#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import hashlib
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
try:
    from google import genai
    from google.genai import types
    GENAI_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    GENAI_IMPORT_ERROR = str(exc)

try:
    from google.cloud import storage
    STORAGE_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    storage = None  # type: ignore[assignment]
    STORAGE_IMPORT_ERROR = str(exc)

try:
    from google.cloud import texttospeech
    TTS_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    texttospeech = None  # type: ignore[assignment]
    TTS_IMPORT_ERROR = str(exc)

from PIL import Image, ImageOps
from pydantic import BaseModel


dotenv.load_dotenv()

APP_NAME = "Linger Live Capture"
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "session_cache"
LOCAL_UPLOAD_DIR = BASE_DIR / "local_uploads"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TTS_DIR = LOCAL_UPLOAD_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ID = os.getenv("PROJECT_ID", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
REGION = os.getenv("REGION", "us-central1")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
LIVE_LOCATION = os.getenv("LIVE_LOCATION", "global")
DEFAULT_STORAGE_MODE: Literal["auto", "gcs", "local"] = os.getenv("STORAGE_MODE", "auto").lower()  # type: ignore[assignment]
SIGNED_URL_EXPIRY_MIN = int(os.getenv("SIGNED_URL_EXPIRY_MIN", "1440"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))
FRAME_MAX_DIM = int(os.getenv("FRAME_MAX_DIM", "1280"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "88"))
STAGE1_MODEL = os.getenv("STAGE1_MODEL", "gemini-2.5-flash")
FRAME_MODEL = os.getenv("FRAME_MODEL", "gemini-2.5-flash-lite")
LIVE_TEXT_MODEL = os.getenv("LIVE_TEXT_MODEL", "gemini-2.0-flash-live-preview-04-09")
ENABLE_LIVE_AGENT = os.getenv("ENABLE_LIVE_AGENT", "true").lower() in {"1", "true", "yes", "on"}
KEEP_BEST_LIMIT = int(os.getenv("KEEP_BEST_LIMIT", "10"))
MIN_KEEP_SCORE = int(os.getenv("MIN_KEEP_SCORE", "74"))
MAX_RECENT_UPLOADS = int(os.getenv("MAX_RECENT_UPLOADS", "24"))
VIDEO_PIPELINE_SCRIPT = os.getenv("VIDEO_PIPELINE_SCRIPT", "")
VIDEO_PIPELINE_BRIEF_FILE = os.getenv("VIDEO_PIPELINE_BRIEF_FILE", "")
VIDEO_PIPELINE_EXTRA_ARGS = os.getenv("VIDEO_PIPELINE_EXTRA_ARGS", "")
DEBUG_LOG_PROMPTS = os.getenv("DEBUG_LOG_PROMPTS", "true").lower() in {"1", "true", "yes", "on"}
GUIDE_TTS_ENABLED = os.getenv("GUIDE_TTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
GUIDE_TTS_LANGUAGE_CODE = os.getenv("GUIDE_TTS_LANGUAGE_CODE", "en-US")
GUIDE_TTS_VOICE_NAME = os.getenv("GUIDE_TTS_VOICE_NAME", "en-US-Chirp3-HD-Aoede")
GUIDE_TTS_ENDPOINT = os.getenv("CLOUD_TTS_ENDPOINT", "")
AUTO_STOP_KEEP_COUNT = int(os.getenv("AUTO_STOP_KEEP_COUNT", "3"))
AUTO_STOP_BEST_SCORE = int(os.getenv("AUTO_STOP_BEST_SCORE", "90"))
AUTO_STOP_MAX_FRAMES = int(os.getenv("AUTO_STOP_MAX_FRAMES", "24"))
EXPORT_BEST_COUNT = int(os.getenv("EXPORT_BEST_COUNT", "3"))

ROLE_PRIORITY = (
    "hero_object",
    "room_opportunity",
    "fit_check",
    "reveal_context",
    "detail_texture",
    "problem_area",
)
MODEL_TIMEOUT_SEC = int(os.getenv("MODEL_TIMEOUT_SEC", "30"))
DEBUG_EVENT_LIMIT = int(os.getenv("DEBUG_EVENT_LIMIT", "200"))
REACT_DIST_DIR_ENV = os.getenv("REACT_DIST_DIR", "")


def resolve_react_dist_dir() -> Optional[Path]:
    candidates: List[Path] = []
    if REACT_DIST_DIR_ENV:
        candidates.append(Path(REACT_DIST_DIR_ENV))
    candidates.extend(
        [
            BASE_DIR / "fe",
            BASE_DIR / "dist",
            BASE_DIR.parent / "linger_react_poc" / "dist",
            BASE_DIR / "../linger_react_poc/dist",
        ]
    )
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if (candidate / "index.html").exists():
            return candidate
    return None


REACT_DIST_DIR = resolve_react_dist_dir()

app = FastAPI(title=APP_NAME)
app.mount("/local_uploads", StaticFiles(directory=str(LOCAL_UPLOAD_DIR)), name="local_uploads")
app.mount("/session_cache", StaticFiles(directory=str(CACHE_DIR)), name="session_cache")
if REACT_DIST_DIR and (REACT_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_DIST_DIR / "assets")), name="react_assets")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        json_log(
            "http_error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=str(exc),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        raise
    response.headers["x-request-id"] = request_id
    json_log(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return response


# ---------- helpers ----------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return now_utc().isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "session"


DEBUG_EVENTS: List[Dict[str, Any]] = []


def json_log(event_type: str, **payload: Any) -> None:
    event = {"ts": utc_iso(), "event_type": event_type, **payload}
    DEBUG_EVENTS.append(event)
    if len(DEBUG_EVENTS) > DEBUG_EVENT_LIMIT:
        del DEBUG_EVENTS[:-DEBUG_EVENT_LIMIT]
    print(json.dumps(event, ensure_ascii=False), flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def pretty_bytes(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num}B"


def pil_to_jpeg_bytes(image: Image.Image, max_dim: int, quality: int = JPEG_QUALITY) -> Tuple[bytes, int, int]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), image.width, image.height


def normalize_uploaded_image(file_bytes: bytes, max_dim: int) -> Tuple[bytes, int, int]:
    image = Image.open(io.BytesIO(file_bytes))
    try:
        return pil_to_jpeg_bytes(image, max_dim=max_dim)
    finally:
        image.close()


def image_size_from_bytes(file_bytes: bytes) -> Tuple[int, int]:
    image = Image.open(io.BytesIO(file_bytes))
    try:
        return image.size
    finally:
        image.close()


def average_hash(file_bytes: bytes, hash_size: int = 8) -> int:
    image = Image.open(io.BytesIO(file_bytes))
    try:
        gray = ImageOps.exif_transpose(image).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    finally:
        image.close()
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel >= avg else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def media_part_from_bytes(file_bytes: bytes, mime_type: str = "image/jpeg") -> Any:
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)


def make_sync_client(location: str) -> Any:
    if genai is None or types is None:
        return None
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID or None,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )


def make_live_client(location: str) -> Any:
    if genai is None or types is None:
        return None
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID or None,
        location=location,
        http_options=types.HttpOptions(api_version="v1beta1"),
    )


SYNC_CLIENT: Any = None
LIVE_CLIENT: Any = None
TTS_CLIENT: Any = None


def get_sync_client() -> Any:
    global SYNC_CLIENT
    if SYNC_CLIENT is None:
        SYNC_CLIENT = make_sync_client(VERTEX_LOCATION)
    if SYNC_CLIENT is None:
        raise RuntimeError(f"google-genai is not available: {GENAI_IMPORT_ERROR}")
    return SYNC_CLIENT


def get_live_client() -> Any:
    global LIVE_CLIENT
    if LIVE_CLIENT is None:
        LIVE_CLIENT = make_live_client(LIVE_LOCATION)
    if LIVE_CLIENT is None:
        raise RuntimeError(f"google-genai is not available: {GENAI_IMPORT_ERROR}")
    return LIVE_CLIENT


def make_tts_client() -> Any:
    if texttospeech is None:
        return None
    kwargs: Dict[str, Any] = {}
    if GUIDE_TTS_ENDPOINT:
        endpoint = GUIDE_TTS_ENDPOINT.replace("https://", "").replace("http://", "")
        kwargs["client_options"] = {"api_endpoint": endpoint}
    return texttospeech.TextToSpeechClient(**kwargs)


def get_tts_client() -> Any:
    global TTS_CLIENT
    if TTS_CLIENT is None:
        TTS_CLIENT = make_tts_client()
    if TTS_CLIENT is None:
        raise RuntimeError(f"google-cloud-texttospeech is not available: {TTS_IMPORT_ERROR}")
    return TTS_CLIENT


def sanitize_guide_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    return clean[:180]


def synthesize_guide_audio(text: str) -> Optional[str]:
    clean = sanitize_guide_text(text)
    if not clean or not GUIDE_TTS_ENABLED:
        return None
    if texttospeech is None:
        return None

    key = hashlib.sha1(f"{GUIDE_TTS_VOICE_NAME}|{clean}".encode("utf-8")).hexdigest()[:20]
    out_path = TTS_DIR / f"guide_{key}.mp3"
    if out_path.exists():
        return f"/local_uploads/tts/{out_path.name}"

    client = get_tts_client()
    synthesis_input = texttospeech.SynthesisInput(text=clean)
    voice = texttospeech.VoiceSelectionParams(
        language_code=GUIDE_TTS_LANGUAGE_CODE,
        name=GUIDE_TTS_VOICE_NAME,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    out_path.write_bytes(response.audio_content)
    return f"/local_uploads/tts/{out_path.name}"


# ---------- schemas ----------

STAGE1_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "object_label": {"type": "STRING"},
        "one_line_summary": {"type": "STRING"},
        "story_signal": {"type": "STRING"},
        "why_it_is_visually_workable": {"type": "STRING"},
        "things_to_avoid": {"type": "ARRAY", "items": {"type": "STRING"}},
        "live_capture_goals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "best_angles": {"type": "ARRAY", "items": {"type": "STRING"}},
        "opening_coach_line": {"type": "STRING"},
        "quality_score_1_to_10": {"type": "INTEGER"},
    },
    "required": [
        "object_label",
        "one_line_summary",
        "story_signal",
        "why_it_is_visually_workable",
        "things_to_avoid",
        "live_capture_goals",
        "best_angles",
        "opening_coach_line",
        "quality_score_1_to_10",
    ],
}

FRAME_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "keep": {"type": "BOOLEAN"},
        "score_0_to_100": {"type": "INTEGER"},
        "moment_label": {"type": "STRING"},
        "cinematic_role": {"type": "STRING"},
        "why": {"type": "STRING"},
        "micro_direction": {"type": "STRING"},
        "best_future_use": {"type": "STRING"},
        "duplicate_likelihood_0_to_100": {"type": "INTEGER"},
        "clarity_0_to_100": {"type": "INTEGER"},
        "context_fit_0_to_100": {"type": "INTEGER"},
        "opportunity_signal_0_to_100": {"type": "INTEGER"},
        "novelty_0_to_100": {"type": "INTEGER"},
    },
    "required": [
        "keep",
        "score_0_to_100",
        "moment_label",
        "cinematic_role",
        "why",
        "micro_direction",
        "best_future_use",
        "duplicate_likelihood_0_to_100",
    ],
}


class GuideTTSRequest(BaseModel):
    text: str


# ---------- storage ----------


@dataclass
class StoredAsset:
    storage_mode: str
    object_name: str
    filename: str
    size_bytes: int
    width: int
    height: int
    created_at: str
    view_url: Optional[str] = None
    gs_uri: Optional[str] = None


class StorageBackend:
    def __init__(self) -> None:
        self.mode = self._resolve_mode(DEFAULT_STORAGE_MODE)
        self.bucket_name = BUCKET_NAME
        self.project_id = PROJECT_ID
        self._client: Any = None
        if self.mode == "gcs":
            if storage is None:
                raise RuntimeError(f"google-cloud-storage is not available: {STORAGE_IMPORT_ERROR}")
            self._client = storage.Client(project=self.project_id or None)
        json_log("storage_backend_ready", storage_mode=self.mode, bucket_name=self.bucket_name or None, project_id=self.project_id or None)

    def _resolve_mode(self, requested: str) -> Literal["gcs", "local"]:
        requested = (requested or "auto").lower()
        if requested == "local":
            return "local"
        if requested == "gcs":
            if not BUCKET_NAME:
                raise RuntimeError("STORAGE_MODE=gcs but BUCKET_NAME is not set")
            return "gcs"
        return "gcs" if BUCKET_NAME else "local"

    @property
    def bucket(self) -> Any:
        if not self._client or not self.bucket_name:
            raise RuntimeError("Google Cloud Storage is not configured")
        return self._client.bucket(self.bucket_name)

    def store_bytes(self, *, prefix: str, filename: str, data: bytes, width: int, height: int, metadata: Dict[str, str]) -> StoredAsset:
        object_name = f"{prefix.rstrip('/')}/{filename}"
        created_at = utc_iso()
        if self.mode == "local":
            local_path = LOCAL_UPLOAD_DIR / object_name
            ensure_dir(local_path.parent)
            local_path.write_bytes(data)
            safe_json_dump(local_path.with_suffix(".json"), metadata)
            return StoredAsset(
                storage_mode="local",
                object_name=object_name,
                filename=local_path.name,
                size_bytes=len(data),
                width=width,
                height=height,
                created_at=created_at,
                view_url=f"/local_uploads/{object_name}",
            )

        blob = self.bucket.blob(object_name)
        blob.metadata = metadata
        blob.cache_control = "public, max-age=3600"
        blob.upload_from_string(data, content_type="image/jpeg")
        return StoredAsset(
            storage_mode="gcs",
            object_name=object_name,
            filename=Path(object_name).name,
            size_bytes=len(data),
            width=width,
            height=height,
            created_at=created_at,
            gs_uri=f"gs://{self.bucket_name}/{object_name}",
            view_url=self._signed_url(blob),
        )

    def _signed_url(self, blob: storage.Blob) -> Optional[str]:
        try:
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=SIGNED_URL_EXPIRY_MIN),
                method="GET",
                response_disposition="inline",
            )
        except Exception as exc:
            json_log("signed_url_failed", object_name=blob.name, error=str(exc))
            return None


STORAGE = StorageBackend()


# ---------- live agent ----------


class LiveTextAgent:
    def __init__(self, session_id: str, stage1_payload: Dict[str, Any]) -> None:
        self.session_id = session_id
        self.stage1_payload = stage1_payload
        self.enabled = ENABLE_LIVE_AGENT
        self.model = LIVE_TEXT_MODEL
        self._ctx: Any = None
        self._session: Any = None
        self._lock = asyncio.Lock()
        self.start_error: Optional[str] = None

    async def start(self) -> None:
        if not self.enabled or self._session is not None:
            return
        seed = (
            "You are Linger's live room scout. "
            "The user has already shown the hero object. "
            "Now help them collect 1-second stop-shots of the surrounding space so a later generator can make a grounded second-life concept. "
            "Keep replies under 12 words, concrete, calm, and useful. "
            "Need-first, not object-first. "
            "Talk only when useful. Ask for one better angle at a time. "
            "Prefer guidance that helps capture one of these roles: hero_object, room_opportunity, fit_check, reveal_context, detail_texture, problem_area. "
            "Never hallucinate measurements. Never ramble. Never reveal a final idea unless the evidence is strong. "
            f"Stage 1 dossier: {json.dumps(self.stage1_payload, ensure_ascii=False)}"
        )
        try:
            self._ctx = get_live_client().aio.live.connect(
                model=self.model,
                config=types.LiveConnectConfig(response_modalities=[types.Modality.TEXT]),
            )
            self._session = await self._ctx.__aenter__()
            await self.ask(seed)
            json_log("live_agent_started", session_id=self.session_id, model=self.model)
        except Exception as exc:
            self.enabled = False
            self.start_error = str(exc)
            self._session = None
            json_log("live_agent_start_failed", session_id=self.session_id, error=str(exc))

    async def ask(self, text: str) -> str:
        if not self.enabled:
            return "Move slower. Frame one clean vertical shot."
        if self._session is None:
            await self.start()
        if self._session is None:
            return "Hold steady. Clean the background and fill the frame."

        async with self._lock:
            await self._session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
                turn_complete=True,
            )
            chunks: List[str] = []
            async for message in self._session.receive():
                if getattr(message, "text", None):
                    chunks.append(message.text)
                server_content = getattr(message, "server_content", None)
                if server_content and getattr(server_content, "turn_complete", False):
                    break
            reply = " ".join(part.strip() for part in chunks if part.strip()).strip()
            return reply or "Good. Now hold one cleaner, brighter backup shot."

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None


# ---------- session store ----------


class SessionStore:
    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, List[WebSocket]] = {}
        self._global_lock = asyncio.Lock()

    def _session_dir(self, session_id: str) -> Path:
        return ensure_dir(CACHE_DIR / slugify(session_id))

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"

    def get_or_create(self, session_id: str) -> Dict[str, Any]:
        session_id = slugify(session_id)
        if session_id not in self.sessions:
            state_path = self._state_path(session_id)
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            else:
                state = {
                    "session_id": session_id,
                    "created_at": utc_iso(),
                    "stage1": None,
                    "stage1_upload": None,
                    "live_metrics": {
                        "frames_seen": 0,
                        "frames_kept": 0,
                        "latency_ms_total": 0,
                        "best_score": 0,
                    },
                    "best_frames": [],
                    "fallback_top_frame": None,
                    "recent_events": [],
                    "story_seed": None,
                    "finalized": None,
                    "debug": {},
                }
            state["_lock"] = asyncio.Lock()
            state["_agent"] = None
            self.sessions[session_id] = state
        return self.sessions[session_id]

    def persist(self, session_id: str) -> None:
        state = self.get_or_create(session_id)
        out = {k: v for k, v in state.items() if not k.startswith("_")}
        safe_json_dump(self._state_path(session_id), out)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        session_id = slugify(session_id)
        await websocket.accept()
        self.connections.setdefault(session_id, []).append(websocket)
        await websocket.send_json({"type": "connected", "session_id": session_id})

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        session_id = slugify(session_id)
        conns = self.connections.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)

    async def broadcast(self, session_id: str, payload: Dict[str, Any]) -> None:
        session_id = slugify(session_id)
        dead: List[WebSocket] = []
        for websocket in self.connections.get(session_id, []):
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(session_id, websocket)


STORE = SessionStore()


# ---------- model calls ----------


def part_size_bytes(part: Any) -> int:
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return 0
    data = getattr(inline, "data", b"")
    if data is None:
        return 0
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    return len(data)


def stage1_fallback(note: str = "") -> Dict[str, Any]:
    note_hint = (note or "").strip()
    return {
        "object_label": "Reusable item",
        "one_line_summary": "A usable object that could solve a small room need with the right context.",
        "story_signal": note_hint[:120] if note_hint else "Find where this item could help in the room, then capture proof shots.",
        "why_it_is_visually_workable": "The object can anchor a grounded before-and-after story if we collect one hero frame plus room evidence.",
        "things_to_avoid": ["blur", "busy edges", "dark corners"],
        "live_capture_goals": [
            "find the room problem or empty spot",
            "capture one fit-check or placement proof shot",
            "capture one detail or texture support shot",
        ],
        "best_angles": [
            "clean centered hero",
            "closer texture detail",
            "wider room reveal",
        ],
        "opening_coach_line": "Scan the room for where this could help.",
        "quality_score_1_to_10": 6,
        "_fallback": True,
    }


def quick_frame_fallback(frame_bytes: bytes, width: int, height: int) -> Dict[str, Any]:
    image = Image.open(io.BytesIO(frame_bytes))
    try:
        gray = ImageOps.exif_transpose(image).convert("L")
        tiny = gray.resize((64, 64), Image.Resampling.BILINEAR)
        pixels = list(tiny.getdata())
        mean = sum(pixels) / max(len(pixels), 1)
        variance = sum((p - mean) ** 2 for p in pixels) / max(len(pixels), 1)
        small = gray.resize((32, 32), Image.Resampling.BILINEAR)
        sx = 0
        sy = 0
        for y in range(31):
            for x in range(31):
                p = small.getpixel((x, y))
                sx += abs(p - small.getpixel((x + 1, y)))
                sy += abs(p - small.getpixel((x, y + 1)))
        edge = (sx + sy) / (31 * 31 * 2)
    finally:
        image.close()

    brightness_score = max(0, 100 - int(abs(mean - 138) * 1.1))
    sharpness_score = min(100, int(edge * 1.8 + variance * 0.03))
    center_bonus = 8 if height >= width else 0
    score = int(max(18, min(92, 0.45 * brightness_score + 0.45 * sharpness_score + center_bonus)))
    keep = score >= max(68, MIN_KEEP_SCORE - 6)

    if mean < 75:
        micro = "Find brighter light and hold steady."
        label = "Dim stop-shot"
        role = "reveal_context"
    elif sharpness_score < 40:
        micro = "Hold still and lock one cleaner frame."
        label = "Soft stop-shot"
        role = "fit_check"
    elif keep:
        micro = "Good. Take one closer proof shot."
        label = "Useful room proof"
        role = "room_opportunity" if height >= width else "reveal_context"
    else:
        micro = "Simplify edges and show more room."
        label = "Usable context frame"
        role = "reveal_context"

    return {
        "keep": keep,
        "score_0_to_100": score,
        "moment_label": label,
        "cinematic_role": role,
        "why": "Fallback visual heuristic used after model issue.",
        "micro_direction": micro[:60],
        "best_future_use": "supporting still" if keep else "skip",
        "duplicate_likelihood_0_to_100": 40,
        "clarity_0_to_100": sharpness_score,
        "context_fit_0_to_100": score,
        "opportunity_signal_0_to_100": 60 if keep else 35,
        "novelty_0_to_100": 40,
        "_fallback": True,
    }


async def generate_json_safe(
    *,
    prompt: str,
    schema: Dict[str, Any],
    media_parts: Optional[List[Any]],
    model: str,
    temperature: float = 0.3,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            run_in_threadpool(generate_json, prompt, schema, media_parts, model, temperature),
            timeout=MODEL_TIMEOUT_SEC,
        )
    except Exception as exc:
        json_log("model_fallback", model=model, error=str(exc), fallback=bool(fallback))
        if fallback is not None:
            return fallback
        raise


def generate_json(prompt: str, schema: Dict[str, Any], media_parts: Optional[List[Any]], model: str, temperature: float = 0.3) -> Dict[str, Any]:
    client = get_sync_client()
    contents: List[Any] = [prompt]
    if media_parts:
        contents.extend(media_parts)
    if DEBUG_LOG_PROMPTS:
        json_log(
            "model_request",
            model=model,
            prompt_chars=len(prompt),
            media_parts=len(media_parts or []),
            approx_payload_bytes=len(prompt.encode("utf-8")) + sum(part_size_bytes(p) for p in media_parts or []),
        )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    return json.loads(response.text)


def stage1_prompt(note: str) -> str:
    return (
        "You are Stage 1 of Linger, a need-first second-life room scout. "
        "The user shows one hero object first, and phase 2 will collect 1-second stop-shots of the surrounding space. "
        "Interpret the object as a future reuse opportunity, not as a product identification task. "
        "Return strict JSON only and fill the existing schema exactly. "
        f"User note: {note or 'No extra note.'} "
        "Be need-first, not object-first. Infer reuse affordances, what kind of room need this item might solve, "
        "and what evidence the camera should hunt next in the room. "
        "Use the existing fields like this: "
        "object_label = plain object name; "
        "one_line_summary = what this object could plausibly become; "
        "story_signal = one sentence linking object to a room need or opportunity; "
        "why_it_is_visually_workable = why this can anchor a grounded before/after story; "
        "things_to_avoid = capture mistakes; "
        "live_capture_goals = 3 short goals for phase 2 stop-shots, especially problem area, fit-check, and room proof; "
        "best_angles = hero/detail/context suggestions; "
        "opening_coach_line = one short spoken-ready sentence under 12 words; "
        "quality_score_1_to_10 = confidence. "
        "Do not over-index on the exact brand or exact product identity. "
        "Optimize for grounded reuse inspiration and clean downstream story generation."
    )


def frame_prompt(stage1_payload: Dict[str, Any], note: str, recent_best: List[Dict[str, Any]]) -> str:
    best_summary = [
        {
            "score": frame.get("score_0_to_100"),
            "moment_label": frame.get("moment_label"),
            "cinematic_role": frame.get("cinematic_role"),
        }
        for frame in recent_best[:4]
    ]
    return (
        "You are judging one isolated 1-second stop-shot from phase 2 of a mobile capture session. "
        "The app already has the hero object. Your job now is to find grounded inspiration in the surrounding room. "
        "Return strict JSON only. "
        f"Stage 1 dossier: {json.dumps(stage1_payload, ensure_ascii=False)}. "
        f"Recent best frames: {json.dumps(best_summary, ensure_ascii=False)}. "
        f"Operator note: {note or 'No extra note.'}. "
        "Use this score rubric: "
        "30 points = clarity and steadiness, "
        "25 points = readability of the object or target area, "
        "20 points = context usefulness for a later generated scene, "
        "15 points = opportunity signal that suggests where this object could belong, "
        "10 points = novelty versus already kept frames. "
        "Pick cinematic_role from this set when possible: "
        "hero_object, room_opportunity, fit_check, reveal_context, detail_texture, problem_area. "
        "A keep-worthy frame should help prove one of three things: "
        "what the object is, what room problem exists, or why the object fits there. "
        "Do not obsess over matching the exact item from stage 1. "
        "Reward frames that would be useful inputs for a later storyboard or image generation step. "
        "If weak, say why briefly. "
        "Micro direction must be one short spoken-friendly sentence under 10 words. "
        "If you have enough evidence, be decisive. If the frame is repetitive, raise duplicate_likelihood_0_to_100."
    )


def chat_prompt(stage1_payload: Dict[str, Any], best_frames: List[Dict[str, Any]], user_text: str) -> str:
    return (
        "You are Linger's live room scout. Reply in 1 short sentence, spoken-friendly, no fluff. "
        "Need-first, not object-first. Help the user capture room evidence for a second-life idea. "
        f"Stage 1 interpretation: {json.dumps(stage1_payload, ensure_ascii=False)}. "
        f"Best frames so far: {json.dumps(best_frames[:3], ensure_ascii=False)}. "
        f"User says: {user_text}"
    )


# ---------- ranking and persistence ----------


def best_frames_dir(session_id: str) -> Path:
    return ensure_dir(CACHE_DIR / slugify(session_id) / "best_frames")


def stage1_dir(session_id: str) -> Path:
    return ensure_dir(CACHE_DIR / slugify(session_id) / "stage1")


def export_dir(session_id: str) -> Path:
    return ensure_dir(CACHE_DIR / slugify(session_id) / "export")


def save_stage1_local(session_id: str, image_bytes: bytes) -> Path:
    filename = f"stage1_{now_utc().strftime('%Y%m%dT%H%M%SZ')}.jpg"
    path = stage1_dir(session_id) / filename
    path.write_bytes(image_bytes)
    return path


def save_best_frame_local(session_id: str, frame_bytes: bytes, frame_index: int, score: int) -> Path:
    filename = f"frame_{frame_index:05d}_score_{score:03d}.jpg"
    path = best_frames_dir(session_id) / filename
    path.write_bytes(frame_bytes)
    return path


def normalize_cinematic_role(value: str) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    mapping = {
        "hero": "hero_object",
        "hero_shot": "hero_object",
        "object": "hero_object",
        "detail": "detail_texture",
        "texture": "detail_texture",
        "context": "reveal_context",
        "wide": "reveal_context",
        "reveal": "reveal_context",
        "problem": "problem_area",
        "problem_corner": "problem_area",
        "opportunity": "room_opportunity",
        "room": "room_opportunity",
        "fit": "fit_check",
        "fitcheck": "fit_check",
    }
    normalized = mapping.get(raw, raw)
    return normalized if normalized in set(ROLE_PRIORITY) else "reveal_context"


def frame_sort_key(frame: Dict[str, Any]) -> Tuple[int, int, int]:
    role = normalize_cinematic_role(frame.get("cinematic_role", ""))
    score = int(frame.get("score_0_to_100", 0))
    duplicate_likelihood = int(frame.get("duplicate_likelihood_0_to_100", 0))
    role_bonus = 8 if role in {"hero_object", "room_opportunity", "fit_check"} else 0
    return (score + role_bonus, -duplicate_likelihood, int(frame.get("frame_index", 0)))


def rebalance_frame_pool(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for frame in frames:
        normalized = {**frame, "cinematic_role": normalize_cinematic_role(frame.get("cinematic_role", ""))}
        grouped.setdefault(normalized["cinematic_role"], []).append(normalized)

    for bucket in grouped.values():
        bucket.sort(key=frame_sort_key, reverse=True)

    ordered: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for role in ROLE_PRIORITY:
        bucket = grouped.pop(role, [])
        if bucket:
            ordered.append(bucket[0])
            leftovers.extend(bucket[1:])

    for _, bucket in sorted(grouped.items(), key=lambda item: frame_sort_key(item[1][0]), reverse=True):
        ordered.append(bucket[0])
        leftovers.extend(bucket[1:])

    leftovers.sort(key=frame_sort_key, reverse=True)

    deduped: List[Dict[str, Any]] = []
    seen_paths = set()
    for frame in ordered + leftovers:
        key = frame.get("local_path") or f"{frame.get('ts_ms')}-{frame.get('moment_label')}"
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped.append(frame)

    return deduped


def update_fallback_top_frame(existing: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], bool]:
    candidate = {**candidate, "cinematic_role": normalize_cinematic_role(candidate.get("cinematic_role", ""))}
    candidate_score = int(candidate.get("score_0_to_100", 0))
    candidate_role = candidate.get("cinematic_role", "reveal_context")

    if not existing:
        return candidate, None, True

    existing_score = int(existing.get("score_0_to_100", 0))
    existing_role = normalize_cinematic_role(existing.get("cinematic_role", ""))

    should_replace = False
    if candidate_score > existing_score:
        should_replace = True
    elif candidate_score == existing_score:
        role_rank = {name: idx for idx, name in enumerate(ROLE_PRIORITY)}
        should_replace = role_rank.get(candidate_role, 999) < role_rank.get(existing_role, 999)

    if should_replace:
        return candidate, existing.get("local_path"), True

    return existing, None, False


def select_top_frames(existing: List[Dict[str, Any]], candidate: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    candidate = {**candidate, "cinematic_role": normalize_cinematic_role(candidate.get("cinematic_role", ""))}
    candidate_score = int(candidate.get("score_0_to_100", 0))
    candidate_hash = int(candidate["ahash"], 16)
    replaced_path: Optional[str] = None

    for index, frame in enumerate(list(existing)):
        frame_hash = int(frame["ahash"], 16)
        if hamming_distance(candidate_hash, frame_hash) <= 6:
            same_role = normalize_cinematic_role(frame.get("cinematic_role", "")) == candidate["cinematic_role"]
            required_margin = 4 if same_role else 8
            if candidate_score > int(frame.get("score_0_to_100", 0)) + required_margin:
                replaced_path = frame.get("local_path")
                existing[index] = candidate
                return rebalance_frame_pool(existing)[:KEEP_BEST_LIMIT], True, replaced_path
            return rebalance_frame_pool(existing)[:KEEP_BEST_LIMIT], False, None

    if int(candidate.get("duplicate_likelihood_0_to_100", 0)) >= 96 and candidate_score < 90:
        return rebalance_frame_pool(existing)[:KEEP_BEST_LIMIT], False, None

    combined = rebalance_frame_pool(existing + [candidate])
    kept = combined[:KEEP_BEST_LIMIT]
    kept_paths = {frame.get("local_path") for frame in kept}

    if candidate.get("local_path") not in kept_paths:
        return kept, False, candidate.get("local_path")

    dropped_old_path = None
    for frame in existing:
        path = frame.get("local_path")
        if path not in kept_paths:
            dropped_old_path = path
            break

    return kept, True, dropped_old_path


async def ensure_live_agent(state: Dict[str, Any]) -> Optional[LiveTextAgent]:
    if state.get("_agent") is None and state.get("stage1") is not None:
        agent = LiveTextAgent(session_id=state["session_id"], stage1_payload=state["stage1"]["analysis"])
        state["_agent"] = agent
        await agent.start()
    return state.get("_agent")


async def maybe_live_reply(state: Dict[str, Any], user_text: str) -> str:
    agent = await ensure_live_agent(state)
    if agent is not None and agent.enabled:
        try:
            return await agent.ask(user_text)
        except Exception as exc:
            json_log("live_agent_ask_failed", session_id=state["session_id"], error=str(exc))
    prompt = chat_prompt(state["stage1"]["analysis"], state.get("best_frames", []), user_text)
    response = get_sync_client().models.generate_content(model=STAGE1_MODEL, contents=[prompt])
    return (response.text or "Hold steady and simplify the frame.").strip()


def build_story_seed(state: Dict[str, Any]) -> Dict[str, Any]:
    stage1 = (state.get("stage1") or {}).get("analysis") or {}
    best_frames = rebalance_frame_pool(list(state.get("best_frames") or []))
    fallback = state.get("fallback_top_frame")
    if best_frames:
        export_frames = best_frames[:EXPORT_BEST_COUNT]
    else:
        export_frames = [fallback] if fallback else []

    hero_local = (state.get("stage1") or {}).get("local_path")
    hero_preview_url = None
    if hero_local:
        p = Path(hero_local)
        hero_preview_url = f"/session_cache/{slugify(state['session_id'])}/stage1/{p.name}"

    selected_roles = [normalize_cinematic_role(frame.get("cinematic_role", "")) for frame in export_frames]
    selected_labels = [frame.get("moment_label") for frame in export_frames if frame.get("moment_label")]

    hook = stage1.get("story_signal") or "This object could solve a small room need if we find the right place for it."
    description = stage1.get("one_line_summary") or "A clean hero frame plus a few grounded support shots."
    idea = hook
    if selected_labels:
        idea = f"{hook} Supporting moments: {', '.join(selected_labels[:3])}."

    return {
        "session_id": state["session_id"],
        "title": stage1.get("object_label") or "Story seed",
        "description": description,
        "idea": idea,
        "hook": hook,
        "tone": "calm, warm, grounded, inventive",
        "visual_style": "clean motion-comic grounded in real room photos",
        "generation_strategy": "hero still + 2-3 complementary support stills, then storyboard JSON and optional single hero clip",
        "hero_preview_url": hero_preview_url,
        "hero_image": hero_preview_url,
        "best_count": len(best_frames),
        "fallback_used": not bool(best_frames) and bool(fallback),
        "selected_roles": selected_roles,
        "selected_frames": [
            {
                "moment_label": frame.get("moment_label"),
                "cinematic_role": frame.get("cinematic_role"),
                "best_future_use": frame.get("best_future_use"),
                "score_0_to_100": frame.get("score_0_to_100"),
                "preview_url": frame.get("preview_url") or frame.get("local_preview_url"),
                "local_preview_url": frame.get("local_preview_url"),
                "local_path": frame.get("local_path"),
                "frame_index": frame.get("frame_index"),
            }
            for frame in export_frames
        ],
        "created_at": utc_iso(),
    }


def build_idea_text(story_seed: Dict[str, Any], export_frames: List[Dict[str, Any]]) -> str:
    lines = [
        story_seed.get("title") or "Story seed",
        "",
        f"Hook: {story_seed.get('hook') or story_seed.get('idea') or ''}",
        f"Tone: {story_seed.get('tone') or ''}",
        f"Visual style: {story_seed.get('visual_style') or ''}",
        f"Generation strategy: {story_seed.get('generation_strategy') or ''}",
        "",
        "Selected shots:",
    ]
    for index, frame in enumerate(export_frames, start=1):
        lines.append(
            f"{index}. {frame.get('cinematic_role', 'shot')} — "
            f"{frame.get('moment_label', 'Moment')} — "
            f"score {frame.get('score_0_to_100', 0)} — "
            f"use: {frame.get('best_future_use', 'supporting still')}"
        )
    lines.append("")
    lines.append(f"Idea: {story_seed.get('idea') or ''}")
    return "\n".join(lines).strip() + "\n"


# ---------- html ----------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = (REACT_DIST_DIR / "index.html") if REACT_DIST_DIR else None
    if index_path and index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse()


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": APP_NAME, "frontend": "react" if REACT_DIST_DIR else "legacy"}


@app.get("/api/debug/events")
async def api_debug_events(limit: int = 80) -> Dict[str, Any]:
    limit = max(1, min(limit, DEBUG_EVENT_LIMIT))
    return {"events": DEBUG_EVENTS[-limit:]}


@app.post("/api/tts/guide")
async def api_tts_guide(payload: GuideTTSRequest) -> Dict[str, Any]:
    text = sanitize_guide_text(payload.text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty guide text")
    if not GUIDE_TTS_ENABLED:
        return {"ok": False, "reason": "disabled"}
    try:
        url = synthesize_guide_audio(text)
    except Exception as exc:
        json_log("guide_tts_failed", error=str(exc), text=text[:120])
        raise HTTPException(status_code=500, detail=f"Guide TTS failed: {exc}")
    return {"ok": True, "text": text, "url": url, "voice": GUIDE_TTS_VOICE_NAME}


@app.get("/api/config")
async def api_config() -> Dict[str, Any]:
    return {
        "app_name": APP_NAME,
        "storage_mode": STORAGE.mode,
        "project_id": PROJECT_ID,
        "bucket_name": BUCKET_NAME,
        "region": REGION,
        "vertex_location": VERTEX_LOCATION,
        "live_location": LIVE_LOCATION,
        "stage1_model": STAGE1_MODEL,
        "frame_model": FRAME_MODEL,
        "live_text_model": LIVE_TEXT_MODEL,
        "enable_live_agent": ENABLE_LIVE_AGENT,
        "guide_tts_enabled": GUIDE_TTS_ENABLED and texttospeech is not None,
        "guide_tts_voice": GUIDE_TTS_VOICE_NAME,
        "video_pipeline_script": bool(VIDEO_PIPELINE_SCRIPT),
        "react_dist": str(REACT_DIST_DIR) if REACT_DIST_DIR else None,
        "model_timeout_sec": MODEL_TIMEOUT_SEC,
        "auto_stop_keep_count": AUTO_STOP_KEEP_COUNT,
        "auto_stop_best_score": AUTO_STOP_BEST_SCORE,
        "auto_stop_max_frames": AUTO_STOP_MAX_FRAMES,
        "export_best_count": EXPORT_BEST_COUNT,
    }


@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    session_id = slugify(session_id)
    await STORE.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        STORE.disconnect(session_id, websocket)


@app.post("/api/stage1/analyze-photo")
async def api_stage1_analyze_photo(
    session_id: str = Form(...),
    note: str = Form(""),
    file: UploadFile = File(...),
) -> JSONResponse:
    started = time.time()
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    orig_width, orig_height = image_size_from_bytes(file_bytes)
    normalized_bytes, width, height = normalize_uploaded_image(file_bytes, max_dim=MAX_IMAGE_DIM)
    local_path = save_stage1_local(session_id, normalized_bytes)
    prompt = stage1_prompt(note)
    analysis = await generate_json_safe(
        prompt=prompt,
        schema=STAGE1_SCHEMA,
        media_parts=[media_part_from_bytes(normalized_bytes)],
        model=STAGE1_MODEL,
        temperature=0.2,
        fallback=stage1_fallback(note),
    )

    upload = await run_in_threadpool(
        STORAGE.store_bytes,
        prefix=f"captures/{session_id}/stage1",
        filename=local_path.name,
        data=normalized_bytes,
        width=width,
        height=height,
        metadata={
            "session_id": session_id,
            "capture_type": "stage1",
            "note": note[:400],
            "created_at": utc_iso(),
        },
    )
    local_preview_url = f"/session_cache/{session_id}/stage1/{local_path.name}"
    state["stage1"] = {
        "note": note,
        "analysis": analysis,
        "local_path": str(local_path),
        "width": width,
        "height": height,
        "original_width": orig_width,
        "original_height": orig_height,
        "size_bytes": len(normalized_bytes),
        "local_preview_url": local_preview_url,
    }
    state["stage1_upload"] = upload.__dict__
    state["best_frames"] = []
    state["live_metrics"] = {
        "frames_seen": 0,
        "frames_kept": 0,
        "latency_ms_total": 0,
        "best_score": 0,
    }
    state["story_seed"] = build_story_seed(state)
    STORE.persist(session_id)
    await ensure_live_agent(state)

    json_log(
        "stage1_complete",
        session_id=session_id,
        filename=file.filename,
        original_size=f"{orig_width}x{orig_height}",
        normalized_size=f"{width}x{height}",
        uploaded_bytes=len(normalized_bytes),
        latency_ms=round((time.time() - started) * 1000),
    )
    await STORE.broadcast(session_id, {"type": "stage1_ready", "analysis": analysis, "story_seed": state.get("story_seed")})
    return JSONResponse(
        {
            "ok": True,
            "session_id": session_id,
            "analysis": analysis,
            "upload": upload.__dict__,
            "local_path": str(local_path),
            "local_preview_url": local_preview_url,
            "story_seed": state.get("story_seed"),
            "latency_ms": round((time.time() - started) * 1000),
        }
    )


@app.post("/api/live/start")
async def api_live_start(session_id: str = Form(...)) -> Dict[str, Any]:
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    if not state.get("stage1"):
        raise HTTPException(status_code=400, detail="Run stage 1 first")

    agent = await ensure_live_agent(state)
    opening = state["stage1"]["analysis"].get("opening_coach_line") or "Scan the room for where this could help."

    if agent is not None and agent.enabled:
        try:
            opening = await agent.ask(
                "We are starting phase 2 now. Give one short first instruction for a one-second stop-shot room scan."
            )
        except Exception:
            pass

    opening = sanitize_guide_text(opening or "Scan the room for where this could help.")
    await STORE.broadcast(session_id, {"type": "live_guidance", "text": opening})
    return {"ok": True, "session_id": session_id, "guidance": opening}


@app.post("/api/live/frame")
async def api_live_frame(
    session_id: str = Form(...),
    note: str = Form(""),
    frame_index: int = Form(...),
    ts_ms: int = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    started = time.time()
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    if not state.get("stage1"):
        raise HTTPException(status_code=400, detail="Run stage 1 first")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty frame")

    normalized_bytes, width, height = normalize_uploaded_image(raw, max_dim=FRAME_MAX_DIM)
    ahash_hex = f"{average_hash(normalized_bytes):016x}"
    prompt = frame_prompt(
        state["stage1"]["analysis"],
        note or state["stage1"].get("note", ""),
        state.get("best_frames", []),
    )
    analysis = await generate_json_safe(
        prompt=prompt,
        schema=FRAME_SCHEMA,
        media_parts=[media_part_from_bytes(normalized_bytes)],
        model=FRAME_MODEL,
        temperature=0.15,
        fallback=quick_frame_fallback(normalized_bytes, width, height),
    )
    analysis["cinematic_role"] = normalize_cinematic_role(analysis.get("cinematic_role", ""))
    score = int(analysis.get("score_0_to_100", 0))
    duplicate_likelihood = int(analysis.get("duplicate_likelihood_0_to_100", 0))
    keep_requested = bool(analysis.get("keep")) and score >= MIN_KEEP_SCORE and not (
        duplicate_likelihood >= 97 and score < 92
    )
    local_path = save_best_frame_local(session_id, normalized_bytes, frame_index, score)
    candidate = {
        **analysis,
        "frame_index": frame_index,
        "ts_ms": ts_ms,
        "width": width,
        "height": height,
        "size_bytes": len(normalized_bytes),
        "ahash": ahash_hex,
        "local_path": str(local_path),
        "local_preview_url": f"/session_cache/{session_id}/best_frames/{local_path.name}",
        "preview_url": None,
    }

    selected = False
    replaced_path: Optional[str] = None
    fallback_replaced_path: Optional[str] = None
    is_fallback_top = False

    if keep_requested:
        state["best_frames"], selected, replaced_path = select_top_frames(state.get("best_frames", []), candidate)

    state["fallback_top_frame"], fallback_replaced_path, is_fallback_top = update_fallback_top_frame(
        state.get("fallback_top_frame"),
        candidate,
    )

    candidate_path = str(local_path)
    best_paths = {frame.get("local_path") for frame in state.get("best_frames", [])}
    fallback_path = (state.get("fallback_top_frame") or {}).get("local_path")

    if candidate_path not in best_paths and candidate_path != fallback_path:
        try:
            local_path.unlink(missing_ok=True)
        except TypeError:
            if local_path.exists():
                local_path.unlink()

    if replaced_path and replaced_path != candidate_path:
        still_used = replaced_path in best_paths or replaced_path == fallback_path
        if not still_used:
            old_path = Path(replaced_path)
            if old_path.exists():
                old_path.unlink()

    if fallback_replaced_path and fallback_replaced_path != candidate_path:
        still_used = fallback_replaced_path in best_paths or fallback_replaced_path == fallback_path
        if not still_used:
            old_path = Path(fallback_replaced_path)
            if old_path.exists():
                old_path.unlink()

    metrics = state["live_metrics"]
    metrics["frames_seen"] += 1
    metrics["latency_ms_total"] += round((time.time() - started) * 1000)
    metrics["frames_kept"] = len(state.get("best_frames", []))
    metrics["best_score"] = max(metrics.get("best_score", 0), score)
    avg_latency_ms = int(metrics["latency_ms_total"] / max(metrics["frames_seen"], 1))

    guidance_text = sanitize_guide_text(analysis.get("micro_direction") or "")
    if not guidance_text:
        guidance_text = "Shift, steady, and simplify the frame."

    kept_total = len(state.get("best_frames", []))
    effective_kept_total = max(kept_total, 1 if state.get("fallback_top_frame") else 0)
    should_stop = False
    stop_reason = ""
    if score >= 88 and selected:
        guidance_text = "Nice. Take one safer backup, then stop."
    if kept_total >= AUTO_STOP_KEEP_COUNT and metrics.get("best_score", 0) >= AUTO_STOP_BEST_SCORE:
        should_stop = True
        stop_reason = "Enough strong shots collected."
        guidance_text = "Enough strong shots. Tap Enough moment."
    elif metrics["frames_seen"] >= AUTO_STOP_MAX_FRAMES:
        should_stop = True
        stop_reason = "Reached capture limit. Use the best shots now."
        guidance_text = "We have enough. Use the best shots now."

    recent_event = {
        "type": "frame",
        "frame_index": frame_index,
        "selected": selected,
        "fallback_selected": is_fallback_top,
        "score": score,
        "moment_label": analysis.get("moment_label"),
        "cinematic_role": analysis.get("cinematic_role"),
        "latency_ms": round((time.time() - started) * 1000),
        "guidance": guidance_text,
        "should_stop": should_stop,
    }
    state.setdefault("recent_events", []).append(recent_event)
    state["recent_events"] = state["recent_events"][-50:]
    state["story_seed"] = build_story_seed(state)
    STORE.persist(session_id)

    json_log(
        "live_frame_processed",
        session_id=session_id,
        frame_index=frame_index,
        ts_ms=ts_ms,
        raw_bytes=len(raw),
        normalized_bytes=len(normalized_bytes),
        image_size=f"{width}x{height}",
        keep_requested=keep_requested,
        selected=selected,
        fallback_selected=is_fallback_top,
        score=score,
        role=analysis.get("cinematic_role"),
        kept_total=kept_total,
        avg_latency_ms=avg_latency_ms,
        local_name=local_path.name,
    )

    await STORE.broadcast(
        session_id,
        {
            "type": "frame_result",
            "selected": selected,
            "fallback_selected": is_fallback_top,
            "frame": {k: v for k, v in candidate.items() if k != "ahash"},
            "kept_total": kept_total,
            "effective_kept_total": effective_kept_total,
            "avg_latency_ms": avg_latency_ms,
            "guidance": guidance_text,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        },
    )
    return JSONResponse(
        {
            "ok": True,
            "selected": selected,
            "fallback_selected": is_fallback_top,
            "analysis": analysis,
            "score": score,
            "candidate": {k: v for k, v in candidate.items() if k != "ahash"},
            "kept_total": kept_total,
            "effective_kept_total": effective_kept_total,
            "avg_latency_ms": avg_latency_ms,
            "guidance": guidance_text,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "latency_ms": round((time.time() - started) * 1000),
        }
    )


@app.post("/api/live/harvest/stop")
async def api_live_harvest_stop(session_id: str = Form(...), reason: str = Form("manual")) -> Dict[str, Any]:
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    state.setdefault("recent_events", []).append({"type": "harvest_stop", "reason": reason, "ts": utc_iso()})
    state["recent_events"] = state["recent_events"][-50:]
    state["story_seed"] = build_story_seed(state)
    STORE.persist(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "reason": reason,
        "kept_count": max(len(state.get("best_frames", [])), 1 if state.get("fallback_top_frame") else 0),
        "frames_seen": state.get("live_metrics", {}).get("frames_seen", 0),
        "story_seed": state.get("story_seed"),
    }


@app.post("/api/story/seed")
async def api_story_seed(session_id: str = Form(...)) -> Dict[str, Any]:
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    if not state.get("stage1"):
        raise HTTPException(status_code=400, detail="Run stage 1 first")
    state["story_seed"] = build_story_seed(state)
    STORE.persist(session_id)
    return {"ok": True, "session_id": session_id, "story_seed": state["story_seed"]}


@app.post("/api/live/chat")
async def api_live_chat(session_id: str = Form(...), message: str = Form(...)) -> Dict[str, Any]:
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    if not state.get("stage1"):
        raise HTTPException(status_code=400, detail="Run stage 1 first")
    reply = await maybe_live_reply(state, message)
    event = {"type": "chat_reply", "user": message, "reply": reply}
    state.setdefault("recent_events", []).append(event)
    state["recent_events"] = state["recent_events"][-50:]
    STORE.persist(session_id)
    await STORE.broadcast(session_id, event)
    return {"ok": True, "reply": reply}


@app.get("/api/session/{session_id}")
async def api_session_state(session_id: str) -> Dict[str, Any]:
    state = STORE.get_or_create(slugify(session_id))
    if state.get("stage1") and not state.get("story_seed"):
        state["story_seed"] = build_story_seed(state)
    public = {k: v for k, v in state.items() if not k.startswith("_")}
    return public


@app.post("/api/session/finalize")
async def api_session_finalize(session_id: str = Form(...), enough_moment: str = Form("enough moment")) -> Dict[str, Any]:
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    if not state.get("stage1"):
        raise HTTPException(status_code=400, detail="Run stage 1 first")

    state["best_frames"] = rebalance_frame_pool(list(state.get("best_frames") or []))[:KEEP_BEST_LIMIT]

    export_frames = list(state.get("best_frames") or [])
    used_fallback_frame = False
    if not export_frames:
        fallback = state.get("fallback_top_frame")
        if fallback:
            export_frames = [fallback]
            used_fallback_frame = True
        else:
            raise HTTPException(status_code=400, detail="No frames captured yet")

    export_frames = export_frames[:EXPORT_BEST_COUNT]
    basket_assets: List[Dict[str, Any]] = []
    shot_manifest: List[Dict[str, Any]] = []
    basket_prefix = f"captures/{session_id}/basket"
    export_root = export_dir(session_id)
    input_media_dir = ensure_dir(export_root / "input_media")
    summary_path = export_root / "session_summary.json"
    story_seed_path = export_root / "story_seed.json"
    shot_manifest_path = export_root / "shot_manifest.json"
    idea_path = export_root / "idea.txt"

    hero_export_path: Optional[Path] = None
    stage1_path = Path(state["stage1"]["local_path"])
    if stage1_path.exists():
        hero_bytes = stage1_path.read_bytes()
        hero_export_path = input_media_dir / "hero.jpg"
        hero_export_path.write_bytes(hero_bytes)

        hero_upload = STORAGE.store_bytes(
            prefix=basket_prefix,
            filename="hero.jpg",
            data=hero_bytes,
            width=int(state["stage1"].get("width", 0)),
            height=int(state["stage1"].get("height", 0)),
            metadata={
                "session_id": session_id,
                "capture_type": "hero",
                "created_at": utc_iso(),
            },
        )
        basket_assets.append(hero_upload.__dict__)

    for rank, frame in enumerate(export_frames, start=1):
        local_path = Path(frame["local_path"])
        if not local_path.exists():
            continue

        frame_bytes = local_path.read_bytes()
        target_name = f"best_{rank:02d}.jpg"
        copied = input_media_dir / target_name
        copied.write_bytes(frame_bytes)

        upload = STORAGE.store_bytes(
            prefix=basket_prefix,
            filename=target_name,
            data=frame_bytes,
            width=int(frame.get("width", 0)),
            height=int(frame.get("height", 0)),
            metadata={
                "session_id": session_id,
                "capture_type": "best_frame",
                "moment_label": str(frame.get("moment_label", "")),
                "cinematic_role": str(frame.get("cinematic_role", "")),
                "best_future_use": str(frame.get("best_future_use", "")),
                "score": str(frame.get("score_0_to_100", 0)),
                "created_at": utc_iso(),
            },
        )

        frame["preview_url"] = upload.view_url
        frame["basket_upload"] = upload.__dict__
        basket_assets.append(upload.__dict__)
        shot_manifest.append(
            {
                "rank": rank,
                "file_name": target_name,
                "local_path": str(copied),
                "moment_label": frame.get("moment_label"),
                "cinematic_role": frame.get("cinematic_role"),
                "best_future_use": frame.get("best_future_use"),
                "score_0_to_100": frame.get("score_0_to_100"),
                "preview_url": upload.view_url,
                "local_preview_url": frame.get("local_preview_url"),
            }
        )

    if not shot_manifest:
        raise HTTPException(status_code=400, detail="No saved frame files available for export")

    story_seed = build_story_seed(
        {
            **state,
            "best_frames": export_frames,
            "fallback_top_frame": export_frames[0] if used_fallback_frame else state.get("fallback_top_frame"),
        }
    )
    story_seed = {
        **story_seed,
        "hero_image": "input_media/hero.jpg" if hero_export_path else None,
        "selected_frames": [f"input_media/{shot['file_name']}" for shot in shot_manifest],
        "selected_frame_items": shot_manifest,
        "selected_roles": [shot.get("cinematic_role") for shot in shot_manifest],
        "fallback_used": used_fallback_frame,
    }

    safe_json_dump(story_seed_path, story_seed)
    safe_json_dump(
        shot_manifest_path,
        {
            "session_id": session_id,
            "hero_image": "input_media/hero.jpg" if hero_export_path else None,
            "shots": shot_manifest,
            "created_at": utc_iso(),
        },
    )
    idea_path.write_text(build_idea_text(story_seed, export_frames), encoding="utf-8")

    summary = {
        "session_id": session_id,
        "enough_moment": enough_moment,
        "stage1": state["stage1"],
        "stage1_upload": state.get("stage1_upload"),
        "best_frames": [{k: v for k, v in frame.items() if k != "ahash"} for frame in export_frames],
        "fallback_top_frame": {k: v for k, v in (state.get("fallback_top_frame") or {}).items() if k != "ahash"} if state.get("fallback_top_frame") else None,
        "basket_assets": basket_assets,
        "shot_manifest_path": str(shot_manifest_path),
        "story_seed_path": str(story_seed_path),
        "idea_path": str(idea_path),
        "finalized_at": utc_iso(),
        "storage_mode": STORAGE.mode,
    }
    safe_json_dump(summary_path, summary)

    pipeline_result: Dict[str, Any] = {"status": "not_started"}
    if VIDEO_PIPELINE_SCRIPT:
        cmd = [
            "python",
            VIDEO_PIPELINE_SCRIPT,
            "--input",
            str(input_media_dir),
            "--out-dir",
            str(export_root / "render_output"),
        ]
        if VIDEO_PIPELINE_BRIEF_FILE:
            cmd.extend(["--brief-file", VIDEO_PIPELINE_BRIEF_FILE])
        if VIDEO_PIPELINE_EXTRA_ARGS:
            cmd.extend(shlex.split(VIDEO_PIPELINE_EXTRA_ARGS))
        try:
            started = time.time()
            proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
            pipeline_result = {
                "status": "finished" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "latency_sec": round(time.time() - started, 2),
                "stdout_tail": proc.stdout[-3000:],
                "stderr_tail": proc.stderr[-3000:],
                "command": cmd,
            }
        except Exception as exc:
            pipeline_result = {"status": "failed", "error": str(exc), "command": cmd}

    state["story_seed"] = story_seed
    state["finalized"] = {
        "summary_path": str(summary_path),
        "story_seed_path": str(story_seed_path),
        "shot_manifest_path": str(shot_manifest_path),
        "idea_path": str(idea_path),
        "basket_assets": basket_assets,
        "shot_manifest": shot_manifest,
        "pipeline_result": pipeline_result,
        "input_media_dir": str(input_media_dir),
        "export_root": str(export_root),
        "enough_moment": enough_moment,
        "selected_count": len(shot_manifest),
        "selected_roles": [shot.get("cinematic_role") for shot in shot_manifest],
        "used_fallback_frame": used_fallback_frame,
        "story_seed": story_seed,
        "finalized_at": utc_iso(),
    }
    STORE.persist(session_id)
    await STORE.broadcast(session_id, {"type": "finalized", "finalized": state["finalized"]})

    agent = state.get("_agent")
    if agent is not None:
        await agent.close()

    return {
        "ok": True,
        "session_id": session_id,
        "finalized": state["finalized"],
    }
