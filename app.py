#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import dotenv
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
from google.genai import types

dotenv.load_dotenv()

APP_NAME = "Linger Live Capture"
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CACHE_DIR = BASE_DIR / "session_cache"
LOCAL_UPLOAD_DIR = BASE_DIR / "input"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ID = os.getenv("PROJECT_ID", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
REGION = os.getenv("REGION", "us-central1")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
LIVE_LOCATION = os.getenv("LIVE_LOCATION", "global")
DEFAULT_STORAGE_MODE: Literal["auto", "gcs", "local"] = os.getenv("STORAGE_MODE", "auto").lower()  # type: ignore[assignment]
SIGNED_URL_EXPIRY_MIN = int(os.getenv("SIGNED_URL_EXPIRY_MIN", "1440"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))
FRAME_MAX_DIM = int(os.getenv("FRAME_MAX_DIM", "1280"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))
STAGE1_MODEL = os.getenv("STAGE1_MODEL", "gemini-2.5-flash")
FRAME_MODEL = os.getenv("FRAME_MODEL", "gemini-2.5-flash-lite")
LIVE_TEXT_MODEL = os.getenv("LIVE_TEXT_MODEL", "gemini-2.0-flash-live-preview-04-09")
ENABLE_LIVE_AGENT = os.getenv("ENABLE_LIVE_AGENT", "true").lower() in {"1", "true", "yes", "on"}
KEEP_BEST_LIMIT = int(os.getenv("KEEP_BEST_LIMIT", "10"))
MIN_KEEP_SCORE = int(os.getenv("MIN_KEEP_SCORE", "74"))
MAX_RECENT_UPLOADS = int(os.getenv("MAX_RECENT_UPLOADS", "24"))

DEBUG_LOG_PROMPTS = os.getenv("DEBUG_LOG_PROMPTS", "true").lower() in {"1", "true", "yes", "on"}

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/local_uploads", StaticFiles(directory=str(LOCAL_UPLOAD_DIR)), name="local_uploads")

# ---------- helpers ----------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return now_utc().isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "session"


def json_log(event_type: str, **payload: Any) -> None:
    print(json.dumps({"ts": utc_iso(), "event_type": event_type, **payload}, ensure_ascii=False), flush=True)


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

from google import genai

SYNC_CLIENT = None

def create_sync_client():
    project_id = os.environ["PROJECT_ID"]
    location = os.environ.get("REGION", "global")
    use_vertex = str(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true")).lower() == "true"

    if not use_vertex:
        raise RuntimeError("GOOGLE_GENAI_USE_VERTEXAI must be true for this app.")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    return client

def get_sync_client():
    global SYNC_CLIENT
    if SYNC_CLIENT is None:
        SYNC_CLIENT = create_sync_client()
        json_log(
            "client_init_ok",
            project=os.environ.get("PROJECT_ID"),
            region=os.environ.get("REGION", "global"),
        )
    return SYNC_CLIENT


def get_live_client() -> Any:
    global LIVE_CLIENT
    if LIVE_CLIENT is None:
        LIVE_CLIENT = make_live_client(LIVE_LOCATION)
    return LIVE_CLIENT


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
            "You are a short, practical live visual coach for a mobile capture session. "
            "Keep replies under 20 words, concrete, calm, and cinematic. "
            "Never write paragraphs. Speak like a helpful director. "
            f"Stage 1 interpretation: {json.dumps(self.stage1_payload, ensure_ascii=False)}"
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
            return "Move a little slower and look for one clean hero angle."
        if self._session is None:
            await self.start()
        if self._session is None:
            return "Try a steadier frame and fill more of the vertical shot."

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
            return reply or "Good. Now hold for one cleaner, brighter frame."

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
                    "recent_events": [],
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

def generate_json(prompt: str, schema: Dict[str, Any], media_parts: Optional[List[Any]], model: str, temperature: float = 0.3) -> Dict[str, Any]:
    contents: List[Any] = [prompt]
    if media_parts:
        contents.extend(media_parts)
    if DEBUG_LOG_PROMPTS:
        json_log(
            "model_request",
            model=model,
            prompt_chars=len(prompt),
            media_parts=len(media_parts or []),
            approx_payload_bytes=len(prompt.encode("utf-8")) + sum(
                part_size_bytes(p) for p in media_parts or []
            ),
        )
    response = SYNC_CLIENT.models.generate_content(
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
        "You are preparing a mobile-first cinematic capture session for a later 30-second story reel. "
        "Interpret this object photo. Be practical, visual, and concise. "
        "Return strict JSON only. "
        f"User note: {note or 'No extra note.'} "
        "Focus on: what the item is, what makes it visually usable, what shots to collect next, and what to avoid."
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
        "You are judging a single frame from a live mobile capture session for a IG reel 30-second video story. "
        "Be brutal and practical. Keep only frames with clear composition and story value. "
        "Return strict JSON only. "
        f"Stage 1 interpretation: {json.dumps(stage1_payload, ensure_ascii=False)}. "
        f"Recent best frames: {json.dumps(best_summary, ensure_ascii=False)}. "
        f"Operator note: {note or 'No extra note.'}. "
        "Prefer moments that feel like hero, reveal, detail, or context shots. "
        "Micro direction must be one short spoken-friendly sentence under 12 words."
    )


def chat_prompt(stage1_payload: Dict[str, Any], best_frames: List[Dict[str, Any]], user_text: str) -> str:
    return (
        "You are a live capture coach. Reply in 1 short sentence, spoken-friendly, no fluff. "
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


def select_top_frames(existing: List[Dict[str, Any]], candidate: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    candidate_score = int(candidate.get("score_0_to_100", 0))
    candidate_hash = int(candidate["ahash"], 16)
    replaced_path: Optional[str] = None
    for index, frame in enumerate(list(existing)):
        frame_hash = int(frame["ahash"], 16)
        if hamming_distance(candidate_hash, frame_hash) <= 6:
            if candidate_score > int(frame.get("score_0_to_100", 0)) + 4:
                replaced_path = frame.get("local_path")
                existing[index] = candidate
                existing.sort(key=lambda item: int(item.get("score_0_to_100", 0)), reverse=True)
                return existing[:KEEP_BEST_LIMIT], True, replaced_path
            return existing, False, None
    existing.append(candidate)
    existing.sort(key=lambda item: int(item.get("score_0_to_100", 0)), reverse=True)
    if len(existing) > KEEP_BEST_LIMIT:
        dropped = existing.pop()
        dropped_path = dropped.get("local_path")
        return existing, dropped is candidate, dropped_path
    return existing, True, None


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
    return (response.text or "Move closer and hold for a cleaner hero frame.").strip()


# ---------- html ----------

@app.on_event("startup")
async def on_startup():
    global SYNC_CLIENT
    try:
        SYNC_CLIENT = create_sync_client()
        json_log(
            "startup_ok",
            sync_client_ready=SYNC_CLIENT is not None,
            project=os.environ.get("PROJECT_ID"),
            region=os.environ.get("REGION", "global"),
        )
    except Exception as e:
        SYNC_CLIENT = None
        print("Failed to initialize SYNC_CLIENT")
        json_log("startup_failed", error=str(e))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": APP_NAME}


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
    analysis = generate_json(
        prompt=prompt,
        schema=STAGE1_SCHEMA,
        media_parts=[media_part_from_bytes(normalized_bytes)],
        model=STAGE1_MODEL,
        temperature=0.2,
    )

    upload = STORAGE.store_bytes(
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
    state["stage1"] = {
        "note": note,
        "analysis": analysis,
        "local_path": str(local_path),
        "width": width,
        "height": height,
        "original_width": orig_width,
        "original_height": orig_height,
        "size_bytes": len(normalized_bytes),
    }
    state["stage1_upload"] = upload.__dict__
    state["best_frames"] = []
    state["live_metrics"] = {
        "frames_seen": 0,
        "frames_kept": 0,
        "latency_ms_total": 0,
        "best_score": 0,
    }
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
    await STORE.broadcast(session_id, {"type": "stage1_ready", "analysis": analysis})
    return JSONResponse(
        {
            "ok": True,
            "session_id": session_id,
            "analysis": analysis,
            "upload": upload.__dict__,
            "local_path": str(local_path),
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
    opening = state["stage1"]["analysis"].get("opening_coach_line") or "Let us find one clean hero angle first."
    if agent is not None and agent.enabled:
        try:
            opening = await agent.ask("We are starting now. Give the first short camera instruction.")
        except Exception:
            pass
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
    prompt = frame_prompt(state["stage1"]["analysis"], note or state["stage1"].get("note", ""), state.get("best_frames", []))
    analysis = generate_json(
        prompt=prompt,
        schema=FRAME_SCHEMA,
        media_parts=[media_part_from_bytes(normalized_bytes)],
        model=FRAME_MODEL,
        temperature=0.15,
    )
    score = int(analysis.get("score_0_to_100", 0))
    keep_requested = bool(analysis.get("keep")) and score >= MIN_KEEP_SCORE
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
        "preview_url": None,
    }

    selected = False
    replaced_path: Optional[str] = None
    if keep_requested:
        state["best_frames"], selected, replaced_path = select_top_frames(state.get("best_frames", []), candidate)
    else:
        try:
            local_path.unlink(missing_ok=True)
        except TypeError:
            if local_path.exists():
                local_path.unlink()

    if replaced_path and replaced_path != str(local_path):
        old_path = Path(replaced_path)
        if old_path.exists():
            old_path.unlink()

    metrics = state["live_metrics"]
    metrics["frames_seen"] += 1
    metrics["latency_ms_total"] += round((time.time() - started) * 1000)
    metrics["frames_kept"] = len(state.get("best_frames", []))
    metrics["best_score"] = max(metrics.get("best_score", 0), score)
    avg_latency_ms = int(metrics["latency_ms_total"] / max(metrics["frames_seen"], 1))

    guidance_text = analysis.get("micro_direction") or "Adjust a little and try one steadier frame."
    if score >= 88 and selected:
        guidance_text = f"Good. Keep this energy and grab one safer backup."

    recent_event = {
        "type": "frame",
        "frame_index": frame_index,
        "selected": selected,
        "score": score,
        "moment_label": analysis.get("moment_label"),
        "latency_ms": round((time.time() - started) * 1000),
        "guidance": guidance_text,
    }
    state.setdefault("recent_events", []).append(recent_event)
    state["recent_events"] = state["recent_events"][-50:]
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
        score=score,
        kept_total=len(state.get("best_frames", [])),
        avg_latency_ms=avg_latency_ms,
        local_name=local_path.name,
    )

    await STORE.broadcast(
        session_id,
        {
            "type": "frame_result",
            "selected": selected,
            "frame": {k: v for k, v in candidate.items() if k != "ahash"},
            "kept_total": len(state.get("best_frames", [])),
            "avg_latency_ms": avg_latency_ms,
            "guidance": guidance_text,
        },
    )
    return JSONResponse(
        {
            "ok": True,
            "selected": selected,
            "analysis": analysis,
            "score": score,
            "candidate": {k: v for k, v in candidate.items() if k != "ahash"},
            "kept_total": len(state.get("best_frames", [])),
            "avg_latency_ms": avg_latency_ms,
            "guidance": guidance_text,
            "latency_ms": round((time.time() - started) * 1000),
        }
    )


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
    public = {k: v for k, v in state.items() if not k.startswith("_")}
    return public


@app.post("/api/session/finalize")
async def api_session_finalize(session_id: str = Form(...), enough_moment: str = Form("enough moment")) -> Dict[str, Any]:
    session_id = slugify(session_id)
    state = STORE.get_or_create(session_id)
    if not state.get("stage1"):
        raise HTTPException(status_code=400, detail="Run stage 1 first")
    if not state.get("best_frames"):
        raise HTTPException(status_code=400, detail="No selected frames yet")

    basket_assets: List[Dict[str, Any]] = []
    basket_prefix = f"captures/{session_id}/basket"
    export_root = export_dir(session_id)
    input_media_dir = ensure_dir(export_root / "input_media")
    summary_path = export_root / "session_summary.json"

    for rank, frame in enumerate(sorted(state["best_frames"], key=lambda item: int(item.get("score_0_to_100", 0)), reverse=True), start=1):
        local_path = Path(frame["local_path"])
        if not local_path.exists():
            continue
        target_name = f"best_{rank:02d}_{local_path.name}"
        copied = input_media_dir / target_name
        copied.write_bytes(local_path.read_bytes())
        upload = STORAGE.store_bytes(
            prefix=basket_prefix,
            filename=target_name,
            data=local_path.read_bytes(),
            width=int(frame.get("width", 0)),
            height=int(frame.get("height", 0)),
            metadata={
                "session_id": session_id,
                "capture_type": "best_frame",
                "moment_label": str(frame.get("moment_label", "")),
                "score": str(frame.get("score_0_to_100", 0)),
                "created_at": utc_iso(),
            },
        )
        frame["preview_url"] = upload.view_url
        frame["basket_upload"] = upload.__dict__
        basket_assets.append(upload.__dict__)

    stage1_path = Path(state["stage1"]["local_path"])
    if stage1_path.exists():
        hero_copy = input_media_dir / f"hero_{stage1_path.name}"
        hero_copy.write_bytes(stage1_path.read_bytes())

    summary = {
        "session_id": session_id,
        "enough_moment": enough_moment,
        "stage1": state["stage1"],
        "stage1_upload": state.get("stage1_upload"),
        "best_frames": [{k: v for k, v in frame.items() if k != "ahash"} for frame in state["best_frames"]],
        "basket_assets": basket_assets,
        "finalized_at": utc_iso(),
        "storage_mode": STORAGE.mode,
    }
    safe_json_dump(summary_path, summary)

    pipeline_result: Dict[str, Any] = {"status": "not_started"}
    print(f"Should trigger a pipeline for footage in: {input_media_dir}")

    state["finalized"] = {
        "summary_path": str(summary_path),
        "basket_assets": basket_assets,
        "pipeline_result": pipeline_result,
        "input_media_dir": str(input_media_dir),
        "export_root": str(export_root),
        "enough_moment": enough_moment,
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


