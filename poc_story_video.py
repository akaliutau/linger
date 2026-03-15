#!/usr/bin/env python3
"""
Local PoC for validating a multimodal 30s story pipeline on Google Cloud.

What it validates:
1) The model can understand input images and extract usable creative context.
2) The model can brainstorm meaningful transformation ideas.
3) The model can generate a 30-second storyboard JSON.
4) The image model can generate keyframes that are not obvious slop.
5) Cloud TTS can narrate the story.
6) A local Python renderer can assemble a final MP4 from stills + existing clips.

This is intentionally simple and opinionated.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import sys
import textwrap
import time
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import dotenv
from PIL import Image
from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips

from google import genai
from google.cloud import texttospeech
try:
    from google.cloud import storage
    STORAGE_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    storage = None  # type: ignore[assignment]
    STORAGE_IMPORT_ERROR = str(exc)
from google.genai import types

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}
DEFAULT_SIZE = (720, 1280)


@dataclass
class MediaAsset:
    file_id: str
    path: str
    kind: str  # image | video
    mime_type: str
    width: int
    height: int
    duration_sec: Optional[float] = None
    file_name: str = ""
    moment_label: str = ""
    cinematic_role: str = ""
    best_future_use: str = ""
    score_0_to_100: Optional[int] = None


class RunLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.metrics: Dict[str, Any] = {
            "started_at_epoch": time.time(),
            "steps": [],
        }

    def record_step(self, name: str, started: float, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "name": name,
            "latency_sec": round(time.time() - started, 3),
        }
        if extra:
            payload.update(extra)
        self.metrics["steps"].append(payload)
        self.save()

    def save(self) -> None:
        self.metrics["finished_at_epoch"] = time.time()
        (self.out_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")


# ---------- General helpers ----------


def debug(msg: str) -> None:
    print(f"[debug] {msg}")


def read_text(path: Optional[Path]) -> str:
    if not path:
        return ""
    return path.read_text(encoding="utf-8").strip()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    if path.suffix.lower() in IMAGE_EXTS:
        return "image/png"
    if path.suffix.lower() in VIDEO_EXTS:
        return "video/mp4"
    return "application/octet-stream"


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def media_part_from_file(path: Path) -> types.Part:
    mime_type = guess_mime(path)
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def approx_payload_bytes(prompt: str, media_paths: Optional[List[Path]] = None) -> int:
    total = len(prompt.encode("utf-8"))
    for media_path in media_paths or []:
        total += file_size_bytes(media_path)
    return total


def aspect_ratio_for_size(size: Tuple[int, int]) -> str:
    width, height = size
    if width == height:
        return "1:1"
    if height > width:
        return "9:16"
    return "16:9"


def shorten_error(exc: Exception, limit: int = 220) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).upper()
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    retry_markers = [
        "RESOURCE_EXHAUSTED",
        "RATE LIMIT",
        "TOO MANY REQUESTS",
        "TIMED OUT",
        "TIMEOUT",
        "UNAVAILABLE",
        "INTERNAL",
        "BAD GATEWAY",
        "SERVICE UNAVAILABLE",
        "GATEWAY TIMEOUT",
        "CONNECTION RESET",
        "REMOTE PROTOCOL ERROR",
    ]
    return any(marker in message for marker in retry_markers)


def retry_call(
    fn: Callable[[], Any],
    *,
    op_name: str,
    max_attempts: int = 6,
    initial_delay_sec: float = 2.0,
    max_delay_sec: float = 20.0,
) -> Any:
    delay = initial_delay_sec
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            retryable = is_retryable_error(exc)
            if attempt >= max_attempts or not retryable:
                raise
            jitter = random.uniform(0.0, min(1.0, delay * 0.2))
            sleep_for = min(max_delay_sec, delay) + jitter
            debug(
                f"{op_name} failed on attempt {attempt}/{max_attempts} with {exc.__class__.__name__}: "
                f"{shorten_error(exc)} | retrying in {sleep_for:.1f}s"
            )
            time.sleep(sleep_for)
            delay = min(max_delay_sec, delay * 2.0)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{op_name} failed without an exception.")


def existing_generated_file(scene_dir: Path, scene_id: str) -> Optional[Path]:
    matches = sorted(scene_dir.glob(f"{scene_id}_*.png"))
    return matches[0] if matches else None


# ---------- Client ----------


def make_genai_client(project: Optional[str], location: str) -> Any:
    effective_location = location or "global"
    kwargs: Dict[str, Any] = {
        "http_options": {"api_version": "v1"},
    }
    if project:
        kwargs.update({"vertexai": True, "project": project, "location": effective_location})
    debug(f"Creating GenAI client. project={project or '<env/default>'}, location={effective_location}")
    if effective_location != "global":
        debug("Using a regional Vertex endpoint. For burstier image generation, --location global is usually more resilient to 429s.")
    return genai.Client(**kwargs)


# ---------- Media discovery ----------


def normalize_input_ref(value: str) -> str:
    raw = value.strip()
    if raw.startswith("gs://"):
        return raw.rstrip("/")
    if raw.startswith("https://storage.googleapis.com/") or raw.startswith("http://storage.googleapis.com/"):
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"gs://{parts[0]}/{'/'.join(parts[1:])}".rstrip("/")
    if raw.startswith("https://storage.cloud.google.com/"):
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"gs://{parts[0]}/{'/'.join(parts[1:])}".rstrip("/")
    return raw


def is_gcs_uri(value: str) -> bool:
    return normalize_input_ref(value).startswith("gs://")


def split_gcs_uri(uri: str) -> Tuple[str, str]:
    normalized = normalize_input_ref(uri)
    if not normalized.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket_and_prefix = normalized[5:]
    bucket_name, _, prefix = bucket_and_prefix.partition("/")
    if not bucket_name:
        raise ValueError(f"Missing bucket name in URI: {uri}")
    return bucket_name, prefix.rstrip("/")


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_file_hints(bundle_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    hints: Dict[str, Dict[str, Any]] = {}
    story_seed = bundle_context.get("story_seed") or {}
    if story_seed.get("hero_image"):
        hints[Path(story_seed["hero_image"]).name] = {
            "moment_label": "Hero object",
            "cinematic_role": "hero_object",
            "best_future_use": "anchor still",
            "score_0_to_100": 100,
        }

    frame_items = story_seed.get("selected_frame_items") or (bundle_context.get("shot_manifest") or {}).get("shots") or []
    for item in frame_items:
        name = item.get("file_name") or Path(item.get("local_path") or item.get("preview_url") or "").name
        if not name:
            continue
        hints[name] = {
            "moment_label": item.get("moment_label", ""),
            "cinematic_role": item.get("cinematic_role", ""),
            "best_future_use": item.get("best_future_use", ""),
            "score_0_to_100": item.get("score_0_to_100"),
        }
    return hints


def load_bundle_context(bundle_dir: Path) -> Dict[str, Any]:
    context = {
        "story_seed": read_json_if_exists(bundle_dir / "story_seed.json"),
        "shot_manifest": read_json_if_exists(bundle_dir / "shot_manifest.json"),
        "session_summary": read_json_if_exists(bundle_dir / "session_summary.json"),
        "idea_text": read_text(bundle_dir / "idea.txt" if (bundle_dir / "idea.txt").exists() else None),
    }
    context["file_hints"] = build_file_hints(context)
    return context


def compose_pipeline_brief(base_brief: str, bundle_context: Dict[str, Any]) -> str:
    parts = []
    if base_brief:
        parts.append(base_brief.strip())

    story_seed = bundle_context.get("story_seed") or {}
    if story_seed:
        parts.append(
            textwrap.dedent(
                f"""
                Capture handoff:
                - title: {story_seed.get('title') or 'Story seed'}
                - hook: {story_seed.get('hook') or story_seed.get('idea') or ''}
                - tone: {story_seed.get('tone') or ''}
                - visual_style: {story_seed.get('visual_style') or ''}
                - generation_strategy: {story_seed.get('generation_strategy') or ''}
                - selected_roles: {', '.join(story_seed.get('selected_roles') or [])}
                """
            ).strip()
        )

    if bundle_context.get("idea_text"):
        parts.append(f"Capture operator note:\n{bundle_context['idea_text']}")

    return "\n\n".join(part for part in parts if part).strip()


def prepare_input_bundle(input_ref: str, out_dir: Path, project: Optional[str]) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    normalized = normalize_input_ref(input_ref)
    if is_gcs_uri(normalized):
        if storage is None:
            raise RuntimeError(f"google-cloud-storage is required for gs:// input: {STORAGE_IMPORT_ERROR}")
        bucket_name, prefix = split_gcs_uri(normalized)
        target_dir = ensure_dir(out_dir / "downloaded_bundle")
        client = storage.Client(project=project or None)
        blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
        if not blobs:
            raise SystemExit(f"No objects found under {normalized}")
        prefix_root = f"{prefix.rstrip('/')}/" if prefix else ""
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel_name = blob.name[len(prefix_root):] if prefix_root and blob.name.startswith(prefix_root) else Path(blob.name).name
            if not rel_name:
                continue
            local_path = target_dir / rel_name
            ensure_dir(local_path.parent)
            blob.download_to_filename(str(local_path))
        bundle_dir = target_dir
        source_info = {"input": normalized, "mode": "gcs", "local_bundle_dir": str(bundle_dir)}
    else:
        bundle_dir = Path(normalized)
        source_info = {"input": normalized, "mode": "local", "local_bundle_dir": str(bundle_dir)}

    if not bundle_dir.exists():
        raise SystemExit(f"Media directory does not exist: {bundle_dir}")

    bundle_context = load_bundle_context(bundle_dir)
    dump_json(out_dir / "input_source.json", source_info)
    dump_json(out_dir / "bundle_context.json", {k: v for k, v in bundle_context.items() if k != "idea_text"})
    if bundle_context.get("idea_text"):
        (out_dir / "bundle_idea.txt").write_text(bundle_context["idea_text"], encoding="utf-8")
    return bundle_dir, bundle_context, source_info


def discover_media(media_dir: Path, file_hints: Optional[Dict[str, Dict[str, Any]]] = None) -> List[MediaAsset]:
    assets: List[MediaAsset] = []
    file_hints = file_hints or {}
    debug(f"Scanning media directory: {media_dir}")
    asset_index = 0
    for path in sorted(p for p in media_dir.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        asset_index += 1
        hint = file_hints.get(path.name, {})
        if suffix in IMAGE_EXTS:
            with Image.open(path) as img:
                width, height = img.size
            asset = MediaAsset(
                file_id=f"asset_{asset_index:02d}",
                path=str(path),
                kind="image",
                mime_type=guess_mime(path),
                width=width,
                height=height,
                file_name=path.name,
                moment_label=str(hint.get("moment_label") or ""),
                cinematic_role=str(hint.get("cinematic_role") or ""),
                best_future_use=str(hint.get("best_future_use") or ""),
                score_0_to_100=hint.get("score_0_to_100"),
            )
            assets.append(asset)
        else:
            clip = VideoFileClip(str(path))
            try:
                width, height = clip.size
                duration_sec = float(clip.duration)
            finally:
                clip.close()
            asset = MediaAsset(
                file_id=f"asset_{asset_index:02d}",
                path=str(path),
                kind="video",
                mime_type=guess_mime(path),
                width=int(width),
                height=int(height),
                duration_sec=duration_sec,
                file_name=path.name,
                moment_label=str(hint.get("moment_label") or ""),
                cinematic_role=str(hint.get("cinematic_role") or ""),
                best_future_use=str(hint.get("best_future_use") or ""),
                score_0_to_100=hint.get("score_0_to_100"),
            )
            assets.append(asset)

        debug(
            "Discovered "
            f"{asset.file_id}: {Path(asset.path).name} | {asset.kind} | {asset.width}x{asset.height}"
            + (f" | {asset.duration_sec:.2f}s" if asset.duration_sec is not None else "")
            + (f" | role={asset.cinematic_role}" if asset.cinematic_role else "")
            + f" | {format_bytes(file_size_bytes(Path(asset.path)))}"
        )
    return assets



# ---------- Prompt schemas ----------


def image_interpretation_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "primary_subject": {"type": "STRING"},
            "room_context": {"type": "STRING"},
            "best_use_in_story": {"type": "STRING"},
            "visual_strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
            "visual_issues": {"type": "ARRAY", "items": {"type": "STRING"}},
            "usable_for": {"type": "ARRAY", "items": {"type": "STRING"}},
            "quality_score_1_to_5": {"type": "INTEGER"},
        },
        "required": [
            "primary_subject",
            "room_context",
            "best_use_in_story",
            "visual_strengths",
            "visual_issues",
            "usable_for",
            "quality_score_1_to_5",
        ],
    }


IDEAS_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "selected_idea_id": {"type": "INTEGER"},
        "ideas": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "idea_id": {"type": "INTEGER"},
                    "title": {"type": "STRING"},
                    "one_line_hook": {"type": "STRING"},
                    "transformation": {"type": "STRING"},
                    "why_it_is_meaningful": {"type": "STRING"},
                    "estimated_cost_band": {"type": "STRING"},
                    "story_potential_1_to_10": {"type": "INTEGER"},
                    "aesthetic_potential_1_to_10": {"type": "INTEGER"},
                    "feasibility_1_to_10": {"type": "INTEGER"},
                    "best_existing_assets": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": [
                    "idea_id",
                    "title",
                    "one_line_hook",
                    "transformation",
                    "why_it_is_meaningful",
                    "estimated_cost_band",
                    "story_potential_1_to_10",
                    "aesthetic_potential_1_to_10",
                    "feasibility_1_to_10",
                    "best_existing_assets",
                ],
            },
        },
    },
    "required": ["selected_idea_id", "ideas"],
}


STORYBOARD_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "logline": {"type": "STRING"},
        "visual_style": {"type": "STRING"},
        "voice_style": {"type": "STRING"},
        "target_duration_sec": {"type": "INTEGER"},
        "scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "scene_id": {"type": "STRING"},
                    "duration_sec": {"type": "NUMBER"},
                    "asset_mode": {
                        "type": "STRING",
                        "enum": ["existing_image", "existing_clip", "generated_image"],
                    },
                    "asset_ref": {"type": "STRING"},
                    "purpose": {"type": "STRING"},
                    "camera_motion": {
                        "type": "STRING",
                        "enum": [
                            "none",
                            "slow_zoom_in",
                            "slow_zoom_out",
                            "pan_left",
                            "pan_right",
                        ],
                    },
                    "narration": {"type": "STRING"},
                    "image_prompt": {"type": "STRING"},
                },
                "required": [
                    "scene_id",
                    "duration_sec",
                    "asset_mode",
                    "asset_ref",
                    "purpose",
                    "camera_motion",
                    "narration",
                    "image_prompt",
                ],
            },
        },
    },
    "required": ["title", "logline", "visual_style", "voice_style", "target_duration_sec", "scenes"],
}


IMAGE_QC_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "meaningful": {"type": "BOOLEAN"},
        "aesthetically_ok": {"type": "BOOLEAN"},
        "artifact_free_score_1_to_10": {"type": "INTEGER"},
        "prompt_alignment_score_1_to_10": {"type": "INTEGER"},
        "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "meaningful",
        "aesthetically_ok",
        "artifact_free_score_1_to_10",
        "prompt_alignment_score_1_to_10",
        "notes",
    ],
}


# ---------- Model calls ----------


def generate_json(
    client: Any,
    model: str,
    prompt: str,
    schema: Dict[str, Any],
    media_parts: Optional[List[types.Part]] = None,
) -> Dict[str, Any]:
    contents: List[Any] = [prompt]
    if media_parts:
        contents.extend(media_parts)

    def _call() -> Dict[str, Any]:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        return json.loads(response.text)

    return retry_call(
        _call,
        op_name=f"generate_json[{model}]",
        max_attempts=6,
        initial_delay_sec=2.0,
        max_delay_sec=18.0,
    )


def interpret_images(
    client: Any,
    image_assets: List[MediaAsset],
    brief: str,
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for asset in image_assets:
        started = time.time()
        asset_path = Path(asset.path)
        prompt = textwrap.dedent(
            f"""
            You are evaluating a user-provided image for a short AI-generated vertical story reel.
            The product brief is below.

            Brief:
            {brief or 'No extra brief provided.'}

            Known capture metadata:
            - moment_label: {asset.moment_label or 'unknown'}
            - cinematic_role: {asset.cinematic_role or 'unknown'}
            - best_future_use: {asset.best_future_use or 'unknown'}
            - capture_score: {asset.score_0_to_100 if asset.score_0_to_100 is not None else 'unknown'}

            Analyze this image with brutal practicality.
            Return only JSON.
            Focus on:
            - what the object/scene actually is
            - what story role this image could play in a grounded home-reuse reel
            - obvious quality issues that will hurt generation or composition
            - whether this is better as an establishing shot, detail shot, reveal shot, or proof-of-fit shot
            """
        ).strip()
        debug(
            f"Interpreting image {asset.file_id} ({asset_path.name}) | "
            f"file={format_bytes(file_size_bytes(asset_path))} | "
            f"prompt_chars={len(prompt)} | approx_payload={format_bytes(approx_payload_bytes(prompt, [asset_path]))}"
        )
        payload = generate_json(
            client=client,
            model=model,
            prompt=prompt,
            schema=image_interpretation_schema(),
            media_parts=[media_part_from_file(asset_path)],
        )
        payload["file_id"] = asset.file_id
        payload["path"] = asset.path
        payload["file_name"] = asset.file_name or asset_path.name
        payload["capture_hint"] = {
            "moment_label": asset.moment_label,
            "cinematic_role": asset.cinematic_role,
            "best_future_use": asset.best_future_use,
            "score_0_to_100": asset.score_0_to_100,
        }
        results.append(payload)
        logger.record_step(
            name=f"interpret_{asset_path.name}",
            started=started,
            extra={
                "model": model,
                "input_file": asset_path.name,
                "input_bytes": file_size_bytes(asset_path),
                "prompt_chars": len(prompt),
            },
        )
    dump_json(out_dir / "image_interpretations.json", results)
    return results


def brainstorm_ideas(
    client: Any,
    brief: str,
    interpretations: List[Dict[str, Any]],
    bundle_context: Dict[str, Any],
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> Dict[str, Any]:
    started = time.time()
    prompt = textwrap.dedent(
        f"""
        You are a creative director designing a cinematic-quality 30-second vertical video story.

        Project brief:
        {brief or 'No extra brief provided.'}

        Available interpreted assets:
        {json.dumps(interpretations, indent=2)}

        Capture handoff metadata:
        {json.dumps({
            "story_seed": bundle_context.get("story_seed"),
            "shot_manifest": bundle_context.get("shot_manifest"),
            "idea_text": bundle_context.get("idea_text"),
        }, indent=2)}

        Produce 5 ideas only.
        Constraints:
        - ideas must be visually clean and plausible, not nonsense
        - cheap to compose from existing photos, existing short clips, and a few AI-generated frames
        - avoid requiring a full 30-second generative video
        - maximize emotional clarity and the 'beyond text' feel
        - pick one best idea for the PoC
        """
    ).strip()
    debug(
        f"Brainstorming ideas | prompt_chars={len(prompt)} | "
        f"approx_payload={format_bytes(approx_payload_bytes(prompt))}"
    )
    ideas = generate_json(client, model, prompt, IDEAS_SCHEMA)
    dump_json(out_dir / "ideas.json", ideas)
    logger.record_step(
        "brainstorm_ideas",
        started,
        {"model": model, "prompt_chars": len(prompt), "ideas_count": len(ideas.get("ideas", []))},
    )
    return ideas


def plan_storyboard(
    client: Any,
    brief: str,
    interpretations: List[Dict[str, Any]],
    chosen_idea: Dict[str, Any],
    assets: List[MediaAsset],
    bundle_context: Dict[str, Any],
    target_duration_sec: int,
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> Dict[str, Any]:
    started = time.time()
    asset_manifest = [asset.__dict__ for asset in assets]
    prompt = textwrap.dedent(
        f"""
        Create a tight, competition-quality {target_duration_sec}-second storyboard for a multimodal video story.

        Brief:
        {brief or 'No extra brief provided.'}

        Chosen idea:
        {json.dumps(chosen_idea, indent=2)}

        Available assets:
        {json.dumps(asset_manifest, indent=2)}

        Image interpretations:
        {json.dumps(interpretations, indent=2)}

        Capture handoff metadata:
        {json.dumps({
            "story_seed": bundle_context.get("story_seed"),
            "shot_manifest": bundle_context.get("shot_manifest"),
            "idea_text": bundle_context.get("idea_text"),
        }, indent=2)}

        Rules:
        - total duration must be {target_duration_sec} seconds exactly after normalization
        - 4 to 6 scenes
        - prefer existing assets when possible
        - use generated_image only for scenes that truly need it
        - hero and fit-proof shots should stay grounded in the real capture when possible
        - narration should fit a {target_duration_sec}-second voiceover, so keep it concise
        - image_prompt must be production-ready and avoid text artifacts, watermark, gibberish, ugly anatomy, low detail
        - asset_ref must match a real file_id when asset_mode is existing_image or existing_clip
        - asset_ref can be an empty string for generated_image
        """
    ).strip()
    debug(
        f"Planning storyboard | prompt_chars={len(prompt)} | "
        f"approx_payload={format_bytes(approx_payload_bytes(prompt))}"
    )
    story = generate_json(client, model, prompt, STORYBOARD_SCHEMA)
    normalize_storyboard(story, assets, target_duration_sec)
    dump_json(out_dir / "storyboard.json", story)
    logger.record_step(
        "plan_storyboard",
        started,
        {"model": model, "prompt_chars": len(prompt), "scene_count": len(story.get("scenes", []))},
    )
    return story


def normalize_storyboard(story: Dict[str, Any], assets: List[MediaAsset], target_duration_sec: int) -> None:
    valid_ids = {asset.file_id for asset in assets}
    scenes = story.get("scenes", [])
    if not scenes:
        raise ValueError("Storyboard returned no scenes.")

    first_image = next((a.file_id for a in assets if a.kind == "image"), "")
    first_video = next((a.file_id for a in assets if a.kind == "video"), "")
    for scene in scenes:
        if scene["asset_mode"] == "existing_image" and scene["asset_ref"] not in valid_ids:
            scene["asset_ref"] = first_image
        elif scene["asset_mode"] == "existing_clip" and scene["asset_ref"] not in valid_ids:
            scene["asset_ref"] = first_video or first_image
            if not first_video:
                scene["asset_mode"] = "existing_image"
        elif scene["asset_mode"] == "generated_image":
            scene["asset_ref"] = ""

    durations = [max(2.0, float(scene.get("duration_sec", 5))) for scene in scenes]
    total = sum(durations)
    scaled = [d * target_duration_sec / total for d in durations]
    rounded = [round(d, 2) for d in scaled]
    diff = round(target_duration_sec - sum(rounded), 2)
    rounded[-1] = round(max(2.0, rounded[-1] + diff), 2)

    cursor = 0.0
    for scene, duration in zip(scenes, rounded):
        scene["duration_sec"] = duration
        scene["start_sec"] = round(cursor, 2)
        cursor += duration
    story["target_duration_sec"] = target_duration_sec


def extract_response_parts(response: Any) -> List[Any]:
    direct_parts = list(getattr(response, "parts", []) or [])
    if direct_parts:
        return direct_parts
    parts: List[Any] = []
    for candidate in getattr(response, "candidates", []) or []:
        parts.extend(list(getattr(getattr(candidate, "content", None), "parts", []) or []))
    return parts


def choose_scene_fallback_asset(scene: Dict[str, Any], image_assets: List[MediaAsset]) -> Optional[MediaAsset]:
    if not image_assets:
        return None
    scene_text = " ".join(
        [
            str(scene.get("purpose") or ""),
            str(scene.get("narration") or ""),
            str(scene.get("image_prompt") or ""),
        ]
    ).lower()

    def score(asset: MediaAsset) -> int:
        value = int(asset.score_0_to_100 or 0)
        role = (asset.cinematic_role or "").lower()
        name = (asset.file_name or Path(asset.path).name).lower()
        if role == "hero_object":
            value += 35
        if role == "fit_check" and any(token in scene_text for token in ["fit", "proof", "place", "shelf", "desk"]):
            value += 28
        if role == "room_opportunity" and any(token in scene_text for token in ["room", "space", "corner", "opportunity", "before"]):
            value += 24
        if role == "detail_texture" and any(token in scene_text for token in ["detail", "texture", "close", "material"]):
            value += 18
        if "hero" in name:
            value += 12
        if name.startswith("best_01"):
            value += 10
        return value

    return sorted(image_assets, key=score, reverse=True)[0]


def save_scene_image_result_files(out_dir: Path, generated_paths: Dict[str, str], scene_results: List[Dict[str, Any]]) -> None:
    dump_json(out_dir / "generated_image_map.json", generated_paths)
    dump_json(out_dir / "generated_image_results.json", scene_results)


def generate_scene_images(
    client: Any,
    story: Dict[str, Any],
    assets: List[MediaAsset],
    out_dir: Path,
    logger: RunLogger,
    model: str,
    size: Tuple[int, int],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    scene_dir = ensure_dir(out_dir / "generated_images")
    generated_paths: Dict[str, str] = {}
    scene_results: List[Dict[str, Any]] = []

    reference_images = sorted(
        [a for a in assets if a.kind == "image"],
        key=lambda asset: (asset.cinematic_role != "hero_object", -(asset.score_0_to_100 or 0), asset.file_id),
    )[:2]
    ref_paths = [Path(a.path) for a in reference_images]
    ref_parts = [media_part_from_file(path) for path in ref_paths]
    aspect_ratio = aspect_ratio_for_size(size)
    debug(
        f"Scene image generation configured for aspect_ratio={aspect_ratio} | "
        f"references={[path.name for path in ref_paths]}"
    )

    for scene in story["scenes"]:
        if scene["asset_mode"] != "generated_image":
            continue

        scene_id = scene["scene_id"]
        started = time.time()
        prompt = scene["image_prompt"]
        prompt_compact = " ".join(prompt.split())
        scene_result: Dict[str, Any] = {
            "scene_id": scene_id,
            "status": "pending",
            "path": None,
            "attempts": [],
        }

        cached_path = existing_generated_file(scene_dir, scene_id)
        if cached_path and cached_path.exists():
            generated_paths[scene_id] = str(cached_path)
            scene_result.update({
                "status": "reused_generated",
                "path": str(cached_path),
            })
            scene_results.append(scene_result)
            debug(f"Reusing previously generated image for {scene_id} -> {cached_path.name}")
            save_scene_image_result_files(out_dir, generated_paths, scene_results)
            logger.record_step(
                f"generate_image_{scene_id}",
                started,
                {
                    "model": model,
                    "status": scene_result["status"],
                    "path": cached_path.name,
                    "reference_files": [path.name for path in ref_paths],
                    "aspect_ratio": aspect_ratio,
                },
            )
            continue

        debug(
            f"Generating image for {scene_id} | prompt_chars={len(prompt)} | "
            f"ref_count={len(ref_paths)} | ref_files={[path.name for path in ref_paths]} | "
            f"approx_payload={format_bytes(approx_payload_bytes(prompt, ref_paths))}"
        )

        ref_count_plan: List[int] = []
        for count in [len(ref_parts), 1, 0]:
            if count not in ref_count_plan and count <= len(ref_parts):
                ref_count_plan.append(count)
        if not ref_count_plan:
            ref_count_plan = [0]

        image_saved = False
        last_error: Optional[Exception] = None

        for ref_count in ref_count_plan:
            used_ref_parts = ref_parts[:ref_count]
            used_ref_files = [path.name for path in ref_paths[:ref_count]]
            if ref_count == 0:
                effective_prompt = (
                    prompt_compact
                    + "\nPreserve realism and composition continuity from the user capture. No text, logos, watermark, or UI."
                )
            else:
                effective_prompt = prompt_compact

            attempt_record = {
                "ref_count": ref_count,
                "ref_files": used_ref_files,
                "status": "started",
            }

            def _call() -> Any:
                return client.models.generate_content(
                    model=model,
                    contents=[effective_prompt, *used_ref_parts],
                    config={
                        "response_modalities": ["TEXT", "IMAGE"],
                        "image_config": {"aspect_ratio": aspect_ratio},
                    },
                )

            try:
                response = retry_call(
                    _call,
                    op_name=f"generate_image[{scene_id}|refs={ref_count}]",
                    max_attempts=4 if ref_count > 0 else 3,
                    initial_delay_sec=2.0 if ref_count > 0 else 1.5,
                    max_delay_sec=16.0,
                )
                parts = extract_response_parts(response)
                for index, part in enumerate(parts):
                    if getattr(part, "inline_data", None):
                        img = part.as_image()
                        image_path = scene_dir / f"{scene_id}_{index + 1}.png"
                        img.save(image_path)
                        generated_paths[scene_id] = str(image_path)
                        scene_result.update({
                            "status": "generated",
                            "path": str(image_path),
                            "used_ref_count": ref_count,
                            "used_ref_files": used_ref_files,
                        })
                        attempt_record.update({
                            "status": "ok",
                            "path": image_path.name,
                        })
                        scene_result["attempts"].append(attempt_record)
                        debug(f"Saved generated image for {scene_id} -> {image_path.name}")
                        image_saved = True
                        break
                if image_saved:
                    break
                attempt_record.update({
                    "status": "no_image",
                    "error": "Model response contained no inline image.",
                })
                scene_result["attempts"].append(attempt_record)
                debug(f"Model returned no inline image for {scene_id} with ref_count={ref_count}; trying safer path")
            except Exception as exc:
                last_error = exc
                attempt_record.update({
                    "status": "error",
                    "error": shorten_error(exc),
                })
                scene_result["attempts"].append(attempt_record)
                debug(
                    f"Image generation path failed for {scene_id} with ref_count={ref_count}: "
                    f"{shorten_error(exc)}"
                )

        if not image_saved:
            fallback_asset = choose_scene_fallback_asset(scene, [a for a in assets if a.kind == "image"])
            if not fallback_asset:
                raise RuntimeError(
                    f"Image generation failed for {scene_id} and there is no fallback reference image. "
                    f"Last error: {shorten_error(last_error) if last_error else 'none'}"
                )
            generated_paths[scene_id] = fallback_asset.path
            scene_result.update({
                "status": "fallback_existing",
                "path": fallback_asset.path,
                "fallback_asset_file_id": fallback_asset.file_id,
                "fallback_asset_name": Path(fallback_asset.path).name,
            })
            debug(
                f"Falling back to existing asset for {scene_id} -> {Path(fallback_asset.path).name} "
                f"after generation issue: {shorten_error(last_error) if last_error else 'no image returned'}"
            )

        scene_results.append(scene_result)
        save_scene_image_result_files(out_dir, generated_paths, scene_results)
        logger.record_step(
            f"generate_image_{scene_id}",
            started,
            {
                "model": model,
                "status": scene_result["status"],
                "path": Path(scene_result["path"]).name if scene_result.get("path") else "",
                "prompt": prompt[:160],
                "prompt_chars": len(prompt),
                "reference_files": [path.name for path in ref_paths],
                "aspect_ratio": aspect_ratio,
                "attempt_count": len(scene_result["attempts"]),
            },
        )

    save_scene_image_result_files(out_dir, generated_paths, scene_results)
    return generated_paths, scene_results


def judge_generated_images(
    client: Any,
    story: Dict[str, Any],
    generated_paths: Dict[str, str],
    scene_image_results: List[Dict[str, Any]],
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> List[Dict[str, Any]]:
    result_lookup = {item.get("scene_id"): item for item in scene_image_results}
    results: List[Dict[str, Any]] = []
    for scene in story["scenes"]:
        image_path = generated_paths.get(scene["scene_id"])
        if not image_path:
            continue
        scene_result = result_lookup.get(scene["scene_id"], {})
        if scene_result.get("status") not in {"generated", "reused_generated"}:
            results.append({
                "scene_id": scene["scene_id"],
                "path": image_path,
                "skipped": True,
                "reason": scene_result.get("status") or "not_generated",
            })
            continue
        started = time.time()
        image_file = Path(image_path)
        prompt = textwrap.dedent(
            f"""
            Judge whether this generated frame is good enough for a competition demo.
            Scene purpose: {scene['purpose']}
            Intended prompt: {scene['image_prompt']}

            Be strict about slop: gibberish text, ugly artifacts, unclear subject, muddy composition.
            Return JSON only.
            """
        ).strip()
        debug(
            f"Judging generated image for {scene['scene_id']} ({image_file.name}) | "
            f"file={format_bytes(file_size_bytes(image_file))} | prompt_chars={len(prompt)} | "
            f"approx_payload={format_bytes(approx_payload_bytes(prompt, [image_file]))}"
        )
        qc = generate_json(
            client=client,
            model=model,
            prompt=prompt,
            schema=IMAGE_QC_SCHEMA,
            media_parts=[media_part_from_file(image_file)],
        )
        qc["scene_id"] = scene["scene_id"]
        qc["path"] = image_path
        results.append(qc)
        logger.record_step(
            f"judge_{scene['scene_id']}",
            started,
            {"model": model, "input_file": image_file.name, "input_bytes": file_size_bytes(image_file)},
        )
    dump_json(out_dir / "generated_image_qc.json", results)
    return results


# ---------- TTS ----------


def build_full_narration(story: Dict[str, Any]) -> str:
    lines = [scene["narration"].strip() for scene in story["scenes"] if scene.get("narration")]
    return " ".join(line for line in lines if line)


def synthesize_tts(
    text: str,
    out_path: Path,
    voice_name: str,
    speaking_rate: float,
) -> None:
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput()
    synthesis_input.text = text

    voice = texttospeech.VoiceSelectionParams()
    voice.language_code = "en-US"
    voice.name = voice_name

    audio_config = texttospeech.AudioConfig()
    audio_config.audio_encoding = texttospeech.AudioEncoding.LINEAR16
    audio_config.speaking_rate = speaking_rate

    debug(
        f"Synthesizing TTS | voice={voice_name} | speaking_rate={speaking_rate} | "
        f"text_chars={len(text)} | text_bytes={format_bytes(len(text.encode('utf-8')))}"
    )
    response = retry_call(
        lambda: client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config),
        op_name="synthesize_tts",
        max_attempts=4,
        initial_delay_sec=1.5,
        max_delay_sec=12.0,
    )
    out_path.write_bytes(response.audio_content)
    debug(f"Saved narration audio -> {out_path.name} ({format_bytes(file_size_bytes(out_path))})")


# ---------- Rendering ----------


def cover_clip(clip: Any, size: Tuple[int, int]) -> Any:
    target_w, target_h = size
    clip = clip.resized(height=target_h)
    if clip.w < target_w:
        clip = clip.resized(width=target_w)
    return clip.cropped(x_center=clip.w / 2, y_center=clip.h / 2, width=target_w, height=target_h)


def build_scene_clip(
    scene: Dict[str, Any],
    asset_path: str,
    size: Tuple[int, int],
) -> Any:
    duration = float(scene["duration_sec"])
    motion = scene.get("camera_motion", "none")
    suffix = Path(asset_path).suffix.lower()

    debug(
        f"Building scene {scene['scene_id']} | source={Path(asset_path).name} | mode={scene['asset_mode']} | "
        f"duration={duration:.2f}s | motion={motion}"
    )

    if suffix in VIDEO_EXTS:
        base = VideoFileClip(asset_path).without_audio()
        if base.duration >= duration:
            base = base.subclipped(0, duration)
        else:
            hold_frame = ImageClip(base.get_frame(max(base.duration - 0.05, 0.0))).with_duration(duration - base.duration)
            base = concatenate_videoclips([base, hold_frame], method="compose")
        clip = cover_clip(base, size)
    else:
        clip = cover_clip(ImageClip(asset_path).with_duration(duration), size)
        if motion == "slow_zoom_in":
            clip = clip.resized(lambda t: 1.0 + 0.05 * (t / max(duration, 0.01)))
            clip = cover_clip(clip, size)
        elif motion == "slow_zoom_out":
            clip = clip.resized(lambda t: 1.05 - 0.05 * (t / max(duration, 0.01)))
            clip = cover_clip(clip, size)
        elif motion == "pan_left":
            clip = clip.with_position(lambda t: (-40 * (t / max(duration, 0.01)), "center"))
        elif motion == "pan_right":
            clip = clip.with_position(lambda t: (40 * (t / max(duration, 0.01)), "center"))

    return clip.with_duration(duration)


def render_video(
    story: Dict[str, Any],
    assets: List[MediaAsset],
    generated_paths: Dict[str, str],
    narration_path: Path,
    out_dir: Path,
    size: Tuple[int, int],
    fps: int,
) -> Path:
    asset_lookup = {asset.file_id: asset for asset in assets}
    visual_clips: List[Any] = []
    video: Optional[Any] = None
    narration: Optional[Any] = None
    background: Optional[Any] = None
    final: Optional[Any] = None
    try:
        for scene in story["scenes"]:
            if scene["asset_mode"] == "generated_image":
                asset_path = generated_paths[scene["scene_id"]]
            else:
                asset_path = asset_lookup[scene["asset_ref"]].path
            visual_clips.append(build_scene_clip(scene, asset_path, size))

        video = concatenate_videoclips(visual_clips, method="compose")
        narration = AudioFileClip(str(narration_path))

        video_duration = float(video.duration)
        audio_duration = float(narration.duration)
        debug(
            f"Render prep | video_duration={video_duration:.2f}s | audio_duration={audio_duration:.2f}s | "
            f"fps={fps} | size={size[0]}x{size[1]}"
        )
        if audio_duration > video_duration:
            delta = audio_duration - video_duration
            last_scene = story["scenes"][-1]
            last_path = generated_paths.get(last_scene["scene_id"])
            if not last_path and last_scene["asset_ref"]:
                last_path = asset_lookup[last_scene["asset_ref"]].path
            extension = build_scene_clip({**last_scene, "duration_sec": delta}, last_path, size)
            visual_clips.append(extension)
            video = concatenate_videoclips([video, extension], method="compose")
            debug(f"Extended last scene by {delta:.2f}s to match narration audio")

        background = ColorClip(size=size, color=(0, 0, 0), duration=video.duration)
        final = CompositeVideoClip([background, video.with_position("center")], size=size).with_audio(narration)
        out_path = out_dir / "final_story.mp4"
        debug(f"Writing final video -> {out_path} | codec=libx264 | audio_codec=aac")
        final.write_videofile(
            str(out_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=max(1, os.cpu_count() or 1),
            logger="bar",
        )
        debug(f"Saved final video -> {out_path.name} ({format_bytes(file_size_bytes(out_path))})")
        return out_path
    finally:
        for clip in visual_clips:
            try:
                clip.close()
            except Exception:
                pass
        for clip in [final, background, video, narration]:
            if clip is None:
                continue
            try:
                clip.close()
            except Exception:
                pass


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a multimodal short-story PoC on Google Cloud from a local folder or gs:// basket.")
    parser.add_argument("--input", required=True, help="Local bundle directory or gs:// bucket/prefix with collected files and metadata.")
    parser.add_argument("--brief-file", help="Optional .txt file with the project or story brief.")
    parser.add_argument("--out-dir", default="output", help="Output directory.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID"), help="Google Cloud project id.")
    parser.add_argument("--location", default=os.getenv("REGION", "global"), help="Vertex AI location.")
    parser.add_argument("--target-seconds", type=int, default=15, help="Target story duration.")
    parser.add_argument("--fps", type=int, default=24, help="Video fps.")
    parser.add_argument("--size", default="720x1280", help="Output size, e.g. 720x1280 for vertical video.")
    parser.add_argument("--understand-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--brainstorm-model", default="gemini-2.5-flash")
    parser.add_argument("--story-model", default="gemini-2.5-flash")
    parser.add_argument("--image-model", default="gemini-2.5-flash-image")
    parser.add_argument("--judge-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--tts-voice", default="en-US-Chirp3-HD-Charon")
    parser.add_argument("--tts-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height = [int(x) for x in args.size.lower().split("x", 1)]
    size = (width, height)

    out_dir = ensure_dir(Path(args.out_dir))
    base_brief = read_text(Path(args.brief_file) if args.brief_file else None)
    logger = RunLogger(out_dir)

    debug(f"Run config | target_seconds={args.target_seconds} | size={width}x{height} | fps={args.fps}")
    if args.brief_file:
        debug(f"Using brief file: {args.brief_file}")

    bundle_dir, bundle_context, source_info = prepare_input_bundle(args.input, out_dir, args.project)
    brief = compose_pipeline_brief(base_brief, bundle_context)

    assets = discover_media(bundle_dir, bundle_context.get("file_hints"))
    if not assets:
        raise SystemExit("No supported media found. Add at least one image to the input bundle.")
    dump_json(out_dir / "media_inventory.json", [asset.__dict__ for asset in assets])
    debug(f"Media inventory written -> {out_dir / 'media_inventory.json'}")

    client = make_genai_client(project=args.project, location=args.location)

    image_assets = [asset for asset in assets if asset.kind == "image"]
    debug(f"Image assets: {[Path(asset.path).name for asset in image_assets]}")
    interpretations = interpret_images(
        client=client,
        image_assets=image_assets,
        brief=brief,
        out_dir=out_dir,
        logger=logger,
        model=args.understand_model,
    )
    print(f"{len(interpretations)} images interpreted.")

    ideas = brainstorm_ideas(
        client=client,
        brief=brief,
        interpretations=interpretations,
        bundle_context=bundle_context,
        out_dir=out_dir,
        logger=logger,
        model=args.brainstorm_model,
    )
    print(f"{len(ideas.get('ideas', []))} ideas generated.")

    selected_idea_id = ideas.get("selected_idea_id", 1)
    chosen_idea = next((item for item in ideas["ideas"] if item["idea_id"] == selected_idea_id), ideas["ideas"][0])
    print(f"Selected idea {selected_idea_id}: {chosen_idea.get('title', '<untitled>')}")

    story = plan_storyboard(
        client=client,
        brief=brief,
        interpretations=interpretations,
        chosen_idea=chosen_idea,
        assets=assets,
        bundle_context=bundle_context,
        target_duration_sec=args.target_seconds,
        out_dir=out_dir,
        logger=logger,
        model=args.story_model,
    )
    print(f"Storyboard generated with {len(story.get('scenes', []))} scenes.")
    for scene in story.get("scenes", []):
        debug(
            f"Scene {scene['scene_id']} | mode={scene['asset_mode']} | asset_ref={scene['asset_ref'] or '<generated>'} | "
            f"duration={scene['duration_sec']}s"
        )

    generated_paths, scene_image_results = generate_scene_images(
        client=client,
        story=story,
        assets=assets,
        out_dir=out_dir,
        logger=logger,
        model=args.image_model,
        size=size,
    )
    if generated_paths:
        debug(f"Resolved scene image files: {[Path(path).name for path in generated_paths.values()]}")
    else:
        debug("No generated images were needed for this storyboard.")

    judge_generated_images(
        client=client,
        story=story,
        generated_paths=generated_paths,
        scene_image_results=scene_image_results,
        out_dir=out_dir,
        logger=logger,
        model=args.judge_model,
    )

    narration_text = build_full_narration(story)
    script = out_dir / "narration.txt"
    script.write_text(narration_text, encoding="utf-8")
    print(f"Narration written to {script}")

    tts_started = time.time()
    narration_path = out_dir / "narration.wav"
    synthesize_tts(
        text=narration_text,
        out_path=narration_path,
        voice_name=args.tts_voice,
        speaking_rate=args.tts_rate,
    )
    logger.record_step(
        "tts_narration",
        tts_started,
        {"voice": args.tts_voice, "text_chars": len(narration_text), "audio_file": narration_path.name},
    )

    print("Render engine started")
    render_started = time.time()
    final_video = render_video(
        story=story,
        assets=assets,
        generated_paths=generated_paths,
        narration_path=narration_path,
        out_dir=out_dir,
        size=size,
        fps=args.fps,
    )
    logger.record_step(
        "render_video",
        render_started,
        {"output": str(final_video), "output_bytes": file_size_bytes(final_video)},
    )
    print("Rendering finished")

    summary = {
        "input_source": source_info,
        "local_input_dir": str(bundle_dir),
        "output_video": str(final_video),
        "storyboard": str(out_dir / "storyboard.json"),
        "ideas": str(out_dir / "ideas.json"),
        "metrics": str(out_dir / "metrics.json"),
        "narration": str(narration_path),
        "bundle_context": str(out_dir / "bundle_context.json"),
        "generated_image_results": str(out_dir / "generated_image_results.json"),
        "generated_image_qc": str(out_dir / "generated_image_qc.json"),
    }
    dump_json(out_dir / "run_summary.json", summary)
    logger.save()

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        dotenv.load_dotenv()
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
