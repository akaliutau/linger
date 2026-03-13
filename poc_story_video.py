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
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from PIL import Image
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)

from google import genai
from google.genai import types
from google.cloud import texttospeech

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}
DEFAULT_SIZE = (1280, 720)


@dataclass
class MediaAsset:
    file_id: str
    path: str
    kind: str  # image | video
    mime_type: str
    width: int
    height: int
    duration_sec: Optional[float] = None


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


def media_part_from_file(path: Path) -> Any:
    mime_type = guess_mime(path)
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hh = millis // 3_600_000
    millis %= 3_600_000
    mm = millis // 60_000
    millis %= 60_000
    ss = millis // 1000
    millis %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{millis:03d}"


# ---------- Client ----------


def make_genai_client(project: Optional[str], location: str) -> Any:
    kwargs: Dict[str, Any] = {
        "http_options": {"api_version": "v1"},
    }
    if project:
        kwargs.update({"vertexai": True, "project": project, "location": location})
    return genai.Client(**kwargs)


# ---------- Media discovery ----------


def discover_media(media_dir: Path) -> List[MediaAsset]:
    assets: List[MediaAsset] = []
    for index, path in enumerate(sorted(p for p in media_dir.rglob("*") if p.is_file())):
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        if suffix in IMAGE_EXTS:
            with Image.open(path) as img:
                width, height = img.size
            assets.append(
                MediaAsset(
                    file_id=f"asset_{index+1:02d}",
                    path=str(path),
                    kind="image",
                    mime_type=guess_mime(path),
                    width=width,
                    height=height,
                )
            )
        else:
            clip = VideoFileClip(str(path))
            try:
                width, height = clip.size
                duration_sec = float(clip.duration)
            finally:
                clip.close()
            assets.append(
                MediaAsset(
                    file_id=f"asset_{index+1:02d}",
                    path=str(path),
                    kind="video",
                    mime_type=guess_mime(path),
                    width=int(width),
                    height=int(height),
                    duration_sec=duration_sec,
                )
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
                    "subtitle": {"type": "STRING"},
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
                    "subtitle",
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
    media_parts: Optional[List[Any]] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    contents: List[Any] = [prompt]
    if media_parts:
        contents.extend(media_parts)

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
        prompt = textwrap.dedent(
            f"""
            You are evaluating a user-provided image for a 15-second AI-generated video story.
            The product brief is below.

            Brief:
            {brief or 'No extra brief provided.'}

            Analyze this image with brutal practicality.
            Return only JSON.
            Focus on:
            - what the object/scene actually is
            - what story role this image could play
            - obvious quality issues that will hurt generation or composition
            - whether this is better as an establishing shot, detail shot, or reveal shot
            """
        ).strip()
        payload = generate_json(
            client=client,
            model=model,
            prompt=prompt,
            schema=image_interpretation_schema(),
            media_parts=[media_part_from_file(Path(asset.path))],
            temperature=0.2,
        )
        payload["file_id"] = asset.file_id
        payload["path"] = asset.path
        results.append(payload)
        logger.record_step(
            name=f"interpret_{Path(asset.path).name}",
            started=started,
            extra={"model": model},
        )
    dump_json(out_dir / "image_interpretations.json", results)
    return results


def brainstorm_ideas(
    client: Any,
    brief: str,
    interpretations: List[Dict[str, Any]],
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> Dict[str, Any]:
    started = time.time()
    prompt = textwrap.dedent(
        f"""
        You are a creative director designing a cinematic-quality 15-second video story.

        Project brief:
        {brief or 'No extra brief provided.'}

        Available interpreted assets:
        {json.dumps(interpretations, indent=2)}

        Produce 5 ideas only.
        Constraints:
        - ideas must be visually clean and plausible, not nonsense
        - cheap to compose from existing photos, existing short clips, and a few AI-generated frames
        - avoid requiring a full 15-second generative video
        - maximize emotional clarity and the 'beyond text' feel
        - pick one best idea for the PoC
        """
    ).strip()
    ideas = generate_json(client, model, prompt, IDEAS_SCHEMA, temperature=0.7)
    dump_json(out_dir / "ideas.json", ideas)
    logger.record_step("brainstorm_ideas", started, {"model": model})
    return ideas


def plan_storyboard(
    client: Any,
    brief: str,
    interpretations: List[Dict[str, Any]],
    chosen_idea: Dict[str, Any],
    assets: List[MediaAsset],
    target_duration_sec: int,
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> Dict[str, Any]:
    started = time.time()
    asset_manifest = [asset.__dict__ for asset in assets]
    prompt = textwrap.dedent(
        f"""
        Create a tight, competition-quality 15-second storyboard for a multimodal video story.

        Brief:
        {brief or 'No extra brief provided.'}

        Chosen idea:
        {json.dumps(chosen_idea, indent=2)}

        Available assets:
        {json.dumps(asset_manifest, indent=2)}

        Image interpretations:
        {json.dumps(interpretations, indent=2)}

        Rules:
        - total duration must be {target_duration_sec} seconds exactly after normalization
        - 4 to 6 scenes
        - prefer existing assets when possible
        - use generated_image only for scenes that truly need it
        - narration should fit a 15-second voiceover, so keep it concise
        - subtitles must be short and readable
        - image_prompt must be production-ready and avoid text artifacts, watermark, gibberish, ugly anatomy, low detail
        - asset_ref must match a real file_id when asset_mode is existing_image or existing_clip
        - asset_ref can be an empty string for generated_image
        """
    ).strip()
    story = generate_json(client, model, prompt, STORYBOARD_SCHEMA, temperature=0.5)
    normalize_storyboard(story, assets, target_duration_sec)
    dump_json(out_dir / "storyboard.json", story)
    logger.record_step("plan_storyboard", started, {"model": model})
    return story


def normalize_storyboard(story: Dict[str, Any], assets: List[MediaAsset], target_duration_sec: int) -> None:
    valid_ids = {asset.file_id for asset in assets}
    scenes = story.get("scenes", [])
    if not scenes:
        raise ValueError("Storyboard returned no scenes.")

    # Repair invalid refs.
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


def generate_scene_images(
    client: Any,
    story: Dict[str, Any],
    assets: List[MediaAsset],
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> Dict[str, str]:
    scene_dir = ensure_dir(out_dir / "generated_images")
    asset_lookup = {asset.file_id: asset for asset in assets}
    generated_paths: Dict[str, str] = {}

    # Use up to two reference images for consistency if available.
    reference_images = [a for a in assets if a.kind == "image"][:2]
    ref_parts = [media_part_from_file(Path(a.path)) for a in reference_images]

    for scene in story["scenes"]:
        if scene["asset_mode"] != "generated_image":
            continue
        started = time.time()
        prompt = scene["image_prompt"]
        response = client.models.generate_content(
            model=model,
            contents=[prompt, *ref_parts],
            config={
                "response_modalities": ["TEXT", "IMAGE"],
                "image_config": {"aspect_ratio": "16:9"},
            },
        )
        image_saved = False
        for index, part in enumerate(getattr(response, "parts", []) or []):
            if getattr(part, "inline_data", None):
                img = part.as_image()
                image_path = scene_dir / f"{scene['scene_id']}_{index+1}.png"
                img.save(image_path)
                generated_paths[scene["scene_id"]] = str(image_path)
                image_saved = True
                break
        if not image_saved:
            # Fallback: duplicate the first image asset if the model returned text only.
            fallback_asset = reference_images[0] if reference_images else None
            if not fallback_asset:
                raise RuntimeError("Image generation returned no image and there is no fallback reference image.")
            generated_paths[scene["scene_id"]] = fallback_asset.path
        logger.record_step(
            f"generate_image_{scene['scene_id']}",
            started,
            {"model": model, "prompt": prompt[:120]},
        )

    dump_json(out_dir / "generated_image_map.json", generated_paths)
    return generated_paths


def judge_generated_images(
    client: Any,
    story: Dict[str, Any],
    generated_paths: Dict[str, str],
    out_dir: Path,
    logger: RunLogger,
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for scene in story["scenes"]:
        image_path = generated_paths.get(scene["scene_id"])
        if not image_path:
            continue
        started = time.time()
        prompt = textwrap.dedent(
            f"""
            Judge whether this generated frame is good enough for a competition demo.
            Scene purpose: {scene['purpose']}
            Intended prompt: {scene['image_prompt']}
            Subtitle: {scene['subtitle']}

            Be strict about slop: gibberish text, ugly artifacts, unclear subject, muddy composition.
            Return JSON only.
            """
        ).strip()
        qc = generate_json(
            client=client,
            model=model,
            prompt=prompt,
            schema=IMAGE_QC_SCHEMA,
            media_parts=[media_part_from_file(Path(image_path))],
            temperature=0.1,
        )
        qc["scene_id"] = scene["scene_id"]
        qc["path"] = image_path
        results.append(qc)
        logger.record_step(f"judge_{scene['scene_id']}", started, {"model": model})
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
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=speaking_rate,
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    out_path.write_bytes(response.audio_content)


# ---------- Rendering ----------


def cover_clip(clip: Any, size: Tuple[int, int]) -> Any:
    target_w, target_h = size
    clip = clip.resized(height=target_h)
    if clip.w < target_w:
        clip = clip.resized(width=target_w)
    return clip.cropped(x_center=clip.w / 2, y_center=clip.h / 2, width=target_w, height=target_h)


def make_text_overlay(text: str, duration: float, size: Tuple[int, int]) -> Any:
    safe_text = (text or "").strip()
    if not safe_text:
        return None
    return (
        TextClip(
            text=safe_text,
            font_size=44,
            size=(size[0] - 120, None),
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            text_align="center",
            transparent=True,
            duration=duration,
        )
        .with_position(("center", size[1] - 180))
        .with_duration(duration)
    )


def build_scene_clip(
    scene: Dict[str, Any],
    asset_path: str,
    size: Tuple[int, int],
) -> Any:
    duration = float(scene["duration_sec"])
    motion = scene.get("camera_motion", "none")
    suffix = Path(asset_path).suffix.lower()

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

    overlay = make_text_overlay(scene.get("subtitle", ""), duration, size)
    if overlay is None:
        return clip.with_duration(duration)
    return CompositeVideoClip([clip.with_duration(duration), overlay], size=size).with_duration(duration)


def write_srt(story: Dict[str, Any], path: Path) -> None:
    chunks: List[str] = []
    for idx, scene in enumerate(story["scenes"], start=1):
        start = float(scene["start_sec"])
        end = start + float(scene["duration_sec"])
        text = scene.get("subtitle") or scene.get("narration") or ""
        chunks.append(f"{idx}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    path.write_text("\n".join(chunks), encoding="utf-8")


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
    visual_clips = []
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
    if audio_duration > video_duration:
        delta = audio_duration - video_duration
        last_scene = story["scenes"][-1]
        last_path = generated_paths.get(last_scene["scene_id"])
        if not last_path and last_scene["asset_ref"]:
            last_path = asset_lookup[last_scene["asset_ref"]].path
        extension = build_scene_clip({**last_scene, "duration_sec": delta}, last_path, size)
        video = concatenate_videoclips([video, extension], method="compose")

    background = ColorClip(size=size, color=(0, 0, 0), duration=video.duration)
    final = CompositeVideoClip([background, video.with_position("center")], size=size).with_audio(narration)
    out_path = out_dir / "final_story.mp4"
    final.write_videofile(
        str(out_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        threads=max(1, os.cpu_count() or 1),
        logger="bar",
    )

    narration.close()
    final.close()
    video.close()
    for clip in visual_clips:
        clip.close()
    return out_path


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a local multimodal 30-second story PoC on Google Cloud.")
    parser.add_argument("--input", required=True, help="Directory with input photos and optional short clips.")
    parser.add_argument("--brief-file", help="Optional .txt file with the project or story brief.")
    parser.add_argument("--out-dir", default="output", help="Output directory.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID"), help="Google Cloud project id.")
    parser.add_argument("--location", default=os.getenv("REGION", "global"), help="Vertex AI location.")
    parser.add_argument("--target-seconds", type=int, default=15, help="Target story duration.")
    parser.add_argument("--fps", type=int, default=24, help="Video fps.")
    parser.add_argument("--size", default="1280x720", help="Output size, e.g. 1280x720.")
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

    media_dir = Path(args.input)
    out_dir = ensure_dir(Path(args.out_dir))
    brief = read_text(Path(args.brief_file) if args.brief_file else None)
    logger = RunLogger(out_dir)

    if not media_dir.exists():
        raise SystemExit(f"Media directory does not exist: {media_dir}")

    assets = discover_media(media_dir)
    if not assets:
        raise SystemExit("No supported media found. Add at least one image to the media directory.")
    dump_json(out_dir / "media_inventory.json", [asset.__dict__ for asset in assets])

    client = make_genai_client(project=args.project, location=args.location)

    image_assets = [asset for asset in assets if asset.kind == "image"]
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
        out_dir=out_dir,
        logger=logger,
        model=args.brainstorm_model,
    )
    print(f"{len(ideas)} ideas generated.")

    selected_idea_id = ideas.get("selected_idea_id", 1)
    chosen_idea = next((item for item in ideas["ideas"] if item["idea_id"] == selected_idea_id), ideas["ideas"][0])
    print(f"{selected_idea_id} selected.")

    story = plan_storyboard(
        client=client,
        brief=brief,
        interpretations=interpretations,
        chosen_idea=chosen_idea,
        assets=assets,
        target_duration_sec=args.target_seconds,
        out_dir=out_dir,
        logger=logger,
        model=args.story_model,
    )
    print(f"sections of story generated: {story.keys()}")

    generated_paths = generate_scene_images(
        client=client,
        story=story,
        assets=assets,
        out_dir=out_dir,
        logger=logger,
        model=args.image_model,
    )

    judge_generated_images(
        client=client,
        story=story,
        generated_paths=generated_paths,
        out_dir=out_dir,
        logger=logger,
        model=args.judge_model,
    )

    narration_text = build_full_narration(story)
    script = out_dir / "narration.txt"
    script.write_text(narration_text, encoding="utf-8")
    print(f"narration written to {script}")

    tts_started = time.time()
    narration_path = out_dir / "narration.wav"
    synthesize_tts(
        text=narration_text,
        out_path=narration_path,
        voice_name=args.tts_voice,
        speaking_rate=args.tts_rate,
    )
    logger.record_step("tts_narration", tts_started, {"voice": args.tts_voice})

    write_srt(story, out_dir / "subtitles.srt")
    print(f"subtitles written to {out_dir / 'subtitles.srt'}")

    print(f"render engine started")
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
    logger.record_step("render_video", render_started, {"output": str(final_video)})
    print(f"rendering finished")

    summary = {
        "output_video": str(final_video),
        "storyboard": str(out_dir / "storyboard.json"),
        "ideas": str(out_dir / "ideas.json"),
        "metrics": str(out_dir / "metrics.json"),
        "narration": str(narration_path),
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
