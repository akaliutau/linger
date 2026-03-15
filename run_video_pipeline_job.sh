#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_URI:?INPUT_URI is required}"
: "${OUTPUT_GCS_URI:?OUTPUT_GCS_URI is required}"

WORKDIR="${WORKDIR:-/tmp/linger-output}"
LOCATION="${LOCATION:-${VERTEX_LOCATION:-${REGION:-global}}}"
TARGET_SECONDS="${TARGET_SECONDS:-15}"
FPS="${FPS:-24}"
SIZE="${SIZE:-720x1280}"

rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

args=(
  python /app/poc_story_video.py
  --input "${INPUT_URI}"
  --out-dir "${WORKDIR}"
  --target-seconds "${TARGET_SECONDS}"
  --fps "${FPS}"
  --size "${SIZE}"
)

if [[ -n "${PROJECT_ID:-}" ]]; then
  args+=(--project "${PROJECT_ID}")
fi
if [[ -n "${LOCATION:-}" ]]; then
  args+=(--location "${LOCATION}")
fi
if [[ -n "${BRIEF_FILE:-}" ]]; then
  args+=(--brief-file "${BRIEF_FILE}")
fi
if [[ -n "${UNDERSTAND_MODEL:-}" ]]; then
  args+=(--understand-model "${UNDERSTAND_MODEL}")
fi
if [[ -n "${BRAINSTORM_MODEL:-}" ]]; then
  args+=(--brainstorm-model "${BRAINSTORM_MODEL}")
fi
if [[ -n "${STORY_MODEL:-}" ]]; then
  args+=(--story-model "${STORY_MODEL}")
fi
if [[ -n "${IMAGE_MODEL:-}" ]]; then
  args+=(--image-model "${IMAGE_MODEL}")
fi
if [[ -n "${JUDGE_MODEL:-}" ]]; then
  args+=(--judge-model "${JUDGE_MODEL}")
fi
if [[ -n "${TTS_VOICE:-}" ]]; then
  args+=(--tts-voice "${TTS_VOICE}")
fi

echo "[job] Running: ${args[*]}"
"${args[@]}"

echo "[job] Uploading ${WORKDIR} -> ${OUTPUT_GCS_URI}"
upload_args=(python /app/upload_dir_to_gcs.py --local-dir "${WORKDIR}" --output-uri "${OUTPUT_GCS_URI}")
if [[ -n "${PROJECT_ID:-}" ]]; then
  upload_args+=(--project "${PROJECT_ID}")
fi
"${upload_args[@]}"

echo "[job] Done. Output available at ${OUTPUT_GCS_URI}"
