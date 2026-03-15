#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${REGION:?REGION must be set}"

INPUT_URI="${1:-}"
OUTPUT_GCS_URI="${2:-}"
JOB_NAME="${VIDEO_JOB_NAME:-linger-video-job}"
WAIT_FLAG="${WAIT_FLAG:---wait}"

if [[ -z "${INPUT_URI}" ]]; then
  echo "Usage: $0 gs://bucket/path/to/basket gs://bucket/path/to/output-prefix" >&2
  exit 1
fi

if [[ -z "${OUTPUT_GCS_URI}" ]]; then
  run_id="$(date +%Y%m%d-%H%M%S)"
  OUTPUT_GCS_URI="gs://${BUCKET_NAME}/renders/${run_id}"
fi

gcloud run jobs execute "${JOB_NAME}" \
  --region "${REGION}" \
  ${WAIT_FLAG} \
  --update-env-vars "INPUT_URI=${INPUT_URI},OUTPUT_GCS_URI=${OUTPUT_GCS_URI}"

echo "Execution started for ${JOB_NAME}"
echo "Input : ${INPUT_URI}"
echo "Output: ${OUTPUT_GCS_URI}"
