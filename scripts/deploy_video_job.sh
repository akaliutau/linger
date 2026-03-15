#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "project dir: ${ROOT_DIR}"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${REGION:?REGION must be set}"
: "${BUCKET_NAME:?BUCKET_NAME must be set}"

JOB_NAME="${VIDEO_JOB_NAME:-linger-video-job}"
AR_REPO="${AR_REPO:-linger}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
SA_NAME="${SA_NAME:-${PROJECT_ID}-vertex-sa}"
SA_EMAIL="${SA_EMAIL:-${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${JOB_NAME}:${TAG}"
BUILD_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

cp "${ROOT_DIR}/poc_story_video.py" "${BUILD_DIR}/"
cp "${ROOT_DIR}/requirements.job.txt" "${BUILD_DIR}/"
cp "${ROOT_DIR}/Dockerfile.job" "${BUILD_DIR}/Dockerfile"
cp "${ROOT_DIR}/run_video_pipeline_job.sh" "${BUILD_DIR}/"
cp "${ROOT_DIR}/upload_dir_to_gcs.py" "${BUILD_DIR}/"

# Optional runtime permissions required by the job.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" >/dev/null

gcloud artifacts repositories describe "${AR_REPO}" \
  --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${AR_REPO}" \
    --location="${REGION}" \
    --repository-format=docker \
    --description="Container images for Linger"

gcloud builds submit "${BUILD_DIR}" --tag "${IMAGE_URI}"

gcloud run jobs deploy "${JOB_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --memory "${JOB_MEMORY:-4Gi}" \
  --cpu "${JOB_CPU:-2}" \
  --tasks 1 \
  --parallelism 1 \
  --max-retries "${JOB_MAX_RETRIES:-0}" \
  --task-timeout "${JOB_TIMEOUT:-3600}" \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},REGION=${REGION},LOCATION=${VIDEO_LOCATION:-global},TARGET_SECONDS=${TARGET_SECONDS:-15},FPS=${FPS:-24},SIZE=${SIZE:-720x1280},UNDERSTAND_MODEL=${UNDERSTAND_MODEL:-gemini-2.5-flash-lite},BRAINSTORM_MODEL=${BRAINSTORM_MODEL:-gemini-2.5-flash},STORY_MODEL=${STORY_MODEL:-gemini-2.5-flash},IMAGE_MODEL=${IMAGE_MODEL:-gemini-2.5-flash-image},JUDGE_MODEL=${JUDGE_MODEL:-gemini-2.5-flash-lite},TTS_VOICE=${TTS_VOICE:-en-US-Chirp3-HD-Charon}"

echo "Deployed ${JOB_NAME} -> ${IMAGE_URI}"
