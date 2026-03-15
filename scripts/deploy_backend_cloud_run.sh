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

SERVICE_NAME="${SERVICE_NAME:-linger-app}"
AR_REPO="${AR_REPO:-linger}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
SA_NAME="${SA_NAME:-${PROJECT_ID}-vertex-sa}"
SA_EMAIL="${SA_EMAIL:-${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:${TAG}"
BUILD_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

mkdir -p "${BUILD_DIR}/fe"
cp "${ROOT_DIR}/app.py" "${BUILD_DIR}/"
cp "${ROOT_DIR}/static/generating_reel.mp4" "${BUILD_DIR}/static/"
cp "${ROOT_DIR}/fe/package.json" "${BUILD_DIR}/fe/"
if [[ -f "${ROOT_DIR}/fe/package-lock.json" ]]; then
  cp "${ROOT_DIR}/fe/package-lock.json" "${BUILD_DIR}/fe/"
fi
cp -R "${ROOT_DIR}/fe/src" "${BUILD_DIR}/fe/"
cp "${ROOT_DIR}/fe/index.html" "${BUILD_DIR}/fe/"
cp "${ROOT_DIR}/fe/vite.config.js" "${BUILD_DIR}/fe/"
cp "${ROOT_DIR}/requirements.app.txt" "${BUILD_DIR}/"
cp "${ROOT_DIR}/Dockerfile.app" "${BUILD_DIR}/Dockerfile"

# Optional runtime permissions required by this app.
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

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --allow-unauthenticated \
  --port 8080 \
  --memory "${APP_MEMORY:-1Gi}" \
  --cpu "${APP_CPU:-1}" \
  --concurrency "${APP_CONCURRENCY:-8}" \
  --max-instances "${APP_MAX_INSTANCES:-3}" \
  --timeout "${APP_TIMEOUT:-1800}" \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},BUCKET_NAME=${BUCKET_NAME},REGION=${REGION},STORAGE_MODE=gcs,VERTEX_LOCATION=${VERTEX_LOCATION:-global},LIVE_LOCATION=${LIVE_LOCATION:-global},GUIDE_TTS_ENABLED=${GUIDE_TTS_ENABLED:-true},VIDEO_PIPELINE_TRIGGER_MODE=${VIDEO_PIPELINE_TRIGGER_MODE:-mock},VIDEO_PIPELINE_SUBMIT_URL=${VIDEO_PIPELINE_SUBMIT_URL:-https://console.cloud.google.com/run/jobs/details/${REGION}/${VIDEO_JOB_NAME:-linger-video-job}?project=${PROJECT_ID}}"

echo "Deployed ${SERVICE_NAME} -> ${IMAGE_URI}"
