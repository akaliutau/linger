# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim AS fe-build
WORKDIR /src/fe
COPY fe/package*.json ./
RUN npm ci
COPY fe/ ./
RUN npm run build

FROM python:3.12-slim
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    REACT_DIST_DIR=/app/dist
WORKDIR /app
COPY requirements.app.txt ./
RUN pip install --no-cache-dir -r requirements.app.txt
COPY app.py ./app.py
COPY --from=fe-build /src/fe/dist ./dist
RUN mkdir -p /app/local_uploads /app/session_cache

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
