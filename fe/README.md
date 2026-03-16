# Linger Story Camera — React PoC

A small React frontend for the existing FastAPI backend.

## What it does

- start with one button: **Start the story**
- opens the camera directly in the app
- first tap captures the hero image and runs stage 1 analysis
- second tap starts context harvesting from the **same** camera view
- backend keeps scoring stop-shots; frontend only shows compact score / quality feedback
- on enough good material, the app moves to a final review card
- from the review card, **Generate story** calls the existing backend finalize endpoint

## Why this frontend is simpler and more reliable

- no file picker in the main workflow
- no WebSocket required for the main happy path
- one camera surface, one linear state machine
- request timeouts on every network call
- backpressure: only one frame-analysis request in flight at a time
- safe stop after repeated model failures
- visible debug drawer for demos and troubleshooting

## Files

- `src/App.jsx` — all UI + capture loop logic in one file
- `src/styles.css` — minimalist camera-first visual system
- `vite.config.js` — local proxy to the FastAPI backend

## Run locally

### 1) Start the backend

From your existing backend project:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2) Start the React app

```bash
cd fe
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Build for demo / deployment

```bash
npm run build
```

This creates `dist/`.

You can then:

- serve `dist/` from any static host
- or copy `dist/` into your FastAPI static folder and serve both app + API from one origin

## Notes on voice

This frontend tries to use `/api/tts/guide` first.

- if your backend has that route, it will play server-generated guide audio
- otherwise it falls back to browser speech synthesis

NOTE: we didn't test thoroughly this path and on some phones we observed issues

## Observability

Tap **Debug** in the bottom-right corner to inspect:

- current phase and session id
- last network action
- last raw backend guidance
- last error
- config returned by backend
- recent app events

That is the fastest way to debug probabilistic model behavior during demos.
