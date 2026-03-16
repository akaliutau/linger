# Linger — Give Things a Second Life 🌿

<p>
 <a><img alt="Status" src="https://img.shields.io/badge/status-mvp-orange"></a>
 <a><img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-blue"></a>
 <a><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
 <a><img alt="Hackathon" src="https://img.shields.io/badge/Hackathon-Gemini_Live_Agent_Challenge-purple"></a>
</p>

**Linger** is a mobile-first, multimodal AI agent designed to reframe household reuse and spatial creativity as a mindful act rather than a chore. 
The name “Linger” suggests slowing down, noticing, and letting an object or place reveal a second life. 

Instead of a text-box chat, Linger acts as your real-time spatial creative director. It "sees" an object you are about to discard, 
scans your room for inspiration, and generates a compelling second-life concept in the form of a 15-30 second mixed-media video reel.

---

## 🏆 Hackathon Alignment: The "Beyond Text" Factor

This project was built for the **Gemini Live Agent Challenge** to develop a NEW next-generation AI Agent that utilizes multimodal inputs 
and outputs and moves beyond simple text-in/text-out interactions. 

We chose the **Creative Storyteller** category, which focuses on multimodal storytelling with interleaved output.

### 🗺️ The User Journey (How It Works)

In the current PoC, the user does not upload a batch of files or fill out a form. 
Instead, the app opens directly into a guided, beyond-text camera flow:

1. **Screen A (Intro):** Minimal landing page to "Start the story."
2. **Screen B (Hero Capture):** The user takes one clear photo of the item. The agent runs a Gemini visual interpretation and extracts a practical live capture plan.
3. **Harvest Mode (Live Session):** The user opens the phone camera in vertical mode. The app grabs stop-shots every few seconds while the user records the surrounding context.
4. **AI Scoring:** The backend analyzes each frame with Gemini Flash-Lite, ranking and keeping only the best local frames. It talks to the user with a short live coach loop.
5. **The "Enough" Moment:** Once the agent has enough strong visual evidence or got a good idea, the collection finishes.
6. **Final Review & Generation:** The app shows a compact story card (small image, short description, concise idea) and triggers the story video pipeline.

---

## 🧠 Dual AI Roles & High-Level Architecture

The architecture deliberately keeps the genuinely agentic parts as agentic, and keeps the rest deterministic. 
There are two conceptual intelligence layers:

### 1. The Scout (Capture Intelligence)
A mobile-first PWA for two-stage multimodal capture. This smart app interprets the hero frame, scores later stop-shots, and gives very short guidance. 
It acts like a cinematography coach, not just an object detector.
* **Frontend:** React app, mobile-first camera UI, uses browser `getUserMedia` APIs, and sends blobs to the backend.
* **Backend:** FastAPI app that owns session state, calls Gemini models via Google GenAI SDK, and provides frame scoring.

### 2. The Director (Storyteller Intelligence)
An async backend pipeline that takes the chosen frames, generates story ideas and storyboard structure, 
and hands off to still generation, narration, and video composition.
* **Pipeline Flow:** 
`gemini-2.5-flash-lite` interprets images -> `gemini-2.5-flash` brainstorms concepts -> 
   creates a 30-second storyboard JSON -> `gemini-2.5-flash-image` generates missing keyframes -> 
                      Cloud TTS (`Chirp 3 HD`) generates narration -> `moviepy` assembles the final MP4.

---

## 🤝 System Handoff Contract

The handoff between the Live Capture app and the Story Pipeline is a clean, composable JSON object ("The Basket"). This ensures the downstream pipeline has stable, high-quality inputs:

```json
{
  "session_id": "linger-abc123",
  "hero_image": "/session_cache/.../hero.jpg",
  "selected_frames": [
    "/session_cache/.../frame_03.jpg",
    "/session_cache/.../frame_07.jpg"
  ],
  "stage1_analysis": {
    "object_label": "Glass jar",
    "story_signal": "A humble discarded container that can become a useful home detail."
  },
  "story_seed": {
    "hook": "This nearly became trash, then found a place in the room.",
    "visual_style": "clean motion-comic",
    "generation_strategy": "4 stills + 1 optional hero clip"
  }
}

```

---

## 🛡️ Reliability & Observability

AI calls are probabilistic, so reliability matters more than purity. If the live AI path becomes unstable, 
the system is designed to degrade into a simpler guided capture flow instead of collapsing entirely.

* **Frontend:** Features explicit request timeouts, fallback camera constraints, and an always-available **Debug Drawer** for observability.
* **Backend:** Includes sync client initialization guards, storage fallbacks to local (if cloud writes fail), 
  and structured JSON logging for every major event (`trace_id`, `latency_ms`, `event_type`).

---

## ⚙️ Quick Start & Spin-Up Instructions

### 1. Dev Environment Setup

1. **Install system dependency**

MoviePy needs FFmpeg on your machine.

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: install FFmpeg and add it to `PATH`

Ensure you have the Google Cloud SDK installed and authenticated.

2. **Clone the repository**

```bash
git clone https://github.com/akaliutau/linger
cd linger
```

3. **Create and activate a Conda environment**

```bash
conda create -n linger python=3.12 -y
conda activate linger
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Cloud settings and Deployment**

We use Vertex AI + Cloud TTS with ADC:

```bash
gcloud auth application-default login
```

6. **Deploy infra and apps**

```bash
scripts/deploy_infra.sh
```

```bash
scripts/deploy_backend_cloud_run.sh
scripts/deploy_video_job.sh
```

## Technologies used

### Pipeline:

1. `gemini-2.5-flash-lite` interprets the input images.
2. `gemini-2.5-flash` brainstorms 5 story concepts.
3. `gemini-2.5-flash` creates a strict JSON storyboard for **30 seconds**.
4. `gemini-2.5-flash-image` generates only the missing keyframes.
5. `gemini-2.5-flash-lite` judges those generated images for slop / quality.
6. Cloud TTS (`Chirp 3 HD`) generates narration.
7. `moviepy` assembles a local MP4 with subtitles.

Output:
- `media_inventory.json`
- `image_interpretations.json`
- `ideas.json`
- `storyboard.json`
- `generated_image_qc.json`
- `narration.wav`
- `metrics.json`
- `final_story.mp4`

## Local test runs

1. We have created a simple PoC for the video generation pipeline which can be run locally:

```bash
python poc_story_video.py \
  --input ./input \
  --brief-file input/goals.txt \
  --out-dir ./output \
  --size 720x1280 --target-seconds 15 
```
It generates a short 15-sec clip using sample footage we've added in `input/` folder.
The cloud version is built on the same algorithm

2. We have created a simple app


## Project structure

## ⚖️ License

Linger AI is open-source software distributed under the **MIT License**. 




