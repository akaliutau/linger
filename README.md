# Linger
### Give the Things 2nd Life

<p>
 <a><img alt="Status" src="https://img.shields.io/badge/status-mvp-orange"></a>
 <a><img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-blue"></a>
 <a><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

**Linger** is a multimodal AI agent designed to actively slow down the consumption cycle of household goods by acting as a real-time, 
spatial creative director.
We will use Gemini's multimodal vision and Live API. The app allows users to combine the information about that object (its photo), 
the pictures of room/house, and creative combinatorial power of AI to compute a highly contextual "second life" for the item. 
The idea is generated in the form of 20-sec video. 

## key features

## ⚡ Quick Start

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
- `subtitles.srt`
- `metrics.json`
- `final_story.mp4`

## Quick run

We have created a simple PoC for the video generation pipeline which can be run locally:

```bash
 python poc_story_video.py --input input/ --brief-file input/goals.txt 
```
It generates a short 15-sec clip using sample footage we've added in `input/` folder.

The cloud version is built on the same algorithm

## Brief Architecture

**Phone/PWA client**
Captures mic + camera, guides the user to take item photos, streams the room scan, shows live overlays, and later plays the generated video.

**Cloud Run gateway / orchestrator**
Owns auth, session state, uploads, and the live websocket/session bridge to Gemini Live API. It also calls helper tools for frame scoring, 
image generation, and final rendering.

**Live analysis lane**
Handles the room walkthrough: Gemini Live listens, watches, comments, and triggers tool calls. 
In parallel, a cheap frame-selection service scores candidate frames and saves the best “stop frames.”

**Story/render lane**
After the live scan, a planner turns the chosen object + chosen room frame(s) into a structured storyboard. 
Image models create edited stills, Veo makes 1–2 short hero clips, TTS generates narration, and a renderer assembles the final 30-second video.

**State + assets**
Use **Cloud Storage** for raw photos, sampled frames, stills, audio, Veo clips, and the final MP4; **Firestore** for session metadata, 
frame rankings, selected concept, and render status; and **Cloud Tasks or Pub/Sub** for async jobs. 


## Project structure

## ⚖️ License

LingerAI is open-source software distributed under the **MIT License**. 




