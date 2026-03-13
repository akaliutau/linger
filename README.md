# Linger

### Give the things 2nd life
=========================

<p>
 <a><img alt="Status" src="https://img.shields.io/badge/status-research_prototype-6a5acd"></a>
 <a><img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-blue"></a>
 <a><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

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
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
```


## Technologies

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



