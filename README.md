# 🛡️ AI Guardian

> A real-time AI-generated content detection web app — detecting deepfakes, AI images, AI text, and fake news using EfficientNet-B4 (PyTorch) with a HuggingFace fallback.

---

## 📌 Overview

AI Guardian is a Flask-based web application that analyzes media and text to determine whether it was generated or manipulated by AI. Built for a college hackathon, it combines computer vision, deep learning, and NLP to provide multi-modal detection across images, videos, audio, and text.

---

## ✨ Features

- 🖼️ **Image Deepfake Detection** — EfficientNet-B4 (PyTorch) as primary model, HuggingFace API as fallback, OpenCV noise analysis as last resort
- 🎥 **Video Analysis** — Frame-by-frame blur and artifact detection using OpenCV
- 🔊 **Audio Detection** — Pattern analysis for synthetic voice detection
- 📝 **AI Text Detection** — Structural language analysis to flag AI-generated writing
- 📰 **Fake News Detection** — Writing style suspicion scoring
- 🔁 **Graceful Fallback Chain** — App never crashes; always returns a result
- 📧 **Subscriber Notifications** — Email alert system via `subscribers.json`

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Deep Learning | PyTorch, TorchVision (EfficientNet-B4) |
| Fallback Model | HuggingFace Inference API |
| Computer Vision | OpenCV, MediaPipe |
| AI Assistant | Google Gemini API |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Heroku (Procfile + runtime.txt) |

---

## 📁 Project Structure

```
AI_Guardian/
│
├── app.py                  # Main Flask app — all routes and logic
├── model.py                # EfficientNet-B4 model definition and prediction
├── hf_fallback.py          # HuggingFace Inference API fallback
├── extract_frames.py       # Video → frame extractor for dataset prep
├── train.py                # Model training script
├── evaluate.py             # Test set evaluation script
│
├── templates/
│   └── index.html          # Main frontend UI
│
├── static/                 # CSS, JS, and static assets
│
├── subscribers.json        # Email subscriber list
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku process config
└── runtime.txt             # Python version for Heroku
```

---

## ⚙️ Detection Pipeline

When a file is uploaded, AI Guardian runs through this fallback chain automatically:

```
Upload
  │
  ▼
EfficientNet-B4 (local PyTorch model)
  │  confidence < 75%? or model missing?
  ▼
HuggingFace API  ←  dima806/deepfake_vs_real_image_detection
  │  API fails or no token?
  ▼
OpenCV Noise Analysis  (always available)
  │
  ▼
Result returned to frontend
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- A HuggingFace account (free) for the fallback API

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aryan1684/AI_Guardian.git
cd AI_Guardian

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root or set these directly:

```env
HF_TOKEN=hf_your_huggingface_token_here
XCEPTION_WEIGHTS_PATH=xception_deepfake.pt   # optional, only if you have trained weights
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Getting a HuggingFace token:** Sign up at [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New Token (read access is enough)

### Run the App

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web UI |
| `POST` | `/api/analyze` | Analyze image, video, or audio file |
| `POST` | `/api/analyze-text` | Analyze text or news content |
| `POST` | `/api/subscribe` | Subscribe to email alerts |

### `/api/analyze` — Request

```
Content-Type: multipart/form-data
file: <your file>
type: image | video | deepfake | audio
```

### `/api/analyze` — Response

```json
{
  "ai_probability": 0.87,
  "classification": "Deepfake Detected",
  "confidence": 0.87,
  "details": "EfficientNet-B4 deepfake detection model.",
  "specific_findings": ["GAN artifacts detected"],
  "recommendations": ["Flag for manual review"],
  "model_used": "EfficientNet-B4 (PyTorch)"
}
```

---

## 🤖 Models Used

### Primary — EfficientNet-B4 (PyTorch)

- Architecture: EfficientNet-B4 pretrained on ImageNet, fine-tuned for 2-class deepfake detection
- Input size: 380×380 px
- Output: `[real_prob, fake_prob]` via softmax
- Weights file: `xception_deepfake.pt` (place in project root after training)

### Fallback — HuggingFace

- Model: [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
- Triggered when: local model is missing, fails, or returns confidence < 75%
- Requires: `HF_TOKEN` environment variable

### Last Resort — OpenCV

- Method: Laplacian variance (blur/noise analysis)
- Always available, no model weights needed
- Lower accuracy but ensures the app never returns an empty result

---

## 📦 Requirements

```
flask
werkzeug
opencv-python
mediapipe
numpy
torch
torchvision
Pillow
requests
beautifulsoup4
google-generativeai
scikit-learn
```

---

## 🌐 Deployment (Heroku)

```bash
# Login and create app
heroku login
heroku create ai-guardian-app

# Set environment variables
heroku config:set HF_TOKEN=hf_yourtoken
heroku config:set GEMINI_API_KEY=your_key

# Deploy
git push heroku main
```

> ⚠️ **Note:** PyTorch (~800MB) may exceed Heroku's free slug limit. Consider using [Railway](https://railway.app) or [Render](https://render.com) for larger deployments, or store model weights on AWS S3 and download at startup.

---

## 🔧 Simulation Mode

If OpenCV or MediaPipe are not installed, the app automatically enters **simulation mode** — returning randomized but structured results so the frontend always works. This is useful for frontend development without a full ML setup.

---

## 📊 Supported File Types

| Type | Formats |
|---|---|
| Image | `.jpg`, `.jpeg`, `.png`, `.webp` |
| Video | `.mp4`, `.avi`, `.mov` |
| Audio | `.mp3`, `.wav` |

Max upload size: **200MB**

---

## 🙌 Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Forensic dataset reference
- [HuggingFace](https://huggingface.co/dima806/deepfake_vs_real_image_detection) — Fallback deepfake model
- [MediaPipe](https://mediapipe.dev) — Face landmark detection
- [Google Gemini](https://ai.google.dev) — Generative AI integration

---

## 📄 License

This project was built for a college hackathon. Feel free to fork and extend it.
