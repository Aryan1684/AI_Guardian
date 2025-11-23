import os
import json
import hashlib
import re
import tempfile
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import requests
from bs4 import BeautifulSoup

from PIL import Image
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("⚠️ google-generativeai not installed. Gemini features disabled.")

# ============================================
# FLASK APP INITIALIZATION
# ============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# ============================================
# EMAIL STORAGE (FILE-BASED)
# ============================================

EMAIL_FILE = 'subscribers.json'


def load_subscribers():
    """Load subscribers from JSON file"""
    if os.path.exists(EMAIL_FILE):
        try:
            with open(EMAIL_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading subscribers: {e}")
            return []
    return []


def save_subscriber(email):
    """Save single subscriber to file"""
    subscribers = load_subscribers()

    # Check if email already exists
    existing_emails = [s['email'] if isinstance(s, dict) else s for s in subscribers]
    if email in existing_emails:
        return False

    new_subscriber = {
        'email': email,
        'subscribed_at': datetime.now().isoformat(),
        'ip_address': request.remote_addr if request else None,
        'status': 'active'
    }

    subscribers.append(new_subscriber)

    try:
        with open(EMAIL_FILE, 'w') as f:
            json.dump(subscribers, f, indent=2)
        print(f"✅ Saved subscriber to {EMAIL_FILE}")
        return True
    except Exception as e:
        print(f"❌ Error saving subscriber: {e}")
        return False


def get_subscriber_emails():
    subscribers = load_subscribers()
    if subscribers and isinstance(subscribers[0], dict):
        return [s['email'] for s in subscribers]
    return subscribers


def get_subscriber_count():
    return len(load_subscribers())


# ============================================
# GEMINI AI CONFIGURATION
# ============================================

# ============================================
# GEMINI AI CONFIGURATION WITH SMART MODEL FALLBACK
# ============================================

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

def init_gemini_model():
    if not GEMINI_API_KEY or not genai:
        print("⚠️ Gemini API key not found or package missing.")
        return None
    
    genai.configure(api_key=GEMINI_API_KEY)

    preferred_models = [
        "models/gemini-2.5-flash-image",     # 🎯 Best for image/video deepfake analysis
        "models/gemini-2.5-pro",             # 🧠 High-accuracy AI content inspection
        "models/gemini-2.5-flash-live-preview",  # 📹 Live detection / streaming
        "models/gemini-2.0-flash"            # ⚡ Fast fallback
    ]
    
    for model_name in preferred_models:
        try:
            print(f"🔍 Trying Gemini Model: {model_name}")
            return genai.GenerativeModel(model_name)
        except Exception as e:
            print(f"⚠️ {model_name} not available: {str(e)}")

    print("🚨 No valid Gemini model found. Using local ML only.")
    return None

# Initialize Gemini model
gemini_model = init_gemini_model()

# ============================================
# FILE TYPE CONFIGURATION
# ============================================

ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'avi', 'mov', 'mkv'},
    'image': {'jpg', 'jpeg', 'png', 'webp', 'gif'},
    'audio': {'mp3', 'wav', 'm4a', 'ogg'}
}


def allowed_file(filename, file_type):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """Analyze uploaded file WITHOUT storing it permanently"""
    temp_filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        file_type = request.form.get('type', 'video')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, file_type):
            return jsonify({'error': f'Invalid file type for {file_type}'}), 400

        print(f"📤 Received {file_type}: {file.filename}")

        # Create temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name

        print(f"💾 Temporary file created: {temp_filepath}")

        # Route based on type
        if file_type == 'video':
            result = analyze_video(temp_filepath)
        elif file_type == 'image':
            result = analyze_image(temp_filepath)
        elif file_type == 'audio':
            result = analyze_audio(temp_filepath)
        elif file_type == 'deepfake':
            result = analyze_deepfake(temp_filepath)
        else:
            result = {'error': 'Unknown file type'}

        print(f"✅ Analysis complete: {result.get('classification', 'N/A')}")
        return jsonify(convert_to_serializable(result))


    except Exception as e:
        print(f"❌ Error in analyze_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.unlink(temp_filepath)
                print(f"🗑️  Deleted temporary file: {temp_filepath}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete temp file: {e}")


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text content"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        analysis_type = data.get('type', 'text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text) < 50:
            return jsonify({'error': 'Text too short (minimum 50 characters)'}), 400

        print(f"📝 Analyzing {len(text)} characters of {analysis_type}")

        if analysis_type == 'news':
            result = analyze_news_text(text)
        else:
            result = analyze_text_content(text)

        print(f"✅ Text analysis complete: {result.get('classification', 'N/A')}")
        return jsonify(convert_to_serializable(result))


    except Exception as e:
        print(f"❌ Error in analyze_text: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze news article from URL"""
    try:
        data = request.get_json()
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400

        print(f"🌐 Fetching URL: {url}")
        result = fetch_and_analyze_url(url)

        print(f"✅ URL analysis complete")
        return jsonify(convert_to_serializable(result))


    except Exception as e:
        print(f"❌ Error in analyze_url: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/subscribe', methods=['POST'])
def subscribe_email():
    """Subscribe email for updates"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return jsonify({'error': 'Invalid email format'}), 400

        current_subscribers = get_subscriber_emails()
        if email in current_subscribers:
            print(f"📧 Duplicate subscription attempt: {email}")
            return jsonify({
                'success': True,
                'message': 'You\'re already on our list! 🎉',
                'already_subscribed': True,
                'subscriber_count': len(current_subscribers)
            })

        if save_subscriber(email):
            new_count = get_subscriber_count()
            print(f"📧 New subscriber: {email} (Total: {new_count})")
            return jsonify({
                'success': True,
                'message': f'Thanks for subscribing! 🚀',
                'subscriber_count': new_count
            })
        else:
            return jsonify({'error': 'Failed to save subscription'}), 500

    except Exception as e:
        print(f"❌ Error in subscribe_email: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================
# ADVANCED DEEPFAKE / VIDEO ANALYSIS
# ============================================

def analyze_deepfake(filepath):
    """Main deepfake analysis entry – video or image"""
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS['video']:
        return analyze_video(filepath)
    else:
        return analyze_image(filepath)


def analyze_video(filepath):
    """
    Hybrid video analysis:
    - Local ML: OpenCV + face_recognition (blur, consistency)
    - Optional cloud: Gemini
    """
    try:
        print("🎥 Starting advanced video deepfake analysis...")
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise RuntimeError("Cannot read video file")

        frame_step = max(1, total_frames // 30)
        blur_scores = []
        face_consistency_scores = []
        prev_encoding = None
        frames_used = 0

        while True:
            frame_idx = frames_used * frame_step
            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frames_used += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb, model='hog')
            if face_locs:
                encodings = face_recognition.face_encodings(rgb, face_locs)
                if encodings:
                    if prev_encoding is not None:
                        dist = face_recognition.face_distance([prev_encoding], encodings[0])[0]
                        face_consistency_scores.append(1 - dist)
                    prev_encoding = encodings[0]

        cap.release()

        avg_blur = np.mean(blur_scores) if blur_scores else 0
        avg_consistency = np.mean(face_consistency_scores) if face_consistency_scores else 0.5

        local_prob = 0.15
        if avg_blur < 100:
            local_prob += 0.3
        if avg_consistency < 0.7:
            local_prob += 0.3

        local_prob = max(0.05, min(0.95, local_prob))

        api_prob = call_gemini_deepfake(filepath, content_type="video")
        final_prob = round(0.6 * local_prob + 0.4 * api_prob, 3)

        classification = classify_probability(final_prob)

        return {
            'ai_probability': final_prob,
            'classification': classification,
            'confidence': round(0.6 + abs(final_prob - 0.5), 3),
            'details': f"Frames used: {frames_used}, Blur: {avg_blur:.1f}, Face consistency: {avg_consistency:.2f}",
            'specific_findings': [
                f"Average blur: {avg_blur:.1f}",
                f"Face consistency score: {avg_consistency:.2f}",
                "Local ML + Gemini hybrid detection"
            ],
            'recommendations': generate_recommendations(final_prob)
        }

    except Exception as e:
        print(f"❌ Error in analyze_video: {e}")
        return get_fallback_result("video")


def analyze_image(filepath):
    """
    Hybrid image analysis:
    - Local ML: DeepFace + OpenCV stats
    - Optional cloud: Gemini
    """
    try:
        print("🖼️ Starting advanced image deepfake analysis...")

        img_cv = cv2.imread(filepath)
        if img_cv is None:
            raise RuntimeError("Cannot read image")

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        h, w = gray.shape[:2]
        resolution = w * h

        # DeepFace analysis
        deepface_prob_boost = 0.0
        deepface_info = {}
        try:
            df_res = DeepFace.analyze(
                img_path=filepath,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False
            )
            if isinstance(df_res, list):
                df_res = df_res[0]
            deepface_info = {
                'age': df_res.get('age'),
                'gender': df_res.get('gender'),
                'emotion': df_res.get('dominant_emotion'),
                'face_confidence': df_res.get('face_confidence', 0.8)
            }
            if df_res.get('face_confidence', 0.8) < 0.7:
                deepface_prob_boost += 0.2
        except Exception as e:
            print(f"⚠️ DeepFace analysis failed: {e}")

        local_prob = 0.2 + deepface_prob_boost
        if blur_score < 100:
            local_prob += 0.25
        if brightness < 50 or brightness > 200:
            local_prob += 0.15
        if resolution in [512 * 512, 768 * 768, 1024 * 1024]:
            local_prob += 0.2

        local_prob = max(0.05, min(0.95, local_prob))

        api_prob = call_gemini_deepfake(filepath, content_type="image")
        final_prob = round(0.6 * local_prob + 0.4 * api_prob, 3)

        classification = classify_probability(final_prob)

        findings = [
            f"Blur score: {blur_score:.1f}",
            f"Brightness: {brightness:.1f}",
            f"Resolution: {w}x{h} ({resolution} px)"
        ]
        if deepface_info:
            findings.append(f"DeepFace -> age: {deepface_info.get('age')}, "
                            f"gender: {deepface_info.get('gender')}, "
                            f"emotion: {deepface_info.get('emotion')}")

        return {
            'ai_probability': final_prob,
            'classification': classification,
            'confidence': round(0.6 + abs(final_prob - 0.5), 3),
            'details': f"Advanced image analysis (OpenCV + DeepFace + Gemini).",
            'specific_findings': findings,
            'recommendations': generate_recommendations(final_prob),
            'deepface': deepface_info
        }

    except Exception as e:
        print(f"❌ Error in analyze_image: {e}")
        return get_fallback_result("image")


# ============================================
# AUDIO ANALYSIS
# ============================================

def analyze_audio(filepath):
    """Simple heuristic + optional Gemini audio analysis"""
    try:
        file_stats = os.stat(filepath)
        file_size = file_stats.st_size

        base_prob = 0.18
        if file_size < 50_000:
            base_prob += 0.15
        elif file_size > 5_000_000:
            base_prob -= 0.05

        api_prob = call_gemini_audio(filepath)
        final_prob = round(0.5 * base_prob + 0.5 * api_prob, 3)

        classification = classify_probability(final_prob, mode='audio')

        return {
            'ai_probability': final_prob,
            'classification': classification,
            'confidence': round(0.6 + abs(final_prob - 0.5), 3),
            'details': f"Audio size: {file_size / 1024:.1f} KB",
            'specific_findings': ["Basic acoustic pattern analysis", "Optional Gemini speech analysis"],
            'recommendations': generate_recommendations(final_prob)
        }
    except Exception as e:
        print(f"❌ Error in analyze_audio: {e}")
        return get_fallback_result("audio")


# ============================================
# TEXT / NEWS ANALYSIS
# ============================================

def analyze_text_content(text):
    """Analyze text for AI generation (simple heuristic + optional Gemini)"""
    if gemini_model:
        try:
            prompt = f"""You are an AI text detector. Analyze if the following text is AI-generated.
Return ONLY JSON: {{"ai_probability": float, "classification": str, "details": str}}
TEXT:
{text[:2000]}"""
            response = gemini_model.generate_content(prompt)
            result = parse_ai_response(response.text)
            if 'ai_probability' not in result:
                raise ValueError("No ai_probability from Gemini")
            return result
        except Exception as e:
            print(f"⚠️ Gemini text analysis failed: {e}")

    # Fallback smart heuristic
    return analyze_text_smart(text)


def analyze_text_smart(text):
    indicators = {
        'delve into': 0.15, 'multifaceted': 0.10, 'paradigm': 0.12,
        'leverage': 0.08, 'synergy': 0.10, 'holistic': 0.11,
        'furthermore': 0.04, 'moreover': 0.04
    }

    text_lower = text.lower()
    indicator_score = sum(weight for phrase, weight in indicators.items() if phrase in text_lower)
    found = [phrase for phrase in indicators if phrase in text_lower]

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_len = sum(len(s.split()) for s in sentences) /max(len(sentences), 1)
    length_factor = 0.08 if 15 < avg_len < 25 else -0.05

    personal_count = text_lower.count(' i ') + text_lower.count(' my ')
    personal_factor = -0.08 if personal_count > 2 else 0.05

    ai_prob = 0.25 + indicator_score + length_factor + personal_factor
    ai_prob = max(0.05, min(0.95, ai_prob))

    classification = (
        "Likely Human-Written" if ai_prob < 0.30 else
        "Possibly AI-Assisted" if ai_prob < 0.50 else
        "Likely AI-Generated" if ai_prob < 0.70 else
        "AI-Generated"
    )

    findings = [f"Found {len(found)} AI-typical phrases"] if found else ["Natural writing style"]

    return {
        'ai_probability': round(ai_prob, 3),
        'classification': classification,
        'confidence': 0.75,
        'score': round(ai_prob, 3),
        'details': f"Text analysis: {len(text)} chars, avg sentence: {int(avg_len)} words",
        'specific_findings': findings,
        'recommendations': generate_recommendations(ai_prob)
    }


def analyze_news_text(text):
    return analyze_news_smart(text)


def analyze_news_smart(text):
    text_lower = text.lower()

    sensational = {
        'shocking': 0.12, "you won't believe": 0.18, 'doctors hate': 0.20,
        'one simple trick': 0.20, 'secret': 0.08, 'breaking': 0.05,
        'miracle': 0.14, 'guaranteed': 0.10
    }

    sensational_score = sum(weight for phrase, weight in sensational.items() if phrase in text_lower)

    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    legit_acronyms = {'HTTP', 'HTTPS', 'NASA', 'FBI', 'CIA', 'NATO', 'COVID', 'USA'}
    excessive_caps = [w for w in caps_words if w not in legit_acronyms]
    caps_score = min(0.20, len(excessive_caps) * 0.05)

    punct_score = min(0.15, (text.count('!!!') + text.count('???')) * 0.05)

    has_citations = bool(re.search(r'(according to|study|research|university)', text_lower))
    citation_factor = -0.15 if has_citations else 0.08

    fake_score = 0.20 + sensational_score + caps_score + punct_score + citation_factor
    fake_score = max(0.05, min(0.95, fake_score))

    classification = (
        "Appears Reliable" if fake_score < 0.30 else
        "Questionable" if fake_score < 0.50 else
        "Likely Misleading" if fake_score < 0.70 else
        "Likely Fake News"
    )

    findings = []
    if sensational_score > 0.1:
        findings.append("Contains sensationalist language")
    if excessive_caps:
        findings.append(f"Excessive caps: {len(excessive_caps)} instances")
    if has_citations:
        findings.append("Includes citations - positive sign")
    if not findings:
        findings.append("Balanced writing style")

    result = {
        'ai_probability': round(fake_score, 3),
        'score': round(fake_score, 3),
        'classification': classification,
        'confidence': 0.80,
        'details': f"Detected {len(findings)} indicators. Sensationalism: {sensational_score:.2f}",
        'specific_findings': findings,
        'recommendations': [
            'Cross-check with fact-checking websites',
            'Verify through multiple sources',
            'Check author credentials',
            'Look for original sources'
        ]
    }

    print(f"📰 News analysis: {classification} ({fake_score:.2%})")
    return result


# ============================================
# URL ANALYSIS
# ============================================

def fetch_and_analyze_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])

        if not text or len(text) < 200:
            return {'error': 'Could not extract article text'}

        result = analyze_news_text(text)
        result['url'] = url
        return result

    except Exception as e:
        return {'error': f'Failed to fetch URL: {str(e)}'}


# ============================================
# GEMINI HELPERS
# ============================================

def call_gemini_deepfake(filepath, content_type="video"):
    if not gemini_model:
        return 0.55
    try:
        file_obj = genai.upload_file(path=filepath)
        prompt = f"""You are a deepfake detection expert.
Analyze this {content_type} and respond ONLY JSON:
{{
  "ai_probability": float  // between 0 and 1
}}"""
        response = gemini_model.generate_content([file_obj, prompt])
        result = parse_ai_response(response.text)
        return float(result.get('ai_probability', 0.55))
    except Exception as e:
        print(f"⚠️ Gemini deepfake error: {e}")
        return 0.55


def call_gemini_audio(filepath):
    if not gemini_model:
        return 0.5
    try:
        file_obj = genai.upload_file(path=filepath)
        prompt = """Analyze this audio for AI voice synthesis.
Respond ONLY JSON: {"ai_probability": float}"""
        response = gemini_model.generate_content([file_obj, prompt])
        result = parse_ai_response(response.text)
        return float(result.get('ai_probability', 0.5))
    except Exception as e:
        print(f"⚠️ Gemini audio error: {e}")
        return 0.5


# ============================================
# HELPER FUNCTIONS
# ============================================

def parse_ai_response(response_text):
    try:
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"⚠️ Failed to parse AI JSON: {e}")
        return {'ai_probability': 0.5}


def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return "0" * 32


def classify_probability(prob, mode='generic'):
    if prob < 0.30:
        return "Real Content" if mode != 'audio' else "Real Voice"
    elif prob < 0.60:
        return "Possibly Manipulated"
    elif prob < 0.85:
        return "Likely Deepfake" if mode != 'audio' else "Likely AI-Generated"
    else:
        return "Deepfake Detected" if mode != 'audio' else "AI-Generated Voice"


def generate_recommendations(ai_prob):
    if ai_prob < 0.30:
        return ["Content appears authentic", "Standard verification recommended"]
    elif ai_prob < 0.60:
        return ["Exercise caution", "Verify through multiple sources", "Use additional tools"]
    else:
        return ["High AI probability", "Do not share without verification", "Consult fact-checkers"]


def get_fallback_result(content_type):
    return {
        'ai_probability': 0.50,
        'classification': 'Analysis Inconclusive',
        'confidence': 0.40,
        'details': f'Unable to analyze {content_type}',
        'specific_findings': ['Limited data'],
        'recommendations': ['Try re-uploading']
    }


def convert_to_serializable(obj):
    """Recursively convert numpy and non-serializable types to JSON serializable"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal error'}), 500


# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🔥 HYBRID AI / DEEPFAKE DETECTION - DESTINY CODER'S 2025")
    print("=" * 60)
    print(f"🤖 Gemini AI: {'✅ Enabled' if gemini_model else '⚠️ Disabled (local ML only)'}")
    print(f"🌐 Server: http://127.0.0.1:5000")
    print(f"📧 Subscribers file: {EMAIL_FILE} ({get_subscriber_count()} users)")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
