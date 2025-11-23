import os
import json
import hashlib
import re
import tempfile
from datetime import datetime
import traceback

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import requests
from bs4 import BeautifulSoup

from PIL import Image
import cv2
import numpy as np
import mediapipe as mp

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("⚠️ google-generativeai not installed. Gemini features disabled.")

# ============================================
# MEDIAPIPE INITIALIZATION
# ============================================

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe models with error handling
try:
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe initialized successfully")
except Exception as e:
    print(f"⚠️ MediaPipe initialization warning: {e}")
    face_detection = None
    face_mesh = None

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

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

def init_gemini_model():
    if not GEMINI_API_KEY or not genai:
        print("⚠️ Gemini API key not found or package missing.")
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        preferred_models = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision"
        ]
        
        for model_name in preferred_models:
            try:
                print(f"🔍 Trying Gemini Model: {model_name}")
                model = genai.GenerativeModel(model_name)
                # Test the model
                model.generate_content("test")
                print(f"✅ Using {model_name}")
                return model
            except Exception as e:
                print(f"⚠️ {model_name} not available: {str(e)}")
        
        print("🚨 No valid Gemini model found. Using local ML only.")
        return None
    except Exception as e:
        print(f"❌ Gemini initialization failed: {e}")
        return None

gemini_model = init_gemini_model()

# ============================================
# FILE TYPE CONFIGURATION
# ============================================

ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm'},
    'image': {'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'},
    'audio': {'mp3', 'wav', 'm4a', 'ogg', 'flac'}
}

def allowed_file(filename, file_type):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS.get(file_type, set())

# ============================================
# MEDIAPIPE HELPER FUNCTIONS
# ============================================

def extract_face_landmarks(image_rgb):
    """Extract face landmarks using MediaPipe Face Mesh"""
    if face_mesh is None:
        return None
    
    try:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image_rgb.shape[:2]
            points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
            return points
        return None
    except Exception as e:
        print(f"⚠️ Landmark extraction failed: {e}")
        return None

def calculate_face_similarity(landmarks1, landmarks2):
    """Calculate similarity between two sets of face landmarks"""
    if landmarks1 is None or landmarks2 is None:
        return 0.0
    
    try:
        key_indices = [1, 33, 61, 199, 263, 291]
        if len(landmarks1) < max(key_indices) or len(landmarks2) < max(key_indices):
            return 0.5
        
        points1 = landmarks1[key_indices]
        points2 = landmarks2[key_indices]
        
        dist1 = np.linalg.norm(points1.max(axis=0) - points1.min(axis=0))
        dist2 = np.linalg.norm(points2.max(axis=0) - points2.min(axis=0))
        
        if dist1 == 0 or dist2 == 0:
            return 0.0
        
        points1_norm = (points1 - points1.mean(axis=0)) / dist1
        points2_norm = (points2 - points2.mean(axis=0)) / dist2
        
        distance = np.mean(np.linalg.norm(points1_norm - points2_norm, axis=1))
        similarity = max(0, 1 - distance)
        
        return similarity
    except Exception as e:
        print(f"⚠️ Similarity calculation failed: {e}")
        return 0.5

def detect_face_quality(image_rgb):
    """Analyze face quality metrics using MediaPipe"""
    if face_detection is None:
        return {'face_detected': False, 'confidence': 0.0, 'face_size': 0}
    
    try:
        results = face_detection.process(image_rgb)
        
        if not results.detections:
            return {'face_detected': False, 'confidence': 0.0, 'face_size': 0}
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = image_rgb.shape[:2]
        face_area = bbox.width * bbox.height * w * h
        
        return {
            'face_detected': True,
            'confidence': detection.score[0],
            'face_size': face_area,
            'bbox': bbox
        }
    except Exception as e:
        print(f"⚠️ Face quality detection failed: {e}")
        return {'face_detected': False, 'confidence': 0.0, 'face_size': 0}

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
        # ===== VALIDATION =====
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        file_type = request.form.get('type', 'image')  # Default to image if not specified

        if not file or file.filename == '':
            print("❌ Empty file")
            return jsonify({'error': 'No file selected'}), 400

        print(f"📤 Received request - Type: {file_type}, File: {file.filename}")

        # Validate file type
        if not allowed_file(file.filename, file_type):
            print(f"❌ Invalid file type: {file.filename} for {file_type}")
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS.get(file_type, []))}'}), 400

        # ===== CREATE TEMPORARY FILE =====
        suffix = os.path.splitext(secure_filename(file.filename))[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name

        print(f"💾 Temporary file created: {temp_filepath}")

        # Verify file was saved
        if not os.path.exists(temp_filepath):
            raise RuntimeError("Failed to save temporary file")

        file_size = os.path.getsize(temp_filepath)
        print(f"📊 File size: {file_size / 1024:.2f} KB")

        # ===== ROUTE TO APPROPRIATE ANALYZER =====
        if file_type == 'video' or file_type == 'deepfake':
            result = analyze_video(temp_filepath)
        elif file_type == 'image':
            result = analyze_image(temp_filepath)
        elif file_type == 'audio':
            result = analyze_audio(temp_filepath)
        else:
            return jsonify({'error': f'Unknown file type: {file_type}'}), 400

        print(f"✅ Analysis complete: {result.get('classification', 'N/A')}")
        
        # Ensure result is serializable
        serializable_result = convert_to_serializable(result)
        
        return jsonify(serializable_result), 200

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"❌ Error in analyze_file: {error_msg}")
        print(f"📋 Traceback:\n{error_trace}")
        
        return jsonify({
            'error': f'Analysis failed: {error_msg}',
            'details': 'Check server logs for more information'
        }), 500

    finally:
        # Cleanup
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.unlink(temp_filepath)
                print(f"🗑️ Deleted temporary file: {temp_filepath}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temp file: {e}")

# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def analyze_video(filepath):
    """Video analysis with MediaPipe"""
    try:
        print("🎥 Starting video analysis...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        cap = cv2.VideoCapture(filepath)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise RuntimeError("Video has no frames")

        print(f"📊 Video info: {total_frames} frames")

        frame_step = max(1, total_frames // 30)
        blur_scores = []
        face_confidences = []
        face_consistency_scores = []
        prev_landmarks = None
        frames_used = 0
        faces_detected = 0

        while frames_used < 30:  # Limit to 30 frames
            frame_idx = frames_used * frame_step
            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                break

            frames_used += 1

            # Blur detection
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(blur_score)
            except Exception as e:
                print(f"⚠️ Blur detection failed: {e}")

            # Face analysis
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                face_info = detect_face_quality(rgb)
                if face_info['face_detected']:
                    faces_detected += 1
                    face_confidences.append(face_info['confidence'])
                    
                    landmarks = extract_face_landmarks(rgb)
                    if landmarks is not None and prev_landmarks is not None:
                        similarity = calculate_face_similarity(landmarks, prev_landmarks)
                        face_consistency_scores.append(similarity)
                    prev_landmarks = landmarks
            except Exception as e:
                print(f"⚠️ Face analysis failed on frame {frames_used}: {e}")

        cap.release()

        # Calculate metrics
        avg_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
        avg_face_conf = float(np.mean(face_confidences)) if face_confidences else 0.0
        avg_consistency = float(np.mean(face_consistency_scores)) if face_consistency_scores else 0.5
        face_detection_rate = faces_detected / max(frames_used, 1)

        # Scoring
        local_prob = 0.15
        
        if avg_blur < 100:
            local_prob += 0.25
        if avg_face_conf < 0.6 and faces_detected > 0:
            local_prob += 0.20
        if avg_consistency < 0.65 and len(face_consistency_scores) > 0:
            local_prob += 0.25
        if face_detection_rate < 0.5 and frames_used > 5:
            local_prob += 0.15

        local_prob = max(0.05, min(0.95, local_prob))

        # Optional Gemini
        api_prob = call_gemini_deepfake(filepath, content_type="video")
        final_prob = round(0.65 * local_prob + 0.35 * api_prob, 3)

        classification = classify_probability(final_prob)

        findings = [
            f"Analyzed {frames_used} frames",
            f"Faces detected: {faces_detected}/{frames_used}",
            f"Avg blur score: {avg_blur:.1f}",
            f"Face confidence: {avg_face_conf:.2f}",
            f"Consistency score: {avg_consistency:.2f}"
        ]

        return {
            'ai_probability': final_prob,
            'classification': classification,
            'confidence': round(0.65 + abs(final_prob - 0.5) * 0.7, 3),
            'details': f"Analyzed {frames_used} video frames",
            'specific_findings': findings,
            'recommendations': generate_recommendations(final_prob),
            'metadata': {
                'frames_analyzed': frames_used,
                'faces_detected': faces_detected,
                'avg_blur': round(avg_blur, 2)
            }
        }

    except Exception as e:
        print(f"❌ Error in analyze_video: {e}")
        traceback.print_exc()
        return get_fallback_result("video")

def analyze_image(filepath):
    """Image analysis with MediaPipe"""
    try:
        print("🖼️ Starting image analysis...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        # Read image
        img_cv = cv2.imread(filepath)
        
        if img_cv is None:
            raise RuntimeError("Failed to read image file")

        print(f"📊 Image shape: {img_cv.shape}")

        # Basic stats
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        h, w = gray.shape[:2]
        resolution = w * h

        # Face analysis
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        face_info = detect_face_quality(rgb)
        landmarks = extract_face_landmarks(rgb)

        # Scoring
        local_prob = 0.20
        
        if blur_score < 100:
            local_prob += 0.25
        if brightness < 50 or brightness > 200:
            local_prob += 0.15
        if resolution in [262144, 589824, 1048576]:
            local_prob += 0.20
        
        if face_info['face_detected']:
            if face_info['confidence'] < 0.7:
                local_prob += 0.15
        else:
            local_prob += 0.10

        local_prob = max(0.05, min(0.95, local_prob))

        # Optional Gemini
        api_prob = call_gemini_deepfake(filepath, content_type="image")
        final_prob = round(0.65 * local_prob + 0.35 * api_prob, 3)

        classification = classify_probability(final_prob)

        findings = [
            f"Resolution: {w}x{h}",
            f"Blur score: {blur_score:.1f}",
            f"Brightness: {brightness:.1f}",
            f"Face detected: {face_info['face_detected']}"
        ]

        if face_info['face_detected']:
            findings.append(f"Face confidence: {face_info['confidence']:.2f}")

        return {
            'ai_probability': final_prob,
            'classification': classification,
            'confidence': round(0.65 + abs(final_prob - 0.5) * 0.7, 3),
            'details': f"Image analysis complete",
            'specific_findings': findings,
            'recommendations': generate_recommendations(final_prob),
            'metadata': {
                'resolution': f"{w}x{h}",
                'face_detected': face_info['face_detected']
            }
        }

    except Exception as e:
        print(f"❌ Error in analyze_image: {e}")
        traceback.print_exc()
        return get_fallback_result("image")

def analyze_audio(filepath):
    """Basic audio analysis"""
    try:
        print("🎵 Starting audio analysis...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
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
            'details': f"Audio file analyzed",
            'specific_findings': [f"File size: {file_size / 1024:.1f} KB"],
            'recommendations': generate_recommendations(final_prob)
        }
        
    except Exception as e:
        print(f"❌ Error in analyze_audio: {e}")
        traceback.print_exc()
        return get_fallback_result("audio")

# ============================================
# TEXT ANALYSIS ROUTES
# ============================================

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text content"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
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

        return jsonify(convert_to_serializable(result)), 200

    except Exception as e:
        print(f"❌ Error in analyze_text: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze news article from URL"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400

        print(f"🌐 Fetching URL: {url}")
        result = fetch_and_analyze_url(url)

        return jsonify(convert_to_serializable(result)), 200

    except Exception as e:
        print(f"❌ Error in analyze_url: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe_email():
    """Subscribe email for updates"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        email = data.get('email', '').strip().lower()

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return jsonify({'error': 'Invalid email format'}), 400

        current_subscribers = get_subscriber_emails()
        if email in current_subscribers:
            return jsonify({
                'success': True,
                'message': 'You\'re already on our list! 🎉',
                'already_subscribed': True,
                'subscriber_count': len(current_subscribers)
            }), 200

        if save_subscriber(email):
            new_count = get_subscriber_count()
            return jsonify({
                'success': True,
                'message': f'Thanks for subscribing! 🚀',
                'subscriber_count': new_count
            }), 200
        else:
            return jsonify({'error': 'Failed to save subscription'}), 500

    except Exception as e:
        print(f"❌ Error in subscribe_email: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================
# TEXT ANALYSIS FUNCTIONS
# ============================================

def analyze_text_content(text):
    """Analyze text for AI generation"""
    indicators = {
        'delve into': 0.15, 'multifaceted': 0.10, 'paradigm': 0.12,
        'leverage': 0.08, 'synergy': 0.10, 'holistic': 0.11,
        'furthermore': 0.04, 'moreover': 0.04
    }

    text_lower = text.lower()
    indicator_score = sum(weight for phrase, weight in indicators.items() if phrase in text_lower)
    found = [phrase for phrase in indicators if phrase in text_lower]

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
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
        'details': f"Text analysis: {len(text)} chars",
        'specific_findings': findings,
        'recommendations': generate_recommendations(ai_prob)
    }

def analyze_news_text(text):
    """Analyze news for fake news indicators"""
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

    return {
        'ai_probability': round(fake_score, 3),
        'score': round(fake_score, 3),
        'classification': classification,
        'confidence': 0.80,
        'details': f"News analysis complete",
        'specific_findings': findings,
        'recommendations': [
            'Cross-check with fact-checking websites',
            'Verify through multiple sources',
            'Check author credentials'
        ]
    }

def fetch_and_analyze_url(url):
    """Fetch and analyze URL content"""
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
    """Call Gemini API for deepfake detection"""
    if not gemini_model:
        return 0.5
    
    try:
        file_obj = genai.upload_file(path=filepath)
        prompt = f"""Analyze this {content_type} for deepfake/AI generation.
Respond with ONLY a JSON object: {{"ai_probability": float}}
The probability should be between 0 and 1."""
        
        response = gemini_model.generate_content([file_obj, prompt])
        result = parse_ai_response(response.text)
        return float(result.get('ai_probability', 0.5))
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return 0.5

def call_gemini_audio(filepath):
    """Call Gemini API for audio analysis"""
    if not gemini_model:
        return 0.5
    
    try:
        file_obj = genai.upload_file(path=filepath)
        prompt = """Analyze this audio for AI voice synthesis.
Respond with ONLY a JSON object: {"ai_probability": float}"""
        
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
    """Parse AI JSON response"""
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

def classify_probability(prob, mode='generic'):
    """Classify probability into categories"""
    if prob < 0.30:
        return "Real Content" if mode != 'audio' else "Real Voice"
    elif prob < 0.60:
        return "Possibly Manipulated"
    elif prob < 0.85:
        return "Likely Deepfake" if mode != 'audio' else "Likely AI-Generated"
    else:
        return "Deepfake Detected" if mode != 'audio' else "AI-Generated Voice"

def generate_recommendations(ai_prob):
    """Generate recommendations based on probability"""
    if ai_prob < 0.30:
        return ["Content appears authentic", "Standard verification recommended"]
    elif ai_prob < 0.60:
        return ["Exercise caution", "Verify through multiple sources"]
    else:
        return ["High AI probability", "Do not share without verification"]

def get_fallback_result(content_type):
    """Return fallback result on error"""
    return {
        'ai_probability': 0.50,
        'classification': 'Analysis Inconclusive',
        'confidence': 0.40,
        'details': f'Unable to fully analyze {content_type}',
        'specific_findings': ['Analysis completed with limited data'],
        'recommendations': ['Try uploading a different file or format']
    }

def convert_to_serializable(obj):
    """Convert numpy/complex types to JSON serializable"""
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

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(502)
def bad_gateway(error):
    return jsonify({'error': 'Bad gateway'}), 502

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🔥 MEDIAPIPE + OPENCV DEEPFAKE DETECTION - DESTINY CODER'S 2025")
    print("=" * 70)
    print(f"🤖 Gemini AI: {'✅ Enabled' if gemini_model else '⚠️ Disabled (local ML only)'}")
    print(f"👤 MediaPipe: {'✅ Enabled' if face_detection else '⚠️ Disabled'}")
    print(f"🌐 Server: http://127.0.0.1:5000")
    print(f"📧 Subscribers: {get_subscriber_count()} users")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')