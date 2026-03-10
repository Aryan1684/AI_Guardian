import requests
import base64

# Best HuggingFace model for deepfake detection:
# "dima806/deepfake_vs_real_image_detection" or
# "prithivMLmods/Deepfake-Detector-Model-v2"
HF_MODEL = "dima806/deepfake_vs_real_image_detection"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def predict_huggingface(image_path, hf_token):
    """
    Calls HuggingFace Inference API as fallback.
    Returns same dict format as predict_xception.
    """
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=15)
    
    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace API error: {response.status_code} — {response.text}")
    
    results = response.json()
    # Response is a list like: [{"label": "Fake", "score": 0.97}, ...]
    
    label_map = {}
    for item in results:
        label_map[item['label'].upper()] = item['score']
    
    fake_conf = label_map.get('FAKE', 0.0)
    real_conf = label_map.get('REAL', 1.0 - fake_conf)
    label = 'FAKE' if fake_conf > real_conf else 'REAL'
    
    return {
        'label': label,
        'confidence': max(fake_conf, real_conf),
        'fake_prob': fake_conf,
        'real_prob': real_conf,
        'is_confident': True,
        'model_used': 'HuggingFace'
    }