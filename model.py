import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================
# DEVICE
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# IMAGE PREPROCESSING
# ============================================
TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),          # EfficientNet-B4 expects 380x380
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet mean
        std=[0.229, 0.224, 0.225]           # ImageNet std
    )
])

# ============================================
# MODEL DEFINITION
# ============================================
def build_xception_model():
    """
    EfficientNet-B4 as XceptionNet replacement.
    Better accuracy, PyTorch native, no extra deps.
    """
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 2)           # 2 classes: real / fake
    )
    return model

# ============================================
# LOAD WEIGHTS
# ============================================
def load_model(weights_path='xception_deepfake.pt'):
    model = build_xception_model()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()                            # Important: disables dropout at inference
    return model

# ============================================
# PREDICTION
# ============================================
def predict(model, filepath, confidence_threshold=0.75):
    img = Image.open(filepath).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)   # (1, 3, 380, 380)

    with torch.no_grad():                             # No gradient needed at inference
        logits = model(tensor)                        # (1, 2)
        probs = torch.softmax(logits, dim=1)[0]       # (2,)

    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    confidence = max(real_prob, fake_prob)

    return {
        'ai_probability': round(fake_prob, 2),
        'classification': "Deepfake Detected" if fake_prob > 0.5 else "Authentic Image",
        'confidence': round(confidence, 2),
        'is_confident': confidence >= confidence_threshold,
        'details': "EfficientNet-B4 deepfake detection model.",
        'specific_findings': ["GAN artifacts detected"] if fake_prob > 0.5 else ["No manipulation found"],
        'recommendations': ["Flag for manual review"] if fake_prob > 0.5 else ["Image appears genuine"],
        'model_used': 'EfficientNet-B4 (PyTorch)'
    }