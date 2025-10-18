#!/usr/bin/env python3
"""
Emotion prediction with face detection and cropping.
Detects faces in photos and predicts emotions on the detected faces.
"""

import sys
sys.path.insert(0, '/Users/romanminakov/Personal/Studies/Sem6/RI/RI-Project/src')

import cv2
import numpy as np
from pathlib import Path
import keras
from PIL import Image

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMAGE_SIZE = 128

def load_model(model_path):
    """Load trained emotion recognition model."""
    return keras.models.load_model(model_path)

def load_and_preprocess_image(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Load and preprocess image for prediction."""
    try:
        img = Image.open(image_path).convert('RGB')  # Convert to RGB
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)  # Add batch dim
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_emotion(model, image_path, return_all_probs=False):
    """Predict emotion from image."""
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return None
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_emotion = EMOTION_LABELS[predicted_idx]
    confidence = float(predictions[predicted_idx])
    
    result = {
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'predicted_class': predicted_idx
    }
    
    if return_all_probs:
        result['all_probabilities'] = {
            emotion: float(prob) for emotion, prob in zip(EMOTION_LABELS, predictions)
        }
    
    return result

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_and_predict_faces(model, image_path):
    """
    Detect faces in image and predict emotion for each face.
    
    Args:
        model: Trained emotion recognition model
        image_path (str): Path to image file
        
    Returns:
        list: List of predictions for each detected face
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"❌ No faces detected in {image_path}")
        return []
    
    print(f"\n✓ Detected {len(faces)} face(s) in image")
    print("="*70)
    
    results = []
    
    for i, (x, y, w, h) in enumerate(faces, 1):
        face_roi = image[y:y+h, x:x+w]
        
        temp_face_path = f"/tmp/face_{i}.jpg"
        cv2.imwrite(temp_face_path, face_roi)
        
        result = predict_emotion(model, temp_face_path, return_all_probs=True)
        results.append(result)
        
        print(f"\nFace {i}:")
        print(f"  Location: ({x}, {y}), Size: {w}x{h}")
        print(f"  Emotion: {result['predicted_emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  All probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"    {emotion:<10} [{bar}] {prob:.2%}")
    
    print("\n" + "="*70)
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect faces and predict emotions')
    parser.add_argument('model_path', help='Path to trained emotion model')
    parser.add_argument('--image', required=True, help='Path to image file')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    print(f"\nProcessing image: {args.image}")
    results = detect_and_predict_faces(model, args.image)
    
    if results:
        print(f"\n✓ Successfully predicted emotions for {len(results)} face(s)")
    else:
        print("\n❌ Failed to detect or predict faces")
