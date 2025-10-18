#!/usr/bin/env python3
"""
Emotion prediction module.
Provides functions to load models and make predictions on images.
"""

import numpy as np
import keras
from PIL import Image
from pathlib import Path

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
    """
    Predict emotion from a single image.
    
    Args:
        model: Trained emotion recognition model
        image_path (str): Path to image file
        return_all_probs (bool): Whether to return probabilities for all emotions
        
    Returns:
        dict: Prediction result with 'predicted_emotion', 'confidence', and optionally 'all_probabilities'
    """
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


def predict_batch(model, directory_path, return_all_probs=False):
    """
    Make predictions on all images in a directory.
    
    Args:
        model: Trained emotion recognition model
        directory_path (str): Path to directory containing images
        return_all_probs (bool): Whether to return probabilities for all emotions
        
    Returns:
        list: List of prediction results
    """
    results = []
    dir_path = Path(directory_path)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = [f for f in dir_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    for i, image_file in enumerate(image_files, 1):
        try:
            result = predict_emotion(model, str(image_file), return_all_probs=return_all_probs)
            if result is not None:
                result['image_path'] = str(image_file)
                results.append(result)
                print(f"[{i}/{len(image_files)}] {image_file.name}: {result['predicted_emotion']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return results
