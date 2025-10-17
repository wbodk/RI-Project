import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from pathlib import Path

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMAGE_SIZE = (128, 128)

def load_and_preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image file
        image_size (tuple): Target image size
        
    Returns:
        np.ndarray: Preprocessed image ready for prediction
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = img.resize(image_size)
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array

def predict_emotion(model, image_path, image_size=IMAGE_SIZE, return_all_probs=False):
    """
    Predict emotion for a single image.
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the image
        image_size (tuple): Target image size
        return_all_probs (bool): Whether to return all class probabilities
        
    Returns:
        dict: Dictionary containing prediction results
    """
    # Preprocess image
    img_array = load_and_preprocess_image(image_path, image_size)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Get predicted class and confidence
    pred_idx = np.argmax(predictions)
    confidence = predictions[pred_idx]
    
    result = {
        'image_path': str(image_path),
        'predicted_emotion': EMOTION_LABELS[pred_idx],
        'confidence': float(confidence),
        'predicted_class': int(pred_idx)
    }
    
    if return_all_probs:
        result['all_probabilities'] = {
            label: float(prob) for label, prob in zip(EMOTION_LABELS, predictions)
        }
    
    return result

def predict_batch(model, image_dir, image_extensions=['.jpg', '.jpeg', '.png'], return_all_probs=False):
    """
    Predict emotions for all images in a directory.
    
    Args:
        model: Trained Keras model
        image_dir (str): Path to directory containing images
        image_extensions (list): List of valid image extensions
        return_all_probs (bool): Whether to return all class probabilities
        
    Returns:
        list: List of prediction results
    """
    image_dir = Path(image_dir)
    results = []
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process...")
    
    for idx, image_path in enumerate(sorted(image_files), 1):
        try:
            result = predict_emotion(model, image_path, return_all_probs=return_all_probs)
            results.append(result)
            print(f"[{idx}/{len(image_files)}] {image_path.name}: {result['predicted_emotion']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {image_path.name}: Error - {str(e)}")
    
    return results

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        keras.Model: Loaded model
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict emotions using trained CNN model')
    parser.add_argument('model_path', help='Path to the trained model')
    parser.add_argument('--image', help='Path to a single image for prediction')
    parser.add_argument('--dir', help='Path to directory with multiple images')
    parser.add_argument('--all-probs', action='store_true', help='Return all class probabilities')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Make predictions
    if args.image:
        print(f"\nPredicting for image: {args.image}")
        result = predict_emotion(model, args.image, return_all_probs=args.all_probs)
        print(f"\n✓ Prediction: {result['predicted_emotion']}")
        print(f"✓ Confidence: {result['confidence']:.4f}")
        if args.all_probs:
            print("\n✓ All probabilities:")
            for emotion, prob in result['all_probabilities'].items():
                print(f"  {emotion}: {prob:.4f}")
    
    elif args.dir:
        print(f"\nPredicting for all images in: {args.dir}")
        results = predict_batch(model, args.dir, return_all_probs=args.all_probs)
        print(f"\n✓ Processed {len(results)} images successfully!")
    
    else:
        parser.print_help()
