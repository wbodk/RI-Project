#!/usr/bin/env python3
"""
Unified command runner for all CNN training, evaluation, and prediction tasks.
Provides a menu-based interface for common operations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path so imports work
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

def print_menu():
    """Print the main menu."""
    print("\n" + "="*70)
    print("Emotion Recognition CNN - Command Menu")
    print("="*70)
    print("\n[1] Train the CNN model")
    print("[2] Evaluate trained model")
    print("[3] Make predictions on image(s)")
    print("[4] Analyze dataset")
    print("[5] Run integration tests")
    print("[6] Exit")
    print("\n" + "-"*70)

def train_model():
    """Run training."""
    from train_cnn import train_cnn
    print("\nStarting model training...")
    print("(This may take a while depending on your hardware)\n")
    train_cnn()

def evaluate_model():
    """Evaluate trained model."""
    from pathlib import Path
    
    # Get models directory relative to this script
    script_dir = Path(__file__).parent.parent.parent  # Goes from src/cnn/ to root
    model_dir = script_dir / "models"
    if not model_dir.exists():
        print("\n✗ No models directory found. Train a model first!")
        return
    
    model_files = list(model_dir.glob("*.keras"))
    if not model_files:
        print("\n✗ No .keras model files found. Train a model first!")
        return
    
    # Use final model if available, otherwise latest
    if (model_dir / "final_model.keras").exists():
        model_path = model_dir / "final_model.keras"
    else:
        model_path = sorted(model_files)[-1]
    
    print(f"\nEvaluating model: {model_path}")
    
    from dataset_utils import get_combined_datasets
    from evaluate import evaluate_model_comprehensive, plot_per_class_metrics
    
    print("Loading dataset...")
    _, _, test_ds = get_combined_datasets(batch_size=32)
    
    print("Running evaluation...")
    model, predictions, true_labels = evaluate_model_comprehensive(
        str(model_path),
        test_ds
    )
    
    print("\nGenerating per-class metrics plot...")
    plot_per_class_metrics(true_labels, predictions)
    
    print("\n✓ Evaluation complete!")

def predict_images():
    """Make predictions on images."""
    from pathlib import Path
    
    # Get models directory relative to this script
    script_dir = Path(__file__).parent.parent.parent  # Goes from src/cnn/ to root
    model_dir = script_dir / "models"
    if not model_dir.exists() or not list(model_dir.glob("*.keras")):
        print("\n✗ No trained models found. Train a model first!")
        return
    
    # Use final model if available
    if (model_dir / "final_model.keras").exists():
        model_path = model_dir / "final_model.keras"
    else:
        model_files = list(model_dir.glob("*.keras"))
        model_path = sorted(model_files)[-1]
    
    print(f"\nUsing model: {model_path}")
    
    pred_type = input("\nPredict on (1) single image or (2) directory? [1/2]: ").strip()
    
    if pred_type == "1":
        image_path = input("Enter image path: ").strip()
        from cnn.predict import load_model, predict_emotion
        
        model = load_model(str(model_path))
        result = predict_emotion(model, image_path, return_all_probs=True)
        
        print(f"\n✓ Predicted emotion: {result['predicted_emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("\n  All probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"    {emotion:<10} [{bar}] {prob:.2%}")
    
    elif pred_type == "2":
        dir_path = input("Enter directory path: ").strip()
        from cnn.predict import load_model, predict_batch
        
        model = load_model(str(model_path))
        results = predict_batch(model, dir_path, return_all_probs=False)
        
        print(f"\n✓ Processed {len(results)} images")
        
        # Summary statistics
        from collections import Counter
        emotion_counts = Counter(r['predicted_emotion'] for r in results)
        print("\nSummary:")
        for emotion, count in emotion_counts.most_common():
            print(f"  {emotion}: {count}")
    
    else:
        print("Invalid option!")

def analyze_dataset():
    """Analyze dataset."""
    print("\nAnalyzing dataset...")
    print("(This downloads and analyzes both datasets)\n")
    
    from data_analysis import compare_datasets, visualize_sample_images
    from dataset_utils import get_combined_datasets
    
    compare_datasets()
    
    print("\nVisualizing sample images...")
    train_ds, _, test_ds = get_combined_datasets(batch_size=32)
    visualize_sample_images(test_ds, num_images=12, dataset_name="Combined Dataset")
    
    print("\n✓ Dataset analysis complete!")

def run_tests():
    """Run integration tests."""
    print("\nRunning integration tests...\n")
    from test_integration import main
    sys.exit(main())

def main():
    """Main menu loop."""
    parser = argparse.ArgumentParser(description='Emotion Recognition CNN Command Runner')
    parser.add_argument('--train', action='store_true', help='Run training directly')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation directly')
    parser.add_argument('--predict', type=str, metavar='IMAGE', help='Make prediction on image')
    parser.add_argument('--predict-dir', type=str, metavar='DIR', help='Make predictions on directory')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset')
    parser.add_argument('--test', action='store_true', help='Run integration tests')
    
    args = parser.parse_args()
    
    # Handle direct command arguments
    if args.train:
        train_model()
        return
    
    if args.evaluate:
        evaluate_model()
        return
    
    if args.predict:
        from cnn.predict import load_model, predict_emotion
        script_dir = Path(__file__).parent.parent.parent  # Goes from src/cnn/ to root
        model_path = script_dir / "models" / "final_model.keras"
        if not model_path.exists():
            print("✗ Model not found at:", model_path)
            return
        model = load_model(str(model_path))
        result = predict_emotion(model, args.predict, return_all_probs=True)
        print(f"\n✓ Emotion: {result['predicted_emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        return
    
    if args.predict_dir:
        from cnn.predict import load_model, predict_batch
        script_dir = Path(__file__).parent.parent.parent  # Goes from src/cnn/ to root
        model_path = script_dir / "models" / "final_model.keras"
        if not model_path.exists():
            print("✗ Model not found at:", model_path)
            return
        model = load_model(str(model_path))
        predict_batch(model, args.predict_dir)
        return
    
    if args.analyze:
        analyze_dataset()
        return
    
    if args.test:
        run_tests()
        return
    
    # Interactive menu
    print("\n" + "="*70)
    print("Welcome to Emotion Recognition CNN!")
    print("="*70)
    
    while True:
        print_menu()
        choice = input("Enter your choice [1-6]: ").strip()
        
        try:
            if choice == "1":
                train_model()
            elif choice == "2":
                evaluate_model()
            elif choice == "3":
                predict_images()
            elif choice == "4":
                analyze_dataset()
            elif choice == "5":
                run_tests()
            elif choice == "6":
                print("\nGoodbye!")
                break
            else:
                print("\n✗ Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Please check the error and try again.")

if __name__ == "__main__":
    main()
