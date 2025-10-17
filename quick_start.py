#!/usr/bin/env python3
"""
Quick start script for training the emotion recognition CNN.
Run this script to start training immediately with default settings.
"""

import os
import sys
from pathlib import Path

# Change to src directory
os.chdir(Path(__file__).parent / 'src')

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Emotion Recognition CNN - Quick Start")
    print("="*80 + "\n")
    
    print("Welcome! This script will train a CNN on the combined RAF-DB and FER-2013 datasets.")
    print("\nWhat will happen:")
    print("  1. Download datasets from Kaggle (if not already present)")
    print("  2. Combine RAF-DB and FER-2013 datasets")
    print("  3. Train a CNN model for emotion recognition")
    print("  4. Save the trained model and results\n")
    
    response = input("Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Exiting...")
        sys.exit(0)
    
    print("\nStarting training...\n")
    
    try:
        from src.train_cnn import train_cnn
        model, history = train_cnn()
        
        print("\n" + "="*80)
        print("✓ Training completed successfully!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Evaluate the model:")
        print("     python src/evaluate.py")
        print("\n  2. Make predictions on new images:")
        print("     python src/predict.py ../models/final_model.keras --image <image_path>")
        print("\n  3. Analyze the dataset:")
        print("     python src/data_analysis.py")
        print("\nFor more details, see TRAINING_GUIDE.md\n")
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        print("\nMake sure:")
        print("  1. You have installed all dependencies: pip install -r requirements.txt")
        print("  2. Your Kaggle API credentials are set up")
        print("  3. You have internet connection to download datasets")
        sys.exit(1)
