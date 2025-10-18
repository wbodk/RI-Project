"""
Integration test script to verify all components work correctly.
Run this before starting full training to catch any issues early.
"""

import sys
import os
from pathlib import Path

# Change to src directory
os.chdir(Path(__file__).parent / 'src')

def test_imports():
    """Test that all required imports work."""
    print("\n[Test 1/5] Testing imports...")
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__}")
        
        from tensorflow import keras
        print(f"  ✓ Keras {keras.__version__}")
        
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
        
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
        
        from sklearn.metrics import confusion_matrix
        print(f"  ✓ Scikit-learn")
        
        import seaborn as sns
        print(f"  ✓ Seaborn {sns.__version__}")
        
        from PIL import Image
        print(f"  ✓ Pillow")
        
        import kaggle
        print(f"  ✓ Kaggle API")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_dataset_utils():
    """Test that dataset utilities can be imported and initialized."""
    print("\n[Test 2/5] Testing dataset utilities...")
    try:
        from dataset_utils import (
            get_raf_db_dataset,
            get_fer_2013_dataset,
            get_combined_datasets,
            RAF_DB_TO_UNIFIED,
            FER_2013_TO_UNIFIED
        )
        print("  ✓ Dataset utilities imported")
        
        # Check mappings
        assert len(RAF_DB_TO_UNIFIED) == 7, "RAF-DB mapping should have 7 classes"
        assert len(FER_2013_TO_UNIFIED) == 7, "FER-2013 mapping should have 7 classes"
        print("  ✓ Emotion mappings correct (7 classes)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_model_building():
    """Test that the CNN model can be built."""
    print("\n[Test 3/5] Testing model building...")
    try:
        from train_cnn import build_cnn_model, compile_model
        
        model = build_cnn_model(input_shape=(128, 128, 3), num_classes=7)
        print(f"  ✓ Model built with {len(model.layers)} layers")
        
        compile_model(model)
        print("  ✓ Model compiled successfully")
        
        # Check model structure
        params = model.count_params()
        print(f"  ✓ Total parameters: {params:,}")
        
        if params > 1_000_000:
            print(f"  ✓ Model has sufficient complexity")
        else:
            print(f"  ⚠ Warning: Model might be too small")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_predict_module():
    """Test that prediction utilities can be imported."""
    print("\n[Test 4/5] Testing prediction module...")
    try:
        from cnn.predict import (
            load_and_preprocess_image,
            predict_emotion,
            load_model,
            EMOTION_LABELS,
            IMAGE_SIZE
        )
        print("  ✓ Prediction module imported")
        
        assert len(EMOTION_LABELS) == 7, "Should have 7 emotion labels"
        print(f"  ✓ Emotion labels: {', '.join(EMOTION_LABELS)}")
        
        assert IMAGE_SIZE == (128, 128), "Image size should be (128, 128)"
        print(f"  ✓ Image size configured correctly: {IMAGE_SIZE}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_evaluation_module():
    """Test that evaluation utilities can be imported."""
    print("\n[Test 5/5] Testing evaluation module...")
    try:
        from evaluate import (
            evaluate_model_comprehensive,
            plot_per_class_metrics,
            EMOTION_LABELS
        )
        print("  ✓ Evaluation module imported")
        
        assert len(EMOTION_LABELS) == 7, "Should have 7 emotion labels"
        print(f"  ✓ Emotion labels match: {len(EMOTION_LABELS)} classes")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("="*70)
    print("Integration Test Suite - Emotion Recognition CNN")
    print("="*70)
    
    tests = [
        test_imports,
        test_dataset_utils,
        test_model_building,
        test_predict_module,
        test_evaluation_module
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to start training.")
        print("\nNext step: Run 'python train_cnn.py' to start training")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed!")
        print("\nPlease fix the issues above and try again.")
        print("\nCommon issues:")
        print("  1. Missing dependencies: pip install -r requirements.txt")
        print("  2. Wrong working directory: Run from project root or src/")
        print("  3. TensorFlow issues: Try 'pip install --upgrade tensorflow'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
