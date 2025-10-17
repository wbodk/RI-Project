# RI-Project: Emotion Recognition CNN

A comprehensive Convolutional Neural Network (CNN) implementation for facial emotion recognition trained on the combination of **RAF-DB** and **FER-2013** datasets.

## 🎯 Project Overview

This project implements a complete emotion recognition pipeline:
- **7 emotion classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Combined dataset**: RAF-DB (~20k images) + FER-2013 (~35k images)
- **Deep CNN model**: 4-block architecture with batch normalization and dropout
- **Complete toolchain**: Training, evaluation, visualization, and inference scripts

## 📁 Project Structure

```
├── src/
│   ├── dataset_utils.py          # Dataset loading and preprocessing
│   ├── train_cnn.py              # Main training script
│   ├── evaluate.py               # Comprehensive evaluation
│   ├── visualization.py          # Training visualization
│   ├── predict.py                # Inference on new images
│   ├── data_analysis.py          # Dataset analysis utilities
│   ├── test_integration.py       # Integration tests
│   └── __pycache__/              # Python cache
├── data/                          # Dataset storage (auto-downloaded)
│   ├── fer2013/                  # FER-2013 dataset
│   └── raf-db-dataset/           # RAF-DB dataset
├── models/                        # Trained models (created after training)
├── results/                       # Training results and plots
├── quick_start.py                # Interactive quick start
├── TRAINING_GUIDE.md             # Detailed training documentation
├── IMPLEMENTATION_SUMMARY.md     # Implementation details
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Kaggle API credentials (for dataset download)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd RI-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle credentials
# Create ~/.kaggle/kaggle.json with your API key
# See: https://www.kaggle.com/account
```

### Training

**Option 1: Interactive Quick Start (Recommended)**
```bash
python quick_start.py
```

**Option 2: Direct Training**
```bash
cd src
python train_cnn.py
```

The training script will:
- Download datasets automatically (if needed)
- Combine RAF-DB and FER-2013 datasets
- Train the CNN model
- Save the best model and final model
- Evaluate on test set

### Making Predictions

**Single image prediction:**
```bash
python src/predict.py models/final_model.keras --image path/to/image.jpg
```

**Batch prediction:**
```bash
python src/predict.py models/final_model.keras --dir path/to/images/
```

**With all probabilities:**
```bash
python src/predict.py models/final_model.keras --image photo.jpg --all-probs
```

### Dataset Analysis

```bash
python src/data_analysis.py
```

Generates:
- Class distribution plots
- Image statistics
- Sample visualizations
- Dataset comparisons

## 🧠 Model Architecture

```
Input (128×128×3)
    ↓
Block 1: Conv(32) → BN → Conv(32) → BN → MaxPool → Dropout(0.25)
    ↓
Block 2: Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv(128) → BN → Conv(128) → BN → MaxPool → Dropout(0.25)
    ↓
Block 4: Conv(256) → BN → Conv(256) → BN → MaxPool → Dropout(0.25)
    ↓
Global Average Pooling
    ↓
Dense(512) → BN → Dropout(0.5)
    ↓
Dense(256) → BN → Dropout(0.5)
    ↓
Dense(7) + Softmax (Emotion output)
```

**Parameters**: ~3.5M trainable parameters

## 📊 Training Configuration

Default configuration in `train_cnn.py`:

```python
CONFIG = {
    'image_size': (128, 128),      # Input image size
    'batch_size': 32,              # Training batch size
    'epochs': 50,                  # Maximum training epochs
    'learning_rate': 0.001,        # Initial learning rate
    'num_classes': 7,              # Number of emotion classes
    'validation_split': 0.15       # Validation/training split
}
```

### Training Features

- ✅ **Early Stopping**: Stops if validation loss doesn't improve for 10 epochs
- ✅ **Model Checkpointing**: Saves best model during training
- ✅ **Learning Rate Scheduling**: Reduces LR by 0.5x if loss plateaus
- ✅ **TensorBoard Logging**: Monitor training in real-time
- ✅ **GPU Support**: Automatically uses GPU if available
- ✅ **Mixed Precision**: Optimized for Apple Silicon (TF Metal)

## 📈 Outputs

After training, the following files are created:

```
models/
├── best_model_YYYYMMDD_HHMMSS.keras    # Best checkpoint
├── final_model.keras                    # Final trained model
└── logs_YYYYMMDD_HHMMSS/               # TensorBoard logs

results/
├── training_history.png                 # Loss & accuracy curves
├── confusion_matrix.png                 # Confusion matrix
├── per_class_accuracy.png              # Per-class metrics
├── sample_predictions.png               # Sample predictions
└── class_distribution.png               # Dataset distribution
```

## 📝 Files Description

### Training Scripts
- **`train_cnn.py`**: Main training pipeline with model building, training, and evaluation
- **`quick_start.py`**: Interactive wrapper around training script

### Analysis & Evaluation
- **`evaluate.py`**: Comprehensive evaluation with classification reports and confusion matrix
- **`data_analysis.py`**: Dataset statistics, class distribution, and visualizations
- **`visualization.py`**: Training history plots and sample predictions

### Inference
- **`predict.py`**: Make predictions on new images (single or batch)

### Utilities
- **`dataset_utils.py`**: Dataset loading, preprocessing, and label standardization
- **`test_integration.py`**: Integration tests to verify all components work

## 🎓 Usage Examples

### Train Model
```bash
python quick_start.py
```

### Evaluate Trained Model
```python
from evaluate import evaluate_model_comprehensive
model, preds, labels = evaluate_model_comprehensive(
    'models/final_model.keras',
    test_dataset
)
```

### Predict on Image
```python
from predict import load_model, predict_emotion

model = load_model('models/final_model.keras')
result = predict_emotion(model, 'face.jpg', return_all_probs=True)
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Visualize Training
```python
from visualization import plot_training_history
plot_training_history(history)
```

## 🔧 Customization

### Adjust Training Parameters
Edit `CONFIG` in `train_cnn.py`:
```python
CONFIG = {
    'batch_size': 64,          # Increase batch size
    'learning_rate': 0.0005,   # Lower learning rate
    'epochs': 100,             # More epochs
    'image_size': (256, 256),  # Larger images
}
```

### Add Data Augmentation
Modify `dataset_utils.py` to add augmentation:
```python
# Add after image loading
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

### Modify Model Architecture
Edit the `build_cnn_model()` function in `train_cnn.py` to add/remove layers

## 📚 Documentation

- **`TRAINING_GUIDE.md`**: Comprehensive guide on training, configuration, and troubleshooting
- **`IMPLEMENTATION_SUMMARY.md`**: Detailed implementation overview and architecture

## 🧪 Testing

Run integration tests to verify setup:
```bash
python src/test_integration.py
```

## 📊 Datasets

### RAF-DB (Real-world Affective Faces Database)
- ~20,000 images across 7 emotions
- More diverse facial expressions
- Better represents real-world scenarios

### FER-2013 (Facial Expression Recognition 2013)
- ~35,000 images across 7 emotions
- Controlled environment
- Complements RAF-DB with more varied conditions

### Label Mapping
Both datasets are standardized to the same emotion labels:
0. Angry
1. Disgust
2. Fear
3. Happy
4. Neutral
5. Sad
6. Surprise

## ⚡ Performance Tips

1. **GPU Training**: The model automatically uses GPU. For faster training, use a GPU-equipped machine
2. **Batch Size**: Larger batches train faster but use more memory
3. **Image Size**: Larger images improve accuracy but slow training
4. **Early Stopping**: Saves training time by stopping when performance plateaus

## 🐛 Troubleshooting

### Issue: "Failed to download dataset"
- Check Kaggle API credentials: `~/.kaggle/kaggle.json`
- Ensure you have internet connection
- Try installing: `pip install kagglehub`

### Issue: "No module named 'tensorflow'"
- Reinstall: `pip install --upgrade tensorflow`
- For Apple Silicon: `pip install tensorflow-macos tensorflow-metal`

### Issue: "CUDA not found" (on GPU)
- Ensure CUDA is installed and available
- Or use CPU-only version: `pip install tensorflow-cpu`

### Issue: Out of memory
- Reduce `batch_size` in CONFIG
- Reduce `image_size` in CONFIG
- Try on a machine with more GPU memory

## 📖 References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- RAF-DB Dataset: https://www.kaggle.com/shuvoalok/raf-db-dataset
- FER-2013 Dataset: https://www.kaggle.com/msambare/fer2013

## 📄 License

This project uses publicly available datasets from Kaggle.

## 👤 Author

Created for Emotion Recognition research and deep learning study.

---

**Happy training! 🚀**