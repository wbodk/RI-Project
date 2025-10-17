# CNN Emotion Recognition System

This project implements a Convolutional Neural Network (CNN) for emotion recognition trained on a combination of RAF-DB and FER-2013 datasets.

## Project Structure

```
src/
├── dataset_utils.py       # Dataset loading and preprocessing utilities
├── train_cnn.py          # Main training script
├── evaluate.py           # Comprehensive model evaluation
├── visualization.py      # Training visualization utilities
└── predict.py            # Inference script for predictions
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

The script will automatically download the datasets from Kaggle on first run. Make sure you have:
- Kaggle API credentials configured (`~/.kaggle/kaggle.json`)

Datasets will be stored in `data/` directory:
- `data/raf-db-dataset/` - RAF-DB dataset
- `data/fer2013/` - FER-2013 dataset

## Usage

### Training the Model

```bash
cd src
python train_cnn.py
```

**What this does:**
1. Downloads and loads both datasets (if not already present)
2. Combines RAF-DB and FER-2013 datasets
3. Builds and compiles the CNN model
4. Trains the model with:
   - Early stopping (patience: 10 epochs)
   - Model checkpointing (saves best model)
   - Learning rate reduction on plateau
   - TensorBoard logging
5. Evaluates on test set
6. Saves the final model to `models/final_model.keras`

**Configuration (in `train_cnn.py`):**
```python
CONFIG = {
    'image_size': (128, 128),      # Input image size
    'batch_size': 32,              # Training batch size
    'epochs': 50,                  # Maximum epochs
    'learning_rate': 0.001,        # Initial learning rate
    'num_classes': 7,              # Number of emotion classes
    'validation_split': 0.15       # Validation split ratio
}
```

### Model Architecture

The CNN consists of 4 convolutional blocks with the following structure:

```
Input (128x128x3)
    ↓
Conv Block 1: 32 filters → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: 64 filters → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 3: 128 filters → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 4: 256 filters → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Global Average Pooling
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(7) with Softmax (output layer)
```

### Emotion Classes

The model recognizes 7 emotions:
0. Angry
1. Disgust
2. Fear
3. Happy
4. Neutral
5. Sad
6. Surprise

### Making Predictions

After training, use the prediction script to classify emotions:

**Single image:**
```bash
cd src
python predict.py ../models/final_model.keras --image path/to/image.jpg
```

**Batch prediction:**
```bash
cd src
python predict.py ../models/final_model.keras --dir path/to/images/
```

**With all probabilities:**
```bash
cd src
python predict.py ../models/final_model.keras --image path/to/image.jpg --all-probs
```

### Model Evaluation

For comprehensive evaluation:

```python
from evaluate import evaluate_model_comprehensive, plot_per_class_metrics

model, predictions, true_labels = evaluate_model_comprehensive(
    '../models/final_model.keras',
    test_dataset
)
plot_per_class_metrics(true_labels, predictions)
```

This generates:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Per-class accuracy plot

### Visualization

Plot training history:

```python
from visualization import plot_training_history
from train_cnn import train_cnn

model, history = train_cnn()
plot_training_history(history)
```

This generates plots for:
- Training vs validation accuracy
- Training vs validation loss

## Output Files

During training, the following files are created:

```
models/
├── best_model_YYYYMMDD_HHMMSS.keras    # Best model checkpoint
├── final_model.keras                    # Final trained model
└── logs_YYYYMMDD_HHMMSS/               # TensorBoard logs

results/
├── training_history.png                 # Training curves
├── sample_predictions.png               # Sample predictions
├── confusion_matrix.png                 # Confusion matrix
└── per_class_accuracy.png              # Per-class metrics
```

## Training Tips

1. **GPU Usage**: The model automatically uses GPU if available (with TensorFlow Metal on Apple Silicon Macs)

2. **Dataset Balance**: The combined dataset includes:
   - RAF-DB: ~16k training images, ~4k test images
   - FER-2013: ~28k training images, ~7k test images
   
3. **Hyperparameter Tuning**: Adjust `CONFIG` in `train_cnn.py` for:
   - Different learning rates
   - Batch sizes
   - Number of epochs
   - Image sizes

4. **Data Augmentation**: Consider adding data augmentation in `dataset_utils.py` for better generalization

5. **Early Stopping**: The model stops training if validation loss doesn't improve for 10 epochs

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- Keras 3.11+
- NumPy, Pandas, Pillow
- Matplotlib, Seaborn, Scikit-learn
- Kaggle API

## Dataset Attribution

- **RAF-DB**: Real-world Affective Faces Database
- **FER-2013**: Facial Expression Recognition 2013 Challenge

## License

This project uses publicly available datasets from Kaggle.
