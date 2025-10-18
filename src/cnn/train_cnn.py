import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from dataset_utils import get_combined_datasets

tf.random.set_seed(42)

# Configuration
CONFIG = {
    'image_size': (128, 128),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'num_classes': 7,  
    'validation_split': 0.15
}

def build_cnn_model(input_shape=(128, 128, 3), num_classes=7):
    """
    Build a simple CNN model for emotion recognition.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of emotion classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling and Dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate optimizer, loss, and metrics.
    
    Args:
        model (keras.Model): Model to compile
        learning_rate (float): Learning rate for the optimizer
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

def calculate_class_weights(train_ds, num_classes=7):
    """
    Calculate class weights to balance underrepresented classes.
    
    Args:
        train_ds: Training dataset
        num_classes (int): Number of emotion classes
        
    Returns:
        dict: Dictionary mapping class indices to weights
    """
    print("\nCalculating class weights for balanced training...")
    
    # Collect all labels from the dataset
    all_labels = []
    for images, labels in train_ds:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    
    # Convert to dictionary format
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("✓ Class weights calculated:")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    for class_idx, weight in class_weight_dict.items():
        count = np.sum(all_labels == class_idx)
        print(f"  {emotion_labels[class_idx]:<12}: weight={weight:.4f}, samples={count}")
    
    return class_weight_dict

def create_callbacks(model_dir='../models'):
    """
    Create training callbacks for early stopping and checkpointing.
    
    Args:
        model_dir (str): Directory to save model checkpoints
        
    Returns:
        list: List of Keras callbacks
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(model_dir) / f"best_model_{timestamp}.keras"
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=Path(model_dir) / f'logs_{timestamp}',
            histogram_freq=0,
            write_graph=False
        )
    ]
    
    return callbacks, str(model_path)

def train_cnn():
    """
    Main training function that loads datasets, builds, and trains the CNN model.
    """
    print("=" * 80)
    print("Emotion Recognition CNN - Training Script")
    print("=" * 80)
    
    # Load combined datasets
    print("\nLoading combined datasets (RAF-DB + FER-2013)...")
    train_ds, val_ds, test_ds = get_combined_datasets(
        image_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        validation_split=CONFIG['validation_split'],
        standardize_labels=True,
        normalize=True
    )
    print("✓ Datasets loaded successfully!")
    
    # Build model
    print("\nBuilding CNN model...")
    model = build_cnn_model(
        input_shape=(*CONFIG['image_size'], 3),
        num_classes=CONFIG['num_classes']
    )
    print("✓ Model built successfully!")
    print(f"\nModel Summary:")
    model.summary()
    
    # Compile model
    print("\nCompiling model...")
    compile_model(model, learning_rate=CONFIG['learning_rate'])
    print("✓ Model compiled!")
    
    # Create callbacks
    print("\nSetting up training callbacks...")
    callbacks, best_model_path = create_callbacks()
    print(f"✓ Best model will be saved to: {best_model_path}")
    
    # Calculate class weights for balanced training
    class_weight_dict = calculate_class_weights(train_ds, CONFIG['num_classes'])
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting training with class balancing...")
    print("=" * 80 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80 + "\n")
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n✓ Test Accuracy: {test_accuracy:.4f}")
    print(f"✓ Test Loss: {test_loss:.4f}")
    
    # Save final model
    final_model_path = Path("../models") / "final_model.keras"
    Path("../models").mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    return model, history

if __name__ == "__main__":
    # Change to src directory if not already there
    if not os.path.exists("dataset_utils.py"):
        os.chdir(Path(__file__).parent)
    
    model, history = train_cnn()
