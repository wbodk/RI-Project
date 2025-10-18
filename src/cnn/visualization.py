import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_history(history, save_dir='../results'):
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Keras training history object
        save_dir (str): Directory to save plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(save_dir) / 'training_history.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {plot_path}")
    plt.close()

def plot_predictions_sample(model, dataset, num_samples=12, save_dir='../results', emotion_labels=None):
    """
    Plot sample predictions from the model.
    
    Args:
        model: Trained Keras model
        dataset: tf.data.Dataset containing test samples
        num_samples (int): Number of samples to visualize
        save_dir (str): Directory to save plots
        emotion_labels (list): List of emotion labels
    """
    if emotion_labels is None:
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get a batch of images
    images, labels = next(iter(dataset.take(1)))
    
    # Make predictions
    predictions = model.predict(images[:num_samples], verbose=0)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (image, true_label, pred_probs) in enumerate(zip(
        images[:num_samples], labels[:num_samples], predictions[:num_samples]
    )):
        ax = axes[idx]
        
        # Denormalize image (assuming it was normalized to [0, 1])
        img = (image.numpy() * 255).astype(np.uint8)
        if img.shape[-1] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img.squeeze(), cmap='gray')
        
        pred_label = np.argmax(pred_probs)
        confidence = np.max(pred_probs)
        
        true_emotion = emotion_labels[true_label.numpy()]
        pred_emotion = emotion_labels[pred_label]
        
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(
            f"True: {true_emotion}\nPred: {pred_emotion}\nConf: {confidence:.2f}",
            color=color,
            fontsize=10,
            fontweight='bold'
        )
        ax.axis('off')
    
    plt.tight_layout()
    plot_path = Path(save_dir) / 'sample_predictions.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"✓ Sample predictions plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    print("This module provides visualization utilities for the CNN training.")
    print("Import and use plot_training_history() and plot_predictions_sample() functions.")
