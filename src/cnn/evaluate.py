import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def evaluate_model_comprehensive(model_path, test_dataset, save_dir='../results'):
    """
    Comprehensive model evaluation with classification report and confusion matrix.
    
    Args:
        model_path (str): Path to saved model
        test_dataset: tf.data.Dataset containing test data
        save_dir (str): Directory to save evaluation results
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model = keras.models.load_model(model_path)
    
    print("Making predictions on test set...")
    all_predictions = []
    all_true_labels = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        all_predictions.extend(pred_labels)
        all_true_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"\n✓ Overall Accuracy: {accuracy:.4f}")
    
    print("\n" + "="*80)
    print("Classification Report:")
    print("="*80)
    print(classification_report(
        all_true_labels, all_predictions,
        target_names=EMOTION_LABELS,
        digits=4
    ))
    
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = Path(save_dir) / 'confusion_matrix.png'
    plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {cm_path}")
    plt.close()
    
    return model, all_predictions, all_true_labels

def plot_per_class_metrics(true_labels, pred_labels, save_dir='../results'):
    """
    Plot per-class accuracy and other metrics.
    
    Args:
        true_labels (np.ndarray): Array of true labels
        pred_labels (np.ndarray): Array of predicted labels
        save_dir (str): Directory to save plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    per_class_acc = []
    for class_idx in range(len(EMOTION_LABELS)):
        mask = true_labels == class_idx
        if mask.sum() > 0:
            class_acc = (pred_labels[mask] == true_labels[mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(EMOTION_LABELS, per_class_acc, color='steelblue', alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(save_dir) / 'per_class_accuracy.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"✓ Per-class accuracy plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    print("This module provides comprehensive evaluation utilities for the trained CNN.")
    print("Use evaluate_model_comprehensive() and plot_per_class_metrics() functions.")
