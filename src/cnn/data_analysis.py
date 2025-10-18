"""
Data analysis utilities for understanding the combined dataset.
"""

import tensorflow as tf
from dataset_utils import get_raf_db_dataset, get_fer_2013_dataset, get_combined_datasets
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def count_samples_per_class(dataset, dataset_name="Dataset"):
    """
    Count number of samples per emotion class in a dataset.
    
    Args:
        dataset: tf.data.Dataset containing (image, label) pairs
        dataset_name (str): Name of the dataset for display
        
    Returns:
        dict: Dictionary with class counts
    """
    class_counts = Counter()
    total_samples = 0
    
    for images, labels in dataset:
        for label in labels.numpy():
            class_counts[label] += 1
            total_samples += 1
    
    print(f"\n{dataset_name} - Class Distribution:")
    print("=" * 50)
    print(f"{'Class':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    
    for class_idx in range(len(EMOTION_LABELS)):
        count = class_counts.get(class_idx, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{EMOTION_LABELS[class_idx]:<15} {count:<10} {percentage:.2f}%")
    
    print("-" * 50)
    print(f"{'Total':<15} {total_samples:<10}")
    print("=" * 50)
    
    return dict(class_counts)

def plot_class_distribution(dataset_dict, save_dir='../results'):
    """
    Create a bar plot comparing class distributions across datasets.
    
    Args:
        dataset_dict (dict): Dictionary with dataset names as keys and class counts as values
        save_dir (str): Directory to save the plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(dataset_dict), figsize=(5*len(dataset_dict), 5))
    
    if len(dataset_dict) == 1:
        axes = [axes]
    
    for ax, (dataset_name, class_counts) in zip(axes, dataset_dict.items()):
        counts = [class_counts.get(i, 0) for i in range(len(EMOTION_LABELS))]
        
        bars = ax.bar(EMOTION_LABELS, counts, color='steelblue', alpha=0.8, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = Path(save_dir) / 'class_distribution.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"\n✓ Class distribution plot saved to: {plot_path}")
    plt.close()

def analyze_image_statistics(dataset, num_batches=5, dataset_name="Dataset"):
    """
    Analyze image statistics (mean, std, min, max pixel values).
    
    Args:
        dataset: tf.data.Dataset containing (image, label) pairs
        num_batches (int): Number of batches to analyze
        dataset_name (str): Name of the dataset for display
    """
    print(f"\n{dataset_name} - Image Statistics:")
    print("=" * 50)
    
    all_pixels = []
    num_images = 0
    
    for images, _ in dataset.take(num_batches):
        all_pixels.extend(images.numpy().flatten())
        num_images += images.shape[0]
    
    all_pixels = np.array(all_pixels)
    
    print(f"Images analyzed: {num_images}")
    print(f"Mean pixel value: {all_pixels.mean():.4f}")
    print(f"Std pixel value: {all_pixels.std():.4f}")
    print(f"Min pixel value: {all_pixels.min():.4f}")
    print(f"Max pixel value: {all_pixels.max():.4f}")
    print(f"Median pixel value: {np.median(all_pixels):.4f}")
    print("=" * 50)

def compare_datasets():
    """
    Compare statistics across RAF-DB, FER-2013, and combined datasets.
    """
    print("\n" + "="*80)
    print("Dataset Comparison Analysis")
    print("="*80)
    
    print("\nLoading RAF-DB dataset...")
    raf_train, raf_val, raf_test = get_raf_db_dataset(batch_size=32)
    
    print("Loading FER-2013 dataset...")
    fer_train, fer_val, fer_test = get_fer_2013_dataset(batch_size=32)
    
    print("Loading combined dataset...")
    combined_train, combined_val, combined_test = get_combined_datasets(batch_size=32)
    
    raf_counts = count_samples_per_class(raf_test, "RAF-DB Test Set")
    fer_counts = count_samples_per_class(fer_test, "FER-2013 Test Set")
    combined_counts = count_samples_per_class(combined_test, "Combined Test Set")
    
    plot_class_distribution({
        'RAF-DB': raf_counts,
        'FER-2013': fer_counts,
        'Combined': combined_counts
    })
    
    analyze_image_statistics(raf_test, num_batches=3, dataset_name="RAF-DB")
    analyze_image_statistics(fer_test, num_batches=3, dataset_name="FER-2013")
    analyze_image_statistics(combined_test, num_batches=3, dataset_name="Combined")

def visualize_sample_images(dataset, num_images=12, dataset_name="Dataset", save_dir='../results'):
    """
    Visualize sample images from a dataset.
    
    Args:
        dataset: tf.data.Dataset containing (image, label) pairs
        num_images (int): Number of images to visualize
        dataset_name (str): Name of the dataset
        save_dir (str): Directory to save the plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    images, labels = next(iter(dataset.take(1)))
    
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    axes = axes.flatten()
    
    for idx in range(min(num_images, len(images))):
        ax = axes[idx]
        
        img = images[idx].numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        if img.shape[-1] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img.squeeze(), cmap='gray')
        
        label = EMOTION_LABELS[labels[idx].numpy()]
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name} - Sample Images', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = Path(save_dir) / f'sample_images_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"✓ Sample images saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    compare_datasets()
    
    print("\nLoading combined dataset for visualization...")
    train_ds, val_ds, test_ds = get_combined_datasets(batch_size=32)
    
    print("\nVisualizing sample images...")
    visualize_sample_images(test_ds, num_images=12, dataset_name="Combined Dataset")
    
    print("\n✓ Analysis complete!")
