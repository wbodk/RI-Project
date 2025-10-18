import kaggle
import keras
import tensorflow as tf
from pathlib import Path
from typing import Optional, Any

DATA_ROOT = Path("../data")

RAF_DB_DATASET = "shuvoalok/raf-db-dataset"
RAF_DB_TO_UNIFIED = tf.constant([6, 2, 1, 3, 5, 0, 4])

FER_2013_DATASET = "msambare/fer2013"
FER_2013_TO_UNIFIED = tf.constant([0, 1, 2, 3, 4, 5, 6])
# 0 - angry, 1 - disgust, 2 - fear, 3 - happy, 4 - neutral, 5 - sad, 6 - surprise 

def _download_dataset(dataset_url: str) -> Path:
    """
    Download and extract a dataset from Kaggle if not already present.
    Args:
        dataset_url (str): Kaggle dataset identifier (e.g., 'username/dataset-name').
    Returns:
        Path: Path to the downloaded dataset directory.
    """
    dataset_name = dataset_url.split("/")[1]
    dataset_path = DATA_ROOT / dataset_name
    DATA_ROOT.mkdir(exist_ok=True)
    if dataset_path.exists():
        print("Dataset is already downloaded")
        return dataset_path
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset_url, path=dataset_path, unzip=True)
        print(f"Downloaded dataset to {dataset_path}")
        return dataset_path
    except Exception as e:
        print("Failed to download dataset: ", e)
        raise

def _standardize_labels(dataset: tf.data.Dataset, mapping: Any) -> tf.data.Dataset:
    """
    Remap dataset labels using a mapping tensor to a unified label order.
    Args:
        dataset (tf.data.Dataset): Input dataset with (image, label) pairs.
        mapping (tf.Tensor): 1-D tensor where mapping[old_index] == new_index.
    Returns:
        tf.data.Dataset: Dataset with remapped labels.
    """
    def _map_fn(images, labels):
        labels = tf.cast(labels, tf.int64)
        new_labels = tf.gather(mapping, labels)
        return images, tf.cast(new_labels, tf.int32)
    return dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

def _normalize_images(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Normalize image pixel values to [0, 1].
    Args:
        dataset (tf.data.Dataset): Input dataset with (image, label) pairs.
    Returns:
        tf.data.Dataset: Dataset with normalized images.
    """
    def _norm_fn(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels
    return dataset.map(_norm_fn, num_parallel_calls=tf.data.AUTOTUNE)

def _augment_images(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply data augmentation to increase dataset diversity.
    Includes random rotation, shift, zoom, and horizontal flip.
    
    Args:
        dataset (tf.data.Dataset): Input dataset with (image, label) pairs.
        
    Returns:
        tf.data.Dataset: Dataset with augmented images.
    """
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomFlip("horizontal"),
    ])
    
    def _aug_fn(images, labels):
        images = data_augmentation(images, training=True)
        return images, labels
    
    return dataset.map(_aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

def get_raf_db_dataset(
    image_size=(128,128),
    batch_size=32,
    validation_split=0.15,
    standardize_labels=True,
    normalize=True
):
    """
    Load the RAF-DB dataset, split into train, validation, and test sets, and optionally standardize labels and normalize images.
    Args:
        image_size (tuple): Image size (height, width) for resizing.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of training data to use for validation.
        standardize_labels (bool): Whether to remap labels to unified order.
        normalize (bool): Whether to normalize pixel values to [0, 1].
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    raf_db_path = _download_dataset(RAF_DB_DATASET)
    test_path = raf_db_path/"DATASET/test"
    train_path = raf_db_path/"DATASET/train"

    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=666,
        validation_split=validation_split,
        subset="training"
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=666,
        validation_split=validation_split,
        subset="validation"
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        batch_size=batch_size
    )
    if standardize_labels:
        train_ds = _standardize_labels(train_ds, RAF_DB_TO_UNIFIED)
        val_ds = _standardize_labels(val_ds, RAF_DB_TO_UNIFIED)
        test_ds = _standardize_labels(test_ds, RAF_DB_TO_UNIFIED)
    if normalize:
        train_ds = _normalize_images(train_ds)
        val_ds = _normalize_images(val_ds)
        test_ds = _normalize_images(test_ds)
    return train_ds, val_ds, test_ds

def get_fer_2013_dataset(
    image_size=(128,128),
    batch_size=32,
    validation_split=0.15,
    standardize_labels=True,
    normalize=True
):
    """
    Load the FER-2013 dataset, split into train, validation, and test sets, and optionally standardize labels and normalize images.
    Args:
        image_size (tuple): Image size (height, width) for resizing.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of training data to use for validation.
        standardize_labels (bool): Whether to remap labels to unified order.
        normalize (bool): Whether to normalize pixel values to [0, 1].
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    fer_2013_path = _download_dataset(FER_2013_DATASET)
    test_path = fer_2013_path/"test"
    train_path = fer_2013_path/"train"

    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=666,
        validation_split=validation_split,
        subset="training"
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=666,
        validation_split=validation_split,
        subset="validation"
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        batch_size=batch_size
    )
    if standardize_labels:
        train_ds = _standardize_labels(train_ds, FER_2013_TO_UNIFIED)
        val_ds = _standardize_labels(val_ds, FER_2013_TO_UNIFIED)
        test_ds = _standardize_labels(test_ds, FER_2013_TO_UNIFIED)
    if normalize:
        train_ds = _normalize_images(train_ds)
        val_ds = _normalize_images(val_ds)
        test_ds = _normalize_images(test_ds)
    return train_ds, val_ds, test_ds

def get_combined_datasets(
    image_size=(128,128),
    batch_size=32,
    validation_split=0.15,
    standardize_labels=True,
    normalize=True
):
    """
    Combine RAF-DB and FER-2013 datasets into unified train, validation, and test sets.
    Args:
        image_size (tuple): Image size (height, width) for resizing.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of training data to use for validation.
        standardize_labels (bool): Whether to remap labels to unified order.
        normalize (bool): Whether to normalize pixel values to [0, 1].
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    raf_train, raf_val, raf_test = get_raf_db_dataset(
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        standardize_labels=standardize_labels,
        normalize=normalize
    )
    fer_train, fer_val, fer_test = get_fer_2013_dataset(
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        standardize_labels=standardize_labels,
        normalize=normalize
    )

    train_ds = raf_train.concatenate(fer_train).shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = raf_val.concatenate(fer_val).shuffle(2000).prefetch(tf.data.AUTOTUNE)
    test_ds = raf_test.concatenate(fer_test).prefetch(tf.data.AUTOTUNE)
    
    # Apply data augmentation to training set only
    print("Applying data augmentation to training set...")
    train_ds = _augment_images(train_ds)

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    get_combined_datasets()