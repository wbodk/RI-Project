import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from src.dataset_utils import get_combined_datasets
from src.train_cnn import create_callbacks
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Reproducibility
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'image_size': (128, 128),
    'batch_size': 32,
    'epochs_head': 10,
    'epochs_finetune': 10,
    'lr_head': 1e-4,
    'lr_finetune': 1e-5,
    'num_classes': 7,
    'validation_split': 0.15,
    'target_balance': 'max',  # strategy to balance: 'max' or 'median'
    'unfreeze_top_n_layers': 50,  # number of top ResNet layers to unfreeze for fine-tuning
    'clipnorm': 1.0,  # gradient clipping norm for fine-tune optimizer
    'model_dir': '../models'  # where to save checkpoints and final model
}

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def build_resnet_model(input_shape=(128, 128, 3), num_classes=7):
    """Build transfer-learning model using ResNet50 as backbone and custom head.

    Head: GlobalAveragePooling2D -> BatchNormalization -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(num_classes, softmax)
    """
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False  # freeze for head training

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def _make_augmentation_model():
    """Return a keras Sequential augmentation model matching the spec:
    flip, rotation +/-15deg (~0.0833), translation +/-10%, zoom +/-5%, contrast +/-20%.
    """
    aug = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.0833),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.2)
    ], name='augmentation')
    return aug


def compute_class_counts(dataset, num_classes):
    """Compute class counts by iterating over the (batched) dataset's labels.
    Works with tf.data.Dataset returning (images, labels) batches.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for images, labels in dataset.unbatch().as_numpy_iterator():
        counts[int(labels)] += 1
    return counts


def create_balanced_train_dataset(train_ds, batch_size, num_classes, strategy='max'):
    """Create a balanced training dataset by oversampling minority classes with augmentation.

    Approach:
    - Unbatch original train_ds
    - Compute class counts
    - For each class, filter examples and (if needed) repeat+augment to reach target_count
    - Concatenate per-class datasets, shuffle, batch and prefetch
    """
    AUTOTUNE = tf.data.AUTOTUNE
    aug_model = _make_augmentation_model()

    # Work on unbatched dataset of single examples
    unbatched = train_ds.unbatch()

    # Compute counts
    print('\nComputing class distribution on training set...')
    counts = compute_class_counts(train_ds, num_classes)
    print('\nClass counts:', counts)

    if strategy == 'max':
        target = int(counts.max())
    elif strategy == 'median':
        target = int(np.median(counts))
    else:
        target = int(counts.max())

    print(f"Target samples per class for balancing: {target} (strategy={strategy})")

    per_class_datasets = []

    for cls in range(num_classes):
        # Filter for class
        def _filter_fn(x, y, cls=cls):
            return tf.equal(y, cls)

        ds_cls = unbatched.filter(_filter_fn)

        # If class is minority, augment and repeat to reach target
        cls_count = int(counts[cls])
        if cls_count == 0:
            # Skip empty classes
            print(f"Warning: class {cls} has 0 samples; skipping.")
            continue

        if cls_count < target:
            # Apply augmentation to produce variety;
            # expand dims so augmentation layers operate batch-wise then squeeze
            def _augment_image(x, y):
                x_aug = aug_model(tf.expand_dims(x, 0), training=True)[0]
                return x_aug, y

            ds_aug = ds_cls.map(_augment_image, num_parallel_calls=AUTOTUNE)
            # create repeated augmented dataset to reach target
            ds_combined = ds_cls.concatenate(ds_aug).repeat()
            ds_combined = ds_combined.take(target)
        else:
            ds_combined = ds_cls.take(cls_count)

        per_class_datasets.append(ds_combined)

    # Concatenate all class datasets
    balanced = None
    for ds in per_class_datasets:
        if balanced is None:
            balanced = ds
        else:
            balanced = balanced.concatenate(ds)

    if balanced is None:
        raise RuntimeError('No training data available after balancing.')

    # Shuffle, batch, prefetch
    balanced = balanced.shuffle(10000).batch(batch_size).prefetch(AUTOTUNE)
    return balanced


def train_resnet():
    print("\n" + "=" * 80)
    print("Emotion Recognition - ResNet50 Transfer Learning (Phase 2)")
    print("=" * 80 + "\n")

    # Load datasets
    print("Loading combined datasets (RAF-DB + FER-2013)...")
    train_ds, val_ds, test_ds, raf_test, fer_test = get_combined_datasets(
        image_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        validation_split=CONFIG['validation_split'],
        standardize_labels=True,
        normalize=True
    )
    print('\u2713 Datasets loaded')

    # Create balanced training dataset with augmentation for minorities
    print('\nPreparing balanced training dataset with augmentation...')
    balanced_train = create_balanced_train_dataset(
        train_ds,
        batch_size=CONFIG['batch_size'],
        num_classes=CONFIG['num_classes'],
        strategy=CONFIG['target_balance']
    )
    print('\u2713 Balanced training dataset ready')

    # Build model
    print('\nBuilding ResNet50-based model...')
    model, base_model = build_resnet_model(
        input_shape=(*CONFIG['image_size'], 3),
        num_classes=CONFIG['num_classes']
    )
    model.summary()

    # Compile head
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['lr_head'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks, best_model_path = create_callbacks(model_dir=CONFIG.get('model_dir', '../models'))

    # Train head
    print(f"\nTraining head for {CONFIG['epochs_head']} epochs with lr={CONFIG['lr_head']}...")
    history_head = model.fit(
        balanced_train,
        validation_data=val_ds,
        epochs=CONFIG['epochs_head'],
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning: unfreeze top layers of base_model
    print('\nStarting fine-tuning...')
    base_model.trainable = True
    # Freeze lower layers, unfreeze top N layers
    if CONFIG['unfreeze_top_n_layers'] is not None:
        for layer in base_model.layers[:-CONFIG['unfreeze_top_n_layers']]:
            layer.trainable = False
    else:
        # unfreeze all
        for layer in base_model.layers:
            layer.trainable = True

    # Additionally, keep all BatchNormalization layers non-trainable to stabilize fine-tuning
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Re-compile with lower learning rate and gradient clipping (clipnorm)
    optimizer_finetune = keras.optimizers.Adam(
        learning_rate=CONFIG['lr_finetune'],
        clipnorm=CONFIG.get('clipnorm', 1.0)
    )
    model.compile(
        optimizer=optimizer_finetune,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_ft = model.fit(
        balanced_train,
        validation_data=val_ds,
        epochs=CONFIG['epochs_finetune'],
        callbacks=callbacks,
        verbose=1
    )

    # Final evaluation
    print('\nEvaluating on test set...')
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f'\n\u2713 Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

    # Post-hoc evaluation: predictions on whole test set and per-source
    def _gather_preds(dataset):
        y_true = []
        y_pred = []
        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            y_pred.extend(list(np.argmax(preds, axis=1)))
            y_true.extend(list(labels.numpy()))
        return np.array(y_true), np.array(y_pred)

    results_dir = Path('../results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # overall
    y_true_all, y_pred_all = _gather_preds(test_ds)
    print('\nClassification report (overall):')
    print(classification_report(y_true_all, y_pred_all, target_names=EMOTION_LABELS, digits=4))

    # confusion matrix overall
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Overall)')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix_overall.png')
    plt.close()

    # per-dataset evaluation
    print('\nEvaluating on RAF test set...')
    y_true_raf, y_pred_raf = _gather_preds(raf_test)
    print(classification_report(y_true_raf, y_pred_raf, target_names=EMOTION_LABELS, digits=4))
    cm_r = confusion_matrix(y_true_raf, y_pred_raf)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_r, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix (RAF test)')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix_raf.png')
    plt.close()

    print('\nEvaluating on FER-2013 test set...')
    y_true_fer, y_pred_fer = _gather_preds(fer_test)
    print(classification_report(y_true_fer, y_pred_fer, target_names=EMOTION_LABELS, digits=4))
    cm_f = confusion_matrix(y_true_fer, y_pred_fer)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_f, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix (FER test)')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix_fer.png')
    plt.close()

    # Save final model
    # Save final model into configured model_dir
    final_model_path = Path(CONFIG.get('model_dir', '../models')) / 'final_model_resnet.keras'
    Path(CONFIG.get('model_dir', '../models')).mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    print(f"\n\u2713 Final ResNet model saved to: {final_model_path}")

    return model, history_head, history_ft


if __name__ == '__main__':
    # Ensure script runs from src directory
    if not os.path.exists('dataset_utils.py'):
        os.chdir(Path(__file__).parent)
    train_resnet()
