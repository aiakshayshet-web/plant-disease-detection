#!/usr/bin/env python3
"""
Train a CNN for plant disease classification.
Modern TensorFlow 2.x implementation.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Build a simple but effective CNN."""
    model = keras.Sequential(
        [
            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Head
            layers.Flatten(),
            layers.Dense(1024),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="plant_disease_cnn",
    )
    return model


def plot_history(history: keras.callbacks.History, output_dir: Path) -> None:
    """Save training curves."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] Training curves saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/plant_dataset",
        help="Path to dataset root (folders = class names)",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )
    parser.add_argument("--validation-split", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Please place your class folders under data/plant_dataset/"
        )

    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=args.validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())

    print(f"[INFO] Found {num_classes} classes: {class_names}")
    print(f"[INFO] Training samples: {train_generator.samples}")
    print(f"[INFO] Validation samples: {val_generator.samples}")

    # Build & compile model
    model = build_model(input_shape=(*img_size, 3), num_classes=num_classes)

    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks
    checkpoint_path = model_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_path = model_dir / "plant_disease_model.keras"
    model.save(final_path)
    print(f"[INFO] Final model saved to {final_path}")

    # Also save class indices for inference
    import json

    labels_path = model_dir / "class_indices.json"
    with open(labels_path, "w") as f:
        json.dump(train_generator.class_indices, f, indent=2)
    print(f"[INFO] Class indices saved to {labels_path}")

    # Plot curves
    plot_history(history, model_dir)

    # Final evaluation
    print("[INFO] Evaluating on validation set...")
    scores = model.evaluate(val_generator, verbose=0)
    print(f"[INFO] Validation Loss: {scores[0]:.4f}")
    print(f"[INFO] Validation Accuracy: {scores[1] * 100:.2f}%")


if __name__ == "__main__":
    main()
