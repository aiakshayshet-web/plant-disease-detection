#!/usr/bin/env python3
"""
Predict plant disease from a single image.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


def load_class_names(model_dir: Path) -> list[str]:
    """Load class names from class_indices.json (sorted by index)."""
    labels_path = model_dir / "class_indices.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"class_indices.json not found in {model_dir}. "
            "Train the model first or provide the file."
        )
    with open(labels_path) as f:
        class_indices = json.load(f)
    # Invert: index -> name
    idx_to_name = {v: k for k, v in class_indices.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]


def preprocess_image(image_path: str, img_size: int = 256) -> np.ndarray:
    """Load and preprocess a single image for the model."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_path: str, model_path: str, top_k: int = 3) -> None:
    model_path = Path(model_path)
    model_dir = model_path.parent

    print(f"[INFO] Loading model from {model_path} ...")
    model = keras.models.load_model(model_path)

    class_names = load_class_names(model_dir)
    print(f"[INFO] Classes: {class_names}")

    img_array = preprocess_image(image_path)
    preds = model.predict(img_array, verbose=0)[0]

    # Top-k results
    top_indices = np.argsort(preds)[::-1][:top_k]

    print("\n=== Prediction Results ===")
    print(f"Image: {image_path}\n")
    for rank, idx in enumerate(top_indices, 1):
        conf = preds[idx] * 100
        print(f"{rank}. {class_names[idx]:<40} {conf:6.2f}%")
    print("==========================\n")


def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from an image")
    parser.add_argument("--image", required=True, help="Path to the leaf image")
    parser.add_argument(
        "--model",
        default="models/plant_disease_model.keras",
        help="Path to the trained .keras model",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    predict(args.image, args.model, top_k=args.top_k)


if __name__ == "__main__":
    main()
