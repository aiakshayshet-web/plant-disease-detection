#!/usr/bin/env python3
"""
Gradio web interface for Plant Disease Detection.
"""

from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import json

MODEL_PATH = Path("models/plant_disease_model.keras")
LABELS_PATH = Path("models/class_indices.json")

model = None
class_names = []


def load_resources():
    global model, class_names
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )
    model = keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
    idx_to_name = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]


def predict_image(image: Image.Image):
    if model is None:
        load_resources()

    # Preprocess
    img = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    # Return dictionary of class -> confidence for Gradio Label
    results = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return results


def main():
    load_resources()

    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload a leaf image"),
        outputs=gr.Label(num_top_classes=5, label="Predictions"),
        title="🌿 Plant Disease Detection",
        description=(
            "Upload a photo of a plant leaf and the model will predict "
            "the most likely disease (or healthy status)."
        ),
        examples=None,  # You can add example image paths here later
        allow_flagging="never",
    )

    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
