# 🌿 Plant Disease Detection

A modern deep-learning project that classifies plant diseases from leaf images using a Convolutional Neural Network (CNN) built with TensorFlow / Keras.

This is a cleaned-up, modernized version of a university project originally written around 2020.

---

## Features

- Clean project structure
- Fully updated to **TensorFlow 2.x** (no mixed Keras imports)
- Proper categorical cross-entropy for multi-class classification
- Modern `model.fit()` API (no deprecated `fit_generator`)
- Image data generators with augmentation
- Easy-to-use inference script
- Optional Gradio web demo for interactive predictions
- Ready for GitHub / reproducibility

---

## Project Structure

```
plant-disease-detection/
├── data/                  # Place your dataset here
│   └── plant_dataset/     # Class folders (e.g. tomato_early_blight/, apple_scab/, ...)
├── models/                # Saved models go here
├── notebooks/             # Original + improved notebooks
├── src/
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   └── app.py             # Gradio web interface
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

Place your plant disease images in the following structure:

```
data/plant_dataset/
├── class_1/
│   ├── image1.jpg
│   └── ...
├── class_2/
│   └── ...
└── ...
```

You can use popular public datasets such as:
- PlantVillage
- PlantDoc
- Or the original university dataset that came with this project

---

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py --data-dir data/plant_dataset --epochs 25 --batch-size 32
```

Options:
- `--data-dir`     Path to dataset root
- `--epochs`       Number of training epochs (default: 25)
- `--batch-size`   Batch size (default: 32)
- `--img-size`     Image size (default: 256)
- `--model-dir`    Where to save the model (default: models/)

---

## Inference

```bash
python src/predict.py --image path/to/leaf.jpg --model models/plant_disease_model.keras
```

---

## Web Demo (Gradio)

```bash
python src/app.py
```

Then open the local URL shown in the terminal.

---

## Model Architecture

Simple but effective CNN:

- 3 convolutional blocks (32 → 64 → 128 filters)
- Batch Normalization + Dropout
- Dense head with 1024 units
- Softmax output

---

## Original vs Upgraded

| Aspect              | Original (2020)              | Upgraded (2026)                     |
|---------------------|------------------------------|-------------------------------------|
| Framework           | Mixed Keras + TF             | Pure TensorFlow 2.x / Keras 3       |
| Training API        | `fit_generator`              | `model.fit`                         |
| Loss                | binary_crossentropy          | categorical_crossentropy            |
| Optimizer           | Adam(lr=..., decay=...)      | Adam with learning rate schedule    |
| Project structure   | Flat / messy                 | Clean modular structure             |
| Reproducibility     | Hardcoded Windows paths      | Configurable CLI + relative paths   |
| Inference           | Basic script                 | CLI + Gradio demo                   |

---

## License

This project is provided for educational purposes.
