"""
models/test_real_cnn.py
-----------------------
Quick prediction test for trained CNN.
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = "models/real_cnn_model.h5"
THRESHOLD_PATH = "models/cnn_threshold.json"

X_PATH = "data/processed/satellite_images.npy"
Y_PATH = "data/processed/image_labels.npy"


def main():
    if not os.path.exists(MODEL_PATH):
        print("\nERROR: Model not found:", MODEL_PATH)
        print("Run this first:")
        print("python -m models.train_real_cnn")
        return

    threshold = 0.5

    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            threshold = float(json.load(f).get("cnn_threshold", 0.5))

    print("[TEST] Loading model...")
    model = load_model(MODEL_PATH, compile=False)

    print("[TEST] Loading dataset...")
    X = np.load(X_PATH).astype("float32")
    y = np.load(Y_PATH).astype("int32")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Threshold:", threshold)

    print("[TEST] Running predictions...")
    probs = model.predict(X[:10]).ravel()

    for i, prob in enumerate(probs):
        predicted_label = 1 if prob >= threshold else 0

        print(
            f"Sample {i + 1}: "
            f"Actual={y[i]} | "
            f"Predicted={predicted_label} | "
            f"Flood Probability={prob:.3f}"
        )


if __name__ == "__main__":
    main()