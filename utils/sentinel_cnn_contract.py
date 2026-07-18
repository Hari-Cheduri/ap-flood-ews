"""
utils/sentinel_cnn_contract.py
------------------------------
Single source of truth for the Sentinel-1 CNN input format.

The existing historical dataset was created by utils/sentinel_dataset_builder.py
as 64x64 RGB PNG thumbnails over an 18 km buffered point AOI using:

    channel 0: VV visualized from -25 to 0 dB
    channel 1: VH visualized from -30 to -5 dB
    channel 2: VV - VH visualized from 0 to 20 dB

The live fetcher must use the same contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


CONTRACT_NAME = "sentinel1_png_rgb_v1"

IMAGE_SIZE = 64
INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)

BUFFER_KM = 18.0
TRAINING_WINDOW_DAYS = 12

VIS_MIN = (-25.0, -30.0, 0.0)
VIS_MAX = (0.0, -5.0, 20.0)

X_PATH = Path("data/processed/satellite_images.npy")
Y_PATH = Path("data/processed/image_labels.npy")
METADATA_PATH = Path("data/processed/satellite_metadata.csv")

LIVE_PATCH_PATH = Path("data/processed/sentinel1_live_patch.npy")
LIVE_METADATA_PATH = Path("data/processed/sentinel1_live_patch.json")

MODEL_KERAS_PATH = Path("models/real_cnn_model.keras")
MODEL_H5_PATH = Path("models/real_cnn_model.h5")
THRESHOLD_PATH = Path("models/cnn_threshold.json")
METRICS_PATH = Path("models/cnn_metrics.json")
SPLIT_PATH = Path("models/cnn_split_indices.npz")
HISTORY_PATH = Path("models/cnn_history.json")
LIVE_RESULT_PATH = Path("data/processed/live_cnn_prediction.json")


def validate_patch(patch: np.ndarray, name: str = "patch") -> np.ndarray:
    """Validate and return a float32 CNN patch."""
    array = np.asarray(patch, dtype=np.float32)

    if array.shape != INPUT_SHAPE:
        raise ValueError(
            f"{name} must have shape {INPUT_SHAPE}; received {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    minimum = float(array.min())
    maximum = float(array.max())

    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"{name} must be normalized to [0, 1]; "
            f"received range {minimum:.6f} to {maximum:.6f}"
        )

    return np.clip(array, 0.0, 1.0).astype(np.float32)
