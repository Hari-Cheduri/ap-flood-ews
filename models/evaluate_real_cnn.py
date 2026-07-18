"""
models/evaluate_real_cnn.py
---------------------------
Re-evaluate the trained CNN using the exact saved test indices.

Run:
    python -m models.evaluate_real_cnn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.sentinel_cnn_contract import (
    INPUT_SHAPE,
    METRICS_PATH,
    MODEL_H5_PATH,
    MODEL_KERAS_PATH,
    SPLIT_PATH,
    THRESHOLD_PATH,
    X_PATH,
    Y_PATH,
)


RECOMPUTED_PATH = Path("models/cnn_metrics_recomputed.json")


def load_model() -> tf.keras.Model:
    if MODEL_KERAS_PATH.exists():
        return tf.keras.models.load_model(str(MODEL_KERAS_PATH), compile=False)
    if MODEL_H5_PATH.exists():
        return tf.keras.models.load_model(str(MODEL_H5_PATH), compile=False)
    raise FileNotFoundError(
        "No CNN model found. Run: python -m models.train_real_cnn"
    )


def specificity(matrix: np.ndarray) -> float:
    tn, fp, fn, tp = matrix.ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def main() -> None:
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError("CNN dataset files are missing")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_PATH}. Retrain with the new training script."
        )
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {THRESHOLD_PATH}. Retrain the CNN."
        )

    X = np.load(X_PATH, allow_pickle=False).astype(np.float32)
    y = np.load(Y_PATH, allow_pickle=False).astype(np.int32).reshape(-1)

    if tuple(X.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"Unexpected CNN input shape: {X.shape}")

    split = np.load(SPLIT_PATH, allow_pickle=False)
    test_idx = split["test_idx"].astype(np.int64)

    if test_idx.size == 0 or int(test_idx.max()) >= len(X):
        raise ValueError("Saved test indices do not match the dataset")

    threshold_payload = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )
    threshold = float(threshold_payload["cnn_threshold"])

    model = load_model()
    X_test = X[test_idx]
    y_test = y[test_idx]

    probs = model.predict(X_test, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(np.int32)

    matrix = confusion_matrix(y_test, preds, labels=[0, 1])
    roc_auc = (
        float(roc_auc_score(y_test, probs))
        if len(np.unique(y_test)) == 2
        else 0.5
    )
    pr_auc = (
        float(average_precision_score(y_test, probs))
        if len(np.unique(y_test)) == 2
        else float(np.mean(y_test))
    )

    stats = {
        "min": float(probs.min()),
        "max": float(probs.max()),
        "mean": float(probs.mean()),
        "std": float(probs.std()),
        "p05": float(np.percentile(probs, 5)),
        "p95": float(np.percentile(probs, 95)),
        "spread_p95_p05": float(
            np.percentile(probs, 95) - np.percentile(probs, 5)
        ),
    }
    compressed = bool(
        stats["std"] < 0.02
        or stats["spread_p95_p05"] < 0.05
    )

    collapsed = bool(
        stats["spread_p95_p05"] < 0.005
        and roc_auc < 0.60
    )

    result: Dict[str, Any] = {
        "threshold": threshold,
        "test_samples": int(len(test_idx)),
        "class_distribution": {
            "non_flood": int(np.sum(y_test == 0)),
            "flood": int(np.sum(y_test == 1)),
        },
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, preds)
        ),
        "precision": float(
            precision_score(y_test, preds, zero_division=0)
        ),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "specificity": specificity(matrix),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, preds)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix.tolist(),
        "probability_stats": stats,
        "collapsed_probability_warning": collapsed,
        "compressed_probability_warning": compressed,
        "majority_class_accuracy": float(
            max(np.mean(y_test == 0), np.mean(y_test == 1))
        ),
    }

    RECOMPUTED_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n==============================")
    print("CNN TEST EVALUATION")
    print("==============================")
    print("Threshold         :", threshold)
    print("Accuracy          :", result["accuracy"])
    print("Balanced accuracy :", result["balanced_accuracy"])
    print("Precision         :", result["precision"])
    print("Recall            :", result["recall"])
    print("Specificity       :", result["specificity"])
    print("F1                :", result["f1"])
    print("MCC               :", result["mcc"])
    print("ROC-AUC           :", result["roc_auc"])
    print("PR-AUC            :", result["pr_auc"])
    print("Majority baseline :", result["majority_class_accuracy"])
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Probability stats :", stats)

    if collapsed:
        print(
            "\n[FAILED MODEL] Predictions are effectively constant and "
            "class ranking is poor."
        )
    elif result["balanced_accuracy"] <= 0.50:
        print(
            "\n[FAILED MODEL] Balanced accuracy is not better than random."
        )
    elif compressed:
        print(
            "\n[USABLE PROTOTYPE] The model separates classes, but probabilities "
            "are compressed. Use the saved threshold, not 0.5."
        )
    else:
        print("\n[OK] Model produces class-separating probabilities.")

if __name__ == "__main__":
    main()
