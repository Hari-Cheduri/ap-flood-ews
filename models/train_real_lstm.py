from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from utils.lstm_weather_contract import (
    CONTRACT_NAME, FEATURES, HISTORY_PATH, INPUT_SHAPE, METRICS_PATH,
    MODEL_PATH, THRESHOLD_PATH, X_TEST, X_TRAIN, X_VAL,
    Y_TEST, Y_TRAIN, Y_VAL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_arrays() -> Tuple[np.ndarray, ...]:
    paths = (X_TRAIN, X_VAL, X_TEST, Y_TRAIN, Y_VAL, Y_TEST)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LSTM arrays:\n" + "\n".join(missing))

    arrays = [
        np.load(path, allow_pickle=False)
        for path in paths
    ]
    X_train, X_val, X_test = [
        arr.astype(np.float32) for arr in arrays[:3]
    ]
    y_train, y_val, y_test = [
        arr.astype(np.int32).reshape(-1) for arr in arrays[3:]
    ]

    for name, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        if tuple(X.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"{name} shape is {X.shape}, expected (N,{INPUT_SHAPE})")
        if len(X) != len(y):
            raise ValueError(f"{name} X/y mismatch")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinity")

    return X_train, X_val, X_test, y_train, y_val, y_test


def distribution(y: np.ndarray) -> Dict[str, int]:
    return {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}


def build_model(seed: int) -> tf.keras.Model:
    seed_everything(seed)
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="weather_sequence")
    x = tf.keras.layers.GaussianNoise(0.015)(inputs)
    x = tf.keras.layers.LSTM(
        64, return_sequences=True, dropout=0.15, name="lstm_1"
    )(x)
    x = tf.keras.layers.LSTM(
        32, return_sequences=False, dropout=0.15, name="lstm_2"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="weather_risk"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="RealWeatherLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def threshold_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predictions)),
        "confusion_matrix": matrix.tolist(),
        "predicted_0": int(np.sum(predictions == 0)),
        "predicted_1": int(np.sum(predictions == 1)),
    }


def select_threshold(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_score = -1.0
    best_result: Dict[str, Any] = {}

    for threshold in np.linspace(0.05, 0.95, 181):
        result = threshold_metrics(y, probabilities, float(threshold))
        score = (
            0.55 * result["balanced_accuracy"]
            + 0.25 * result["f1"]
            + 0.20 * result["recall"]
        )
        if result["predicted_0"] == 0 or result["predicted_1"] == 0:
            score -= 0.25
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_result = result

    best_result["selection_score"] = float(best_score)
    return best_threshold, best_result


def evaluate_probabilities(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    result = threshold_metrics(y, probabilities, threshold)
    roc_auc = float(roc_auc_score(y, probabilities))
    pr_auc = float(average_precision_score(y, probabilities))
    p05 = float(np.percentile(probabilities, 5))
    p95 = float(np.percentile(probabilities, 95))
    spread = p95 - p05

    result.update({
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "probability_stats": {
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "std": float(probabilities.std()),
            "p05": p05,
            "p95": p95,
            "spread_p95_p05": spread,
        },
        "collapsed_probability_warning": bool(
            spread < 0.005 and roc_auc < 0.60
        ),
    })
    return result


def main() -> None:
    args = parse_args()
    X_train, X_val, X_test, y_train, y_val, y_test = load_arrays()

    print("\n[LSTM] Dataset")
    print("Train:", X_train.shape, distribution(y_train))
    print("Val  :", X_val.shape, distribution(y_val))
    print("Test :", X_test.shape, distribution(y_test))
    print("Features:", FEATURES)

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight = {
        int(cls): float(weight)
        for cls, weight in zip(classes, weights)
    }
    print("Class weights:", class_weight)

    checkpoint_dir = Path("models/lstm_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trial_records: List[Dict[str, Any]] = []
    selected_model: Optional[tf.keras.Model] = None
    selected_record: Optional[Dict[str, Any]] = None
    selected_key: Optional[Tuple[int, float, float]] = None

    for trial in range(args.trials):
        trial_seed = args.seed + trial * 100
        seed_everything(trial_seed)
        tf.keras.backend.clear_session()

        checkpoint = checkpoint_dir / f"trial_{trial + 1}.h5"
        model = build_model(trial_seed)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=10,
                restore_best_weights=True, min_delta=1e-4, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", factor=0.5,
                patience=4, min_lr=1e-6, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint), monitor="val_pr_auc",
                mode="max", save_best_only=True, verbose=0,
            ),
        ]

        print(f"\n[LSTM] Trial {trial + 1}/{args.trials}, seed={trial_seed}")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

        if checkpoint.exists():
            model = tf.keras.models.load_model(str(checkpoint))

        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        threshold, threshold_selection = select_threshold(y_val, val_prob)
        validation = evaluate_probabilities(y_val, val_prob, threshold)
        validation["threshold_selection"] = threshold_selection

        record = {
            "trial": trial + 1,
            "seed": trial_seed,
            "epochs": len(history.history.get("loss", [])),
            "validation": validation,
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        }
        trial_records.append(record)

        key = (
            0 if validation["collapsed_probability_warning"] else 1,
            validation["pr_auc"],
            validation["balanced_accuracy"],
        )
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_model = model
            selected_record = record

        print(json.dumps({
            "threshold": threshold,
            "balanced_accuracy": validation["balanced_accuracy"],
            "recall": validation["recall"],
            "f1": validation["f1"],
            "roc_auc": validation["roc_auc"],
            "pr_auc": validation["pr_auc"],
            "collapsed": validation["collapsed_probability_warning"],
        }, indent=2))

    if selected_model is None or selected_record is None:
        raise RuntimeError("No LSTM trial completed")

    threshold = float(selected_record["validation"]["threshold"])
    test_prob = selected_model.predict(X_test, verbose=0).reshape(-1)
    test_result = evaluate_probabilities(y_test, test_prob, threshold)

    status = "usable_prototype"
    warnings: List[str] = []
    if selected_record["validation"]["collapsed_probability_warning"]:
        status = "collapsed_model"
        warnings.append("Validation probabilities are collapsed.")
    if test_result["collapsed_probability_warning"]:
        status = "collapsed_model"
        warnings.append("Test probabilities are collapsed.")
    if test_result["balanced_accuracy"] <= 0.50:
        status = "failed_model"
        warnings.append("Balanced accuracy is not better than random.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_model.save(str(MODEL_PATH), save_format="h5")

    THRESHOLD_PATH.write_text(json.dumps({
        "lstm_threshold": threshold,
        "contract": CONTRACT_NAME,
        "selected_trial": selected_record["trial"],
        "validation_metrics": selected_record["validation"],
    }, indent=2), encoding="utf-8")

    METRICS_PATH.write_text(json.dumps({
        "status": status,
        "warnings": warnings,
        "contract": CONTRACT_NAME,
        "features": list(FEATURES),
        "input_shape": list(INPUT_SHAPE),
        "selected_trial": selected_record["trial"],
        "threshold": threshold,
        "dataset": {
            "train": distribution(y_train),
            "val": distribution(y_val),
            "test": distribution(y_test),
        },
        "validation": selected_record["validation"],
        "test": test_result,
        "label_limitation": (
            "Target is a weather-risk proxy derived from future rainfall, "
            "not verified physical flood inundation."
        ),
    }, indent=2), encoding="utf-8")

    HISTORY_PATH.write_text(
        json.dumps({"trials": trial_records}, indent=2),
        encoding="utf-8",
    )

    print("\n========================================")
    print("FINAL LSTM RESULT")
    print("========================================")
    print("Status            :", status)
    print("Selected trial    :", selected_record["trial"])
    print("Threshold         :", threshold)
    for key in (
        "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc", "pr_auc", "mcc",
    ):
        print(f"{key:18s}:", test_result[key])
    print("Confusion matrix:")
    print(np.asarray(test_result["confusion_matrix"]))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print("-", warning)


if __name__ == "__main__":
    main()
