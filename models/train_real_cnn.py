"""
models/train_real_cnn.py
------------------------
Train a compact CNN on the existing real Sentinel-1 thumbnail dataset.

Key safeguards:
- validates the dataset and duplicate labels;
- uses district-grouped splitting when satellite_metadata.csv is available;
- otherwise uses a deterministic stratified split;
- balances only the training batches;
- applies augmentation only to training data;
- trains multiple random restarts;
- chooses the best validation model;
- detects collapsed near-constant predictions;
- saves split indices so evaluation uses the exact same test set.

Run:
    python -m models.train_real_cnn
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils.sentinel_cnn_contract import (
    CONTRACT_NAME,
    HISTORY_PATH,
    INPUT_SHAPE,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_H5_PATH,
    MODEL_KERAS_PATH,
    SPLIT_PATH,
    THRESHOLD_PATH,
    X_PATH,
    Y_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {X_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Missing labels: {Y_PATH}")

    X = np.load(X_PATH, allow_pickle=False).astype(np.float32)
    y = np.load(Y_PATH, allow_pickle=False).astype(np.int32).reshape(-1)

    if X.ndim != 4 or tuple(X.shape[1:]) != INPUT_SHAPE:
        raise ValueError(
            f"Expected X shape (N, {INPUT_SHAPE}); received {X.shape}"
        )
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} versus {len(y)}")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(
            f"Labels must contain exactly 0 and 1; received {np.unique(y)}"
        )
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values")
    if float(X.min()) < -1e-6 or float(X.max()) > 1.0 + 1e-6:
        raise ValueError(
            f"Expected normalized [0,1] data; range is "
            f"{X.min()} to {X.max()}"
        )

    return np.clip(X, 0.0, 1.0), y


def inspect_duplicates(X: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    hash_to_labels: Dict[str, set[int]] = {}
    hash_counts: Dict[str, int] = {}

    for image, label in zip(X, y):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        hash_to_labels.setdefault(digest, set()).add(int(label))
        hash_counts[digest] = hash_counts.get(digest, 0) + 1

    conflicting = sum(1 for labels in hash_to_labels.values() if len(labels) > 1)
    duplicates = sum(count - 1 for count in hash_counts.values() if count > 1)

    if conflicting:
        raise ValueError(
            f"Found {conflicting} exact image(s) assigned to both labels. "
            "Fix the dataset before training."
        )

    return {
        "exact_duplicate_rows": int(duplicates),
        "conflicting_duplicate_images": int(conflicting),
    }


def load_groups(sample_count: int) -> Optional[np.ndarray]:
    if not METADATA_PATH.exists():
        return None

    with METADATA_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != sample_count:
        print(
            f"[Split] Metadata has {len(rows)} rows but dataset has "
            f"{sample_count}; using stratified split instead."
        )
        return None

    if not rows or "district" not in rows[0]:
        return None

    groups = np.asarray(
        [row.get("district", "").strip() for row in rows],
        dtype=str,
    )
    if np.any(groups == "") or len(np.unique(groups)) < 4:
        return None

    return groups


def has_both_classes(y: np.ndarray, indices: np.ndarray) -> bool:
    return len(np.unique(y[indices])) == 2


def grouped_split(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    all_indices = np.arange(len(y))

    for attempt in range(500):
        first = GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=seed + attempt,
        )
        train_idx, temp_idx = next(
            first.split(all_indices, y, groups=groups)
        )

        if not has_both_classes(y, train_idx):
            continue
        if len(np.unique(groups[temp_idx])) < 2:
            continue

        second = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=seed + 10_000 + attempt,
        )
        temp_relative = np.arange(len(temp_idx))
        val_rel, test_rel = next(
            second.split(
                temp_relative,
                y[temp_idx],
                groups=groups[temp_idx],
            )
        )
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        if (
            has_both_classes(y, val_idx)
            and has_both_classes(y, test_idx)
        ):
            return (
                np.sort(train_idx),
                np.sort(val_idx),
                np.sort(test_idx),
            )

    return None


def stratified_split(
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=y[temp_idx],
    )
    return (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
    )


def create_split(
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[np.ndarray]]:
    groups = load_groups(len(y))

    if groups is not None:
        result = grouped_split(y, groups, seed)
        if result is not None:
            return (*result, "district_grouped", groups)

        print(
            "[Split] Could not create grouped splits containing both classes; "
            "falling back to deterministic stratified split."
        )

    train_idx, val_idx, test_idx = stratified_split(y, seed)
    return train_idx, val_idx, test_idx, "stratified", groups


def make_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> Tuple[tf.data.Dataset, int]:
    X_zero = X[y == 0]
    y_zero = y[y == 0]
    X_one = X[y == 1]
    y_one = y[y == 1]

    if len(X_zero) == 0 or len(X_one) == 0:
        raise ValueError("Training split must contain both classes")

    ds_zero = (
        tf.data.Dataset.from_tensor_slices((X_zero, y_zero))
        .shuffle(len(X_zero), seed=seed, reshuffle_each_iteration=True)
        .repeat()
    )
    ds_one = (
        tf.data.Dataset.from_tensor_slices((X_one, y_one))
        .shuffle(len(X_one), seed=seed + 1, reshuffle_each_iteration=True)
        .repeat()
    )

    balanced = tf.data.Dataset.sample_from_datasets(
        [ds_zero, ds_one],
        weights=[0.5, 0.5],
        seed=seed,
        stop_on_empty_dataset=False,
    )

    balanced = (
        balanced
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    # One epoch exposes roughly twice the majority-class count.
    steps = max(
        1,
        int(math.ceil((2 * max(len(X_zero), len(X_one))) / batch_size)),
    )
    return balanced, steps


def build_model(seed: int) -> tf.keras.Model:
    set_seed(seed)

    regularizer = tf.keras.regularizers.l2(1e-4)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal_and_vertical",
                seed=seed,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
                seed=seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08),
                fill_mode="reflect",
                seed=seed + 2,
            ),
            tf.keras.layers.RandomContrast(
                factor=0.10,
                seed=seed + 3,
            ),
        ],
        name="train_only_augmentation",
    )

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="sentinel_rgb")
    x = augmentation(inputs)
    x = tf.keras.layers.GaussianNoise(0.015)(x)

    x = tf.keras.layers.Conv2D(
        24,
        5,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.SpatialDropout2D(0.10)(x)

    x = tf.keras.layers.SeparableConv2D(
        48,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.SpatialDropout2D(0.15)(x)

    x = tf.keras.layers.SeparableConv2D(
        96,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.SpatialDropout2D(0.20)(x)

    x = tf.keras.layers.SeparableConv2D(
        128,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        48,
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="flood_probability",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="sentinel1_flood_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def specificity_from_confusion(matrix: np.ndarray) -> float:
    tn, fp, fn, tp = matrix.ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def probability_stats(probs: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "p05": float(np.percentile(probs, 5)),
        "p95": float(np.percentile(probs, 95)),
        "spread_p95_p05": float(
            np.percentile(probs, 95) - np.percentile(probs, 5)
        ),
    }


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, float(np.mean(y_true))
    return (
        float(roc_auc_score(y_true, probs)),
        float(average_precision_score(y_true, probs)),
    )


def threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(np.int32)
    matrix = confusion_matrix(y_true, preds, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, preds)
        ),
        "precision": float(
            precision_score(y_true, preds, zero_division=0)
        ),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "specificity": specificity_from_confusion(matrix),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "confusion_matrix": matrix.tolist(),
        "predicted_non_flood": int(np.sum(preds == 0)),
        "predicted_flood": int(np.sum(preds == 1)),
    }


def choose_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_score = -1.0
    best_result: Dict[str, Any] = {}

    for threshold in np.linspace(0.05, 0.95, 181):
        result = threshold_metrics(y_true, probs, float(threshold))

        # This score prevents the all-positive solution from winning simply
        # because the minority class has a superficially high recall.
        score = (
            0.65 * result["balanced_accuracy"]
            + 0.35 * result["f1"]
        )

        # Prefer thresholds that predict at least one example of each class.
        both_predicted = (
            result["predicted_non_flood"] > 0
            and result["predicted_flood"] > 0
        )
        if not both_predicted:
            score -= 0.20

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_result = result

    best_result["selection_score"] = best_score
    best_result["selection_method"] = (
        "0.65*balanced_accuracy + 0.35*f1 with one-class penalty"
    )
    return best_threshold, best_result


def evaluate_probabilities(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    result = threshold_metrics(y_true, probs, threshold)
    roc_auc, pr_auc = safe_auc(y_true, probs)
    stats = probability_stats(probs)

    compressed = bool(
        stats["std"] < 0.02
        or stats["spread_p95_p05"] < 0.05
    )

    collapsed = bool(
        stats["spread_p95_p05"] < 0.005
        and roc_auc < 0.60
    )

    result.update(
        {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "probability_stats": stats,
            "compressed_probability_warning": compressed,
            "compressed_probability_warning": compressed,
            "collapsed_probability_warning": collapsed,
        }
    )
    return result


def serializable_history(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
    }


def main() -> None:
    args = parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2")

    X, y = load_dataset()
    duplicate_report = inspect_duplicates(X, y)

    print("\n[CNN] Dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Range  :", float(X.min()), "to", float(X.max()))
    print("Classes:", class_distribution(y))
    print("Duplicates:", duplicate_report)

    print("\n[CNN] Per-class channel means")
    for label in (0, 1):
        means = X[y == label].mean(axis=(0, 1, 2))
        stds = X[y == label].std(axis=(0, 1, 2))
        print(
            f"class {label}: means={np.round(means, 4)} "
            f"stds={np.round(stds, 4)}"
        )

    train_idx, val_idx, test_idx, split_strategy, groups = create_split(
        y=y,
        seed=args.seed,
    )

    MODEL_KERAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SPLIT_PATH,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_strategy=np.asarray(split_strategy),
        seed=np.asarray(args.seed),
    )

    print(f"\n[CNN] Split strategy: {split_strategy}")
    for name, indices in (
        ("Train", train_idx),
        ("Val", val_idx),
        ("Test", test_idx),
    ):
        print(
            f"{name:5s}: {len(indices)} samples "
            f"{class_distribution(y[indices])}"
        )
        if groups is not None and split_strategy == "district_grouped":
            print(
                f"       districts={sorted(np.unique(groups[indices]).tolist())}"
            )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds, steps_per_epoch = make_balanced_dataset(
        X_train,
        y_train,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    checkpoint_dir = Path("models/cnn_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trial_records: List[Dict[str, Any]] = []
    best_checkpoint: Optional[Path] = None
    best_key: Optional[Tuple[int, float, float]] = None
    best_trial_record: Optional[Dict[str, Any]] = None

    for trial in range(args.trials):
        trial_seed = args.seed + trial * 100
        set_seed(trial_seed)
        tf.keras.backend.clear_session()

        checkpoint = checkpoint_dir / f"trial_{trial + 1}.h5"
        model = build_model(trial_seed)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=12,
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
                verbose=0,
            ),
        ]

        print(
            f"\n[CNN] Trial {trial + 1}/{args.trials}, seed={trial_seed}"
        )
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        if checkpoint.exists():
            model = tf.keras.models.load_model(str(checkpoint))

        val_probs = model.predict(X_val, verbose=0).reshape(-1)
        threshold, threshold_result = choose_threshold(y_val, val_probs)
        val_result = evaluate_probabilities(
            y_val,
            val_probs,
            threshold,
        )
        val_result["threshold_selection"] = threshold_result

        record = {
            "trial": trial + 1,
            "seed": trial_seed,
            "epochs_completed": len(history.history.get("loss", [])),
            "validation": val_result,
            "history": serializable_history(history.history),
        }
        trial_records.append(record)

        collapsed = val_result["collapsed_probability_warning"]
        selection_key = (
            0 if collapsed else 1,
            float(val_result["pr_auc"]),
            float(val_result["balanced_accuracy"]),
        )

        print(
            "[CNN] Validation:",
            json.dumps(
                {
                    "threshold": threshold,
                    "balanced_accuracy": val_result["balanced_accuracy"],
                    "f1": val_result["f1"],
                    "roc_auc": val_result["roc_auc"],
                    "pr_auc": val_result["pr_auc"],
                    "probability_std": val_result["probability_stats"]["std"],
                    "collapsed": collapsed,
                },
                indent=2,
            ),
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_checkpoint = checkpoint
            best_trial_record = record

    if best_checkpoint is None or best_trial_record is None:
        raise RuntimeError("No CNN trial completed")

    best_model = tf.keras.models.load_model(best_checkpoint)

    selected_threshold = float(
        best_trial_record["validation"]["threshold"]
    )

    test_probs = best_model.predict(X_test, verbose=0).reshape(-1)
    test_result = evaluate_probabilities(
        y_test,
        test_probs,
        selected_threshold,
    )

    majority_accuracy = float(
        max(np.mean(y_test == 0), np.mean(y_test == 1))
    )

    status = "usable_prototype"
    warnings: List[str] = []

    if best_trial_record["validation"]["collapsed_probability_warning"]:
        status = "collapsed_model"
        warnings.append(
            "Validation probabilities are nearly constant. Do not use this "
            "model for live decisions."
        )
    if test_result["collapsed_probability_warning"]:
        status = "collapsed_model"
        warnings.append(
            "Test probabilities are nearly constant. Do not use this model "
            "for live decisions."
        )
    if test_result["balanced_accuracy"] <= 0.50:
        status = "failed_model"
        warnings.append(
            "Test balanced accuracy is not better than random guessing."
        )
    if test_result["accuracy"] <= majority_accuracy:
        warnings.append(
            "Test accuracy does not beat the majority-class baseline."
        )
    best_model.save(str(MODEL_H5_PATH), save_format="h5")
    threshold_payload = {
        "cnn_threshold": selected_threshold,
        "contract": CONTRACT_NAME,
        "selected_trial": int(best_trial_record["trial"]),
        "selection_method": best_trial_record["validation"][
            "threshold_selection"
        ]["selection_method"],
        "validation_metrics": {
            key: value
            for key, value in best_trial_record["validation"].items()
            if key != "threshold_selection"
        },
    }
    THRESHOLD_PATH.write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )

    metrics_payload = {
        "status": status,
        "warnings": warnings,
        "contract": CONTRACT_NAME,
        "split_strategy": split_strategy,
        "seed": args.seed,
        "dataset": {
            "samples": int(len(X)),
            "shape": list(X.shape),
            "class_distribution": class_distribution(y),
            "duplicates": duplicate_report,
            "label_provenance": (
                "Weak labels generated from Sentinel-1 water fraction, "
                "historical rainfall, and monsoon season. These are flood-risk "
                "proxy labels, not manually verified flood polygons."
            ),
        },
        "split": {
            "train_samples": int(len(train_idx)),
            "validation_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "train_distribution": class_distribution(y_train),
            "validation_distribution": class_distribution(y_val),
            "test_distribution": class_distribution(y_test),
        },
        "selected_trial": int(best_trial_record["trial"]),
        "threshold": selected_threshold,
        "validation": best_trial_record["validation"],
        "test": test_result,
        "majority_class_test_accuracy": majority_accuracy,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    HISTORY_PATH.write_text(
        json.dumps({"trials": trial_records}, indent=2),
        encoding="utf-8",
    )

    print("\n========================================")
    print("FINAL CNN RESULT")
    print("========================================")
    print("Status            :", status)
    print("Selected trial    :", best_trial_record["trial"])
    print("Threshold         :", selected_threshold)
    print("Test accuracy     :", test_result["accuracy"])
    print("Balanced accuracy :", test_result["balanced_accuracy"])
    print("Precision         :", test_result["precision"])
    print("Recall            :", test_result["recall"])
    print("Specificity       :", test_result["specificity"])
    print("F1                :", test_result["f1"])
    print("ROC-AUC           :", test_result["roc_auc"])
    print("PR-AUC            :", test_result["pr_auc"])
    print("MCC               :", test_result["mcc"])
    print("Confusion matrix  :")
    print(np.asarray(test_result["confusion_matrix"]))
    print("Probability stats :", test_result["probability_stats"])
    print("Majority baseline :", majority_accuracy)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print("-", warning)

    print("\nSaved:")
    print(MODEL_KERAS_PATH)
    print(MODEL_H5_PATH)
    print(THRESHOLD_PATH)
    print(METRICS_PATH)
    print(SPLIT_PATH)
    print(HISTORY_PATH)


if __name__ == "__main__":
    main()


