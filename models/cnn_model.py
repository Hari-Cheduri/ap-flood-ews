"""
models/cnn_model.py
--------------------
Three-block CNN for binary flood / no-flood classification of 64×64×3
satellite image patches, plus a detachable 128-dim feature extractor for
later fusion with the LSTM branch in the hybrid model.

Architecture
------------
  Input  (64, 64, 3)
    Conv2D(32,  3×3, relu, same) → BatchNorm → MaxPool(2×2)   → (32, 32, 32)
    Conv2D(64,  3×3, relu, same) → BatchNorm → MaxPool(2×2)   → (16, 16, 64)
    Conv2D(128, 3×3, relu, same) → BatchNorm → MaxPool(2×2)   → ( 8,  8, 128)
    GlobalAveragePooling2D                                      → (128,)
    Dense(256, relu) → Dropout(0.4)
    Dense(128, relu)                                 ← feature_vector
    Dense(1,   sigmoid)                              ← flood probability

Public API
----------
  build_cnn_model(input_shape=(64,64,3)) → keras.Model
  feature_extractor(model)               → 128-dim extractor keras.Model
  train_cnn(epochs=50, batch_size=64)    → (model, extractor, history, metrics)

Outputs written
---------------
  models/best_cnn.h5                        best checkpoint (val_loss)
  models/cnn_final.h5                       end-of-training weights
  models/cnn_feature_extractor.h5           128-dim feature extractor
  outputs/reports/cnn_training.png          4-panel learning curves
  outputs/reports/cnn_confusion.png         dual confusion matrix
  outputs/reports/cnn_metrics.json          full metrics + history
"""

from __future__ import annotations

from typing import Any, Literal, cast

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import tensorflow as tf  # type: ignore[import]
    from tensorflow import keras  # type: ignore[import]
    from tensorflow.keras import layers  # type: ignore[import]
    from tensorflow.keras.callbacks import (  # type: ignore[import]
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[import]
    except ImportError:
        from keras.preprocessing.image import ImageDataGenerator  # type: ignore[import]
except ImportError:
    import keras as keras
    from keras import layers
    from keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from keras.preprocessing.image import ImageDataGenerator
    tf = None
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR   = Path("data/processed")
MODELS_DIR      = Path("models")
REPORTS_DIR     = Path("outputs/reports")

CHECKPOINT_PATH  = MODELS_DIR  / "best_cnn.h5"
FINAL_PATH       = MODELS_DIR  / "cnn_final.h5"
EXTRACTOR_PATH   = MODELS_DIR  / "cnn_feature_extractor.h5"
TRAIN_PLOT_PATH  = REPORTS_DIR / "cnn_training.png"
CM_PLOT_PATH     = REPORTS_DIR / "cnn_confusion.png"
METRICS_PATH     = REPORTS_DIR / "cnn_metrics.json"

# ── Palette (matches lstm_model.py style) ─────────────────────────────────
P = {
    "bg":      "#0d1117", "panel":   "#161b22", "border":  "#30363d",
    "blue":    "#58a6ff", "green":   "#3fb950", "orange":  "#d29922",
    "red":     "#f85149", "purple":  "#bc8cff", "text":    "#e6edf3",
    "subtext": "#8b949e",
}


# ══════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════

def build_cnn_model(
    input_shape:    tuple = (64, 64, 3),
    filters:        tuple = (32, 64, 128),
    dense_units:    tuple = (256, 128),
    dropout_rate:   float = 0.4,
    learning_rate:  float = 0.001,
) -> keras.Model:
    """
    Build and compile the three-block CNN flood classifier.

    Parameters
    ----------
    input_shape   : (H, W, C) of input images
    filters       : conv filters per block — (block1, block2, block3)
    dense_units   : (classification_head_units, feature_vector_units)
    dropout_rate  : spatial dropout after first dense layer
    learning_rate : Adam learning rate

    Returns
    -------
    keras.Model  — compiled, ready for .fit()
    """
    inputs: Any = keras.Input(shape=input_shape, name="satellite_patch")

    # ── Block 1: 64×64 → 32×32 ──────────────────────────────────────
    x = layers.Conv2D(filters[0], 3, padding="same", activation="relu",
                      name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # ── Block 2: 32×32 → 16×16 ──────────────────────────────────────
    x = layers.Conv2D(filters[1], 3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # ── Block 3: 16×16 → 8×8 ────────────────────────────────────────
    x = layers.Conv2D(filters[2], 3, padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)

    # ── Spatial pooling ───────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Classification head ───────────────────────────────────────────
    x = layers.Dense(dense_units[0], activation="relu", name="dense_256")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # 128-dim feature vector (named so feature_extractor can find it)
    features = layers.Dense(dense_units[1], activation="relu",
                             name="feature_vector")(x)

    outputs  = layers.Dense(1, activation="sigmoid",
                             name="flood_prob")(features)

    model = keras.Model(inputs, outputs, name="CNN_FloodClassifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════
#  Feature extractor
# ══════════════════════════════════════════════════════════════════════════

def feature_extractor(model: keras.Model) -> keras.Model:
    """
    Slice off the classification head and return a Model that outputs the
    128-dim 'feature_vector' dense layer — used for LSTM-CNN fusion.

    Parameters
    ----------
    model : trained (or freshly built) CNN_FloodClassifier

    Returns
    -------
    keras.Model  input=(64,64,3), output=(128,)
    """
    feat_layer = model.get_layer("feature_vector")
    extractor  = keras.Model(
        inputs  = model.input,
        outputs = feat_layer.output,
        name    = "CNN_FeatureExtractor",
    )
    return extractor


# ══════════════════════════════════════════════════════════════════════════
#  Label loader  (CNN split is independent from LSTM split)
# ══════════════════════════════════════════════════════════════════════════

def _load_cnn_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the same 70/15/15 stratified split that the preprocessor
    applied to satellite_images.npy, returning (y_train, y_val, y_test)
    aligned to the saved X_cnn_* arrays.
    """
    labels = np.load(PROCESSED_DIR / "image_labels.npy").astype(np.int8)
    idx    = np.arange(len(labels))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_idx, temp_idx = next(sss1.split(idx, labels))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    vl_rel, te_rel  = next(sss2.split(temp_idx, labels[temp_idx]))
    vl_idx, te_idx  = temp_idx[vl_rel], temp_idx[te_rel]

    return labels[tr_idx], labels[vl_idx], labels[te_idx]


# ══════════════════════════════════════════════════════════════════════════
#  Plot helpers  (same dark style as lstm_model.py)
# ══════════════════════════════════════════════════════════════════════════

def _apply_dark(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in axes:
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["subtext"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(P["border"])
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])


def _plot_training_curves(history: dict, best_ep: int, save_path: Path):
    """4-panel dark-themed learning curve: Loss | Accuracy | AUC | Prec+Rec."""
    h   = history
    eps = range(1, len(h["loss"]) + 1)

    fig = plt.figure(figsize=(18, 5), facecolor=P["bg"])
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                            left=0.05, right=0.97, top=0.84, bottom=0.14)
    axes = []

    panels = [
        ("Loss",     "loss",     "val_loss",     P["red"],    P["orange"]),
        ("Accuracy", "accuracy", "val_accuracy", P["blue"],   P["purple"]),
        ("AUC",      "auc",      "val_auc",      P["green"],  P["orange"]),
    ]
    for col, (title, tr_k, vl_k, c_tr, c_vl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(eps, h[tr_k], color=c_tr, lw=2, label="Train")
        ax.plot(eps, h[vl_k], color=c_vl, lw=2, label="Val", linestyle="--")
        ax.axvline(best_ep, color=P["subtext"], lw=1, linestyle=":",
                   label=f"Best ep {best_ep}")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=8, facecolor=P["panel"],
                  labelcolor=P["text"], edgecolor=P["border"])
        axes.append(ax)

    # Panel 4: Precision + Recall
    ax4 = fig.add_subplot(gs[0, 3])
    for key, label, col in [
        ("precision",     "Train Prec", P["blue"]),
        ("val_precision", "Val Prec",   P["blue"]),
        ("recall",        "Train Rec",  P["green"]),
        ("val_recall",    "Val Rec",    P["green"]),
    ]:
        ls = "--" if key.startswith("val") else "-"
        ax4.plot(eps, h[key], color=col, lw=2, label=label, linestyle=ls)
    ax4.axhline(0.80, color=P["orange"], lw=1.2, linestyle=":",
                label="0.80 threshold")
    ax4.axvline(best_ep, color=P["subtext"], lw=1, linestyle=":")
    ax4.set_title("Precision & Recall", fontsize=11, fontweight="bold", pad=8)
    ax4.set_xlabel("Epoch", fontsize=9)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=7.5, facecolor=P["panel"], labelcolor=P["text"],
               edgecolor=P["border"], ncol=2, loc="lower right")
    axes.append(ax4)

    _apply_dark(fig, axes)
    fig.suptitle("CNN Flood Classifier — Training Curves",
                 fontsize=14, fontweight="bold", color=P["text"], y=0.97)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  Training curves   → {save_path}")


def _plot_confusion_matrix(y_true, y_pred, save_path: Path):
    """Dual confusion matrix: raw counts (left) + row-normalised (right)."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ["No-Flood (0)", "Flood (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=P["bg"])
    for ax, data, fmt, title, cmap in zip(
        axes,
        [cm,      cm_norm],
        ["d",     ".2%"],
        ["Counts","Row-Normalised"],
        ["Blues", "RdYlGn"],
    ):
        im = ax.imshow(data, cmap=cmap, vmin=0,
                       vmax=(None if fmt == "d" else 1))
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels, fontsize=10, rotation=90, va="center")
        ax.set_xlabel("Predicted",  fontsize=10, color=P["text"])
        ax.set_ylabel("Actual",     fontsize=10, color=P["text"])
        ax.set_title(f"Confusion Matrix — {title}", fontsize=11,
                     fontweight="bold", color=P["text"], pad=10)
        thresh = data.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = data[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if val < thresh else P["bg"])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=P["subtext"], labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=P["subtext"])
        # Some matplotlib versions/types may not expose an outline with a
        # set_edgecolor method; guard to avoid runtime/type-checker errors.
        outline = getattr(cb, "outline", None)
        if outline is not None:
            set_edgecolor = getattr(outline, "set_edgecolor", None)
            if callable(set_edgecolor):
                try:
                    set_edgecolor(P["border"])
                except Exception:
                    pass
        ax.set_facecolor(P["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(P["border"])
        ax.tick_params(colors=P["subtext"])

    fig.suptitle("CNN Flood Classifier — Test Set Confusion Matrix",
                 fontsize=13, fontweight="bold", color=P["text"], y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  Confusion matrix  → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Training function
# ══════════════════════════════════════════════════════════════════════════

def train_cnn(
    epochs:        int              = 50,
    batch_size:    int              = 64,
    threshold:     float            = 0.50,
    min_recall:    float            = 0.80,
    min_precision: float            = 0.80,
    verbose:       Literal[0, 1, 2] = 1,
) -> tuple[keras.Model, keras.Model, keras.callbacks.History, dict]:
    """
    Full CNN training pipeline: load → augment → train → evaluate → save.

    Parameters
    ----------
    epochs        : maximum epochs (EarlyStopping may stop earlier)
    batch_size    : mini-batch size for both augmented train and val
    threshold     : sigmoid decision threshold
    min_recall    : WARNING if flood recall falls below this on test set
    min_precision : WARNING if flood precision falls below this on test set
    verbose       : Keras fit verbosity (0 silent, 1 bar, 2 epoch-line)

    Returns
    -------
    model     : trained CNN_FloodClassifier  (keras.Model)
    extractor : 128-dim CNN_FeatureExtractor (keras.Model)
    history   : keras History object
    metrics   : dict  with all evaluation metrics and training history
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    SEP  = "═" * 64
    DASH = "─" * 64

    print(f"\n{SEP}")
    print("  CNN Flood Classifier  —  Training Pipeline")
    print(SEP)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1 / 5]  Loading image splits and labels …")
    t0 = time.perf_counter()

    X_train = np.load(PROCESSED_DIR / "X_cnn_train.npy")   # (14000, 64, 64, 3)
    X_val   = np.load(PROCESSED_DIR / "X_cnn_val.npy")     # (3000,  64, 64, 3)
    X_test  = np.load(PROCESSED_DIR / "X_cnn_test.npy")    # (3000,  64, 64, 3)

    # Labels reproduced from the same stratified split used by preprocessor
    y_train, y_val, y_test = _load_cnn_labels()
    y_train = y_train.astype(np.float32)
    y_val   = y_val.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    load_time = time.perf_counter() - t0
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    print(f"  y_train flood={int(y_train.sum())}  "
          f"y_val flood={int(y_val.sum())}  "
          f"y_test flood={int(y_test.sum())}")
    print(f"  Loaded in {load_time:.2f}s")

    # ── 2. Build model ────────────────────────────────────────────────
    print(f"\n[2 / 5]  Building CNN model …")
    model = build_cnn_model()
    model.summary(line_length=67, print_fn=lambda s: print("  " + s))

    # ── 3. Augmentation ───────────────────────────────────────────────
    print(f"\n[3 / 5]  Setting up ImageDataGenerator augmentation …")
    aug = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_gen = ImageDataGenerator()   # no augmentation on val

    train_flow: Any = aug.flow(X_train, y_train, batch_size=batch_size, seed=42)
    val_flow: Any   = val_gen.flow(X_val,   y_val,   batch_size=batch_size, shuffle=False)

    steps_per_epoch  = int(np.ceil(len(X_train) / batch_size))
    val_steps        = int(np.ceil(len(X_val)   / batch_size))
    print(f"  Train batches/epoch: {steps_per_epoch}  "
          f"Val batches/epoch: {val_steps}")
    print(f"  Augmentation: rotation±15° | shift±10% | hflip | zoom±10%")

    # ── Class weights ─────────────────────────────────────────────────
    classes    = np.unique(y_train)
    raw_w      = compute_class_weight("balanced", classes=classes, y=y_train)
    cw         = {int(c): float(w) for c, w in zip(classes, raw_w)}
    print(f"  Class weights: {cw}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=cast(int, 1e-6), verbose=1),
        ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_loss",
                        save_best_only=True, verbose=1),
    ]

    # ── 4. Training ───────────────────────────────────────────────────
    print(f"\n[4 / 5]  Training  (epochs={epochs}, batch_size={batch_size}) …")
    t_train = time.perf_counter()

    history = model.fit(
        train_flow,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_flow,
        validation_steps=val_steps,
        epochs=epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=verbose,
    )

    train_time = time.perf_counter() - t_train
    actual_eps = len(history.history["loss"])
    best_ep    = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\n  Trained {actual_eps} epochs  |  best epoch = {best_ep}  "
          f"|  wall time = {train_time:.1f}s")

    # ── Save final model ──────────────────────────────────────────────
    model.save(str(FINAL_PATH))
    print(f"  ✓  Final model       → {FINAL_PATH}")

    # ── Build & save feature extractor ───────────────────────────────
    extractor = feature_extractor(model)
    extractor.save(str(EXTRACTOR_PATH))
    print(f"  ✓  Feature extractor → {EXTRACTOR_PATH}  "
          f"(output shape: {extractor.output_shape})")

    # ── 5. Evaluation ─────────────────────────────────────────────────
    print(f"\n[5 / 5]  Evaluating on test set …")

    y_prob = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_test.astype(int)

    report_str = cast(
        str,
        classification_report(
            y_true, y_pred, target_names=["No-Flood", "Flood"],
            output_dict=False,
        ),
    )
    report_dict = cast(
        dict[str, Any],
        classification_report(
            y_true, y_pred, target_names=["No-Flood", "Flood"],
            output_dict=True
        ),
    )

    roc_auc   = roc_auc_score(y_true, y_prob)
    pr_auc    = average_precision_score(y_true, y_prob)
    recall    = report_dict["Flood"]["recall"]
    precision = report_dict["Flood"]["precision"]
    f1        = report_dict["Flood"]["f1-score"]
    accuracy  = report_dict["accuracy"]

    print(f"\n  Classification Report (test set):")
    print(f"  {DASH}")
    for line in report_str.strip().splitlines():
        print(f"  {line}")
    print(f"  {DASH}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # ── Precision / Recall target checks ─────────────────────────────
    print(f"\n  Performance targets (flood class):")
    for name, val, target in [
        ("Recall",    recall,    min_recall),
        ("Precision", precision, min_precision),
    ]:
        ok     = val >= target
        symbol = "✓" if ok else "⚠  WARNING"
        note   = "" if ok else f" ← BELOW TARGET ({target:.2f})"
        print(f"    {symbol}  {name:<10} = {val:.4f}{note}")
        if not ok:
            print(f"    ⚠  {name} {val:.4f} < {target:.2f}. Consider more "
                  f"epochs, stronger augmentation, or lower threshold.")

    # ── Feature extractor sanity check ───────────────────────────────
    sample_feats = extractor.predict(X_test[:4], verbose=0)
    print(f"\n  Feature extractor check (first 4 test images):")
    print(f"    output shape : {sample_feats.shape}   "
          f"(expected (4, 128))")
    print(f"    value range  : [{sample_feats.min():.4f}, "
          f"{sample_feats.max():.4f}]  (relu → all ≥ 0)")

    # ── Metrics dict ──────────────────────────────────────────────────
    keras_scores = model.evaluate(X_test, y_test, batch_size=batch_size,
                                  verbose=0)
    if not isinstance(keras_scores, (list, tuple)):
        keras_scores = [keras_scores]
    metrics = {
        "model":          "CNN_FloodClassifier",
        "input_shape":    [64, 64, 3],
        "feature_dim":    128,
        "epochs_trained": actual_eps,
        "best_epoch":     best_ep,
        "train_time_s":   round(train_time, 2),
        "threshold":      threshold,
        "augmentation": {
            "rotation_range": 15,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "horizontal_flip": True,
            "zoom_range": 0.1,
        },
        "test": {
            "accuracy":         round(accuracy,  4),
            "precision_flood":  round(precision, 4),
            "recall_flood":     round(recall,    4),
            "f1_flood":         round(f1,        4),
            "roc_auc":          round(roc_auc,   4),
            "pr_auc":           round(pr_auc,    4),
        },
        "test_keras": {
            n: round(float(v), 4)
            for n, v in zip(model.metrics_names, keras_scores)
        },
        "history": {
            k: [round(float(x), 5) for x in v]
            for k, v in history.history.items()
        },
        "classification_report": report_dict,
        "targets": {
            "min_recall":    min_recall,
            "min_precision": min_precision,
            "recall_met":    bool(recall    >= min_recall),
            "precision_met": bool(precision >= min_precision),
        },
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓  Metrics dict      → {METRICS_PATH}")

    # ── Plots ─────────────────────────────────────────────────────────
    _plot_training_curves(history.history, best_ep, TRAIN_PLOT_PATH)
    _plot_confusion_matrix(y_true, y_pred, CM_PLOT_PATH)

    # ── Summary banner ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(SEP)
    rows = [
        ("Epochs trained",    f"{actual_eps}  (best={best_ep})"),
        ("Wall time",         f"{train_time:.1f}s"),
        ("Accuracy",          f"{accuracy:.4f}"),
        ("Precision (flood)", f"{precision:.4f}  {'✓' if precision >= min_precision else '⚠'}"),
        ("Recall (flood)",    f"{recall:.4f}  {'✓' if recall >= min_recall else '⚠'}"),
        ("F1 (flood)",        f"{f1:.4f}"),
        ("ROC-AUC",           f"{roc_auc:.4f}"),
        ("PR-AUC",            f"{pr_auc:.4f}"),
        ("Feature extractor", f"(4, 128)  shape  sample_feats verified"),
    ]
    for label, val in rows:
        print(f"  {label:<24}  {val}")
    print(SEP)
    artefacts = [
        ("Best checkpoint",    CHECKPOINT_PATH),
        ("Final model",        FINAL_PATH),
        ("Feature extractor",  EXTRACTOR_PATH),
        ("Training curves",    TRAIN_PLOT_PATH),
        ("Confusion matrix",   CM_PLOT_PATH),
        ("Metrics JSON",       METRICS_PATH),
    ]
    print("\n  Artefacts saved:")
    for name, path in artefacts:
        sz_kb = Path(path).stat().st_size / 1024
        print(f"    {name:<22}  {path}  ({sz_kb:.1f} KB)")
    print(f"{SEP}\n")

    return model, extractor, history, metrics


# ══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if tf is not None:
        tf.random.set_seed(42)
    np.random.seed(42)
    model, extractor, history, metrics = train_cnn(epochs=50, batch_size=64)
