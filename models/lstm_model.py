"""
models/lstm_model.py
---------------------
Stacked LSTM classifier for binary flood / no-flood prediction.

Architecture
------------
  Input  (24, 15)                    ← 24 timesteps, 15 sensor features
    LSTM(128, return_sequences=True, dropout=0.2)
    LSTM(64,  return_sequences=False, dropout=0.2)
    BatchNormalization()
    Dense(32, relu)
    Dropout(0.3)
    Dense(1,  sigmoid)               ← flood probability [0, 1]

Public API
----------
  build_lstm_model(input_shape=(24,15)) → keras.Model
  train_lstm(epochs=50, batch_size=64) → (model, history, metrics_dict)

Outputs written
---------------
  models/best_lstm.h5                    ← best checkpoint (val_loss)
  models/lstm_final.h5                   ← weights after full training
  outputs/reports/lstm_training.png      ← loss / metric learning curves
  outputs/reports/lstm_confusion.png     ← normalised confusion matrix
  outputs/reports/lstm_metrics.json      ← full metrics dict
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Tuple, cast

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress TF C++ info
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import tensorflow as tf  # type: ignore[import]
from tensorflow import keras  # type: ignore[import]
from tensorflow.keras import layers  # type: ignore[import]
from tensorflow.keras.callbacks import (  # type: ignore[import]
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
REPORTS_DIR   = Path("outputs/reports")

CHECKPOINT_PATH = MODELS_DIR  / "best_lstm.h5"
FINAL_PATH      = MODELS_DIR  / "lstm_final.h5"
TRAIN_PLOT_PATH = REPORTS_DIR / "lstm_training.png"
CM_PLOT_PATH    = REPORTS_DIR / "lstm_confusion.png"
METRICS_PATH    = REPORTS_DIR / "lstm_metrics.json"

# ── Style ─────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "border":   "#30363d",
    "blue":     "#58a6ff",
    "green":    "#3fb950",
    "orange":   "#d29922",
    "red":      "#f85149",
    "purple":   "#bc8cff",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}


# ══════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════

def build_lstm_model(
    input_shape: tuple = (24, 15),
    lstm_units:  tuple = (128, 64),
    dense_units: int   = 32,
    dropout_lstm: float = 0.2,
    dropout_dense: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build and compile the stacked LSTM binary classifier.

    Parameters
    ----------
    input_shape   : (timesteps, features)
    lstm_units    : number of units in the two LSTM layers
    dense_units   : units in the dense hidden layer
    dropout_lstm  : recurrent dropout applied inside LSTM
    dropout_dense : spatial dropout after dense layer
    learning_rate : Adam learning rate

    Returns
    -------
    keras.Model  — compiled, ready for .fit()
    """
    inputs = keras.Input(shape=input_shape, name="sensor_sequence")

    # ── Temporal encoding ────────────────────────────────────────────
    x = layers.LSTM(
        lstm_units[0],
        return_sequences=True,
        dropout=dropout_lstm,
        name="lstm_1",
    )(inputs)

    x = layers.LSTM(
        lstm_units[1],
        return_sequences=False,
        dropout=dropout_lstm,
        name="lstm_2",
    )(x)

    # ── Feature refinement ───────────────────────────────────────────
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_1")(x)

    # ── Output ───────────────────────────────────────────────────────
    outputs = layers.Dense(1, activation="sigmoid", name="flood_prob")(x)

    model = keras.Model(inputs, outputs, name="LSTM_FloodClassifier")

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
#  Plot helpers
# ══════════════════════════════════════════════════════════════════════════

def _apply_dark_style(fig, axes_flat):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes_flat:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])


def _plot_training_curves(history, save_path: Path):
    """
    4-panel learning curve plot:
      [Loss]  [Accuracy]  [Precision / Recall]  [AUC]
    """
    h   = history.history
    eps = range(1, len(h["loss"]) + 1)
    best_ep = int(np.argmin(h["val_loss"])) + 1

    fig = plt.figure(figsize=(18, 5), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                            left=0.05, right=0.97, top=0.84, bottom=0.14)

    panels = [
        ("Loss",           "loss",     "val_loss",
         PALETTE["red"],   PALETTE["orange"]),
        ("Accuracy",       "accuracy", "val_accuracy",
         PALETTE["blue"],  PALETTE["purple"]),
        ("AUC",            "auc",      "val_auc",
         PALETTE["green"], PALETTE["orange"]),
    ]

    axes = []
    for col, (title, tr_key, vl_key, c_tr, c_vl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(eps, h[tr_key], color=c_tr,    lw=2, label="Train")
        ax.plot(eps, h[vl_key], color=c_vl,    lw=2, label="Val",   linestyle="--")
        ax.axvline(best_ep, color=PALETTE["subtext"], lw=1,
                   linestyle=":", label=f"Best ep {best_ep}")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=8, facecolor=PALETTE["panel"],
                  labelcolor=PALETTE["text"], edgecolor=PALETTE["border"])
        axes.append(ax)

    # Panel 4: Precision + Recall together
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(eps, h["precision"],     color=PALETTE["blue"],   lw=2, label="Train Prec")
    ax4.plot(eps, h["val_precision"], color=PALETTE["blue"],   lw=2, label="Val Prec",  linestyle="--")
    ax4.plot(eps, h["recall"],        color=PALETTE["green"],  lw=2, label="Train Rec")
    ax4.plot(eps, h["val_recall"],    color=PALETTE["green"],  lw=2, label="Val Rec",   linestyle="--")
    ax4.axhline(0.80, color=PALETTE["orange"], lw=1, linestyle=":",
                label="0.80 threshold")
    ax4.axvline(best_ep, color=PALETTE["subtext"], lw=1, linestyle=":")
    ax4.set_title("Precision & Recall", fontsize=11, fontweight="bold", pad=8)
    ax4.set_xlabel("Epoch", fontsize=9)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=7.5, facecolor=PALETTE["panel"],
               labelcolor=PALETTE["text"], edgecolor=PALETTE["border"],
               ncol=2, loc="lower right")
    axes.append(ax4)

    _apply_dark_style(fig, axes)

    fig.suptitle(
        "LSTM Flood Classifier — Training Curves",
        fontsize=14, fontweight="bold", color=PALETTE["text"], y=0.97,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓  Training curves  → {save_path}")


def _plot_confusion_matrix(y_true, y_pred_bin, save_path: Path):
    """
    Dual confusion matrix: raw counts (left) + row-normalised rates (right).
    """
    cm      = confusion_matrix(y_true, y_pred_bin)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ["No-Flood (0)", "Flood (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])

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
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10,
                                                    rotation=90, va="center")
        ax.set_xlabel("Predicted",  fontsize=10, color=PALETTE["text"])
        ax.set_ylabel("Actual",     fontsize=10, color=PALETTE["text"])
        ax.set_title(f"Confusion Matrix — {title}",
                     fontsize=11, fontweight="bold",
                     color=PALETTE["text"], pad=10)

        # Annotate cells
        thresh = data.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = data[i, j]
                txt = f"{val:{fmt}}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if val < thresh else PALETTE["bg"])

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=PALETTE["subtext"], labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"])
        cb.ax.spines["outline"].set_edgecolor(PALETTE["border"])

        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.tick_params(colors=PALETTE["subtext"])

    fig.suptitle("LSTM Flood Classifier — Test Set Confusion Matrix",
                 fontsize=13, fontweight="bold",
                 color=PALETTE["text"], y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓  Confusion matrix → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Training function
# ══════════════════════════════════════════════════════════════════════════

def train_lstm(
    epochs:         int   = 50,
    batch_size:     int   = 64,
    threshold:      float = 0.50,
    min_recall:     float = 0.80,
    min_precision:  float = 0.80,
    verbose:        int   = 1,
) -> Tuple["keras.Model", Any, dict]:
    """
    Load preprocessed LSTM splits, build the model, train, evaluate, and
    save all artefacts.

    Parameters
    ----------
    epochs        : maximum training epochs (EarlyStopping may stop earlier)
    batch_size    : mini-batch size
    threshold     : classification threshold applied to sigmoid output
    min_recall    : assert target — prints WARNING if not met on test set
    min_precision : assert target — prints WARNING if not met on test set
    verbose       : Keras fit verbosity (0=silent, 1=progress, 2=epoch only)

    Returns
    -------
    model   : trained keras.Model
    history : keras History object
    metrics : dict with all evaluation metrics
    """

    # ── Setup ─────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _sep  = "═" * 62
    _dash = "─" * 62

    print(f"\n{_sep}")
    print("  LSTM Flood Classifier  —  Training Pipeline")
    print(_sep)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1 / 5]  Loading preprocessed splits …")
    t0 = time.perf_counter()

    X_train = np.load(PROCESSED_DIR / "X_lstm_train.npy")
    X_val   = np.load(PROCESSED_DIR / "X_lstm_val.npy")
    X_test  = np.load(PROCESSED_DIR / "X_lstm_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    y_val   = np.load(PROCESSED_DIR / "y_val.npy").astype(np.float32)
    y_test  = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)

    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    print(f"  Loaded in {time.perf_counter()-t0:.2f}s")

    # ── Class weights ─────────────────────────────────────────────────
    classes     = np.unique(y_train)
    raw_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw          = {int(c): float(w) for c, w in zip(classes, raw_weights)}
    print(f"  Class weights: {cw}")

    # ── Build model ───────────────────────────────────────────────────
    print(f"\n[2 / 5]  Building model …")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary(line_length=65, print_fn=lambda s: print("  " + s))

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,  # type: ignore[arg-type]
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────
    print(f"\n[3 / 5]  Training  (epochs={epochs}, batch_size={batch_size}) …")
    t_train = time.perf_counter()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=callbacks,
        verbose=verbose,
    )

    train_time = time.perf_counter() - t_train
    actual_eps = len(history.history["loss"])
    best_ep    = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\n  Trained {actual_eps} epochs  |  best epoch = {best_ep}  "
          f"|  wall time = {train_time:.1f}s")

    # ── Save final weights ────────────────────────────────────────────
    model.save(str(FINAL_PATH))
    print(f"  ✓  Final model     → {FINAL_PATH}")

    # ── Evaluation ────────────────────────────────────────────────────
    print(f"\n[4 / 5]  Evaluating on test set …")

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # sklearn metrics
    report_dict = cast(
        dict[str, Any],
        classification_report(
            y_test.astype(int), y_pred,
            target_names=["No-Flood", "Flood"],
            output_dict=True,
        ),
    )
    # force string type to satisfy type checkers that may infer dict
    report_str = str(classification_report(
        y_test.astype(int), y_pred,
        target_names=["No-Flood", "Flood"],
    ))

    roc_auc   = roc_auc_score(y_test, y_prob)
    pr_auc    = average_precision_score(y_test, y_prob)
    recall    = float(report_dict["Flood"]["recall"])
    precision = float(report_dict["Flood"]["precision"])
    f1        = float(report_dict["Flood"]["f1-score"])
    accuracy  = float(report_dict["accuracy"])

    print("\n  Classification Report (test set):")
    print("  " + _dash)
    for line in report_str.strip().splitlines():
        print("  " + line)
    print("  " + _dash)
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # ── Assert recall / precision targets ─────────────────────────────
    print(f"\n  Performance targets (flood class):")
    for metric_name, actual, target in [
        ("Recall",    recall,    min_recall),
        ("Precision", precision, min_precision),
    ]:
        symbol = "✓" if actual >= target else "⚠  WARNING"
        colour = "" if actual >= target else " ← BELOW TARGET"
        print(f"    {symbol}  {metric_name:<10} = {actual:.4f}  "
              f"(target ≥ {target:.2f}){colour}")
        if actual < target:
            print(f"    ⚠  {metric_name} {actual:.4f} is below the required "
                  f"threshold of {target:.2f}. Consider increasing epochs, "
                  f"adjusting class_weight, or lowering the decision threshold.")

    # ── Keras test evaluation ─────────────────────────────────────────
    keras_metrics_raw = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(keras_metrics_raw, (float, int)):
        keras_metrics = [keras_metrics_raw]
    else:
        keras_metrics = list(keras_metrics_raw)
    keras_names = model.metrics_names

    # ── Metrics dict ──────────────────────────────────────────────────
    metrics = {
        "model":         "LSTM_FloodClassifier",
        "input_shape":   list(X_train.shape[1:]),
        "epochs_trained": actual_eps,
        "best_epoch":    best_ep,
        "train_time_s":  round(train_time, 2),
        "threshold":     threshold,
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
            for n, v in zip(keras_names, keras_metrics)
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
    print(f"\n  ✓  Metrics dict    → {METRICS_PATH}")

    # ── Plots ─────────────────────────────────────────────────────────
    print(f"\n[5 / 5]  Generating plots …")
    _plot_training_curves(history, TRAIN_PLOT_PATH)
    _plot_confusion_matrix(y_test.astype(int), y_pred, CM_PLOT_PATH)

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{_sep}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(_sep)
    rows = [
        ("Epochs trained",    f"{actual_eps}  (best={best_ep})"),
        ("Wall time",         f"{train_time:.1f}s"),
        ("Accuracy",          f"{accuracy:.4f}"),
        ("Precision (flood)", f"{precision:.4f}  {'✓' if precision >= min_precision else '⚠'}"),
        ("Recall (flood)",    f"{recall:.4f}  {'✓' if recall >= min_recall else '⚠'}"),
        ("F1 (flood)",        f"{f1:.4f}"),
        ("ROC-AUC",           f"{roc_auc:.4f}"),
        ("PR-AUC",            f"{pr_auc:.4f}"),
    ]
    for label, val in rows:
        print(f"  {label:<22}  {val}")
    print(_sep)
    artefacts = [
        ("Checkpoint",      CHECKPOINT_PATH),
        ("Final model",     FINAL_PATH),
        ("Training curves", TRAIN_PLOT_PATH),
        ("Confusion matrix",CM_PLOT_PATH),
        ("Metrics JSON",    METRICS_PATH),
    ]
    print("\n  Artefacts saved:")
    for name, path in artefacts:
        size_kb = Path(path).stat().st_size / 1024
        print(f"    {name:<20}  {path}  ({size_kb:.1f} KB)")
    print(f"{_sep}\n")

    return model, history, metrics


# ══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    model, history, metrics = train_lstm(epochs=50, batch_size=64)
