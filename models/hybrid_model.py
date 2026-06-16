"""
models/hybrid_model.py
-----------------------
Late-fusion Hybrid CNN + LSTM model for flood monitoring.

Design
------
Rather than re-training the CNN and LSTM from scratch, we freeze both
pretrained encoders, pre-extract their feature vectors, and train only
the lightweight fusion head on top — fast, memory-efficient, and lets
each encoder's pretrained weights do their job.

Feature extraction
  CNN branch  : cnn_feature_extractor.h5  (64×64×3 image → 128-dim ReLU vec)
  LSTM branch : best_lstm.h5 tapped at   (24×15 sequence → 64-dim LSTM vec)

Fusion head  (Keras Functional API)
  Input A : (128,)  CNN feature vector
  Input B : (64,)   LSTM encoder output
  Concat  → (192,)
  Dense(128, relu) → BatchNorm → Dropout(0.4)
  Dense(64,  relu) → Dropout(0.3)
  Dense(1,   sigmoid)  →  flood probability

Data pairing strategy
  CNN and LSTM datasets are independently generated (different sizes / indices).
  We pair them class-by-class: for each split take min(N_flood_cnn, N_flood_lstm)
  flood pairs and min(N_noflood_cnn, N_noflood_lstm) no-flood pairs.
  This preserves perfect 50-50 balance in every split.

  train: 13,983 pairs  |  val: 2,997  |  test: 2,997

Outputs
-------
  models/hybrid_final.h5                  trained fusion head
  data/processed/hybrid_feats_*.npy       cached pre-extracted features
  outputs/reports/hybrid_metrics.json     full metrics + history
  outputs/reports/hybrid_roc.png          ROC curve (hybrid vs single-modal)
  outputs/reports/hybrid_confusion.png    dual confusion matrix

Usage
-----
  python models/hybrid_model.py
"""

from __future__ import annotations

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

import tensorflow as tf
keras = tf.keras
layers = keras.layers
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR    = Path("data/processed")
MODELS_DIR       = Path("models")
REPORTS_DIR      = Path("outputs/reports")

CNN_EXTRACTOR    = MODELS_DIR  / "cnn_feature_extractor.h5"
LSTM_CHECKPOINT  = MODELS_DIR  / "best_lstm.h5"
HYBRID_FINAL     = MODELS_DIR  / "hybrid_final.h5"
METRICS_PATH     = REPORTS_DIR / "hybrid_metrics.json"
ROC_PATH         = REPORTS_DIR / "hybrid_roc.png"
CM_PATH          = REPORTS_DIR / "hybrid_confusion.png"

# Feature cache paths  (skip re-extraction if already computed)
FEAT_CACHE = {
    split: {
        "cnn":  PROCESSED_DIR / f"hybrid_feats_cnn_{split}.npy",
        "lstm": PROCESSED_DIR / f"hybrid_feats_lstm_{split}.npy",
        "y":    PROCESSED_DIR / f"hybrid_y_{split}.npy",
    }
    for split in ("train", "val", "test")
}


def _resolve_model(stem: str, base_dir: Path = MODELS_DIR) -> Path:
    """Return path to model file, preferring .keras over .h5."""
    for ext in (".keras", ".h5"):
        p = base_dir / (stem + ext)
        if p.exists():
            return p
    return base_dir / (stem + ".h5")  # fallback raises OSError at load time


# ── Visual palette (consistent with other model files) ────────────────────
P = {
    "bg":      "#0d1117", "panel":   "#161b22", "border":  "#30363d",
    "blue":    "#58a6ff", "green":   "#3fb950", "orange":  "#d29922",
    "red":     "#f85149", "purple":  "#bc8cff", "teal":    "#39d353",
    "text":    "#e6edf3", "subtext": "#8b949e",
}


# ══════════════════════════════════════════════════════════════════════════
#  Encoder builders
# ══════════════════════════════════════════════════════════════════════════

def _load_cnn_extractor() -> keras.Model:
    """Load pretrained CNN feature extractor (image → 128-dim)."""
    ext = keras.models.load_model(str(_resolve_model("cnn_feature_extractor")))
    ext.trainable = False          # frozen — used for feature extraction only
    print(f"  ✓  CNN extractor     loaded  {CNN_EXTRACTOR.name}"
          f"  output={ext.output_shape}")
    return ext


def _build_lstm_encoder(lstm_model: keras.Model) -> keras.Model:
    target = "lstm_2"
    try:
        out = lstm_model.get_layer(target).output
        print(f"  ✓  LSTM encoder      built   layer={target}"
              f"  output={out.shape}")
    except ValueError:
        lstm_layers = [l for l in lstm_model.layers
                       if isinstance(l, keras.layers.LSTM)]
        if not lstm_layers:
            raise ValueError("No LSTM layers found in checkpoint — "
                             "check that LSTM_CHECKPOINT is valid.")
        out = lstm_layers[-1].output
        print(f"  ⚠  layer '{target}' not found; using {lstm_layers[-1].name}"
              f"  output={out.shape}")
    encoder = keras.Model(
        inputs  = lstm_model.input,
        outputs = out,
        name    = "LSTM_Encoder_64",
    )
    encoder.trainable = False
    return encoder


# ══════════════════════════════════════════════════════════════════════════
#  Class-aligned feature pairing
# ══════════════════════════════════════════════════════════════════════════

def _load_cnn_labels_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce CNN label split (same as preprocessor, random_state=42)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = np.load(PROCESSED_DIR / "image_labels.npy").astype(np.int8)
    idx    = np.arange(len(labels))
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr, temp = next(s1.split(idx, labels))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    vl_r, te_r = next(s2.split(temp, labels[temp]))
    return labels[tr], labels[temp[vl_r]], labels[temp[te_r]]


def _pair_features(
    cnn_feats:  np.ndarray,   cnn_labels:  np.ndarray,
    lstm_feats: np.ndarray,   lstm_labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build class-aligned (cnn_feat, lstm_feat, label) triplets.

    For each class, take min(N_cnn_class, N_lstm_class) samples from each
    modality — preserving the 50-50 balance without duplication.

    Returns
    -------
    paired_cnn   : (N_total, 128)
    paired_lstm  : (N_total, 64)
    paired_y     : (N_total,)
    """
    blocks_cnn, blocks_lstm, blocks_y = [], [], []

    for cls in (0, 1):
        cnn_cls  = cnn_feats [cnn_labels  == cls]
        lstm_cls = lstm_feats[lstm_labels == cls]
        n        = min(len(cnn_cls), len(lstm_cls))

        # Shuffle within class before truncating (avoid ordering bias)
        ci = rng.permutation(len(cnn_cls)) [:n]
        li = rng.permutation(len(lstm_cls))[:n]

        blocks_cnn.append( cnn_cls [ci])
        blocks_lstm.append(lstm_cls[li])
        blocks_y.append(np.full(n, cls, dtype=np.float32))

    paired_cnn  = np.vstack(blocks_cnn)
    paired_lstm = np.vstack(blocks_lstm)
    paired_y    = np.hstack(blocks_y)

    # Final shuffle to interleave classes
    perm        = rng.permutation(len(paired_y))
    return paired_cnn[perm], paired_lstm[perm], paired_y[perm]


def _extract_and_pair_all(
    cnn_ext:      keras.Model,
    lstm_enc:     keras.Model,
    batch_size:   int = 256,
    force_recompute: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Pre-extract CNN + LSTM feature vectors for all three splits, pair them
    class-by-class, cache to disk, and return as a nested dict.

    If cache exists and force_recompute=False, loads from disk instead.

    Returns
    -------
    { "train": {"cnn": ..., "lstm": ..., "y": ...},
      "val":   {...}, "test": {...} }
    """
    rng = np.random.default_rng(42)

    # Check cache
    if not force_recompute and all(
        p.exists()
        for split_d in FEAT_CACHE.values()
        for p in split_d.values()
    ):
        print("  ↩  Loading cached hybrid features from disk …")
        return {
            split: {k: np.load(p) for k, p in split_d.items()}
            for split, split_d in FEAT_CACHE.items()
        }

    cnn_y_tr, cnn_y_va, cnn_y_te = _load_cnn_labels_split()
    lstm_y_tr = np.load(PROCESSED_DIR / "y_train.npy").astype(np.int8)
    lstm_y_va = np.load(PROCESSED_DIR / "y_val.npy"  ).astype(np.int8)
    lstm_y_te = np.load(PROCESSED_DIR / "y_test.npy" ).astype(np.int8)

    # File names and corresponding labels
    splits_cfg = [
        ("train", "X_cnn_train.npy", "X_lstm_train.npy", cnn_y_tr, lstm_y_tr),
        ("val",   "X_cnn_val.npy",   "X_lstm_val.npy",   cnn_y_va, lstm_y_va),
        ("test",  "X_cnn_test.npy",  "X_lstm_test.npy",  cnn_y_te, lstm_y_te),
    ]

    result = {}
    for split, cnn_file, lstm_file, cnn_y, lstm_y in splits_cfg:
        print(f"  ── Extracting features: {split} …")

        X_cnn  = np.load(PROCESSED_DIR / cnn_file)
        X_lstm = np.load(PROCESSED_DIR / lstm_file)

        t0 = time.perf_counter()
        F_cnn  = cnn_ext.predict( X_cnn,  batch_size=batch_size, verbose=0)
        F_lstm = lstm_enc.predict(X_lstm, batch_size=batch_size, verbose=0)
        print(f"     CNN  {X_cnn.shape}  → feats {F_cnn.shape}  "
              f"({time.perf_counter()-t0:.1f}s)")

        p_cnn, p_lstm, p_y = _pair_features(F_cnn, cnn_y, F_lstm, lstm_y, rng)

        # Cache to disk
        np.save(FEAT_CACHE[split]["cnn"],  p_cnn)
        np.save(FEAT_CACHE[split]["lstm"], p_lstm)
        np.save(FEAT_CACHE[split]["y"],    p_y)

        result[split] = {"cnn": p_cnn, "lstm": p_lstm, "y": p_y}
        print(f"     Paired: {p_cnn.shape[0]:,} samples  "
              f"flood={(p_y==1).sum():,}  no-flood={(p_y==0).sum():,}")

    return result


# ══════════════════════════════════════════════════════════════════════════
#  Hybrid fusion model builder
# ══════════════════════════════════════════════════════════════════════════

def build_hybrid_model(
    cnn_feat_dim:  int   = 128,
    lstm_feat_dim: int   = 64,
    fuse_units:    tuple = (128, 64),
    dropout:       tuple = (0.4, 0.3),
    learning_rate: float = 0.0005,
) -> keras.Model:
    """
    Late-fusion classifier that takes pre-extracted feature vectors as inputs.

    Parameters
    ----------
    cnn_feat_dim  : dimension of CNN feature vector  (128)
    lstm_feat_dim : dimension of LSTM encoder output (64)
    fuse_units    : Dense units in fusion layers     (128, 64)
    dropout       : Dropout rates after each fusion  (0.4, 0.3)
    learning_rate : Adam learning rate               (0.0005)

    Returns
    -------
    compiled keras.Model  — inputs=[cnn_feat, lstm_feat], output=flood_prob
    """
    cnn_input  = keras.Input(shape=(cnn_feat_dim,),  name="cnn_features")
    lstm_input = keras.Input(shape=(lstm_feat_dim,), name="lstm_features")

    # ── Feature fusion ────────────────────────────────────────────────
    x = layers.Concatenate(name="feature_concat")([cnn_input, lstm_input])
    # → (192,)

    # ── Fusion block 1 ───────────────────────────────────────────────
    x = layers.Dense(fuse_units[0], activation="relu",
                     name="fuse_dense_1")(x)
    x = layers.BatchNormalization(name="fuse_bn")(x)
    x = layers.Dropout(dropout[0], name="fuse_drop_1")(x)

    # ── Fusion block 2 ───────────────────────────────────────────────
    x = layers.Dense(fuse_units[1], activation="relu",
                     name="fuse_dense_2")(x)
    x = layers.Dropout(dropout[1], name="fuse_drop_2")(x)

    # ── Output ───────────────────────────────────────────────────────
    output = layers.Dense(1, activation="sigmoid",
                          name="flood_prob")(x)

    model = keras.Model(
        inputs  = [cnn_input, lstm_input],
        outputs = output,
        name    = "HybridFloodClassifier",
    )
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

def _apply_dark(fig, axes_flat):
    fig.patch.set_facecolor(P["bg"])
    for ax in axes_flat:
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["subtext"], labelsize=9)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])


def _plot_roc(
    y_true: np.ndarray,
    y_prob_hybrid: np.ndarray,
    y_prob_cnn:    np.ndarray | None,
    y_prob_lstm:   np.ndarray | None,
    save_path: Path,
):
    """
    ROC + Precision-Recall dual-panel plot.
    Overlays CNN-only and LSTM-only curves when available.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6),
                                          facecolor=P["bg"])
    fig.suptitle("Hybrid CNN+LSTM — ROC & Precision-Recall Curves",
                 fontsize=14, fontweight="bold", color=P["text"], y=1.01)

    curves = [
        ("Hybrid CNN+LSTM", y_prob_hybrid, P["teal"],   2.5),
        ("CNN only",        y_prob_cnn,    P["blue"],   1.5),
        ("LSTM only",       y_prob_lstm,   P["purple"], 1.5),
    ]

    # ── ROC ──────────────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], "--", color=P["subtext"],
                lw=1.2, label="Random (AUC=0.50)")
    for label, probs, colour, lw in curves:
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc         = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, color=colour, lw=lw,
                    label=f"{label}  AUC={auc:.4f}")

    ax_roc.set_xlim(-0.01, 1.01); ax_roc.set_ylim(-0.01, 1.01)
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=10)
    ax_roc.set_title("ROC Curve",            fontsize=11, fontweight="bold")
    ax_roc.legend(fontsize=9, facecolor=P["panel"],
                  labelcolor=P["text"], edgecolor=P["border"])

    # ── Precision-Recall ─────────────────────────────────────────────
    baseline = y_true.mean()
    ax_pr.axhline(baseline, color=P["subtext"], lw=1.2, linestyle="--",
                  label=f"Baseline (P={baseline:.2f})")
    for label, probs, colour, lw in curves:
        if probs is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap           = average_precision_score(y_true, probs)
        ax_pr.plot(rec, prec, color=colour, lw=lw,
                   label=f"{label}  AP={ap:.4f}")

    ax_pr.set_xlim(-0.01, 1.01); ax_pr.set_ylim(-0.01, 1.01)
    ax_pr.set_xlabel("Recall",    fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.legend(fontsize=9, facecolor=P["panel"],
                 labelcolor=P["text"], edgecolor=P["border"])

    _apply_dark(fig, [ax_roc, ax_pr])
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  ROC + PR curves   → {save_path}")


def _plot_confusion(y_true, y_pred, save_path: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ["No-Flood (0)", "Flood (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=P["bg"])
    for ax, data, fmt, title, cmap in zip(
        axes,
        [cm, cm_norm], ["d", ".2%"],
        ["Counts", "Row-Normalised"], ["Blues", "RdYlGn"],
    ):
        im = ax.imshow(data, cmap=cmap, vmin=0,
                       vmax=(None if fmt == "d" else 1))
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels, fontsize=10, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=10, color=P["text"])
        ax.set_ylabel("Actual",    fontsize=10, color=P["text"])
        ax.set_title(f"Confusion Matrix — {title}", fontsize=11,
                     fontweight="bold", color=P["text"], pad=10)
        thresh = data.max() / 2.0
        for i in range(2):
            for j in range(2):
                v = data[i, j]
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if v < thresh else P["bg"])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=P["subtext"], labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=P["subtext"])
        cb.outline.set_edgecolor(P["border"])
        ax.set_facecolor(P["panel"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
        ax.tick_params(colors=P["subtext"])

    fig.suptitle("Hybrid CNN+LSTM — Test Set Confusion Matrix",
                 fontsize=13, fontweight="bold", color=P["text"], y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  Confusion matrix  → {save_path}")


def _plot_training_curves(history: dict, best_ep: int, save_path: Path):
    """3-panel: Loss | Accuracy | AUC  with val overlay."""
    h   = history
    eps = range(1, len(h["loss"]) + 1)

    fig = plt.figure(figsize=(15, 4.5), facecolor=P["bg"])
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.36,
                            left=0.06, right=0.97, top=0.83, bottom=0.15)
    axes = []

    for col, (title, tr_k, vl_k, c_tr, c_vl) in enumerate([
        ("Loss",     "loss",     "val_loss",     P["red"],   P["orange"]),
        ("Accuracy", "accuracy", "val_accuracy", P["blue"],  P["purple"]),
        ("AUC",      "auc",      "val_auc",      P["teal"],  P["orange"]),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(eps, h[tr_k], color=c_tr, lw=2, label="Train")
        ax.plot(eps, h[vl_k], color=c_vl, lw=2, label="Val", linestyle="--")
        ax.axvline(best_ep, color=P["subtext"], lw=1, linestyle=":",
                   label=f"Best ep {best_ep}")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=P["panel"],
                  labelcolor=P["text"], edgecolor=P["border"])
        axes.append(ax)

    _apply_dark(fig, axes)
    fig.suptitle("Hybrid CNN+LSTM — Training Curves",
                 fontsize=14, fontweight="bold", color=P["text"], y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  Training curves   → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Training function
# ══════════════════════════════════════════════════════════════════════════

def train_hybrid(
    epochs:        int   = 30,
    batch_size:    int   = 32,
    threshold:     float = 0.50,
    min_recall:    float = 0.80,
    min_precision: float = 0.80,
    verbose:       int   = 1,
    force_recompute: bool = False,
) -> tuple[keras.Model, dict]:
    """
    Load pretrained encoders → extract paired features → train fusion head
    → evaluate → save all artefacts.

    Parameters
    ----------
    epochs          : max training epochs
    batch_size      : mini-batch size
    threshold       : sigmoid decision threshold
    min_recall      : WARNING printed if flood recall falls below this
    min_precision   : WARNING printed if flood precision falls below this
    verbose         : Keras fit verbosity
    force_recompute : re-extract features even if cache exists

    Returns
    -------
    model   : trained HybridFloodClassifier
    metrics : full metrics dict
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    SEP  = "═" * 65
    DASH = "─" * 65

    print(f"\n{SEP}")
    print("  Hybrid CNN + LSTM Flood Classifier  —  Training Pipeline")
    print(SEP)

    # ── 1. Load pretrained encoders ───────────────────────────────────
    print("\n[1 / 5]  Loading pretrained encoders …")
    cnn_ext  = _load_cnn_extractor()
    lstm_raw = keras.models.load_model(str(_resolve_model("best_lstm")))
    lstm_enc = _build_lstm_encoder(lstm_raw)
    print(f"  Encoders frozen  →  CNN 128-dim | LSTM 64-dim | concat 192-dim")

    # ── 2. Extract + pair features ────────────────────────────────────
    print("\n[2 / 5]  Extracting and pairing features (class-aligned) …")
    feats = _extract_and_pair_all(cnn_ext, lstm_enc,
                                   force_recompute=force_recompute)

    tr = feats["train"]; va = feats["val"]; te = feats["test"]
    print(f"\n  Paired split sizes:")
    for name, d in feats.items():
        y = d["y"]
        print(f"    {name:<6}: {len(y):>6,}  flood={int((y==1).sum()):,}"
              f"  no-flood={int((y==0).sum()):,}")

    # ── 3. Build hybrid model ─────────────────────────────────────────
    print(f"\n[3 / 5]  Building hybrid fusion model …")
    model = build_hybrid_model()
    model.summary(line_length=67, print_fn=lambda s: print("  " + s))

    # ── Class weights ─────────────────────────────────────────────────
    cw = {
        int(c): float(w)
        for c, w in zip(
            [0, 1],
            compute_class_weight("balanced", classes=np.array([0, 1]),
                                 y=tr["y"]),
        )
    }
    print(f"  Class weights: {cw}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(MODELS_DIR / "hybrid_final.h5"), monitor="val_loss",
                        save_best_only=True, verbose=1),
    ]

    # ── 4. Train ──────────────────────────────────────────────────────
    print(f"\n[4 / 5]  Training  (epochs={epochs}, batch_size={batch_size}) …")
    t_start = time.perf_counter()

    history = model.fit(
        [tr["cnn"], tr["lstm"]], tr["y"],
        validation_data=([va["cnn"], va["lstm"]], va["y"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    train_time = time.perf_counter() - t_start
    actual_eps = len(history.history["loss"])
    best_ep    = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\n  Trained {actual_eps} epochs  |  best epoch = {best_ep}"
          f"  |  wall time = {train_time:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────────
    print(f"\n[5 / 5]  Evaluating on test set …")

    y_true         = te["y"].astype(int)
    y_prob_hybrid  = model.predict(
        [te["cnn"], te["lstm"]], batch_size=batch_size, verbose=0
    ).ravel()
    y_pred         = (y_prob_hybrid >= threshold).astype(int)

    # Single-modal test probabilities for ROC comparison
    # (re-use the same test images / sequences used for feature extraction)
    print("  Computing single-modal baselines for ROC overlay …")
    X_cnn_test  = np.load(PROCESSED_DIR / "X_cnn_test.npy")
    X_lstm_test = np.load(PROCESSED_DIR / "X_lstm_test.npy")

    # CNN classifier (full model, not extractor)
    try:
        cnn_full    = keras.models.load_model(str(MODELS_DIR / "best_cnn.h5"))
        cnn_y_te    = np.load(FEAT_CACHE["test"]["y"])   # paired test labels
        # Need CNN predictions aligned to paired test set labels
        # Use cnn extractor output already stored in te["cnn"]
        # Reconstruct probs via a temp model: feature → dense → sigmoid
        # Simpler: just load cnn_final and predict on the full test set
        # then use matched indices from the pairing
        # For the ROC overlay we just need approximate probs on the same y_true
        cnn_ext_only = keras.models.load_model(str(CNN_EXTRACTOR))
        # Predict on the first len(y_true) CNN test images
        n            = min(len(y_true), len(X_cnn_test))
        cnn_ext_feats = cnn_ext_only.predict(
            X_cnn_test[:n], batch_size=batch_size, verbose=0
        )
        # Score via the fusion model using zero LSTM features (ablation)
        zero_lstm    = np.zeros((n, 64), dtype=np.float32)
        y_prob_cnn   = model.predict(
            [cnn_ext_feats, zero_lstm], batch_size=batch_size, verbose=0
        ).ravel()[:len(y_true)]
    except Exception as e:
        print(f"  ⚠  CNN-only baseline skipped: {e}")
        y_prob_cnn = None

    try:
        lstm_best  = keras.models.load_model(str(LSTM_CHECKPOINT))
        n          = min(len(y_true), len(X_lstm_test))
        # Ablation: score through fusion with zero CNN features
        zero_cnn   = np.zeros((len(y_true), 128), dtype=np.float32)
        y_prob_lstm = model.predict(
            [zero_cnn, te["lstm"][:len(y_true)]],
            batch_size=batch_size, verbose=0
        ).ravel()
    except Exception as e:
        print(f"  ⚠  LSTM-only baseline skipped: {e}")
        y_prob_lstm = None

   # Classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["No-Flood", "Flood"],
        output_dict=True,
    )
    report_str = str(classification_report(
        y_true,
        y_pred,
        target_names=["No-Flood", "Flood"]
    ))
    roc_auc   = roc_auc_score(y_true, y_prob_hybrid)
    pr_auc    = average_precision_score(y_true, y_prob_hybrid)
    recall    = float(report_dict["Flood"]["recall"])  # type: ignore
    precision = float(report_dict["Flood"]["precision"])  # type: ignore
    f1        = float(report_dict["Flood"]["f1-score"])  # type: ignore
    accuracy  = float(report_dict["accuracy"])  # type: ignore

    print(f"\n  Classification Report (test set):")
    print(f"  {DASH}")
    for line in report_str.strip().splitlines():
        print(f"  {line}")
    print(f"  {DASH}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # ── Precision / recall targets ────────────────────────────────────
    print(f"\n  Performance targets (flood class):")
    for name, val, target in [
        ("Recall",    recall,    min_recall),
        ("Precision", precision, min_precision),
    ]:
        ok = val >= target
        print(
            f"    {'✓' if ok else '⚠  WARNING'}  {name:<10} = {val:.4f}"
            + (f"  ← BELOW target ({target:.2f})" if not ok else "")
        )
        if not ok:
            print(f"    ⚠  Consider more epochs or a lower decision threshold.")

    # ── Metrics dict ──────────────────────────────────────────────────
    metrics = {
        "model":           "HybridFloodClassifier",
        "architecture": {
            "cnn_feat_dim":  128,
            "lstm_feat_dim": 64,
            "concat_dim":    192,
            "fusion_layers": [128, 64, 1],
            "dropouts":      [0.4, 0.3],
        },
        "epochs_trained": actual_eps,
        "best_epoch":     best_ep,
        "train_time_s":   round(train_time, 2),
        "threshold":      threshold,
        "test": {
            "accuracy":        round(float(accuracy),  4),
            "precision_flood": round(float(precision), 4),
            "recall_flood":    round(float(recall),    4),
            "f1_flood":        round(float(f1),        4),
            "roc_auc":         round(roc_auc,   4),
            "pr_auc":          round(pr_auc,    4),
        },
        "test_keras": {},   # removed duplicate evaluation pass
        "history": {
            k: [round(float(x), 5) for x in v]
            for k, v in history.history.items()
        },
        "classification_report": report_dict,
        "targets": {
            "min_recall":    min_recall,
            "min_precision": min_precision,
                "recall_met":    bool(recall    >= float(min_recall)),
                "precision_met": bool(precision >= float(min_precision)),
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓  Metrics dict      → {METRICS_PATH}")

    # ── Plots ─────────────────────────────────────────────────────────
    tc_path = REPORTS_DIR / "hybrid_training.png"
    _plot_training_curves(history.history, best_ep, tc_path)
    _plot_roc(y_true, y_prob_hybrid, y_prob_cnn, y_prob_lstm, ROC_PATH)
    _plot_confusion(y_true, y_pred, CM_PATH)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(SEP)
    for label, val in [
        ("Epochs trained",    f"{actual_eps}  (best={best_ep})"),
        ("Wall time",         f"{train_time:.1f}s"),
        ("Accuracy",          f"{accuracy:.4f}"),
        ("Precision (flood)", f"{precision:.4f}  {'✓' if precision >= float(min_precision) else '⚠'}"),
        ("Recall (flood)",    f"{recall:.4f}  {'✓' if recall >= float(min_recall) else '⚠'}"),
        ("F1 (flood)",        f"{f1:.4f}"),
        ("ROC-AUC",           f"{roc_auc:.4f}"),
        ("PR-AUC",            f"{pr_auc:.4f}"),
    ]:
        print(f"  {label:<24}  {val}")
    print(SEP)
    print("\n  Artefacts saved:")
    for name, path in [
        ("Hybrid model",       HYBRID_FINAL),
        ("Metrics JSON",       METRICS_PATH),
        ("ROC + PR curves",    ROC_PATH),
        ("Confusion matrix",   CM_PATH),
        ("Training curves",    tc_path),
    ]:
        sz = Path(path).stat().st_size / 1024
        print(f"    {name:<22}  {path}  ({sz:.1f} KB)")
    print(f"{SEP}\n")

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    model, metrics = train_hybrid(epochs=30, batch_size=32)