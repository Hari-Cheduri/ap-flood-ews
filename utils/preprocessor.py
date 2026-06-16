"""
utils/preprocessor.py
----------------------
FloodDataPreprocessor — unified preprocessing for both the LSTM tabular
pipeline and the CNN satellite image pipeline.

Tabular / LSTM pipeline
-----------------------
1. Load data/raw/flood_dataset.csv
2. Parse timestamp → extract hour_sin (cyclical, diurnal feature) → 15 features
3. StandardScaler on all 15 numeric features
4. Sliding-window sequences  (window = 24 timesteps)
5. Stratified 70 / 15 / 15 split  (StratifiedShuffleSplit, random_state=42)
6. Persist scaler → models/scaler.pkl

CNN image pipeline
------------------
1. Load data/processed/satellite_images.npy + image_labels.npy
2. Verify / enforce pixel range [0, 1]
3. Same 70 / 15 / 15 stratified split  (independent, same random_state=42)

Output variables
----------------
  X_lstm_train / val / test  shape  (N, 24, 15)   float32
  X_cnn_train  / val / test  shape  (N, 64, 64, 3) float32
  y_train      / val / test  shape  (N,)            int8

All arrays are also saved to data/processed/ as .npy files.

Feature list (15 columns)
--------------------------
  0  rainfall_mm            9  antecedent_rainfall_72h
  1  river_level_m         10  elevation_m
  2  soil_moisture_pct     11  slope_deg
  3  temperature_c         12  distance_to_river_km
  4  humidity_pct          13  ndvi_index
  5  wind_speed_kmh        14  hour_sin  ← cyclical (engineered from timestamp)
  6  upstream_flow_m3s
  7  water_discharge
  8  pressure_hpa

Usage
-----
  from utils.preprocessor import FloodDataPreprocessor
  prep   = FloodDataPreprocessor()
  splits = prep.run()          # returns dict with all arrays
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════

TABULAR_CSV      = Path("data/raw/flood_dataset.csv")
IMAGES_NPY       = Path("data/processed/satellite_images.npy")
LABELS_NPY       = Path("data/processed/image_labels.npy")
PROCESSED_DIR    = Path("data/processed")
SCALER_PATH      = Path("models/scaler.pkl")

SEQ_LEN          = 24      # LSTM look-back window (timesteps)
RANDOM_STATE     = 42
TRAIN_FRAC       = 0.70
VAL_FRAC         = 0.15    # of total; test_frac = 1 - TRAIN_FRAC - VAL_FRAC

RAW_FEATURE_COLS = [
    "rainfall_mm", "river_level_m", "soil_moisture_pct",
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "upstream_flow_m3s", "water_discharge", "pressure_hpa",
    "antecedent_rainfall_72h", "elevation_m", "slope_deg",
    "distance_to_river_km", "ndvi_index",
]
ENGINEERED_COLS  = ["hour_sin"]          # +1 = 15 total
ALL_FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_COLS


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _bar(label: str, n: int, total: int, width: int = 36) -> str:
    filled = int(n / total * width)
    return f"  {label:<14} [{'█'*filled}{'░'*(width-filled)}] {n:>6,}/{total:,}"


def _class_balance(y: np.ndarray) -> str:
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    return (f"no-flood={n0:,} ({n0/len(y)*100:.1f}%)  "
            f"flood={n1:,} ({n1/len(y)*100:.1f}%)")


def _split_indices(
    y: np.ndarray,
    train_frac:   float = TRAIN_FRAC,
    random_state: int   = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-step StratifiedShuffleSplit → (train_idx, val_idx, test_idx).
    Step 1: carve out train (70 %) vs remaining (30 %).
    Step 2: split remaining 50/50 → val (15 %) and test (15 %).
    """
    idx = np.arange(len(y))

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=round(1 - train_frac, 6), random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(idx, y))

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state
    )
    rel_val, rel_test = next(sss2.split(temp_idx, y[temp_idx]))
    val_idx  = temp_idx[rel_val]
    test_idx = temp_idx[rel_test]

    return train_idx, val_idx, test_idx


# ════════════════════════════════════════════════════════════════════════════
#  Main class
# ════════════════════════════════════════════════════════════════════════════

class FloodDataPreprocessor:
    """
    Preprocesses tabular sensor data and satellite imagery for the hybrid
    CNN + LSTM flood monitoring model.

    Attributes
    ----------
    scaler : sklearn StandardScaler  (fitted on training sequences)
    splits : dict                     (populated after calling .run())
    """

    def __init__(
        self,
        tabular_csv:   Path | str = TABULAR_CSV,
        images_npy:    Path | str = IMAGES_NPY,
        labels_npy:    Path | str = LABELS_NPY,
        processed_dir: Path | str = PROCESSED_DIR,
        scaler_path:   Path | str = SCALER_PATH,
        seq_len:       int        = SEQ_LEN,
        random_state:  int        = RANDOM_STATE,
    ):
        self.tabular_csv   = Path(tabular_csv)
        self.images_npy    = Path(images_npy)
        self.labels_npy    = Path(labels_npy)
        self.processed_dir = Path(processed_dir)
        self.scaler_path   = Path(scaler_path)
        self.seq_len       = seq_len
        self.random_state  = random_state
        self.scaler        = StandardScaler()
        self.splits: dict  = {}

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  TABULAR PIPELINE                                                    #
    # ------------------------------------------------------------------ #

    def _load_tabular(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load CSV, engineer hour_sin, return (features_array, labels_array).

        Returns
        -------
        X : (N, 15)  raw unscaled features
        y : (N,)     flood labels  int8
        """
        print("  ├─ Loading CSV …", end=" ", flush=True)
        t0 = time.perf_counter()
        df = pd.read_csv(self.tabular_csv, parse_dates=["timestamp"])
        print(f"done  ({len(df):,} rows, {time.perf_counter()-t0:.2f}s)")

        # ── Engineered feature: cyclical hour-of-day ──────────────────
        # sin(2π × h / 24) maps 0 h and 24 h to the same value (periodic)
        hour = df["timestamp"].dt.hour.astype(np.float32)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)

        X = df[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
        y = df["flood_label"].to_numpy(dtype=np.int8)
        return X, y

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window sequence construction.

        Each sequence contains `seq_len` consecutive rows.
        The label is taken from the **last** row of the window
        (the timestep being predicted).

        Returns
        -------
        X_seq : (N-seq_len+1, seq_len, 15)
        y_seq : (N-seq_len+1,)
        """
        n_seq = len(X) - self.seq_len + 1

        print(f"  ├─ Building {n_seq:,} sliding-window sequences "
              f"(window={self.seq_len}) …", end=" ", flush=True)
        t0 = time.perf_counter()

        # Pre-allocate for speed
        X_seq = np.empty((n_seq, self.seq_len, X.shape[1]), dtype=np.float32)
        y_seq = np.empty(n_seq, dtype=np.int8)

        for i in range(n_seq):
            X_seq[i] = X[i : i + self.seq_len]
            y_seq[i] = y[i + self.seq_len - 1]

        print(f"done  ({time.perf_counter()-t0:.2f}s)")
        return X_seq, y_seq

    def _scale_sequences(
        self,
        X_train: np.ndarray,
        X_val:   np.ndarray,
        X_test:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit StandardScaler on training sequences, transform all splits.
        The scaler operates on the feature axis (last dim) by reshaping.
        """
        print("  ├─ Fitting StandardScaler on training sequences …",
              end=" ", flush=True)
        t0    = time.perf_counter()
        n_tr, s, f = X_train.shape

        # Reshape to (N×T, F), fit, reshape back
        self.scaler.fit(X_train.reshape(-1, f))

        X_train = self.scaler.transform(
            X_train.reshape(-1, f)).reshape(n_tr, s, f)
        X_val   = self.scaler.transform(
            X_val.reshape(-1, f)).reshape(X_val.shape)
        X_test  = self.scaler.transform(
            X_test.reshape(-1, f)).reshape(X_test.shape)

        joblib.dump(self.scaler, self.scaler_path)
        print(f"done  (saved → {self.scaler_path}, {time.perf_counter()-t0:.2f}s)")
        return X_train, X_val, X_test

    def preprocess_tabular(self) -> dict:
        """
        Full tabular preprocessing pipeline.

        Returns dict with keys:
          X_lstm_train, X_lstm_val, X_lstm_test  (N, 24, 15)
          y_train, y_val, y_test                  (N,)
        """
        print("\n┌─ [TABULAR / LSTM PIPELINE] ─────────────────────────────────")
        X_raw, y_raw = self._load_tabular()
        X_seq, y_seq = self._make_sequences(X_raw, y_raw)

        print("  ├─ Stratified split (70 / 15 / 15) …", end=" ", flush=True)
        t0 = time.perf_counter()
        train_idx, val_idx, test_idx = _split_indices(y_seq, TRAIN_FRAC, self.random_state)
        print(f"done  ({time.perf_counter()-t0:.2f}s)")

        X_tr  = X_seq[train_idx]
        X_val = X_seq[val_idx]
        X_te  = X_seq[test_idx]
        y_tr  = y_seq[train_idx]
        y_val = y_seq[val_idx]
        y_te  = y_seq[test_idx]

        X_tr, X_val, X_te = self._scale_sequences(X_tr, X_val, X_te)

        return {
            "X_lstm_train": X_tr,  "X_lstm_val": X_val,  "X_lstm_test": X_te,
            "y_train":      y_tr,  "y_val":      y_val,  "y_test":      y_te,
            "_train_idx":   train_idx, "_val_idx": val_idx, "_test_idx": test_idx,
        }

    # ------------------------------------------------------------------ #
    #  CNN IMAGE PIPELINE                                                  #
    # ------------------------------------------------------------------ #

    def preprocess_images(
        self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """
        Full CNN image preprocessing pipeline.

        Uses its own stratified split (independent from tabular) because
        the image dataset has 20,000 samples while the sequence dataset
        has 19,977. Same random_state=42 ensures identical class balance.

        Returns dict with keys:
          X_cnn_train, X_cnn_val, X_cnn_test  (N, 64, 64, 3)
        """
        print("├─ [CNN IMAGE PIPELINE] ──────────────────────────────────────")

        print("  ├─ Loading satellite_images.npy …", end=" ", flush=True)
        t0     = time.perf_counter()
        images = np.load(self.images_npy).astype(np.float32)
        labels = np.load(self.labels_npy).astype(np.int8)
        print(f"done  ({images.shape}, {time.perf_counter()-t0:.2f}s)")

        # ── Normalise to [0, 1] if needed ─────────────────────────────
        pmin, pmax = images.min(), images.max()
        if pmax > 1.0 + 1e-6:
            print(f"  ├─ Pixel range [{pmin:.3f}, {pmax:.3f}] → normalising …",
                  end=" ", flush=True)
            images = (images - pmin) / (pmax - pmin + 1e-8)
            print("done")
        else:
            print(f"  ├─ Pixel range [{pmin:.4f}, {pmax:.4f}] — already [0, 1] ✓")

        # ── Stratified split ──────────────────────────────────────────
        print("  ├─ Stratified split (70 / 15 / 15) …", end=" ", flush=True)
        t0 = time.perf_counter()
        tr_idx, vl_idx, te_idx = _split_indices(labels, TRAIN_FRAC, self.random_state)
        print(f"done  ({time.perf_counter()-t0:.2f}s)")

        return {
            "X_cnn_train": images[tr_idx],
            "X_cnn_val":   images[vl_idx],
            "X_cnn_test":  images[te_idx],
            "_cnn_y_train": labels[tr_idx],
            "_cnn_y_val":   labels[vl_idx],
            "_cnn_y_test":  labels[te_idx],
        }

    # ------------------------------------------------------------------ #
    #  SAVE                                                                #
    # ------------------------------------------------------------------ #

    def _save_splits(self, splits: dict):
        """Persist all arrays to data/processed/ as .npy files."""
        print("├─ [SAVING] ──────────────────────────────────────────────────")
        save_keys = [
            "X_lstm_train", "X_lstm_val", "X_lstm_test",
            "X_cnn_train",  "X_cnn_val",  "X_cnn_test",
            "y_train",       "y_val",       "y_test",
        ]
        total_bytes = 0
        for key in save_keys:
            arr  = splits[key]
            path = self.processed_dir / f"{key}.npy"
            t0   = time.perf_counter()
            np.save(path, arr)
            sz   = path.stat().st_size
            total_bytes += sz
            print(f"  ├─ {key:<18}  {str(arr.shape):<20}  "
                  f"{sz/1e6:>7.2f} MB  ({time.perf_counter()-t0:.2f}s)")
        print(f"  └─ Total written: {total_bytes/1e6:.1f} MB  → {self.processed_dir}/")

    # ------------------------------------------------------------------ #
    #  REPORT                                                              #
    # ------------------------------------------------------------------ #

    def _print_report(self, splits: dict):
        """Print shapes, class balance, and a visual summary table."""
        print()
        print("═" * 70)
        print("  PREPROCESSING REPORT")
        print("═" * 70)

        # ── LSTM splits ───────────────────────────────────────────────
        print("\n  LSTM SEQUENCES  (window=24, features=15)")
        print(f"  {'Split':<8}  {'Shape':<22}  {'dtype':<8}  Class balance")
        print("  " + "─" * 66)
        for tag, xk, yk in [
            ("Train", "X_lstm_train", "y_train"),
            ("Val",   "X_lstm_val",   "y_val"),
            ("Test",  "X_lstm_test",  "y_test"),
        ]:
            x = splits[xk]; y = splits[yk]
            print(f"  {tag:<8}  {str(x.shape):<22}  {str(x.dtype):<8}  "
                  f"{_class_balance(y)}")

        # ── CNN splits ────────────────────────────────────────────────
        print("\n  CNN IMAGES  (64×64×3 float32)")
        print(f"  {'Split':<8}  {'Shape':<26}  {'dtype':<8}  Class balance")
        print("  " + "─" * 66)
        for tag, xk, yk in [
            ("Train", "X_cnn_train", "_cnn_y_train"),
            ("Val",   "X_cnn_val",   "_cnn_y_val"),
            ("Test",  "X_cnn_test",  "_cnn_y_test"),
        ]:
            x = splits[xk]; y = splits[yk]
            print(f"  {tag:<8}  {str(x.shape):<26}  {str(x.dtype):<8}  "
                  f"{_class_balance(y)}")

        # ── Label arrays ─────────────────────────────────────────────
        print("\n  SHARED LABEL ARRAYS  (tabular split, aligned to LSTM)")
        print(f"  {'Split':<8}  {'Shape':<12}  {'dtype':<8}  Class balance")
        print("  " + "─" * 66)
        for tag, yk in [("y_train", "y_train"), ("y_val", "y_val"), ("y_test", "y_test")]:
            y = splits[yk]
            print(f"  {tag:<8}  {str(y.shape):<12}  {str(y.dtype):<8}  "
                  f"{_class_balance(y)}")

        # ── Visual bar chart ─────────────────────────────────────────
        lstm_total = sum(len(splits[k]) for k in ("X_lstm_train","X_lstm_val","X_lstm_test"))
        cnn_total  = sum(len(splits[k]) for k in ("X_cnn_train","X_cnn_val","X_cnn_test"))
        print("\n  SAMPLE COUNTS")
        for tag, key, total in [
            ("LSTM train", "X_lstm_train", lstm_total),
            ("LSTM val",   "X_lstm_val",   lstm_total),
            ("LSTM test",  "X_lstm_test",  lstm_total),
            ("CNN  train", "X_cnn_train",  cnn_total),
            ("CNN  val",   "X_cnn_val",    cnn_total),
            ("CNN  test",  "X_cnn_test",   cnn_total),
        ]:
            n = len(splits[key])
            print(_bar(tag, n, total))

        # ── Scaler info ───────────────────────────────────────────────
        print(f"\n  SCALER")
        print(f"  {'Type':<18}  StandardScaler  (fit on LSTM training set)")
        print(f"  {'Saved to':<18}  {self.scaler_path}")
        print(f"  {'Features (15)':<18}  {', '.join(ALL_FEATURE_COLS[:5])} …")

        print("\n" + "═" * 70)
        print("  Preprocessing complete — all arrays saved to data/processed/")
        print("═" * 70 + "\n")

    # ------------------------------------------------------------------ #
    #  PUBLIC ENTRY POINT                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the full preprocessing pipeline for both modalities.

        Returns
        -------
        splits : dict  containing all X_lstm_*, X_cnn_*, y_* arrays
                       plus the fitted scaler at self.scaler
        """
        t_start = time.perf_counter()
        print()
        print("╔" + "═" * 68 + "╗")
        print("║   FloodDataPreprocessor  —  Flood Monitoring System" + " " * 17 + "║")
        print("╚" + "═" * 68 + "╝")

        # ── Tabular ───────────────────────────────────────────────────
        tab_splits = self.preprocess_tabular()

        # ── Images ────────────────────────────────────────────────────
        img_splits = self.preprocess_images(
            tab_splits["y_train"], tab_splits["y_val"], tab_splits["y_test"]
        )

        # ── Merge ─────────────────────────────────────────────────────
        self.splits = {**tab_splits, **img_splits}

        # ── Save ──────────────────────────────────────────────────────
        self._save_splits(self.splits)

        # ── Report ────────────────────────────────────────────────────
        self._print_report(self.splits)

        elapsed = time.perf_counter() - t_start
        print(f"  Total wall time: {elapsed:.1f}s\n")
        return self.splits

    # ------------------------------------------------------------------ #
    #  CONVENIENCE UNPACKER                                                #
    # ------------------------------------------------------------------ #

    def unpack(self) -> tuple:
        """
        Return all split arrays as a flat tuple for easy unpacking.

        Usage
        -----
          (X_lstm_tr, X_lstm_val, X_lstm_te,
           X_cnn_tr,  X_cnn_val,  X_cnn_te,
           y_tr,      y_val,      y_te)   = prep.unpack()
        """
        s = self.splits
        return (
            s["X_lstm_train"], s["X_lstm_val"],  s["X_lstm_test"],
            s["X_cnn_train"],  s["X_cnn_val"],   s["X_cnn_test"],
            s["y_train"],      s["y_val"],        s["y_test"],
        )

    # ------------------------------------------------------------------ #
    #  RELOAD FROM DISK                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_splits(cls, processed_dir: str = str(PROCESSED_DIR)) -> dict:
        """
        Reload all preprocessed arrays from disk without rerunning the pipeline.

        Usage
        -----
          splits = FloodDataPreprocessor.load_splits()
        """
        d   = Path(processed_dir)
        keys = [
            "X_lstm_train", "X_lstm_val",  "X_lstm_test",
            "X_cnn_train",  "X_cnn_val",   "X_cnn_test",
            "y_train",       "y_val",        "y_test",
        ]
        splits = {}
        print(f"\nLoading preprocessed splits from {d} …")
        for k in keys:
            path = d / f"{k}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run FloodDataPreprocessor().run() first."
                )
            splits[k] = np.load(path)
            print(f"  {k:<20}  {str(splits[k].shape)}")
        return splits


# ════════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    prep   = FloodDataPreprocessor()
    splits = prep.run()