"""
utils/data_generator.py
------------------------
Generates a realistic synthetic flood monitoring dataset with 20,000 rows
and a perfectly balanced 50-50 class split (10,000 flood / 10,000 no-flood).

Design for ~90% model accuracy
--------------------------------
The original generator produced zero inter-class feature overlap, letting any
depth-1 decision stump achieve 100% accuracy — not useful for benchmarking
a real model. This version fixes that with three mechanisms:

1. Compressed primary-feature distributions
   Primary features (rainfall, river_level, soil_moisture, upstream_flow) are
   drawn from tighter Beta distributions close to their rule thresholds.
   This reduces the signal margin the model can exploit.

2. Secondary-feature confusion
   Temperature, humidity, pressure, wind, elevation, and NDVI are drawn from
   broadly overlapping distributions across both classes, so the model cannot
   use them as easy shortcuts.

3. Label noise  (LABEL_NOISE_RATE = 0.05 → 5 %)
   After generation, 5 % of each class is randomly re-labelled to the other
   class. These mislabelled rows retain their original feature values, creating
   "hard" examples where the signal contradicts the label. This sets a Bayes
   error floor of ~5 %, making the achievable ceiling ~95 % and the realistic
   trained-model accuracy ~88–93 %.

Class-balance invariance
   Because we flip N samples from 0→1 AND N samples from 1→0, the 50-50
   split is preserved exactly after noise.

Rules (still directionally correct for the CLEAN 95 % of each class)
----------------------------------------------------------------------
  Flood   (label=1): rainfall_mm > 80, river_level_m > 4.5,
                     soil_moisture_pct > 85, upstream_flow_m3s > 350
  No-flood(label=0): rainfall_mm < 40, river_level_m < 2.5,
                     soil_moisture_pct < 60

Usage
-----
  python utils/data_generator.py
  → saves  data/raw/flood_dataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────
SEED             = 42
RNG              = np.random.default_rng(SEED)

# ── Dataset constants ──────────────────────────────────────────────────────
N_TOTAL          = 20_000
N_FLOOD          = 10_000
N_NO_FLOOD       = 10_000
LABEL_NOISE_RATE = 0.05          # 5 % of each class will be re-labelled
START_TS         = datetime(2018, 1, 1, 0, 0, 0)


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def _noisy(base: np.ndarray, scale: float,
           lo: float = -np.inf, hi: float = np.inf) -> np.ndarray:
    """Additive Gaussian noise clipped to [lo, hi]."""
    return np.clip(base + RNG.normal(0, scale, len(base)), lo, hi)

def _beta_scaled(n: int, a: float, b: float,
                 lo: float, hi: float) -> np.ndarray:
    """Beta-distributed samples rescaled to [lo, hi]."""
    return lo + RNG.beta(a, b, n) * (hi - lo)

def _uniform(n: int, lo: float, hi: float) -> np.ndarray:
    return RNG.uniform(lo, hi, n).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Flood samples  (label = 1 before noise)
# ══════════════════════════════════════════════════════════════════════════

def _generate_flood(n: int) -> pd.DataFrame:
    """
    Flood conditions — primary features satisfy the rules but distributions
    are compressed toward the thresholds to reduce signal margin.

    Secondary features (humidity, pressure, temp, wind, elevation …) are
    drawn from distributions that OVERLAP heavily with the no-flood class,
    forcing the model to weight the primary features carefully.
    """
    # ── Primary features: tight distributions near thresholds ─────────
    # rainfall: mostly 81–130 mm  (was 82–300)
    rainfall_mm       = _noisy(_beta_scaled(n, 2.5, 3.5, 81.5, 130.0),
                                10.0, 80.1, 200.0)
    # upstream_flow: mostly 351–620 m³/s  (was 351–1200)
    upstream_flow_m3s = _noisy(_beta_scaled(n, 2.5, 3.5, 352.0, 620.0),
                                18.0, 350.1, 900.0)
    # river level: driven by upstream flow, compressed to 4.5–7 m
    _flow_norm        = (upstream_flow_m3s - 350) / 270.0
    river_level_m     = _noisy(4.52 + np.clip(_flow_norm, 0, 1) * 2.5,
                                0.30, 4.51, 8.5)
    # antecedent rainfall: moderate correlation with current rainfall
    antecedent_72h    = _noisy(rainfall_mm * 0.45 + _uniform(n, 8, 30),
                                12.0, 40.0, 200.0)
    # soil moisture: compressed range 85–95 %  (was 85–100)
    soil_moisture_pct = _noisy(85.2 + (antecedent_72h / 200.0) * 9.8,
                                2.5, 85.1, 98.0)

    # ── Secondary features: BROAD overlap with no-flood ────────────────
    # temperature: same range as no-flood
    temperature_c     = _noisy(_uniform(n, 16, 34),      4.0,  5.0, 45.0)
    # humidity: slightly elevated but wide → overlapping
    humidity_pct      = _noisy(_beta_scaled(n, 3, 2, 62, 98),
                                6.0, 30.0, 100.0)
    # pressure: somewhat lower → but wide distribution overlaps
    pressure_hpa      = _noisy(_beta_scaled(n, 2, 4, 985, 1020),
                                5.0, 955.0, 1035.0)
    # wind: overlapping range
    wind_speed_kmh    = _noisy(_beta_scaled(n, 2, 2.5, 12, 75),
                                7.0,  0.0, 120.0)
    # water discharge: scales with flow × level but noisy
    water_discharge   = _noisy(upstream_flow_m3s * river_level_m * 0.10,
                                35.0, 20.0, 1500.0)
    # elevation: flood-prone → low, but occasionally mid (confusing)
    elevation_m       = _noisy(_beta_scaled(n, 1.8, 4.5, 10, 250),
                                12.0,  5.0, 600.0)
    # slope: gentle → but wide range overlaps with no-flood
    slope_deg         = _noisy(_beta_scaled(n, 1.8, 4, 0, 18),
                                2.5,  0.0,  45.0)
    # distance to river: close → but moderate noise
    distance_to_river_km = _noisy(_beta_scaled(n, 1.5, 4, 0.05, 3.5),
                                   0.4, 0.01, 12.0)
    # NDVI: suppressed but noisy → some overlap with no-flood
    ndvi_index        = _noisy(_beta_scaled(n, 2, 4.5, -0.15, 0.50),
                                0.08, -1.0,  1.0)

    return pd.DataFrame({
        "rainfall_mm":             np.round(rainfall_mm,           3),
        "river_level_m":           np.round(river_level_m,         3),
        "soil_moisture_pct":       np.round(soil_moisture_pct,     3),
        "temperature_c":           np.round(temperature_c,         3),
        "humidity_pct":            np.round(humidity_pct,          3),
        "wind_speed_kmh":          np.round(wind_speed_kmh,        3),
        "upstream_flow_m3s":       np.round(upstream_flow_m3s,     3),
        "water_discharge":         np.round(water_discharge,       3),
        "pressure_hpa":            np.round(pressure_hpa,          3),
        "antecedent_rainfall_72h": np.round(antecedent_72h,        3),
        "elevation_m":             np.round(elevation_m,           3),
        "slope_deg":               np.round(slope_deg,             3),
        "distance_to_river_km":    np.round(distance_to_river_km,  4),
        "ndvi_index":              np.round(ndvi_index,            4),
        "flood_label":             np.ones(n, dtype=int),
    })


# ══════════════════════════════════════════════════════════════════════════
#  No-flood samples  (label = 0 before noise)
# ══════════════════════════════════════════════════════════════════════════

def _generate_no_flood(n: int) -> pd.DataFrame:
    """
    Normal / dry conditions — primary features raised closer to thresholds
    (higher rainfall, higher river level, higher soil moisture) to reduce
    the inter-class gap; secondary features broadly overlapping with flood.
    """
    # ── Primary features: pushed toward thresholds ────────────────────
    # rainfall: 10–39 mm  (was 0–39.9; mean now ~28 instead of ~11)
    rainfall_mm       = _noisy(_beta_scaled(n, 2.5, 2, 10.0, 39.4),
                                5.0,  0.0, 39.9)
    # upstream_flow: 40–340 m³/s  (was 10–340)
    upstream_flow_m3s = _noisy(_beta_scaled(n, 2.5, 2, 40.0, 338.0),
                                15.0,  0.0, 349.9)
    _flow_norm        = np.clip(upstream_flow_m3s / 340.0, 0, 1)
    river_level_m     = _noisy(0.5 + _flow_norm * 1.9,
                                0.20,  0.1, 2.49)
    antecedent_72h    = _noisy(rainfall_mm * 0.50 + _uniform(n, 0, 18),
                                6.0,  0.0, 80.0)
    # soil moisture: 35–59 %  (was 10–59.9; mean now ~48 instead of ~26)
    soil_moisture_pct = _noisy(35.0 + (antecedent_72h / 80.0) * 23.0,
                                5.0, 10.0, 59.9)

    # ── Secondary features: heavily overlapping with flood ─────────────
    temperature_c     = _noisy(_uniform(n, 14, 35),      4.0,  0.0, 50.0)
    humidity_pct      = _noisy(_beta_scaled(n, 2.5, 2.5, 45, 92),
                                6.0, 15.0,  99.0)
    pressure_hpa      = _noisy(_beta_scaled(n, 3, 2.5, 998, 1030),
                                5.0, 970.0, 1050.0)
    wind_speed_kmh    = _noisy(_beta_scaled(n, 1.8, 2.5, 5, 65),
                                6.0,  0.0,  90.0)
    water_discharge   = _noisy(upstream_flow_m3s * river_level_m * 0.09,
                                8.0,  0.0, 180.0)
    elevation_m       = _noisy(_beta_scaled(n, 3, 2, 80, 650),
                                20.0, 10.0, 2000.0)
    slope_deg         = _noisy(_beta_scaled(n, 2.5, 2, 4, 32),
                                3.0,  0.0,  60.0)
    distance_to_river_km = _noisy(_beta_scaled(n, 2.5, 1.8, 0.8, 12),
                                   1.2, 0.1, 50.0)
    ndvi_index        = _noisy(_beta_scaled(n, 4, 2.5, 0.25, 0.82),
                                0.07, -1.0,  1.0)

    return pd.DataFrame({
        "rainfall_mm":             np.round(rainfall_mm,           3),
        "river_level_m":           np.round(river_level_m,         3),
        "soil_moisture_pct":       np.round(soil_moisture_pct,     3),
        "temperature_c":           np.round(temperature_c,         3),
        "humidity_pct":            np.round(humidity_pct,          3),
        "wind_speed_kmh":          np.round(wind_speed_kmh,        3),
        "upstream_flow_m3s":       np.round(upstream_flow_m3s,     3),
        "water_discharge":         np.round(water_discharge,       3),
        "pressure_hpa":            np.round(pressure_hpa,          3),
        "antecedent_rainfall_72h": np.round(antecedent_72h,        3),
        "elevation_m":             np.round(elevation_m,           3),
        "slope_deg":               np.round(slope_deg,             3),
        "distance_to_river_km":    np.round(distance_to_river_km,  4),
        "ndvi_index":              np.round(ndvi_index,            4),
        "flood_label":             np.zeros(n, dtype=int),
    })


# ══════════════════════════════════════════════════════════════════════════
#  Label noise injection
# ══════════════════════════════════════════════════════════════════════════

def _apply_label_noise(
    flood_df:    pd.DataFrame,
    noflood_df:  pd.DataFrame,
    noise_rate:  float = LABEL_NOISE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly flip `noise_rate` fraction of labels in each class.

    Flip N flood→0 AND N no-flood→1, so total class balance is preserved:
      label=1 : (N_FLOOD - N_flip) + N_flip = N_FLOOD  ✓
      label=0 : (N_NO_FLOOD - N_flip) + N_flip = N_NO_FLOOD  ✓

    The feature values are UNCHANGED — flipped samples keep their
    class-consistent features but get the wrong label, creating
    genuinely hard / ambiguous training examples that prevent 100 % accuracy.
    """
    n_flip  = int(len(flood_df) * noise_rate)

    flip_f  = RNG.choice(len(flood_df),   n_flip, replace=False)
    flip_nf = RNG.choice(len(noflood_df), n_flip, replace=False)

    flood_df  = flood_df.copy();   flood_df.iloc[flip_f,  -1] = 0
    noflood_df = noflood_df.copy(); noflood_df.iloc[flip_nf, -1] = 1

    return flood_df, noflood_df


# ══════════════════════════════════════════════════════════════════════════
#  Timestamps
# ══════════════════════════════════════════════════════════════════════════

def _make_timestamps(n: int, start: datetime) -> list[str]:
    jitter = RNG.integers(0, 4, n)
    return [
        (start + timedelta(hours=i, minutes=int(jitter[i]))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════
#  Main generator
# ══════════════════════════════════════════════════════════════════════════

def generate_dataset(
    output_path:     str   = "data/raw/flood_dataset.csv",
    label_noise_rate: float = LABEL_NOISE_RATE,
) -> pd.DataFrame:
    """
    Build the 20,000-row dataset, inject label noise, shuffle, save.

    Returns
    -------
    pd.DataFrame  — shape (20000, 16)
    """
    print("=" * 64)
    print("  Flood Monitoring — Synthetic Dataset Generator  (v2)")
    print(f"  Seed={SEED}  |  Rows={N_TOTAL:,}  |  LabelNoise={label_noise_rate:.0%}")
    print("=" * 64)

    print("\n[1/4]  Generating flood samples     (n=10,000) …")
    flood_df    = _generate_flood(N_FLOOD)

    print("[2/4]  Generating no-flood samples  (n=10,000) …")
    noflood_df  = _generate_no_flood(N_NO_FLOOD)

    print(f"[3/4]  Applying {label_noise_rate:.0%} label noise "
          f"({int(N_FLOOD * label_noise_rate):,} flips per class) …")
    flood_df, noflood_df = _apply_label_noise(flood_df, noflood_df,
                                               label_noise_rate)

    print("[4/4]  Combining, shuffling, attaching timestamps, saving …")
    combined = (
        pd.concat([flood_df, noflood_df], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    combined.insert(0, "timestamp", _make_timestamps(N_TOTAL, START_TS))

    col_order = [
        "timestamp",
        "rainfall_mm", "river_level_m", "soil_moisture_pct",
        "temperature_c", "humidity_pct", "wind_speed_kmh",
        "upstream_flow_m3s", "water_discharge", "pressure_hpa",
        "antecedent_rainfall_72h", "elevation_m", "slope_deg",
        "distance_to_river_km", "ndvi_index",
        "flood_label",
    ]
    combined = combined[col_order]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # ── Verification report ───────────────────────────────────────────
    vc          = combined["flood_label"].value_counts().sort_index()
    flood_rows  = combined[combined["flood_label"] == 1]
    normal_rows = combined[combined["flood_label"] == 0]

    # Note: after noise injection some "flood" rows have no-flood features,
    # so rule compliance will be ~95 % (not 100 %) — this is intentional.
    print("\n" + "=" * 64)
    print("  DATASET VERIFICATION REPORT")
    print("=" * 64)

    print(f"\n  CLASS DISTRIBUTION  (after {label_noise_rate:.0%} label noise)")
    print(f"  {'Label':<13} {'Count':>8}  {'Share':>8}")
    print(f"  {'-'*32}")
    for label, count in vc.items():
        name = "Flood (1)" if label == 1 else "No-Flood (0)"
        bar  = "█" * int(count / N_TOTAL * 50)
        print(f"  {name:<13} {count:>8,}  {count/N_TOTAL*100:>7.2f}%  {bar}")
    print(f"  {'TOTAL':<13} {N_TOTAL:>8,}  {'100.00%':>8}")

    print(f"\n  RULE COMPLIANCE  (≈95% expected due to {label_noise_rate:.0%} noise)")
    checks = [
        ("Flood  | rainfall_mm > 80",        flood_rows,  "rainfall_mm",       ">",  80),
        ("Flood  | river_level_m > 4.5",      flood_rows,  "river_level_m",     ">",   4.5),
        ("Flood  | soil_moisture_pct > 85",   flood_rows,  "soil_moisture_pct", ">",  85),
        ("Flood  | upstream_flow_m3s > 350",  flood_rows,  "upstream_flow_m3s", ">", 350),
        ("NoFlood| rainfall_mm < 40",         normal_rows, "rainfall_mm",       "<",  40),
        ("NoFlood| river_level_m < 2.5",      normal_rows, "river_level_m",     "<",   2.5),
        ("NoFlood| soil_moisture_pct < 60",   normal_rows, "soil_moisture_pct", "<",  60),
    ]
    for desc, df_, col, op, thresh in checks:
        if op == ">":
            pct = (df_[col] > thresh).mean()
        else:
            pct = (df_[col] < thresh).mean()
        status = "✓" if pct >= 0.93 else "~"
        print(f"  {status}  {desc:<38}  {pct*100:6.2f}% compliant")

    print(f"\n  FEATURE STATISTICS BY CLASS  (key separation features)")
    print(f"  {'Feature':<26} {'Flood mean':>11} {'Flood std':>10} │ "
          f"{'Normal mean':>12} {'Normal std':>10}")
    print("  " + "─" * 74)
    for feat in ["rainfall_mm", "river_level_m", "soil_moisture_pct",
                 "upstream_flow_m3s", "humidity_pct", "pressure_hpa",
                 "elevation_m", "ndvi_index"]:
        fm  = flood_rows[feat].mean()
        fs  = flood_rows[feat].std()
        nm  = normal_rows[feat].mean()
        ns  = normal_rows[feat].std()
        print(f"  {feat:<26} {fm:>11.3f} {fs:>10.3f} │ {nm:>12.3f} {ns:>10.3f}")

    print(f"\n  NOISY SAMPLES INJECTED")
    print(f"  Flood→0  (flood features, label 0): "
          f"{int(N_FLOOD * label_noise_rate):>5,} rows")
    print(f"  NoFlood→1(no-flood features, label 1): "
          f"{int(N_NO_FLOOD * label_noise_rate):>5,} rows")
    print(f"  Expected Bayes error floor : ~{label_noise_rate*100:.0f}%")
    print(f"  Expected trained accuracy  : ~{(1-label_noise_rate)*100-5:.0f}"
          f"–{(1-label_noise_rate)*100:.0f}%")

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"\n  FILE  {Path(output_path).resolve()}  ({size_kb:.1f} KB)")
    print(f"  Rows={len(combined):,}  Cols={len(combined.columns)}")
    print("=" * 64 + "\n")
    return combined


if __name__ == "__main__":
    generate_dataset("data/raw/flood_dataset.csv")
