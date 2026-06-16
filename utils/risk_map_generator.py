"""
utils/risk_map_generator.py
----------------------------
FloodRiskMapGenerator — converts pretrained hybrid CNN+LSTM model
predictions into geo-referenced flood risk maps for the Andhra Pradesh region.

Pipeline per date
-----------------
1. Build a 50×50 spatial grid  (lat 17.0–17.5, lon 78.3–78.8)
2. For every grid cell simulate:
   • a 64×64×3 satellite patch  (SAR / NDVI / DEM, spatially coherent)
   • a 24×15 sensor sequence    (scaled with the training scaler)
3. Batch-extract CNN features (128-dim) and LSTM features (64-dim)
4. Run the hybrid fusion model → 2500 flood probabilities → (50,50) grid
5. Persist four artefacts:

   outputs/flood_risk_maps/risk_map_{date}.npy       numpy probability grid
   outputs/flood_risk_maps/risk_map_{date}.png        matplotlib choropleth
   outputs/flood_risk_maps/interactive_map_{date}.html  folium HTML map
   outputs/predictions/risk_summary_{date}.json       summary statistics

Risk thresholds
---------------
  0.0 – 0.3  Normal       ██ green
  0.3 – 0.6  Watch        ██ yellow
  0.6 – 0.8  Warning      ██ orange
  0.8 – 1.0  Emergency    ██ red

Usage
-----
  from utils.risk_map_generator import FloodRiskMapGenerator
  gen = FloodRiskMapGenerator()
  gen.generate_risk_map("2024-07-15")
  gen.generate_historical_maps(n_days=7)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import joblib
import cv2

import folium
from folium.plugins import HeatMap
from scipy.ndimage import gaussian_filter

import tensorflow as tf
import tensorflow.keras as keras

# ── Paths ─────────────────────────────────────────────────────────────────
MODELS_DIR       = Path("models")
RISK_MAPS_DIR    = Path("outputs/flood_risk_maps")
PREDICTIONS_DIR  = Path("outputs/predictions")

CNN_EXTRACTOR    = MODELS_DIR / "cnn_feature_extractor.h5"
LSTM_CHECKPOINT  = MODELS_DIR / "best_lstm.h5"
HYBRID_MODEL     = MODELS_DIR / "hybrid_final.h5"
SCALER_PATH      = MODELS_DIR / "scaler.pkl"

def _resolve_model(stem: str, base_dir: Path = MODELS_DIR) -> Path:
    """Probe for .keras first (preferred), fall back to .h5."""
    for ext in (".keras", ".h5"):
        p = base_dir / (stem + ext)
        if p.exists():
            return p
    return base_dir / (stem + ".h5")  # fallback raises OSError at load time

# Andhra Pradesh coverage

LAT_MIN, LAT_MAX = 12.5, 19.5
LON_MIN, LON_MAX = 76.0, 85.5

STATE_CENTER = (15.9129, 79.7400)

# Major rivers
GODAVARI_LAT   = 17.25
KRISHNA_LAT    = 16.15
PENNA_LAT      = 14.45
VAMSADHARA_LAT = 18.35

# ── Risk palette ─────────────────────────────────────────────────────────
RISK_LEVELS = [
    ("Normal",      0.0, 0.3, "#2ecc71", "#1a8a4a"),
    ("Watch", 0.3, 0.6, "#f1c40f", "#c9a800"),
    ("Warning",     0.6, 0.8, "#e67e22", "#b35f0c"),
    ("Emergency",  0.8, 1.001, "#e74c3c", "#9b1c1c"),
]

# ── Sensor feature column order (must match preprocessor.py) ─────────────
FEATURE_COLS = [
    "rainfall_mm", "river_level_m", "soil_moisture_pct",
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "upstream_flow_m3s", "water_discharge", "pressure_hpa",
    "antecedent_rainfall_72h", "elevation_m", "slope_deg",
    "distance_to_river_km", "ndvi_index", "hour_sin",
]

# ── Visual style ─────────────────────────────────────────────────────────
P = {
    "bg":     "#0d1117", "panel":  "#161b22", "border": "#30363d",
    "text":   "#e6edf3", "sub":    "#8b949e",
}


# ════════════════════════════════════════════════════════════════════════
#  Main class
# ════════════════════════════════════════════════════════════════════════

class FloodRiskMapGenerator:
    """
    Generate geo-referenced flood risk maps using the pretrained
    Hybrid CNN + LSTM model.

    Parameters
    ----------
    models_dir   : directory containing saved .h5 model files
    output_dir   : root output directory (subdirs are created automatically)
    grid_size    : default spatial grid resolution (grid_size × grid_size)
    batch_size   : inference batch size for model prediction
    """

    def __init__(
        self,
        models_dir:  str | Path = MODELS_DIR,
        output_dir:  str | Path = Path("outputs"),
        batch_size:  int        = 256,
    ):
        self.models_dir = Path(models_dir)
        self.risk_dir   = Path(output_dir) / "flood_risk_maps"
        self.pred_dir   = Path(output_dir) / "predictions"
        self.batch_size = batch_size

        self.risk_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        self._cnn_ext     : keras.Model | None = None
        self._lstm_enc    : keras.Model | None = None
        self._hybrid      : keras.Model | None = None
        self._scaler                           = None

    # ── Lazy model loading ────────────────────────────────────────────

    def _load_models(self):
        if self._hybrid is not None:
            return   # already loaded

        print("  [Models] Loading pretrained encoders and hybrid model …")
        t0 = time.perf_counter()

        cnn_ext = keras.models.load_model(str(_resolve_model("cnn_feature_extractor")))
        cnn_ext.trainable = False
        self._cnn_ext = cnn_ext

        lstm_full      = keras.models.load_model(str(_resolve_model("best_lstm")))
        lstm_enc = keras.Model(
            inputs  = lstm_full.input,
            outputs = lstm_full.get_layer("lstm_2").output,
            name    = "LSTM_Encoder",
        )
        lstm_enc.trainable = False
        self._lstm_enc = lstm_enc

        hybrid = keras.models.load_model(str(_resolve_model("hybrid_final")))
        hybrid.trainable = False
        self._hybrid = hybrid

        self._scaler   = joblib.load(str(SCALER_PATH))

        print(f"  [Models] Loaded in {time.perf_counter()-t0:.1f}s  "
              f"| CNN→128  LSTM→64  Hybrid→1")

    # ── Spatial simulation ────────────────────────────────────────────

    def _spatial_field(
        self,
        grid_size: int,
        rng: np.random.Generator,
        smooth: float = 4.0,
    ) -> np.ndarray:
        """
        Generate a spatially correlated random field on the grid using
        Gaussian smoothing of white noise.  Returns (grid_size, grid_size)
        float32 in [0, 1].
        """
        raw    = rng.random((grid_size, grid_size)).astype(np.float32)
        smooth_field = gaussian_filter(raw, sigma=smooth)
        mn, mx = smooth_field.min(), smooth_field.max()
        return (smooth_field - mn) / (mx - mn + 1e-8)
    
    def _river_proximity(self, lats, lons):
        """
        Combined influence of major Andhra Pradesh rivers.
        """

        godavari = np.exp(
            -(((lats - 17.0) ** 2 + (lons - 81.8) ** 2) / 0.5)
        )

        krishna = np.exp(
            -(((lats - 16.5) ** 2 + (lons - 80.6) ** 2) / 0.5)
        )

        penna = np.exp(
            -(((lats - 14.5) ** 2 + (lons - 79.9) ** 2) / 0.5)
        )

        vamsa = np.exp(
            -(((lats - 18.5) ** 2 + (lons - 84.0) ** 2) / 0.5)
        )

        prox = (
            0.35 * godavari +
            0.35 * krishna +
            0.20 * penna +
            0.10 * vamsa
        )

        return (prox - prox.min()) / (prox.max() - prox.min() + 1e-8)

    def _simulate_satellite_patches(
        self,
        grid_size: int,
        rng: np.random.Generator,
        risk_field: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate N=grid_size² satellite patches (64×64×3) with spatial
        coherence driven by `risk_field`.

        Channels:
          0  NDVI  — inversely correlated with risk (low veg near flooded cells)
          1  SAR   — positively correlated with risk (Warning return near water)
          2  DEM   — inversely correlated with risk (low elev near flood)
        """
        N    = grid_size * grid_size
        S    = 64
        risk = risk_field.ravel()                 # (N,)

        # SAR: risk → Warninger backscatter (water)
        sar_center  = 0.40 + risk * 0.45          # [0.40, 0.85]
        # NDVI: risk → suppressed vegetation
        ndvi_center = 0.60 - risk * 0.50          # [0.10, 0.60]
        # DEM: risk → lower elevation
        dem_center  = 0.55 - risk * 0.40          # [0.15, 0.55]

        patches = np.empty((N, S, S, 3), dtype=np.float32)
        for i in range(N):
            # Per-channel spatial texture
            ndvi = self._perlin_patch(S, ndvi_center[i], 0.12, rng)
            sar  = self._perlin_patch(S, sar_center[i],  0.12, rng)
            dem  = self._perlin_patch(S, dem_center[i],  0.10, rng)
            patches[i] = np.stack([ndvi, sar, dem], axis=-1)

        return patches

    @staticmethod
    def _perlin_patch(
        size: int,
        center: float,
        noise_std: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Single-channel texture: smooth bilinear base + per-pixel noise."""
        coarse = rng.random((max(4, size // 8), max(4, size // 8))).astype(np.float32)
        # cv2.resize can return objects that don't support numpy scalar ops in some
        # environments; ensure we have a plain numpy float32 array before math.
        smooth = np.asarray(
            cv2.resize(coarse, (size, size), interpolation=cv2.INTER_LINEAR),
            dtype=np.float32,
        )
        pixel  = np.clip(
            center + (smooth - 0.5) * 0.3 + rng.normal(0, noise_std, (size, size)),
            0.0, 1.0,
        )
        return pixel.astype(np.float32)

    def _simulate_sensor_sequences(
        self,
        grid_size: int,
        rng: np.random.Generator,
        risk_field: np.ndarray,
        date_str: str,
        seq_len: int = 24,
    ) -> np.ndarray:
        """
        Simulate N scaled sensor sequences (seq_len × 15) with feature
        values proportional to the local risk field.

        Returns raw (unscaled) array — scaling applied just before LSTM inference.
        """
        N    = grid_size * grid_size
        risk = risk_field.ravel()                  # (N,)

        # Parse date for temporal features
        try:
            dt  = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            dt  = datetime.now()

        # Pre-allocate: (N, seq_len, n_features)
        sequences = np.zeros((N, seq_len, len(FEATURE_COLS)), dtype=np.float32)

        for t in range(seq_len):
            hour       = (dt.hour + t) % 24
            hour_sin   = np.sin(2 * np.pi * hour / 24)

            # Sensor values linearly interpolated between calm and flood ranges
            # weighted by local risk, plus small timestep-specific noise
            noise_t    = rng.normal(0, 0.02, N).astype(np.float32)

            col_values = {
                "rainfall_mm":           5   + risk * 155  + rng.normal(0, 8,  N),
                "river_level_m":         0.5 + risk * 7.5  + rng.normal(0, 0.3, N),
                "soil_moisture_pct":     25  + risk * 65   + rng.normal(0, 4,  N),
                "temperature_c":         22  + rng.normal(0, 4,  N),
                "humidity_pct":          45  + risk * 50   + rng.normal(0, 5,  N),
                "wind_speed_kmh":        5   + rng.normal(0, 8,  N),
                "upstream_flow_m3s":     30  + risk * 870  + rng.normal(0, 25, N),
                "water_discharge":       10  + risk * 490  + rng.normal(0, 20, N),
                "pressure_hpa":          1015 - risk * 45  + rng.normal(0, 5,  N),
                "antecedent_rainfall_72h": 10 + risk * 200 + rng.normal(0, 15, N),
                "elevation_m":           300 - risk * 250  + rng.normal(0, 20, N),
                "slope_deg":             8   - risk * 6    + rng.normal(0, 2,  N),
                "distance_to_river_km":  5   - risk * 4.5  + rng.normal(0, 0.5, N),
                "ndvi_index":            0.7 - risk * 0.65 + rng.normal(0, 0.05, N),
                "hour_sin":              np.full(N, hour_sin, dtype=np.float32),
            }

            for fi, col in enumerate(FEATURE_COLS):
                sequences[:, t, fi] = col_values[col].astype(np.float32)

        # Clip to physically plausible ranges
        sequences[:, :, 0]  = np.clip(sequences[:, :, 0],  0,   300)   # rainfall
        sequences[:, :, 1]  = np.clip(sequences[:, :, 1],  0.1,  12)   # river level
        sequences[:, :, 2]  = np.clip(sequences[:, :, 2],  0,   100)   # soil moisture
        sequences[:, :, 3]  = np.clip(sequences[:, :, 3],  5,    45)   # temperature
        sequences[:, :, 4]  = np.clip(sequences[:, :, 4],  10,  100)   # humidity
        sequences[:, :, 5]  = np.clip(sequences[:, :, 5],  0,   120)   # wind
        sequences[:, :, 6]  = np.clip(sequences[:, :, 6],  0,  1200)   # flow
        sequences[:, :, 7]  = np.clip(sequences[:, :, 7],  0,  1500)   # discharge
        sequences[:, :, 8]  = np.clip(sequences[:, :, 8],  940, 1050)  # pressure
        sequences[:, :, 9]  = np.clip(sequences[:, :, 9],  0,   280)   # antecedent
        sequences[:, :, 10] = np.clip(sequences[:, :, 10], 5,  2000)   # elevation
        sequences[:, :, 11] = np.clip(sequences[:, :, 11], 0,    60)   # slope
        sequences[:, :, 12] = np.clip(sequences[:, :, 12], 0.01, 50)   # dist river
        sequences[:, :, 13] = np.clip(sequences[:, :, 13], -1,    1)   # ndvi
        sequences[:, :, 14] = np.clip(sequences[:, :, 14], -1,    1)   # hour_sin

        return sequences   # (N, 24, 15)  unscaled

    def _scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Apply fitted StandardScaler to (N, T, F) sequences."""
        assert self._scaler is not None, (
            "Scaler is not loaded. Call _load_models() before inference."
        )
        N, T, F = sequences.shape
        flat    = sequences.reshape(-1, F)
        scaled  = self._scaler.transform(flat).astype(np.float32)
        return scaled.reshape(N, T, F)

    # ── Inference ─────────────────────────────────────────────────────

    def _predict_probabilities(
        self,
        patches:   np.ndarray,   # (N, 64, 64, 3)
        sequences: np.ndarray,   # (N, 24, 15)  unscaled
    ) -> np.ndarray:
        """
        Extract CNN and LSTM features, run hybrid model.
        Returns (N,) float32 flood probabilities.
        """
        assert self._cnn_ext is not None and self._lstm_enc is not None and self._hybrid is not None, (
            "Models are not loaded. Call _load_models() before inference."
        )
        N = len(patches)
        scaled_seq = self._scale_sequences(sequences)

        # CNN features  (N, 128)
        cnn_feats  = self._cnn_ext.predict(
            patches, batch_size=self.batch_size, verbose=0
        )

        # LSTM features  (N, 64)
        lstm_feats = self._lstm_enc.predict(
            scaled_seq, batch_size=self.batch_size, verbose=0
        )

        # Hybrid prediction  (N, 1)
        probs = self._hybrid.predict(
            [cnn_feats, lstm_feats],
            batch_size=self.batch_size, verbose=0,
        ).ravel()

        return probs.astype(np.float32)

    # ── Output writers ────────────────────────────────────────────────

    def _save_numpy(self, prob_grid: np.ndarray, date_str: str) -> Path:
        path = self.risk_dir / f"risk_map_{date_str}.npy"
        np.save(path, prob_grid)
        return path

    def _save_matplotlib(
        self,
        prob_grid: np.ndarray,
        date_str:  str,
        grid_size: int,
    ) -> Path:
        """
        Dark-themed choropleth risk map with:
          • custom 4-band colormap
          • lat/lon tick labels
          • state centre marker + Andhra Pradesh River line
          • risk-level legend patches
          • colorbar
        """
        lats = np.linspace(LAT_MIN, LAT_MAX, grid_size)
        lons = np.linspace(LON_MIN, LON_MAX, grid_size)

        lat_grid, lon_grid = np.meshgrid(
            lats,
            lons,
            indexing="ij"
        )

        river_prox = self._river_proximity(
            lat_grid,
            lon_grid
        )

        # Build a 4-segment discrete-like colormap
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "flood_risk",
            [(0.0, "#2ecc71"), (0.3, "#f1c40f"),
             (0.6, "#e67e22"), (0.8, "#e74c3c"), (1.0, "#8e1a1a")],
        )
        norm = mcolors.Normalize(vmin=0, vmax=1)

        fig, ax = plt.subplots(figsize=(11, 9), facecolor=P["bg"])
        ax.set_facecolor(P["panel"])
        fig.set_facecolor(P["bg"])

        im = ax.imshow(
            prob_grid,
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower",
            cmap=cmap, norm=norm,
            aspect="auto", interpolation="bilinear",
        )

        # ── state centre ───────────────────────────────────────────────
        ax.plot(
                *STATE_CENTER[::-1],
                marker="*",
                markersize=12,
                color="white"
       )

        ax.annotate(
           "Andhra Pradesh",
           xy=STATE_CENTER[::-1],
           xytext=(6, 6),
           textcoords="offset points",
           fontsize=9,
           color="white",
           fontweight="bold"
        )
        # ── Andhra Pradesh River (approximate line) ────────────────────────────
        for river_lat, river_name in [
              (GODAVARI_LAT, "Godavari"),
              (KRISHNA_LAT, "Krishna"),
              (PENNA_LAT, "Penna"),
              (VAMSADHARA_LAT, "Vamsadhara"),
              ]:
            ax.axhline(
               river_lat,
               color="#58a6ff",
               linewidth=1,
               linestyle="--",
               alpha=0.6,
            )

            ax.text(
               LON_MIN + 0.2,
               river_lat + 0.05,
               river_name,
               color="cyan",
               fontsize=8
            )

        # ── Axes labels ───────────────────────────────────────────────
        ax.set_xlabel("Longitude (°E)", color=P["text"], fontsize=10)
        ax.set_ylabel("Latitude (°N)",  color=P["text"], fontsize=10)
        ax.tick_params(colors=P["sub"], labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f°E"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f°N"))
        plt.setp(ax.get_xticklabels(), color=P["sub"])
        plt.setp(ax.get_yticklabels(), color=P["sub"])

        # ── Colorbar ─────────────────────────────────────────────────
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Flood Probability", color=P["text"], fontsize=10)
        cbar.set_ticks([0.0, 0.3, 0.6, 0.8, 1.0])
        cbar.ax.yaxis.set_tick_params(color=P["sub"], labelsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=P["sub"])
        for sp in cbar.ax.spines.values():
            sp.set_edgecolor(P["border"])
        cbar.ax.set_facecolor(P["panel"])

        # ── Legend patches ────────────────────────────────────────────
        patches = [
            mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.5,
                           label=f"{name}  ({lo:.1f}–{hi:.1f})")
            for name, lo, hi, c, _ in RISK_LEVELS
        ]
        leg = ax.legend(handles=patches, loc="lower left",
                        fontsize=8.5, framealpha=0.85,
                        facecolor=P["panel"], edgecolor=P["border"],
                        labelcolor=P["text"])

        # ── Summary stats annotation ──────────────────────────────────
        n_total    = grid_size * grid_size
        n_Warning     = int((prob_grid >= 0.6).sum())
        n_Emergency  = int((prob_grid >= 0.8).sum())
        stats_text = (
            f"Mean risk : {prob_grid.mean():.3f}\n"
            f"Max risk  : {prob_grid.max():.3f}\n"
            f"Warning+Ext  : {(n_Warning/n_total)*100:.1f}%\n"
            f"Emergency   : {(n_Emergency/n_total)*100:.1f}%"
        )
        ax.text(0.01, 0.99, stats_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=8.5,
                color=P["text"], fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=P["panel"],
                          edgecolor=P["border"], alpha=0.90))

        # ── Title ─────────────────────────────────────────────────────
        ax.set_title(
            f"Flood Risk Map — Andhra Pradesh Region\n{date_str}  "
            f"({grid_size}×{grid_size} grid | {n_total:,} cells)",
            color=P["text"], fontsize=12, fontweight="bold", pad=12,
        )

        plt.tight_layout()
        path = self.risk_dir / f"risk_map_{date_str}.png"
        fig.savefig(str(path), dpi=130, bbox_inches="tight", facecolor=P["bg"])
        plt.close(fig)
        return path

    def _save_folium(
        self,
        prob_grid: np.ndarray,
        date_str:  str,
        grid_size: int,
    ) -> Path:
        """
        Interactive folium map with:
          • semi-transparent GeoJSON choropleth for every cell
          • CircleMarker for every Warning and Emergency cell (popups)
          • HeatMap overlay (toggleable layer)
          • Layer control, legend HTML, fullscreen plugin
        """
        lats     = np.linspace(LAT_MIN, LAT_MAX, grid_size + 1)
        lons     = np.linspace(LON_MIN, LON_MAX, grid_size + 1)
        cell_lat = np.linspace(LAT_MIN + (LAT_MAX-LAT_MIN)/(2*grid_size),
                               LAT_MAX - (LAT_MAX-LAT_MIN)/(2*grid_size),
                               grid_size)
        cell_lon = np.linspace(LON_MIN + (LON_MAX-LON_MIN)/(2*grid_size),
                               LON_MAX - (LON_MAX-LON_MIN)/(2*grid_size),
                               grid_size)

        # ── Base map ──────────────────────────────────────────────────
        m = folium.Map(
            location   = list(STATE_CENTER),
            zoom_start = 7,
            tiles      = "CartoDB dark_matter",
        )

        # ── GeoJSON choropleth layer ─────────────────────────────────
        def risk_colour(p: float) -> str:
            for _, lo, hi, col, _ in RISK_LEVELS:
                if lo <= p < hi:
                    return col
            return RISK_LEVELS[-1][3]

        geojson_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                p   = float(prob_grid[i, j])
                lvl = next(n for n, lo, hi, *_ in RISK_LEVELS if lo <= p < hi
                           ) if p < 1.0 else "Emergency"
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lons[j],   lats[i]],
                            [lons[j+1], lats[i]],
                            [lons[j+1], lats[i+1]],
                            [lons[j],   lats[i+1]],
                            [lons[j],   lats[i]],
                        ]],
                    },
                    "properties": {
                        "risk":  round(p, 4),
                        "level": lvl,
                        "color": risk_colour(p),
                        "lat":   round(float(cell_lat[i]), 4),
                        "lon":   round(float(cell_lon[j]),  4),
                    },
                })

        geojson_layer = folium.FeatureGroup(name="Risk Grid", show=True)
        for feat in geojson_features:
            props = feat["properties"]
            folium.GeoJson(
                feat,
                style_function=lambda f, c=props["color"]: {
                    "fillColor":   c,
                    "color":       "none",
                    "fillOpacity": 0.55,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["level", "risk", "lat", "lon"],
                    aliases=["Risk Level:", "Probability:", "Lat:", "Lon:"],
                    localize=True,
                ),
            ).add_to(geojson_layer)
        geojson_layer.add_to(m)

        # ── HeatMap layer ────────────────────────────────────────────
        heat_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                p = float(prob_grid[grid_size - 1 - i, j])
                if p > 0.1:
                    heat_data.append([float(cell_lat[grid_size-1-i]),
                                      float(cell_lon[j]), p])
        heat_layer = folium.FeatureGroup(name="Heat Map", show=False)
        HeatMap(
            heat_data,
            min_opacity=0.2, max_opacity=0.8,
            radius=18, blur=15,
            gradient={"0.3": "#2ecc71", "0.6": "#f1c40f",
                      "0.8": "#e67e22", "1.0": "#e74c3c"},
        ).add_to(heat_layer)
        heat_layer.add_to(m)

        # ── CircleMarkers for Warning + Emergency cells ────────────────────
        alert_layer = folium.FeatureGroup(name="⚠ Warning/Emergency Alerts", show=True)
        for i in range(grid_size):
            for j in range(grid_size):
                p = float(prob_grid[grid_size - 1 - i, j])
                if p >= 0.6:
                    _, _, _, col, dark_col = next(
                        r for r in RISK_LEVELS if r[1] <= p < r[2]
                    ) if p < 1.0 else RISK_LEVELS[-1]
                    lvl = next(r[0] for r in RISK_LEVELS if r[1] <= p < r[2]
                               ) if p < 1.0 else "Emergency"
                    lat_c = float(cell_lat[grid_size - 1 - i])
                    lon_c = float(cell_lon[j])
                    folium.CircleMarker(
                        location  = [lat_c, lon_c],
                        radius    = 5 if p >= 0.8 else 3,
                        color     = dark_col,
                        fill      = True,
                        fill_color= col,
                        fill_opacity = 0.85,
                        popup     = folium.Popup(
                            f"<b>{lvl} Risk</b><br>"
                            f"Probability: {p:.3f}<br>"
                            f"Lat: {lat_c:.4f}°N<br>"
                            f"Lon: {lon_c:.4f}°E",
                            max_width="200",
                        ),
                        tooltip   = f"{lvl}: {p:.3f}",
                    ).add_to(alert_layer)
        alert_layer.add_to(m)

        # ── state centre marker ─────────────────────────────────────────
        folium.Marker(
            location = list(STATE_CENTER),
            popup    = folium.Popup("<b>Andhra Pradesh State Centre</b>", max_width="150"),
            tooltip  = "Andhra Pradesh",
            icon     = folium.Icon(color="blue", icon="map-marker", prefix="fa"),
        ).add_to(m)

        # ── Andhra Pradesh Rivers ──────────────────────────────────────

        GODAVARI_PATH = [
           [17.0, 81.8],
           [17.2, 81.7],
           [17.4, 81.6]
       ]

        KRISHNA_PATH = [
           [16.2, 80.5],
           [16.4, 80.7],
           [16.6, 80.9]
       ]

        PENNA_PATH = [
           [14.3, 79.8],
           [14.6, 80.0],
           [14.8, 80.2]
       ]

        VAMSADHARA_PATH = [
           [18.5, 83.6],
           [18.8, 84.0]
       ]
        folium.PolyLine(
            GODAVARI_PATH,
            color="blue",
            weight=3,
            tooltip="Godavari River"
        ).add_to(m)

        folium.PolyLine(
            KRISHNA_PATH,
            color="cyan",
            weight=3,
            tooltip="Krishna River"
        ).add_to(m)

        folium.PolyLine(
            PENNA_PATH,
            color="green",
            weight=3,
            tooltip="Penna River"
        ).add_to(m)

        folium.PolyLine(
            VAMSADHARA_PATH,
            color="purple",
            weight=3,
            tooltip="Vamsadhara River"
        ).add_to(m)

        # ── Legend HTML ───────────────────────────────────────────────
        legend_html = f"""
        <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                    background:#1a1a2e;border:1px solid #30363d;
                    border-radius:8px;padding:14px 18px;
                    font-family:monospace;font-size:12px;color:#e6edf3;">
          <b>Flood Risk Levels</b><br>
          <span style="color:#2ecc71">██</span>  Normal (0.0–0.3)<br>
          <span style="color:#f1c40f">██</span>  Watch (0.3–0.6)<br>
          <span style="color:#e67e22">██</span>  Warning (0.6–0.8)<br>
          <span style="color:#e74c3c">██</span>  Emergency (0.8–1.0)<br>
          <hr style="border-color:#30363d;margin:6px 0">
          <span style="font-size:10px;color:#8b949e">{date_str}</span>
        </div>
        """
        # Avoid accessing nonexistent 'html' attribute on the root; add the
        # legend Element directly to the root so static analyzers won't flag it.
        m.get_root().add_child(folium.Element(legend_html))

        # ── Layer control ─────────────────────────────────────────────
        folium.LayerControl(collapsed=False).add_to(m)

        path = self.risk_dir / f"interactive_map_{date_str}.html"
        m.save(str(path))
        return path

    def _save_json_summary(
        self,
        prob_grid: np.ndarray,
        date_str:  str,
        grid_size: int,
        elapsed:   float,
    ) -> Path:
        n_total = grid_size * grid_size
        counts  = {}
        for name, lo, hi, *_ in RISK_LEVELS:
            mask       = (prob_grid >= lo) & (prob_grid < hi)
            counts[name] = int(mask.sum())

        Warning_cells = [
            {
                "lat": round(LAT_MIN + (i + 0.5) * (LAT_MAX - LAT_MIN) / grid_size, 4),
                "lon": round(LON_MIN + (j + 0.5) * (LON_MAX - LON_MIN) / grid_size, 4),
                "risk": round(float(prob_grid[i, j]), 4),
                "level": next(n for n, lo, hi, *_ in RISK_LEVELS
                              if lo <= prob_grid[i, j] < hi)
                         if prob_grid[i, j] < 1.0 else "Emergency",
            }
            for i in range(grid_size)
            for j in range(grid_size)
            if prob_grid[i, j] >= 0.6
        ]
        Warning_cells.sort(key=lambda x: -x["risk"])

        # Percentage of affected cells
        affected_pct = (
            counts["Warning"] + counts["Emergency"]
        ) / n_total * 100

        emergency_pct = counts["Emergency"] / n_total * 100
        watch_pct = counts["Watch"] / n_total * 100

        # Consistent alert logic
        if affected_pct >= 15:
           alert_level = "Emergency"
        elif affected_pct >= 8:
           alert_level = "Warning"
        elif affected_pct >= 3:
           alert_level = "Watch"
        else:
           alert_level = "Normal"

        summary = {
            "date":            date_str,
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "region": {
                "lat_min": LAT_MIN, "lat_max": LAT_MAX,
                "lon_min": LON_MIN, "lon_max": LON_MAX,
                "state":    "Andhra Pradesh, India",
            },
            "grid_size":       grid_size,
            "n_cells":         n_total,
            "inference_time_s": round(elapsed, 2),
            "statistics": {
                "mean_risk":       round(float(prob_grid.mean()), 4),
                "max_risk":        round(float(prob_grid.max()),  4),
                "min_risk":        round(float(prob_grid.min()),  4),
                "std_risk":        round(float(prob_grid.std()),  4),
                "pct_Warning_Emergency": round(
                       ((counts["Warning"] + counts["Emergency"]) / n_total) * 100, 2
                    ),
                "pct_Emergency": round(
                       counts["Emergency"] / n_total * 100, 2
                    ),
            },
            "risk_level_counts": counts,
            "risk_level_pct": {
                name: round(cnt / n_total * 100, 2)
                for name, cnt in counts.items()
            },
            "top_Warning_risk_cells": Warning_cells[:20],
            "alert_level": alert_level,
        }

        path = self.pred_dir / f"risk_summary_{date_str}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    # ── Public API ────────────────────────────────────────────────────

    def generate_risk_map(
        self,
        date_str:  str,
        grid_size: int  = 50,
    ) -> dict:
        """
        Generate all four risk-map outputs for a given date.

        Parameters
        ----------
        date_str  : ISO date string, e.g. "2024-07-15"
        grid_size : spatial grid resolution (grid_size × grid_size cells)

        Returns
        -------
        dict with keys: prob_grid, paths (npy/png/html/json), summary
        """
        self._load_models()

        SEP = "─" * 60
        print(f"\n{'═'*60}")
        print(f"  Flood Risk Map Generator — {date_str}")
        print(f"  Grid: {grid_size}×{grid_size} = {grid_size**2:,} cells"
              f"  |  Region: Andhra Pradesh")
        print(f"{'═'*60}")

        # ── Derive date-seeded spatial risk field ─────────────────────
        date_seed = int(date_str.replace("-", "")) % (2**31)
        rng       = np.random.default_rng(date_seed)

        print(f"\n[1/5]  Generating spatially coherent risk field …")
        t0 = time.perf_counter()

        # Multi-scale risk field: river proximity + large + fine spatial variation
        lats = np.linspace(LAT_MIN, LAT_MAX, grid_size)
        lons = np.linspace(LON_MIN, LON_MAX, grid_size)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        river_prox = self._river_proximity(lat_grid, lon_grid)

        field_large = self._spatial_field(grid_size, rng, smooth=6.0)
        field_fine  = self._spatial_field(grid_size, rng, smooth=2.0)

        # Blend: 50% river proximity, 35% large-scale, 15% fine-scale
        risk_field  = np.clip(
            0.50 * river_prox + 0.35 * field_large + 0.15 * field_fine,
            0.0, 1.0,
        ).astype(np.float32)

        # Normalise so some cells always reach Warning risk in rainy season
        season_boost = 0.15 * np.sin(2 * np.pi * (datetime.strptime(date_str, "%Y-%m-%d").month - 6) / 12)
        risk_field   = np.clip(risk_field + season_boost, 0.0, 1.0)

        print(f"       Risk field  mean={risk_field.mean():.3f}"
              f"  max={risk_field.max():.3f}  ({time.perf_counter()-t0:.2f}s)")

        # ── Simulate input data ───────────────────────────────────────
        N = grid_size * grid_size
        print(f"\n[2/5]  Simulating {N:,} satellite patches …", end=" ", flush=True)
        t1 = time.perf_counter()
        patches   = self._simulate_satellite_patches(grid_size, rng, risk_field)
        print(f"done  ({time.perf_counter()-t1:.2f}s)")

        print(f"[3/5]  Simulating {N:,} sensor sequences …", end=" ", flush=True)
        t2 = time.perf_counter()
        sequences = self._simulate_sensor_sequences(
            grid_size, rng, risk_field, date_str
        )
        print(f"done  ({time.perf_counter()-t2:.2f}s)")

        # ── Hybrid inference ──────────────────────────────────────────
        print(f"[4/5]  Running hybrid model inference on {N:,} cells …",
              end=" ", flush=True)
        t3 = time.perf_counter()
        probs    = self._predict_probabilities(patches, sequences)
        infer_t  = time.perf_counter() - t3
        prob_grid = probs.reshape(grid_size, grid_size)
        print(f"done  ({infer_t:.2f}s | {N/infer_t:.0f} cells/s)")
        print(f"       Probabilities  mean={prob_grid.mean():.3f}"
              f"  max={prob_grid.max():.3f}")

        # ── Save outputs ──────────────────────────────────────────────
        print(f"\n[5/5]  Saving outputs …")
        t4 = time.perf_counter()

        p_npy  = self._save_numpy(prob_grid, date_str)
        print(f"  ✓  Numpy array      → {p_npy}")

        p_png  = self._save_matplotlib(prob_grid, date_str, grid_size)
        print(f"  ✓  Risk map (PNG)   → {p_png}")

        p_html = self._save_folium(prob_grid, date_str, grid_size)
        print(f"  ✓  Interactive map  → {p_html}")

        p_json = self._save_json_summary(prob_grid, date_str, grid_size, infer_t)
        print(f"  ✓  Summary JSON     → {p_json}")

        # ── Load summary for return ───────────────────────────────────
        with open(p_json) as f:
            summary = json.load(f)

        total_t = time.perf_counter() - t0
        print(f"\n{'═'*60}")
        print(f"  Date         : {date_str}")
        print(f"  Mean risk    : {prob_grid.mean():.4f}")
        print(f"  Max risk     : {prob_grid.max():.4f}")
        print(f"  Alert level  : {summary['alert_level']}")
        print(f"  Warning+Emergency : {summary['statistics']['pct_Warning_Emergency']}%"
              f"  ({summary['risk_level_counts'].get('Warning',0) + summary['risk_level_counts'].get('Emergency',0)} cells)")
        print(f"  Total time   : {total_t:.1f}s")
        print(f"{'═'*60}\n")

        return {
            "date":      date_str,
            "prob_grid": prob_grid,
            "summary":   summary,
            "paths": {
                "npy":  str(p_npy),
                "png":  str(p_png),
                "html": str(p_html),
                "json": str(p_json),
            },
        }

    def generate_historical_maps(
        self,
        n_days:    int = 7,
        grid_size: int = 50,
        end_date:  str | None = None,
    ) -> list[dict]:
        """
        Generate risk maps for the last `n_days` dates ending at `end_date`
        (defaults to today), suitable for dashboard animation.

        Parameters
        ----------
        n_days    : number of daily maps to generate
        grid_size : spatial resolution
        end_date  : ISO date string for the last day; None = today

        Returns
        -------
        List of result dicts from generate_risk_map(), oldest first.
        """
        if end_date is None:
            base = datetime.utcnow().date()
        else:
            base = datetime.strptime(end_date, "%Y-%m-%d").date() 

        dates   = [(base - timedelta(days=n_days - 1 - i)) for i in range(n_days)]
        results = []

        print(f"\n{'═'*60}")
        print(f"  Historical Risk Maps  —  {n_days} days")
        print(f"  {dates[0]} → {dates[-1]}")
        print(f"{'═'*60}")

        for idx, d in enumerate(dates, 1):
            date_str = d.strftime("%Y-%m-%d")
            print(f"\n  [{idx}/{n_days}]  Processing {date_str} …")
            result = self.generate_risk_map(date_str, grid_size=grid_size)
            results.append(result)
        # ── Combined summary ──────────────────────────────────────────
        # State-wide average risk
        mean_risks = [
            r["summary"]["statistics"]["mean_risk"]
            for r in results
        ]

        # Percentage of cells in Warning or Emergency categories
        warning_pct = [
            r["summary"]["statistics"]["pct_Warning_Emergency"]
            for r in results
        ]

        # Peak flood day = largest affected area
        peak_idx = int(np.argmax(warning_pct)) if len(warning_pct) > 0 else 0

        print(f"\n{'═'*60}")
        print(f"  HISTORICAL SUMMARY  ({n_days} days)")
        print(f"  {'Date':<14}  {'Mean':>7}  {'Alert':<10}  Warning+Ext%")
        print(f"  {'─'*56}")
        for r in results:
            s    = r["summary"]
            star = " ◄ PEAK" if r["date"] == results[peak_idx]["date"] else ""
            print(f"  {r['date']:<14}  "
                  f"{s['statistics']['mean_risk']:>7.4f}  "
                  f"{s['alert_level']:<10}  "
                  f"{s['statistics']['pct_Warning_Emergency']:>5.1f}%{star}")
        print(f"  {'─'*56}")
        print(f"  {f'{n_days}-day mean':<14}  {np.mean(mean_risks):>7.4f}  "
              f"{np.mean(warning_pct):>6.2f}")
        print(f"{'═'*60}\n")

        return results


# ════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import tensorflow as tf
    import numpy as np

    tf.random.set_seed(42)
    np.random.seed(42)

    gen = FloodRiskMapGenerator()

    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Historical maps ending on chosen date
    gen.generate_historical_maps(
        n_days=7,
        end_date=date_str
    )