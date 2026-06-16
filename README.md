# 🌊 Real-Time Flood Monitoring System

A production-ready **Hybrid CNN + LSTM** deep learning pipeline for satellite-based flood detection and hydrological time-series forecasting. The system ingests Sentinel-1 SAR imagery, Sentinel-2 multispectral tiles, and ground-based sensor streams to produce per-pixel flood risk maps and 24-hour ahead probabilistic flood forecasts.

---

## Architecture

```
                    ┌────────────────────────────────────┐
  Satellite Image ──►  CNN Spatial Encoder (U-Net)        │
  (256 × 256 × 7)  │  Red · Green · Blue · NIR · SWIR   ├──► Spatial Feature Vec (256-D)
                   │  DEM · Slope                         │              │
                   └────────────────────────────────────┘              │
                                                                        ▼
                   ┌────────────────────────────────────┐    ┌─────────────────────┐
  Sensor Stream ───►  BiLSTM Temporal Encoder            │    │   Feature Fusion    │
  (72 h × 6 ch)   │  Rainfall · River Level             ├──► │  Concatenation +    ├──► Flood Risk Score
                   │  Soil Moisture · Flow               │    │  Dense Head         │    + Flood Risk Map
                   │  Temperature · Humidity             │    └─────────────────────┘
                   └────────────────────────────────────┘
```

### Spatial Branch – `CNNFloodDetector`
A U-Net encoder-decoder with skip connections performs **semantic segmentation** of flooded vs non-flooded pixels. Trained with a combined Dice + BCE loss to handle severe class imbalance (flooded pixels are rare).

**Input bands:**

| Index | Band | Source |
|-------|------|--------|
| 0 | Red (B4) | Sentinel-2 |
| 1 | Green (B3) | Sentinel-2 |
| 2 | Blue (B2) | Sentinel-2 |
| 3 | NIR (B8) | Sentinel-2 |
| 4 | SWIR (B11) | Sentinel-2 |
| 5 | DEM (elevation m) | SRTM |
| 6 | Slope (degrees) | SRTM |

### Temporal Branch – `LSTMFloodPredictor`
A stacked **Bidirectional LSTM with self-attention** processes a 72-hour sliding window of six sensor channels and forecasts flood probability for each of the next 24 hours.

**Sensor channels:** rainfall (mm), river level (m), soil moisture (%), upstream flow (m³/s), temperature (°C), humidity (%)

### Hybrid Model – `HybridCNNLSTM`
The two branches are fused via concatenation and a two-layer dense head, producing:
1. `flood_risk_score` – scalar [0, 1] for the monitoring region
2. `spatial_flood_map` – (H × W × 1) pixel-wise probability map

---

## Project Structure

```
flood_monitoring/
├── data/
│   ├── raw/                 ← Raw sensor CSVs and downloaded assets
│   ├── processed/           ← Normalised tensors and feature tables
│   └── satellite_images/    ← GeoTIFF tiles (Sentinel-1/2, DEM)
├── models/
│   ├── __init__.py
│   ├── cnn_model.py         ← U-Net semantic segmentation (spatial)
│   ├── lstm_model.py        ← Bidirectional LSTM forecaster (temporal)
│   └── hybrid_model.py      ← End-to-end fusion model
├── outputs/
│   ├── flood_risk_maps/     ← Generated GeoJSON / PNG risk maps
│   ├── predictions/         ← Saved model checkpoints and .h5 files
│   └── reports/             ← Alert logs, system log, JSON summaries
├── dashboard/
│   ├── __init__.py
│   ├── app.py               ← Flask server + Dash interactive UI
│   └── templates/
│       └── index.html       ← Landing page
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       ← GeoTIFF, CSV, and GEE ingestion
│   ├── preprocessor.py      ← Feature engineering and windowing
│   └── alert_system.py      ← Multi-level alert dispatch (email, webhook)
├── requirements.txt
└── main.py                  ← CLI entrypoint
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine (optional, for live satellite data)

```bash
earthengine authenticate
```

### 3. Train the hybrid model

```bash
python main.py train --epochs 50 --batch-size 8 --lr 5e-4
```

### 4. Run inference for a station

```bash
python main.py predict --station G-04
```

### 5. Launch the dashboard

```bash
python main.py serve --port 8050
```

Then open `http://localhost:8050` in your browser.

### 6. Full pipeline (ingest → predict → alert)

```bash
python main.py pipeline --station G-04
```

---

## Alert Levels

| Level | Risk Score | Action |
|-------|-----------|--------|
| 🟢 LOW | < 0.40 | No action required |
| 🟡 MODERATE | 0.40 – 0.65 | Monitoring increased |
| 🔴 HIGH | 0.65 – 0.85 | Authorities notified |
| 🟣 CRITICAL | > 0.85 | Emergency evacuation advisory |

Alerts are written to `outputs/reports/alert_log.jsonl` and can be dispatched via SMTP email or HTTP webhook by configuring `AlertSystem`.

---

## Spectral Indices

The preprocessor computes the following water-detection indices from raw spectral bands:

| Index | Formula | Water Indicator |
|-------|---------|----------------|
| NDWI | (Green − NIR) / (Green + NIR) | > 0 → open water |
| MNDWI | (Green − SWIR) / (Green + SWIR) | > 0 → water (urban robust) |
| NDVI | (NIR − Red) / (NIR + Red) | < 0 → non-vegetated / water |

---

## Dashboard

The Dash web application auto-refreshes every 30 seconds and displays:

- **Flood Risk Map** – interactive Mapbox scatter-geo of all monitoring stations coloured by current risk score
- **24-Hour Forecast** – time-series of predicted flood probability with HIGH / CRITICAL threshold lines
- **Sensor Readings (72 h)** – dual-axis chart of hourly rainfall and river level
- **Alert History** – tabular log of all recent alerts with acknowledgement status

REST endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Landing page |
| GET | `/dashboard/` | Dash interactive dashboard |
| GET | `/api/status` | System health JSON |
| GET | `/api/alerts` | Last 50 alert records |
| POST | `/api/predict` | On-demand inference (JSON body) |

---

## Data Sources

| Source | Product | Resolution | Access |
|--------|---------|-----------|--------|
| ESA Copernicus | Sentinel-1 SAR GRD | 10 m | Google Earth Engine |
| ESA Copernicus | Sentinel-2 L2A | 10 m | Google Earth Engine |
| NASA / JAXA | SRTM DEM | 30 m | Google Earth Engine |
| Ground sensors | Telemetry CSV | 1-hour | HDMC / CWC API |

---

## Configuration

Key parameters can be tuned in each module's constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_shape` | (256, 256, 7) | CNN input tile dimensions |
| `window_size` | 72 | Hours of sensor history for LSTM |
| `n_ahead` | 24 | Hours forecast horizon |
| `cnn_filters` | 64 | Base filter count (U-Net) |
| `lstm_units` | (128, 64) | BiLSTM hidden dimensions |
| `dropout_rate` | 0.3 | Dropout regularisation |

---

## Extending the System

- **New sensor streams** – add columns to `Preprocessor.FEATURE_COLS` and retrain
- **Additional satellites** – extend `DataLoader.load_geotiff()` for Landsat or MODIS tiles
- **Custom alert channels** – subclass `AlertSystem` and override `dispatch()`
- **Deployment** – serve with `gunicorn "dashboard.app:server"` behind an Nginx reverse proxy

---

## References

- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- McFeeters, S.K. (1996). *The use of the Normalised Difference Water Index (NDWI) in the delineation of open water features*
- Xu, H. (2006). *Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery*
- Copernicus Emergency Management Service – [https://emergency.copernicus.eu](https://emergency.copernicus.eu)

---

## License

MIT License. See `LICENSE` for details.
