# 🌊 AP Flood EWS — Andhra Pradesh Flood Early Warning System

A production-ready **Hybrid CNN + LSTM** deep learning pipeline for satellite-based flood
detection and hydrological time-series forecasting across Andhra Pradesh's major river
basins — Godavari, Krishna, Penna, and Vamsadhara.

---

## Architecture

```
                   ┌────────────────────────────────────┐
 Satellite Image ──►  CNN Spatial Encoder (U-Net)        │
 (256 × 256 × 7)  │  Red · Green · Blue · NIR · SWIR   ├──► Spatial Feature Vec (256-D)
                   │  DEM · Slope                        │              │
                   └────────────────────────────────────┘              │
                                                                        ▼
                   ┌────────────────────────────────────┐    ┌─────────────────────┐
 Sensor Stream ───►  BiLSTM Temporal Encoder            │    │   Feature Fusion    │
 (72 h × 6 ch)    │  Rainfall · River Level            ├──► │  Concatenation +    ├──► Flood Risk Score
                   │  Soil Moisture · Flow              │    │  Dense Head         │  + Flood Risk Map
                   │  Temperature · Humidity            │    └─────────────────────┘
                   └────────────────────────────────────┘
```

### Spatial Branch — `CNNFloodDetector`
A U-Net encoder-decoder with skip connections performs **semantic segmentation** of flooded
vs non-flooded pixels. Trained with a combined Dice + BCE loss to handle severe class
imbalance (flooded pixels are rare).

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

### Temporal Branch — `LSTMFloodPredictor`
A stacked **Bidirectional LSTM with self-attention** processes a 72-hour sliding window of
six sensor channels and forecasts flood probability for each of the next 24 hours.

**Sensor channels:** rainfall (mm), river level (m), soil moisture (%), upstream flow
(m³/s), temperature (°C), humidity (%)

### Hybrid Model — `HybridCNNLSTM`
The two branches are fused via concatenation and a two-layer dense head, producing:
1. `flood_risk_score` — scalar [0, 1] for the monitoring region
2. `spatial_flood_map` — (H × W × 1) pixel-wise probability map

---

## Project Structure

```
ap-flood-ews/
├── data/
│   └── satellite_images/     ← Sample flood/no-flood image patches (PNG)
├── models/
│   ├── cnn_model.py          ← U-Net semantic segmentation (spatial branch)
│   ├── lstm_model.py         ← Bidirectional LSTM forecaster (temporal branch)
│   └── hybrid_model.py       ← End-to-end fusion model
├── outputs/
│   ├── flood_risk_maps/      ← Generated PNG risk maps (daily)
│   ├── predictions/          ← JSON risk summaries (daily)
│   └── reports/              ← Confusion matrices, training plots, metrics JSON
├── dashboard/
│   └── app.py                ← Plotly Dash command-centre + Flask API endpoints
├── utils/
│   ├── alert_system.py       ← Multi-level SMS alert dispatch (Twilio / Fast2SMS)
│   ├── constants.py          ← AP river basin definitions and risk thresholds
│   ├── data_generator.py     ← Synthetic sensor data generation
│   ├── preprocessor.py       ← Feature engineering and windowing
│   ├── risk_map_generator.py ← Folium / matplotlib risk map rendering
│   └── satellite_generator.py← Synthetic satellite image patch generation
├── requirements.txt
└── main.py                   ← CLI entrypoint
```

---

## ⚠️ Model Files — Must Regenerate After Cloning

Trained model weights (`.keras`) and preprocessed data arrays (`.npy`, `.npz`) are
**not included in this repository** (too large for GitHub). Run the training step first
before launching inference or the dashboard:

```bash
python main.py train   # generates .keras weights + processed data
python main.py predict # then run inference
```

The sample `outputs/reports/` (confusion matrices, training plots, metrics JSON) **are**
committed so you can review model performance without retraining.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Hari-Cheduri/ap-flood-ews.git
cd ap-flood-ews

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the hybrid model

```bash
python main.py train
```

Generates synthetic AP river basin sensor data, trains the CNN, LSTM, and hybrid fusion
model, and saves checkpoints to `outputs/`.

### 3. Run batch inference

```bash
python main.py predict
```

Outputs daily risk summaries to `outputs/predictions/` and PNG risk maps to
`outputs/flood_risk_maps/`.

### 4. Launch the dashboard

```bash
python main.py serve
```

Open `http://localhost:8050` in your browser to access the live command-centre dashboard.

### 5. Full pipeline (train → predict → alert)

```bash
python main.py pipeline
```

Runs all stages end-to-end: data generation, training, inference, risk mapping, and alert
dispatch.

---

## Alert Levels

| Level | Risk Score | Action |
|-------|-----------|--------|
| 🟢 LOW | < 0.40 | No action required |
| 🟡 MODERATE | 0.40 – 0.65 | Monitoring increased |
| 🔴 HIGH | 0.65 – 0.85 | Authorities notified |
| 🟣 CRITICAL | > 0.85 | Emergency evacuation advisory |

---

## SMS Alert System

Real-time SMS alerts are dispatched via **Twilio** or **Fast2SMS** whenever risk scores
cross the HIGH or CRITICAL threshold.

**To enable live SMS:**

1. Create your credentials file:
   ```bash
   # Windows PowerShell
   New-Item -Path config -ItemType Directory -Force
   New-Item -Path config\sms_config.json -ItemType File
   ```
2. Add your API credentials (Twilio Account SID + Auth Token, or Fast2SMS API key)
3. Refer to `SMS_SETUP_GUIDE.md` for full setup including TRAI/DLT template compliance

**Without credentials**, the system runs in **mock mode** — alerts are printed to the
console and logged to `outputs/sms_log.csv` but not dispatched.

> ⚠️ `config/sms_config.json` is listed in `.gitignore` — never commit credentials.

---

## Dashboard

The Plotly Dash command-centre (`dashboard/app.py`) auto-refreshes every 30 seconds:

- **Flood Risk Map** — interactive Folium map of AP monitoring stations coloured by risk score
- **24-Hour Forecast** — predicted flood probability with HIGH/CRITICAL threshold lines
- **Sensor Readings (72 h)** — dual-axis chart of hourly rainfall and river level
- **Alert History** — tabular log of all dispatched alerts
- **SMS Alert Panel** — send and manage SMS alerts directly from the UI

**Flask REST endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | System health JSON |
| GET | `/api/alerts` | Last 50 alert records |
| POST | `/api/predict` | On-demand inference (JSON body) |
| POST | `/api/sms` | Trigger manual SMS alert |

---

## Geographic Scope

Covers Andhra Pradesh's four major river basins with AP-specific flood zone definitions
and dual-monsoon (Southwest + Northeast) seasonal risk multipliers:

| Basin | Key Districts Monitored |
|-------|------------------------|
| Godavari | East Godavari, West Godavari, Eluru |
| Krishna | Krishna, NTR District, Guntur |
| Penna | Nellore, Prakasam |
| Vamsadhara | Srikakulam, Vizianagaram |

---

## Spectral Indices

The preprocessor computes the following water-detection indices from raw spectral bands:

| Index | Formula | Water Indicator |
|-------|---------|----------------|
| NDWI | (Green − NIR) / (Green + NIR) | > 0 → open water |
| MNDWI | (Green − SWIR) / (Green + SWIR) | > 0 → water (urban robust) |
| NDVI | (NIR − Red) / (NIR + Red) | < 0 → non-vegetated / water |

---

## Configuration

Key parameters can be tuned in each module's constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_shape` | (256, 256, 7) | CNN input tile dimensions |
| `window_size` | 72 | Hours of sensor history for LSTM |
| `n_ahead` | 24 | Forecast horizon (hours) |
| `cnn_filters` | 64 | Base filter count (U-Net) |
| `lstm_units` | (128, 64) | BiLSTM hidden dimensions |
| `dropout_rate` | 0.3 | Dropout regularisation |

---

## References

- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- McFeeters, S.K. (1996). *The use of the Normalised Difference Water Index (NDWI) in the delineation of open water features*
- Xu, H. (2006). *Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery*
- Copernicus Emergency Management Service — https://emergency.copernicus.eu

---

## License

MIT License. See `LICENSE` for details.