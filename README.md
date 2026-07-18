# Andhra Pradesh Flood Early Warning System

A district-level academic flood early-warning prototype that combines:

- **Sentinel-1 SAR satellite imagery**
- **Hourly weather sequences**
- **A CNN model**
- **An LSTM model**
- **Freshness-aware decision-level fusion**
- **A 26-district Andhra Pradesh risk dashboard**
- **A controlled SMS alert workflow**

> **Important:** This project is an academic and research prototype. It is not an official government warning system and must not be used alone for evacuation, emergency-response, or public-safety decisions.

---

## 1. Project overview

The system analyses two independent sources of evidence:

1. A **CNN** analyses a recent Sentinel-1 radar patch and produces a flood-like surface score.
2. An **LSTM** analyses the previous 72 hours of weather and estimates next-24-hour heavy-weather risk.
3. A transparent hybrid layer combines both scores.
4. The final result is assigned one of four levels:

| Level | Meaning |
|---|---|
| GREEN | No strong combined flood signal |
| YELLOW | Possible or borderline risk; continue monitoring |
| ORANGE | Meaningful combined evidence; verify district conditions |
| RED | Strong usable satellite and weather evidence; urgent official verification |

The system processes representative coordinates for all 26 Andhra Pradesh districts and generates district-level JSON, CSV, GeoJSON, maps, alert logs, and a Dash dashboard.

---

## 2. Current project status

| Component | Status |
|---|---|
| Real Sentinel-1 fetch through Google Earth Engine | Complete |
| CNN training and evaluation | Complete |
| Live CNN prediction | Complete |
| Historical weather dataset | Complete |
| LSTM training and evaluation | Complete |
| Live 72-hour weather prediction | Complete |
| Freshness-aware CNN/LSTM fusion | Complete |
| Single-location live pipeline | Complete |
| All 26 AP district processing | Complete |
| District result archiving | Complete |
| JSON, CSV, and GeoJSON generation | Complete |
| Interactive and static risk maps | Complete |
| Alert system | Complete |
| Dash dashboard | Complete |
| Mock SMS workflow | Complete |
| Real SMS credentials | Optional deployment step |
| Final report and presentation | Project documentation step |

---

## 3. High-level architecture

```text
                         AP FLOOD EARLY WARNING SYSTEM
                                      |
                       +--------------+--------------+
                       |                             |
              SATELLITE PIPELINE             WEATHER PIPELINE
                       |                             |
            District latitude/longitude   District latitude/longitude
                       |                             |
             Google Earth Engine              Open-Meteo API
                       |                             |
             Recent Sentinel-1 SAR        Latest 72 hourly records
                       |                             |
              64 x 64 x 3 patch            Six weather features
                       |                             |
                  CNN model                    RobustScaler
                       |                             |
          Flood-like surface evidence          LSTM model
                       |                             |
                       +--------------+--------------+
                                      |
                         Freshness-aware fusion
                                      |
                         GREEN / YELLOW / ORANGE / RED
                                      |
                  District archive + JSON + CSV + GeoJSON
                                      |
                       Risk map + alert system + dashboard
                                      |
                       Controlled mock or real SMS gateway
```

---

## 4. Models

### 4.1 CNN

The CNN accepts:

```text
Input shape: (64, 64, 3)
Data type: float32
Value range: 0.0 to 1.0
```

The current Sentinel-1 visual contract represents:

1. VV radar response
2. VH radar response
3. VV minus VH

The CNN output is interpreted as a **flood-like surface score**, not a guaranteed real-world flood probability.

Current stored model:

```text
models/real_cnn_model.h5
```

Associated files:

```text
models/cnn_threshold.json
models/cnn_metrics.json
models/cnn_history.json
models/cnn_split_indices.npz
```

Current recorded test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 91.30% |
| Balanced accuracy | 94.44% |
| Precision | 71.43% |
| Recall | 100.00% |
| Specificity | 88.89% |
| F1-score | 83.33% |
| ROC-AUC | 96.67% |
| PR-AUC | 90.29% |

The CNN test set is small, so these values must be reported with caution.

### 4.2 LSTM

The LSTM accepts:

```text
Input shape: (72, 6)
Sequence period: previous 72 hours
Forecast target: next 24-hour heavy-weather risk
```

Weather features:

1. Temperature at 2 metres
2. Relative humidity at 2 metres
3. Precipitation
4. Rain
5. Wind speed at 10 metres
6. Surface pressure

Current model files:

```text
models/real_lstm_model.h5
models/lstm_scaler.joblib
models/lstm_threshold.json
models/lstm_metrics.json
models/lstm_history.json
```

Current recorded test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 61.03% |
| Balanced accuracy | 69.81% |
| Precision | 41.76% |
| Recall | 90.44% |
| Specificity | 49.19% |
| F1-score | 57.13% |
| ROC-AUC | 80.02% |
| PR-AUC | 60.14% |
| MCC | 36.97% |

The LSTM is configured as a high-recall early-warning component. It detects most risky weather events but can create false-positive warnings.

### 4.3 Hybrid fusion

The old trained hybrid network was removed because the historical CNN and LSTM samples were not reliably paired by identical location and time.

The current project uses transparent decision-level fusion:

```text
CNN raw score
    -> normalize relative to CNN threshold
    -> reduce weight when the satellite scene is old

LSTM raw score
    -> normalize relative to LSTM threshold

Freshness-adjusted CNN score + LSTM score
    -> final fusion score
    -> GREEN / YELLOW / ORANGE / RED
```

The fusion layer does not claim a separate historical hybrid accuracy because a verified paired event-level test dataset is not yet available.

---

## 5. Satellite freshness policy

A satellite image becomes less representative as it becomes older. The current project records the actual Sentinel-1 acquisition timestamp and applies this project policy:

| Satellite age | Class | CNN freshness factor |
|---|---|---:|
| 0–3 days | FRESH | 1.00 |
| 4–6 days | RECENT | 0.85 |
| 7–12 days | AGING | 0.60 |
| 13–24 days | STALE | 0.30 |
| More than 24 days | EXPIRED | 0.00 |

This is a project design policy, not an official disaster-management standard.

The hybrid layer automatically increases LSTM influence when satellite evidence becomes old.

---

## 6. Data sources

### Sentinel-1

Source:

```text
COPERNICUS/S1_GRD through Google Earth Engine
```

Why Sentinel-1 is useful:

- Radar operates during day and night.
- Radar can observe through most cloud cover.
- It is more usable during storm conditions than ordinary optical imagery.

### Weather

Source:

```text
Open-Meteo historical and forecast APIs
```

No API key is required for the current weather fetcher.

### Labels

The CNN dataset currently contains weakly labelled flood-like and non-flood-like satellite patches.

The LSTM target is a rainfall-based risk proxy. It is not a verified flood-inundation label.

---

## 7. Andhra Pradesh district coverage

The live runner uses representative district coordinates for:

1. Alluri Sitharama Raju
2. Anakapalli
3. Ananthapuramu
4. Annamayya
5. Bapatla
6. Chittoor
7. East Godavari
8. Eluru
9. Guntur
10. Kakinada
11. Dr. B.R. Ambedkar Konaseema
12. Krishna
13. Kurnool
14. Nandyal
15. NTR Vijayawada
16. Palnadu
17. Parvathipuram Manyam
18. Prakasam
19. SPSR Nellore
20. Sri Sathya Sai
21. Srikakulam
22. Tirupati
23. Visakhapatnam
24. Vizianagaram
25. West Godavari
26. YSR Kadapa

The current map is a **district representative-point risk map**. It is not a district polygon inundation map.

---

## 8. Project structure

```text
flood_monitor/
|
|-- config/
|   `-- sms_config.json
|
|-- dashboard/
|   `-- app.py
|
|-- data/
|   |-- raw/
|   `-- processed/
|       |-- ap_district_risk.json
|       |-- ap_district_risk.csv
|       |-- ap_district_risk.geojson
|       |-- current_alert.json
|       |-- alert_history.jsonl
|       |-- satellite_images.npy
|       |-- image_labels.npy
|       `-- district_results/
|
|-- models/
|   |-- real_cnn_model.h5
|   |-- real_lstm_model.h5
|   |-- cnn_threshold.json
|   |-- lstm_threshold.json
|   |-- cnn_metrics.json
|   |-- lstm_metrics.json
|   |-- lstm_scaler.joblib
|   |-- train_real_cnn.py
|   |-- evaluate_real_cnn.py
|   |-- train_real_lstm.py
|   `-- evaluate_real_lstm.py
|
|-- outputs/
|   |-- flood_risk_maps/
|   |   |-- ap_live_hybrid_risk_map.html
|   |   `-- ap_live_hybrid_risk_map.png
|   `-- predictions/
|       |-- alert_log.csv
|       |-- sms_log.csv
|       |-- dashboard_state.json
|       `-- ap_live_risk_summary.json
|
|-- realtime/
|   |-- fetch_live_weather_sequence.py
|   |-- predict_live_cnn.py
|   |-- predict_live_lstm.py
|   |-- hybrid_decision.py
|   |-- run_full_pipeline.py
|   |-- run_all_districts.py
|   `-- build_ap_risk_map_data.py
|
|-- utils/
|   |-- alert_system.py
|   |-- ap_districts.py
|   |-- build_real_lstm_dataset.py
|   |-- constants.py
|   |-- lstm_weather_contract.py
|   |-- risk_map_generator.py
|   |-- sentinel_cnn_contract.py
|   |-- sentinel_fetcher.py
|   `-- weather_fetcher.py
|
|-- main.py
|-- README.md
|-- requirements.txt
`-- requirements-lock.txt
```

---

## 9. System requirements

Recommended development environment:

```text
Operating system: Windows 10 or Windows 11
Python: 3.11
TensorFlow: 2.13.0
Google Earth Engine project: ap-flood-monitor
```

The project was developed and tested using PowerShell.

---

## 10. Installation

### 10.1 Open the project

```powershell
cd C:\Users\user\Desktop\flood_monitor
```

### 10.2 Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 10.3 Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 10.4 Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The project keeps `numpy==1.24.3` because TensorFlow 2.13 is not compatible with NumPy 2.x.

For the exact working environment snapshot, use:

```powershell
python -m pip install -r requirements-lock.txt
```

Use the lock file only when reproducing the same tested Windows environment.

---

## 11. Google Earth Engine setup

Authenticate:

```powershell
earthengine authenticate
```

Test initialization:

```powershell
python -c "import ee; ee.Initialize(project='ap-flood-monitor'); print('EARTH ENGINE READY')"
```

Expected:

```text
EARTH ENGINE READY
```

If initialization reports that no project was found, pass the project explicitly:

```python
ee.Initialize(project="ap-flood-monitor")
```

---

## 12. Run a single-location live pipeline

Example for NTR Vijayawada:

```powershell
python -m realtime.run_full_pipeline `
    --project ap-flood-monitor `
    --lat 16.51 `
    --lon 80.65 `
    --name ntr_vijayawada
```

This performs:

```text
Sentinel-1 fetch
-> CNN prediction
-> 72-hour weather fetch
-> LSTM prediction
-> freshness-aware hybrid fusion
-> current alert
-> alert history
```

Main outputs:

```text
data/processed/current_alert.json
data/processed/alert_history.jsonl
```

---

## 13. Run all 26 districts

```powershell
python -m realtime.run_all_districts `
    --project ap-flood-monitor
```

If the process stops because of a temporary network or Earth Engine error:

```powershell
python -m realtime.run_all_districts `
    --project ap-flood-monitor `
    --resume
```

District archives are saved under:

```text
data/processed/district_results/<district_slug>/
```

Do not use `--resume` when you intentionally need to refresh every district with newly available satellite and weather data.

---

## 14. Aggregate district risk data

```powershell
python -m realtime.build_ap_risk_map_data
```

Outputs:

```text
data/processed/ap_district_risk.json
data/processed/ap_district_risk.csv
data/processed/ap_district_risk.geojson
```

Verify:

```powershell
python -c "import json; d=json.load(open('data/processed/ap_district_risk.json')); print('TOTAL:',d['district_count']); print('SUCCESS:',d['successful_count']); print('UNAVAILABLE:',d['unavailable_count'])"
```

Target:

```text
TOTAL: 26
SUCCESS: 26
UNAVAILABLE: 0
```

---

## 15. Generate risk maps

```powershell
python -m utils.risk_map_generator
```

Outputs:

```text
outputs/flood_risk_maps/ap_live_hybrid_risk_map.html
outputs/flood_risk_maps/ap_live_hybrid_risk_map.png
outputs/predictions/ap_live_risk_summary.json
```

Open the interactive map:

```powershell
Start-Process .\outputs\flood_risk_maps\ap_live_hybrid_risk_map.html
```

---

## 16. Run the alert system

Evaluate current district results without real SMS:

```powershell
python -m utils.alert_system --mode live
```

Test SMS formatting and provider configuration:

```powershell
python -m utils.alert_system --mode sms-test
```

Validate configured phone numbers:

```powershell
python -m utils.alert_system --mode validate-phones
```

Real transmission is optional and should be enabled only after project validation.

---

## 17. Start the dashboard

```powershell
python .\dashboard\app.py
```

Open:

```text
http://127.0.0.1:5050
```

PowerShell shortcut:

```powershell
Start-Process "http://127.0.0.1:5050"
```

Dashboard features:

- Statewide highest alert
- GREEN, YELLOW, ORANGE, and RED district counts
- 26-district map
- District filter
- District-level details
- CNN and LSTM scores
- Satellite age and freshness
- Effective model weights
- Rainfall summaries
- Alert history
- Real model metrics
- SMS provider and recipient management
- Manual eligible-alert test
- Controlled automatic-alert option

---

## 18. SMS configuration

Configuration file:

```text
config/sms_config.json
```

Recommended academic configuration:

```json
{
  "provider": "mock",
  "recipients": [
    {
      "name": "AP District Authority",
      "phone": "+919XXXXXXXXX"
    }
  ],
  "alert_on_levels": [
    "ORANGE",
    "RED"
  ],
  "cooldown_minutes": 30,
  "max_sms_per_hour": 10,
  "dashboard_auto_sms": false,
  "minimum_fusion_for_sms": 0.6
}
```

Mock mode does not send a real message.

Real SMS must not be triggered only by the CNN score. The project requires an eligible final hybrid alert:

```text
ORANGE or RED
AND
fusion score >= configured minimum
AND
recipient is valid
AND
cooldown/hourly limits permit sending
```

Do not commit real credentials:

```text
config/sms_config.json
```

Add it to `.gitignore`.

---

## 19. Training

The final saved models are already available. Training is not required during normal live operation.

### Train CNN

```powershell
python -m models.train_real_cnn
```

Evaluate:

```powershell
python -m models.evaluate_real_cnn
```

### Build LSTM dataset

```powershell
python -m utils.build_real_lstm_dataset
```

### Train LSTM

```powershell
python -m models.train_real_lstm
```

Evaluate:

```powershell
python -m models.evaluate_real_lstm
```

Retrain only when:

- New verified labels are collected
- The preprocessing contract changes
- Model input features change
- The model performs poorly on newer verified events
- The system is expanded to a substantially different region

Do not retrain merely because a new live image or weather sequence arrives.

---

## 20. End-to-end operational sequence

```powershell
python -m realtime.run_all_districts `
    --project ap-flood-monitor

python -m realtime.build_ap_risk_map_data

python -m utils.risk_map_generator

python -m utils.alert_system --mode live

python .\dashboard\app.py
```

Operational pipeline:

```text
Fetch latest data
-> run CNN and LSTM
-> freshness-aware fusion
-> archive all district results
-> build statewide dataset
-> generate map
-> evaluate alerts
-> display dashboard
```

---

## 21. Important output files

| File | Purpose |
|---|---|
| `data/processed/current_alert.json` | Latest single-location result |
| `data/processed/alert_history.jsonl` | Pipeline execution history |
| `data/processed/ap_district_risk.json` | Dashboard and alert source |
| `data/processed/ap_district_risk.csv` | Tabular district risk data |
| `data/processed/ap_district_risk.geojson` | Mapping data |
| `outputs/flood_risk_maps/ap_live_hybrid_risk_map.html` | Interactive map |
| `outputs/flood_risk_maps/ap_live_hybrid_risk_map.png` | Static map |
| `outputs/predictions/alert_log.csv` | Alert history |
| `outputs/predictions/sms_log.csv` | SMS attempts and status |
| `outputs/predictions/dashboard_state.json` | Automatic SMS duplicate protection |

---

## 22. Troubleshooting

### `ModuleNotFoundError: No module named 'ee'`

```powershell
python -m pip install earthengine-api
```

### `ee.Initialize: no project found`

```python
ee.Initialize(project="ap-flood-monitor")
```

### `ModuleNotFoundError: No module named 'matplotlib'`

```powershell
python -m pip install matplotlib folium pandas colorama
```

### NumPy 2.x / TensorFlow error

```powershell
python -m pip uninstall numpy -y
python -m pip install numpy==1.24.3
```

### TensorFlow protobuf error

```powershell
python -m pip install protobuf==4.25.3
```

### `ModuleNotFoundError: No module named 'sklearn'`

```powershell
python -m pip install scikit-learn
```

### Dashboard starts but no real SMS is delivered

Expected in academic mode:

```text
[SMS] Mock mode — no real message will be transmitted
```

Real SMS also requires a valid non-placeholder recipient and real provider credentials.

### No new Sentinel image

The fetcher searches older windows. The fusion system then reduces CNN influence according to scene age.

---

## 23. Limitations

1. The CNN dataset currently contains only 150 images.
2. CNN labels are weak labels rather than complete official ground truth.
3. The LSTM target is a heavy-rainfall risk proxy.
4. LSTM recall is high, but false-positive warnings are frequent.
5. The CNN and LSTM do not yet have a large verified paired event dataset.
6. Hybrid thresholds and weights are transparent project rules rather than a trained paired fusion model.
7. Each district currently uses one representative coordinate.
8. The map does not show exact flood boundaries or village-level inundation.
9. The system does not currently integrate:
   - Live river gauges
   - Reservoir or dam releases
   - Drainage capacity
   - Soil moisture
   - Detailed DEM-derived flow
   - Official verified flood reports
10. Real emergency decisions require official field and government verification.

---

## 24. Clean submission copy

The Python virtual environment can occupy more than 1 GB and must not be included in the submitted project.

Exclude:

```text
.venv/
.git/
__pycache__/
*.pyc
models/cnn_checkpoints/
models/lstm_checkpoints/
temporary upgrade folders
```

The submitted project should contain:

- Source code
- Final trained models
- Thresholds and scaler
- Required live result samples
- README
- Requirements
- Dashboard
- Maps
- Documentation

---

## 25. Future improvements

- Collect a larger verified Sentinel-1 flood dataset
- Use grouped event-level train/validation/test splitting
- Integrate official flood-event databases
- Add river-gauge and reservoir-release data
- Add soil moisture and DEM-derived terrain features
- Use district polygon boundaries
- Produce pixel-level flood extent maps
- Calibrate freshness and hybrid thresholds on verified events
- Add scheduled execution and secure production deployment
- Enable a verified production SMS gateway
- Add user authentication and role-based dashboard access

---

## 26. Project summary

The Andhra Pradesh Flood Early Warning System is a district-level deep-learning prototype. It combines recent Sentinel-1 radar evidence with 72-hour weather sequences, adjusts satellite influence according to image freshness, produces transparent hybrid risk levels for all 26 Andhra Pradesh districts, generates maps and logs, and presents results through an interactive dashboard with controlled SMS support.

The current implementation is suitable for academic demonstration, experimentation, and further research. It must not be represented as an official public disaster-warning service.
