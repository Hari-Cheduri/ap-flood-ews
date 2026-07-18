from pathlib import Path

SEQUENCE_LENGTH = 72
FORECAST_HORIZON = 24
FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "surface_pressure",
)
INPUT_SHAPE = (SEQUENCE_LENGTH, len(FEATURES))
CONTRACT_NAME = "open_meteo_hourly_72x6_v1"

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

X_TRAIN = PROCESSED_DIR / "X_lstm_train.npy"
X_VAL = PROCESSED_DIR / "X_lstm_val.npy"
X_TEST = PROCESSED_DIR / "X_lstm_test.npy"
Y_TRAIN = PROCESSED_DIR / "y_lstm_train.npy"
Y_VAL = PROCESSED_DIR / "y_lstm_val.npy"
Y_TEST = PROCESSED_DIR / "y_lstm_test.npy"
DATASET_SUMMARY = PROCESSED_DIR / "lstm_dataset_summary.json"

SCALER_PATH = MODELS_DIR / "lstm_scaler.joblib"
MODEL_PATH = MODELS_DIR / "real_lstm_model.h5"
THRESHOLD_PATH = MODELS_DIR / "lstm_threshold.json"
METRICS_PATH = MODELS_DIR / "lstm_metrics.json"
HISTORY_PATH = MODELS_DIR / "lstm_history.json"

LIVE_RAW_PATH = PROCESSED_DIR / "lstm_live_raw.npy"
LIVE_SEQUENCE_PATH = PROCESSED_DIR / "lstm_live_sequence.npy"
LIVE_METADATA_PATH = PROCESSED_DIR / "lstm_live_sequence.json"
LIVE_RESULT_PATH = PROCESSED_DIR / "live_lstm_prediction.json"
HYBRID_RESULT_PATH = PROCESSED_DIR / "hybrid_live_result.json"
