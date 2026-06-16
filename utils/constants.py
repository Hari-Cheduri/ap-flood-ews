"""
utils/constants.py
------------------
Single source of truth for all shared constants across the AP Flood EWS.
Import from here in risk_map_generator.py, alert_system.py, and app.py.
"""

# ── Andhra Pradesh geographic bounding box ────────────────────────────────
LAT_MIN: float = 12.5
LAT_MAX: float = 19.5
LON_MIN: float = 76.0
LON_MAX: float = 85.5
STATE_CENTER: tuple[float, float] = (15.9129, 79.7400)

# ── Major river centre latitudes ──────────────────────────────────────────
GODAVARI_LAT:   float = 17.25
KRISHNA_LAT:    float = 16.15
PENNA_LAT:      float = 14.45
VAMSADHARA_LAT: float = 18.35

# ── Risk level thresholds (used by BOTH risk_map_generator and alert_system)
# Names: GREEN / YELLOW / ORANGE / RED  (consistent across all files)
# ─────────────────────────────────────────────────────────────────────────
ALERT_THRESHOLDS: dict[str, dict] = {
    "GREEN":  {"min": 0.00, "max": 0.30, "label": "Normal"},
    "YELLOW": {"min": 0.30, "max": 0.60, "label": "Watch"},
    "ORANGE": {"min": 0.60, "max": 0.80, "label": "Warning"},
    "RED":    {"min": 0.80, "max": 1.001, "label": "Emergency"},
}

# ── Model file stems (extension resolved at runtime: .keras preferred) ────
CNN_EXTRACTOR_STEM   = "cnn_feature_extractor"
LSTM_CHECKPOINT_STEM = "best_lstm"
HYBRID_FINAL_STEM    = "hybrid_final"
