"""
dashboard/app.py
================

Andhra Pradesh Flood Early Warning System — Live 26-District Dashboard.

Data source
-----------
data/processed/ap_district_risk.json

This dashboard uses the current real-data architecture:
- 26 district Sentinel-1 CNN results
- 26 district weather LSTM results
- freshness-aware hybrid fusion
- district alert history
- controlled Twilio / Fast2SMS / mock SMS dispatch

It deliberately does NOT:
- create synthetic 50x50 risk grids
- apply a second monsoon multiplier
- report a fake hybrid-model accuracy
- send SMS from raw CNN probability alone
- send the same automatic alert every refresh

Run
---
cd <project_root>
python dashboard/app.py
Open http://127.0.0.1:5050
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate
from flask import jsonify, request


# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.alert_system import FloodAlertSystem, SMSAlertSystem  # noqa: E402


LOG = logging.getLogger("ap_flood_dashboard")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REFRESH_MS = 30_000

RISK_DATA = ROOT / "data" / "processed" / "ap_district_risk.json"
ALERT_LOG = ROOT / "outputs" / "predictions" / "alert_log.csv"
SMS_LOG = ROOT / "outputs" / "predictions" / "sms_log.csv"
SMS_CONFIG = ROOT / "config" / "sms_config.json"
DASHBOARD_STATE = (
    ROOT / "outputs" / "predictions" / "dashboard_state.json"
)

CNN_METRICS = ROOT / "models" / "cnn_metrics.json"
LSTM_METRICS = ROOT / "models" / "lstm_metrics.json"

ALERT_HEX = {
    "GREEN": "#22c55e",
    "YELLOW": "#eab308",
    "ORANGE": "#f97316",
    "RED": "#ef4444",
    "UNKNOWN": "#64748b",
}

ALERT_PRIORITY = {
    "UNKNOWN": -1,
    "GREEN": 0,
    "YELLOW": 1,
    "ORANGE": 2,
    "RED": 3,
}

HIGH_SMS_LEVELS = {"ORANGE", "RED"}

DEFAULT_DASHBOARD_SMS = {
    "dashboard_auto_send_enabled": False,
    "dashboard_min_fusion_for_sms": 0.60,
}

DISTRICT_COLUMNS = [
    {"name": "District", "id": "district_name"},
    {"name": "Alert", "id": "alert_level"},
    {"name": "Fusion %", "id": "fusion_percent"},
    {"name": "CNN %", "id": "cnn_percent"},
    {"name": "CNN confidence", "id": "cnn_confidence"},
    {"name": "Satellite age", "id": "satellite_age"},
    {"name": "Freshness", "id": "satellite_freshness_class"},
    {"name": "LSTM %", "id": "lstm_percent"},
    {"name": "Rain 24h", "id": "rain_24h"},
    {"name": "Rain 72h", "id": "rain_72h"},
    {"name": "Updated", "id": "updated"},
]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #04090f;
  --surface: #08121e;
  --card: #0c1a28;
  --card2: #0f2035;
  --border: rgba(0,212,255,.12);
  --cyan: #00d4ff;
  --purple: #a371f7;
  --text: #e2eaf3;
  --muted: #8b949e;
  --dim: #485466;
  --green: #22c55e;
  --yellow: #eab308;
  --orange: #f97316;
  --red: #ef4444;
  --mono: 'JetBrains Mono', monospace;
  --display: 'Rajdhani', sans-serif;
  --body: 'Inter', sans-serif;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(ellipse 60% 45% at 10% 30%, rgba(0,212,255,.035), transparent 70%),
    radial-gradient(ellipse 50% 40% at 90% 15%, rgba(163,113,247,.04), transparent 70%),
    var(--bg);
  color: var(--text);
  font-family: var(--body);
}

.card {
  background: linear-gradient(155deg, var(--card), var(--card2));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 10px 35px rgba(0,0,0,.38);
}

.card-purple {
  border-color: rgba(163,113,247,.22);
}

.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
  color: var(--cyan);
  font-family: var(--display);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
}

.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, rgba(0,212,255,.25), transparent);
}

.section-purple {
  color: var(--purple);
}

.section-purple::after {
  background: linear-gradient(to right, rgba(163,113,247,.30), transparent);
}

.metric-label {
  color: var(--muted);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.metric-value {
  margin-top: 3px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 24px;
  font-weight: 600;
}

.metric-small {
  font-size: 15px;
}

.status-title {
  font-family: var(--display);
  font-size: 56px;
  font-weight: 700;
  letter-spacing: .06em;
  line-height: 1;
}

.live-dot {
  width: 7px;
  height: 7px;
  display: inline-block;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 9px var(--green);
  animation: livePulse 2s infinite;
}

@keyframes livePulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50% { opacity: .45; transform: scale(1.45); }
}

.btn {
  padding: 8px 15px;
  border: 1px solid rgba(0,212,255,.35);
  border-radius: 7px;
  color: var(--cyan);
  background: rgba(0,212,255,.09);
  cursor: pointer;
  font-family: var(--display);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.btn:hover {
  background: rgba(0,212,255,.18);
}

.btn-purple {
  color: var(--purple);
  border-color: rgba(163,113,247,.38);
  background: rgba(163,113,247,.10);
}

.btn-red {
  color: var(--red);
  border-color: rgba(239,68,68,.40);
  background: rgba(239,68,68,.08);
}

.note {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.55;
}

.warning-note {
  padding: 10px 12px;
  border: 1px solid rgba(234,179,8,.22);
  border-radius: 7px;
  color: #f5d95e;
  background: rgba(234,179,8,.06);
  font-size: 11px;
  line-height: 1.5;
}

/* Form controls --------------------------------------------------------- */
:root,
body {
  color-scheme: dark;
}

html,
body,
#react-entry-point {
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}

.card,
.two-col > *,
.three-col > *,
.six-col > * {
  min-width: 0;
}

/* Dash 4 / React-Select uses generated class names.  The scoped attribute
   selectors below work with both the older Select-* classes and Dash 4. */
.dark-dropdown,
.dark-dropdown > div {
  width: 100%;
}

.dark-dropdown .Select-control,
.dark-dropdown [class*="-control"] {
  min-height: 38px !important;
  color: var(--text) !important;
  background: #08121e !important;
  border: 1px solid rgba(0,212,255,.22) !important;
  border-radius: 7px !important;
  box-shadow: none !important;
}

.dark-dropdown .Select-control:hover,
.dark-dropdown [class*="-control"]:hover {
  border-color: rgba(0,212,255,.48) !important;
}

.dark-dropdown .is-focused:not(.is-open) > .Select-control,
.dark-dropdown [class*="-control"]:focus-within {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,.10) !important;
}

.dark-dropdown .Select-value-label,
.dark-dropdown .Select-placeholder,
.dark-dropdown .Select-input,
.dark-dropdown [class*="-singleValue"],
.dark-dropdown [class*="-placeholder"],
.dark-dropdown [class*="-Input"] {
  color: var(--text) !important;
}

.dark-dropdown input {
  color: var(--text) !important;
  background: transparent !important;
}

.dark-dropdown .Select-arrow,
.dark-dropdown [class*="-indicatorContainer"] svg {
  color: var(--muted) !important;
  fill: var(--muted) !important;
}

.dark-dropdown .Select-menu-outer,
.dark-dropdown .Select-menu,
.dark-dropdown [class*="-menu"] {
  z-index: 9999 !important;
  color: var(--text) !important;
  background: #08121e !important;
  border-color: rgba(0,212,255,.22) !important;
}

.dark-dropdown .VirtualizedSelectOption,
.dark-dropdown [class*="-option"] {
  color: var(--text) !important;
  background: #08121e !important;
}

.dark-dropdown .VirtualizedSelectFocusedOption,
.dark-dropdown [class*="-option"]:hover {
  color: #ffffff !important;
  background: #12304a !important;
}

/* Native numeric inputs */
.dark-number-input {
  width: 100% !important;
  height: 38px !important;
  margin-top: 6px !important;
  padding: 8px 11px !important;
  color: var(--text) !important;
  background: #08121e !important;
  border: 1px solid rgba(0,212,255,.22) !important;
  border-radius: 7px !important;
  outline: none !important;
  box-shadow: none !important;
  font-family: var(--mono) !important;
  font-size: 13px !important;
  line-height: 1.2 !important;
  caret-color: var(--cyan) !important;
  -moz-appearance: textfield;
  appearance: textfield;
}

.dark-number-input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,.10) !important;
}

.dark-number-input::-webkit-outer-spin-button,
.dark-number-input::-webkit-inner-spin-button {
  margin: 0;
  -webkit-appearance: none;
}

/* Checklists */
.dark-checklist {
  color: var(--text) !important;
}

.dark-checklist label,
.dark-checklist span {
  color: var(--text) !important;
}

.dark-checklist input[type="checkbox"] {
  width: 15px;
  height: 15px;
  margin-right: 7px !important;
  accent-color: var(--purple);
  vertical-align: middle;
}

/* Editable DataTable cells should remain dark while editing. */
.dash-spreadsheet-container,
.dash-table-container {
  width: 100%;
  max-width: 100%;
  min-width: 0;
}

.dash-spreadsheet-container input {
  color: var(--text) !important;
  background: #08121e !important;
  border: 1px solid var(--cyan) !important;
}

.config-state-note {
  min-height: 18px;
  margin: -5px 0 14px;
  font-size: 11px;
  line-height: 1.45;
}

.dash-spreadsheet-container .dash-spreadsheet-inner th {
  background: #08121e !important;
  color: var(--muted) !important;
  border: 1px solid rgba(0,212,255,.10) !important;
  font-size: 10px !important;
  text-transform: uppercase;
}

.dash-spreadsheet-container .dash-spreadsheet-inner td {
  background: #0c1a28 !important;
  color: var(--text) !important;
  border: 1px solid rgba(0,212,255,.06) !important;
  font-size: 11px !important;
}

@media (max-width: 1050px) {
  .two-col, .three-col, .six-col {
    grid-template-columns: 1fr !important;
  }
}
"""


# ---------------------------------------------------------------------------
# JSON, CSV and configuration helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON safely without reusing one fixed .tmp filename.

    A unique temporary file avoids Windows failures caused by a stale,
    read-only, antivirus-scanned, or editor-locked ``sms_config.json.tmp``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2)
    temporary_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)

        os.replace(temporary_path, path)
        temporary_path = None
    except PermissionError as exc:
        # A direct write is a useful fallback when only rename/replace is
        # blocked by Windows security software or an editor.
        try:
            path.write_text(
                serialized,
                encoding="utf-8",
            )
        except PermissionError as direct_exc:
            raise PermissionError(
                f"Cannot write {path}. Close editors using the file, remove "
                "read-only attributes, and confirm that the config folder is "
                "writable."
            ) from direct_exc
        LOG.warning(
            "Atomic replace was blocked for %s; direct write succeeded: %s",
            path,
            exc,
        )
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOG.warning(
                    "Could not remove temporary config file: %s",
                    temporary_path,
                )


def _merge_dict(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _ensure_sms_config(
    *,
    persist_normalised: bool = False,
) -> Dict[str, Any]:
    """Load SMS settings and merge defaults without writing on every refresh.

    Dash callbacks call this function frequently. Reading configuration must
    never require write permission. Configuration is persisted only from an
    explicit Save action, or when ``persist_normalised`` is deliberately used.
    """
    base = dict(SMSAlertSystem.DEFAULT_CONFIG)
    current = _read_json(SMS_CONFIG, {}) or {}
    config = _merge_dict(base, current)
    for key, value in DEFAULT_DASHBOARD_SMS.items():
        config.setdefault(key, value)

    # High-risk SMS only. Remove accidental GREEN/YELLOW triggers.
    eligible = [
        level
        for level in config.get("alert_on_levels", ["ORANGE", "RED"])
        if level in HIGH_SMS_LEVELS
    ]
    config["alert_on_levels"] = eligible or ["ORANGE", "RED"]

    if persist_normalised and config != current:
        _atomic_write_json(SMS_CONFIG, config)

    return config


def _load_risk_payload() -> Dict[str, Any]:
    payload = _read_json(RISK_DATA)
    if not isinstance(payload, dict):
        return {
            "generated_utc": None,
            "district_count": 26,
            "successful_count": 0,
            "unavailable_count": 26,
            "districts": [],
            "load_error": (
                f"Missing or invalid {RISK_DATA}. Run "
                "python -m realtime.build_ap_risk_map_data"
            ),
        }
    return payload


def _normalise_level(value: Any) -> str:
    level = str(value or "UNKNOWN").upper()
    return level if level in ALERT_HEX else "UNKNOWN"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for raw in payload.get("districts") or []:
        record = dict(raw)
        record["alert_level"] = _normalise_level(
            record.get("alert_level")
        )
        record["fusion_score"] = _float_or_none(
            record.get("fusion_score")
        )
        record["cnn_probability"] = _float_or_none(
            record.get("cnn_probability")
        )
        record["lstm_probability"] = _float_or_none(
            record.get("lstm_probability")
        )
        record["latitude"] = _float_or_none(record.get("latitude"))
        record["longitude"] = _float_or_none(record.get("longitude"))
        record["satellite_age_days"] = _float_or_none(
            record.get("satellite_age_days")
        )
        record["rain_last_24h_mm"] = _float_or_none(
            record.get("rain_last_24h_mm")
        )
        record["rain_last_72h_mm"] = _float_or_none(
            record.get("rain_last_72h_mm")
        )
        output.append(record)
    return output


def _safe_percent(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _safe_number(
    value: Optional[float],
    suffix: str = "",
    decimals: int = 1,
) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def _format_time(value: Any) -> str:
    if not value:
        return "—"
    text = str(value)
    try:
        timestamp = pd.to_datetime(text, utc=True)
        return timestamp.tz_convert("Asia/Kolkata").strftime(
            "%d-%m-%Y %H:%M"
        )
    except Exception:
        return text[:19]


def _risk_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append({
            "district_slug": record.get("district_slug"),
            "district_name": record.get("district_name"),
            "alert_level": record.get("alert_level"),
            "fusion_percent": _safe_percent(
                record.get("fusion_score")
            ),
            "cnn_percent": _safe_percent(
                record.get("cnn_probability")
            ),
            "cnn_confidence": record.get("cnn_confidence") or "—",
            "satellite_age": _safe_number(
                record.get("satellite_age_days"),
                " d",
                1,
            ),
            "satellite_freshness_class": (
                record.get("satellite_freshness_class") or "UNKNOWN"
            ),
            "lstm_percent": _safe_percent(
                record.get("lstm_probability")
            ),
            "rain_24h": _safe_number(
                record.get("rain_last_24h_mm"),
                " mm",
                1,
            ),
            "rain_72h": _safe_number(
                record.get("rain_last_72h_mm"),
                " mm",
                1,
            ),
            "updated": _format_time(
                record.get("pipeline_completed_utc")
            ),
            "_fusion": record.get("fusion_score"),
            "_priority": ALERT_PRIORITY.get(
                record.get("alert_level", "UNKNOWN"),
                -1,
            ),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["_priority", "_fusion"],
        ascending=[False, False],
        na_position="last",
    )


def _load_csv(path: Path, limit: int = 50) -> pd.DataFrame:
    try:
        return pd.read_csv(path).tail(limit).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _sms_sent_last_24h() -> int:
    frame = _load_csv(SMS_LOG, 1000)
    if frame.empty or "timestamp" not in frame.columns:
        return 0
    timestamps = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
        utc=True,
    )
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    status = frame.get("status", pd.Series("", index=frame.index)).astype(str)
    sent = status.str.startswith("sent:") | status.eq("mock_sent")
    return int(((timestamps > cutoff) & sent).sum())


def _last_sms() -> Tuple[str, str, str]:
    frame = _load_csv(SMS_LOG, 100)
    if frame.empty:
        return "—", "—", "—"
    status = frame.get("status", pd.Series("", index=frame.index)).astype(str)
    sent = frame[
        status.str.startswith("sent:")
        | status.eq("mock_sent")
    ]
    if sent.empty:
        return "—", "—", "—"
    row = sent.iloc[-1]
    return (
        str(row.get("name", "—")),
        _format_time(row.get("timestamp")),
        str(row.get("status", "—")),
    )


def _cooldown_status(phone: str, config: Dict[str, Any]) -> str:
    if not phone:
        return "invalid"
    frame = _load_csv(SMS_LOG, 1000)
    if frame.empty or "phone" not in frame.columns:
        return "clear"
    timestamps = pd.to_datetime(
        frame.get("timestamp"),
        errors="coerce",
        utc=True,
    )
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
        minutes=int(config.get("cooldown_minutes", 30))
    )
    recent = frame[
        frame["phone"].astype(str).eq(str(phone))
        & (timestamps > cutoff)
    ]
    return "active" if not recent.empty else "clear"


def _recipient_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "name": recipient.get("name", ""),
            "phone": recipient.get("phone", ""),
            "levels": ", ".join(
                config.get("alert_on_levels", ["ORANGE", "RED"])
            ),
            "cooldown": _cooldown_status(
                str(recipient.get("phone", "")),
                config,
            ),
        }
        for recipient in config.get("recipients", [])
    ]


def _sms_history_rows(limit: int = 12) -> List[Dict[str, Any]]:
    frame = _load_csv(SMS_LOG, limit)
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iloc[::-1].iterrows():
        rows.append({
            "time": _format_time(row.get("timestamp")),
            "recipient": row.get("name", "—"),
            "provider": row.get("provider", "—"),
            "status": row.get("status", "—"),
            "message": row.get("message", "—"),
        })
    return rows


# ---------------------------------------------------------------------------
# Alert evaluation, idempotent logging and SMS
# ---------------------------------------------------------------------------

def _alert_system() -> FloodAlertSystem:
    return FloodAlertSystem(
        risk_path=RISK_DATA,
        log_path=ALERT_LOG,
    )


def _evaluate_alert() -> Dict[str, Any]:
    try:
        return _alert_system().evaluate_live_results()
    except Exception as exc:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state_level": "UNKNOWN",
            "level": "UNKNOWN",
            "available_districts": 0,
            "total_districts": 26,
            "green_count": 0,
            "yellow_count": 0,
            "orange_count": 0,
            "red_count": 0,
            "max_fusion_score": None,
            "top_district": None,
            "districts_at_risk": [],
            "message": str(exc),
            "source_generated_utc": None,
        }


def _event_signature(alert: Dict[str, Any]) -> str:
    urgent = sorted(
        (
            str(item.get("district_slug")),
            str(item.get("alert_level")),
            round(float(item.get("fusion_score") or 0.0), 4),
        )
        for item in alert.get("districts_at_risk", [])
        if item.get("alert_level") in HIGH_SMS_LEVELS
    )
    raw = json.dumps(
        {
            "source": alert.get("source_generated_utc"),
            "level": alert.get("state_level"),
            "urgent": urgent,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dashboard_state() -> Dict[str, Any]:
    return _read_json(DASHBOARD_STATE, {}) or {}


def _save_dashboard_state(state: Dict[str, Any]) -> None:
    _atomic_write_json(DASHBOARD_STATE, state)


def _log_alert_once(alert: Dict[str, Any]) -> None:
    signature = _event_signature(alert)
    state = _dashboard_state()
    if state.get("last_logged_signature") == signature:
        return

    try:
        _alert_system().log_alert(alert)
        state["last_logged_signature"] = signature
        state["last_logged_utc"] = datetime.now(timezone.utc).isoformat()
        _save_dashboard_state(state)
    except Exception as exc:
        LOG.warning("Could not log current alert: %s", exc)


def _automatic_sms(alert: Dict[str, Any]) -> str:
    config = _ensure_sms_config()
    if not bool(config.get("dashboard_auto_send_enabled", False)):
        return "Auto-SMS disabled"

    level = _normalise_level(alert.get("state_level"))
    if level not in set(config.get("alert_on_levels", [])):
        return f"Auto-SMS armed; current level {level} is not eligible"

    minimum = float(
        config.get("dashboard_min_fusion_for_sms", 0.60)
    )
    score = _float_or_none(alert.get("max_fusion_score"))
    if score is None or score < minimum:
        return (
            "Auto-SMS armed; highest hybrid fusion "
            f"{_safe_percent(score)} is below {minimum * 100:.0f}%"
        )

    signature = _event_signature(alert)
    state = _dashboard_state()
    if state.get("last_auto_sms_signature") == signature:
        return "Auto-SMS already evaluated for this prediction cycle"

    try:
        results = SMSAlertSystem(
            config_path=str(SMS_CONFIG)
        ).send_alert_sms(alert)
        state["last_auto_sms_signature"] = signature
        state["last_auto_sms_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        state["last_auto_sms_results"] = results
        _save_dashboard_state(state)

        sent = sum(
            str(result.get("status", "")).startswith("sent:")
            or result.get("status") == "mock_sent"
            for result in results
        )
        return (
            f"Auto-SMS evaluated: {sent} sent/mock-sent, "
            f"{len(results) - sent} skipped/error"
        )
    except Exception as exc:
        LOG.exception("Auto-SMS failed")
        return f"Auto-SMS error: {exc}"


def _send_current_alert_manually() -> str:
    alert = _evaluate_alert()
    level = _normalise_level(alert.get("state_level"))
    config = _ensure_sms_config()
    minimum = float(
        config.get("dashboard_min_fusion_for_sms", 0.60)
    )
    score = _float_or_none(alert.get("max_fusion_score"))

    if level not in HIGH_SMS_LEVELS:
        return (
            f"Blocked: current hybrid alert is {level}. "
            "Only ORANGE/RED can send SMS."
        )
    if score is None or score < minimum:
        return (
            f"Blocked: highest fusion {_safe_percent(score)} "
            f"is below configured {minimum * 100:.0f}%."
        )

    results = SMSAlertSystem(
        config_path=str(SMS_CONFIG)
    ).send_alert_sms(alert)
    if not results:
        return "No SMS dispatched; check levels, recipients and cooldown."
    summary = ", ".join(
        f"{item.get('name')}: {item.get('status')}"
        for item in results
    )
    return summary


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _empty_figure(message: str, height: int = 300) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#8b949e", "size": 12},
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _district_map(
    records: List[Dict[str, Any]],
    selected_level: str = "ALL",
) -> go.Figure:
    valid = [
        record
        for record in records
        if record.get("latitude") is not None
        and record.get("longitude") is not None
        and (
            selected_level == "ALL"
            or record.get("alert_level") == selected_level
        )
    ]
    if not valid:
        return _empty_figure("No matching district results", 430)

    colors = [
        ALERT_HEX.get(record["alert_level"], ALERT_HEX["UNKNOWN"])
        for record in valid
    ]
    sizes = [
        11 + 27 * float(record.get("fusion_score") or 0.0)
        for record in valid
    ]
    customdata = [
        [
            record.get("district_name"),
            record.get("alert_level"),
            _safe_percent(record.get("fusion_score")),
            _safe_percent(record.get("cnn_probability")),
            record.get("cnn_confidence") or "—",
            _safe_number(record.get("satellite_age_days"), " days"),
            record.get("satellite_freshness_class") or "UNKNOWN",
            _safe_percent(record.get("lstm_probability")),
            _safe_number(record.get("rain_last_24h_mm"), " mm"),
            _safe_number(record.get("rain_last_72h_mm"), " mm"),
            _format_time(record.get("pipeline_completed_utc")),
        ]
        for record in valid
    ]

    figure = go.Figure(
        go.Scattergeo(
            lon=[record["longitude"] for record in valid],
            lat=[record["latitude"] for record in valid],
            mode="markers+text",
            text=[
                record.get("district_name", "")
                if record.get("alert_level") in {"ORANGE", "RED"}
                else ""
                for record in valid
            ],
            textposition="top center",
            textfont={"size": 9, "color": "#e2eaf3"},
            customdata=customdata,
            marker={
                "size": sizes,
                "color": colors,
                "line": {"width": 1, "color": "#111827"},
                "opacity": 0.88,
            },
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Alert: %{customdata[1]}<br>"
                "Hybrid fusion: %{customdata[2]}<br>"
                "CNN: %{customdata[3]} (%{customdata[4]})<br>"
                "Satellite age: %{customdata[5]}<br>"
                "Freshness: %{customdata[6]}<br>"
                "LSTM: %{customdata[7]}<br>"
                "Rain 24h: %{customdata[8]}<br>"
                "Rain 72h: %{customdata[9]}<br>"
                "Updated: %{customdata[10]}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_geos(
        projection_type="equirectangular",
        lonaxis_range=[76.5, 85.0],
        lataxis_range=[12.0, 19.5],
        showland=True,
        landcolor="#0b1725",
        showocean=True,
        oceancolor="#06111d",
        showlakes=True,
        lakecolor="#071c2c",
        showcoastlines=True,
        coastlinecolor="#1f4d68",
        showcountries=True,
        countrycolor="#334155",
        showsubunits=True,
        subunitcolor="#243447",
        bgcolor="rgba(0,0,0,0)",
        resolution=50,
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=430,
        showlegend=False,
    )
    return figure


def _alert_distribution(records: List[Dict[str, Any]]) -> go.Figure:
    counts = {
        level: sum(
            record.get("alert_level") == level
            for record in records
        )
        for level in ("GREEN", "YELLOW", "ORANGE", "RED", "UNKNOWN")
    }
    figure = go.Figure(
        go.Pie(
            labels=list(counts),
            values=list(counts.values()),
            hole=0.62,
            marker={
                "colors": [
                    ALERT_HEX[level]
                    for level in counts
                ]
            },
            textinfo="label+value",
            hovertemplate="%{label}: %{value}<extra></extra>",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin={"l": 0, "r": 0, "t": 5, "b": 5},
        legend={"font": {"color": "#8b949e", "size": 10}},
        font={"color": "#e2eaf3"},
    )
    return figure


def _timeline() -> go.Figure:
    frame = _load_csv(ALERT_LOG, 60)
    if frame.empty:
        return _empty_figure("No live alert history yet", 235)

    timestamp = pd.to_datetime(
        frame.get("timestamp"),
        errors="coerce",
    )
    score = pd.to_numeric(
        frame.get("max_fusion_score"),
        errors="coerce",
    ) * 100

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=timestamp,
            y=score,
            mode="lines+markers",
            name="Highest district fusion",
            line={"color": "#00d4ff", "width": 2},
            marker={"size": 5},
            customdata=frame.get(
                "top_district",
                pd.Series("", index=frame.index),
            ),
            hovertemplate=(
                "%{x}<br>Fusion: %{y:.1f}%<br>"
                "Top district: %{customdata}<extra></extra>"
            ),
        )
    )

    for level, threshold in (
        ("YELLOW", 35),
        ("ORANGE", 60),
        ("RED", 80),
    ):
        figure.add_hline(
            y=threshold,
            line={
                "color": ALERT_HEX[level],
                "width": 1,
                "dash": "dot",
            },
            annotation_text=level,
            annotation_font_color=ALERT_HEX[level],
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=235,
        margin={"l": 45, "r": 15, "t": 10, "b": 35},
        xaxis={
            "showgrid": False,
            "color": "#64748b",
        },
        yaxis={
            "range": [0, 100],
            "ticksuffix": "%",
            "gridcolor": "rgba(255,255,255,.05)",
            "color": "#64748b",
        },
        legend={
            "orientation": "h",
            "font": {"color": "#8b949e", "size": 10},
        },
    )
    return figure


def _normalise_metric_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _find_metric(
    payload: Any,
    aliases: Iterable[str],
) -> Optional[float]:
    target = {_normalise_metric_key(alias) for alias in aliases}
    if isinstance(payload, dict):
        # Prefer an exact key in the current object.
        for key, value in payload.items():
            if _normalise_metric_key(str(key)) in target:
                number = _float_or_none(value)
                if number is not None:
                    return number
        # Then search nested dictionaries.
        preferred = [
            payload.get("test"),
            payload.get("test_metrics"),
            payload.get("metrics"),
        ]
        for child in preferred:
            found = _find_metric(child, aliases)
            if found is not None:
                return found
        for value in payload.values():
            found = _find_metric(value, aliases)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_metric(item, aliases)
            if found is not None:
                return found
    return None


def _model_metrics_figure() -> go.Figure:
    sources = {
        "CNN": _read_json(CNN_METRICS, {}) or {},
        "LSTM": _read_json(LSTM_METRICS, {}) or {},
    }
    metric_aliases = {
        "Balanced accuracy": [
            "balanced_accuracy",
            "balanced accuracy",
            "balancedaccuracy",
        ],
        "Precision": [
            "precision",
            "precision_flood",
            "precisionpositive",
        ],
        "Recall": [
            "recall",
            "recall_flood",
            "sensitivity",
        ],
        "F1": [
            "f1",
            "f1_score",
            "f1_flood",
            "f1score",
        ],
        "ROC-AUC": [
            "roc_auc",
            "rocauc",
            "auc",
        ],
    }

    figure = go.Figure()
    colors = {
        "Balanced accuracy": "#00d4ff",
        "Precision": "#a371f7",
        "Recall": "#f97316",
        "F1": "#22c55e",
        "ROC-AUC": "#eab308",
    }

    for metric, aliases in metric_aliases.items():
        x: List[str] = []
        y: List[float] = []
        for model, payload in sources.items():
            value = _find_metric(payload, aliases)
            if value is not None:
                x.append(model)
                y.append(value)
        if y:
            figure.add_trace(
                go.Bar(
                    name=metric,
                    x=x,
                    y=y,
                    marker_color=colors[metric],
                    hovertemplate=(
                        f"{metric}: " + "%{y:.3f}<extra></extra>"
                    ),
                )
            )

    if not figure.data:
        return _empty_figure(
            "CNN/LSTM metric files were not readable",
            260,
        )

    figure.add_annotation(
        text=(
            "Hybrid is transparent decision fusion; "
            "no paired hybrid test accuracy is claimed."
        ),
        x=0.5,
        y=1.13,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#8b949e", "size": 10},
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        barmode="group",
        margin={"l": 40, "r": 10, "t": 50, "b": 35},
        xaxis={"color": "#64748b"},
        yaxis={
            "range": [0, 1],
            "tickformat": ".0%",
            "gridcolor": "rgba(255,255,255,.05)",
            "color": "#64748b",
        },
        legend={
            "orientation": "h",
            "y": 1.02,
            "font": {"color": "#8b949e", "size": 9},
        },
    )
    return figure


# ---------------------------------------------------------------------------
# Layout helper functions
# ---------------------------------------------------------------------------

def _section(title: str, purple: bool = False) -> html.Div:
    css = "section-title section-purple" if purple else "section-title"
    return html.Div(title, className=css)


def _metric_card(
    title: str,
    component_id: str,
    color: Optional[str] = None,
) -> html.Div:
    return html.Div(
        className="card",
        children=[
            html.Div(title, className="metric-label"),
            html.Div(
                "—",
                id=component_id,
                className="metric-value",
                style={"color": color} if color else {},
            ),
        ],
    )


def _district_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frame = _risk_dataframe(records)
    if frame.empty:
        return []
    return frame.drop(
        columns=["_fusion", "_priority"],
        errors="ignore",
    ).to_dict("records")


def _alert_log_rows() -> List[Dict[str, Any]]:
    frame = _load_csv(ALERT_LOG, 12)
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iloc[::-1].iterrows():
        rows.append({
            "time": _format_time(row.get("timestamp")),
            "level": row.get("state_level", "—"),
            "top_district": row.get("top_district", "—"),
            "max_fusion": _safe_percent(
                _float_or_none(row.get("max_fusion_score"))
            ),
            "at_risk": row.get("districts_at_risk", "—"),
        })
    return rows


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

config_initial = _ensure_sms_config()
payload_initial = _load_risk_payload()
records_initial = _records(payload_initial)

app = Dash(
    __name__,
    title="AP Flood EWS — Live 26 Districts",
    update_title="Updating live district risk...",
)
server = app.server

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>AP Flood EWS</title>
  {{%favicon%}}
  {{%css%}}
  <style>{CSS}</style>
</head>
<body>
  {{%app_entry%}}
  <footer>
    {{%config%}}
    {{%scripts%}}
    {{%renderer%}}
  </footer>
</body>
</html>
"""


app.layout = html.Div(
    style={
        "maxWidth": "1500px",
        "margin": "0 auto",
        "padding": "0 20px 45px",
    },
    children=[
        dcc.Store(id="alert-store"),
        dcc.Interval(
            id="refresh-tick",
            interval=REFRESH_MS,
            n_intervals=0,
        ),

        # Header
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "22px 0 18px",
                "borderBottom": "1px solid rgba(0,212,255,.12)",
                "marginBottom": "22px",
            },
            children=[
                html.Div([
                    html.Div(
                        "ANDHRA PRADESH · 26 DISTRICTS",
                        style={
                            "fontFamily": "var(--display)",
                            "fontWeight": 700,
                            "fontSize": "11px",
                            "letterSpacing": ".18em",
                            "color": "#64748b",
                        },
                    ),
                    html.Div(
                        "LIVE FLOOD EARLY WARNING SYSTEM",
                        style={
                            "fontFamily": "var(--display)",
                            "fontWeight": 700,
                            "fontSize": "28px",
                            "letterSpacing": ".05em",
                        },
                    ),
                    html.Div(
                        "Sentinel-1 CNN + 72-hour weather LSTM + freshness-aware hybrid fusion",
                        className="note",
                    ),
                ]),
                html.Div([
                    html.Div([
                        html.Span(className="live-dot"),
                        html.Span(
                            " LIVE DATA",
                            style={
                                "color": "#22c55e",
                                "fontFamily": "var(--display)",
                                "fontWeight": 700,
                                "fontSize": "11px",
                                "letterSpacing": ".12em",
                                "marginLeft": "7px",
                            },
                        ),
                    ]),
                    html.Div(
                        id="last-refresh",
                        className="note",
                        style={"marginTop": "5px", "textAlign": "right"},
                    ),
                ]),
            ],
        ),

        # Current state + distribution
        html.Div(
            className="two-col",
            style={
                "display": "grid",
                "gridTemplateColumns": "1.15fr .85fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    className="card",
                    children=[
                        _section("Current Statewide Monitoring Status"),
                        html.Div(
                            id="state-level",
                            className="status-title",
                        ),
                        html.Div(
                            id="state-message",
                            className="note",
                            style={"margin": "10px 0 20px"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3,1fr)",
                                "gap": "14px",
                            },
                            children=[
                                html.Div([
                                    html.Div(
                                        "Highest Fusion",
                                        className="metric-label",
                                    ),
                                    html.Div(
                                        id="highest-fusion",
                                        className="metric-value metric-small",
                                    ),
                                ]),
                                html.Div([
                                    html.Div(
                                        "Top District",
                                        className="metric-label",
                                    ),
                                    html.Div(
                                        id="top-district",
                                        className="metric-value metric-small",
                                    ),
                                ]),
                                html.Div([
                                    html.Div(
                                        "District Results",
                                        className="metric-label",
                                    ),
                                    html.Div(
                                        id="district-availability",
                                        className="metric-value metric-small",
                                    ),
                                ]),
                            ],
                        ),
                        html.Div(
                            id="data-warning",
                            className="warning-note",
                            style={"marginTop": "18px"},
                        ),
                    ],
                ),
                html.Div(
                    className="card",
                    children=[
                        _section("District Alert Distribution"),
                        dcc.Graph(
                            id="alert-distribution",
                            figure=_alert_distribution(records_initial),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),

        # Six count cards
        html.Div(
            className="six-col",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(6,1fr)",
                "gap": "13px",
                "marginBottom": "18px",
            },
            children=[
                _metric_card("Total", "count-total"),
                _metric_card("Available", "count-available", "#00d4ff"),
                _metric_card("Green", "count-green", ALERT_HEX["GREEN"]),
                _metric_card("Yellow", "count-yellow", ALERT_HEX["YELLOW"]),
                _metric_card("Orange", "count-orange", ALERT_HEX["ORANGE"]),
                _metric_card("Red", "count-red", ALERT_HEX["RED"]),
            ],
        ),

        # Map and selected district details
        html.Div(
            className="two-col",
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 1fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "gap": "15px",
                            },
                            children=[
                                _section("Live District Hybrid Risk Map"),
                                dcc.Dropdown(
                                    id="map-level-filter",
                                    options=[
                                        {"label": "All levels", "value": "ALL"},
                                        {"label": "GREEN", "value": "GREEN"},
                                        {"label": "YELLOW", "value": "YELLOW"},
                                        {"label": "ORANGE", "value": "ORANGE"},
                                        {"label": "RED", "value": "RED"},
                                        {"label": "Unavailable", "value": "UNKNOWN"},
                                    ],
                                    value="ALL",
                                    clearable=False,
                                    style={"width": "185px", "fontSize": "11px"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="district-map",
                            figure=_district_map(records_initial),
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                            },
                        ),
                        html.Div(
                            "Circle size represents hybrid fusion score. "
                            "This is a district-centre risk map, not a pixel-level inundation map.",
                            className="note",
                        ),
                    ],
                ),
                html.Div(
                    className="card",
                    children=[
                        _section("District Detail"),
                        dcc.Dropdown(
                            id="district-selector",
                            options=[
                                {
                                    "label": record.get("district_name"),
                                    "value": record.get("district_slug"),
                                }
                                for record in records_initial
                            ],
                            value=(
                                records_initial[0].get("district_slug")
                                if records_initial
                                else None
                            ),
                            clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        html.Div(id="district-detail"),
                    ],
                ),
            ],
        ),

        # Table
        html.Div(
            className="card",
            style={"marginBottom": "18px"},
            children=[
                _section("All 26 District Results"),
                dash_table.DataTable(
                    id="district-table",
                    columns=DISTRICT_COLUMNS,
                    data=_district_table(records_initial),
                    sort_action="native",
                    filter_action="native",
                    page_size=26,
                    fixed_rows={"headers": True},
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "650px",
                        "overflowY": "auto",
                    },
                    style_cell={
                        "padding": "7px 9px",
                        "minWidth": "95px",
                        "maxWidth": "180px",
                        "whiteSpace": "normal",
                    },
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": '{alert_level} = "GREEN"'
                            },
                            "color": ALERT_HEX["GREEN"],
                        },
                        {
                            "if": {
                                "filter_query": '{alert_level} = "YELLOW"'
                            },
                            "color": ALERT_HEX["YELLOW"],
                        },
                        {
                            "if": {
                                "filter_query": '{alert_level} = "ORANGE"'
                            },
                            "color": ALERT_HEX["ORANGE"],
                        },
                        {
                            "if": {
                                "filter_query": '{alert_level} = "RED"'
                            },
                            "color": ALERT_HEX["RED"],
                        },
                    ],
                ),
            ],
        ),

        # History + metrics
        html.Div(
            className="two-col",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "18px",
                "marginBottom": "24px",
            },
            children=[
                html.Div(
                    className="card",
                    children=[
                        _section("Highest District Fusion History"),
                        dcc.Graph(
                            id="fusion-timeline",
                            figure=_timeline(),
                            config={"displayModeBar": False},
                        ),
                        dash_table.DataTable(
                            id="alert-history-table",
                            columns=[
                                {"name": "Time", "id": "time"},
                                {"name": "Level", "id": "level"},
                                {
                                    "name": "Top district",
                                    "id": "top_district",
                                },
                                {
                                    "name": "Fusion",
                                    "id": "max_fusion",
                                },
                                {"name": "Watch list", "id": "at_risk"},
                            ],
                            data=_alert_log_rows(),
                            page_size=7,
                            style_table={"overflowX": "auto"},
                            style_cell={"padding": "6px 8px"},
                        ),
                    ],
                ),
                html.Div(
                    className="card",
                    children=[
                        _section("Measured CNN and LSTM Performance"),
                        dcc.Graph(
                            id="model-metrics",
                            figure=_model_metrics_figure(),
                            config={"displayModeBar": False},
                        ),
                        html.Div(
                            "The hybrid layer is transparent decision fusion. "
                            "A hybrid accuracy is not shown because the CNN and LSTM "
                            "historical samples are not a paired event-level test set.",
                            className="note",
                        ),
                    ],
                ),
            ],
        ),

        # SMS panel
        html.Div(
            className="card card-purple",
            children=[
                _section("SMS Alert Configuration", purple=True),
                html.Div(
                    className="warning-note",
                    style={"marginBottom": "17px"},
                    children=(
                        "SMS decisions use the final freshness-aware hybrid alert. "
                        "Raw CNN probability alone never sends an SMS. "
                        "Automatic SMS is disabled by default and is restricted "
                        "to ORANGE/RED events above the configured fusion threshold."
                    ),
                ),

                html.Div(
                    className="three-col",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr 1fr",
                        "gap": "15px",
                        "marginBottom": "17px",
                    },
                    children=[
                        html.Div([
                            html.Div("Provider", className="metric-label"),
                            dcc.Dropdown(
                                id="sms-provider-input",
                                className="dark-dropdown",
                                options=[
                                    {"label": "Mock", "value": "mock"},
                                    {"label": "Twilio", "value": "twilio"},
                                    {"label": "Fast2SMS", "value": "fast2sms"},
                                ],
                                value=config_initial.get("provider", "mock"),
                                clearable=False,
                            ),
                        ]),
                        html.Div([
                            html.Div(
                                "High-risk trigger levels",
                                className="metric-label",
                            ),
                            dcc.Checklist(
                                id="sms-levels-input",
                                className="dark-checklist",
                                options=[
                                    {"label": " ORANGE", "value": "ORANGE"},
                                    {"label": " RED", "value": "RED"},
                                ],
                                value=config_initial.get(
                                    "alert_on_levels",
                                    ["ORANGE", "RED"],
                                ),
                                inputStyle={"marginRight": "6px"},
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "18px",
                                    "marginTop": "10px",
                                },
                            ),
                        ]),
                        html.Div([
                            html.Div(
                                "Automatic dispatch",
                                className="metric-label",
                            ),
                            dcc.Checklist(
                                id="sms-auto-input",
                                className="dark-checklist",
                                options=[
                                    {
                                        "label": " Enable automatic ORANGE/RED SMS",
                                        "value": "enabled",
                                    }
                                ],
                                value=(
                                    ["enabled"]
                                    if config_initial.get(
                                        "dashboard_auto_send_enabled",
                                        False,
                                    )
                                    else []
                                ),
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"marginTop": "10px"},
                            ),
                        ]),
                    ],
                ),

                html.Div(
                    className="three-col",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr 1fr",
                        "gap": "15px",
                        "marginBottom": "17px",
                    },
                    children=[
                        html.Div([
                            html.Div(
                                "Minimum hybrid fusion for SMS",
                                className="metric-label",
                            ),
                            dcc.Input(
                                id="sms-min-fusion-input",
                                type="number",
                                min=0.60,
                                max=1.00,
                                step=0.01,
                                value=float(
                                    config_initial.get(
                                        "dashboard_min_fusion_for_sms",
                                        0.60,
                                    )
                                ),
                                className="dark-number-input",
                            ),
                        ]),
                        html.Div([
                            html.Div(
                                "Recipient cooldown (minutes)",
                                className="metric-label",
                            ),
                            dcc.Input(
                                id="sms-cooldown-input",
                                type="number",
                                min=1,
                                max=1440,
                                step=1,
                                value=int(
                                    config_initial.get(
                                        "cooldown_minutes",
                                        30,
                                    )
                                ),
                                className="dark-number-input",
                            ),
                        ]),
                        html.Div([
                            html.Div(
                                "Maximum SMS per hour",
                                className="metric-label",
                            ),
                            dcc.Input(
                                id="sms-cap-input",
                                type="number",
                                min=1,
                                max=100,
                                step=1,
                                value=int(
                                    config_initial.get(
                                        "max_sms_per_hour",
                                        10,
                                    )
                                ),
                                className="dark-number-input",
                            ),
                        ]),
                    ],
                ),

                html.Div(
                    id="sms-config-change-state",
                    className="config-state-note",
                    children=(
                        "The controls above show editable values. "
                        "Press Save SMS Configuration to activate changes."
                    ),
                ),

                html.Div(
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "alignItems": "center",
                        "flexWrap": "wrap",
                        "marginBottom": "18px",
                    },
                    children=[
                        html.Button(
                            "Save SMS Configuration",
                            id="save-sms-config",
                            className="btn btn-purple",
                        ),
                        html.Button(
                            "Test SMS",
                            id="test-sms",
                            className="btn btn-purple",
                        ),
                        html.Button(
                            "Send Current Eligible Alert",
                            id="send-current-alert",
                            className="btn btn-red",
                        ),
                        html.Div(
                            id="sms-action-result",
                            className="note",
                        ),
                    ],
                ),

                html.Div(
                    className="three-col",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4,1fr)",
                        "gap": "13px",
                        "marginBottom": "18px",
                    },
                    children=[
                        html.Div([
                            html.Div("Active provider (saved)", className="metric-label"),
                            html.Div(
                                id="sms-provider-status",
                                className="metric-value metric-small",
                            ),
                        ]),
                        html.Div([
                            html.Div("Sent last 24h", className="metric-label"),
                            html.Div(
                                id="sms-sent-today",
                                className="metric-value metric-small",
                            ),
                        ]),
                        html.Div([
                            html.Div("Last recipient", className="metric-label"),
                            html.Div(
                                id="sms-last-recipient",
                                className="metric-value metric-small",
                            ),
                        ]),
                        html.Div([
                            html.Div("Active automatic status", className="metric-label"),
                            html.Div(
                                id="auto-sms-status",
                                className="note",
                                style={"marginTop": "5px"},
                            ),
                        ]),
                    ],
                ),

                html.Div(
                    className="two-col",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "17px",
                    },
                    children=[
                        html.Div([
                            _section("Recipients", purple=True),
                            dash_table.DataTable(
                                id="recipients-table",
                                columns=[
                                    {
                                        "name": "Name",
                                        "id": "name",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Phone",
                                        "id": "phone",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Levels",
                                        "id": "levels",
                                    },
                                    {
                                        "name": "Cooldown",
                                        "id": "cooldown",
                                    },
                                ],
                                data=_recipient_rows(config_initial),
                                editable=True,
                                row_deletable=True,
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "7px 8px"},
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "marginTop": "10px",
                                    "alignItems": "center",
                                },
                                children=[
                                    html.Button(
                                        "Add Recipient",
                                        id="add-recipient",
                                        className="btn btn-purple",
                                    ),
                                    html.Button(
                                        "Save Recipients",
                                        id="save-recipients",
                                        className="btn btn-purple",
                                    ),
                                    html.Div(
                                        id="recipient-save-result",
                                        className="note",
                                    ),
                                ],
                            ),
                        ]),
                        html.Div([
                            _section("Recent SMS History", purple=True),
                            dash_table.DataTable(
                                id="sms-history-table",
                                columns=[
                                    {"name": "Time", "id": "time"},
                                    {
                                        "name": "Recipient",
                                        "id": "recipient",
                                    },
                                    {
                                        "name": "Provider",
                                        "id": "provider",
                                    },
                                    {"name": "Status", "id": "status"},
                                    {"name": "Message", "id": "message"},
                                ],
                                data=_sms_history_rows(),
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "padding": "7px 8px",
                                    "maxWidth": "180px",
                                    "whiteSpace": "normal",
                                },
                            ),
                        ]),
                    ],
                ),
                html.Div(
                    "Twilio/Fast2SMS secrets stay in config/sms_config.json. "
                    "The dashboard never displays authentication tokens.",
                    className="note",
                    style={"marginTop": "14px"},
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Main dashboard callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("state-level", "children"),
        Output("state-level", "style"),
        Output("state-message", "children"),
        Output("highest-fusion", "children"),
        Output("top-district", "children"),
        Output("district-availability", "children"),
        Output("data-warning", "children"),
        Output("count-total", "children"),
        Output("count-available", "children"),
        Output("count-green", "children"),
        Output("count-yellow", "children"),
        Output("count-orange", "children"),
        Output("count-red", "children"),
        Output("alert-distribution", "figure"),
        Output("district-map", "figure"),
        Output("district-table", "data"),
        Output("district-selector", "options"),
        Output("fusion-timeline", "figure"),
        Output("alert-history-table", "data"),
        Output("alert-store", "data"),
        Output("last-refresh", "children"),
        Output("sms-provider-status", "children"),
        Output("sms-sent-today", "children"),
        Output("sms-last-recipient", "children"),
        Output("auto-sms-status", "children"),
        Output("sms-history-table", "data"),
    ],
    [
        Input("refresh-tick", "n_intervals"),
        Input("map-level-filter", "value"),
        Input("save-sms-config", "n_clicks"),
        Input("save-recipients", "n_clicks"),
    ],
)
def refresh_dashboard(
    _n_intervals: int,
    selected_level: str,
    _save_sms_clicks: Optional[int],
    _save_recipient_clicks: Optional[int],
):
    payload = _load_risk_payload()
    records = _records(payload)
    alert = _evaluate_alert()

    _log_alert_once(alert)
    auto_status = _automatic_sms(alert)

    level = _normalise_level(alert.get("state_level"))
    color = ALERT_HEX[level]
    style = {
        "color": color,
        "textShadow": f"0 0 32px {color}",
    }

    available = int(alert.get("available_districts") or 0)
    total = int(alert.get("total_districts") or payload.get(
        "district_count", 26
    ))
    highest = _float_or_none(alert.get("max_fusion_score"))

    load_error = payload.get("load_error")
    unavailable = total - available
    freshness_unknown = sum(
        str(record.get("satellite_freshness_class", "UNKNOWN")).upper()
        in {"UNKNOWN", "EXPIRED"}
        for record in records
        if record.get("status") == "success"
    )

    warning_parts: List[str] = []
    if load_error:
        warning_parts.append(str(load_error))
    if unavailable:
        warning_parts.append(
            f"{unavailable} district result(s) are unavailable."
        )
    if freshness_unknown:
        warning_parts.append(
            f"{freshness_unknown} district(s) have expired/unknown "
            "satellite freshness; CNN influence is reduced or zero."
        )
    if not warning_parts:
        warning_parts.append(
            "All available district results are loaded. "
            "Verify ORANGE/RED alerts with official river and field data."
        )

    config = _ensure_sms_config()
    last_name, last_time, _last_status = _last_sms()

    options = [
        {
            "label": (
                f"{record.get('district_name')} — "
                f"{record.get('alert_level')}"
            ),
            "value": record.get("district_slug"),
        }
        for record in sorted(
            records,
            key=lambda item: str(item.get("district_name", "")),
        )
    ]

    return (
        level,
        style,
        alert.get("message", ""),
        _safe_percent(highest),
        alert.get("top_district") or "—",
        f"{available}/{total}",
        " ".join(warning_parts),
        str(total),
        str(available),
        str(alert.get("green_count", 0)),
        str(alert.get("yellow_count", 0)),
        str(alert.get("orange_count", 0)),
        str(alert.get("red_count", 0)),
        _alert_distribution(records),
        _district_map(records, selected_level or "ALL"),
        _district_table(records),
        options,
        _timeline(),
        _alert_log_rows(),
        alert,
        (
            f"Risk data: {_format_time(payload.get('generated_utc'))} · "
            f"Dashboard: {datetime.now():%H:%M:%S}"
        ),
        str(config.get("provider", "mock")).upper(),
        str(_sms_sent_last_24h()),
        f"{last_name} · {last_time}",
        auto_status,
        _sms_history_rows(),
    )


@app.callback(
    Output("district-detail", "children"),
    [
        Input("district-selector", "value"),
        Input("refresh-tick", "n_intervals"),
    ],
)
def update_district_detail(
    district_slug: Optional[str],
    _n_intervals: int,
):
    records = _records(_load_risk_payload())
    if not records:
        return html.Div(
            "No district results available.",
            className="warning-note",
        )

    record = next(
        (
            item
            for item in records
            if item.get("district_slug") == district_slug
        ),
        records[0],
    )
    level = record.get("alert_level", "UNKNOWN")
    color = ALERT_HEX.get(level, ALERT_HEX["UNKNOWN"])

    values = [
        ("Alert", level),
        ("Hybrid fusion", _safe_percent(record.get("fusion_score"))),
        ("CNN score", _safe_percent(record.get("cnn_probability"))),
        ("CNN confidence", record.get("cnn_confidence") or "—"),
        (
            "CNN effective weight",
            _safe_percent(record.get("cnn_effective_weight")),
        ),
        (
            "Latest Sentinel scene",
            _format_time(record.get("satellite_latest_scene_utc")),
        ),
        (
            "Satellite age",
            _safe_number(record.get("satellite_age_days"), " days"),
        ),
        (
            "Freshness",
            (
                f"{record.get('satellite_freshness_class', 'UNKNOWN')} "
                f"(factor "
                f"{_safe_number(record.get('satellite_freshness_factor'), '', 2)})"
            ),
        ),
        ("LSTM score", _safe_percent(record.get("lstm_probability"))),
        ("LSTM confidence", record.get("lstm_confidence") or "—"),
        (
            "LSTM effective weight",
            _safe_percent(record.get("lstm_effective_weight")),
        ),
        (
            "Rain previous 24h",
            _safe_number(record.get("rain_last_24h_mm"), " mm"),
        ),
        (
            "Rain previous 72h",
            _safe_number(record.get("rain_last_72h_mm"), " mm"),
        ),
        (
            "Evidence agreement",
            record.get("evidence_agreement") or "—",
        ),
        (
            "Updated",
            _format_time(record.get("pipeline_completed_utc")),
        ),
    ]

    return html.Div([
        html.Div(
            record.get("district_name", "District"),
            style={
                "fontFamily": "var(--display)",
                "fontSize": "22px",
                "fontWeight": 700,
                "color": color,
                "marginBottom": "12px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(label, className="metric-label"),
                        html.Div(
                            value,
                            style={
                                "fontFamily": "var(--mono)",
                                "fontSize": "12px",
                                "color": color if label == "Alert" else "#e2eaf3",
                                "marginTop": "3px",
                            },
                        ),
                    ],
                    style={
                        "padding": "8px 0",
                        "borderBottom": "1px solid rgba(255,255,255,.04)",
                    },
                )
                for label, value in values
            ]
        ),
        html.Div(
            record.get("message") or "",
            className="warning-note",
            style={"marginTop": "14px"},
        ),
    ])


# ---------------------------------------------------------------------------
# SMS configuration and action callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("sms-config-change-state", "children"),
        Output("sms-config-change-state", "style"),
    ],
    [
        Input("sms-provider-input", "value"),
        Input("sms-levels-input", "value"),
        Input("sms-auto-input", "value"),
        Input("sms-min-fusion-input", "value"),
        Input("sms-cooldown-input", "value"),
        Input("sms-cap-input", "value"),
        Input("save-sms-config", "n_clicks"),
    ],
)
def show_sms_config_state(
    provider: str,
    levels: List[str],
    auto_values: List[str],
    min_fusion: float,
    cooldown: int,
    hourly_cap: int,
    _save_clicks: Optional[int],
):
    saved = _ensure_sms_config()

    selected_levels = sorted(
        level
        for level in (levels or [])
        if level in HIGH_SMS_LEVELS
    )
    saved_levels = sorted(
        level
        for level in saved.get("alert_on_levels", [])
        if level in HIGH_SMS_LEVELS
    )

    current = {
        "provider": str(provider or "mock").lower(),
        "levels": selected_levels,
        "auto": "enabled" in (auto_values or []),
        "minimum": round(float(min_fusion or 0.60), 4),
        "cooldown": int(cooldown or 30),
        "cap": int(hourly_cap or 10),
    }
    active = {
        "provider": str(saved.get("provider", "mock")).lower(),
        "levels": saved_levels,
        "auto": bool(saved.get("dashboard_auto_send_enabled", False)),
        "minimum": round(
            float(saved.get("dashboard_min_fusion_for_sms", 0.60)),
            4,
        ),
        "cooldown": int(saved.get("cooldown_minutes", 30)),
        "cap": int(saved.get("max_sms_per_hour", 10)),
    }

    if current == active:
        return (
            "Configuration matches the active saved settings.",
            {
                "color": "#22c55e",
                "fontSize": "11px",
                "lineHeight": "1.45",
            },
        )

    provider_note = ""
    if current["provider"] != active["provider"]:
        provider_note = (
            f" Selected provider: {current['provider'].upper()}; "
            f"active provider: {active['provider'].upper()}."
        )

    return (
        "Unsaved SMS configuration changes. Press Save SMS Configuration "
        "before testing or dispatching." + provider_note,
        {
            "color": "#eab308",
            "fontSize": "11px",
            "lineHeight": "1.45",
        },
    )


@app.callback(
    Output("sms-action-result", "children"),
    [
        Input("save-sms-config", "n_clicks"),
        Input("test-sms", "n_clicks"),
        Input("send-current-alert", "n_clicks"),
    ],
    [
        State("sms-provider-input", "value"),
        State("sms-levels-input", "value"),
        State("sms-auto-input", "value"),
        State("sms-min-fusion-input", "value"),
        State("sms-cooldown-input", "value"),
        State("sms-cap-input", "value"),
    ],
    prevent_initial_call=True,
)
def handle_sms_actions(
    save_clicks: Optional[int],
    test_clicks: Optional[int],
    send_clicks: Optional[int],
    provider: str,
    levels: List[str],
    auto_values: List[str],
    min_fusion: float,
    cooldown: int,
    hourly_cap: int,
):
    from dash import ctx

    trigger = ctx.triggered_id
    if trigger == "save-sms-config":
        selected = [
            level for level in (levels or [])
            if level in HIGH_SMS_LEVELS
        ]
        if not selected:
            return "Select ORANGE and/or RED."

        minimum = float(min_fusion or 0.60)
        if not 0.60 <= minimum <= 1.0:
            return "Minimum fusion must be between 0.60 and 1.00."

        config = _ensure_sms_config()
        config.update({
            "provider": str(provider or "mock").lower(),
            "alert_on_levels": selected,
            "dashboard_auto_send_enabled": (
                "enabled" in (auto_values or [])
            ),
            "dashboard_min_fusion_for_sms": minimum,
            "cooldown_minutes": int(cooldown or 30),
            "max_sms_per_hour": int(hourly_cap or 10),
        })
        _atomic_write_json(SMS_CONFIG, config)
        return (
            f"Saved. Active provider={config['provider'].upper()}; "
            "Auto-SMS is "
            + (
                "ENABLED"
                if config["dashboard_auto_send_enabled"]
                else "disabled"
            )
            + f"; threshold={minimum * 100:.0f}%."
        )

    if trigger == "test-sms":
        try:
            results = SMSAlertSystem(
                config_path=str(SMS_CONFIG)
            ).test_connection()
            if not results:
                return "No test message dispatched. Check recipients/config."
            return ", ".join(
                f"{item.get('name')}: {item.get('status')}"
                for item in results
            )
        except Exception as exc:
            return f"Test SMS failed: {exc}"

    if trigger == "send-current-alert":
        try:
            return _send_current_alert_manually()
        except Exception as exc:
            return f"Current alert dispatch failed: {exc}"

    raise PreventUpdate


@app.callback(
    Output("recipients-table", "data"),
    Input("add-recipient", "n_clicks"),
    State("recipients-table", "data"),
    prevent_initial_call=True,
)
def add_recipient(
    n_clicks: Optional[int],
    current_rows: List[Dict[str, Any]],
):
    if not n_clicks:
        raise PreventUpdate
    rows = list(current_rows or [])
    rows.append({
        "name": "",
        "phone": "+91",
        "levels": "ORANGE, RED",
        "cooldown": "clear",
    })
    return rows


@app.callback(
    Output("recipient-save-result", "children"),
    Input("save-recipients", "n_clicks"),
    State("recipients-table", "data"),
    prevent_initial_call=True,
)
def save_recipients(
    n_clicks: Optional[int],
    rows: List[Dict[str, Any]],
):
    if not n_clicks:
        raise PreventUpdate

    recipients = []
    for row in rows or []:
        name = str(row.get("name", "")).strip()
        phone = str(row.get("phone", "")).strip()
        if not name and not phone:
            continue
        if not name or not phone:
            return "Every retained recipient needs both name and phone."
        recipients.append({"name": name, "phone": phone})

    config = _ensure_sms_config()
    config["recipients"] = recipients
    _atomic_write_json(SMS_CONFIG, config)
    return f"Saved {len(recipients)} recipient(s)."


# ---------------------------------------------------------------------------
# Flask API endpoints
# ---------------------------------------------------------------------------

@server.route("/api/district_risk")
def api_district_risk():
    return jsonify(_load_risk_payload())


@server.route("/api/alert_status")
def api_alert_status():
    return jsonify(_evaluate_alert())


@server.route("/api/sms_history")
def api_sms_history():
    return jsonify(_sms_history_rows(50))


@server.route("/api/test_sms", methods=["POST"])
def api_test_sms():
    try:
        results = SMSAlertSystem(
            config_path=str(SMS_CONFIG)
        ).test_connection()
        return jsonify({"status": "ok", "results": results})
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500


@server.route("/api/send_current_alert", methods=["POST"])
def api_send_current_alert():
    try:
        return jsonify({
            "status": "ok",
            "message": _send_current_alert_manually(),
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500


@server.route("/api/sms_config", methods=["GET", "POST"])
def api_sms_config():
    if request.method == "GET":
        config = _ensure_sms_config()
        safe = dict(config)
        # Do not expose secrets through the API.
        if isinstance(safe.get("twilio"), dict):
            safe["twilio"] = {
                "configured": all(
                    value
                    and "YOUR_" not in str(value)
                    for value in safe["twilio"].values()
                )
            }
        if isinstance(safe.get("fast2sms"), dict):
            safe["fast2sms"] = {
                "configured": bool(
                    safe["fast2sms"].get("api_key")
                    and "YOUR_" not in str(
                        safe["fast2sms"].get("api_key")
                    )
                )
            }
        return jsonify(safe)

    incoming = request.get_json(silent=True) or {}
    allowed = {
        "provider",
        "recipients",
        "alert_on_levels",
        "cooldown_minutes",
        "max_sms_per_hour",
        "dashboard_auto_send_enabled",
        "dashboard_min_fusion_for_sms",
    }
    config = _ensure_sms_config()
    for key in allowed:
        if key in incoming:
            config[key] = incoming[key]
    _atomic_write_json(SMS_CONFIG, config)
    return jsonify({"status": "saved"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AP Flood EWS live 26-district dashboard"
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print(" AP FLOOD EWS — LIVE 26-DISTRICT DASHBOARD")
    print(f" http://{args.host}:{args.port}")
    print(f" Risk source: {RISK_DATA.relative_to(ROOT)}")
    print("=" * 72 + "\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
