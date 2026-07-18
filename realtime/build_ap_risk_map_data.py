"""
Aggregate archived district results, including satellite freshness fields.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from utils.ap_districts import AP_DISTRICTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_ROOT = PROCESSED_DIR / "district_results"

JSON_OUTPUT = PROCESSED_DIR / "ap_district_risk.json"
CSV_OUTPUT = PROCESSED_DIR / "ap_district_risk.csv"
GEOJSON_OUTPUT = PROCESSED_DIR / "ap_district_risk.geojson"

LEVEL_COLORS = {
    "GREEN": "#2ca25f",
    "YELLOW": "#ffd92f",
    "ORANGE": "#f28e2b",
    "RED": "#d62728",
    "UNKNOWN": "#808080",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_outputs() -> Dict[str, Any]:
    generated_utc = datetime.now(timezone.utc).isoformat()
    records: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []

    for district in AP_DISTRICTS:
        alert_path = (
            RESULTS_ROOT / district["slug"] / "current_alert.json"
        )

        record: Dict[str, Any] = {
            "district_slug": district["slug"],
            "district_name": district["name"],
            "latitude": district["latitude"],
            "longitude": district["longitude"],
            "status": "unavailable",
            "alert_level": "UNKNOWN",
            "alert_color": LEVEL_COLORS["UNKNOWN"],
            "fusion_score": None,
            "risk_percent": None,
            "message": "No completed live prediction is available.",
            "cnn_probability": None,
            "cnn_threshold": None,
            "cnn_prediction": None,
            "cnn_confidence": None,
            "cnn_effective_weight": None,
            "satellite_latest_scene_utc": None,
            "satellite_scene_count": None,
            "satellite_age_days": None,
            "satellite_freshness_class": "UNKNOWN",
            "satellite_freshness_factor": None,
            "satellite_usable": False,
            "lstm_probability": None,
            "lstm_threshold": None,
            "lstm_prediction": None,
            "lstm_confidence": None,
            "lstm_effective_weight": None,
            "rain_last_24h_mm": None,
            "rain_last_72h_mm": None,
            "mean_humidity_last_24h": None,
            "evidence_agreement": None,
            "pipeline_completed_utc": None,
        }

        if alert_path.exists():
            try:
                alert = _load_json(alert_path)
                cnn = alert.get("cnn") or {}
                lstm = alert.get("lstm") or {}
                hybrid = alert.get("hybrid") or {}
                weather = lstm.get("weather_summary") or {}
                freshness = cnn.get("satellite_freshness") or {}

                level = str(
                    hybrid.get("alert_level") or "UNKNOWN"
                ).upper()
                if level not in LEVEL_COLORS:
                    level = "UNKNOWN"

                fusion_score = _number(hybrid.get("fusion_score"))
                record.update({
                    "status": alert.get("status", "success"),
                    "alert_level": level,
                    "alert_color": LEVEL_COLORS[level],
                    "fusion_score": fusion_score,
                    "risk_percent": (
                        round(fusion_score * 100.0, 2)
                        if fusion_score is not None
                        else None
                    ),
                    "message": hybrid.get("message"),
                    "cnn_probability": _number(cnn.get("probability")),
                    "cnn_threshold": _number(cnn.get("threshold")),
                    "cnn_prediction": cnn.get("prediction"),
                    "cnn_confidence": cnn.get("confidence"),
                    "cnn_effective_weight": _number(
                        cnn.get("effective_weight")
                    ),
                    "satellite_latest_scene_utc": cnn.get(
                        "latest_scene_utc"
                    ),
                    "satellite_scene_count": cnn.get("scene_count"),
                    "satellite_age_days": _number(
                        freshness.get("age_days")
                    ),
                    "satellite_freshness_class": freshness.get(
                        "class", "UNKNOWN"
                    ),
                    "satellite_freshness_factor": _number(
                        freshness.get("factor")
                    ),
                    "satellite_usable": bool(
                        freshness.get("usable", False)
                    ),
                    "lstm_probability": _number(lstm.get("probability")),
                    "lstm_threshold": _number(lstm.get("threshold")),
                    "lstm_prediction": lstm.get("prediction"),
                    "lstm_confidence": lstm.get("confidence"),
                    "lstm_effective_weight": _number(
                        lstm.get("effective_weight")
                    ),
                    "rain_last_24h_mm": _number(
                        weather.get("rain_last_24h_mm")
                    ),
                    "rain_last_72h_mm": _number(
                        weather.get("rain_last_72h_mm")
                    ),
                    "mean_humidity_last_24h": _number(
                        weather.get("mean_humidity_last_24h")
                    ),
                    "evidence_agreement": hybrid.get(
                        "evidence_agreement"
                    ),
                    "pipeline_completed_utc": alert.get(
                        "pipeline_completed_utc"
                    ),
                })
            except Exception as exc:
                record["status"] = "invalid_result"
                record["message"] = f"Could not read saved result: {exc}"

        records.append(record)
        features.append({
            "type": "Feature",
            "id": district["slug"],
            "geometry": {
                "type": "Point",
                "coordinates": [
                    district["longitude"],
                    district["latitude"],
                ],
            },
            "properties": dict(record),
        })

    successful = sum(
        1 for record in records if record["status"] == "success"
    )

    summary = {
        "generated_utc": generated_utc,
        "district_count": len(AP_DISTRICTS),
        "successful_count": successful,
        "unavailable_count": len(AP_DISTRICTS) - successful,
        "risk_method": (
            "Freshness-aware CNN + LSTM threshold-centered fusion"
        ),
        "geometry_note": (
            "Points are representative district locations, not boundaries."
        ),
        "districts": records,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    fieldnames = list(records[0].keys()) if records else []
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    geojson = {
        "type": "FeatureCollection",
        "name": "ap_live_freshness_aware_hybrid_flood_risk",
        "generated_utc": generated_utc,
        "features": features,
    }
    GEOJSON_OUTPUT.write_text(
        json.dumps(geojson, indent=2),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    summary = build_outputs()
    print("\n========================================")
    print("AP DISTRICT RISK MAP DATA")
    print("========================================")
    print("Districts  :", summary["district_count"])
    print("Successful :", summary["successful_count"])
    print("Unavailable:", summary["unavailable_count"])
    print("JSON       :", JSON_OUTPUT.relative_to(PROJECT_ROOT))
    print("CSV        :", CSV_OUTPUT.relative_to(PROJECT_ROOT))
    print("GeoJSON    :", GEOJSON_OUTPUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
