"""
Run the complete live pipeline with data-lineage and location consistency checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SENTINEL_METADATA = PROCESSED_DIR / "sentinel1_live_patch.json"
WEATHER_METADATA = PROCESSED_DIR / "lstm_live_sequence.json"
CNN_RESULT = PROCESSED_DIR / "live_cnn_prediction.json"
LSTM_RESULT = PROCESSED_DIR / "live_lstm_prediction.json"
HYBRID_RESULT = PROCESSED_DIR / "hybrid_live_result.json"
CURRENT_ALERT = PROCESSED_DIR / "current_alert.json"
ALERT_HISTORY = PROCESSED_DIR / "alert_history.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run freshness-aware CNN + LSTM live flood pipeline."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--skip-sentinel-fetch",
        action="store_true",
        help="Reuse satellite data only when coordinates match.",
    )
    parser.add_argument(
        "--skip-weather-fetch",
        action="store_true",
        help="Reuse weather data only when coordinates match.",
    )
    return parser.parse_args()


def run_stage(
    stage_name: str,
    module: str,
    extra_args: List[str] | None = None,
) -> None:
    command = [sys.executable, "-m", module]
    if extra_args:
        command.extend(extra_args)

    print("\n" + "=" * 68)
    print(f"STAGE: {stage_name}")
    print("=" * 68)
    print("Command:", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {stage_name} "
            f"(module={module}, exit_code={completed.returncode})"
        )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not created: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON file: {path}") from exc


def append_history(record: Dict[str, Any]) -> None:
    ALERT_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_location(
    path: Path,
    requested_lat: float,
    requested_lon: float,
    source_name: str,
    tolerance: float = 0.02,
) -> Dict[str, Any]:
    payload = load_json(path)
    actual_lat = payload.get("latitude")
    actual_lon = payload.get("longitude")

    if not isinstance(actual_lat, (int, float)):
        raise RuntimeError(f"{source_name} metadata has no latitude")
    if not isinstance(actual_lon, (int, float)):
        raise RuntimeError(f"{source_name} metadata has no longitude")

    if abs(float(actual_lat) - requested_lat) > tolerance:
        raise RuntimeError(
            f"{source_name} latitude mismatch: requested={requested_lat}, "
            f"saved={actual_lat}. Do not reuse another district's data."
        )
    if abs(float(actual_lon) - requested_lon) > tolerance:
        raise RuntimeError(
            f"{source_name} longitude mismatch: requested={requested_lon}, "
            f"saved={actual_lon}. Do not reuse another district's data."
        )
    return payload


def main() -> None:
    args = parse_args()

    if not (-90.0 <= args.lat <= 90.0):
        raise SystemExit("Latitude must be between -90 and 90.")
    if not (-180.0 <= args.lon <= 180.0):
        raise SystemExit("Longitude must be between -180 and 180.")

    started_utc = datetime.now(timezone.utc).isoformat()

    print("\n" + "#" * 68)
    print("AP FLOOD MONITOR — FRESHNESS-AWARE LIVE PIPELINE")
    print("#" * 68)
    print("Project  :", args.project)
    print("Location :", args.name)
    print("Coords   :", args.lat, args.lon)
    print("Started  :", started_utc)

    try:
        if not args.skip_sentinel_fetch:
            run_stage(
                "Fetch Sentinel-1 patch",
                "utils.sentinel_fetcher",
                [
                    "--project", args.project,
                    "--lat", str(args.lat),
                    "--lon", str(args.lon),
                    "--name", args.name,
                ],
            )

        sentinel_metadata = validate_location(
            SENTINEL_METADATA,
            args.lat,
            args.lon,
            "Sentinel-1",
        )

        run_stage("Run live CNN", "realtime.predict_live_cnn")

        if not args.skip_weather_fetch:
            run_stage(
                "Fetch live weather sequence",
                "realtime.fetch_live_weather_sequence",
                [
                    "--lat", str(args.lat),
                    "--lon", str(args.lon),
                    "--name", args.name,
                ],
            )

        validate_location(
            WEATHER_METADATA,
            args.lat,
            args.lon,
            "Weather",
        )

        run_stage("Run live LSTM", "realtime.predict_live_lstm")
        run_stage("Run freshness-aware hybrid", "realtime.hybrid_decision")

        cnn = load_json(CNN_RESULT)
        lstm = load_json(LSTM_RESULT)
        hybrid = load_json(HYBRID_RESULT)
        completed_utc = datetime.now(timezone.utc).isoformat()

        freshness = (
            hybrid.get("cnn", {}).get("freshness")
            or cnn.get("satellite_freshness")
            or sentinel_metadata.get("satellite_freshness")
            or {}
        )

        record = {
            "pipeline_started_utc": started_utc,
            "pipeline_completed_utc": completed_utc,
            "project": args.project,
            "location_name": args.name,
            "latitude": args.lat,
            "longitude": args.lon,
            "cnn": {
                "probability": cnn.get("probability"),
                "threshold": cnn.get("threshold"),
                "prediction": cnn.get("prediction"),
                "confidence": (
                    cnn.get("confidence_label")
                    or cnn.get("confidence")
                ),
                "satellite_freshness": freshness,
                "latest_scene_utc": sentinel_metadata.get(
                    "latest_scene_utc"
                ),
                "scene_count": sentinel_metadata.get(
                    "sentinel1_scene_count"
                ),
                "composite_method": sentinel_metadata.get(
                    "composite_method"
                ),
                "effective_weight": hybrid.get("cnn", {}).get("weight"),
            },
            "lstm": {
                "probability": lstm.get("probability"),
                "threshold": lstm.get("threshold"),
                "prediction": lstm.get("prediction"),
                "confidence": lstm.get("confidence"),
                "weather_summary": lstm.get("weather_summary", {}),
                "effective_weight": hybrid.get("lstm", {}).get("weight"),
            },
            "hybrid": {
                "fusion_score": hybrid.get("fusion_score"),
                "alert_level": hybrid.get("alert_level"),
                "message": hybrid.get("message"),
                "evidence_agreement": hybrid.get("evidence_agreement"),
            },
            "status": "success",
        }

        CURRENT_ALERT.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        append_history(record)

        print("\n" + "#" * 68)
        print("FULL PIPELINE COMPLETE")
        print("#" * 68)
        print("Location       :", args.name)
        print("Latest S1 scene:", record["cnn"]["latest_scene_utc"])
        print(
            "Satellite age :",
            freshness.get("age_days", "unknown"),
            "days",
        )
        print(
            "Freshness     :",
            freshness.get("class", "UNKNOWN"),
            f"factor={float(freshness.get('factor', 0.0)):.2f}",
        )
        print(
            "CNN           :",
            f"{record['cnn']['probability']:.4f}",
            f"weight={float(record['cnn']['effective_weight'] or 0.0):.3f}",
        )
        print(
            "LSTM          :",
            f"{record['lstm']['probability']:.4f}",
            f"weight={float(record['lstm']['effective_weight'] or 0.0):.3f}",
        )
        print(
            "Fusion score  :",
            f"{float(record['hybrid']['fusion_score']):.4f}",
        )
        print("Alert level   :", record["hybrid"]["alert_level"])
        print("Message       :", record["hybrid"]["message"])
        print("Current alert :", CURRENT_ALERT.relative_to(PROJECT_ROOT))
        print("History       :", ALERT_HISTORY.relative_to(PROJECT_ROOT))

    except Exception as exc:
        failed_record = {
            "pipeline_started_utc": started_utc,
            "pipeline_failed_utc": datetime.now(timezone.utc).isoformat(),
            "project": args.project,
            "location_name": args.name,
            "latitude": args.lat,
            "longitude": args.lon,
            "status": "failed",
            "error": str(exc),
        }
        append_history(failed_record)
        print("\n" + "!" * 68, file=sys.stderr)
        print("PIPELINE FAILED", file=sys.stderr)
        print("!" * 68, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
