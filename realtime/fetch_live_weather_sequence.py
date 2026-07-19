from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import requests

from utils.lstm_weather_contract import (
    CONTRACT_NAME, FEATURES, LIVE_METADATA_PATH, LIVE_RAW_PATH,
    LIVE_SEQUENCE_PATH, SCALER_PATH, SEQUENCE_LENGTH,
)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--name", default="requested_location")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SCALER_PATH}. Build the LSTM dataset first."
        )

    params = {
        "latitude": args.lat,
        "longitude": args.lon,
        "hourly": ",".join(FEATURES),
        "past_days": 4,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }

    retryable_statuses = {429, 500, 502, 503, 504}
    max_attempts = 6
    response = None

    for attempt in range(1, max_attempts + 1):
        try:
            candidate = requests.get(
                FORECAST_URL,
                params=params,
                timeout=(15, 90),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_attempts:
                raise

            delay = min(60, 5 * (2 ** (attempt - 1)))
            print(
                f"[Weather] Network failure on attempt "
                f"{attempt}/{max_attempts}: {exc}"
            )
            print(f"[Weather] Retrying in {delay} seconds...")
            time.sleep(delay)
            continue

        if (
            candidate.status_code in retryable_statuses
            and attempt < max_attempts
        ):
            delay = min(60, 5 * (2 ** (attempt - 1)))
            print(
                f"[Weather] HTTP {candidate.status_code} on attempt "
                f"{attempt}/{max_attempts}"
            )
            print(f"[Weather] Retrying in {delay} seconds...")
            time.sleep(delay)
            continue

        candidate.raise_for_status()
        response = candidate
        break

    if response is None:
        raise RuntimeError(
            "Weather API did not return a usable response after retries."
        )
    hourly = response.json().get("hourly") or {}
    times = hourly.get("time") or []

    if len(times) < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Only {len(times)} hourly rows returned; "
            f"{SEQUENCE_LENGTH} required."
        )

    columns = []
    for feature in FEATURES:
        values = hourly.get(feature)
        if values is None or len(values) != len(times):
            raise RuntimeError(f"Missing live feature: {feature}")
        columns.append(np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=np.float32,
        ))

    raw = np.column_stack(columns).astype(np.float32)
    positions = np.arange(len(raw))
    for index in range(raw.shape[1]):
        column = raw[:, index]
        finite = np.isfinite(column)
        if not finite.any():
            raise RuntimeError(f"No valid values for {FEATURES[index]}")
        if (~finite).any():
            column[~finite] = np.interp(
                positions[~finite], positions[finite], column[finite]
            )
            raw[:, index] = column

    now_local = datetime.now()
    eligible = [
        index for index, timestamp in enumerate(times)
        if datetime.fromisoformat(timestamp) <= now_local
    ]
    if len(eligible) < SEQUENCE_LENGTH:
        raise RuntimeError("Not enough completed/current hourly rows")

    end_index = eligible[-1] + 1
    start_index = end_index - SEQUENCE_LENGTH
    raw_sequence = raw[start_index:end_index]

    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(raw_sequence).astype(np.float32)

    LIVE_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(LIVE_RAW_PATH, raw_sequence, allow_pickle=False)
    np.save(LIVE_SEQUENCE_PATH, scaled, allow_pickle=False)

    rain_i = FEATURES.index("precipitation")
    humidity_i = FEATURES.index("relative_humidity_2m")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT_NAME,
        "location_name": args.name,
        "latitude": args.lat,
        "longitude": args.lon,
        "features": list(FEATURES),
        "shape": list(scaled.shape),
        "start_time": times[start_index],
        "end_time": times[end_index - 1],
        "raw_summary": {
            "rain_last_24h_mm": float(raw_sequence[-24:, rain_i].sum()),
            "rain_last_72h_mm": float(raw_sequence[:, rain_i].sum()),
            "mean_humidity_last_24h": float(
                raw_sequence[-24:, humidity_i].mean()
            ),
        },
    }
    LIVE_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n[LSTM Live] Weather sequence saved")
    print("Location :", args.name)
    print("Raw      :", LIVE_RAW_PATH)
    print("Scaled   :", LIVE_SEQUENCE_PATH)
    print("Metadata :", LIVE_METADATA_PATH)
    print("Shape    :", scaled.shape)
    print("Rain 24h :", metadata["raw_summary"]["rain_last_24h_mm"], "mm")
    print("Rain 72h :", metadata["raw_summary"]["rain_last_72h_mm"], "mm")


if __name__ == "__main__":
    main()
