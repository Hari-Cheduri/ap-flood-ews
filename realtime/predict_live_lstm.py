import json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from utils.lstm_weather_contract import (
    CONTRACT_NAME, LIVE_METADATA_PATH, LIVE_RESULT_PATH,
    LIVE_SEQUENCE_PATH, METRICS_PATH, MODEL_PATH, THRESHOLD_PATH,
)


def main() -> None:
    for path in (
        LIVE_SEQUENCE_PATH, LIVE_METADATA_PATH, MODEL_PATH, THRESHOLD_PATH
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    status = "unknown"
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        status = metrics.get("status", "unknown")
        if status in {"collapsed_model", "failed_model"}:
            raise RuntimeError(
                f"Saved LSTM status is '{status}'. Do not use it live."
            )

    metadata = json.loads(
        LIVE_METADATA_PATH.read_text(encoding="utf-8")
    )
    if metadata.get("contract") != CONTRACT_NAME:
        raise ValueError("Live LSTM preprocessing contract mismatch")

    threshold = float(json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )["lstm_threshold"])

    sequence = np.load(
        LIVE_SEQUENCE_PATH, allow_pickle=False
    ).astype(np.float32)
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    probability = float(
        model.predict(sequence[None, ...], verbose=0).reshape(-1)[0]
    )
    prediction = int(probability >= threshold)
    distance = probability - threshold

    confidence = (
        "LOW" if abs(distance) < 0.05
        else "MEDIUM" if abs(distance) < 0.15
        else "HIGH"
    )

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT_NAME,
        "model_status": status,
        "location_name": metadata.get("location_name"),
        "latitude": metadata.get("latitude"),
        "longitude": metadata.get("longitude"),
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": (
            "HIGH_WEATHER_RISK" if prediction else "LOW_WEATHER_RISK"
        ),
        "confidence": confidence,
        "weather_summary": metadata.get("raw_summary", {}),
        "interpretation": (
            "Next-24-hour weather-driven flood-risk proxy, "
            "not confirmed inundation."
        ),
    }
    LIVE_RESULT_PATH.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\n================================")
    print("LIVE WEATHER LSTM PREDICTION")
    print("================================")
    print("Location   :", result["location_name"])
    print("Score      :", f"{probability:.4f}")
    print("Threshold  :", f"{threshold:.4f}")
    print("Result     :", result["prediction_label"])
    print("Confidence :", confidence)
    print("Weather    :", result["weather_summary"])
    print("Saved      :", LIVE_RESULT_PATH)


if __name__ == "__main__":
    main()
