"""
Predict flood-like surface conditions and preserve satellite freshness/lineage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from utils.sentinel_cnn_contract import (
    CONTRACT_NAME,
    LIVE_METADATA_PATH,
    LIVE_PATCH_PATH,
    LIVE_RESULT_PATH,
    METRICS_PATH,
    MODEL_H5_PATH,
    MODEL_KERAS_PATH,
    THRESHOLD_PATH,
    validate_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=Path, default=LIVE_PATCH_PATH)
    parser.add_argument(
        "--patch-metadata",
        type=Path,
        default=LIVE_METADATA_PATH,
    )
    parser.add_argument("--output", type=Path, default=LIVE_RESULT_PATH)
    return parser.parse_args()


def load_model() -> tuple[tf.keras.Model, Path]:
    if MODEL_KERAS_PATH.exists():
        return (
            tf.keras.models.load_model(
                str(MODEL_KERAS_PATH),
                compile=False,
            ),
            MODEL_KERAS_PATH,
        )
    if MODEL_H5_PATH.exists():
        return (
            tf.keras.models.load_model(
                str(MODEL_H5_PATH),
                compile=False,
            ),
            MODEL_H5_PATH,
        )
    raise FileNotFoundError(
        "No trained CNN model found. Run: "
        "python -m models.train_real_cnn"
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unknown_freshness(reason: str) -> Dict[str, Any]:
    return {
        "class": "UNKNOWN",
        "age_days": None,
        "factor": 0.0,
        "usable": False,
        "reason": reason,
    }


def main() -> None:
    args = parse_args()

    if not args.patch.exists():
        raise FileNotFoundError(
            f"Missing live patch: {args.patch}\n"
            "Run python -m utils.sentinel_fetcher first."
        )
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing threshold: {THRESHOLD_PATH}\n"
            "Retrain the CNN."
        )

    metadata: Dict[str, Any] = {}
    if args.patch_metadata.exists():
        metadata = load_json(args.patch_metadata)
        contract = metadata.get("contract")
        if contract != CONTRACT_NAME:
            raise ValueError(
                f"Live patch contract is '{contract}', but the trained "
                f"pipeline expects '{CONTRACT_NAME}'."
            )
    else:
        print(
            "[Warning] Live patch metadata is missing. CNN prediction will "
            "be produced, but fusion must ignore untraceable satellite data."
        )

    freshness = metadata.get("satellite_freshness")
    if not isinstance(freshness, dict):
        freshness = unknown_freshness(
            "Actual Sentinel-1 acquisition timestamp was not recorded. "
            "Fetch a new patch with the upgraded sentinel_fetcher."
        )

    if METRICS_PATH.exists():
        metrics = load_json(METRICS_PATH)
        status = metrics.get("status", "unknown")
        if status in {"collapsed_model", "failed_model"}:
            raise RuntimeError(
                f"The saved CNN status is '{status}'. Retrain successfully "
                "before running live prediction."
            )
    else:
        status = "unknown"

    threshold_payload = load_json(THRESHOLD_PATH)
    threshold = float(threshold_payload["cnn_threshold"])

    patch = validate_patch(
        np.load(args.patch, allow_pickle=False),
        "live CNN patch",
    )
    model, model_path = load_model()

    probability = float(
        model.predict(patch[None, ...], verbose=0).reshape(-1)[0]
    )
    probability = float(np.clip(probability, 0.0, 1.0))
    prediction = int(probability >= threshold)

    margin = probability - threshold
    distance = abs(margin)
    if distance < 0.05:
        confidence_label = "LOW"
    elif distance < 0.15:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "HIGH"

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_status": status,
        "contract": CONTRACT_NAME,
        "threshold": threshold,
        "probability": probability,
        "prediction": prediction,
        "prediction_label": (
            "FLOOD_LIKE" if prediction else "NON_FLOOD_LIKE"
        ),
        "distance_from_threshold": margin,
        "confidence_label": confidence_label,
        "satellite_freshness": freshness,
        "satellite_evidence_usable": bool(
            freshness.get("usable", False)
        ),
        "patch": str(args.patch),
        "patch_metadata": metadata,
        "interpretation": (
            "CNN flood-like surface evidence. The hybrid system must "
            "down-weight or ignore this evidence when the scene is old."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n================================")
    print("LIVE SENTINEL-1 CNN PREDICTION")
    print("================================")
    print("Location     :", metadata.get("location_name", "unknown"))
    print(
        "Coordinates  :",
        metadata.get("latitude", "unknown"),
        metadata.get("longitude", "unknown"),
    )
    print("Model status :", status)
    print("Probability  :", f"{probability:.4f}")
    print("Threshold    :", f"{threshold:.4f}")
    print(
        "Prediction   :",
        "FLOOD-LIKE" if prediction else "NON-FLOOD-LIKE",
    )
    print("Model conf.  :", confidence_label)
    print(
        "Scene age    :",
        freshness.get("age_days", "unknown"),
        "days",
    )
    print(
        "Freshness    :",
        freshness.get("class", "UNKNOWN"),
        f"factor={float(freshness.get('factor', 0.0)):.2f}",
    )
    print("Saved result :", args.output)


if __name__ == "__main__":
    main()
