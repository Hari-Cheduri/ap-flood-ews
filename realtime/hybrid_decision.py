"""
Freshness-aware transparent CNN + LSTM decision-level fusion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.lstm_weather_contract import HYBRID_RESULT_PATH, LIVE_RESULT_PATH

CNN_RESULT_PATH = Path("data/processed/live_cnn_prediction.json")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing live result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def threshold_centered_score(probability: float, threshold: float) -> float:
    probability = float(np.clip(probability, 0.0, 1.0))
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    if probability < threshold:
        return 0.5 * probability / threshold
    return 0.5 + 0.5 * (
        (probability - threshold) / (1.0 - threshold)
    )


def _safe_factor(cnn: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
    freshness = cnn.get("satellite_freshness")
    if not isinstance(freshness, dict):
        freshness = {
            "class": "UNKNOWN",
            "age_days": None,
            "factor": 0.0,
            "usable": False,
            "reason": "No freshness metadata.",
        }

    factor = freshness.get("factor", 0.0)
    try:
        factor = float(np.clip(float(factor), 0.0, 1.0))
    except (TypeError, ValueError):
        factor = 0.0
    return freshness, factor


def main() -> None:
    cnn = load_json(CNN_RESULT_PATH)
    lstm = load_json(LIVE_RESULT_PATH)

    cnn_score = threshold_centered_score(
        cnn["probability"], cnn["threshold"]
    )
    lstm_score = threshold_centered_score(
        lstm["probability"], lstm["threshold"]
    )

    cnn_confidence = str(
        cnn.get("confidence_label", "MEDIUM")
    ).upper()

    base_cnn_weight = 0.35 if cnn_confidence == "LOW" else 0.45
    base_lstm_weight = 1.0 - base_cnn_weight

    freshness, freshness_factor = _safe_factor(cnn)

    # Down-weight stale satellite evidence, then renormalize the two weights.
    raw_cnn_weight = base_cnn_weight * freshness_factor
    raw_lstm_weight = base_lstm_weight
    total_weight = raw_cnn_weight + raw_lstm_weight

    if total_weight <= 0.0:
        cnn_weight, lstm_weight = 0.0, 1.0
    else:
        cnn_weight = raw_cnn_weight / total_weight
        lstm_weight = raw_lstm_weight / total_weight

    fusion_score = (
        cnn_weight * cnn_score
        + lstm_weight * lstm_score
    )

    cnn_positive = bool(cnn["prediction"])
    lstm_positive = bool(lstm["prediction"])
    effective_cnn_positive = (
        cnn_positive
        and freshness_factor > 0.0
    )
    evidence_agreement = (
        "BOTH_POSITIVE"
        if effective_cnn_positive and lstm_positive
        else "CNN_ONLY"
        if effective_cnn_positive
        else "LSTM_ONLY"
        if lstm_positive
        else "BOTH_NEGATIVE"
    )

    # Use the combined score as the primary alert rule.
    # RED additionally requires two usable positive evidence sources.
    if (
        fusion_score >= 0.80
        and effective_cnn_positive
        and lstm_positive
    ):
        level = "RED"
    elif fusion_score >= 0.60:
        level = "ORANGE"
    elif fusion_score >= 0.35:
        level = "YELLOW"
    else:
        level = "GREEN"

    messages = {
        "GREEN": "No strong combined flood signal. Continue monitoring.",
        "YELLOW": (
            "Possible or borderline combined risk. Monitor updates and "
            "verify local reports."
        ),
        "ORANGE": (
            "Meaningful combined flood-risk evidence. Verify river levels, "
            "drainage conditions, and district reports."
        ),
        "RED": (
            "Strong usable satellite and weather evidence. Escalate for "
            "official verification and emergency decision-making."
        ),
    }

    freshness_class = str(freshness.get("class", "UNKNOWN"))
    message = messages[level]
    if freshness_class in {"STALE", "EXPIRED", "UNKNOWN"}:
        message += (
            f" Sentinel evidence is {freshness_class.lower()} and received "
            "reduced or zero influence."
        )

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "location_name": (
            cnn.get("patch_metadata", {}).get("location_name")
            or lstm.get("location_name")
        ),
        "cnn": {
            "raw_probability": cnn["probability"],
            "threshold": cnn["threshold"],
            "decision_score": cnn_score,
            "positive": cnn_positive,
            "effective_positive": effective_cnn_positive,
            "confidence": cnn_confidence,
            "base_weight": base_cnn_weight,
            "freshness": freshness,
            "freshness_factor": freshness_factor,
            "weight": cnn_weight,
        },
        "lstm": {
            "raw_probability": lstm["probability"],
            "threshold": lstm["threshold"],
            "decision_score": lstm_score,
            "positive": lstm_positive,
            "confidence": lstm.get("confidence"),
            "base_weight": base_lstm_weight,
            "weight": lstm_weight,
        },
        "evidence_agreement": evidence_agreement,
        "fusion_score": fusion_score,
        "alert_level": level,
        "message": message,
        "method": (
            "Threshold-centered decision-level fusion. CNN influence is "
            "multiplied by a Sentinel-1 freshness factor and weights are "
            "renormalized. Combined score drives GREEN/YELLOW/ORANGE; RED "
            "requires two usable positive evidence sources."
        ),
        "limitation": (
            "Academic prototype; not an official disaster alert."
        ),
    }

    HYBRID_RESULT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n================================")
    print("FRESHNESS-AWARE HYBRID DECISION")
    print("================================")
    print("Location       :", result["location_name"])
    print(
        "Satellite age :",
        freshness.get("age_days", "unknown"),
        "days",
    )
    print(
        "Freshness     :",
        freshness_class,
        f"factor={freshness_factor:.2f}",
    )
    print(
        "CNN score     :",
        f"{cnn_score:.4f}",
        f"weight={cnn_weight:.3f}",
    )
    print(
        "LSTM score    :",
        f"{lstm_score:.4f}",
        f"weight={lstm_weight:.3f}",
    )
    print("Agreement     :", evidence_agreement)
    print("Fusion score  :", f"{fusion_score:.4f}")
    print("Alert level   :", level)
    print("Message       :", message)
    print("Saved         :", HYBRID_RESULT_PATH)


if __name__ == "__main__":
    main()
