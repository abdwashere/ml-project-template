"""
Generic monitoring — logs every prediction and checks if the model
needs retraining based on confidence/accuracy drift.

You should NOT need to edit this file for a new project.
"""

import json
import os
from datetime import datetime


def log_prediction(log_file: str, input_features: dict, result: dict, latency_seconds: float = None):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "input": input_features,
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "latency_seconds": round(latency_seconds, 4) if latency_seconds is not None else None,
    }

    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)

    logs.append(record)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)


def get_logs(log_file: str):
    if not os.path.exists(log_file):
        return []
    with open(log_file, "r") as f:
        return json.load(f)


def check_drift(log_file: str, threshold: float, min_predictions: int):
    """
    Simple drift check: is average confidence dropping below threshold?
    Returns (should_retrain: bool, summary: dict)
    """
    logs = get_logs(log_file)

    if len(logs) < min_predictions:
        return False, {
            "reason": f"Not enough predictions yet ({len(logs)}/{min_predictions})",
            "total_predictions": len(logs),
        }

    confidences = [l["confidence"] for l in logs if l.get("confidence") is not None]

    if not confidences:
        return False, {"reason": "No confidence scores available to check drift"}

    avg_confidence = sum(confidences) / len(confidences)
    should_retrain = avg_confidence < threshold

    summary = {
        "total_predictions": len(logs),
        "avg_confidence": round(avg_confidence, 4),
        "threshold": threshold,
        "should_retrain": should_retrain,
    }

    return should_retrain, summary