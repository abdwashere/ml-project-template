"""
Generic FastAPI app — calls into src/model.py for predictions
and monitoring/monitor.py for logging.

You should NOT need to edit this file for a new project,
unless your input schema needs custom validation.

Run with: uvicorn api.main:app --reload
"""

import sys
import os
import time
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict

# Allow importing from src/ regardless of where uvicorn is run from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "monitoring"))

from model import load_model, predict
from monitor import log_prediction, check_drift

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["save_path"]
LOG_FILE   = config["monitoring"]["log_file"]
THRESHOLD  = config["monitoring"]["retrain_threshold"]
MIN_PREDS  = config["monitoring"]["min_predictions_for_check"]

app = FastAPI(title=config["project"]["name"])

# Load model once at startup, not per-request
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def root():
    return {"message": f"{config['project']['name']} API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    try:
        start = time.time()
        model  = get_model()
        result = predict(model, request.features)
        latency = time.time() - start

        log_prediction(LOG_FILE, request.features, result, latency)

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/monitoring/status")
def monitoring_status():
    should_retrain, summary = check_drift(LOG_FILE, THRESHOLD, MIN_PREDS)
    return JSONResponse({"should_retrain": should_retrain, **summary})