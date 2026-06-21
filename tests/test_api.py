"""
Basic smoke tests for the API.
Add more specific tests for your model's actual behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_monitoring_status():
    response = client.get("/monitoring/status")
    assert response.status_code == 200


# Add a test like this once you have a trained model:
#
# def test_predict():
#     response = client.post("/predict", json={
#         "features": {"feature_1": 1.0, "feature_2": 2.0}
#     })
#     assert response.status_code == 200
#     assert "prediction" in response.json()