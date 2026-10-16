"""
AeroHealth AI - API integration tests using FastAPI TestClient
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "AeroHealth AI"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_fleet_engines_endpoint():
    response = client.get("/fleet/engines")
    assert response.status_code == 200
    engines = response.json()
    assert len(engines) == 100
    assert "engine_id" in engines[0]


def test_predict_endpoint_flow():
    # Fetch real telemetry from engine 1
    engine_resp = client.get("/fleet/engine/1")
    assert engine_resp.status_code == 200
    engine_data = engine_resp.json()
    
    # Send last 10 readings to predict endpoint
    readings = engine_data["records"][-10:]
    payload = {
        "unit_id": 1,
        "readings": readings
    }
    
    pred_resp = client.post("/predict/rul", json=payload)
    if pred_resp.status_code == 503:
        pytest.skip("Model not yet trained / saved.")
        
    assert pred_resp.status_code == 200
    result = pred_resp.json()
    assert "predicted_rul" in result
    assert "health_index_pct" in result
    assert result["alert_level"] in ["NORMAL", "CAUTION", "CRITICAL"]
