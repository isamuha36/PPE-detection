from fastapi.testclient import TestClient
from pathlib import Path
import pytest
from src.service.api import app

client = TestClient(app)

def test_dashboard_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "SafeMine AI" in response.text

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_stats_endpoint():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "active_violations" in data

def test_detect_endpoint():
    sample_img_path = Path("data/samples/sample_worker_1.jpg")
    if not sample_img_path.exists():
        pytest.skip("sample_worker_1.jpg not found")

    with open(sample_img_path, "rb") as f:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("sample_worker_1.jpg", f, "image/jpeg")},
            params={"conf": 0.3}
        )

    assert response.status_code == 200
    data = response.json()
    assert "total_detections" in data
    assert "compliant" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)
