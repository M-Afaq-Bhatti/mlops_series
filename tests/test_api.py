import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

import app.main as main_module
from app.main import app

VALID_PAYLOAD = {
    "age": 21,
    "gender": "female",
    "course": "b.sc",
    "study_hours": 7.91,
    "class_attendance": 98.8,
    "internet_access": "no",
    "sleep_hours": 4.9,
    "sleep_quality": "average",
    "study_method": "online videos",
    "facility_rating": "low",
    "exam_difficulty": "easy",
}

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Build a tiny fake pipeline so the test doesn't depend on a real
    # trained model file or the Kaggle dataset being present.
    dummy_pipeline = Pipeline([("model", DummyRegressor(strategy="constant", constant=72.5))])
    dummy_pipeline.fit(pd.DataFrame({"x": [1, 2, 3]}), [1, 2, 3])

    model_path = tmp_path / "model.pkl"
    joblib.dump(dummy_pipeline, model_path)

    monkeypatch.setattr(main_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(main_module, "_model", None)

    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_score" in body
    assert body["predicted_score"] == 72.5


def test_predict_rejects_invalid_category(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["gender"] = "Unknown"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_attendance(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["class_attendance"] = 250
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(main_module, "_model", None)
    client = TestClient(app)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503