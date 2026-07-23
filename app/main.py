"""FastAPI serving app for the student score predictor."""

import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Score Predictor", version="0.1.0")

MODEL_PATH = Path("model.pkl")
_model = None


class StudentFeatures(BaseModel):
    age: int = Field(..., ge=0)
    gender: Literal["male", "female", "other"]
    course: Literal["b.sc", "diploma", "bca", "b.com", "ba", "bba", "b.tech"]
    study_hours: float = Field(..., ge=0)
    class_attendance: float = Field(..., ge=0, le=100)
    internet_access: Literal["yes", "no"]
    sleep_hours: float = Field(..., ge=0, le=24)
    sleep_quality: Literal["poor", "average", "good"]
    study_method: Literal["self-study", "group study", "online videos", "coaching", "mixed"]
    facility_rating: Literal["low", "medium", "high"]
    exam_difficulty: Literal["easy", "moderate", "hard"]


class PredictionResponse(BaseModel):
    predicted_score: float


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Train it first with "
                "'python -m app.train'.",
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    model = get_model()
    row = pd.DataFrame([student.model_dump()])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    try:
        prediction = model.predict(row)[0]
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(predicted_score=round(float(prediction), 2))