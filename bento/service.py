"""BentoML service for the student score predictor.

Alternative to the raw FastAPI app in app/main.py - BentoML bundles
the model, preprocessing, and serving code into one deployable "Bento"
and generates the Docker image for you, instead of you hand-writing
a Dockerfile.
"""

import bentoml
import joblib
import pandas as pd

from app.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@bentoml.service(resources={"cpu": "1"}, traffic={"timeout": 10})
class StudentScorePredictor:
    def __init__(self) -> None:
        self.model = joblib.load("model.pkl")

    @bentoml.api
    def predict(self, student: dict) -> dict:
        row = pd.DataFrame([student])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        prediction = self.model.predict(row)[0]
        return {"predicted_score": round(float(prediction), 2)}
