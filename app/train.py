"""Train a student exam-score predictor and save it to disk.

Usage:
    python -m app.train --data data/students.csv --out model.pkl
"""

import argparse
import logging

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_and_validate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s%(levelname)s%(name)s -%(message)s"
)
logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200, max_depth=8, random_state=42
                ),
            ),
        ]
    )


def train(data_path: str, output_path: str) -> float:
    df = load_and_validate(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    logger.info("Training on%d rows, validating on%d rows", len(X_train), len(X_val))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    logger.info("Validation MAE:%.3f", mae)

    joblib.dump(pipeline, output_path)
    logger.info("Model saved to%s", output_path)
    return mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/students.csv")
    parser.add_argument("--out", default="model.pkl")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()