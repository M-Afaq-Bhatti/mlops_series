"""Train a student exam-score predictor on the prepared training split.

This is DVC pipeline stage 2: "train". It depends on the output of
app/prepare.py (data/prepared/train.csv), not the raw dataset - so DVC
only re-runs this stage when the prepared training data or this script
actually changes.

Usage:
    python -m app.train --data data/prepared/train.csv --out model.pkl
"""

import argparse
import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s%(levelname)s%(name)s -%(message)s"
)
logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
            ),
        ]
    )


def train(data_path: str, output_path: str) -> Pipeline:
    df = pd.read_csv(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    pipeline = build_pipeline()
    logger.info("Training on%d rows from%s", len(df), data_path)
    pipeline.fit(X, y)

    joblib.dump(pipeline, output_path)
    logger.info("Model saved to%s", output_path)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prepared/train.csv")
    parser.add_argument("--out", default="model.pkl")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
