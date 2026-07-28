"""Train a student exam-score predictor on the prepared training split,
tracked with MLflow.

This is DVC pipeline stage 2: "train". It depends on the output of
app/prepare.py (data/prepared/train.csv). Training is wrapped in an
MLflow run so every hyperparameter, the model artifact, and (via
evaluate.py) the resulting metrics all live under one queryable run.

Usage:
    python -m app.train --data data/prepared/train.csv --out model.pkl
"""

import argparse
import logging

import joblib
import mlflow
import mlflow.sklearn
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

EXPERIMENT_NAME = "student-score-predictor"
RUN_ID_FILE = "run_id.txt"


def build_pipeline(n_estimators: int = 200, max_depth: int = 8) -> Pipeline:
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
                RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                ),
            ),
        ]
    )


def train(
    data_path: str,
    output_path: str,
    n_estimators: int = 200,
    max_depth: int = 8,
) -> Pipeline:
    df = pd.read_csv(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_train_rows", len(df))
        mlflow.log_param("model_type", "RandomForestRegressor")

        pipeline = build_pipeline(n_estimators, max_depth)
        logger.info("Training on%d rows from%s", len(df), data_path)
        pipeline.fit(X, y)

        joblib.dump(pipeline, output_path)
        mlflow.sklearn.log_model(pipeline, name="model")
        logger.info(
            "Model saved to%s and logged to MLflow run%s",
            output_path,
            run.info.run_id,
        )

        # Write the run id so evaluate.py (a separate process, in a
        # separate DVC stage) can log metrics into this same run.
        with open(RUN_ID_FILE, "w") as f:
            f.write(run.info.run_id)

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prepared/train.csv")
    parser.add_argument("--out", default="model.pkl")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()
    train(args.data, args.out, args.n_estimators, args.max_depth)


if __name__ == "__main__":
    main()
