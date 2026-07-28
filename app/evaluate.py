"""Evaluate a trained model against the validation split, log metrics to
MLflow (into the same run as training), and write metrics.json for DVC.

This is DVC pipeline stage 3: "evaluate".

Usage:
    python -m app.evaluate --model model.pkl --data data/prepared/val.csv --out metrics.json
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from app.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from app.train import EXPERIMENT_NAME, RUN_ID_FILE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s%(levelname)s%(name)s -%(message)s"
)
logger = logging.getLogger(__name__)


def evaluate(model_path: str, data_path: str, out_path: str) -> dict:
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    preds = model.predict(X)
    metrics = {
        "mae": round(float(mean_absolute_error(y, preds)), 4),
        "r2": round(float(r2_score(y, preds)), 4),
        "n_val_rows": len(df),
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation metrics:%s", metrics)
    logger.info("Metrics written to%s", out_path)

    # Log into the SAME MLflow run that training created, so params,
    # the model artifact, and metrics all live under one run.
    run_id_path = Path(RUN_ID_FILE)
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.log_metric("n_val_rows", metrics["n_val_rows"])
        logger.info("Metrics logged to MLflow run%s", run_id)
    else:
        logger.warning("%s not found - skipping MLflow metric logging", RUN_ID_FILE)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--data", default="data/prepared/val.csv")
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()
    evaluate(args.model, args.data, args.out)


if __name__ == "__main__":
    main()
