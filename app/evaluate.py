"""Evaluate a trained model against the validation split and write metrics.

This is DVC pipeline stage 3: "evaluate". DVC treats metrics.json as a
tracked metrics file, which lets you run `dvc metrics show` and
`dvc metrics diff` to compare runs without opening MLflow (that
comparison layer comes in Module 4).

Usage:
    python -m app.evaluate --model model.pkl --data data/prepared/val.csv --out metrics.json
"""

import argparse
import json
import logging

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from app.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

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

    with open(out_path, "w", newline="\n") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation metrics:%s", metrics)
    logger.info("Metrics written to%s", out_path)
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
