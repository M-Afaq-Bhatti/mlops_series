"""Register the model from the current training run and promote it to
the 'production' alias. Run this ONLY after the quality gate has
passed and the change has been merged to main - never on a pull
request build.

Usage:
    python -m app.promote --run-id-file run_id.txt
"""

import argparse
import logging
from pathlib import Path

import mlflow

from app.train import EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s -%(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "student-score-predictor"


def promote(run_id_file: str) -> str:
    run_id = Path(run_id_file).read_text().strip()

    mlflow.set_experiment(EXPERIMENT_NAME)
    logged_models = mlflow.search_logged_models(
        experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id]
    )
    row = logged_models[logged_models["source_run_id"] == run_id].iloc[0]
    model_id = row["model_id"]

    registered = mlflow.register_model(
        model_uri=f"models:/{model_id}", name=REGISTERED_MODEL_NAME
    )
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "production", registered.version
    )
    logger.info(
        "Promoted run%s (registered version%s) to production.",
        run_id, registered.version,
    )
    return registered.version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id-file", default="run_id.txt")
    args = parser.parse_args()
    promote(args.run_id_file)


if __name__ == "__main__":
    main()
