"""CI quality gate: compare the newly-trained model's metrics against
whatever is currently registered as "production" in MLflow, and fail
(non-zero exit code) if the new model is meaningfully worse.

This is what GitHub Actions runs after training+evaluating on a pull
request, to decide whether the PR is allowed to merge.

Usage:
    python -m app.quality_gate --new-metrics metrics.json --max-mae-increase 0.5
"""

import argparse
import json
import logging
import sys

import mlflow

from app.train import EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s -%(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "student-score-predictor"


def get_production_mae() -> float | None:
    """Return the MAE of the model currently aliased 'production', or
    None if nothing has been registered yet (e.g. this is the very
    first run ever)."""
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
    except mlflow.exceptions.MlflowException:
        logger.info("No production model registered yet - nothing to compare against.")
        return None

    run = client.get_run(mv.run_id)
    return run.data.metrics.get("mae")


def run_gate(new_metrics_path: str, max_mae_increase: float) -> bool:
    with open(new_metrics_path) as f:
        new_metrics = json.load(f)
    new_mae = new_metrics["mae"]

    mlflow.set_experiment(EXPERIMENT_NAME)
    prod_mae = get_production_mae()

    if prod_mae is None:
        logger.info("New model MAE:%.4f. No baseline to compare against - passing.", new_mae)
        return True

    allowed_mae = prod_mae + max_mae_increase
    logger.info(
        "Production MAE:%.4f | New MAE:%.4f | Allowed up to:%.4f",
        prod_mae, new_mae, allowed_mae,
    )

    if new_mae > allowed_mae:
        logger.error(
            "QUALITY GATE FAILED: new MAE%.4f exceeds production MAE%.4f "
            "by more than the allowed%.4f increase.",
            new_mae, prod_mae, max_mae_increase,
        )
        return False

    logger.info("QUALITY GATE PASSED: new model is not meaningfully worse than production.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-metrics", default="metrics.json")
    parser.add_argument("--max-mae-increase", type=float, default=0.5)
    args = parser.parse_args()

    passed = run_gate(args.new_metrics, args.max_mae_increase)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
