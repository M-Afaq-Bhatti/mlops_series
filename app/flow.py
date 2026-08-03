"""Prefect orchestration for the Student Score Predictor pipeline.

Wraps prepare -> train -> evaluate -> quality_gate -> (conditional) promote
as a single flow, replacing manually running `dvc repro` + the gate +
promote script by hand. This is what a scheduler (or CI) triggers.

Usage:
    python -m app.flow --data data/students.csv
"""

import argparse

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

from app.evaluate import evaluate as evaluate_fn
from app.prepare import prepare as prepare_fn
from app.promote import promote as promote_fn
from app.quality_gate import run_gate
from app.train import train as train_fn

DATA_DIR = "data/prepared"
MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"
RUN_ID_FILE = "run_id.txt"


@task(retries=2, retry_delay_seconds=5, log_prints=True)
def prepare_task(raw_data_path: str) -> None:
    logger = get_run_logger()
    logger.info("Preparing data from%s", raw_data_path)
    prepare_fn(raw_data_path, DATA_DIR)


@task(retries=1)
def train_task(n_estimators: int, max_depth: int) -> None:
    logger = get_run_logger()
    logger.info("Training with n_estimators=%d, max_depth=%d", n_estimators, max_depth)
    train_fn(f"{DATA_DIR}/train.csv", MODEL_PATH, n_estimators, max_depth)


@task
def evaluate_task() -> dict:
    logger = get_run_logger()
    metrics = evaluate_fn(MODEL_PATH, f"{DATA_DIR}/val.csv", METRICS_PATH)
    logger.info("Evaluation metrics:%s", metrics)
    return metrics


@task
def quality_gate_task(max_mae_increase: float) -> bool:
    logger = get_run_logger()
    passed = run_gate(METRICS_PATH, max_mae_increase)
    logger.info("Quality gate passed:%s", passed)
    return passed


@task
def promote_task() -> str:
    logger = get_run_logger()
    version = promote_fn(RUN_ID_FILE)
    logger.info("Promoted to production version%s", version)
    return version


@flow(name="student-score-training-pipeline", task_runner=ThreadPoolTaskRunner(max_workers=1))
def training_pipeline(
    raw_data_path: str = "data/students.csv",
    n_estimators: int = 200,
    max_depth: int = 8,
    max_mae_increase: float = 0.5,
    auto_promote: bool = True,
) -> dict:
    logger = get_run_logger()

    prepare_task(raw_data_path)
    train_task(n_estimators, max_depth)
    metrics = evaluate_task()
    passed = quality_gate_task(max_mae_increase)

    if passed and auto_promote:
        promote_task()
    elif not passed:
        logger.error("Quality gate failed - skipping promotion. Metrics:%s", metrics)
    else:
        logger.info("auto_promote=False - gate passed but promotion skipped by request.")

    return {"metrics": metrics, "gate_passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/students.csv")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-mae-increase", type=float, default=0.5)
    parser.add_argument("--no-auto-promote", action="store_true")
    args = parser.parse_args()

    result = training_pipeline(
        raw_data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_mae_increase=args.max_mae_increase,
        auto_promote=not args.no_auto_promote,
    )
    print(result)


if __name__ == "__main__":
    main()
