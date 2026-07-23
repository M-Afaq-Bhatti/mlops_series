"""Data loading and validation for the student score predictor.

This module is deliberately separate from training logic so it can be
unit-tested on its own (see tests/test_data.py) and reused by both the
training script and any future pipeline step.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "age",
    "study_hours",
    "class_attendance",
    "sleep_hours", 
]

CATEGORICAL_FEATURES = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

TARGET = "exam_score"

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


class DataValidationError(Exception):
    """Raised when the input dataset fails basic sanity checks."""


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV from disk.

    Raises:
        FileNotFoundError: if csv_path does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at{csv_path}. Download it from Kaggle first "
            "(see README) and place it at this path."
        )
    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset with shape%s from%s", df.shape, csv_path)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic data-quality checks before training.

    Raises:
        DataValidationError: if any check fails.
    """
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Dataset is missing required columns:{sorted(missing_cols)}. "
            "If your Kaggle CSV uses different column names, update "
            "NUMERIC_FEATURES / CATEGORICAL_FEATURES / TARGET in app/data.py."
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        bad_cols = df[REQUIRED_COLUMNS].columns[
            df[REQUIRED_COLUMNS].isnull().any()
        ].tolist()
        raise DataValidationError(f"Null values found in columns:{bad_cols}")

    if not df[TARGET].between(0, 100).all():
        raise DataValidationError(
            f"{TARGET} contains values outside the expected 0-100 range."
        )

    if len(df) < 10:
        raise DataValidationError(
            f"Dataset has only{len(df)} rows - too small to train on."
        )

    logger.info("Dataset passed validation:%d rows,%d columns", *df.shape)


def load_and_validate(csv_path: str | Path) -> pd.DataFrame:
    """Convenience wrapper used by train.py."""
    df = load_dataset(csv_path)
    validate_dataset(df)
    return df