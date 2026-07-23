import pandas as pd
import pytest

from app.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    DataValidationError,
    load_dataset,
    validate_dataset,
)


def make_valid_df(n_rows: int = 20) -> pd.DataFrame:
    data = {col: [10] * n_rows for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = ["High"] * n_rows
    data[TARGET] = [75.0] * n_rows
    return pd.DataFrame(data)


def test_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/does_not_exist.csv")


def test_load_dataset_success(tmp_path):
    csv_path = tmp_path / "students.csv"
    make_valid_df().to_csv(csv_path, index=False)
    df = load_dataset(csv_path)
    assert len(df) == 20


def test_validate_dataset_passes_on_good_data():
    df = make_valid_df()
    validate_dataset(df)  # should not raise


def test_validate_dataset_missing_columns_raises():
    df = make_valid_df().drop(columns=["study_hours"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_nulls_raise():
    df = make_valid_df()
    df.loc[0, "class_attendance"] = None
    with pytest.raises(DataValidationError, match="Null values"):
        validate_dataset(df)


def test_validate_dataset_out_of_range_target_raises():
    df = make_valid_df()
    df.loc[0, TARGET] = 150
    with pytest.raises(DataValidationError, match="0-100 range"):
        validate_dataset(df)


def test_validate_dataset_too_few_rows_raises():
    df = make_valid_df(n_rows=3)
    with pytest.raises(DataValidationError, match="too small"):
        validate_dataset(df)