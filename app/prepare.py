"""Split the raw, validated dataset into train/val CSVs.

This is DVC pipeline stage 1: "prepare". Keeping the split as its own
stage (instead of doing it inside train.py) means DVC can cache it
separately - if you only change a training hyperparameter later,
DVC will skip re-running this stage entirely.

Usage:
    python -m app.prepare --data data/students.csv --out-dir data/prepared
"""

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.data import load_and_validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s%(levelname)s%(name)s -%(message)s")
logger = logging.getLogger(__name__)


def prepare(data_path: str, out_dir: str, test_size: float = 0.2, seed: int = 42) -> None:
    df = load_and_validate(data_path)

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    train_path = out_dir_path / "train.csv"
    val_path = out_dir_path / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info("Wrote %d training rows to %s", len(train_df), train_path)
    logger.info("Wrote %d validation rows to %s", len(val_df), val_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/students.csv")
    parser.add_argument("--out-dir", default="data/prepared")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data, args.out_dir, args.test_size, args.seed)


if __name__ == "__main__":
    main()
