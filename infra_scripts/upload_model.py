"""Upload the trained model artifact to S3, versioned by run ID.

Usage:
    python infra_scripts/upload_model.py --bucket my-bucket --run-id abc123 --model-path model.pkl
"""

import argparse
import logging

import boto3

logging.basicConfig(level=logging.INFO, format="%(levelname)s -%(message)s")
logger = logging.getLogger(__name__)


def upload_model(bucket: str, run_id: str, model_path: str) -> str:
    s3 = boto3.client("s3")
    key = f"models/{run_id}/model.pkl"
    s3.upload_file(model_path, bucket, key)
    logger.info("Uploaded%s to s3://%s/%s", model_path, bucket, key)
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", default="model.pkl")
    args = parser.parse_args()
    upload_model(args.bucket, args.run_id, args.model_path)


if __name__ == "__main__":
    main()
