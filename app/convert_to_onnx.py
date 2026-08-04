"""Convert the trained model to ONNX format.

Usage:
    python -m app.convert_to_onnx --model model.pkl --out model.onnx
"""
import argparse
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from app.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def convert(model_path: str, out_path: str) -> None:
    model = joblib.load(model_path)
    initial_types = [(c, FloatTensorType([None, 1])) for c in NUMERIC_FEATURES]
    initial_types += [(c, StringTensorType([None, 1])) for c in CATEGORICAL_FEATURES]
    onnx_model = convert_sklearn(model, initial_types=initial_types)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--out", default="model.onnx")
    args = parser.parse_args()
    convert(args.model, args.out)


if __name__ == "__main__":
    main()
