"""Train a regression model for diabetes progression (v0.1/v0.2 compatible)."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import numpy as np
from joblib import dump
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import load_dataset


def make_pipeline(model: str) -> Pipeline:
    """Create a regression pipeline."""
    if model == "linear":
        return Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    if model == "ridge":
        # modest L2 regularization for v0.2 improvement
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))])
    raise ValueError(f"Unknown model: {model}")


def main():
    """Train, evaluate, and save artifacts (pipeline, features, metrics)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "ridge"], default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipe = make_pipeline(args.model)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dump(pipe, out / "pipeline.pkl")
    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    metrics = {
        "model": args.model,
        "seed": args.seed,
        "test_size": args.test_size,
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.version:
        (out / "MODEL_VERSION").write_text(args.version)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
