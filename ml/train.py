"""Train a baseline linear regression model for diabetes progression.

This script trains a simple regression pipeline (StandardScaler + LinearRegression)
using the scikit-learn diabetes dataset. It outputs trained model artifacts,
feature metadata, and evaluation metrics (RMSE) to an output directory.
"""

import argparse
import json
import pathlib
import random
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from ml.data import load_dataset


def make_pipeline() -> Pipeline:
    """Create a baseline regression pipeline.

    The pipeline standardizes numeric features using StandardScaler
    and fits a LinearRegression model to predict diabetes progression.

    Returns:
        sklearn.pipeline.Pipeline: A fitted pipeline combining preprocessing and regression.
    """
    return Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])


def main():
    """Train and evaluate a baseline regression model.

    Workflow:
        1. Load and split the dataset into train/test sets.
        2. Build and train the pipeline.
        3. Evaluate model performance using RMSE.
        4. Save the trained model and metadata to disk.

    Command-line arguments:
        --seed (int): Random seed for reproducibility (default=42).
        --output-dir (str): Directory to store artifacts (default='artifacts').
        --test-size (float): Fraction of data reserved for testing (default=0.2).
        --version (str): Optional model version identifier (e.g., 'v0.1').

    Outputs:
        - pipeline.pkl: Serialized scikit-learn pipeline.
        - feature_names.json: List of feature columns.
        - metrics.json: Evaluation metrics dictionary.
        - MODEL_VERSION: (if provided) version tag written to file.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--version", default=None)
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load and split data
    x, y, feature_names = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed
    )

    # Train model
    pipe = make_pipeline()
    pipe.fit(x_train, y_train)

    # Evaluate
    preds = pipe.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Save artifacts
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(pipe, out / "pipeline.pkl")
    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    metrics = {
        "model": "linear",
        "seed": args.seed,
        "test_size": args.test_size,
        "rmse": rmse,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.version:
        (out / "MODEL_VERSION").write_text(args.version)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
