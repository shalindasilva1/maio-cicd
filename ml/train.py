import argparse, json, os, pathlib, random
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from ml.data import load_dataset

def make_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--version", default=None)  # e.g., v0.1
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dump(pipe, out / "pipeline.pkl")
    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    metrics = {
        "model": "linear",
        "seed": args.seed,
        "test_size": args.test_size,
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.version:
        (out / "MODEL_VERSION").write_text(args.version)

    print(json.dumps(metrics))

if __name__ == "__main__":
    main()
