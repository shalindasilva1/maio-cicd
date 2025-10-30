"""Data loading utilities for the Virtual Diabetes Clinic Triage project."""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch


def load_dataset(as_frame: bool = True):
    """Load and prepare the scikit-learn diabetes dataset."""
    xy: Bunch = load_diabetes(as_frame=as_frame)
    frame: pd.DataFrame = getattr(xy, "frame", None)
    if frame is None:
        raise ValueError("Expected a DataFrame when as_frame=True.")
    x = frame.drop(columns=["target"])
    y = frame["target"]
    feature_names = list(x.columns)
    return x, y, feature_names
