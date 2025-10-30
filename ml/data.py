"""Data loading utilities for the Virtual Diabetes Clinic Triage project."""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd
from sklearn.datasets import load_diabetes


def load_dataset(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and prepare the scikit-learn diabetes dataset.

    Args:
        as_frame: Return pandas objects if True.

    Returns:
        (X, y, feature_names)
    """
    x, y = load_diabetes(return_X_y=True, as_frame=as_frame)
    feature_names: List[str] = list(x.columns)
    return x, y, feature_names
