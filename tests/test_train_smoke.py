"""Smoke tests for training pipeline."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile


def test_training_smoke():
    """Run training, assert artifacts and sane metrics exist."""
    tmp = tempfile.mkdtemp()
    try:
        cmd = [
            sys.executable,
            "-m",
            "ml.train",
            "--seed",
            "42",
            "--output-dir",
            tmp,
            "--version",
            "ci",
        ]
        out = subprocess.check_output(cmd, text=True)
        metrics = json.loads(out)

        assert "rmse" in metrics and metrics["rmse"] > 0
        assert (pathlib.Path(tmp) / "pipeline.pkl").exists()
        assert (pathlib.Path(tmp) / "feature_names.json").exists()
        assert (pathlib.Path(tmp) / "metrics.json").exists()
    finally:
        shutil.rmtree(tmp)
