"""Smoke tests for model training pipeline.

These tests ensure that the training script runs end-to-end,
produces output artifacts, and returns valid evaluation metrics.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import shutil


def test_training_smoke():
    """Run a minimal training pipeline smoke test.

    This test verifies that the training script:
      - Executes successfully from the command line.
      - Produces expected output files (model, features, metrics).
      - Outputs a valid metrics dictionary with a positive RMSE.

    The test uses a temporary directory for artifacts and cleans it up after execution.
    """
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

        assert "rmse" in metrics and metrics["rmse"] > 0, "RMSE missing or invalid"
        assert (pathlib.Path(tmp) / "pipeline.pkl").exists(), "Model file not found"
        assert (pathlib.Path(tmp) / "feature_names.json").exists(), "Feature list missing"
        assert (pathlib.Path(tmp) / "metrics.json").exists(), "Metrics file missing"
    finally:
        shutil.rmtree(tmp)
