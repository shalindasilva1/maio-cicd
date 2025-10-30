import json, pathlib, subprocess, sys, tempfile, shutil

def test_training_smoke():
    tmp = tempfile.mkdtemp()
    try:
        cmd = [sys.executable, "-m", "ml.train", "--seed", "42", "--output-dir", tmp, "--version", "ci"]
        out = subprocess.check_output(cmd, text=True)
        metrics = json.loads(out)
        assert "rmse" in metrics and metrics["rmse"] > 0
        assert (pathlib.Path(tmp) / "pipeline.pkl").exists()
        assert (pathlib.Path(tmp) / "feature_names.json").exists()
        assert (pathlib.Path(tmp) / "metrics.json").exists()
    finally:
        shutil.rmtree(tmp)
