import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from joblib import load

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
VERSION_PATH = MODEL_DIR / "MODEL_VERSION"

# Lazy-loaded globals
pipe = None
feature_names = None
model_version = VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else "unknown"

class DiabetesFeatures(BaseModel):
    age: float; sex: float; bmi: float; bp: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float

def _ensure_model_loaded():
    global pipe, feature_names
    if pipe is None or feature_names is None:
        if not PIPELINE_PATH.exists() or not FEATURES_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail={"error": "Model artifacts missing. Train and bake artifacts into image."}
            )
        try:
            fns = json.loads(FEATURES_PATH.read_text())
            model = load(PIPELINE_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": "Failed to load model", "reason": str(e)})
        feature_names = fns
        pipe = model

app = FastAPI(title="Virtual Diabetes Triage Scorer", version=model_version)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_version}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    try:
        data = DiabetesFeatures(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"error": "Invalid payload", "issues": e.errors()})

    _ensure_model_loaded()

    x = np.array([[getattr(data, fname) for fname in feature_names]], dtype=float)
    pred = float(pipe.predict(x)[0])
    return {"prediction": pred}