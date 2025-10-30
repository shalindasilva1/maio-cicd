"""FastAPI app for the Virtual Diabetes Clinic Triage service."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from joblib import load
from pydantic import BaseModel, ConfigDict, ValidationError

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
VERSION_PATH = MODEL_DIR / "MODEL_VERSION"

DOCS_URL = os.getenv("DOCS_URL", "/docs")
REDOC_URL = os.getenv("REDOC_URL", "/redoc")
OPENAPI_URL = os.getenv("OPENAPI_URL", "/openapi.json")
if os.getenv("DISABLE_DOCS", "").lower() in {"1", "true", "yes", "y"}:
    DOCS_URL = None
    REDOC_URL = None

model_version = VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else "unknown"


class DiabetesFeatures(BaseModel):
    """Input schema for diabetes prediction."""
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
                "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response schema with the continuous progression score."""
    prediction: float
    model_config = ConfigDict(json_schema_extra={"example": {"prediction": 123.456}})


@lru_cache(maxsize=1)
def load_artifacts() -> Tuple[Any, List[str]]:
    """Load the trained pipeline and feature names once (cached).

    Returns:
        tuple: (pipeline, feature_names)
    """
    if not PIPELINE_PATH.exists() or not FEATURES_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail={"error": "Model artifacts missing. Train and bake artifacts into image."},
        )
    try:
        pipe = load(PIPELINE_PATH)
        feature_names: List[str] = json.loads(FEATURES_PATH.read_text())
        return pipe, feature_names
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to load model", "reason": str(exc)},
        ) from exc


app = FastAPI(
    title="Virtual Diabetes Triage Scorer",
    version=model_version,
    description=(
        "Predicts short-term diabetes progression index (higher = worse). "
        "Use the continuous score to prioritize nurse follow-ups."
    ),
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL,
    openapi_url=OPENAPI_URL,
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect to docs if enabled, else return a small status."""
    if DOCS_URL:
        return RedirectResponse(url=DOCS_URL)
    return {"status": "ok", "model_version": model_version, "docs": "disabled"}


@app.get("/health", tags=["ops"])
def health():
    """Liveness probe (does not force model load)."""
    return {"status": "ok", "model_version": model_version}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe that forces artifacts to be loaded."""
    load_artifacts()
    return {"status": "ready", "model_version": model_version}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={
        400: {"description": "Invalid payload"},
        500: {"description": "Model not available or internal error"},
    },
)
def predict(payload: Dict[str, Any]):
    """Predict the progression index for a single patient feature vector."""
    try:
        data = DiabetesFeatures(**payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid payload", "issues": exc.errors()},
        ) from exc

    pipe, feature_names = load_artifacts()
    row = [getattr(data, name) for name in feature_names]
    x = np.array([row], dtype=float)
    pred = float(pipe.predict(x)[0])
    return {"prediction": pred}
