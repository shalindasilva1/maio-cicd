import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ValidationError, ConfigDict
from joblib import load

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
VERSION_PATH = MODEL_DIR / "MODEL_VERSION"

# Swagger / OpenAPI configuration via env (with sensible defaults)
DOCS_URL = os.getenv("DOCS_URL", "/docs")
REDOC_URL = os.getenv("REDOC_URL", "/redoc")
OPENAPI_URL = os.getenv("OPENAPI_URL", "/openapi.json")
# If DISABLE_DOCS is truthy, hide Swagger/Redoc entirely
if os.getenv("DISABLE_DOCS", "").lower() in ("1", "true", "yes", "y"):
    DOCS_URL = None
    REDOC_URL = None

# Lazy-loaded globals
pipe = None
feature_names = None
model_version = VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else "unknown"

class DiabetesFeatures(BaseModel):
    age: float; sex: float; bmi: float; bp: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
                "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
            }
        }
    )

class PredictionResponse(BaseModel):
    prediction: float
    model_config = ConfigDict(
        json_schema_extra={"example": {"prediction": 123.456}}
    )

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

app = FastAPI(
    title="Virtual Diabetes Triage Scorer",
    version=model_version,
    description=(
        "Predicts short-term diabetes progression index (higher = worse). "
        "Use the continuous score to prioritize nurse follow-ups.\n\n"
        "### Endpoints\n"
        "- **GET `/health`**: service status & model version\n"
        "- **POST `/predict`**: pass standardized diabetes features, get continuous risk score\n"
    ),
    contact={"name": "MLOps Team", "url": "https://github.com"},
    license_info={"name": "MIT"},
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL,
    openapi_url=OPENAPI_URL,
)

@app.get("/", include_in_schema=False)
def root():
    # Redirect root to Swagger UI if enabled; otherwise show a minimal message
    if DOCS_URL:
        return RedirectResponse(url=DOCS_URL)
    return {"status": "ok", "model_version": model_version, "docs": "disabled"}

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_version": model_version}

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={
        400: {
            "description": "Bad Request — invalid payload shape or types",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "Invalid payload",
                            "issues": [
                                {
                                    "type": "float_parsing",
                                    "loc": ["body", "age"],
                                    "msg": "Input should be a valid number"
                                }
                            ]
                        }
                    }
                }
            },
        },
        500: {
            "description": "Server Error — model not loaded or internal failure",
            "content": {
                "application/json": {
                    "example": {"detail": {"error": "Failed to load model", "reason": "..."}}  # noqa: E231
                }
            },
        },
    },
)
def predict(payload: Dict[str, Any]):
    try:
        data = DiabetesFeatures(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"error": "Invalid payload", "issues": e.errors()})

    _ensure_model_loaded()

    x = np.array([[getattr(data, fname) for fname in feature_names]], dtype=float)
    pred = float(pipe.predict(x)[0])
    return {"prediction": pred}