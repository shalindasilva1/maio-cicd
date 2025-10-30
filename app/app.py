"""FastAPI app for the Virtual Diabetes Clinic Triage service.

This service predicts a continuous diabetes progression index from
the standard scikit-learn diabetes dataset features. The endpoint `/predict`
accepts a JSON payload of normalized features and returns a numeric score.
"""

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

# Swagger / OpenAPI configuration via environment variables
DOCS_URL = os.getenv("DOCS_URL", "/docs")
REDOC_URL = os.getenv("REDOC_URL", "/redoc")
OPENAPI_URL = os.getenv("OPENAPI_URL", "/openapi.json")

# Disable documentation endpoints if requested
if os.getenv("DISABLE_DOCS", "").lower() in ("1", "true", "yes", "y"):
    DOCS_URL = None
    REDOC_URL = None

# Lazy-loaded globals
pipe = None
feature_names = None
model_version = VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else "unknown"


class DiabetesFeatures(BaseModel):
    """Input schema for the diabetes prediction model.

    Attributes:
        age (float): Normalized patient age.
        sex (float): Encoded sex indicator.
        bmi (float): Body Mass Index (normalized).
        bp (float): Mean arterial blood pressure.
        s1–s6 (float): Serum and lipid measurements (normalized).
    """
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
    """Response schema for predictions.

    Attributes:
        prediction (float): The predicted progression index (higher = worse).
    """
    prediction: float

    model_config = ConfigDict(
        json_schema_extra={"example": {"prediction": 123.456}}
    )


class ModelState:
    """Manage shared model cache for pipeline and feature names."""

    pipe = None
    feature_names = None

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if the model has already been loaded."""
        return cls.pipe is not None and cls.feature_names is not None

    @classmethod
    def load(cls):
        """Load model artifacts into memory if not already loaded."""
        if cls.is_loaded():
            return
        if not PIPELINE_PATH.exists() or not FEATURES_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail={"error": "Model artifacts missing."}
            )
        try:
            cls.feature_names = json.loads(FEATURES_PATH.read_text())
            cls.pipe = load(PIPELINE_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to load model", "reason": str(e)}
            ) from e


def _ensure_model_loaded():
    """Ensure the model is loaded into memory."""
    ModelState.load()



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
    """Redirect to the Swagger UI or return minimal health info.

    Returns:
        RedirectResponse | dict: Redirects to Swagger if enabled,
        otherwise returns a simple JSON response.
    """
    if DOCS_URL:
        return RedirectResponse(url=DOCS_URL)
    return {"status": "ok", "model_version": model_version, "docs": "disabled"}


@app.get("/health", tags=["ops"])
def health():
    """Health endpoint showing service status and model version.

    Returns:
        dict: `{ "status": "ok", "model_version": "..." }`
    """
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
                    "example": {"detail": {"error": "Failed to load model", "reason": "..."}}
                }
            },
        },
    },
)
def predict(payload: Dict[str, Any]):
    """Predict diabetes progression risk for a given patient feature set.

    Args:
        payload (Dict[str, Any]): JSON mapping of feature names to numeric values.

    Returns:
        dict: A JSON object with a single field `"prediction"` containing the model’s output.
    """
    try:
        data = DiabetesFeatures(**payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid payload", "issues": e.errors()}
        ) from e

    _ensure_model_loaded()

    x = np.array([[getattr(data, fname) for fname in list(feature_names)]], dtype=float)
    pred = float(pipe.predict(x)[0])
    return {"prediction": pred}
