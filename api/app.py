"""
app.py
------
FastAPI application that serves the crop recommendation model.

Run it with:
    uvicorn api.app:app --reload

Then open:
    http://localhost:8000/docs      ← interactive UI to test predictions
    http://localhost:8000/health    ← check the API is alive
    http://localhost:8000/classes   ← see all 22 crop classes
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
import os

from src.predict import predict_crop

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Recommendation API",
    description="Recommends the best crop based on soil and climate conditions.",
    version="1.0.0",
)


# ── Request schema ────────────────────────────────────────────────────────────
# Pydantic validates every incoming request automatically.
# If someone sends temperature="hot" instead of a float,
# FastAPI rejects it with a clear 422 error before it ever hits your model.

class CropInput(BaseModel):
    N:           float = Field(..., ge=0,   le=140, description="Nitrogen content (kg/ha)")
    P:           float = Field(..., ge=5,   le=145, description="Phosphorus content (kg/ha)")
    K:           float = Field(..., ge=5,   le=205, description="Potassium content (kg/ha)")
    temperature: float = Field(..., ge=8,   le=44,  description="Temperature in Celsius")
    humidity:    float = Field(..., ge=14,  le=100, description="Relative humidity (%)")
    ph:          float = Field(..., ge=3.5, le=10,  description="Soil pH value")
    rainfall:    float = Field(..., ge=20,  le=300, description="Rainfall in mm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "N": 90, "P": 42, "K": 43,
                "temperature": 20.8,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.9
            }
        }
    }


# ── Response schema ───────────────────────────────────────────────────────────
class CropOutput(BaseModel):
    crop:       str   = Field(..., description="Recommended crop name")
    confidence: float = Field(..., description="Model confidence (0–1)")
    message:    str   = Field(..., description="Human-readable summary")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    """Check that the API is running."""
    return {"status": "ok"}


@app.get("/classes", tags=["Utility"])
def get_classes():
    """Return all crop classes the model knows about."""
    label_path = os.path.join("models", "label_classes.json")

    if not os.path.exists(label_path):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python -m src.train"
        )

    with open(label_path) as f:
        classes = json.load(f)

    return {"total": len(classes), "crops": classes}


@app.post("/predict", response_model=CropOutput, tags=["Prediction"])
def recommend_crop(data: CropInput):
    """
    Recommend the best crop for the given soil and climate conditions.

    - **N**           : Nitrogen content in soil (kg/ha)
    - **P**           : Phosphorus content in soil (kg/ha)
    - **K**           : Potassium content in soil (kg/ha)
    - **temperature** : Temperature in Celsius
    - **humidity**    : Relative humidity in %
    - **ph**          : pH value of the soil
    - **rainfall**    : Rainfall in mm
    """
    try:
        result = predict_crop(
            N=data.N,
            P=data.P,
            K=data.K,
            temperature=data.temperature,
            humidity=data.humidity,
            ph=data.ph,
            rainfall=data.rainfall,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return CropOutput(
        crop=result["crop"],
        confidence=result["confidence"],
        message=f"We recommend growing {result['crop']} "
                f"(confidence: {result['confidence']*100:.0f}%)"
    )