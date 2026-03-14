import os
import json
import joblib
import pandas as pd

from src.preprocess import FEATURE_COLS

# ── Paths (must match what train.py saved) ────────────────────────────────────
MODELS_DIR   = "models"
MODEL_PATH   = os.path.join(MODELS_DIR, "crop_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_PATH   = os.path.join(MODELS_DIR, "label_classes.json")


# ── Lazy loading ──────────────────────────────────────────────────────────────
# We load the model ONCE when the module is first imported, not on every call.
# This means the FastAPI app loads it at startup — not on every HTTP request.
# Loading a model on every request would make the API ~10x slower.

_model       = None
_scaler      = None
_labels      = None


def _load_artifacts():
    """Load model artifacts from disk (called once on first prediction)."""
    global _model, _scaler, _labels

    if _model is not None:
        return  # already loaded, skip

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model found at '{MODEL_PATH}'.\n"
            f"Run training first:  python -m src.train"
        )

    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

    with open(LABEL_PATH) as f:
        _labels = json.load(f)

    print(f"✓ Model loaded — {len(_labels)} crop classes")


# ── Main prediction function ──────────────────────────────────────────────────
def predict_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
) -> dict:
    """
    Predict the best crop for the given soil and climate conditions.

    Args:
        N           : Nitrogen content in soil (kg/ha)
        P           : Phosphorus content in soil (kg/ha)
        K           : Potassium content in soil (kg/ha)
        temperature : Temperature in Celsius
        humidity    : Relative humidity in %
        ph          : pH value of the soil
        rainfall    : Rainfall in mm

    Returns:
        dict with:
          - 'crop'        : recommended crop name  (e.g. 'rice')
          - 'confidence'  : model's confidence 0–1 (e.g. 0.97)

    Example:
        >>> predict_crop(90, 42, 43, 20.8, 82.0, 6.5, 202.9)
        {'crop': 'rice', 'confidence': 0.97}
    """
    _load_artifacts()

    # Build a single-row DataFrame — MUST use FEATURE_COLS order
    # so the scaler applies the right min/max to the right column
    input_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=FEATURE_COLS
    )

    # Scale using the SAME scaler fitted during training
    input_scaled = _scaler.transform(input_df)

    # Predict class and probability
    predicted_index = _model.predict(input_scaled)[0]
    probabilities   = _model.predict_proba(input_scaled)[0]
    confidence      = round(float(probabilities.max()), 4)

    # predicted_index is already a crop name (e.g. 'rice') for Naive Bayes
    # because we trained it on string labels — no LabelEncoder needed
    crop = predicted_index

    return {
        "crop":       crop,
        "confidence": confidence
    }