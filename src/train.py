import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

from src.preprocess import (
    load_data, split_data,
    build_pipeline, fit_and_transform
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR       = "models"
MODEL_PATH       = os.path.join(MODELS_DIR, "crop_model.pkl")
SCALER_PATH      = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_PATH       = os.path.join(MODELS_DIR, "label_classes.json")
METRICS_PATH     = os.path.join(MODELS_DIR, "metrics.json")


# ── Model definition ──────────────────────────────────────────────────────────
def build_model() -> GaussianNB:
    return GaussianNB()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    """
    Run predictions on the test set and return a metrics dictionary.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall":    round(recall_score(y_test, y_pred,    average="weighted"), 4),
        "f1":        round(f1_score(y_test, y_pred,        average="weighted"), 4),
    }

    print("\n── Evaluation Results ───────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")

    print("\n── Per-class Report ─────────────────────────────")
    print(classification_report(y_test, y_pred))

    return metrics


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model, transformer, label_classes: list, metrics: dict):
    """
    Save everything needed to make predictions later.

    Why save all three?
      - crop_model.pkl      : the trained model
      - scaler.pkl          : MUST match the one used during training.
                              If you scale differently at serving time,
                              predictions will be garbage.
      - label_classes.json  : maps integer indices → crop names.
                              Needed by predict.py to return a readable name.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,       MODEL_PATH)
    joblib.dump(transformer, SCALER_PATH)

    with open(LABEL_PATH,   "w") as f:
        json.dump(label_classes, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Saved Artifacts ──────────────────────────────")
    print(f"  Model    → {MODEL_PATH}")
    print(f"  Scaler   → {SCALER_PATH}")
    print(f"  Labels   → {LABEL_PATH}")
    print(f"  Metrics  → {METRICS_PATH}")


# ── Main training pipeline ────────────────────────────────────────────────────
def train(csv_path: str):
    """
    Full training pipeline — one function, runs top to bottom.
    Each step prints progress so you know exactly where you are.
    """
    print("── Step 1: Load data ─────────────────────────────")
    X, y = load_data(csv_path)

    print("\n── Step 2: Split ─────────────────────────────────")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n── Step 3: Scale features ────────────────────────")
    transformer = build_pipeline()
    X_train_s, X_test_s = fit_and_transform(transformer, X_train, X_test)

    print("\n── Step 4: Train GaussianNB ────────────────────")
    model = build_model()
    model.fit(X_train_s, y_train)
    print(f"✓ Trained on {len(X_train_s)} samples")

    print("\n── Step 5: Evaluate ──────────────────────────────")
    metrics = evaluate(model, X_test_s, y_test)

    print("\n── Step 6: Save ──────────────────────────────────")
    label_classes = sorted(y.unique().tolist())   # e.g. ['apple','banana',...]
    save_artifacts(model, transformer, label_classes, metrics)

    print("\n✓ Training complete!\n")
    return model, transformer


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the crop recommendation model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/Crop_recommendation.csv",
        help="Path to the raw CSV file"
    )
    args = parser.parse_args()
    train(args.data)