# 🌾 Crop Recommendation System

A machine learning web application that recommends the best crop to grow based on soil nutrients and climate conditions. Built with **scikit-learn**, **FastAPI**, and **Streamlit**.

**Live demo:** [crop-recommendation-systm.streamlit.app](https://crop-recommendation-systm.streamlit.app)

---

## What it does

Given 7 soil and climate inputs, the model predicts the most suitable crop from 22 possible classes with a confidence score.

| Input | Description | Range |
|-------|-------------|-------|
| N | Nitrogen content in soil (kg/ha) | 0 – 140 |
| P | Phosphorus content in soil (kg/ha) | 5 – 145 |
| K | Potassium content in soil (kg/ha) | 5 – 205 |
| Temperature | Temperature in Celsius | 8 – 44 |
| Humidity | Relative humidity (%) | 14 – 100 |
| pH | Soil pH value | 3.5 – 10 |
| Rainfall | Rainfall in mm | 20 – 300 |

---

## Project structure

```
Crop-Recommendation/
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # data loading, cleaning, scaling
│   ├── train.py            # training pipeline, saves artifacts
│   └── predict.py          # loads artifacts, serves predictions
├── api/
│   ├── __init__.py
│   └── app.py              # FastAPI REST API
├── data/
│   └── raw/
│       └── Crop_recommendation.csv
├── models/                 # generated after training (not committed)
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   ├── label_classes.json
│   └── metrics.json
├── notebooks/
│   └── Crop_recommendation.ipynb   # original exploration notebook
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/A7med-Foly/Crop-Recommendation.git
cd Crop-Recommendation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m src.train --data data/raw/Crop_recommendation.csv
```

This generates 4 files in `models/`:
- `crop_model.pkl` — trained Random Forest
- `scaler.pkl` — fitted MinMaxScaler
- `label_classes.json` — list of all 22 crop classes
- `metrics.json` — accuracy, F1, precision, recall

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Or run the REST API

```bash
uvicorn api.app:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

---

## REST API

### `GET /health`
Check the API is running.
```json
{"status": "ok"}
```

### `GET /classes`
List all supported crop classes.
```json
{"total": 22, "crops": ["apple", "banana", "coffee", ...]}
```

### `POST /predict`
Get a crop recommendation.

**Request:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

**Response:**
```json
{
  "crop": "rice",
  "confidence": 0.97,
  "message": "We recommend growing rice (confidence: 97%)"
}
```

---

## Model

| Model | Notes |
|-------|-------|
| **Naive Bayes** ✅ | Selected — best accuracy |
| Logistic Regression | Benchmarked |
| Decision Tree | Benchmarked |
| Gradient Boosting | Benchmarked |
| KNN | Benchmarked |
| Random Forest | Benchmarked |
| XGBoost | Benchmarked |

**Preprocessing:** MinMaxScaler on all 7 features via `ColumnTransformer`. Fitted on training data only to prevent data leakage.

**Split:** 80% train / 20% test, stratified by crop class (`random_state=42`).

---

## Dataset

**Crop Recommendation Dataset** by Atharva Ingle on Kaggle.

- 2,200 rows, 22 crop classes (100 samples each)
- Features: N, P, K, temperature, humidity, pH, rainfall
- Target: crop label (string)

[View on Kaggle →](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)


---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select repo → set main file: `streamlit_app.py`
4. Click **Deploy**

The app auto-trains the model on first run if no `models/` artifacts exist.

---

## Tech stack

- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, Pydantic, uvicorn
- **Frontend:** Streamlit
- **Python:** 3.10+