import os
import json
import time
import logging
import mlflow.sklearn
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SCALER_PARAMS_PATH  = "data/processed/scaler_params.json"
MODEL_URI = "file:///app/mlruns/1/models/m-7b1f18cf4db346478afca1d43df00147/artifacts"
NUMERIC_COLS        = ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak']
CATEGORICAL_COLS    = ['cp', 'restecg', 'slope', 'ca', 'thal']

PREDICTION_COUNT    = Counter(
    "prediction_total",
    "Total number of predictions made",
    ["result"]
)
PREDICTION_LATENCY  = Histogram(
    "prediction_latency_seconds",
    "Time taken for a prediction request"
)
REQUEST_COUNT       = Counter(
    "request_total",
    "Total number of requests received",
    ["endpoint"]
)
ERROR_COUNT         = Counter(
    "error_total",
    "Total number of errors",
    ["endpoint"]
)

app = FastAPI(
    title="Heart Disease Diagnostic System",
    description="Predicts whether a patient is at risk of heart disease based on clinical parameters.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   
        "http://localhost:5173",   
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3002",
        "http://localhost:3002",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

model         = None
scaler_params = None

@app.on_event("startup")
def load_model_and_scaler():
    global model, scaler_params

    logger.info("Loading model from MLflow registry...")
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info(f"Model loaded from: {MODEL_URI}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")

    logger.info("Loading scaler params...")
    try:
        with open(SCALER_PARAMS_PATH, "r") as f:
            scaler_params = json.load(f)
        logger.info("Scaler params loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load scaler params: {e}")
        raise RuntimeError(f"Scaler params load failed: {e}")


class PatientInput(BaseModel):
    age:      float = Field(..., ge=1,  le=120, description="Age of the patient")
    sex:      int   = Field(..., ge=0,  le=1,   description="1 = male, 0 = female")
    cp:       int   = Field(..., ge=0,  le=3,   description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol:     float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,  le=1,   description="Fasting blood sugar > 120 mg/dl")
    restecg:  int   = Field(..., ge=0,  le=2,   description="Resting ECG results (0-2)")
    thalachh: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang:    int   = Field(..., ge=0,  le=1,   description="Exercise-induced angina")
    oldpeak:  float = Field(..., ge=0,  le=10,  description="ST depression")
    slope:    int   = Field(..., ge=0,  le=2,   description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,  le=3,   description="Number of major vessels (0-3)")
    thal:     int   = Field(..., ge=1,  le=3,   description="Thalassemia type (1-3)")



class PredictionOutput(BaseModel):
    prediction:  int   
    probability: float 
    result:      str   


def preprocess_input(data: PatientInput) -> pd.DataFrame:
    raw = {
        "age":      data.age,
        "sex":      data.sex,
        "cp":       data.cp,
        "trestbps": data.trestbps,
        "chol":     data.chol,
        "fbs":      data.fbs,
        "restecg":  data.restecg,
        "thalachh": data.thalachh,
        "exang":    data.exang,
        "oldpeak":  data.oldpeak,
        "slope":    data.slope,
        "ca":       data.ca,
        "thal":     data.thal,
    }
    df = pd.DataFrame([raw])

    for col in NUMERIC_COLS:
        min_val = scaler_params[col]["min"]
        max_val = scaler_params[col]["max"]
        df[col] = (df[col] - min_val) / (max_val - min_val)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)

    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    return df


@app.get("/health")
def health():
    """Liveness check — is the server running?"""
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness check — is the model loaded and ready to serve?"""
    REQUEST_COUNT.labels(endpoint="/ready").inc()
    if model is None or scaler_params is None:
        ERROR_COUNT.labels(endpoint="/ready").inc()
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    """
    Takes patient clinical parameters and returns heart disease risk prediction.
    Measures request count, errors, and latency for every request.
    """
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start_time = time.time()
    try:
        logger.info(f"Received prediction request: {patient.dict()}")

        df = preprocess_input(patient)

        prediction = int(model.predict(df)[0])
        probability = round(float(model.predict_proba(df)[0][1]), 4)
        result = "High Risk" if prediction == 1 else "Low Risk"


        PREDICTION_COUNT.labels(result=result).inc()

        logger.info(f"Prediction: {result}, Probability: {probability}")

        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            result=result
        )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    REQUEST_COUNT.labels(endpoint="/metrics").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
