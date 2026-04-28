# Low Level Design (LLD) — Heart Disease Diagnostic System

## 1. API Endpoint Definitions

Base URL: `http://localhost:8000`

---

### 1.1 GET /health

**Description:** Liveness check — confirms the server is running.

**Request:**
```
GET /health
```

**Response — 200 OK:**
```json
{
  "status": "ok"
}
```

---

### 1.2 GET /ready

**Description:** Readiness check — confirms the model and scaler are loaded and ready.

**Request:**
```
GET /ready
```

**Response — 200 OK:**
```json
{
  "status": "ready"
}
```

**Response — 503 Service Unavailable:**
```json
{
  "detail": "Model not ready"
}
```

---

### 1.3 POST /predict

**Description:** Takes patient clinical parameters and returns a heart disease risk prediction.

**Request:**
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "age":      63,
  "sex":      1,
  "cp":       3,
  "trestbps": 145,
  "chol":     233,
  "fbs":      1,
  "restecg":  0,
  "thalachh": 150,
  "exang":    0,
  "oldpeak":  2.3,
  "slope":    0,
  "ca":       0,
  "thal":     1
}
```

**Input Field Specifications:**

| Field | Type | Range | Description |
|---|---|---|---|
| age | float | 1–120 | Age of patient in years |
| sex | int | 0 or 1 | 1 = male, 0 = female |
| cp | int | 0–3 | Chest pain type: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| trestbps | float | 50–250 | Resting blood pressure in mm Hg |
| chol | float | 50–700 | Serum cholesterol in mg/dl |
| fbs | int | 0 or 1 | Fasting blood sugar > 120 mg/dl: 1=True, 0=False |
| restecg | int | 0–2 | Resting ECG: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| thalachh | float | 50–250 | Maximum heart rate achieved in bpm |
| exang | int | 0 or 1 | Exercise-induced angina: 1=Yes, 0=No |
| oldpeak | float | 0–10 | ST depression induced by exercise |
| slope | int | 0–2 | Slope of peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping |
| ca | int | 0–3 | Number of major vessels colored by fluoroscopy |
| thal | int | 1–3 | Thalassemia: 1=Normal, 2=Fixed defect, 3=Reversible defect |

**Response — 200 OK:**
```json
{
  "prediction":  1,
  "probability": 0.87,
  "result":      "High Risk"
}
```

**Output Field Specifications:**

| Field | Type | Values | Description |
|---|---|---|---|
| prediction | int | 0 or 1 | 0 = Low Risk, 1 = High Risk |
| probability | float | 0.0–1.0 | Probability of heart disease |
| result | string | "High Risk" or "Low Risk" | Human readable result |

**Response — 422 Unprocessable Entity (validation error):**
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is less than or equal to 120",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**Response — 500 Internal Server Error:**
```json
{
  "detail": "error message"
}
```

---

### 1.4 GET /metrics

**Description:** Prometheus metrics scrape endpoint. Returns all instrumented metrics in Prometheus text format.

**Request:**
```
GET /metrics
```

**Response — 200 OK:**
```
# HELP prediction_total Total number of predictions made
# TYPE prediction_total counter
prediction_total{result="High Risk"} 10.0
prediction_total{result="Low Risk"} 5.0

# HELP prediction_latency_seconds Time taken for a prediction request
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.005"} 3.0
...

# HELP request_total Total number of requests received
# TYPE request_total counter
request_total{endpoint="/predict"} 15.0
request_total{endpoint="/health"} 2.0

# HELP error_total Total number of errors
# TYPE error_total counter
error_total{endpoint="/predict"} 0.0
```

---

## 2. Preprocessing Pipeline — Function Specifications

### `load_data(filepath: str) -> pd.DataFrame`
- **Input:** Path to raw CSV file
- **Output:** DataFrame with `?` replaced by NaN
- **Side effects:** Prints row/column count and missing value count

### `drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame`
- **Input:** DataFrame with NaN values
- **Output:** DataFrame with all NaN rows removed
- **Side effects:** Prints number of rows dropped

### `convert_types(df: pd.DataFrame) -> pd.DataFrame`
- **Input:** DataFrame post missing row removal
- **Output:** DataFrame with all columns cast to numeric
- **Side effects:** Drops rows with invalid types, prints count

### `validate_ranges(df: pd.DataFrame) -> pd.DataFrame`
- **Input:** DataFrame post type conversion
- **Output:** DataFrame with only medically valid rows
- **Side effects:** Prints number of out-of-range rows dropped

### `normalize_numeric(df: pd.DataFrame) -> pd.DataFrame`
- **Input:** DataFrame post validation
- **Output:** DataFrame with numeric columns scaled to [0, 1]
- **Side effects:** Saves `scaler_params.json` to disk

### `one_hot_encode(df: pd.DataFrame) -> pd.DataFrame`
- **Input:** DataFrame post normalization
- **Output:** DataFrame with categorical columns one-hot encoded as integers
- **Side effects:** None

### `save_data(df: pd.DataFrame, filepath: str) -> None`
- **Input:** Processed DataFrame, output path
- **Output:** None
- **Side effects:** Saves CSV to disk

---

## 3. Data Models

### PatientInput (Pydantic)
```python
class PatientInput(BaseModel):
    age:      float  # ge=1,  le=120
    sex:      int    # ge=0,  le=1
    cp:       int    # ge=0,  le=3
    trestbps: float  # ge=50, le=250
    chol:     float  # ge=50, le=700
    fbs:      int    # ge=0,  le=1
    restecg:  int    # ge=0,  le=2
    thalachh: float  # ge=50, le=250
    exang:    int    # ge=0,  le=1
    oldpeak:  float  # ge=0,  le=10
    slope:    int    # ge=0,  le=2
    ca:       int    # ge=0,  le=3
    thal:     int    # ge=1,  le=3
```

### PredictionOutput (Pydantic)
```python
class PredictionOutput(BaseModel):
    prediction:  int    # 0 or 1
    probability: float  # 0.0 to 1.0
    result:      str    # "High Risk" or "Low Risk"
```

---

## 4. Scaler Parameters Format

Saved at `data/processed/scaler_params.json`:
```json
{
  "age":      { "min": 29.0, "max": 77.0 },
  "trestbps": { "min": 94.0, "max": 200.0 },
  "chol":     { "min": 126.0, "max": 564.0 },
  "thalachh": { "min": 71.0,  "max": 202.0 },
  "oldpeak":  { "min": 0.0,   "max": 6.2 }
}
```

---

## 5. Inference Pipeline

```
Raw Input (JSON)
      ↓
Pydantic Validation (range checks)
      ↓
Normalize numeric cols using scaler_params.json
      ↓
One-hot encode categorical cols
      ↓
Align columns to training feature order
      ↓
model.predict() + model.predict_proba()
      ↓
Return PredictionOutput (JSON)
```