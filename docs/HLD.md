# High Level Design (HLD) — Heart Disease Diagnostic System

## 1. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early and accurate diagnosis
is critical for effective treatment. This system provides a web-based diagnostic tool that predicts
whether a patient is at risk of heart disease based on clinical parameters such as age, cholesterol,
and blood pressure. It addresses the need for faster and more consistent preliminary diagnosis in
healthcare settings.

---

## 2. System Overview

The Heart Disease Diagnostic System is a full-stack AI application built on MLOps principles.
It consists of a React frontend, a FastAPI backend, an MLflow model registry, a data pipeline
orchestrated by Apache Airflow, and a monitoring stack using Prometheus and Grafana. All
components are containerized using Docker and orchestrated via Docker Compose.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP REST
┌───────────────────────────▼─────────────────────────────────────┐
│                   Frontend (React + Vite)                        │
│                     Port: 3000                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP REST (POST /predict)
┌───────────────────────────▼─────────────────────────────────────┐
│                  Backend (FastAPI + Uvicorn)                      │
│                     Port: 8000                                   │
│   /health  /ready  /predict  /metrics                           │
└──────┬──────────────────────────────────┬────────────────────────┘
       │                                  │
┌──────▼──────┐                  ┌────────▼────────┐
│   MLflow    │                  │   Prometheus     │
│  Port: 5000 │                  │   Port: 9090     │
│  Model      │                  │   Scrapes        │
│  Registry   │                  │   /metrics       │
└─────────────┘                  └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │     Grafana       │
                                 │   Port: 3001      │
                                 │   Dashboards      │
                                 └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline (Airflow)                        │
│                       Port: 8080                                 │
│  load → drop_missing → convert_types → validate_ranges →        │
│  normalize → one_hot_encode → save                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Design Choices and Rationale

### 4.1 Frontend — React + Vite
- React chosen for component-based UI and fast development
- Vite for fast build and hot reload during development
- Communicates with backend **only via REST API** — strict loose coupling
- No business logic in frontend — purely presentation layer

### 4.2 Backend — FastAPI
- FastAPI chosen for high performance, automatic docs generation (Swagger UI)
- Pydantic models enforce strict input validation
- Prometheus client integrated directly for metrics export
- Stateless — no session management, each request is independent

### 4.3 Model — Random Forest Classifier
- Multiple models evaluated: Logistic Regression, Random Forest, XGBoost
- Random Forest selected based on highest F1-score and AUC-ROC
- Binary classification: 0 = Low Risk, 1 = High Risk
- Model stored in MLflow registry for versioning and reproducibility

### 4.4 Data Pipeline — Apache Airflow
- Airflow chosen for pipeline orchestration and visualization
- DAG with 7 tasks — each task is a single responsibility function
- XCom used to pass data between tasks
- Schedule: manual trigger (can be set to daily for production)

### 4.5 Experiment Tracking — MLflow
- All experiments logged with metrics, params, and artifacts
- Git commit hash tagged to every run for full reproducibility
- Model registry used to manage versions and promote to production
- Custom artifacts: confusion matrix, classification report, feature importances

### 4.6 Monitoring — Prometheus + Grafana
- Prometheus scrapes `/metrics` endpoint every 15 seconds
- Four key metrics tracked: prediction count, latency, error count, request count
- Grafana provides near-real-time dashboards
- Alerts configured for error rate > 5%

### 4.7 Containerization — Docker + Docker Compose
- Each component runs in its own container — isolation and reproducibility
- Docker Compose manages service dependencies and startup order
- Shared volume for mlruns between MLflow and backend containers
- Environment parity between development and production

### 4.8 Data Versioning — DVC + Git
- Raw and processed data tracked with DVC
- Code tracked with Git
- Every experiment reproducible via Git commit hash + MLflow run ID

---

## 5. Success Metrics

### ML Metrics
| Metric | Target |
|---|---|
| F1 Score | > 0.85 |
| AUC-ROC | > 0.85 |
| Precision | > 0.80 |
| Recall | > 0.85 |

### Business Metrics
| Metric | Target |
|---|---|
| Inference Latency | < 200ms |
| API Uptime | > 99% |
| Error Rate | < 5% |

---

## 6. Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | FastAPI, Uvicorn |
| ML Models | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Data Pipeline | Apache Airflow |
| Data Versioning | DVC, Git |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker, Docker Compose |
| Language | Python 3.12 |

---

## 7. Data Flow

```
Raw CSV → Airflow Pipeline → Cleaned CSV → train.py → MLflow
                                                          ↓
User Input → React UI → POST /predict → FastAPI → Load Model → Prediction
                                            ↓
                                       Prometheus → Grafana
```

---

## 8. Security Considerations
- Input validation enforced via Pydantic with strict value ranges
- CORS configured to allow only known frontend origins
- No sensitive patient data stored — stateless prediction only
- All data encrypted at rest via filesystem permissions