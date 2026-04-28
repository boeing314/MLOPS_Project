# Heart Disease Diagnostic System (MLOps Project)

An end-to-end Machine Learning Operations (MLOps) project that predicts the likelihood of heart disease based on patient clinical parameters. The system includes data preprocessing, model training, experiment tracking, model registry, API serving, frontend UI, monitoring, and workflow orchestration.

---

# Project Overview

This project uses machine learning models to predict whether a patient is at **High Risk** or **Low Risk** of heart disease.

The solution includes:

- Data preprocessing pipeline
- Multiple model training and comparison
- MLflow experiment tracking + model registry
- FastAPI backend for real-time inference
- React frontend UI
- Prometheus + Grafana monitoring
- Dockerized deployment
- Airflow workflow orchestration
- DVC data version control

---

# Tech Stack

## Machine Learning

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy

## MLOps / Deployment

- MLflow
- FastAPI
- React + Vite
- Docker + Docker Compose
- Prometheus
- Grafana
- Apache Airflow
- DVC

---

# Project Structure

```text
final_project/
│── src/                  # Backend + ML code
│   ├── preprocessing.py
│   ├── train.py
│   └── app.py
│
│── frontend/             # React frontend
│── airflow/              # Airflow DAGs
│── data/
│   ├── raw/
│   └── processed/
│
│── tests/                # Pytest unit tests
│── docs/                 # Documentation
│── docker-compose.yml
│── Dockerfile.backend
│── Dockerfile.frontend
│── MLproject
│── prometheus.yml
│── requirements.txt
│── README.md