"""
preprocessing_dag.py - Airflow DAG for Heart Disease Preprocessing Pipeline
Place this file in your airflow dags/ folder.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

# Add your project root to path so preprocessing.py can be imported
sys.path.insert(0, "/home/asher747/final_project/src/")  # ← update this to your project root

from preprocessing import (
    load_data,
    drop_missing_rows,
    convert_types,
    validate_ranges,
    normalize_numeric,
    one_hot_encode,
    save_data,
)

# ── Config ─────────────────────────────────────────────────────────────────────
import os

PROJECT_ROOT = "/home/asher747/final_project"

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/heart_dataset.csv")
OUTPUT_PATH   = os.path.join(PROJECT_ROOT, "data/processed/features.csv")

# ── Default Arguments ──────────────────────────────────────────────────────────
default_args = {
    'owner':           'airflow',
    'depends_on_past': False,
    'retries':         1,
    'retry_delay':     timedelta(minutes=5),
}

# ── Task Functions ─────────────────────────────────────────────────────────────

def task_load(**context):
    df = load_data(RAW_DATA_PATH)
    context["ti"].xcom_push(key="df", value=df.to_json())


def task_drop_missing(**context):
    import pandas as pd

    data = context["ti"].xcom_pull(task_ids="load_data", key="df")
    df = pd.read_json(data)
    df = drop_missing_rows(df)

    context["ti"].xcom_push(key="df", value=df.to_json())


def task_convert_types(**context):
    import pandas as pd

    data = context["ti"].xcom_pull(task_ids="drop_missing_rows", key="df")
    df = pd.read_json(data)
    df = convert_types(df)

    context["ti"].xcom_push(key="df", value=df.to_json())


def task_validate_ranges(**context):
    import pandas as pd

    data = context["ti"].xcom_pull(task_ids="convert_types", key="df")
    df = pd.read_json(data)
    df = validate_ranges(df)

    context["ti"].xcom_push(key="df", value=df.to_json())


def task_normalize(**context):
    import pandas as pd

    data = context["ti"].xcom_pull(task_ids="validate_ranges", key="df")
    df = pd.read_json(data)
    df = normalize_numeric(df)

    context["ti"].xcom_push(key="df", value=df.to_json())


def task_encode(**context):
    import pandas as pd

    data = context["ti"].xcom_pull(task_ids="normalize_numeric", key="df")
    df = pd.read_json(data)
    df = one_hot_encode(df)

    context["ti"].xcom_push(key="df", value=df.to_json())


def task_save(**context):
    import pandas as pd

    data = context["ti"].xcom_pull(task_ids="one_hot_encode", key="df")
    df = pd.read_json(data)

    save_data(df, OUTPUT_PATH)


# ── DAG Definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id='heart_disease_preprocessing',
    default_args=default_args,
    description='Preprocessing pipeline for Heart Disease Diagnostic System',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['heart-disease', 'preprocessing'],
) as dag:

    load          = PythonOperator(task_id='load_data',           python_callable=task_load           )
    drop_missing  = PythonOperator(task_id='drop_missing_rows',   python_callable=task_drop_missing   )
    convert       = PythonOperator(task_id='convert_types',       python_callable=task_convert_types  )
    validate      = PythonOperator(task_id='validate_ranges',     python_callable=task_validate_ranges)
    normalize     = PythonOperator(task_id='normalize_numeric',   python_callable=task_normalize     )
    encode        = PythonOperator(task_id='one_hot_encode',      python_callable=task_encode     )
    save          = PythonOperator(task_id='save_data',           python_callable=task_save          )

    # ── Pipeline Order ─────────────────────────────────────────────────────────
    load >> drop_missing >> convert >> validate >> normalize >> encode >> save