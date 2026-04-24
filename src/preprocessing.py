"""
preprocessing.py - Clean and prepare heart disease dataset
Run: python preprocessing.py
Output: data/processed/features.csv, data/processed/scaler_params.json
"""

import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/heart_dataset.csv"
OUTPUT_PATH = "data/processed/features.csv"
SCALER_PARAMS_PATH = "data/processed/scaler_params.json"

NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak']
BINARY_COLS = ['sex', 'fbs', 'exang']
CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal']
TARGET_COL = 'target'


# ── Step 1: Load Data ──────────────────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv(filepath, na_values='?')
    df.replace('?', pd.NA, inplace=True)

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values found: {df.isnull().sum().sum()}")

    return df


# ── Step 2: Remove Rows With Missing Values ───────────────────────────────────
def drop_missing_rows(df):
    before = len(df)
    df = df.dropna()
    after = len(df)

    print(f"Removed {before - after} row(s) with missing values")
    return df


# ── Step 3: Convert Types ──────────────────────────────────────────────────────
def convert_types(df):
    df = df.copy()

    for col in NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df = df.dropna()
    after = len(df)

    print(f"Removed {before - after} row(s) with invalid datatypes)")
    return df


# ── Step 4: Keep Only Valid Range Rows ────────────────────────────────────────
def validate_ranges(df):
    before = len(df)

    valid = (
        df['age'].between(1, 120) &
        df['sex'].isin([0, 1]) &
        df['cp'].isin([0, 1, 2, 3]) &
        df['trestbps'].between(50, 250) &
        df['chol'].between(50, 700) &
        df['fbs'].isin([0, 1]) &
        df['restecg'].isin([0, 1, 2]) &
        df['thalachh'].between(50, 250) &
        df['exang'].isin([0, 1]) &
        df['oldpeak'].between(0, 10) &
        df['slope'].isin([0, 1, 2]) &
        df['ca'].isin([0, 1, 2, 3]) &
        df['thal'].isin([1, 2, 3]) &
        df['target'].isin([0, 1])
    )

    df = df[valid].copy()

    after = len(df)
    print(f"Removed {before - after} row(s) outside allowed ranges")

    return df


# ── Step 5: Normalize Numeric Columns ─────────────────────────────────────────
def normalize_numeric(df):
    df = df.copy()

    scaler = MinMaxScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    scaler_params = {
        col: {
            "min": float(scaler.data_min_[i]),
            "max": float(scaler.data_max_[i])
        }
        for i, col in enumerate(NUMERIC_COLS)
    }

    os.makedirs(os.path.dirname(SCALER_PARAMS_PATH), exist_ok=True)

    with open(SCALER_PARAMS_PATH, "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"Normalized columns: {NUMERIC_COLS}")
    return df


# ── Step 6: One-Hot Encode ────────────────────────────────────────────────────
def one_hot_encode(df):
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=False,
        dtype=int
    )

    print(f"One-hot encoded columns: {CATEGORICAL_COLS}")
    return df


# ── Step 7: Save ───────────────────────────────────────────────────────────────
def save_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Step 1: Load ---")
    df = load_data(RAW_DATA_PATH)

    print("\n--- Step 2: Remove Missing Rows ---")
    df = drop_missing_rows(df)

    print("\n--- Step 3: Convert Types ---")
    df = convert_types(df)

    print("\n--- Step 4: Validate Allowed Ranges ---")
    df = validate_ranges(df)

    print("\n--- Step 5: Normalize Numeric Columns ---")
    df = normalize_numeric(df)

    print("\n--- Step 6: One-Hot Encode ---")
    df = one_hot_encode(df)

    print("\n--- Step 7: Save ---")
    save_data(df, OUTPUT_PATH)

    print("\nDone! Preview:")
    print(df.head())