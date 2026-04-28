"""
test_preprocessing.py - Unit Tests for Heart Disease Preprocessing Pipeline
Run: pytest tests/test_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import (
    load_data,
    drop_missing_rows,
    convert_types,
    validate_ranges,
    normalize_numeric,
    one_hot_encode,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """A clean valid dataframe with no issues."""
    return pd.DataFrame({
        'age':      [63, 37, 41, 56, 57],
        'sex':      [1,  1,  0,  1,  0],
        'cp':       [3,  2,  1,  1,  0],
        'trestbps': [145,130,130,120,120],
        'chol':     [233,250,204,236,354],
        'fbs':      [1,  0,  0,  0,  0],
        'restecg':  [0,  1,  0,  1,  1],
        'thalachh': [150,187,172,178,163],
        'exang':    [0,  0,  0,  0,  1],
        'oldpeak':  [2.3,3.5,1.4,0.8,0.6],
        'slope':    [0,  0,  2,  2,  2],
        'ca':       [0,  0,  0,  0,  0],
        'thal':     [1,  2,  2,  2,  2],
        'target':   [1,  1,  1,  1,  1],
    })


@pytest.fixture
def df_with_missing():
    """Dataframe with NaN values."""
    return pd.DataFrame({
        'age':      [63, None, 41],
        'sex':      [1,  1,    0],
        'cp':       [3,  2,    1],
        'trestbps': [145,130,  130],
        'chol':     [233,None, 204],
        'fbs':      [1,  0,    0],
        'restecg':  [0,  1,    0],
        'thalachh': [150,187,  172],
        'exang':    [0,  0,    0],
        'oldpeak':  [2.3,3.5,  1.4],
        'slope':    [0,  0,    2],
        'ca':       [0,  0,    0],
        'thal':     [1,  2,    2],
        'target':   [1,  1,    1],
    })


@pytest.fixture
def df_with_invalid_types():
    """Dataframe with invalid string values in numeric columns."""
    return pd.DataFrame({
        'age':      [63, 'abc', 41],
        'sex':      [1,  1,    0],
        'cp':       [3,  2,    1],
        'trestbps': [145,130,  130],
        'chol':     [233,250,  204],
        'fbs':      [1,  0,    0],
        'restecg':  [0,  1,    0],
        'thalachh': [150,187,  172],
        'exang':    [0,  0,    0],
        'oldpeak':  [2.3,3.5,  1.4],
        'slope':    [0,  0,    2],
        'ca':       [0,  0,    0],
        'thal':     [1,  2,    2],
        'target':   [1,  1,    1],
    })


@pytest.fixture
def df_with_out_of_range():
    """Dataframe with out of range values."""
    return pd.DataFrame({
        'age':      [63,  999, 41],   # 999 is invalid
        'sex':      [1,   1,   0],
        'cp':       [3,   2,   1],
        'trestbps': [145, 130, 130],
        'chol':     [233, 250, 800],  # 800 is invalid
        'fbs':      [1,   0,   0],
        'restecg':  [0,   1,   0],
        'thalachh': [150, 187, 172],
        'exang':    [0,   0,   0],
        'oldpeak':  [2.3, 3.5, 1.4],
        'slope':    [0,   0,   2],
        'ca':       [0,   0,   0],
        'thal':     [1,   2,   2],
        'target':   [1,   1,   1],
    })


# ── Tests: drop_missing_rows ───────────────────────────────────────────────────

def test_drop_missing_rows_removes_nan_rows(df_with_missing):
    result = drop_missing_rows(df_with_missing)
    assert result.isnull().sum().sum() == 0


def test_drop_missing_rows_correct_count(df_with_missing):
    result = drop_missing_rows(df_with_missing)
    assert len(result) == 2   


def test_drop_missing_rows_clean_df_unchanged(clean_df):
    result = drop_missing_rows(clean_df)
    assert len(result) == len(clean_df)


# ── Tests: convert_types ───────────────────────────────────────────────────────

def test_convert_types_removes_invalid_rows(df_with_invalid_types):
    result = convert_types(df_with_invalid_types)
    assert len(result) == 2   # row with 'abc' should be dropped


def test_convert_types_all_numeric(clean_df):
    result = convert_types(clean_df)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])


# ── Tests: validate_ranges ─────────────────────────────────────────────────────

def test_validate_ranges_removes_out_of_range(df_with_out_of_range):
    result = validate_ranges(df_with_out_of_range)
    assert len(result) == 1   # only first row is fully valid


def test_validate_ranges_valid_df_unchanged(clean_df):
    result = validate_ranges(clean_df)
    assert len(result) == len(clean_df)


def test_validate_ranges_age_bounds(clean_df):
    result = validate_ranges(clean_df)
    assert result['age'].between(1, 120).all()


def test_validate_ranges_chol_bounds(clean_df):
    result = validate_ranges(clean_df)
    assert result['chol'].between(50, 700).all()


# ── Tests: normalize_numeric ───────────────────────────────────────────────────

def test_normalize_numeric_range(clean_df):
    result = normalize_numeric(clean_df)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak']
    for col in numeric_cols:
        assert result[col].min() >= 0.0
        assert result[col].max() <= 1.0



def test_normalize_numeric_saves_scaler_params(clean_df, tmp_path):
    import json
    from unittest.mock import patch
    scaler_path = str(tmp_path / "scaler_params.json")
    with patch('preprocessing.SCALER_PARAMS_PATH', scaler_path):
        normalize_numeric(clean_df)
    assert os.path.exists(scaler_path)
    with open(scaler_path) as f:
        params = json.load(f)
    assert 'age' in params
    assert 'min' in params['age']
    assert 'max' in params['age']


def test_normalize_binary_cols_unchanged(clean_df):
    result = normalize_numeric(clean_df)
    # Binary columns should still be 0 or 1
    for col in ['sex', 'fbs', 'exang']:
        assert set(result[col].unique()).issubset({0, 1})


# ── Tests: one_hot_encode ──────────────────────────────────────────────────────

def test_one_hot_encode_creates_new_columns(clean_df):
    result = one_hot_encode(clean_df)
    # cp has 4 values (0,1,2,3) so should create 4 columns
    cp_cols = [c for c in result.columns if c.startswith('cp_')]
    assert len(cp_cols) == 4


def test_one_hot_encode_removes_original_cols(clean_df):
    result = one_hot_encode(clean_df)
    for col in ['cp', 'restecg', 'slope', 'ca', 'thal']:
        assert col not in result.columns


def test_one_hot_encode_values_are_integers(clean_df):
    result = one_hot_encode(clean_df)
    ohe_cols = [c for c in result.columns if '_' in c and
                any(c.startswith(p) for p in ['cp_', 'restecg_', 'slope_', 'ca_', 'thal_'])]
    for col in ohe_cols:
        assert set(result[col].unique()).issubset({0, 1})


def test_one_hot_encode_binary_cols_preserved(clean_df):
    result = one_hot_encode(clean_df)
    for col in ['sex', 'fbs', 'exang']:
        assert col in result.columns


def test_one_hot_encode_target_preserved(clean_df):
    result = one_hot_encode(clean_df)
    assert 'target' in result.columns