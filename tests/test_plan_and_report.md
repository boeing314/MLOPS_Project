# Test Plan & Test Report — Heart Disease Diagnostic System

---

## Part 1: Test Plan

### 1.1 Objectives
- Verify that the preprocessing pipeline correctly cleans and transforms data
- Verify that the FastAPI backend returns correct predictions
- Verify that input validation rejects invalid data
- Verify that the system meets defined acceptance criteria

### 1.2 Scope
- Unit tests for all preprocessing functions
- Integration tests for API endpoints
- Acceptance tests against defined business metrics

### 1.3 Tools
- **pytest** — unit and integration testing
- **requests** — API endpoint testing
- **pandas** — data validation

### 1.4 Acceptance Criteria

| Criteria | Target | Test Method |
|---|---|---|
| F1 Score | > 0.85 | Evaluate on test split |
| AUC-ROC | > 0.85 | Evaluate on test split |
| Inference Latency | < 200ms | Time /predict endpoint |
| API uptime | > 99% | /health endpoint |
| Input validation | Reject invalid values | Send out-of-range inputs |
| Missing value handling | No NaN in output | Check processed CSV |
| Normalized values | All in [0, 1] | Check processed CSV |

---

### 1.5 Test Cases

#### Unit Tests — Preprocessing

| ID | Function | Input | Expected Output | Type |
|---|---|---|---|---|
| TC01 | drop_missing_rows | DataFrame with NaN rows | NaN rows removed | Unit |
| TC02 | drop_missing_rows | Clean DataFrame | Unchanged DataFrame | Unit |
| TC03 | drop_missing_rows | DataFrame with 2 NaN rows | 2 rows fewer | Unit |
| TC04 | convert_types | DataFrame with string in numeric col | Row with string dropped | Unit |
| TC05 | convert_types | Clean DataFrame | All columns numeric | Unit |
| TC06 | validate_ranges | Age = 999 | Row removed | Unit |
| TC07 | validate_ranges | Chol = 800 | Row removed | Unit |
| TC08 | validate_ranges | Valid DataFrame | Unchanged | Unit |
| TC09 | normalize_numeric | Numeric columns | All values in [0.0, 1.0] | Unit |
| TC10 | normalize_numeric | Any DataFrame | scaler_params.json created | Unit |
| TC11 | normalize_numeric | Binary columns | Binary columns unchanged | Unit |
| TC12 | one_hot_encode | cp column with 4 values | 4 new cp_ columns | Unit |
| TC13 | one_hot_encode | Categorical columns | Original columns removed | Unit |
| TC14 | one_hot_encode | Encoded values | Values are 0 or 1 integers | Unit |
| TC15 | one_hot_encode | Binary columns | Binary columns preserved | Unit |
| TC16 | one_hot_encode | target column | Target column preserved | Unit |

#### Integration Tests — API

| ID | Endpoint | Input | Expected Output | Type |
|---|---|---|---|---|
| TC17 | GET /health | None | {"status": "ok"} 200 | Integration |
| TC18 | GET /ready | None | {"status": "ready"} 200 | Integration |
| TC19 | POST /predict | Valid patient data | prediction, probability, result | Integration |
| TC20 | POST /predict | High risk patient | prediction=1, result="High Risk" | Integration |
| TC21 | POST /predict | Low risk patient | prediction=0, result="Low Risk" | Integration |
| TC22 | POST /predict | age=999 (invalid) | 422 Unprocessable Entity | Integration |
| TC23 | POST /predict | Missing field | 422 Unprocessable Entity | Integration |
| TC24 | POST /predict | Empty body | 422 Unprocessable Entity | Integration |
| TC25 | GET /metrics | None | Prometheus text format 200 | Integration |

#### Acceptance Tests

| ID | Criteria | Method | Target |
|---|---|---|---|
| AC01 | F1 Score | Evaluate model on test set | > 0.85 |
| AC02 | AUC-ROC | Evaluate model on test set | > 0.85 |
| AC03 | Inference Latency | Time 100 /predict calls | < 200ms average |
| AC04 | Input Validation | Send 10 invalid requests | All return 422 |
| AC05 | Pipeline completeness | Run Airflow DAG | All 7 tasks green |

---

## Part 2: Test Report

### 2.1 Unit Test Results

Run command:
```bash
pytest tests/test_preprocessing.py -v
```

| ID | Test Name | Status |
|---|---|---|
| TC01 | test_drop_missing_rows_removes_nan_rows | ✅ PASSED |
| TC02 | test_drop_missing_rows_clean_df_unchanged | ✅ PASSED |
| TC03 | test_drop_missing_rows_correct_count | ✅ PASSED |
| TC04 | test_convert_types_removes_invalid_rows | ✅ PASSED |
| TC05 | test_convert_types_all_numeric | ✅ PASSED |
| TC06 | test_validate_ranges_removes_out_of_range | ✅ PASSED |
| TC07 | test_validate_ranges_chol_bounds | ✅ PASSED |
| TC08 | test_validate_ranges_valid_df_unchanged | ✅ PASSED |
| TC09 | test_normalize_numeric_range | ✅ PASSED |
| TC10 | test_normalize_numeric_saves_scaler_params | ✅ PASSED |
| TC11 | test_normalize_binary_cols_unchanged | ✅ PASSED |
| TC12 | test_one_hot_encode_creates_new_columns | ✅ PASSED |
| TC13 | test_one_hot_encode_removes_original_cols | ✅ PASSED |
| TC14 | test_one_hot_encode_values_are_integers | ✅ PASSED |
| TC15 | test_one_hot_encode_binary_cols_preserved | ✅ PASSED |
| TC16 | test_one_hot_encode_target_preserved | ✅ PASSED |

**Unit Test Summary:**
- Total: 16
- Passed: 16
- Failed: 0

---

### 2.2 Integration Test Results

| ID | Test | Status | Notes |
|---|---|---|---|
| TC17 | GET /health | ✅ PASSED | Returns 200 {"status":"ok"} |
| TC18 | GET /ready | ✅ PASSED | Returns 200 {"status":"ready"} |
| TC19 | POST /predict valid | ✅ PASSED | Returns prediction JSON |
| TC20 | POST /predict high risk | ✅ PASSED | Returns prediction=1 |
| TC21 | POST /predict low risk | ✅ PASSED | Returns prediction=0 |
| TC22 | POST /predict age=999 | ✅ PASSED | Returns 422 |
| TC23 | POST /predict missing field | ✅ PASSED | Returns 422 |
| TC24 | POST /predict empty body | ✅ PASSED | Returns 422 |
| TC25 | GET /metrics | ✅ PASSED | Returns Prometheus format |

**Integration Test Summary:**
- Total: 9
- Passed: 9
- Failed: 0

---

### 2.3 Acceptance Test Results

| ID | Criteria | Target | Actual | Status |
|---|---|---|---|---|
| AC01 | F1 Score | > 0.85 | 0.89 | ✅ MET |
| AC02 | AUC-ROC | > 0.85 | 0.92 | ✅ MET |
| AC03 | Inference Latency | < 200ms | ~45ms | ✅ MET |
| AC04 | Input Validation | All 422 | All 422 | ✅ MET |
| AC05 | Pipeline completeness | 7 tasks green | 7 tasks green | ✅ MET |

---

### 2.4 Overall Summary

| Category | Total | Passed | Failed |
|---|---|---|---|
| Unit Tests | 16 | 16 | 0 |
| Integration Tests | 9 | 9 | 0 |
| Acceptance Tests | 5 | 5 | 0 |
| **Total** | **30** | **30** | **0** |

**All acceptance criteria met. System is ready for demonstration.**