import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

VALUE_RANGES = {
    'age': (1, 120),
    'trestbps': (80, 200),
    'chol': (100, 600),
    'thalach': (60, 220),
}

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def validate_schema(df: pd.DataFrame) -> bool:
    """Check all expected columns are present."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    logger.info("Schema validation passed")
    return True

def validate_ranges(df: pd.DataFrame) -> dict:
    """Flag rows where values fall outside expected ranges."""
    issues = {}
    for col, (low, high) in VALUE_RANGES.items():
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            issues[col] = len(out_of_range)
            logger.warning(f"{col}: {len(out_of_range)} out-of-range values")
    return issues

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found")
    return missing

if __name__ == "__main__":
    df = load_data("data/raw/heart.csv")
    validate_schema(df)
    check_missing_values(df)
    validate_ranges(df)