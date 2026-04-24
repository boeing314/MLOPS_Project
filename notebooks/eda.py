"""
eda.py - Exploratory Data Analysis for Heart Disease Dataset
Run: python eda.py
Outputs: saves all plots to outputs/eda_plots/ and baseline_stats.json
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw/heart_dataset.csv"
OUTPUT_PLOTS_DIR    = "outputs/eda_plots"
BASELINE_STATS_PATH = "data/processed/baseline_stats.json"

CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal', 'sex', 'fbs', 'exang']
NUMERIC_COLS     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
TARGET_COL       = 'target'

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    return df


# ── 2. Basic Summary ───────────────────────────────────────────────────────────
def basic_summary(df: pd.DataFrame) -> None:
    logger.info("=== Basic Summary ===")
    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    print("\n--- Missing Values per Column ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found")

    print("\n--- Duplicate Rows ---")
    print(f"{df.duplicated().sum()} duplicate rows found")


# ── 3. Class Distribution ──────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    logger.info("Plotting class distribution...")
    counts = df[TARGET_COL].value_counts()
    labels = ['No Disease (0)', 'Disease (1)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Class Distribution", fontsize=14, fontweight='bold')

    # Bar chart
    axes[0].bar(labels, counts.values, color=['#4C9BE8', '#E8684C'], edgecolor='black')
    axes[0].set_title("Count")
    axes[0].set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
                colors=['#4C9BE8', '#E8684C'], startangle=90)
    axes[1].set_title("Proportion")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/class_distribution.png", dpi=150)
    plt.close()
    logger.info(f"Class distribution: {counts.to_dict()}")


# ── 4. Numeric Feature Distributions ──────────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame) -> None:
    logger.info("Plotting numeric feature distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        axes[i].hist(df[col].dropna(), bins=25, color='#4C9BE8',
                     edgecolor='black', alpha=0.8)
        axes[i].axvline(df[col].mean(), color='red', linestyle='--',
                        linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
        axes[i].axvline(df[col].median(), color='green', linestyle='--',
                        linewidth=1.5, label=f"Median: {df[col].median():.1f}")
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)

    # Hide unused subplot if any
    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/numeric_distributions.png", dpi=150)
    plt.close()


# ── 5. Boxplots by Target ──────────────────────────────────────────────────────
def plot_boxplots_by_target(df: pd.DataFrame) -> None:
    logger.info("Plotting boxplots by target...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Numeric Features by Target Class", fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        disease     = df[df[TARGET_COL] == 1][col].dropna()
        no_disease  = df[df[TARGET_COL] == 0][col].dropna()
        axes[i].boxplot([no_disease, disease],
                        labels=['No Disease', 'Disease'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#4C9BE8', alpha=0.7))
        axes[i].set_title(col)
        axes[i].set_ylabel(col)

    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/boxplots_by_target.png", dpi=150)
    plt.close()


# ── 6. Categorical Feature Counts ─────────────────────────────────────────────
def plot_categorical_distributions(df: pd.DataFrame) -> None:
    logger.info("Plotting categorical feature distributions...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Categorical Feature Distributions by Target", fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(CATEGORICAL_COLS):
        ct = pd.crosstab(df[col], df[TARGET_COL])
        ct.plot(kind='bar', ax=axes[i], color=['#4C9BE8', '#E8684C'],
                edgecolor='black', alpha=0.8)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].legend(['No Disease', 'Disease'], fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/categorical_distributions.png", dpi=150)
    plt.close()


# ── 7. Correlation Heatmap ─────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    logger.info("Plotting correlation heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap='coolwarm', center=0, linewidths=0.5,
        ax=ax, annot_kws={"size": 9}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    # Log top correlations with target
    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    logger.info(f"\nTop correlations with target:\n{target_corr}")


# ── 8. Outlier Detection ───────────────────────────────────────────────────────
def plot_outliers(df: pd.DataFrame) -> None:
    logger.info("Plotting outlier detection (IQR method)...")
    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(18, 6))
    fig.suptitle("Outlier Detection — Boxplots", fontsize=14, fontweight='bold')

    for i, col in enumerate(NUMERIC_COLS):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='#A8D8EA', alpha=0.8))
        axes[i].set_title(col, fontsize=10)

        # Count outliers via IQR
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        logger.info(f"{col}: {len(outliers)} outliers detected (IQR method)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/outliers.png", dpi=150)
    plt.close()


# ── 9. Pairplot for Key Features ───────────────────────────────────────────────
def plot_pairplot(df: pd.DataFrame) -> None:
    logger.info("Plotting pairplot for key numeric features...")
    key_cols = ['age', 'chol', 'thalach', 'oldpeak', TARGET_COL]
    pair_df  = df[key_cols].dropna()

    pairplot = sns.pairplot(
        pair_df,
        hue=TARGET_COL,
        palette={0: '#4C9BE8', 1: '#E8684C'},
        diag_kind='kde',
        plot_kws={'alpha': 0.5}
    )
    pairplot.fig.suptitle("Pairplot — Key Features", y=1.02,
                           fontsize=14, fontweight='bold')
    pairplot.savefig(f"{OUTPUT_PLOTS_DIR}/pairplot.png", dpi=150)
    plt.close()


# ── 10. Baseline Statistics (for Drift Detection) ─────────────────────────────
def calculate_baseline_stats(df: pd.DataFrame) -> None:
    logger.info("Calculating baseline statistics for drift detection...")
    baseline = {}

    for col in df.select_dtypes(include=np.number).columns:
        baseline[col] = {
            "mean":  round(df[col].mean(), 4),
            "std":   round(df[col].std(), 4),
            "min":   round(df[col].min(), 4),
            "max":   round(df[col].max(), 4),
            "p25":   round(df[col].quantile(0.25), 4),
            "p50":   round(df[col].quantile(0.50), 4),
            "p75":   round(df[col].quantile(0.75), 4),
        }

    with open(BASELINE_STATS_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    logger.info(f"Baseline stats saved to {BASELINE_STATS_PATH}")
    print(f"\n--- Baseline Stats Preview ---")
    for col, stats in list(baseline.items())[:3]:
        print(f"{col}: {stats}")
    print("...")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logger.info("Starting EDA...")

    df = load_data(RAW_DATA_PATH)
    basic_summary(df)
    plot_class_distribution(df)
    plot_numeric_distributions(df)
    plot_boxplots_by_target(df)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(df)
    plot_outliers(df)
    plot_pairplot(df)
    calculate_baseline_stats(df)

    logger.info(f"EDA complete. All plots saved to: {OUTPUT_PLOTS_DIR}/")
    logger.info(f"Baseline stats saved to: {BASELINE_STATS_PATH}")


if __name__ == "__main__":
    main()