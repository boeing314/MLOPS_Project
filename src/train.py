"""
train.py - Model Training with MLflow Tracking for Heart Disease Diagnostic System
Run directly : python src/train.py
Run via MLflow Projects:
    mlflow run . -e train
    mlflow run . -e train -P n_estimators=200
    mlflow run . -e train -P n_estimators=200 -P learning_rate=0.05 -P random_state=0
"""

import os
import json
import argparse
import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, f1_score,
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ── Argument Parser ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Heart Disease Diagnostic Models"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/processed/features.csv",
        help="Path to processed features CSV"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100,
        help="Number of estimators for Random Forest and XGBoost"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1,
        help="Learning rate for XGBoost"
    )
    parser.add_argument(
        "--max_iter", type=int, default=500,
        help="Max iterations for Logistic Regression"
    )
    return parser.parse_args()


# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_COL        = "target"
MLFLOW_EXPERIMENT = "heart-disease-diagnosis"


# ── Helper: Get Current Git Commit Hash ───────────────────────────────────────
def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "git-not-available"


# ── Helper: Load and Split Data ───────────────────────────────────────────────
def load_and_split(filepath: str, test_size: float, random_state: int):
    df = pd.read_csv(filepath)
    print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── Helper: Compute Metrics ───────────────────────────────────────────────────
def compute_metrics(y_test, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }


# ── Train a Single Model and Log to MLflow ────────────────────────────────────
def train_and_log(name, model, params, X_train, X_test, y_train, y_test, git_hash):
    print(f"\nTraining: {name}")

    with mlflow.start_run(run_name=name, nested=True) as run:

        # ── Train ──────────────────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ── Metrics ────────────────────────────────────────────────────────────
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  F1 Score : {metrics['f1_score']}")
        print(f"  AUC-ROC  : {metrics['auc_roc']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall   : {metrics['recall']}")

        # ── Log Params ─────────────────────────────────────────────────────────
        mlflow.log_params(params)

        # ── Log Metrics ────────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Log Extra Info (beyond autolog) ────────────────────────────────────
        mlflow.set_tag("git_commit",   git_hash)
        mlflow.set_tag("model_name",   name)
        mlflow.set_tag("test_size",    params["test_size"])
        mlflow.set_tag("random_state", params["random_state"])

        # Log confusion matrix as a JSON artifact
        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_path = f"confusion_matrix_{name}.json"
        with open(cm_path, "w") as f:
            json.dump({"confusion_matrix": cm}, f, indent=2)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Log classification report as a text artifact
        report = classification_report(y_test, y_pred)
        report_path = f"classification_report_{name}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # Log feature importances for tree-based models
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(
                X_train.columns.tolist(),
                model.feature_importances_.tolist()
            ))
            imp_path = f"feature_importances_{name}.json"
            with open(imp_path, "w") as f:
                json.dump(importances, f, indent=2)
            mlflow.log_artifact(imp_path)
            os.remove(imp_path)

        # ── Log Model ──────────────────────────────────────────────────────────
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"heart-disease-{name}"
        )

        run_id = run.info.run_id
        print(f"  MLflow Run ID: {run_id}")
        print(f"  Git Commit   : {git_hash}")

        return run_id, metrics


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    args = parse_args()

    print(f"Parameters:")
    print(f"  data_path    : {args.data_path}")
    print(f"  n_estimators : {args.n_estimators}")
    print(f"  random_state : {args.random_state}")
    print(f"  test_size    : {args.test_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  max_iter     : {args.max_iter}")

    # ── Build models using parsed args ────────────────────────────────────────
    MODELS = {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=args.max_iter,
                random_state=args.random_state
            ),
            "params": {
                "max_iter":     args.max_iter,
                "random_state": args.random_state,
                "solver":       "lbfgs",
                "test_size":    args.test_size,
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=args.n_estimators,
                random_state=args.random_state
            ),
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth":    None,
                "random_state": args.random_state,
                "test_size":    args.test_size,
            }
        },
        "xgboost": {
            "model": XGBClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                random_state=args.random_state,
                eval_metric="logloss",
                verbosity=0
            ),
            "params": {
                "n_estimators":  args.n_estimators,
                "learning_rate": args.learning_rate,
                "random_state":  args.random_state,
                "test_size":     args.test_size,
            }
        }
    }

    # Set MLflow experiment
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Get git commit hash for reproducibility
    git_hash = get_git_commit_hash()
    print(f"\nGit Commit: {git_hash}")

    # Load and split data
    print("\n--- Loading Data ---")
    X_train, X_test, y_train, y_test = load_and_split(
        args.data_path, args.test_size, args.random_state
    )

    # Train all models
    results = {}
    for name, config in MODELS.items():
        run_id, metrics = train_and_log(
            name     = name,
            model    = config["model"],
            params   = config["params"],
            X_train  = X_train,
            X_test   = X_test,
            y_train  = y_train,
            y_test   = y_test,
            git_hash = git_hash
        )
        results[name] = {"run_id": run_id, "metrics": metrics}

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n--- Results Summary ---")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1':<12} {'AUC-ROC':<12}")
    print("-" * 60)
    for name, result in results.items():
        m = result["metrics"]
        print(f"{name:<25} {m['accuracy']:<12} {m['f1_score']:<12} {m['auc_roc']:<12}")

    # Pick best model by F1 score
    best_name = max(results, key=lambda x: results[x]["metrics"]["f1_score"])
    best_run  = results[best_name]["run_id"]
    print(f"\nBest Model : {best_name}")
    print(f"Run ID     : {best_run}")
    print(f"Git Commit : {git_hash}")
    print(f"\nTo reproduce: git checkout {git_hash} + mlflow run ID {best_run}")
    print("\nOpen MLflow UI: mlflow ui --port 5000")