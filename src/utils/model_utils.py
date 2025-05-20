import json
import logging
import os
import socket
import subprocess
from datetime import datetime

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "R2": r2}


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)
    return metrics


def save_model(model, model_path):
    joblib.dump(model, model_path)


def save_artifacts(
    model: CatBoostRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    artifacts_dir: str = "artifacts",
    data_path: str = "none",
) -> None:
    logging.info("Saving artifacts...")
    train_metrics = evaluate_model(model, X_train, y_train)
    valid_metrics = evaluate_model(model, X_valid, y_valid)
    test_metrics = evaluate_model(model, X_test, y_test)

    metrics = {
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    save_model(model, os.path.join(run_dir, "model.pkl"))

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    config = {
        "data_path": data_path,
        "params": params,
        "feature_names": list(X_train.columns),
        "created_at": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "hostname": socket.gethostname(),
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    logging.info(f"✅ Artifacts saved in: {run_dir}")
