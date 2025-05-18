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

from src.utils.logging_utils import setup_logging

setup_logging(task_name="train_model")
logger = logging.getLogger(__name__)


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def train_catboost_model(
    df: pd.DataFrame, params: dict = None, artifacts_dir: str = "artifacts"
) -> None:
    logging.info("Starting CatBoost training...")

    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=["target", "timestamp"])
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    if params is None:
        params = {
            "iterations": 500,
            "learning_rate": 0.1,
            "depth": 6,
            "loss_function": "RMSE",
            "verbose": 100,
            "random_seed": 42,
        }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    metrics = {
        "RMSE_val": mean_squared_error(y_val, y_pred_val),
        "MAE_val": mean_absolute_error(y_val, y_pred_val),
        "R2_val": r2_score(y_val, y_pred_val),
        "RMSE_test": mean_squared_error(y_test, y_pred_test),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
        "R2_test": r2_score(y_test, y_pred_test),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    joblib.dump(model, os.path.join(run_dir, "model.pkl"))

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    config = {
        "params": params,
        "feature_names": list(X.columns),
        "created_at": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "hostname": socket.gethostname(),
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    logging.info(f"✅ Artifacts saved in: {run_dir}")
