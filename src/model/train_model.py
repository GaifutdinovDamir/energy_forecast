import json
import logging
import os
import socket
import subprocess
from datetime import datetime

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import setup_logging
from src.utils.model_utils import evaluate_model, save_model

setup_logging(task_name="train_model")
logger = logging.getLogger(__name__)


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict = None,
) -> None:
    logging.info("Starting CatBoost training...")

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
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    return model
