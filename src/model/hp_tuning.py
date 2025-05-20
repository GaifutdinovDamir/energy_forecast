import json
import logging
import os
import socket
from datetime import datetime

import joblib
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logging_utils import setup_logging
from src.utils.model_utils import calculate_metrics

setup_logging(task_name="hp_tuning")
logger = logging.getLogger(__name__)


def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_seed": 42,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "early_stopping_rounds": 100,
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds)
    return rmse


def run_optuna_catboost(
    X_train, y_train, X_valid, y_valid, dataset_name="none", n_trials=50, n_jobs=8
):
    import logging

    logging.info("🔍 Starting Optuna optimization for CatBoost")

    storage = "sqlite:///optuna_catboost.db"
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{dataset_name}_catboost_optuna",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_params = study.best_params
    best_params.update(
        {
            "iterations": 1000,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": 100,
        }
    )

    logging.info(f"✅ Best params: {best_params}")
    logging.info(f"💡 Best trial: {study.best_trial.value}")
    logging.info(f"Fisished optimization with {len(study.trials)} trials")
    return best_params
