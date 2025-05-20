import json
import logging
import os

from src.features.make_features import make_features
from src.model.hp_tuning import run_optuna_catboost
from src.model.train_model import train_catboost_model
from src.utils.data_utils import load_data
from src.utils.logging_utils import setup_logging
from src.utils.model_utils import save_artifacts, split_data

setup_logging(task_name="pipeline")
logger = logging.getLogger(__name__)


def main():
    logging.info("🚀 Starting pipeline")

    with open("./src/pipeline/conf.json", "r") as f:
        config = json.load(f)

    data_path = config.get("data_path", "none")
    artifacts_dir = config.get("artifacts_dir", "artifacts")
    hp_trials = config.get("hp_trials", 50)
    n_jobs = config.get("n_jobs", 8)

    logging.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    logging.info(f"Loaded {len(df)} rows")

    logging.info("Generating features")
    df_feat = make_features(df)
    logging.info(f"Features dataframe shape: {df_feat.shape}")

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df_feat)

    logging.info("Starting hyperparameter tuning")
    best_params = run_optuna_catboost(
        X_train,
        y_train,
        X_valid,
        y_valid,
        dataset_name=data_path.split("/")[-1],
        n_trials=hp_trials,
        n_jobs=n_jobs,
    )
    logging.info("Training model")
    model = train_catboost_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=best_params,
    )
    logging.info("Saving artifacts")
    save_artifacts(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        best_params,
        artifacts_dir=artifacts_dir,
        data_path=data_path,
    )
    logging.info("✅ Pipeline finished")


if __name__ == "__main__":
    main()
