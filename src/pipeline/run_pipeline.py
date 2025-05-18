import json
import logging
import os

from src.features.make_features import make_features
from src.model.train_model import get_git_hash, train_catboost_model
from src.utils.data_utils import load_data

logging.basicConfig(level=logging.INFO)


def main():
    logging.info("🚀 Starting pipeline")

    with open("./src/pipeline/conf.json", "r") as f:
        config = json.load(f)

    data_path = config.get("data_path", "none")
    artifacts_dir = config.get("artifacts_dir", "artifacts")
    train_params = config.get("train_params", None)

    logging.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    logging.info(f"Loaded {len(df)} rows")

    logging.info("Generating features")
    df_feat = make_features(df)
    logging.info(f"Features dataframe shape: {df_feat.shape}")

    logging.info("Training model")
    train_catboost_model(df_feat, params=train_params, artifacts_dir=artifacts_dir)
    logging.info("✅ Pipeline finished")


if __name__ == "__main__":
    main()
