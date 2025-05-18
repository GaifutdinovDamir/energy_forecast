import logging
import os

import pandas as pd

from src.utils.data_utils import load_data
from src.utils.data_validation import validate_time_series
from src.utils.logging_utils import setup_logging

PROCESSED_DIR = "data/processed_data"

setup_logging(task_name="validate_data")


def validate_all_csv_files(data_dir: str = PROCESSED_DIR):

    logging.info("🚀 Starting dataset validation for all CSV files...")

    if not os.path.exists(data_dir):
        logging.error(f"❌ Provided directory does not exist: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "est" not in f]
    if not files:
        logging.warning("⚠️ No CSV files found in the processed data directory.")
        return

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        logging.info(f"📂 Validating file: {filename}")
        try:
            df = load_data(f"{PROCESSED_DIR}/{filename}")
            validate_time_series(df)
            logging.info(f"✅ File passed validation: {filename}")
        except Exception as e:
            logging.error(f"❌ Validation failed for {filename}: {e}")


if __name__ == "__main__":
    validate_all_csv_files()
