import logging
import os
from pathlib import Path

import pandas as pd

from src.utils.data_utils import load_data
from src.utils.logging_utils import setup_logging

RAW_DIR = Path("./data/raw_data")
PROCESSED_DIR = Path("./data/processed_data")

setup_logging(task_name="preprocess_data")
logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")

    df = df.drop_duplicates(subset="Datetime")

    df = df.rename(columns={"Datetime": "timestamp", target_name: "target"})

    df = df.dropna()

    return df


def main():
    print(os.listdir(RAW_DIR))
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in 'data/raw/'.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        logger.info(f"Processing file: {csv_file.name}")
        try:
            df = load_data(csv_file)
            df_clean = preprocess_data(
                df,
                target_name=csv_file.name.replace(".csv", "").replace("hourly", "MW"),
            )
            output_path = PROCESSED_DIR / csv_file.name
            df_clean.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned file to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to process {csv_file.name}: {e}")


if __name__ == "__main__":
    main()
