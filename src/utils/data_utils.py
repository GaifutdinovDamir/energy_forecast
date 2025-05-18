import logging
import os

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)

    return df
