import logging

import pandas as pd

from src.utils.data_utils import load_data


def generate_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def generate_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"target_lag_{lag}"] = df["target"].shift(lag)
    return df


def generate_ewm_features(df: pd.DataFrame, span_list: list[int]) -> pd.DataFrame:
    for span in span_list:
        df[f"target_ewm_span_{span}"] = df["target"].ewm(span=span, adjust=False).mean()
    return df


def generate_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for window in windows:
        df[f"target_roll_mean_{window}"] = df["target"].rolling(window=window).mean()
        df[f"target_roll_std_{window}"] = df["target"].rolling(window=window).std()
    return df


def make_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 24, 48],
    ewm_spans: list[int] = [3, 12, 24, 168],
    rolling_windows: list[int] = [3, 24, 168],
) -> pd.DataFrame:

    df = generate_basic_time_features(df)
    df = generate_lag_features(df, lags)
    df = generate_ewm_features(df, ewm_spans)
    df = generate_rolling_features(df, rolling_windows)

    df = df.dropna().reset_index(drop=True)
    return df


df = load_data("data/processed_data/AEP_hourly.csv")
