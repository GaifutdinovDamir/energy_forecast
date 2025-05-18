import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_time_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    freq: str = "H",
) -> None:
    errors = []

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            errors.append(f"Cannot convert '{timestamp_col}' to datetime: {e}")

    if not df[timestamp_col].is_monotonic_increasing:
        errors.append(f"'{timestamp_col}' is not sorted in ascending order.")

    # expected_diff = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    # actual_diff = df[timestamp_col].diff().dropna()
    # if not all(actual_diff == expected_diff):
    #     errors.append("Time gaps or inconsistent frequency detected in timestamp column.")

    missing = df[target_col].isna().sum()
    if missing > 0:
        errors.append(f"{missing} missing values in '{target_col}' column.")

    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError("Time series validation failed.")
    else:
        logger.info("Time series validation passed.")
