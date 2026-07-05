import numpy as np
import pandas as pd

from src.logger import logging


def clean_baf_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Clean and feature-engineer a BAF dataframe.

    Shared by the training pipeline (data transformation stage) and the
    inference API so both run the exact same logic before the preprocessor.

    df: raw dataframe (with or without the target column)
    schema: parsed config/schema.yaml
    """
    df = df.copy()

    sentinel_cols = schema["sentinel_columns"]
    log1p_cols = schema["log1p_columns"]
    target = schema["target"]
    binary_cols = schema["binary_columns"]
    categorical_cols = schema["categorical_columns"]
    numeric_cols = schema["numeric_columns"]

    logging.info("Initializing cleaning and feature engineering.")

    for col in categorical_cols:
        if col in df.columns:
            if col == "device_os":
                df[col] = df[col].astype(str).str.lower().str.strip()
            else:
                df[col] = df[col].astype(str).str.upper().str.strip()

    if target in df.columns:
        df[target] = df[target].astype(int)

    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in sentinel_cols:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col] == -1).astype("Int64")

    for col in sentinel_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == -1, col] = np.nan

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in log1p_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    logging.info("Done cleaning and feature engineering.")
    return df
