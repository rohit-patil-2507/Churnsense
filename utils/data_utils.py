import os
import pandas as pd
from typing import List

PROCESSED_DATA_PATH = os.path.join("data", "processed_data.csv")

def load_processed_data(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Loads the final processed dataset matching the model's exact feature set.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at: {file_path}")
    return pd.read_csv(file_path)

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns feature names excluding the target variable.
    """
    return [col for col in df.columns if col != "churned"]