import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Resolve project root dynamically (utils/ -> project root)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

def load_trained_model(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = Path(model_path)
        
    if not model_path.exists():
        # Fallback check for root-relative paths
        alt_path = Path("models/best_model.pkl")
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path.resolve()}")

    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    # Unwrap dictionary payload if serialized as a dict
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    return loaded

def predict_single_record(model, input_dict):
    # Set default decision threshold to 0.76 (76%)
    threshold = 0.76
    if isinstance(model, dict):
        threshold = model.get("optimal_threshold", 0.76)
        actual_model = model["model"]
    else:
        actual_model = model

    input_df = pd.DataFrame([input_dict])
    
    if hasattr(actual_model, "predict_proba"):
        prob = float(actual_model.predict_proba(input_df)[0][1])
    else:
        prob = float(actual_model.predict(input_df)[0])
        
    # Classify as churned (1) only if predicted probability >= 76%
    prediction = int(prob >= threshold)
    return prediction, prob

def get_top_feature_drivers(model, feature_cols, top_n=4):
    # Extract estimator if wrapped inside a dictionary
    actual_model = model["model"] if isinstance(model, dict) and "model" in model else model

    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return df_imp.head(top_n).reset_index(drop=True)