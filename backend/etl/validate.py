import pandas as pd

REQUIRED_COLUMNS = [
    "convo_id",
    "msg_count_5min",
    "tokens_5min",
    "avg_tokens_per_msg",
    "model_tier",
    "user_tier",
    "total_cost_usd"
]

def validate_data(df: pd.DataFrame) -> None:
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if (df["tokens_5min"] < 0).any():
        raise ValueError("Negative token values detected.")
    
    if (df["total_cost_usd"] < 0).any():
        raise ValueError("Negative cost values detected.")
    
    if df["convo_id"].isnull().any():
        raise ValueError("Null conversation IDs detected.")
    
    print("Data validation passed successfully.")
