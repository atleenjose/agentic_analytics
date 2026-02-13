# transform.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.fillna(0)

    df["cost_per_token"] = df["total_cost_usd"] / (df["tokens_5min"] + 1)
    df["cost_per_message"] = df["total_cost_usd"] / (df["msg_count_5min"] + 1)

    threshold = df["total_cost_usd"].quantile(0.95)
    df["high_cost_flag"] = df["total_cost_usd"] > threshold

    df["model_tier_encoded"] = df["model_tier"].astype("category").cat.codes
    df["user_tier_encoded"] = df["user_tier"].astype("category").cat.codes

    df["z_score_tokens"] = zscore(df["tokens_5min"])
    df["z_score_cost"] = zscore(df["total_cost_usd"])

    df["z_anomaly_flag"] = (
        (abs(df["z_score_tokens"]) > 3) |
        (abs(df["z_score_cost"]) > 3)
    )

    iso_features = df[[
        "tokens_5min",
        "total_cost_usd",
        "avg_tokens_per_msg"
    ]]

    iso_model = IsolationForest(contamination=0.05, random_state=42)
    df["iso_anomaly_flag"] = iso_model.fit_predict(iso_features)

    df["iso_anomaly_flag"] = df["iso_anomaly_flag"] == -1

    print("Transformation + Anomaly Detection complete.")

    return df
