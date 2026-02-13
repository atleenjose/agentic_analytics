import sqlite3
import pandas as pd
import joblib
from textblob import TextBlob
import os

DB_PATH = "backend/database/usage.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

REFINED_MODEL_PATH = "backend/models/high_cost_model_refined.pkl"

refined_model = joblib.load(REFINED_MODEL_PATH)

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_usage_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM usage_metrics", conn)
    conn.close()
    return df

def compute_kpis(user_tier=None, model_tier=None, start_date=None, end_date=None):
    df = fetch_usage_data()

    if start_date:
        df = df[df['created_at'] >= start_date]
    if end_date:
        df = df[df['created_at'] <= end_date]
    if user_tier:
        df = df[df['user_tier'] == user_tier]
    if model_tier:
        df = df[df['model_tier'] == model_tier]

    tier_summary = df.groupby("user_tier")["total_cost_usd"].agg(
        total_cost="sum",
        avg_cost="mean"
    ).reset_index()

    anomaly_rate = df[["z_anomaly_flag", "iso_anomaly_flag"]].any(axis=1).mean() * 100

    features = ["msg_count_5min", "model_tier_encoded", "user_tier_encoded"]
    df_features = df[features]
    df["high_cost_risk_prob"] = refined_model.predict_proba(df_features)[:,1]

    high_risk_avg = df["high_cost_risk_prob"].mean() * 100

    return {
        "tier_summary": tier_summary.to_dict(orient="records"),
        "anomaly_rate_percent": round(anomaly_rate,2),
        "average_high_cost_risk_percent": round(high_risk_avg,2)
    }


def analyze_text_logs(df):
    """
    df: DataFrame with 'conversation_text' column
    Returns: dict with average sentiment and sample analysis
    """
    if 'conversation_text' not in df.columns:
        return {"average_sentiment": None, "total_logs": 0}

    df['sentiment'] = df['conversation_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    avg_sentiment = df['sentiment'].mean()

    return {
        "average_sentiment": round(avg_sentiment, 2),
        "total_logs": len(df),
        "sample_logs": df[['conversation_text','sentiment']].head(5).to_dict(orient="records")
    }