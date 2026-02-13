from fastapi import FastAPI
from .database import fetch_dataframe
import pandas as pd
from backend.analytics.analytics import compute_kpis
from fastapi import Query

app = FastAPI(title="Agentic Analytics API")

@app.get("/analytics/kpis")
def analytics_kpis(
    user_tier: str | None = Query(None),
    model_tier: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None)
):
    return compute_kpis(user_tier, model_tier, start_date, end_date)


@app.get("/analytics/kpis")
def analytics_kpis():
    return compute_kpis()

@app.get("/kpis")
def get_kpis():

    df = fetch_dataframe("SELECT * FROM usage_metrics")

    total_cost = df["total_cost_usd"].sum()
    avg_cost = df["total_cost_usd"].mean()
    total_conversations = len(df)
    high_cost_count = df["high_cost_flag"].sum()

    return {
        "total_cost_usd": round(float(total_cost), 2),
        "average_cost_per_conversation": round(float(avg_cost), 4),
        "total_conversations": int(total_conversations),
        "high_cost_conversations": int(high_cost_count)
    }

@app.get("/anomalies")
def get_anomalies():

    df = fetch_dataframe(
        "SELECT * FROM usage_metrics WHERE iso_anomaly_flag = 1 OR z_anomaly_flag = 1"
    )

    return {
        "total_anomalies": len(df),
        "anomaly_records": df.head(20).to_dict(orient="records")
    }

@app.get("/summary")
def get_summary():

    df = fetch_dataframe("SELECT * FROM usage_metrics")

    premium_usage = df[df["model_tier"] == "premium"]["total_cost_usd"].sum()
    total_cost = df["total_cost_usd"].sum()

    premium_percentage = (premium_usage / total_cost) * 100 if total_cost > 0 else 0

    return {
        "summary": f"Premium model usage accounts for {round(premium_percentage,2)}% of total cost.",
        "top_5_high_cost_conversations": df.sort_values(
            "total_cost_usd", ascending=False
        ).head(5)[["convo_id", "total_cost_usd"]].to_dict(orient="records")
    }
