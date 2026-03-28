from fastapi import FastAPI, Query
from typing import Optional
from .database import fetch_dataframe
from backend.analytics.analytics import compute_kpis

app = FastAPI(title="Agentic Analytics API")


@app.get("/analytics/kpis")
def analytics_kpis(
    user_tier: Optional[str] = Query(None, description="Filter by user tier: free, pro, enterprise"),
    model_tier: Optional[str] = Query(None, description="Filter by model tier: basic, standard, premium"),
    start_date: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter to date (ISO format)"),
):
    """
    Returns KPIs with optional filters.
    Previously: two @app.get('/analytics/kpis') routes — the unfiltered one
    always won, making the filtered version unreachable.
    """
    return compute_kpis(user_tier, model_tier, start_date, end_date)


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
        "high_cost_conversations": int(high_cost_count),
    }


@app.get("/anomalies")
def get_anomalies():
    df = fetch_dataframe(
        "SELECT * FROM usage_metrics WHERE iso_anomaly_flag = 1 OR z_anomaly_flag = 1"
    )
    return {
        "total_anomalies": len(df),
        "anomaly_records": df.head(20).to_dict(orient="records"),
    }


@app.get("/summary")
def get_summary():
    df = fetch_dataframe("SELECT * FROM usage_metrics")
    premium_usage = df[df["model_tier"] == 3]["total_cost_usd"].sum()
    total_cost = df["total_cost_usd"].sum()
    premium_percentage = (premium_usage / total_cost) * 100 if total_cost > 0 else 0
    return {
        "summary": f"Premium model usage accounts for {round(premium_percentage, 2)}% of total cost.",
        "top_5_high_cost_conversations": df.sort_values("total_cost_usd", ascending=False)
        .head(5)[["convo_id", "total_cost_usd"]]
        .to_dict(orient="records"),
    }
