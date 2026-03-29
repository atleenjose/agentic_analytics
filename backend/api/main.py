"""
backend/api/main.py  —  Layer 5a: API

Clean REST API exposing each layer's outputs.
Run: uvicorn backend.api.main:app --reload --port 8000
"""
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH  = BASE_DIR / "data" / "usage.db"

app = FastAPI(title="Agentic Analytics API", version="2.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_df(table: str = "usage_metrics", where: str = "") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    sql = f"SELECT * FROM {table}"
    if where: sql += f" WHERE {where}"
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


class AskRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "running", "version": "2.0", "docs": "/docs"}


@app.post("/ask")
def ask(req: AskRequest):
    """Agentic endpoint — natural language → grounded answer."""
    try:
        from backend.agent.agent import run_agent
        r = run_agent(req.question)
        return {"answer": r.answer, "tool_calls": r.tool_calls,
                "sql_results": r.sql_results, "has_figures": len(r.figures) > 0,
                "error": r.error}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/kpis")
def kpis(model_tier: Optional[int] = None, user_tier: Optional[int] = None):
    """Core KPIs with optional tier filters."""
    where_clauses = []
    if model_tier: where_clauses.append(f"model_tier={model_tier}")
    if user_tier:  where_clauses.append(f"user_tier={user_tier}")
    df = get_df(where=" AND ".join(where_clauses) if where_clauses else "")
    return {
        "total_conversations":   len(df),
        "total_cost_usd":        round(float(df["total_cost_usd"].sum()), 2),
        "avg_cost_usd":          round(float(df["total_cost_usd"].mean()), 4),
        "max_cost_usd":          round(float(df["total_cost_usd"].max()), 4),
        "anomaly_count":         int(df["iso_anomaly_flag"].sum()),
        "anomaly_rate_pct":      round(float(df["iso_anomaly_flag"].mean() * 100), 2),
        "high_cost_count":       int(df["high_cost_flag"].sum()),
        "high_cost_rate_pct":    round(float(df["high_cost_flag"].mean() * 100), 2),
    }


@app.get("/anomalies")
def anomalies(limit: int = 20, method: str = Query("iso", enum=["iso", "zscore", "both"])):
    """Anomalous conversations. method: iso=IsolationForest, zscore=z-score, both=either."""
    where = {
        "iso":    "iso_anomaly_flag=1",
        "zscore": "z_anomaly_flag=1",
        "both":   "iso_anomaly_flag=1 OR z_anomaly_flag=1",
    }[method]
    df = get_df(where=where)
    df = df.sort_values("total_cost_usd", ascending=False).head(limit)
    return {"total": len(df), "method": method, "records": df.to_dict(orient="records")}


@app.get("/tier-breakdown")
def tier_breakdown():
    df = get_df()
    by_model = df.groupby("model_tier")["total_cost_usd"].agg(
        conversations="count", avg_cost="mean", total_cost="sum"
    ).round(4).reset_index().to_dict(orient="records")
    by_user = df.groupby("user_tier")["total_cost_usd"].agg(
        conversations="count", avg_cost="mean", total_cost="sum"
    ).round(4).reset_index().to_dict(orient="records")
    return {"by_model": by_model, "by_user": by_user}


@app.get("/statistics")
def statistics():
    """Statistical summary including hypothesis test results."""
    from scipy import stats
    df = get_df()
    groups = [df[df["model_tier"]==t]["total_cost_usd"].values for t in [1,2,3]]
    h, p = stats.kruskal(*groups)
    r, _ = stats.pearsonr(df["tokens_5min"], df["total_cost_usd"])
    return {
        "kruskal_wallis": {"H": round(float(h),4), "p": round(float(p),6),
                           "significant": bool(p < 0.05), "interpretation":
                           "Model tier cost differences are statistically significant" if p < 0.05
                           else "No significant difference detected"},
        "pearson_tokens_cost": round(float(r), 4),
        "cost_distribution": {
            "mean":   round(float(df["total_cost_usd"].mean()), 4),
            "median": round(float(df["total_cost_usd"].median()), 4),
            "std":    round(float(df["total_cost_usd"].std()), 4),
            "skew":   round(float(df["total_cost_usd"].skew()), 4),
            "p25":    round(float(df["total_cost_usd"].quantile(0.25)), 4),
            "p75":    round(float(df["total_cost_usd"].quantile(0.75)), 4),
            "p95":    round(float(df["total_cost_usd"].quantile(0.95)), 4),
        }
    }


@app.get("/schema")
def schema():
    conn = sqlite3.connect(DB_PATH)
    cols = [{"name": r[1], "type": r[2]} for r in
            conn.execute("PRAGMA table_info(usage_metrics)").fetchall()]
    n = conn.execute("SELECT COUNT(*) FROM usage_metrics").fetchone()[0]
    conn.close()
    return {"table": "usage_metrics", "columns": cols, "total_rows": n}
