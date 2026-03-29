"""
backend/agent/tools.py
The concrete tools the agent can invoke.
Each tool runs real code against real data and returns structured results.
"""
import sqlite3
import traceback
import io
import sys
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "usage.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_sql(query: str) -> dict:
    """
    Execute a SQL SELECT query against the usage_metrics table.
    Returns rows as a list of dicts plus column names.
    """
    try:
        conn = get_db()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return {
            "success": True,
            "columns": list(df.columns),
            "rows": df.head(50).to_dict(orient="records"),
            "total_rows": len(df),
            "dataframe": df,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "dataframe": pd.DataFrame()}


def run_python(code: str) -> dict:
    """
    Execute arbitrary Python code with pandas/numpy/scipy available.
    The variable `df` is pre-loaded with the full usage_metrics table.
    Any value assigned to `result` is captured and returned.
    """
    try:
        conn = get_db()
        df = pd.read_sql_query("SELECT * FROM usage_metrics", conn)
        conn.close()

        local_vars: dict[str, Any] = {"df": df, "pd": pd, "np": np, "result": None}
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        exec(code, local_vars)
        sys.stdout = old_stdout
        printed = buffer.getvalue()

        return {
            "success": True,
            "result": local_vars.get("result"),
            "printed": printed,
            "dataframe": local_vars.get("result") if isinstance(local_vars.get("result"), pd.DataFrame) else df,
        }
    except Exception as e:
        sys.stdout = old_stdout if 'old_stdout' in dir() else sys.stdout
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


TIER_COLORS = {"Basic": "#5a4ee0", "Standard": "#4a9de0", "Premium": "#7c6af7"}
USER_COLORS = {"Free": "#e8a328", "Pro": "#2dbdaa", "Enterprise": "#2dbd7c"}

def _tier_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model_tier" in df.columns:
        df["model_tier"] = df["model_tier"].map({1: "Basic", 2: "Standard", 3: "Premium"}).fillna(df["model_tier"])
    if "user_tier" in df.columns:
        df["user_tier"] = df["user_tier"].map({1: "Free", 2: "Pro", 3: "Enterprise"}).fillna(df["user_tier"])
    return df


def generate_plot(plot_type: str, **kwargs) -> dict:
    if not HAS_PLOTLY:
        return {"success": False, "error": "plotly not installed. Run: pip install plotly"}
    """
    Generate a Plotly chart. Returns JSON that Streamlit renders.
    Supported types: cost_histogram, tier_bar, scatter_tokens_cost,
                     anomaly_scatter, user_tier_bar, donut_cost_share
    """
    try:
        conn = get_db()
        df = pd.read_sql_query("SELECT * FROM usage_metrics", conn)
        conn.close()
        df = _tier_label(df)

        template = "plotly_white"

        if plot_type == "cost_histogram":
            fig = px.histogram(df, x="total_cost_usd", nbins=20,
                               color_discrete_sequence=["#7c6af7"],
                               labels={"total_cost_usd": "Cost (USD)", "count": "Conversations"},
                               template=template)
            fig.update_layout(title="Cost distribution", showlegend=False)

        elif plot_type == "tier_bar":
            g = df.groupby("model_tier")["total_cost_usd"].agg(["mean", "sum", "count"]).reset_index()
            g.columns = ["Tier", "Avg cost", "Total cost", "Conversations"]
            fig = px.bar(g, x="Tier", y="Avg cost",
                         color="Tier", color_discrete_map=TIER_COLORS,
                         text="Avg cost", template=template)
            fig.update_traces(texttemplate="$%{text:.3f}", textposition="outside")
            fig.update_layout(title="Average cost by model tier", showlegend=False)

        elif plot_type == "scatter_tokens_cost":
            df["anomaly"] = df["iso_anomaly_flag"].map({1: "Anomaly", 0: "Normal"})
            fig = px.scatter(df, x="tokens_5min", y="total_cost_usd",
                             color="model_tier", color_discrete_map=TIER_COLORS,
                             symbol="anomaly", symbol_map={"Anomaly": "x", "Normal": "circle"},
                             labels={"tokens_5min": "Tokens (5 min)", "total_cost_usd": "Cost (USD)"},
                             template=template, opacity=0.7)
            fig.update_layout(title="Tokens vs cost — coloured by model tier")

        elif plot_type == "anomaly_scatter":
            fig = px.scatter(df, x="tokens_5min", y="total_cost_usd",
                             color="iso_anomaly_flag",
                             color_discrete_map={1: "#e05a5a", 0: "#7c6af766"},
                             labels={"tokens_5min": "Tokens", "total_cost_usd": "Cost (USD)",
                                     "iso_anomaly_flag": "Anomaly"},
                             template=template, opacity=0.75)
            fig.update_layout(title="Anomaly detection — IsolationForest")

        elif plot_type == "user_tier_bar":
            g = df.groupby("user_tier")["total_cost_usd"].agg(["mean", "sum", "count"]).reset_index()
            g.columns = ["Tier", "Avg cost", "Total cost", "Conversations"]
            fig = px.bar(g, x="Tier", y="Total cost",
                         color="Tier", color_discrete_map=USER_COLORS,
                         template=template)
            fig.update_layout(title="Total cost by user tier", showlegend=False)

        elif plot_type == "donut_cost_share":
            g = df.groupby("model_tier")["total_cost_usd"].sum().reset_index()
            fig = px.pie(g, values="total_cost_usd", names="model_tier",
                         color="model_tier", color_discrete_map=TIER_COLORS,
                         hole=0.55, template=template)
            fig.update_layout(title="Cost share by model tier")

        elif plot_type == "box_cost_by_tier":
            fig = px.box(df, x="model_tier", y="total_cost_usd",
                         color="model_tier", color_discrete_map=TIER_COLORS,
                         template=template,
                         labels={"model_tier": "Model tier", "total_cost_usd": "Cost (USD)"})
            fig.update_layout(title="Cost distribution by model tier", showlegend=False)

        else:
            return {"success": False, "error": f"Unknown plot type: {plot_type}"}

        fig.update_layout(
            font_family="Inter, system-ui, sans-serif",
            margin=dict(t=50, b=30, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return {"success": True, "figure": fig}

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_schema() -> dict:
    """Return the database schema so the agent knows what columns exist."""
    conn = get_db()
    cursor = conn.execute("PRAGMA table_info(usage_metrics)")
    cols = [{"name": r["name"], "type": r["type"]} for r in cursor.fetchall()]
    sample = pd.read_sql_query("SELECT * FROM usage_metrics LIMIT 3", conn)
    conn.close()
    return {
        "success": True,
        "table": "usage_metrics",
        "columns": cols,
        "sample_rows": sample.to_dict(orient="records"),
        "total_rows": pd.read_sql_query("SELECT COUNT(*) as n FROM usage_metrics", sqlite3.connect(DB_PATH)).iloc[0, 0],
    }


TOOL_REGISTRY = {
    "run_sql": run_sql,
    "run_python": run_python,
    "generate_plot": generate_plot,
    "get_schema": get_schema,
}

TOOL_DEFINITIONS = [
    {
        "name": "run_sql",
        "description": "Execute a SQL SELECT query against the usage_metrics SQLite table. Use this to get exact numbers, aggregations, filters, and rankings from the chatbot usage data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Valid SQLite SELECT query. Table name: usage_metrics. Available columns: convo_id, msg_count_5min, tokens_5min, avg_tokens_per_msg, model_tier (1=Basic 2=Standard 3=Premium), user_tier (1=Free 2=Pro 3=Enterprise), total_cost_usd, cost_per_token, cost_per_message, high_cost_flag, model_tier_encoded, user_tier_encoded, z_anomaly_flag, iso_anomaly_flag."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_python",
        "description": "Execute Python code for statistical analysis. The variable `df` is pre-loaded with the full usage_metrics table as a pandas DataFrame. Assign your final result to `result`.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. pandas (pd), numpy (np) are available. Assign your answer to `result`."}
            },
            "required": ["code"],
        },
    },
    {
        "name": "generate_plot",
        "description": "Generate a Plotly visualization of the data. Choose the best plot type for the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": ["cost_histogram", "tier_bar", "scatter_tokens_cost", "anomaly_scatter", "user_tier_bar", "donut_cost_share", "box_cost_by_tier"],
                    "description": "Type of plot to generate.",
                }
            },
            "required": ["plot_type"],
        },
    },
    {
        "name": "get_schema",
        "description": "Get the database schema and sample rows. Call this first if you are unsure about column names or data types.",
        "input_schema": {"type": "object", "properties": {}},
    },
]
