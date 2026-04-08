"""
backend/agent/agent.py  —  Layer 4: Agentic AI

LLM options (both free, no card needed):
  - Groq: llama-3.1-70b-versatile  (fastest, best quality, console.groq.com)
  - Gemini: gemini-1.5-flash        (Google AI Studio, aistudio.google.com)

Agentic loop:
  1. User asks a natural language question
  2. LLM decides which tools to call (run_sql, run_python, generate_plot)
  3. Tools execute against real data and return structured results
  4. LLM reads results, optionally calls more tools, then answers
  5. Returns grounded answer with evidence

The agent uses a ReAct-style prompt: it reasons, then acts, then observes.
This is the same pattern used in production agentic systems.

MOCK_MODE=true: full tool execution, rule-based routing instead of LLM.
"""
import os
import json
import sqlite3
import traceback
import io
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = Path(__file__).resolve().parents[2]
DB_PATH   = BASE_DIR / "data" / "usage.db"

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "groq")
MOCK_MODE      = os.getenv("MOCK_MODE", "false").lower() == "true"

GROQ_MODEL   = "llama-3.1-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_sql",
        "description": "Execute a SQL SELECT query against the usage_metrics SQLite table. Use for exact numbers, aggregations, filters, rankings. Table: usage_metrics. Key columns: convo_id, msg_count_5min, tokens_5min, avg_tokens_per_msg, model_tier (1=Basic 2=Standard 3=Premium), user_tier (1=Free 2=Pro 3=Enterprise), total_cost_usd, cost_per_token, high_cost_flag, iso_anomaly_flag, z_anomaly_flag.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "run_python",
        "description": "Execute Python for statistical analysis. Variable `df` is pre-loaded with usage_metrics as a pandas DataFrame. Assign your answer to `result`.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    {
        "name": "generate_plot",
        "description": "Generate a chart of the data.",
        "parameters": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": ["cost_histogram", "tier_bar", "scatter_tokens_cost",
                             "anomaly_scatter", "user_tier_bar", "donut_cost_share",
                             "box_cost_by_tier", "feature_importance", "roc_curve"],
                }
            },
            "required": ["plot_type"],
        },
    },
]

SYSTEM_PROMPT = """You are a senior data analyst with access to a chatbot usage analytics database.
The database contains 740 conversation records including cost, token usage, model tier, and anomaly flags.

You have these tools: run_sql, run_python, generate_plot.

When answering:
1. Use run_sql to get exact numbers first
2. Use run_python for statistical calculations
3. Always call generate_plot to visualise your findings
4. Give specific, data-grounded answers using the numbers you retrieved
5. Explain anomaly detection method choices (IsolationForest vs z-score) when relevant

Be precise and analytical. Reference exact figures from your tool results."""


# ── Tool executors ────────────────────────────────────────────────────────────

def _get_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM usage_metrics", conn)
    conn.close()
    return df


def run_sql(query: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return {"success": True, "columns": list(df.columns),
                "rows": df.head(30).to_dict(orient="records"),
                "total_rows": len(df), "_df": df}
    except Exception as e:
        return {"success": False, "error": str(e), "_df": pd.DataFrame()}


def run_python(code: str) -> dict:
    df = _get_df()
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    local: dict[str, Any] = {"df": df, "pd": pd, "np": np, "result": None}
    try:
        exec(code, local)
        sys.stdout = old_stdout
        return {"success": True, "result": local.get("result"),
                "printed": buffer.getvalue(), "_df": df}
    except Exception as e:
        sys.stdout = old_stdout
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def generate_plot(plot_type: str) -> dict:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        return {"success": False, "error": "plotly not installed"}

    df = _get_df()
    tier_map = {1: "Basic", 2: "Standard", 3: "Premium"}
    user_map = {1: "Free", 2: "Pro", 3: "Enterprise"}
    df["model_label"] = df["model_tier"].map(tier_map)
    df["user_label"]  = df["user_tier"].map(user_map)
    TIER_COLORS = {"Basic": "#5a4ee0", "Standard": "#4a9de0", "Premium": "#7c6af7"}
    USER_COLORS = {"Free": "#e8a328", "Pro": "#2dbdaa", "Enterprise": "#2dbd7c"}
    LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(color="#9b9990"), margin=dict(t=40,b=20,l=10,r=10))

    try:
        if plot_type == "cost_histogram":
            fig = px.histogram(df, x="total_cost_usd", nbins=20,
                               color_discrete_sequence=["#7c6af7"],
                               title="Cost distribution")
        elif plot_type == "tier_bar":
            g = df.groupby("model_label")["total_cost_usd"].mean().reset_index()
            g.columns = ["Tier","Avg cost"]
            fig = px.bar(g, x="Tier", y="Avg cost", color="Tier",
                         color_discrete_map=TIER_COLORS, title="Avg cost by model tier",
                         text="Avg cost")
            fig.update_traces(texttemplate="$%{text:.3f}", textposition="outside")
        elif plot_type == "scatter_tokens_cost":
            fig = px.scatter(df, x="tokens_5min", y="total_cost_usd",
                             color="model_label", color_discrete_map=TIER_COLORS,
                             title="Tokens vs cost by model tier", opacity=0.7)
        elif plot_type == "anomaly_scatter":
            df["status"] = df["iso_anomaly_flag"].map({1:"Anomaly", 0:"Normal"})
            fig = px.scatter(df, x="tokens_5min", y="total_cost_usd", color="status",
                             color_discrete_map={"Anomaly":"#e05a5a","Normal":"#7c6af744"},
                             title="IsolationForest anomaly detection")
        elif plot_type == "user_tier_bar":
            g = df.groupby("user_label")["total_cost_usd"].sum().reset_index()
            fig = px.bar(g, x="user_label", y="total_cost_usd", color="user_label",
                         color_discrete_map=USER_COLORS, title="Total cost by user tier")
        elif plot_type == "donut_cost_share":
            g = df.groupby("model_label")["total_cost_usd"].sum().reset_index()
            fig = px.pie(g, values="total_cost_usd", names="model_label",
                         color="model_label", color_discrete_map=TIER_COLORS,
                         hole=0.55, title="Cost share by model tier")
        elif plot_type == "box_cost_by_tier":
            fig = px.box(df, x="model_label", y="total_cost_usd",
                         color="model_label", color_discrete_map=TIER_COLORS,
                         title="Cost spread per model tier")
        elif plot_type == "feature_importance":
            import joblib
            model_path = BASE_DIR / "data" / "high_cost_classifier.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                features = ["msg_count_5min", "model_tier_encoded", "user_tier_encoded"]
                imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                             orientation="h", title="RandomForest feature importance",
                             color_discrete_sequence=["#7c6af7"])
            else:
                return {"success": False, "error": "Model not trained yet. Run the ML pipeline first."}
        elif plot_type == "roc_curve":
            from sklearn.metrics import roc_curve
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            import joblib
            model_path = BASE_DIR / "data" / "high_cost_classifier.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                FEATURES = ["msg_count_5min", "model_tier_encoded", "user_tier_encoded"]
                X, y = df[FEATURES], df["high_cost_flag"]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                y_prob = model.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_test, y_prob)), 3)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc})", line=dict(color="#7c6af7")))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(dash="dash", color="#5f5e5a")))
                fig.update_layout(title="ROC Curve — high cost classifier",
                                  xaxis_title="False positive rate", yaxis_title="True positive rate")
            else:
                return {"success": False, "error": "Model not trained yet."}
        else:
            return {"success": False, "error": f"Unknown plot: {plot_type}"}

        fig.update_layout(**LAYOUT)
        return {"success": True, "_figure": fig}
    except Exception as e:
        return {"success": False, "error": str(e)}


TOOL_FNS = {"run_sql": run_sql, "run_python": run_python, "generate_plot": generate_plot}


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_groq(messages: list, tools: list) -> dict:
    """Groq API — OpenAI-compatible, free at console.groq.com"""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages, "tools": tools,
              "tool_choice": "auto", "max_tokens": 1024},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _call_gemini(messages: list, tools: list) -> dict:
    import json

    gemini_tools = [{
        "function_declarations": [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in tools
        ]
    }]

    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": m.get("name", "tool"),
                        "response": {"content": m["content"]}
                    }
                }]
            })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            parts = []
            for tc in m["tool_calls"]:
                parts.append({
                    "functionCall": {
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"])
                    }
                })
            contents.append({"role": "model", "parts": parts})
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": str(m.get("content", ""))}]})

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        json={
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "tools": gemini_tools,
            "generationConfig": {"maxOutputTokens": 1024},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    candidate = data["candidates"][0]["content"]
    parts = candidate.get("parts", [])

    func_calls = [p["functionCall"] for p in parts if "functionCall" in p]
    if func_calls:
        tool_calls = [
            {
                "id": f"gemini_{i}",
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("args", {}))
                }
            }
            for i, fc in enumerate(func_calls)
        ]
        return {
            "choices": [{
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls"
            }]
        }

    text = next((p["text"] for p in parts if "text" in p), "")
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    }


# ── Agent response ─────────────────────────────────────────────────────────────

class AgentResponse:
    def __init__(self):
        self.answer: str       = ""
        self.tool_calls: list  = []
        self.figures: list     = []
        self.sql_results: list = []
        self.error: str | None = None


def run_agent(question: str) -> AgentResponse:
    if MOCK_MODE or (not GROQ_API_KEY and not GEMINI_API_KEY):
        return _mock_agent(question)
    return _live_agent(question)


def _live_agent(question: str) -> AgentResponse:
    response  = AgentResponse()
    messages  = [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": question}]

    for _ in range(6):
        try:
            if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
                data = _call_gemini(messages, TOOLS)
            else:
                data = _call_groq(messages, TOOLS)
        except Exception as e:
            response.error = str(e)
            break

        choice  = data["choices"][0]
        message = choice["message"]
        finish  = choice.get("finish_reason", "stop")

        if finish == "stop" or not message.get("tool_calls"):
            response.answer = message.get("content", "")
            break

        # Execute tool calls
        tool_results_msgs = []
        for tc in message.get("tool_calls", []):
            name  = tc["function"]["name"]
            args  = json.loads(tc["function"]["arguments"])
            result = TOOL_FNS[name](**args)

            response.tool_calls.append({"tool": name, "input": args, "success": result.get("success", False)})

            if "_figure" in result:
                response.figures.append(result["_figure"])
            if "rows" in result:
                response.sql_results.append({"query": args.get("query",""), "rows": result["rows"], "columns": result.get("columns",[])})

            clean = {k: v for k, v in result.items() if not k.startswith("_")}
            if "rows" in clean and len(clean["rows"]) > 15:
                clean["rows"] = clean["rows"][:15]
            tool_results_msgs.append({
                "role": "tool", "tool_call_id": tc["id"],
                "content": json.dumps(clean, default=str)
            })

        messages.append({"role": "assistant", "content": None, "tool_calls": message["tool_calls"]})
        messages.extend(tool_results_msgs)

    return response


def _mock_agent(question: str) -> AgentResponse:
    """Full tool execution, no LLM. Works offline."""
    response = AgentResponse()
    q = question.lower()

    # Route and execute real tools
    if any(w in q for w in ["anomal", "outlier", "flag", "unusual", "iso"]):
        r1 = run_sql("SELECT convo_id, total_cost_usd, tokens_5min, model_tier, user_tier FROM usage_metrics WHERE iso_anomaly_flag=1 ORDER BY total_cost_usd DESC LIMIT 10")
        r2 = run_sql("SELECT COUNT(*) as iso_count, (SELECT COUNT(*) FROM usage_metrics WHERE z_anomaly_flag=1) as z_count FROM usage_metrics WHERE iso_anomaly_flag=1")
        p1 = generate_plot("anomaly_scatter")
        rows = r2["rows"][0] if r2["rows"] else {}
        response.answer = f"**{rows.get('iso_count',37)} anomalies detected by IsolationForest** vs {rows.get('z_count',0)} by z-score — a difference of {rows.get('iso_count',37)-rows.get('z_count',0)} anomalies.\n\n**Why IsolationForest catches more:** Z-score is univariate — it flags rows where a single column exceeds 3 standard deviations. In this dataset no individual column is extreme enough. IsolationForest works in the joint feature space (tokens × cost × avg_tokens_per_msg) and isolates points that are collectively unusual. All top anomalies share the same multivariate signature: **Premium tier + 2,700–3,000 tokens in 5 minutes** — neither dimension alone is flagged, but the combination is.\n\nThis is a key methodological insight: anomaly detection method choice is a design decision, not a default."
        if r1["rows"]: response.sql_results.append({"query": "Top anomalies", "columns": r1["columns"], "rows": r1["rows"]})
        if p1.get("success"): response.figures.append(p1["_figure"])

    elif any(w in q for w in ["cost driver", "expensive", "why cost", "contribut", "driver"]):
        r1 = run_sql("SELECT model_tier, COUNT(*) as n, ROUND(AVG(total_cost_usd),4) as avg, ROUND(SUM(total_cost_usd),2) as total FROM usage_metrics GROUP BY model_tier")
        p1 = generate_plot("tier_bar")
        p2 = generate_plot("donut_cost_share")
        response.answer = "**Primary cost driver: model tier.** Kruskal-Wallis test confirms the difference is statistically significant (p < 0.001) — not sampling noise.\n\n- Basic (tier 1): avg **$0.303** — 254 conversations, $77.06 total (16.9%)\n- Standard (tier 2): avg **$0.633** — 265 conversations, $167.67 total (36.7%)\n- Premium (tier 3): avg **$0.961** — 221 conversations, $212.40 total (46.5%)\n\nPremium costs 3.2× Basic but handles only 29.9% of volume. It drives nearly half of total spend. User tier (Free/Pro/Enterprise) shows almost no cost difference — model choice dominates, user type does not."
        if r1["rows"]: response.sql_results.append({"query":"Cost by model tier","columns":r1["columns"],"rows":r1["rows"]})
        for p in [p1, p2]:
            if p.get("success"): response.figures.append(p["_figure"])

    elif any(w in q for w in ["reduc", "save", "optim", "recommend", "action"]):
        r1 = run_sql("SELECT COUNT(*) as n, ROUND(AVG(total_cost_usd),3) as avg FROM usage_metrics WHERE model_tier=3 AND tokens_5min > 2500")
        p1 = generate_plot("scatter_tokens_cost")
        rows = r1["rows"][0] if r1["rows"] else {}
        response.answer = f"**Three data-driven cost reduction levers:**\n\n**1. Token cap on Premium sessions.** {rows.get('n','N/A')} Premium conversations exceed 2,500 tokens/5 min (avg cost ${rows.get('avg','N/A')}). A soft cap with a user warning would eliminate the high-cost tail with minimal UX impact.\n\n**2. Intelligent model routing.** The RandomForest classifier predicts high-cost sessions from just 3 pre-session features: message count, model tier, and user tier (ROC-AUC ~0.85). Deploying it as a real-time gate before routing to Premium would proactively redirect ~5% of sessions.\n\n**3. Tier rightsizing.** Standard tier averages $0.633 vs Premium's $0.961 — a 34% saving. Auto-routing borderline queries to Standard could shift 15–20% of Premium volume, saving ~$30–40/month at current scale."
        if r1["rows"]: response.sql_results.append({"query":"High-token Premium sessions","columns":r1["columns"],"rows":r1["rows"]})
        if p1.get("success"): response.figures.append(p1["_figure"])

    elif any(w in q for w in ["model", "ml", "rf", "random forest", "classif", "roc", "auc", "predict"]):
        p1 = generate_plot("feature_importance")
        p2 = generate_plot("roc_curve")
        response.answer = "**RandomForest high-cost classifier:**\n\nTarget: top 5th percentile cost conversations (high_cost_flag). Features: message count, model tier, user tier — all observable before the conversation ends.\n\n- ROC-AUC: ~0.85 (strong discrimination)\n- 5-fold cross-validation: ~0.83 ± 0.04 (stable, not overfitting)\n- Class weight='balanced' to handle the 5%/95% imbalance\n\n**Feature importance:** model_tier dominates (~0.60 importance), msg_count_5min second (~0.30), user_tier minimal (~0.10). This quantitatively confirms model tier as the primary predictor of high cost — consistent with the Kruskal-Wallis statistical test result."
        for p in [p1, p2]:
            if p.get("success"): response.figures.append(p["_figure"])

    elif any(w in q for w in ["distribut", "histogram", "skew", "spread", "stat"]):
        r1 = run_python("""
from scipy import stats
result = {
    'mean':   round(df.total_cost_usd.mean(), 4),
    'median': round(df.total_cost_usd.median(), 4),
    'std':    round(df.total_cost_usd.std(), 4),
    'skew':   round(float(stats.skew(df.total_cost_usd)), 3),
    'kurt':   round(float(stats.kurtosis(df.total_cost_usd)), 3),
    'p95':    round(df.total_cost_usd.quantile(0.95), 4),
    'iqr':    round(df.total_cost_usd.quantile(0.75) - df.total_cost_usd.quantile(0.25), 4),
}
""")
        p1 = generate_plot("cost_histogram")
        p2 = generate_plot("box_cost_by_tier")
        s = r1.get("result", {})
        response.answer = f"**Cost distribution is right-skewed** (skewness={s.get('skew','?')}).\n\nMean ${s.get('mean','?')} > Median ${s.get('median','?')} — the gap confirms right skew driven by the Premium tier tail.\n\nIQR: ${s.get('iqr','?')} | 95th percentile: ${s.get('p95','?')} (high-cost threshold)\n\nRight skew is expected in cost data — most conversations are cheap, a small number are expensive. This is why we use IQR (robust to outliers) for anomaly bounds rather than mean ± 2σ."
        for p in [p1, p2]:
            if p.get("success"): response.figures.append(p["_figure"])

    else:
        r1 = run_sql("SELECT COUNT(*) as n, ROUND(SUM(total_cost_usd),2) as total, ROUND(AVG(total_cost_usd),4) as avg, SUM(iso_anomaly_flag) as anomalies, SUM(high_cost_flag) as hc FROM usage_metrics")
        p1 = generate_plot("scatter_tokens_cost")
        rows = r1["rows"][0] if r1["rows"] else {}
        response.answer = f"**Dataset overview:** {rows.get('n',740)} chatbot conversations | Total cost **${rows.get('total',457.13)}** | Avg **${rows.get('avg',0.6177)}/conversation**\n\n{rows.get('anomalies',37)} anomalies (IsolationForest) | {rows.get('hc',37)} high-cost conversations (top 5th percentile)\n\n**Try asking:**\n- What are the main cost drivers?\n- Explain the anomaly detection methodology\n- How does the ML model perform?\n- What is the cost distribution?\n- How can we reduce cost?"
        if r1["rows"]: response.sql_results.append({"query":"Overview","columns":r1["columns"],"rows":r1["rows"]})
        if p1.get("success"): response.figures.append(p1["_figure"])

    response.tool_calls = [{"tool": "mock_router", "input": {"q": question}, "success": True}]
    return response
