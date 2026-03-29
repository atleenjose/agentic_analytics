"""
frontend/app.py
Streamlit agentic analytics dashboard.
Run: streamlit run frontend/app.py
"""
import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DB_PATH = BASE_DIR / "data" / "usage.db"
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true" or not os.getenv("ANTHROPIC_API_KEY", "")

from backend.agent.agent import run_agent
from backend.agent.tools import generate_plot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f0f10; }
  [data-testid="stSidebar"] { background: #161618; border-right: 0.5px solid rgba(255,255,255,0.07); }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { color: #e8e6e0 !important; font-weight: 500 !important; }
  p, li, label { color: #9b9990 !important; }
  .stMetric { background: #161618; border: 0.5px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 1rem; }
  .stMetric label { color: #5f5e5a !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.04em; }
  .stMetric [data-testid="stMetricValue"] { color: #e8e6e0 !important; font-family: 'IBM Plex Mono', monospace !important; }
  .stTextInput input, .stTextArea textarea {
    background: #1e1e21 !important; border: 0.5px solid rgba(255,255,255,0.12) !important;
    color: #e8e6e0 !important; border-radius: 8px !important;
  }
  .stButton button {
    background: #7c6af7 !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 500 !important;
  }
  .stButton button:hover { background: #5a4ee0 !important; }
  div[data-testid="stExpander"] { background: #161618; border: 0.5px solid rgba(255,255,255,0.07); border-radius: 10px; }
  .stDataFrame { background: #161618 !important; }
  [data-testid="stMarkdownContainer"] p { color: #9b9990 !important; line-height: 1.7; }
  .agent-answer {
    background: #161618; border: 0.5px solid rgba(124,106,247,0.3);
    border-radius: 10px; padding: 1.2rem 1.4rem;
    color: #e8e6e0; line-height: 1.75; font-size: 14px;
    border-left: 3px solid #7c6af7;
  }
  .tool-chip {
    display: inline-block; background: rgba(124,106,247,0.12);
    color: #7c6af7; border: 0.5px solid rgba(124,106,247,0.3);
    border-radius: 5px; padding: 2px 10px; font-size: 11px;
    font-family: monospace; margin: 2px;
  }
  .mode-badge {
    display: inline-block; padding: 3px 10px; border-radius: 5px;
    font-size: 11px; font-weight: 500; font-family: monospace;
  }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM usage_metrics", conn)
    conn.close()
    tier_map = {1: "Basic", 2: "Standard", 3: "Premium"}
    user_map = {1: "Free", 2: "Pro", 3: "Enterprise"}
    df["model_tier_label"] = df["model_tier"].map(tier_map)
    df["user_tier_label"] = df["user_tier"].map(user_map)
    return df

df = load_data()

TIER_COLORS = {"Basic": "#5a4ee0", "Standard": "#4a9de0", "Premium": "#7c6af7"}
USER_COLORS = {"Free": "#e8a328", "Pro": "#2dbdaa", "Enterprise": "#2dbd7c"}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9b9990", family="Inter, system-ui"),
    margin=dict(t=40, b=20, l=10, r=10),
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ Agentic Analytics")
    mode_color = "#e8a328" if MOCK_MODE else "#2dbd7c"
    mode_label = "Mock mode — no API key" if MOCK_MODE else "Live — Claude API"
    st.markdown(f'<span class="mode-badge" style="background:rgba(124,106,247,0.1);color:{mode_color}">{mode_label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", ["Dashboard", "Agent Chat", "Anomaly Explorer", "Model Explorer"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filters**")
    tier_filter = st.multiselect("Model tier", ["Basic", "Standard", "Premium"], default=["Basic", "Standard", "Premium"])
    user_filter = st.multiselect("User tier", ["Free", "Pro", "Enterprise"], default=["Free", "Pro", "Enterprise"])

    if not MOCK_MODE:
        st.markdown("---")
        st.markdown("**API key set** ✓")
    else:
        st.markdown("---")
        st.info("Add ANTHROPIC_API_KEY to .env for full agentic mode")

df_filtered = df[df["model_tier_label"].isin(tier_filter) & df["user_tier_label"].isin(user_filter)]


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("## Usage analytics dashboard")
    st.markdown(f"*{len(df_filtered):,} conversations · filtered view*")

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Conversations", f"{len(df_filtered):,}")
    c2.metric("Total cost", f"${df_filtered['total_cost_usd'].sum():.2f}")
    c3.metric("Avg cost", f"${df_filtered['total_cost_usd'].mean():.4f}")
    c4.metric("Max cost", f"${df_filtered['total_cost_usd'].max():.2f}")
    c5.metric("Anomalies", f"{df_filtered['iso_anomaly_flag'].sum()}")
    c6.metric("High-cost flag", f"{df_filtered['high_cost_flag'].sum()}")

    st.markdown("---")

    # Row 1: histogram + scatter
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_filtered, x="total_cost_usd", nbins=20,
                           color_discrete_sequence=["#7c6af7"],
                           labels={"total_cost_usd": "Cost (USD)"},
                           title="Cost distribution")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_filtered["anomaly"] = df_filtered["iso_anomaly_flag"].map({1: "Anomaly", 0: "Normal"})
        fig2 = px.scatter(df_filtered, x="tokens_5min", y="total_cost_usd",
                          color="model_tier_label", color_discrete_map=TIER_COLORS,
                          symbol="anomaly", symbol_map={"Anomaly": "x", "Normal": "circle"},
                          labels={"tokens_5min": "Tokens (5 min)", "total_cost_usd": "Cost (USD)",
                                  "model_tier_label": "Tier"},
                          title="Tokens vs cost", opacity=0.7)
        fig2.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: tier bars + user bars + donut
    col3, col4, col5 = st.columns(3)
    with col3:
        g = df_filtered.groupby("model_tier_label")["total_cost_usd"].agg(["mean", "count"]).reset_index()
        g.columns = ["Tier", "Avg cost", "Count"]
        fig3 = px.bar(g, x="Tier", y="Avg cost", color="Tier",
                      color_discrete_map=TIER_COLORS, title="Avg cost by model tier",
                      text="Avg cost")
        fig3.update_traces(texttemplate="$%{text:.3f}", textposition="outside")
        fig3.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        g2 = df_filtered.groupby("user_tier_label")["total_cost_usd"].sum().reset_index()
        g2.columns = ["Tier", "Total cost"]
        fig4 = px.bar(g2, x="Tier", y="Total cost", color="Tier",
                      color_discrete_map=USER_COLORS, title="Total cost by user tier")
        fig4.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        g3 = df_filtered.groupby("model_tier_label")["total_cost_usd"].sum().reset_index()
        fig5 = px.pie(g3, values="total_cost_usd", names="model_tier_label",
                      color="model_tier_label", color_discrete_map=TIER_COLORS,
                      hole=0.55, title="Cost share by model tier")
        fig5.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)

    # Box plots
    fig6 = px.box(df_filtered, x="model_tier_label", y="total_cost_usd",
                  color="model_tier_label", color_discrete_map=TIER_COLORS,
                  labels={"model_tier_label": "Model tier", "total_cost_usd": "Cost (USD)"},
                  title="Cost spread by model tier")
    fig6.update_layout(**CHART_LAYOUT, showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — AGENT CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Agent Chat":
    st.markdown("## Ask the AI analyst")
    if MOCK_MODE:
        st.info("Running in mock mode — tools execute against real data, LLM reasoning is rule-based. Add ANTHROPIC_API_KEY to .env for full Claude-powered analysis.")
    else:
        st.success("Claude API connected — full agentic mode active.")

    # Suggested questions
    st.markdown("**Try asking:**")
    cols = st.columns(4)
    suggestions = [
        "What are the main cost drivers?",
        "Explain the anomaly patterns",
        "How can we reduce cost?",
        "Compare model tiers",
    ]
    for i, s in enumerate(suggestions):
        if cols[i].button(s, key=f"sug_{i}"):
            st.session_state["agent_question"] = s

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a question about the data…",
                              value=st.session_state.get("agent_question", ""),
                              key="agent_input",
                              placeholder="e.g. Why are some conversations so expensive?")

    if st.button("Analyse ↗", key="ask_btn") and question.strip():
        st.session_state["agent_question"] = ""
        with st.spinner("Agent is reasoning…"):
            result = run_agent(question)

        st.session_state.chat_history.append({
            "question": question,
            "result": result,
        })

    # Display history (newest first)
    for item in reversed(st.session_state.chat_history):
        q = item["question"]
        r = item["result"]

        st.markdown(f"**Q: {q}**")

        # Tool chips
        if r.tool_calls:
            chips = " ".join([f'<span class="tool-chip">{t["tool"]}</span>' for t in r.tool_calls])
            st.markdown(f'<div style="margin-bottom:8px">{chips}</div>', unsafe_allow_html=True)

        # Answer
        st.markdown(f'<div class="agent-answer">{r.answer}</div>', unsafe_allow_html=True)

        # Figures
        for fig in r.figures:
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # SQL results
        if r.sql_results:
            with st.expander("View data returned by agent"):
                for sr in r.sql_results:
                    if sr["rows"]:
                        st.dataframe(pd.DataFrame(sr["rows"]), use_container_width=True)

        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Explorer":
    st.markdown("## Anomaly explorer")
    st.markdown("*IsolationForest (multivariate) + Z-score (univariate) detection*")

    anomalies = df[df["iso_anomaly_flag"] == 1].copy()
    normal = df[df["iso_anomaly_flag"] == 0].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total anomalies", len(anomalies))
    c2.metric("Anomaly rate", f"{len(anomalies)/len(df)*100:.1f}%")
    c3.metric("Avg anomaly cost", f"${anomalies['total_cost_usd'].mean():.3f}")
    c4.metric("Avg normal cost", f"${normal['total_cost_usd'].mean():.3f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="tokens_5min", y="total_cost_usd",
                         color=df["iso_anomaly_flag"].map({1: "Anomaly", 0: "Normal"}),
                         color_discrete_map={"Anomaly": "#e05a5a", "Normal": "#7c6af766"},
                         labels={"tokens_5min": "Tokens", "total_cost_usd": "Cost (USD)"},
                         title="IsolationForest anomaly detection")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        model_dist = anomalies["model_tier_label"].value_counts().reset_index()
        model_dist.columns = ["Tier", "Count"]
        fig2 = px.bar(model_dist, x="Tier", y="Count", color="Tier",
                      color_discrete_map=TIER_COLORS, title="Anomalies by model tier")
        fig2.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Top 10 most costly anomalies")
    top = anomalies.sort_values("total_cost_usd", ascending=False).head(10)[
        ["convo_id", "total_cost_usd", "tokens_5min", "msg_count_5min", "model_tier_label", "user_tier_label"]
    ].rename(columns={"model_tier_label": "Model tier", "user_tier_label": "User tier",
                       "total_cost_usd": "Cost (USD)", "tokens_5min": "Tokens",
                       "msg_count_5min": "Messages"})
    st.dataframe(top, use_container_width=True, hide_index=True)

    with st.expander("Why IsolationForest over z-score?"):
        st.markdown("""
**Z-score** flags points where a single column is more than 3 standard deviations from the mean.
In this dataset, no individual column is extreme enough — so z-score finds 0 anomalies.

**IsolationForest** isolates points that are unusual across *multiple dimensions simultaneously*.
A Premium conversation with 2,900 tokens and $1.78 cost isn't extreme on any one axis,
but the *combination* (high tier × high tokens × high cost) makes it easy to isolate.
That's why IsolationForest catches 37 anomalies that z-score misses entirely.

This is a key insight for your portfolio: **anomaly detection method choice matters.**
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Explorer":
    st.markdown("## Model & user tier explorer")

    tab1, tab2 = st.tabs(["Model tier", "User tier"])

    with tab1:
        g = df_filtered.groupby("model_tier_label").agg(
            Conversations=("convo_id", "count"),
            Avg_cost=("total_cost_usd", "mean"),
            Total_cost=("total_cost_usd", "sum"),
            Anomalies=("iso_anomaly_flag", "sum"),
            High_cost=("high_cost_flag", "sum"),
        ).round(4).reset_index().rename(columns={"model_tier_label": "Tier"})
        st.dataframe(g, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(g, x="Tier", y="Avg_cost", color="Tier",
                         color_discrete_map=TIER_COLORS, title="Average cost by model tier",
                         text="Avg_cost")
            fig.update_traces(texttemplate="$%{text:.4f}", textposition="outside")
            fig.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df_filtered, x="model_tier_label", y="total_cost_usd",
                          color="model_tier_label", color_discrete_map=TIER_COLORS,
                          title="Cost distribution by model tier",
                          labels={"model_tier_label": "Tier", "total_cost_usd": "Cost"})
            fig2.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        g2 = df_filtered.groupby("user_tier_label").agg(
            Conversations=("convo_id", "count"),
            Avg_cost=("total_cost_usd", "mean"),
            Total_cost=("total_cost_usd", "sum"),
        ).round(4).reset_index().rename(columns={"user_tier_label": "Tier"})
        st.dataframe(g2, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.bar(g2, x="Tier", y="Total_cost", color="Tier",
                          color_discrete_map=USER_COLORS, title="Total cost by user tier")
            fig3.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = px.violin(df_filtered, x="user_tier_label", y="total_cost_usd",
                             color="user_tier_label", color_discrete_map=USER_COLORS,
                             title="Cost distribution by user tier", box=True,
                             labels={"user_tier_label": "Tier", "total_cost_usd": "Cost"})
            fig4.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
