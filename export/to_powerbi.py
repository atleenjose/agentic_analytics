import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "usage.db"
EXPORT_DIR = BASE_DIR / "data" / "processed"
EXPORT_DIR.mkdir(exist_ok=True)


POWER_BI_STEPS = """
╔══════════════════════════════════════════════════════════════════════╗
║          HOW TO CONNECT THESE FILES TO POWER BI DESKTOP            ║
╚══════════════════════════════════════════════════════════════════════╝

1. OPEN POWER BI DESKTOP
   Download free from: https://powerbi.microsoft.com/desktop

2. GET DATA
   Home → Get Data → Text/CSV
   Load each of these files:
     - data/processed/usage_overview.csv        (KPI summary)
     - data/processed/conversations.csv         (full dataset)
     - data/processed/tier_summary.csv          (tier breakdown)
     - data/processed/anomalies.csv             (anomaly records)

3. BUILD YOUR REPORT — RECOMMENDED VISUALS:

   Page 1 — Executive Overview
   ┌─────────────────────────────────────┐
   │ Card: Total Conversations           │
   │ Card: Total Cost USD                │
   │ Card: Avg Cost USD                  │
   │ Card: Anomaly Count                 │
   │ Bar chart: avg_cost by model_tier   │
   │ Donut: total_cost by model_tier     │
   └─────────────────────────────────────┘

   Page 2 — Cost Analysis
   ┌─────────────────────────────────────┐
   │ Histogram: total_cost_usd           │
   │   (use Play Axis or bin the data)   │
   │ Scatter: tokens_5min vs cost        │
   │   Color by: model_tier              │
   │ Box plot: cost by tier              │
   └─────────────────────────────────────┘

   Page 3 — Anomaly Report
   ┌─────────────────────────────────────┐
   │ Table: anomalies.csv columns        │
   │ Bar: anomalies by model tier        │
   │ Slicer: user_tier                   │
   └─────────────────────────────────────┘

4. ADD SLICERS (interactive filters)
   - model_tier (select multiple)
   - user_tier  (select multiple)
   These will filter all visuals on the page.

5. ADD MEASURES (DAX formulas)
   In the Data pane, right-click your table → New measure:

   Anomaly Rate % =
     DIVIDE(
       COUNTROWS(FILTER('conversations', 'conversations'[iso_anomaly_flag] = 1)),
       COUNTROWS('conversations')
     ) * 100

   Cost per Token =
     DIVIDE(SUM('conversations'[total_cost_usd]), SUM('conversations'[tokens_5min]))

   High Cost Rate % =
     DIVIDE(
       COUNTROWS(FILTER('conversations', 'conversations'[high_cost_flag] = 1)),
       COUNTROWS('conversations')
     ) * 100

6. THEME
   View → Themes → Browse → choose a dark theme JSON for a professional look.
   Free themes: https://community.powerbi.com/t5/Themes-Gallery/

7. PUBLISH (optional)
   If you have a Power BI account:
   Home → Publish → My Workspace
   Then share the report link with a viewer.

"""


def export():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM usage_metrics", conn)
    conn.close()

    # Add human-readable labels
    tier_map = {1: "Basic", 2: "Standard", 3: "Premium"}
    user_map = {1: "Free", 2: "Pro", 3: "Enterprise"}
    df["model_tier_label"] = df["model_tier"].map(tier_map)
    df["user_tier_label"] = df["user_tier"].map(user_map)

    # 1. Full conversations table
    df.to_csv(EXPORT_DIR / "conversations.csv", index=False)
    try:
         df.to_parquet(EXPORT_DIR/"conversations.parquet", index=False)
         print(f"[Export] conversations.parquet — {len(df)} rows")
    except Exception:
        print("[Export] Parquet skipped — CSV is sufficient for Power BI")

    # 2. KPI summary (single-row overview for Power BI cards)
    kpis = pd.DataFrame([{
        "total_conversations": len(df),
        "total_cost_usd": round(df["total_cost_usd"].sum(), 2),
        "avg_cost_usd": round(df["total_cost_usd"].mean(), 4),
        "median_cost_usd": round(df["total_cost_usd"].median(), 4),
        "max_cost_usd": round(df["total_cost_usd"].max(), 4),
        "anomaly_count": int(df["iso_anomaly_flag"].sum()),
        "anomaly_rate_pct": round(df["iso_anomaly_flag"].mean() * 100, 2),
        "high_cost_count": int(df["high_cost_flag"].sum()),
        "high_cost_rate_pct": round(df["high_cost_flag"].mean() * 100, 2),
    }])
    kpis.to_csv(EXPORT_DIR / "usage_overview.csv", index=False)
    print(f"[Export] usage_overview.csv — KPI summary")

    # 3. Tier summary (for bar/donut charts)
    tier_summary = df.groupby(["model_tier", "model_tier_label"]).agg(
        conversations=("convo_id", "count"),
        avg_cost=("total_cost_usd", "mean"),
        total_cost=("total_cost_usd", "sum"),
        anomalies=("iso_anomaly_flag", "sum"),
        high_cost=("high_cost_flag", "sum"),
    ).round(4).reset_index()
    tier_summary.to_csv(EXPORT_DIR / "tier_summary.csv", index=False)
    print(f"[Export] tier_summary.csv — {len(tier_summary)} tiers")

    # 4. Anomalies only
    anomalies = df[df["iso_anomaly_flag"] == 1].sort_values("total_cost_usd", ascending=False)
    anomalies.to_csv(EXPORT_DIR / "anomalies.csv", index=False)
    print(f"[Export] anomalies.csv — {len(anomalies)} records")

    print(f"\nAll files saved to: {EXPORT_DIR}")
    print(POWER_BI_STEPS)


if __name__ == "__main__":
    export()
