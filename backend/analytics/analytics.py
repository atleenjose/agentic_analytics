import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
from insight_engine import generate_insight

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "usage.db"


def load_data(table: str = "usage_metrics") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


def detect_outliers_iqr(df: pd.DataFrame, column: str):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper


def tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("user_tier")["total_cost_usd"]
        .agg(count="count", avg_cost="mean", total_cost="sum")
        .reset_index()
    )


def generate_structured_summary(df: pd.DataFrame) -> dict:
    outliers, lower, upper = detect_outliers_iqr(df, "total_cost_usd")
    summary = {
        "total_rows": len(df),
        "avg_cost": round(df["total_cost_usd"].mean(), 4),
        "max_cost": round(df["total_cost_usd"].max(), 4),
        "min_cost": round(df["total_cost_usd"].min(), 4),
        "outlier_count": len(outliers),
        "cost_iqr_bounds": (round(lower, 4), round(upper, 4)),
        "tier_distribution": tier_distribution(df).to_dict(orient="records"),
    }
    return summary


def build_llm_prompt(summary: dict) -> str:
    prompt = f"""
You are a senior analytics consultant.

Dataset Overview:
- Total conversations: {summary['total_rows']}
- Average cost per conversation: ${summary['avg_cost']}
- Maximum observed cost: ${summary['max_cost']}
- Minimum observed cost: ${summary['min_cost']}
- Statistical outliers detected (IQR method): {summary['outlier_count']}
- Cost IQR bounds: {summary['cost_iqr_bounds']}

User Tier Breakdown:
"""
    for tier in summary["tier_distribution"]:
        prompt += f"""
Tier {tier['user_tier']}:
- Conversations: {tier['count']}
- Average cost: ${round(tier['avg_cost'], 4)}
- Total cost: ${round(tier['total_cost'], 2)}
"""
    prompt += "\nProvide 5 executive-level analytical insights focusing on cost drivers, risk patterns, and business implications."
    return prompt


def compute_kpis(
    user_tier: Optional[str] = None,
    model_tier: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Compute KPIs with optional filters.
    Previously missing — was imported in main.py but not defined here.
    """
    df = load_data()

    if user_tier is not None:
        tier_map = {"free": 1, "pro": 2, "enterprise": 3}
        df = df[df["user_tier"] == tier_map.get(user_tier.lower(), user_tier)]

    if model_tier is not None:
        tier_map = {"basic": 1, "standard": 2, "premium": 3}
        df = df[df["model_tier"] == tier_map.get(model_tier.lower(), model_tier)]

    if start_date:
        df = df[df["ingestion_timestamp"] >= start_date]
    if end_date:
        df = df[df["ingestion_timestamp"] <= end_date]

    total_cost = df["total_cost_usd"].sum()
    avg_cost = df["total_cost_usd"].mean()
    total_conversations = len(df)
    high_cost_count = df["high_cost_flag"].sum() if "high_cost_flag" in df.columns else 0
    anomaly_count = df["iso_anomaly_flag"].sum() if "iso_anomaly_flag" in df.columns else 0

    tier_stats = (
        df.groupby("model_tier")["total_cost_usd"]
        .agg(count="count", avg="mean", total="sum")
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "total_cost_usd": round(float(total_cost), 2),
        "average_cost_per_conversation": round(float(avg_cost), 4),
        "total_conversations": int(total_conversations),
        "high_cost_conversations": int(high_cost_count),
        "anomaly_count": int(anomaly_count),
        "model_tier_breakdown": tier_stats,
    }


def main():
    df = load_data()

    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:\n", check_missing(df))
    print("\nBasic Statistics:\n", basic_stats(df))

    summary = generate_structured_summary(df)

    print("\nStructured Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    prompt = build_llm_prompt(summary)
    insights = generate_insight(summary)

    print("\nExecutive Insights (rule-based):\n")
    print(insights)


if __name__ == "__main__":
    main()
