import sqlite3
import pandas as pd
from pathlib import Path
from insight_engine import generate_insight

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "usage.db"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM chatbot_usage", conn)
    conn.close()
    return df

def check_missing(df):
    return df.isnull().sum()

def basic_stats(df):
    return df.describe()

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

def tier_distribution(df):
    return df.groupby("user_tier")["total_cost_usd"].agg(
        count="count",
        avg_cost="mean",
        total_cost="sum"
    ).reset_index()

def generate_structured_summary(df):
    outliers, lower, upper = detect_outliers_iqr(df, "total_cost_usd")

    summary = {
        "total_rows": len(df),
        "avg_cost": round(df["total_cost_usd"].mean(), 4),
        "max_cost": round(df["total_cost_usd"].max(), 4),
        "min_cost": round(df["total_cost_usd"].min(), 4),
        "outlier_count": len(outliers),
        "cost_iqr_bounds": (round(lower,4), round(upper,4)),
        "tier_distribution": tier_distribution(df).to_dict(orient="records")
    }

    return summary

def build_llm_prompt(summary):
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
- Average cost: ${round(tier['avg_cost'],4)}
- Total cost: ${round(tier['total_cost'],2)}
"""

    prompt += "\nProvide 5 executive-level analytical insights focusing on cost drivers, risk patterns, and business implications."

    return prompt

def main():
    df = load_data()

    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:\n", check_missing(df))
    print("\nBasic Statistics:\n", basic_stats(df))

    summary = generate_structured_summary(df)

    print("\nStructured Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    insights = generate_insight(summary)

    print("\nExecutive Insights:\n")
    print(insights)

if __name__ == "__main__":
    main()