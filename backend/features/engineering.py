"""
backend/features/engineering.py  —  Layer 2: Feature Engineering & Statistics

Demonstrates:
  - Cost efficiency derived features
  - Rolling/lag temporal features
  - IQR-based outlier bounds
  - Z-score standardisation
  - Kruskal-Wallis hypothesis test (non-parametric ANOVA)
    → Proves model_tier cost differences are statistically significant
  - Pearson correlation analysis
  - Writes enriched records to usage_metrics table
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH  = BASE_DIR / "data" / "usage.db"

METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_metrics AS
SELECT * FROM chatbot_usage WHERE 0;
"""


@dataclass
class StatisticalReport:
    """Structured results from the hypothesis testing layer."""
    kruskal_statistic: float = 0.0
    kruskal_pvalue:    float = 0.0
    kruskal_significant: bool = False
    tier_means:        dict  = field(default_factory=dict)
    tier_medians:      dict  = field(default_factory=dict)
    pearson_tokens_cost: float = 0.0
    iqr_lower:         float = 0.0
    iqr_upper:         float = 0.0
    skewness:          float = 0.0
    kurtosis:          float = 0.0

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.kruskal_significant else "NOT SIGNIFICANT"
        return (
            f"Kruskal-Wallis H={self.kruskal_statistic:.3f}, p={self.kruskal_pvalue:.4f} [{sig}]\n"
            f"Tier mean costs: {self.tier_means}\n"
            f"Pearson r (tokens↔cost): {self.pearson_tokens_cost:.3f}\n"
            f"IQR bounds: [{self.iqr_lower:.4f}, {self.iqr_upper:.4f}]\n"
            f"Skewness: {self.skewness:.3f} | Kurtosis: {self.kurtosis:.3f}"
        )


def load_raw(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM chatbot_usage", conn)
    conn.close()
    return df


def add_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cost efficiency metrics from raw usage columns."""
    df = df.copy()
    df["cost_per_token"]   = df["total_cost_usd"] / (df["tokens_5min"] + 1)
    df["cost_per_message"] = df["total_cost_usd"] / (df["msg_count_5min"] + 1)
    df["token_efficiency"] = df["tokens_5min"] / (df["total_cost_usd"] + 0.001)
    df["msg_density"]      = df["msg_count_5min"] / (df["tokens_5min"] + 1)
    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score, IQR bounds, percentile bucket."""
    df = df.copy()

    # Z-score on cost and tokens
    df["z_score_cost"]   = stats.zscore(df["total_cost_usd"])
    df["z_score_tokens"] = stats.zscore(df["tokens_5min"])
    df["z_anomaly_flag"] = (
        (df["z_score_cost"].abs() > 3) | (df["z_score_tokens"].abs() > 3)
    ).astype(int)

    # IQR bounds
    Q1, Q3 = df["total_cost_usd"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df["iqr_lower_bound"] = Q1 - 1.5 * IQR
    df["iqr_upper_bound"] = Q3 + 1.5 * IQR
    df["iqr_outlier_flag"] = (
        (df["total_cost_usd"] < df["iqr_lower_bound"]) |
        (df["total_cost_usd"] > df["iqr_upper_bound"])
    ).astype(int)

    # Cost percentile bucket (decile label 1–10)
    df["cost_decile"] = pd.qcut(df["total_cost_usd"], q=10, labels=False, duplicates="drop") + 1

    # High cost flag: top 5th percentile
    p95 = df["total_cost_usd"].quantile(0.95)
    df["high_cost_flag"] = (df["total_cost_usd"] > p95).astype(int)

    return df


def add_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding for ML layer."""
    df = df.copy()
    df["model_tier_encoded"] = df["model_tier"].astype("category").cat.codes
    df["user_tier_encoded"]  = df["user_tier"].astype("category").cat.codes
    return df


def run_hypothesis_tests(df: pd.DataFrame) -> StatisticalReport:
    """
    Kruskal-Wallis test: are model tier cost differences statistically significant?
    (Non-parametric — does not assume normality, appropriate for skewed cost data.)

    H0: All three tiers have the same cost distribution
    H1: At least one tier has a different distribution

    Result: p < 0.05 → reject H0 → cost differences are real, not sampling noise.
    """
    report = StatisticalReport()

    groups = [
        df[df["model_tier"] == t]["total_cost_usd"].values
        for t in [1, 2, 3]
    ]
    h_stat, p_val = stats.kruskal(*groups)
    report.kruskal_statistic  = round(float(h_stat), 4)
    report.kruskal_pvalue     = round(float(p_val), 6)
    report.kruskal_significant = p_val < 0.05

    report.tier_means   = {t: round(df[df["model_tier"]==t]["total_cost_usd"].mean(), 4) for t in [1,2,3]}
    report.tier_medians = {t: round(df[df["model_tier"]==t]["total_cost_usd"].median(), 4) for t in [1,2,3]}

    r, _ = stats.pearsonr(df["tokens_5min"], df["total_cost_usd"])
    report.pearson_tokens_cost = round(float(r), 4)

    Q1, Q3 = df["total_cost_usd"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    report.iqr_lower = round(float(Q1 - 1.5*IQR), 4)
    report.iqr_upper = round(float(Q3 + 1.5*IQR), 4)
    report.skewness  = round(float(df["total_cost_usd"].skew()), 4)
    report.kurtosis  = round(float(df["total_cost_usd"].kurt()), 4)

    print(f"[Features] Kruskal-Wallis p={report.kruskal_pvalue} → {'significant' if report.kruskal_significant else 'not significant'}")
    return report


def save_metrics(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    df.to_sql("usage_metrics", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_m_cost ON usage_metrics(total_cost_usd)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_m_tier ON usage_metrics(model_tier)")
    conn.commit()
    conn.close()
    print(f"[Features] Saved {len(df):,} rows to usage_metrics")


def run_feature_engineering() -> tuple[pd.DataFrame, StatisticalReport]:
    df = load_raw()
    df = add_cost_features(df)
    df = add_statistical_features(df)
    df = add_encoded_features(df)
    report = run_hypothesis_tests(df)
    save_metrics(df, DB_PATH)
    print("[Features] Layer 2 complete.")
    return df, report


if __name__ == "__main__":
    df, report = run_feature_engineering()
    print(report.summary())
