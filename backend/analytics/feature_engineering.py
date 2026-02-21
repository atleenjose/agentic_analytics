import pandas as pd
import numpy as np

def add_ratio_features(df):
    # Cost per token (if tokens exist)
    if "total_tokens" in df.columns:
        df["cost_per_token"] = df["total_cost_usd"] / df["total_tokens"]

    # Relative cost vs global average
    global_avg = df["total_cost_usd"].mean()
    df["cost_vs_avg_ratio"] = df["total_cost_usd"] / global_avg

    return df

def add_lag_features(df):
    df = df.sort_values("timestamp")

    df["prev_cost"] = df["total_cost_usd"].shift(1)
    df["cost_change"] = df["total_cost_usd"] - df["prev_cost"]

    return df

def add_rolling_features(df, window=7):
    df = df.sort_values("timestamp")

    df["rolling_mean_cost"] = df["total_cost_usd"].rolling(window).mean()
    df["rolling_std_cost"] = df["total_cost_usd"].rolling(window).std()

    return df

def compute_risk_score(df):
    # Normalize features
    df["z_cost"] = (
        (df["total_cost_usd"] - df["total_cost_usd"].mean())
        / df["total_cost_usd"].std()
    )

    df["volatility_score"] = (
        df["rolling_std_cost"] / df["rolling_std_cost"].mean()
    )

    # Composite risk score
    df["risk_score"] = (
        0.6 * df["z_cost"].abs() +
        0.4 * df["volatility_score"].fillna(0)
    )

    return df

def monthly_growth(df):
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")

    monthly = df.groupby("month")["total_cost_usd"].sum()

    growth = monthly.pct_change()

    return monthly, growth