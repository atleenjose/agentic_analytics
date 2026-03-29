"""
backend/etl/pipeline.py  —  Layer 1: Ingestion

Responsibilities:
  - Extract raw CSV
  - Schema validation (required columns, types, constraints)
  - Data integrity checks (no nulls in keys, no negative costs)
  - Type casting and normalisation
  - Load clean records into SQLite (chatbot_usage table)

Design decision: validation runs BEFORE any transformation so downstream
layers never receive corrupt data. Errors raise immediately with clear messages.
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH   = BASE_DIR / "data" / "usage.db"
CSV_PATH  = BASE_DIR / "data" / "raw" / "chatbot_data.csv"

REQUIRED_COLUMNS = {
    "convo_id":          "object",
    "msg_count_5min":    "int64",
    "tokens_5min":       "int64",
    "avg_tokens_per_msg":"float64",
    "model_tier":        "int64",
    "user_tier":         "int64",
    "total_cost_usd":    "float64",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chatbot_usage (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    convo_id              TEXT    NOT NULL UNIQUE,
    msg_count_5min        INTEGER NOT NULL CHECK(msg_count_5min >= 0),
    tokens_5min           INTEGER NOT NULL CHECK(tokens_5min >= 0),
    avg_tokens_per_msg    REAL    NOT NULL,
    model_tier            INTEGER NOT NULL CHECK(model_tier IN (1,2,3)),
    user_tier             INTEGER NOT NULL CHECK(user_tier  IN (1,2,3)),
    total_cost_usd        REAL    NOT NULL CHECK(total_cost_usd >= 0),
    msg_count_5min_norm   REAL,
    tokens_5min_norm      REAL,
    avg_tokens_per_msg_norm REAL,
    model_tier_norm       REAL,
    user_tier_norm        REAL,
    ingested_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_cost   ON chatbot_usage(total_cost_usd);
CREATE INDEX IF NOT EXISTS idx_tier   ON chatbot_usage(model_tier);
CREATE INDEX IF NOT EXISTS idx_user   ON chatbot_usage(user_tier);
"""


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    null_counts: dict = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def summary(self) -> str:
        lines = [f"Rows: {self.row_count} | Passed: {self.passed}"]
        if self.errors:
            lines += [f"  ERROR: {e}" for e in self.errors]
        if self.warnings:
            lines += [f"  WARN:  {w}" for w in self.warnings]
        return "\n".join(lines)


def extract(path: Path = CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    print(f"[ETL] Extracted {len(df):,} rows from {path.name}")
    return df


def validate(df: pd.DataFrame) -> ValidationReport:
    """
    Schema validation and data integrity checks.
    Runs before transformation to prevent downstream corruption.
    """
    report = ValidationReport(row_count=len(df))

    # 1. Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add_error(f"Missing required columns: {missing}")
        return report  # can't continue without schema

    # 2. Null checks on key fields
    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    report.null_counts = null_counts[null_counts > 0].to_dict()
    if df["convo_id"].isnull().any():
        report.add_error("Null conversation IDs detected — primary key violation")
    if df["total_cost_usd"].isnull().any():
        report.add_error("Null cost values detected")

    # 3. Business constraint checks
    if (df["tokens_5min"] < 0).any():
        report.add_error(f"Negative token values: {(df['tokens_5min'] < 0).sum()} rows")
    if (df["total_cost_usd"] < 0).any():
        report.add_error(f"Negative cost values: {(df['total_cost_usd'] < 0).sum()} rows")
    if (~df["model_tier"].isin([1, 2, 3])).any():
        report.add_error("model_tier must be 1, 2, or 3")
    if (~df["user_tier"].isin([1, 2, 3])).any():
        report.add_error("user_tier must be 1, 2, or 3")

    # 4. Duplicate check
    dupes = df["convo_id"].duplicated().sum()
    if dupes > 0:
        report.add_error(f"{dupes} duplicate convo_ids detected")

    # 5. Warnings (non-fatal)
    if df["total_cost_usd"].max() > 5.0:
        report.warnings.append("Unusually high cost detected (>$5.00) — verify source data")

    status = "PASSED" if report.passed else "FAILED"
    print(f"[ETL] Validation {status}: {len(report.errors)} errors, {len(report.warnings)} warnings")
    return report


def _min_max_norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx != mn else pd.Series(0.0, index=series.index)


def cast_and_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, dtype in REQUIRED_COLUMNS.items():
        df[col] = df[col].astype(dtype)

    # Normalised features (for ML layer)
    for col in ["msg_count_5min", "tokens_5min", "avg_tokens_per_msg", "model_tier", "user_tier"]:
        df[f"{col}_norm"] = _min_max_norm(df[col])

    df["ingested_at"] = pd.Timestamp.now().isoformat()
    return df


def load(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    cols = list(REQUIRED_COLUMNS) + [
        "msg_count_5min_norm", "tokens_5min_norm",
        "avg_tokens_per_msg_norm", "model_tier_norm", "user_tier_norm"
    ]
    insert_df = df[cols].copy()
    insert_df.to_sql("chatbot_usage", conn, if_exists="replace", index=False)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM chatbot_usage").fetchone()[0]
    conn.close()
    print(f"[ETL] Loaded {count:,} rows into chatbot_usage")
    return count


def run_pipeline() -> tuple[pd.DataFrame, ValidationReport]:
    df_raw = extract()
    report = validate(df_raw)
    if not report.passed:
        raise ValueError(f"Validation failed:\n{report.summary()}")
    df_clean = cast_and_normalise(df_raw)
    load(df_clean)
    print("[ETL] Layer 1 complete.")
    return df_clean, report


if __name__ == "__main__":
    df, report = run_pipeline()
    print(report.summary())
