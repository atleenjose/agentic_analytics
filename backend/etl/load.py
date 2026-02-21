# load.py
import sqlite3
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DB_PATH = BASE_DIR / "database" / "usage.db"
CSV_PATH = BASE_DIR / "data" / "raw" / "chatbot_data.csv"

def load_csv_to_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open(CSV_PATH, "r") as file:
        reader = csv.DictReader(file)

        rows = [
            (
                row["convo_id"],
                int(row["msg_count_5min"]),
                int(row["tokens_5min"]),
                float(row["avg_tokens_per_msg"]),
                int(row["model_tier"]),
                int(row["user_tier"]),
                float(row["total_cost_usd"]),
                float(row["msg_count_5min_norm"]),
                float(row["tokens_5min_norm"]),
                float(row["avg_tokens_per_msg_norm"]),
                float(row["model_tier_norm"]),
                float(row["user_tier_norm"]),
            )
            for row in reader
        ]

    cursor.executemany("""
        INSERT OR IGNORE INTO chatbot_usage (
            convo_id,
            msg_count_5min,
            tokens_5min,
            avg_tokens_per_msg,
            model_tier,
            user_tier,
            total_cost_usd,
            msg_count_5min_norm,
            tokens_5min_norm,
            avg_tokens_per_msg_norm,
            model_tier_norm,
            user_tier_norm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()

    print(f"{len(rows)} rows inserted.")

if __name__ == "__main__":
    load_csv_to_sqlite()