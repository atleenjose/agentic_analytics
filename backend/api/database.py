import sqlite3
import pandas as pd

DB_PATH = "database/usage.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_dataframe(query: str):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df