import sqlite3
import pandas as pd

conn = sqlite3.connect("database/usage.db")

df = pd.read_sql("SELECT * FROM usage_metrics LIMIT 5", conn)
print(df.head())

conn.close()
