import random
import sqlite3
import os

from backend.analytics.analytics import fetch_usage_data

DB_PATH = "backend/database/usage.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

mock_texts = [
    "I love this service, it works perfectly!",
    "The response time was too slow and frustrating.",
    "Not sure how to use the new feature.",
    "Everything is great, no complaints!",
    "I am unhappy with the support I received.",
    "The AI gave me exactly what I needed, very helpful.",
    "This product is confusing and hard to navigate."
]

df = fetch_usage_data()

def add_mock_conversation_text(df):
    if 'conversation_text' not in df.columns:
        df['conversation_text'] = [random.choice(mock_texts) for _ in range(len(df))]
    return df

df = add_mock_conversation_text(df)

conn = sqlite3.connect(DB_PATH)
df.to_sql("usage_metrics", conn, if_exists="replace", index=False)
conn.close()

print("conversation_text column added and saved to DB successfully!")
