import sqlite3
from pathlib import Path

DB_PATH = Path("../../database/usage.db")
SCHEMA_PATH = Path("schema.sql")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open(Path(__file__).parent / "schema.sql", "r") as f:
        cursor.executescript(f.read())

    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()