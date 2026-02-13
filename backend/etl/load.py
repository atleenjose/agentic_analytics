# load.py
import sqlite3

def load_to_db(df, db_path="database/usage.db", table_name="usage_metrics"):
    
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"Data loaded into {table_name} table.")
