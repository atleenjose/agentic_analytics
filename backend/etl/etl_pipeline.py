# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import os

# RAW_PATH = "data/raw/chatbot_data.csv"
# PROCESSED_PATH = "data/processed/processed_data.csv"

# def run_etl():
#     df = pd.read_csv(RAW_PATH)
#     df = df.dropna()

#     categorical_cols = ["user_type", "region", "topic"]
#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#     numeric_cols = [
#         "message_count",
#         "avg_response_time",
#         "conversation_duration",
#         "sentiment_score"
#     ]

#     scaler = StandardScaler()
#     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#     os.makedirs("data/processed", exist_ok=True)
#     df.to_csv(PROCESSED_PATH, index=False)

# if __name__ == "__main__":
#     run_etl()


# import pandas as pd

# RAW_PATH = "data/raw/chatbot_data.csv"

# df = pd.read_csv(RAW_PATH)
# print(df.columns)

import pandas as pd
from extract import extract_data
from transform import transform_data
from validate import validate_data
from load import load_csv_to_sqlite


def run_pipeline() -> None:
    file_path = "data/raw/chatbot_data.csv"

    # 1. Extract
    df = extract_data(file_path)

    # 2. Validate
    validate_data(df)

    # 3. Add ingestion timestamp
    df["ingestion_timestamp"] = pd.Timestamp.now()

    # 4. Transform (adds anomaly flags, cost features, encoded tiers)
    df_transformed = transform_data(df)

    # 5. Load raw CSV to chatbot_usage table
    load_csv_to_sqlite()

    print("ETL Pipeline completed successfully.")
    print(f"  Rows processed: {len(df_transformed)}")
    print(f"  Anomalies detected: {df_transformed['iso_anomaly_flag'].sum()}")
    print(f"  High cost flagged: {df_transformed['high_cost_flag'].sum()}")


if __name__ == "__main__":
    run_pipeline()
