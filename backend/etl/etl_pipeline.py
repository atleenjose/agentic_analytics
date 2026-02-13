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

# pipeline.py
from extract import extract_data
from transform import transform_data
from load import load_to_db
import pandas as pd
from validate import validate_data

def run_pipeline():
    
    file_path = "data/raw/chatbot_data.csv"
    
    df = extract_data(file_path)
    validate_data(df)
    df["ingestion_timestamp"] = pd.Timestamp.now()
    df_transformed = transform_data(df)
    load_to_db(df_transformed)
    
    print("ETL Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
