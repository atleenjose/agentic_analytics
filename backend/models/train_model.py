# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib
# import os

# DATA_PATH = "data/processed/processed_data.csv"
# MODEL_PATH = "backend/models/knn_model.pkl"

# def train():
#     df = pd.read_csv(DATA_PATH)

#     X = df.drop("price", axis=1)
#     y = df["price"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = KNeighborsRegressor(n_neighbors=5)
#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)

#     print("MSE:", mean_squared_error(y_test, predictions))
#     print("R2:", r2_score(y_test, predictions))

#     os.makedirs("backend/models", exist_ok=True)
#     joblib.dump(model, MODEL_PATH)

# if __name__ == "__main__":
#     train()

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DB_PATH = "database/usage.db"

def train_high_cost_model():

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM usage_metrics", conn)
    conn.close()

    features = [
        "msg_count_5min",
        "tokens_5min",
        "avg_tokens_per_msg",
        "model_tier_encoded",
        "user_tier_encoded"
    ]

    X = df[features]
    y = df["high_cost_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    joblib.dump(model, "backend/models/high_cost_model_baseline.pkl")

    print("Model saved successfully.")

def train_high_cost_model_refined():
    # connect DB
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM usage_metrics", conn)
    conn.close()

    # Features WITHOUT tokens/avg_tokens
    features = ["msg_count_5min", "model_tier_encoded", "user_tier_encoded"]
    X = df[features]
    y = df["high_cost_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("Refined Model Report:\n")
    print(classification_report(y_test, y_pred))
    print("Refined ROC AUC:", roc_auc_score(y_test, y_prob))

    joblib.dump(model, "backend/models/high_cost_model_refined.pkl")
    print("Refined model saved successfully.")

if __name__ == "__main__":
    train_high_cost_model()
    train_high_cost_model_refined()