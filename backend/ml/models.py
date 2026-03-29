"""
backend/ml/models.py  —  Layer 3: Machine Learning

Two models:
  1. IsolationForest  — unsupervised anomaly detection (multivariate)
     Why not z-score? Explained in the report: z-score is univariate and
     finds 0 anomalies here because no single column is extreme enough.
     IsolationForest detects anomalies in the joint feature space.

  2. RandomForest classifier — supervised high-cost prediction
     Features: msg_count_5min, model_tier_encoded, user_tier_encoded
     Target: high_cost_flag (top 5th percentile)
     Evaluation: classification report, ROC-AUC, confusion matrix,
                 feature importance

Demonstrates: ML pipeline, model selection rationale, proper evaluation.
"""
import sqlite3
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
)

BASE_DIR   = Path(__file__).resolve().parents[2]
DB_PATH    = BASE_DIR / "data" / "usage.db"
MODELS_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class MLReport:
    """Structured ML evaluation results for the dashboard and docs."""
    # IsolationForest
    iso_anomaly_count: int   = 0
    iso_anomaly_rate:  float = 0.0
    iso_vs_zscore_gap: int   = 0     # extra anomalies ISO caught over z-score

    # RandomForest
    rf_roc_auc:        float = 0.0
    rf_f1_macro:       float = 0.0
    rf_cv_mean:        float = 0.0
    rf_cv_std:         float = 0.0
    rf_confusion:      list  = field(default_factory=list)
    rf_classification_report: str = ""
    feature_importance: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"IsolationForest: {self.iso_anomaly_count} anomalies "
            f"({self.iso_anomaly_rate:.1f}%) — {self.iso_vs_zscore_gap} more than z-score\n"
            f"RandomForest: ROC-AUC={self.rf_roc_auc:.3f} | F1={self.rf_f1_macro:.3f} "
            f"| CV={self.rf_cv_mean:.3f}±{self.rf_cv_std:.3f}\n"
            f"Feature importance: {self.feature_importance}"
        )


def load_features(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM usage_metrics", conn)
    conn.close()
    return df


def run_isolation_forest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Multivariate anomaly detection.
    Features: tokens, cost, avg_tokens_per_msg — capturing joint behaviour.
    Contamination=0.05: we expect ~5% anomalous conversations.
    """
    features = df[["tokens_5min", "total_cost_usd", "avg_tokens_per_msg"]].copy()
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    preds = iso.fit_predict(features)
    df = df.copy()
    df["iso_anomaly_flag"] = (preds == -1).astype(int)

    # Compare against z-score method
    z_count  = int(df["z_anomaly_flag"].sum()) if "z_anomaly_flag" in df.columns else 0
    iso_count = int(df["iso_anomaly_flag"].sum())

    print(f"[ML] IsolationForest: {iso_count} anomalies | Z-score: {z_count} anomalies")
    print(f"[ML] IsolationForest caught {iso_count - z_count} anomalies z-score missed")

    joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
    return df, {"iso_count": iso_count, "z_count": z_count}


def run_random_forest(df: pd.DataFrame) -> tuple[RandomForestClassifier, MLReport]:
    """
    Supervised classifier: predict high-cost conversations.
    Uses only pre-session features (no leakage of cost-derived cols).
    """
    report = MLReport()

    FEATURES = ["msg_count_5min", "model_tier_encoded", "user_tier_encoded"]
    TARGET   = "high_cost_flag"

    X = df[FEATURES]
    y = df[TARGET]

    # Stratified split preserves class ratio in imbalanced dataset (5% positive)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        max_depth=8,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report.rf_roc_auc   = round(float(roc_auc_score(y_test, y_prob)), 4)
    report.rf_f1_macro  = round(float(f1_score(y_test, y_pred, average="macro")), 4)
    report.rf_confusion = confusion_matrix(y_test, y_pred).tolist()
    report.rf_classification_report = classification_report(y_test, y_pred)

    # 5-fold cross-validation for stability check
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    report.rf_cv_mean = round(float(cv_scores.mean()), 4)
    report.rf_cv_std  = round(float(cv_scores.std()), 4)

    # Feature importance
    report.feature_importance = dict(zip(FEATURES, [
        round(float(v), 4) for v in model.feature_importances_
    ]))

    print(f"[ML] RandomForest ROC-AUC: {report.rf_roc_auc} | CV: {report.rf_cv_mean}±{report.rf_cv_std}")
    print(f"[ML] Feature importance: {report.feature_importance}")

    joblib.dump(model, MODELS_DIR / "high_cost_classifier.pkl")
    return model, report


def save_predictions(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    df.to_sql("usage_metrics", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"[ML] Predictions saved to usage_metrics")


def run_ml_pipeline() -> MLReport:
    df = load_features()
    df, iso_stats = run_isolation_forest(df)
    _, rf_report  = run_random_forest(df)

    rf_report.iso_anomaly_count = iso_stats["iso_count"]
    rf_report.iso_anomaly_rate  = round(iso_stats["iso_count"] / len(df) * 100, 2)
    rf_report.iso_vs_zscore_gap = iso_stats["iso_count"] - iso_stats["z_count"]

    save_predictions(df)
    print("[ML] Layer 3 complete.")
    return rf_report


if __name__ == "__main__":
    report = run_ml_pipeline()
    print(report.summary())
