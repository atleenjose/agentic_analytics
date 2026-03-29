"""
run_all.py  —  Run the complete pipeline in one command.

Usage:
  python run_all.py          # run all layers
  python run_all.py --etl    # layer 1 only
  python run_all.py --features  # layers 1+2
  python run_all.py --ml     # layers 1+2+3
"""
import sys, argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def run_etl():
    print("\n" + "="*50)
    print("LAYER 1 — ETL: Ingestion & Validation")
    print("="*50)
    from backend.etl.pipeline import run_pipeline
    df, report = run_pipeline()
    print(report.summary())
    return df


def run_features():
    print("\n" + "="*50)
    print("LAYER 2 — Feature Engineering & Statistics")
    print("="*50)
    from backend.features.engineering import run_feature_engineering
    df, report = run_feature_engineering()
    print(report.summary())
    return df


def run_ml():
    print("\n" + "="*50)
    print("LAYER 3 — ML: Anomaly Detection & Classification")
    print("="*50)
    from backend.ml.models import run_ml_pipeline
    report = run_ml_pipeline()
    print(report.summary())
    return report


def run_export():
    print("\n" + "="*50)
    print("EXPORT — Power BI CSV files")
    print("="*50)
    from export.to_powerbi import export
    export()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etl",      action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--ml",       action="store_true")
    parser.add_argument("--export",   action="store_true")
    args = parser.parse_args()

    run_any = args.etl or args.features or args.ml or args.export

    if not run_any or args.etl or args.features or args.ml:
        run_etl()
    if not run_any or args.features or args.ml:
        run_features()
    if not run_any or args.ml:
        run_ml()
    if not run_any or args.export:
        run_export()

    print("\n" + "="*50)
    print("All layers complete. Start dashboard:")
    print("  streamlit run frontend/app.py")
    print("="*50)
