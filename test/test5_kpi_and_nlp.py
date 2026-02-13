import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from backend.analytics.analytics import compute_kpis, analyze_text_logs, fetch_usage_data

df = fetch_usage_data()
print("KPIs:", compute_kpis())
print("NLP Metrics:", analyze_text_logs(df))
