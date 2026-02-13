import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from backend.analytics.analytics import compute_kpis, analyze_text_logs

# Test KPIs
kpis = compute_kpis()
print("KPIs:\n", kpis)

# Test NLP
from backend.analytics.analytics import fetch_usage_data, analyze_text_logs

# Fetch updated data from DB
df = fetch_usage_data()

# Now df should have 'conversation_text' column
nlp_metrics = analyze_text_logs(df)
print("\nNLP Metrics:\n", nlp_metrics)
