from backend.analytics.analytics import fetch_usage_data

df = fetch_usage_data()
print(df.columns)
print(df['conversation_text'].head())
