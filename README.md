The ingestion layer extracts structured conversation usage data, applies validation and feature engineering including cost efficiency metrics, flags anomalies based on percentile thresholds, and stores processed records in a relational database to support downstream analytics and dashboard queries.
The ingestion layer includes schema validation and data integrity checks before transformation to prevent downstream corruption.
The pipeline integrates statistical and ML-based anomaly detection during transformation to flag abnormal usage patterns before persistence.
The system follows a layered architecture: ingestion to transformation with anomaly detection to persistence to API exposure for frontend consumption.


# Agentic Analytics — End-to-End AI Analytics System

A portfolio-grade project demonstrating agentic AI applied to real chatbot usage data.
Claude autonomously writes SQL, runs Python analysis, detects anomalies, and explains findings.

## What makes this "agentic"
The AI doesn't just answer from memory — it:
1. Receives a natural language question
2. Decides which tools to call (SQL query, Python analysis, plot generation)
3. Executes those tools against real data
4. Reads the results and iterates if needed
5. Returns a grounded answer with evidence

## Project structure
```
agentic_analytics/
├── data/
│   ├── raw/chatbot_data.csv       # Source data (740 conversations)
│   ├── usage.db                   # SQLite analytical database
│   └── processed/                 # Parquet exports for Power BI
├── backend/
│   ├── etl/                       # Extract → Transform → Load pipeline
│   ├── agent/                     # The agentic AI core (tool use loop)
│   ├── analytics/                 # Feature engineering, anomaly detection
│   ├── models/                    # RandomForest high-cost classifier
│   └── api/                       # FastAPI serving the agent
├── frontend/
│   └── app.py                     # Streamlit dashboard
├── export/
│   └── to_powerbi.py              # Export clean data for Power BI
└── .env.example                   # API key config
```

## Quick start
```bash
pip install -r requirements.txt

# Option A: Full agentic mode (needs Anthropic API key)
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
streamlit run frontend/app.py

# Option B: Mock mode (no API key needed, all tools still run)
MOCK_MODE=true streamlit run frontend/app.py
```

## Stack
- Python 3.11+
- FastAPI + Uvicorn
- Streamlit (dashboard)
- SQLite (analytical store)
- scikit-learn (RandomForest classifier)
- Plotly (interactive charts)
- Anthropic Claude API (agentic reasoning)
- Power BI (executive reporting layer)
