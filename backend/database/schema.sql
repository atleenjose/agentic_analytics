CREATE TABLE IF NOT EXISTS chatbot_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    convo_id TEXT NOT NULL UNIQUE,

    msg_count_5min INTEGER NOT NULL,
    tokens_5min INTEGER NOT NULL,
    avg_tokens_per_msg REAL NOT NULL,

    model_tier INTEGER NOT NULL,
    user_tier INTEGER NOT NULL,

    total_cost_usd REAL NOT NULL,

    msg_count_5min_norm REAL,
    tokens_5min_norm REAL,
    avg_tokens_per_msg_norm REAL,
    model_tier_norm REAL,
    user_tier_norm REAL,

    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_convo_id ON chatbot_usage(convo_id);
CREATE INDEX IF NOT EXISTS idx_cost ON chatbot_usage(total_cost_usd);
CREATE INDEX IF NOT EXISTS idx_user_tier ON chatbot_usage(user_tier);