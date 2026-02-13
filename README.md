The ingestion layer extracts structured conversation usage data, applies validation and feature engineering including cost efficiency metrics, flags anomalies based on percentile thresholds, and stores processed records in a relational database to support downstream analytics and dashboard queries.
The ingestion layer includes schema validation and data integrity checks before transformation to prevent downstream corruption.
The pipeline integrates statistical and ML-based anomaly detection during transformation to flag abnormal usage patterns before persistence.
The system follows a layered architecture: ingestion to transformation with anomaly detection to persistence to API exposure for frontend consumption.
