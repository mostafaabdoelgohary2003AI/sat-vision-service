CREATE TABLE IF NOT EXISTS prediction_log (
    id SERIAL PRIMARY KEY,
    ts BIGINT NOT NULL,
    filename TEXT,
    top_class TEXT,
    probability DOUBLE PRECISION
);
