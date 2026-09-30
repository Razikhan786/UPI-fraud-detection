-- UPI Fraud Detection: Database Schema

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    upi_id TEXT UNIQUE,
    device_id TEXT,
    signup_date TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    sender_upi TEXT,
    receiver_upi TEXT,
    amount REAL,
    timestamp TEXT,
    location TEXT,
    device_id TEXT,
    is_new_device INTEGER,
    hour_of_day INTEGER,
    is_fraud INTEGER,          -- ground truth label (synthetic data)
    predicted_fraud INTEGER,   -- model prediction, filled after scoring
    risk_score REAL,           -- model probability / anomaly score
    FOREIGN KEY (sender_upi) REFERENCES users(upi_id)
);

CREATE TABLE IF NOT EXISTS flagged_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT,
    reason TEXT,
    risk_score REAL,
    flagged_at TEXT,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);
