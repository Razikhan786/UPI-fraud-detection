"""
Step 6: Score the whole transactions table at once and write
predictions + flags back into the database (so the dashboard has data).
Run this once after train_model.py.
"""

import joblib
import pandas as pd

from database import get_connection, update_predictions
from feature_engineering import build_features

if __name__ == "__main__":
    conn = get_connection()
    tx_df = pd.read_sql("SELECT * FROM transactions", conn)

    df, feature_cols = build_features(tx_df)

    rf = joblib.load("rf_model.pkl")
    iso = joblib.load("iso_model.pkl")
    scaler = joblib.load("scaler.pkl")

    X_scaled = scaler.transform(df[feature_cols])
    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    iso_flags = iso.predict(X_scaled)
    iso_scores = (iso_flags == -1).astype(float)

    df["risk_score"] = (0.7 * rf_proba + 0.3 * iso_scores).round(4)
    df["predicted_fraud"] = (df["risk_score"] >= 0.5).astype(int)

    update_predictions(conn, df[["transaction_id", "predicted_fraud", "risk_score"]])
    conn.close()

    print(f"Scored {len(df)} transactions.")
    print(f"Flagged as fraud: {df['predicted_fraud'].sum()}")
