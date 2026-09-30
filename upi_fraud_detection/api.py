"""
Step 5: FastAPI backend.
Run with: uvicorn api:app --reload
Docs available at http://127.0.0.1:8000/docs
"""

from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from database import get_connection, update_predictions
from feature_engineering import build_features
from train_model import score_transaction

app = FastAPI(title="UPI Fraud Detection API")


class Transaction(BaseModel):
    transaction_id: str
    sender_upi: str
    receiver_upi: str
    amount: float
    hour_of_day: int
    is_new_device: int  # 1 if unrecognized device, else 0


@app.get("/")
def root():
    return {"message": "UPI Fraud Detection API is running"}


@app.post("/predict")
def predict(tx: Transaction):
    """Score a single incoming transaction in real time."""
    conn = get_connection()
    history = pd.read_sql("SELECT * FROM transactions WHERE sender_upi = ?", conn, params=(tx.sender_upi,))

    if len(history) > 0:
        sender_avg = history["amount"].mean()
        sender_std = history["amount"].std() or 0
    else:
        sender_avg, sender_std = tx.amount, 0

    amount_deviation = (tx.amount - sender_avg) / (sender_std + 1)
    is_odd_hour = 1 if (tx.hour_of_day < 5 or tx.hour_of_day == 23) else 0
    is_high_amount = 1 if tx.amount > 5000 else 0

    features = {
        "amount": tx.amount,
        "amount_deviation": amount_deviation,
        "is_new_device": tx.is_new_device,
        "is_odd_hour": is_odd_hour,
        "is_high_amount": is_high_amount,
        "hour_of_day": tx.hour_of_day,
    }

    result = score_transaction(features)
    conn.close()

    return {"transaction_id": tx.transaction_id, **result}


@app.get("/flagged")
def get_flagged(limit: int = 20):
    """Return recently flagged (high-risk) transactions."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM transactions WHERE predicted_fraud = 1 ORDER BY risk_score DESC LIMIT ?",
        conn, params=(limit,),
    )
    conn.close()
    return df.to_dict(orient="records")


@app.get("/transactions")
def get_transactions(limit: int = 50):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df.to_dict(orient="records")
