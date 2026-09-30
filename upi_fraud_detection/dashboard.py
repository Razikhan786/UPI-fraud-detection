"""
Step 7: Streamlit dashboard.
Run with: streamlit run dashboard.py
"""

import pandas as pd
import streamlit as st

from database import get_connection
from feature_engineering import build_features
from train_model import score_transaction

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("💳 UPI Fraud Detection Dashboard")

conn = get_connection()
tx_df = pd.read_sql("SELECT * FROM transactions", conn)
tx_df["predicted_fraud"] = pd.to_numeric(tx_df["predicted_fraud"], errors="coerce")
tx_df["risk_score"] = pd.to_numeric(tx_df["risk_score"], errors="coerce")

# --- Top metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(tx_df))
col2.metric("Flagged as Fraud", int(tx_df["predicted_fraud"].fillna(0).sum()))
col3.metric("Fraud Rate (model)", f"{tx_df['predicted_fraud'].fillna(0).mean()*100:.2f}%")

st.divider()

# --- Risk score distribution ---
st.subheader("Risk Score Distribution")
st.bar_chart(tx_df["risk_score"].dropna())

# --- Flagged transactions table ---
st.subheader("🚩 Flagged Transactions")
flagged = tx_df[tx_df["predicted_fraud"] == 1].sort_values("risk_score", ascending=False)
st.dataframe(flagged[["transaction_id", "sender_upi", "receiver_upi", "amount",
                       "timestamp", "location", "risk_score"]], use_container_width=True)

st.divider()

# --- Manual transaction tester ---
st.subheader("🧪 Test a New Transaction")
with st.form("test_transaction"):
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (₹)", min_value=1.0, value=1500.0)
    hour = c1.slider("Hour of day", 0, 23, 14)
    is_new_device = c2.selectbox("New/unknown device?", [0, 1])
    submitted = st.form_submit_button("Check Risk")

    if submitted:
        is_odd_hour = 1 if (hour < 5 or hour == 23) else 0
        is_high_amount = 1 if amount > 5000 else 0
        features = {
            "amount": amount,
            "amount_deviation": 0,  # no history for a manual test
            "is_new_device": is_new_device,
            "is_odd_hour": is_odd_hour,
            "is_high_amount": is_high_amount,
            "hour_of_day": hour,
        }
        result = score_transaction(features)
        if result["predicted_fraud"] == 1:
            st.error(f"⚠️ HIGH RISK — risk score: {result['risk_score']}")
        else:
            st.success(f"✅ Looks normal — risk score: {result['risk_score']}")

conn.close()
