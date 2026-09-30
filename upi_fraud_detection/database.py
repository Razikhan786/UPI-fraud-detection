"""
Step 2: Database layer.
Creates the SQLite DB from schema.sql and loads the CSVs generated
by generate_data.py into the users / transactions tables.
"""

import sqlite3
import pandas as pd

DB_PATH = "upi_fraud.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    with open("schema.sql", "r") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def load_csv_to_db(conn):
    users_df = pd.read_csv("users.csv")
    tx_df = pd.read_csv("transactions.csv")

    # columns not yet computed at insert time
    tx_df["predicted_fraud"] = None
    tx_df["risk_score"] = None

    users_df.to_sql("users", conn, if_exists="replace", index=False)
    tx_df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.commit()
    print(f"Loaded {len(users_df)} users and {len(tx_df)} transactions into {DB_PATH}")


def get_connection():
    return sqlite3.connect(DB_PATH)


def update_predictions(conn, results_df):
    """results_df must have: transaction_id, predicted_fraud, risk_score"""
    cur = conn.cursor()
    for _, row in results_df.iterrows():
        cur.execute(
            "UPDATE transactions SET predicted_fraud = ?, risk_score = ? WHERE transaction_id = ?",
            (int(row["predicted_fraud"]), float(row["risk_score"]), row["transaction_id"]),
        )
        if row["predicted_fraud"] == 1:
            cur.execute(
                "INSERT INTO flagged_transactions (transaction_id, reason, risk_score, flagged_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (row["transaction_id"], "ML model flagged as high risk", float(row["risk_score"])),
            )
    conn.commit()


if __name__ == "__main__":
    conn = init_db()
    load_csv_to_db(conn)
    conn.close()
