"""
Step 3: Feature engineering.
Turns raw transaction rows into numeric features the ML models can use.
"""

import pandas as pd


def build_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    df = tx_df.copy()

    # Per-sender stats (behavioral baseline)
    sender_stats = df.groupby("sender_upi")["amount"].agg(["mean", "std"]).reset_index()
    sender_stats.columns = ["sender_upi", "sender_avg_amount", "sender_std_amount"]
    sender_stats["sender_std_amount"] = sender_stats["sender_std_amount"].fillna(0)
    df = df.merge(sender_stats, on="sender_upi", how="left")

    # How far this transaction's amount deviates from the sender's usual amount
    df["amount_deviation"] = (
        (df["amount"] - df["sender_avg_amount"]) / (df["sender_std_amount"] + 1)
    )

    # Odd hour flag (midnight - 5am is unusual for typical daily transactions)
    df["is_odd_hour"] = df["hour_of_day"].apply(lambda h: 1 if (h < 5 or h == 23) else 0)

    # High value flag
    df["is_high_amount"] = (df["amount"] > df["amount"].quantile(0.90)).astype(int)

    feature_cols = [
        "amount",
        "amount_deviation",
        "is_new_device",
        "is_odd_hour",
        "is_high_amount",
        "hour_of_day",
    ]
    return df, feature_cols


if __name__ == "__main__":
    tx_df = pd.read_csv("transactions.csv")
    df, cols = build_features(tx_df)
    print(df[cols + ["is_fraud"]].head())
    print("\nFeature columns:", cols)
