"""
Step 1: Generate synthetic UPI transaction data.
Real UPI transaction data is private/regulated, so we simulate realistic
patterns: most transactions are normal, a small % are fraud-like
(odd hours, high amounts, new/unknown devices, unusual locations).
"""

import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
Faker.seed(42)

N_USERS = 200
N_TRANSACTIONS = 5000
FRAUD_RATE = 0.05  # 5% fraud

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Akola", "Chennai", "Kolkata"]


def generate_users(n=N_USERS):
    users = []
    for _ in range(n):
        uid = str(uuid.uuid4())[:8]
        users.append({
            "user_id": uid,
            "name": fake.name(),
            "upi_id": f"{fake.user_name()}@upi",
            "device_id": str(uuid.uuid4())[:12],
            "signup_date": fake.date_between(start_date="-2y", end_date="-30d").isoformat(),
        })
    return pd.DataFrame(users)


def generate_transactions(users_df, n=N_TRANSACTIONS):
    transactions = []
    n_fraud = int(n * FRAUD_RATE)
    n_normal = n - n_fraud

    # --- Normal transactions ---
    for _ in range(n_normal):
        sender = users_df.sample(1).iloc[0]
        receiver = users_df.sample(1).iloc[0]
        ts = fake.date_time_between(start_date="-90d", end_date="now")
        amount = round(random.uniform(50, 5000), 2)  # typical UPI amounts

        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "sender_upi": sender["upi_id"],
            "receiver_upi": receiver["upi_id"],
            "amount": amount,
            "timestamp": ts.isoformat(),
            "location": random.choice(CITIES),
            "device_id": sender["device_id"],   # same device as usual
            "is_new_device": 0,
            "hour_of_day": ts.hour if 6 <= ts.hour <= 23 else random.randint(7, 22),
            "is_fraud": 0,
        })

    # --- Fraud-like transactions ---
    for _ in range(n_fraud):
        sender = users_df.sample(1).iloc[0]
        receiver = users_df.sample(1).iloc[0]
        ts = fake.date_time_between(start_date="-90d", end_date="now")
        # fraud traits: odd hour, high amount, new device, sometimes different city
        odd_hour = random.choice([1, 2, 3, 4, 23, 0])
        ts = ts.replace(hour=odd_hour)
        amount = round(random.uniform(8000, 100000), 2)

        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "sender_upi": sender["upi_id"],
            "receiver_upi": receiver["upi_id"],
            "amount": amount,
            "timestamp": ts.isoformat(),
            "location": random.choice(CITIES),
            "device_id": str(uuid.uuid4())[:12],  # new/unknown device
            "is_new_device": 1,
            "hour_of_day": odd_hour,
            "is_fraud": 1,
        })

    df = pd.DataFrame(transactions).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    users_df = generate_users()
    tx_df = generate_transactions(users_df)

    users_df.to_csv("users.csv", index=False)
    tx_df.to_csv("transactions.csv", index=False)

    print(f"Generated {len(users_df)} users and {len(tx_df)} transactions.")
    print(f"Fraud transactions: {tx_df['is_fraud'].sum()} ({tx_df['is_fraud'].mean()*100:.1f}%)")
