# UPI Fraud Detection System

A mini end-to-end project: **Python + SQL + AI/ML**, built and tested to actually run.

Detects likely-fraudulent UPI transactions using a Random Forest classifier
(trained with SMOTE to handle class imbalance) combined with an Isolation
Forest anomaly detector, stores everything in SQLite, and exposes it through
a FastAPI backend + Streamlit dashboard.

> **Note on data:** Real UPI transaction data is private/regulated, so this
> project uses realistic **synthetic data** (via Faker) with deliberately
> engineered fraud patterns (odd hours, unusually high amounts, new/unknown
> devices). This is the standard, expected approach for a project like this
> — mention it explicitly if asked in an interview.

## Project Structure
```
upi_fraud_detection/
├── schema.sql              # SQL schema (users, transactions, flagged_transactions)
├── generate_data.py         # Step 1: create synthetic dataset
├── database.py               # Step 2: SQLite DB setup + load data
├── feature_engineering.py    # Step 3: turn raw transactions into ML features
├── train_model.py            # Step 4: train RandomForest + IsolationForest
├── batch_score.py            # Step 5: score all transactions, save to DB
├── api.py                    # Step 6: FastAPI backend (real-time scoring)
├── dashboard.py               # Step 7: Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to Run (in order)

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (creates users.csv, transactions.csv)
python generate_data.py

# 2. Create SQLite DB and load the data
python database.py

# 3. Train the ML models (saves .pkl files)
python train_model.py

# 4. Score all transactions and store predictions in the DB
python batch_score.py

# 5. Start the API (in one terminal)
uvicorn api:app --reload
# → visit http://127.0.0.1:8000/docs for interactive API docs

# 6. Start the dashboard (in another terminal)
streamlit run dashboard.py
```

## How It Works

1. **Data**: 200 users, 5,000 transactions, ~5% fraud rate, with fraud
   transactions skewed toward odd hours, high amounts, and new devices.
2. **Features**: amount, deviation from the sender's usual amount, odd-hour
   flag, high-amount flag, new-device flag, hour of day.
3. **Models**:
   - Random Forest (supervised) — trained on SMOTE-balanced data since
     fraud is rare and a plain classifier would just predict "not fraud"
     for everything.
   - Isolation Forest (unsupervised) — catches unusual patterns even if
     they don't match the fraud examples seen during training.
   - Final `risk_score = 0.7 × RF_probability + 0.3 × anomaly_flag`.
4. **API**: `/predict` scores a transaction in real time; `/flagged` and
   `/transactions` return stored results.
5. **Dashboard**: shows flagged transactions, risk score distribution, and
   lets you manually test a transaction.

## Talking Points for Interviews / Resume

- Why **SMOTE**: fraud is a rare-event/imbalanced classification problem —
  without rebalancing, the model would just predict "not fraud" for
  everything and still get ~95% accuracy while catching 0 fraud.
- Why **precision/recall over plain accuracy**: with imbalanced classes,
  accuracy is misleading; recall matters because missing real fraud is
  costlier than a false alarm.
- Why combine a **supervised + unsupervised** model: the supervised model
  catches known fraud patterns; the anomaly detector can catch new/unseen
  fraud patterns that don't look like the training examples.
- Synthetic data limitation: honestly acknowledge that a production system
  would need real transaction data, more behavioral features (transaction
  velocity, graph-based relationships between accounts), and periodic
  retraining.

## Possible Extensions (if you want to go further)
- Add a transaction-velocity feature (transactions per user per hour).
- Add graph-based features (money mule ring detection).
- Add authentication to the API.
- Deploy API + dashboard on Render/Railway.
