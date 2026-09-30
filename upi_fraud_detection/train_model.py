"""
Step 4: Train fraud detection models.
- Supervised: Random Forest (with SMOTE to handle class imbalance)
- Unsupervised: Isolation Forest (catches anomalies even without labels)
Final risk_score = blend of both models' outputs.
"""

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_features


def train():
    tx_df = pd.read_csv("transactions.csv")
    df, feature_cols = build_features(tx_df)

    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    # Supervised model
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train_bal, y_train_bal)

    y_pred = rf.predict(X_test_scaled)
    print("=== Random Forest Results ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Unsupervised anomaly detector (trained only on normal-looking data patterns)
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train_scaled)

    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(iso, "iso_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    print("\nSaved: rf_model.pkl, iso_model.pkl, scaler.pkl, feature_cols.pkl")


def score_transaction(feature_dict: dict) -> dict:
    """Load saved models and score a single transaction (used by the API)."""
    rf = joblib.load("rf_model.pkl")
    iso = joblib.load("iso_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")

    X = pd.DataFrame([feature_dict])[feature_cols]
    X_scaled = scaler.transform(X)

    rf_proba = rf.predict_proba(X_scaled)[0][1]              # P(fraud) from RF
    iso_flag = iso.predict(X_scaled)[0]                       # -1 = anomaly, 1 = normal
    iso_score = 1.0 if iso_flag == -1 else 0.0

    # Blend: weight supervised model higher, anomaly detector as a booster
    risk_score = round(0.7 * rf_proba + 0.3 * iso_score, 4)
    predicted_fraud = 1 if risk_score >= 0.5 else 0

    return {"risk_score": risk_score, "predicted_fraud": predicted_fraud}


if __name__ == "__main__":
    train()
