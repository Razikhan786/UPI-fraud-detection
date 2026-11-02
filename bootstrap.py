"""
Prepares everything the dashboard needs (data, database, trained models,
scored transactions) the first time the app starts. This is what lets the
app run on Streamlit Community Cloud, where you cannot run the setup
scripts by hand.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = ["upi_fraud.db", "rf_model.pkl", "iso_model.pkl", "scaler.pkl", "feature_cols.pkl"]

SETUP_STEPS = ["generate_data.py", "database.py", "train_model.py", "batch_score.py"]


def is_ready():
    return all(os.path.exists(os.path.join(BASE_DIR, f)) for f in REQUIRED_FILES)


def ensure_ready():
    """Run the 4 setup scripts in order if the database/models don't exist yet."""
    os.chdir(BASE_DIR)
    if is_ready():
        return
    for script in SETUP_STEPS:
        subprocess.run([sys.executable, script], check=True, cwd=BASE_DIR)
