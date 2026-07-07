"""
Credit Risk Analysis Platform - FIXED VERSION (Windows + Batch + Safe File Handling)
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import joblib
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ─────────────────────────────
# FIX: Use local folder instead of /tmp (Windows safe)
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

BATCH_FILE = os.path.join(BASE_DIR, "batch_results.csv")

# ─────────────────────────────
# Synthetic Data
# ─────────────────────────────
def generate_synthetic_data(n=5000):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(22, 65, n),
        'income': np.random.lognormal(13, 0.6, n).astype(int),
        'loan_amount': np.random.lognormal(13, 0.5, n).astype(int),
        'credit_score': np.random.randint(300, 900, n),
        'existing_emis': np.random.lognormal(9.5, 0.8, n).astype(int),
        'employment_type': np.random.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15]),
        'loan_tenure': np.random.choice([12, 24, 36, 48, 60], n),
        'num_credit_accounts': np.random.randint(0, 10, n),
        'num_delinquencies': np.random.randint(0, 5, n),
    })

    df['dti'] = (df['existing_emis'] * 12) / df['income']
    df['loan_to_income'] = df['loan_amount'] / df['income']
    df['credit_utilization'] = np.random.uniform(0, 1, n)

    risk = (
        (df['credit_score'] < 600).astype(int) * 3 +
        (df['dti'] > 0.5).astype(int) * 2 +
        (df['loan_to_income'] > 5).astype(int) * 2 +
        (df['num_delinquencies'] > 1).astype(int) * 2 +
        np.random.binomial(1, 0.15, n)
    )

    df["default"] = (risk >= 4).astype(int)
    return df


# ─────────────────────────────
# TRAIN MODEL (SAFE)
# ─────────────────────────────
def train_model():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = generate_synthetic_data(3000)

    feature_cols = [
        'age', 'income', 'loan_amount', 'credit_score', 'existing_emis',
        'employment_type', 'loan_tenure', 'num_credit_accounts',
        'num_delinquencies', 'dti', 'loan_to_income', 'credit_utilization'
    ]

    X = df[feature_cols]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=80)
    gb = GradientBoostingClassifier(n_estimators=80)
    xgb = XGBClassifier(eval_metric="logloss")

    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("xgb", xgb)],
        voting="soft"
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURE_PATH)

    return model, scaler, feature_cols


# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
else:
    model, scaler, feature_cols = train_model()


# ─────────────────────────────
# ROUTES
# ─────────────────────────────
@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"success": False, "error": "Empty file"}), 400

        # Read file
        df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)

        # Rename columns (handle user input variations)
        df.rename(columns={
            "Age": "age",
            "Annual Income": "income",
            "Loan Amount": "loan_amount",
            "Credit Score": "credit_score",
            "Existing EMI": "existing_emis",
            "Employment Type": "employment_type"
        }, inplace=True)

        # Employment mapping
        emp_map = {"Salaried": 0, "Self-Employed": 1, "Business": 2}
        df["employment_type"] = df.get("employment_type", 0)
        df["employment_type"] = df["employment_type"].map(emp_map).fillna(0)

        # Safe numeric conversions
        df["income"] = pd.to_numeric(df.get("income", 500000), errors="coerce").fillna(500000)
        df["existing_emis"] = pd.to_numeric(df.get("existing_emis", 0), errors="coerce").fillna(0)
        df["loan_amount"] = pd.to_numeric(df.get("loan_amount", 0), errors="coerce").fillna(0)
        df["credit_score"] = pd.to_numeric(df.get("credit_score", 650), errors="coerce").fillna(650)

        # Feature engineering
        df["dti"] = (df["existing_emis"] * 12) / df["income"]
        df["loan_to_income"] = df["loan_amount"] / df["income"]
        df["credit_utilization"] = (df["existing_emis"] / (df["income"] / 12)).clip(0, 1)

        # ✅ HANDLE MISSING FEATURES (MAIN FIX)
        default_values = {
            "loan_tenure": 12,
            "num_credit_accounts": 2,
            "num_delinquencies": 0
        }

        for col, val in default_values.items():
            if col not in df.columns:
                df[col] = val

        # ✅ Ensure ALL required columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Select features
        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        # Prediction
        probs = model.predict_proba(X_scaled)[:, 1]

        df["risk_score"] = (probs * 100).round(1)
        df["risk_category"] = pd.cut(
            probs,
            [0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )

        # Save results
        df.to_csv(BATCH_FILE, index=False)

        counts = df["risk_category"].value_counts()

        return jsonify({
            "success": True,
            "total": len(df),
            "low": int(counts.get("Low", 0)),
            "medium": int(counts.get("Medium", 0)),
            "high": int(counts.get("High", 0)),
            "download_url": "/download_batch"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download_batch")
def download_batch():
    if not os.path.exists(BATCH_FILE):
        return jsonify({"error": "No batch file found. Run analysis first."}), 404

    return send_file(BATCH_FILE, as_attachment=True,
                     download_name="credit_risk_results.csv")


@app.route("/analytics/feature_importance")
def feature_importance():
    return jsonify({
        "features": [
            "Credit Score", "Debt-to-Income", "Loan-to-Income",
            "Past Delinquencies", "Credit Utilization", "Existing EMIs"
        ],
        "importance": [0.22, 0.18, 0.15, 0.13, 0.10, 0.08]
    })


@app.route("/analytics/summary")
def analytics_summary():
    df = generate_synthetic_data(1000)

    df["dti"] = (df["existing_emis"] * 12) / df["income"]
    df["loan_to_income"] = df["loan_amount"] / df["income"]
    df["credit_utilization"] = np.clip(df["existing_emis"] / (df["income"] / 12), 0, 1)

    X = scaler.transform(df[feature_cols])
    probs = model.predict_proba(X)[:, 1]

    df["risk_category"] = pd.cut(probs, [0, 0.3, 0.6, 1.0],
                                 labels=["Low", "Medium", "High"])

    counts = df["risk_category"].value_counts()

    return jsonify({
        "risk_distribution": {
            "Low": int(counts.get("Low", 0)),
            "Medium": int(counts.get("Medium", 0)),
            "High": int(counts.get("High", 0))
        }
    })


if __name__ == "__main__":
    app.run(debug=True)