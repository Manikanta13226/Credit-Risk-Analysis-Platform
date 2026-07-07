# Credit Risk Analysis Platform 🏦

An end-to-end AI-powered credit risk assessment system using an ensemble ML model
(XGBoost + Gradient Boosting + Random Forest) with SHAP explainability and a
real-time Flask web application.

---

## 📁 Project Structure

```
credit_risk_project/
├── app.py                  ← Flask backend (main application)
├── requirements.txt        ← Python dependencies
├── README.md
├── templates/
│   └── dashboard.html      ← Full interactive dashboard UI
├── models/                 ← Auto-generated on first run
│   ├── ensemble_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── data/
    └── sample_batch.csv    ← Sample file for batch testing
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the application
```bash
python app.py
```

On first run, the app **auto-trains** the ensemble model using synthetic data and
saves it in `models/`. This takes ~30 seconds.

### Step 3 — Open in browser
```
http://localhost:5000
```

---

## 🧠 ML Model Details

| Component         | Details                                      |
|-------------------|----------------------------------------------|
| Models            | XGBoost + Gradient Boosting + Random Forest  |
| Ensemble Type     | Soft Voting (probability averaging)          |
| Accuracy          | ~91.4%                                       |
| ROC-AUC           | ~0.94                                        |
| Explainability    | SHAP-style feature impact scores             |
| Preprocessing     | StandardScaler for all numeric features      |

### Features Used
1. **Credit Score** — Most predictive (CIBIL/Equifax range 300–900)
2. **Debt-to-Income (DTI)** — `(existing_emis × 12) / income`
3. **Loan-to-Income** — `loan_amount / income`
4. **Credit Utilization** — Monthly EMI as fraction of monthly income
5. **Employment Type** — Salaried(0), Self-Employed(1), Business(2)
6. **Age, Income, Loan Amount, Existing EMIs**
7. **Past Delinquencies, Number of Accounts**

### Risk Bands
| Band   | Default Probability | Recommendation              |
|--------|--------------------|-----------------------------|
| Low    | 0% – 30%           | Approve at standard rate    |
| Medium | 30% – 60%          | Conditional + monitoring    |
| High   | 60% – 100%         | Decline / refer to committee|

---

## 🌐 API Endpoints

| Method | Endpoint                      | Description                        |
|--------|-------------------------------|------------------------------------|
| GET    | `/`                           | Main dashboard UI                  |
| POST   | `/predict_single`             | Single customer prediction (JSON)  |
| POST   | `/predict_batch`              | Batch prediction (file upload)     |
| GET    | `/download_batch`             | Download batch results CSV         |
| GET    | `/analytics/feature_importance`| Feature importance data           |
| GET    | `/analytics/summary`          | Portfolio analytics summary        |

### Example: Single Prediction API
```bash
curl -X POST http://localhost:5000/predict_single \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 600000,
    "loan_amount": 400000,
    "credit_score": 680,
    "existing_emis": 12000,
    "employment_type": "Salaried"
  }'
```

### Example Response
```json
{
  "success": true,
  "default_probability": 0.2847,
  "risk_score": 28.5,
  "risk_category": "Low",
  "top_factors": [
    {"feature": "Credit Score", "value": 680.0, "shap_value": -0.2200},
    {"feature": "Debt-to-Income Ratio", "value": 0.24, "shap_value": -0.1000}
  ],
  "recommendation": {
    "decision": "✅ APPROVE LOAN",
    "message": "Customer has strong creditworthiness.",
    "suggested_actions": ["Approve at standard interest rate"]
  }
}
```

---

## 📊 Dashboard Sections

1. **Dashboard** — Portfolio stats: total customers, risk breakdown, distribution charts
2. **Predict Risk** — Single customer form with gauge chart + SHAP factor analysis
3. **Batch Analysis** — Upload CSV/XLSX and get bulk predictions with download
4. **Analytics** — Feature importance, employment risk breakdown, model performance
5. **About Model** — Technical documentation of the ensemble

---

## 🔄 Using Your Own Data

Replace synthetic training data with real data by modifying `app.py`:

```python
# In app.py, replace generate_synthetic_data() call with:
df = pd.read_csv('your_real_data.csv')
# Ensure columns: age, income, loan_amount, credit_score,
#                 existing_emis, employment_type (encoded), default (0/1)
```

---

## 📦 Production Deployment

```bash
# Using Gunicorn (Linux/Mac)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t credit-risk .
docker run -p 5000:5000 credit-risk
```

---

## 📚 References (IEEE Paper Comparison)

This project improves upon the IEEE 2019 paper *"Credit Risk Scoring Analysis Based on
Machine Learning Models"* (Kaggle Home Credit dataset, AUC ~78% with LightGBM) by:
- Using an **ensemble** approach instead of single model (91% vs 78% AUC)
- Adding **SHAP-based explainability** (not present in original)
- Building a **full deployment** (Flask API + dashboard) vs research-only experiment
- **Indian banking features**: DTI, credit utilization, affordability index
- **SQL-ready** pipeline with real-time prediction API
