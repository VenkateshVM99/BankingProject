from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Banking ML APIs")

# ======================
# Load trained models
# ======================
loan_model = joblib.load("models/best_loan_model.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")
kmeans_scaler = joblib.load("models/kmeans_scaler.pkl")

# ======================
# Banking Product Logic
# ======================
CLUSTER_PRODUCT_MAP = {
    0: ["Basic Savings Account", "Recurring Deposit"],
    1: ["Credit Card", "Personal Loan"],
    2: ["Wealth Management", "Fixed Deposit", "Premium Credit Card"]
}

# ======================
# Request Schemas
# ======================
class ClusterRequest(BaseModel):
    monetary: float
    frequency: int
    avg_txn_amount: float
    tenure: float
    income: float
    credit_score: int



@app.post("/predict-loan-default")
def predict_default(data: dict):
    df = pd.DataFrame([data])
    prob = loan_model.predict_proba(df)[0][1]
    prediction = int(prob > 0.5)

    return {
        "default_probability": prob,
        "prediction": prediction,
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    }
# OPTIONAL MODELS (required if endpoints exist)
kmeans_model = joblib.load("models/kmeans_customers.pkl")



@app.post("/segment-customer")
def segment_customer(data: dict):
    df = pd.DataFrame([data])
    cluster = kmeans_model.predict(df)[0]
    return {"cluster": int(cluster)}


# ======================
# Product Recommendation
# ======================
@app.post("/recommend-products")
def recommend_products(req: ClusterRequest):
    df = pd.DataFrame([req.dict()])
    X_scaled = kmeans_scaler.transform(df)
    cluster = int(kmeans_model.predict(X_scaled)[0])
    return {
        "cluster": cluster,
        "recommended_products": CLUSTER_PRODUCT_MAP[cluster]
    }
