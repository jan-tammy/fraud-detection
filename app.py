import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load artifacts ────────────────────────────────────────────────────────────
model = joblib.load("models/model.pkl")      
threshold = joblib.load("models/threshold.pkl")    

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")
st.title("Fraud Detection")
st.caption("Enter transaction details to assess fraud probability.")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        tx_type  = st.selectbox("Transaction Type",
                                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount   = st.number_input("Amount ($)", min_value=0.0, step=100.0)
        old_orig = st.number_input("Origin Balance (Before)", min_value=0.0, step=100.0)
        new_orig = st.number_input("Origin Balance (After)",  min_value=0.0, step=100.0)

    with col2:
        old_dest = st.number_input("Destination Balance (Before)", min_value=0.0, step=100.0)
        new_dest = st.number_input("Destination Balance (After)",  min_value=0.0, step=100.0)
        step     = st.number_input("Time Step (hour)", min_value=0, step=1)

    submitted = st.form_submit_button("Assess Transaction")

# ── Inference ─────────────────────────────────────────────────────────────────
if submitted:
    # Replicate your feature engineering
    type_map = {"TRANSFER": 0, "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "DEBIT": 4}

    hour = step % 24
    day  = (step // 24) % 7

    # Amount z-score — requires mean and std from training set
    amt_stats = joblib.load("models/amt_stats.pkl")
    amt_mean = amt_stats["mean"]
    amt_std  = amt_stats["std"]

    features = pd.DataFrame([{
    "amt_zscore":            (amount - amt_mean) / amt_std if amt_std != 0 else 0,
    "dest_bal_discrepancy":  (new_dest - old_dest) - amount,
    "zeroed_out":            int(new_orig == 0.0),
    "orig_empty":            int(old_orig == 0.0),
    "hour_sin":              np.sin(2 * np.pi * hour / 24),
    "hour_cos":              np.cos(2 * np.pi * hour / 24),
    "day_sin":               np.sin(2 * np.pi * day / 7),
    "day_cos":               np.cos(2 * np.pi * day / 7),
    "is_transfer":           int(tx_type in ["TRANSFER", "CASH_OUT"]),
}])

    prob = model.predict_proba(features)[0][1]
    is_fraud  = prob >= threshold

    # ── Result display ────────────────────────────────────────────────────────
    st.divider()

    if is_fraud:
        st.error(f"⚠️ **Fraudulent Transaction Detected**")
    else:
        st.success(f"✅ **Transaction Appears Legitimate**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Fraud Probability", f"{prob:.2%}")
    col_b.metric("Decision Threshold", f"{threshold:.2%}")
    col_c.metric("Decision", "BLOCK" if is_fraud else "ALLOW")

    # ── Probability gauge ─────────────────────────────────────────────────────
    st.progress(float(prob), text=f"Risk Score: {prob:.2%}")