"""Run:  streamlit run app/streamlit_app.py   (after `python -m src.train`)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Telecom Customer Churn Predictor")


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


bundle = load()
model, thr = bundle["model"], bundle["threshold"]

c1, c2 = st.columns(2)
with c1:
    call_failure = st.number_input("Call failures", 0, 60, 8)
    complains = st.selectbox("Has complained?", [0, 1], format_func=lambda v: "Yes" if v else "No")
    subscription_length = st.number_input("Subscription length (months)", 1, 60, 32)
    charge_amount = st.number_input("Charge amount (0-10 band)", 0, 10, 1)
    seconds_of_use = st.number_input("Seconds of use", 0, 20000, 4400)
    frequency_of_use = st.number_input("Frequency of use (calls)", 0, 300, 70)
with c2:
    frequency_of_sms = st.number_input("Frequency of SMS", 0, 600, 70)
    distinct_called_numbers = st.number_input("Distinct called numbers", 0, 100, 23)
    age_group = st.selectbox("Age group (1-5)", [1, 2, 3, 4, 5], index=2)
    tariff_plan = st.selectbox("Tariff plan", [1, 2])
    customer_value = st.number_input("Customer value", 0.0, 2500.0, 470.0)

row = pd.DataFrame([{
    "call_failure": call_failure, "complains": complains, "subscription_length": subscription_length,
    "charge_amount": charge_amount, "seconds_of_use": seconds_of_use, "frequency_of_use": frequency_of_use,
    "frequency_of_sms": frequency_of_sms, "distinct_called_numbers": distinct_called_numbers,
    "age_group": age_group, "tariff_plan": tariff_plan, "age": {1: 15, 2: 25, 3: 30, 4: 45, 5: 55}[age_group],
    "customer_value": customer_value,
}])[bundle["features"]]

if st.button("Predict", type="primary"):
    p = float(model.predict_proba(row)[:, 1][0])
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(min(p, 1.0))
    if p >= thr:
        st.error(f"High risk (above the {thr:.2f} decision threshold) - prioritise for retention outreach.")
    elif p >= thr / 2:
        st.warning("Medium risk - monitor.")
    else:
        st.success("Low risk.")
