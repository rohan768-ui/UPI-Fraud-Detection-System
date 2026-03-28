import streamlit as st
import pandas as pd

st.set_page_config(page_title="UPI Fraud Detector")

st.title("🚨 UPI Fraud Detection System")

st.write("Enter transaction details:")

# User Inputs
amount = st.number_input("Transaction Amount (₹)", min_value=0)
hour = st.slider("Transaction Hour (0–23)", 0, 23)

# Feature logic
high_amount = 1 if amount > 50000 else 0
night = 1 if hour < 5 else 0

# Risk score
risk_score = high_amount + night

# Button
if st.button("Check Transaction"):

    if risk_score >= 2:
        st.error("⚠️ High Risk Transaction (Possible Fraud)")
    elif risk_score == 1:
        st.warning("⚠️ Medium Risk Transaction")
    else:
        st.success("✅ Safe Transaction")

    # Show details
    st.write("### Details:")
    st.write(f"Amount: ₹{amount}")
    st.write(f"Hour: {hour}")
    st.write(f"High Amount: {high_amount}")
    st.write(f"Night Transaction: {night}")