# This is a Streamlit app for predicting machine maintenance needs based on input parameters.
# It uses a pre-trained Keras model and a scaler for feature normalization.
# The app allows users to input various machine parameters and outputs the probability of machine failure.
# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load Model & Scaler
model = load_model("maintenance_model.keras")
scaler = joblib.load("scaler.pkl")

# Streamlit App Configuration
# Set page configuration for Streamlit app
st.set_page_config(page_title="Machine Maintenance Monitor", layout="centered")

st.title("Real-time Machine Maintenance Prediction")

st.markdown("Enter machine parameters below:")

# Input Form for parameters
with st.form("input_form"):
    air_temp = st.number_input("Air Temperature [K]", value=None, placeholder="e.g. 300.0")
    process_temp = st.number_input("Process Temperature [K]", value=None, placeholder="e.g. 310.0")
    rotation_speed = st.number_input("Rotational Speed [rpm]", value=None, placeholder="e.g. 1500")
    torque = st.number_input("Torque [Nm]", value=None, placeholder="e.g. 40.0")
    tool_wear = st.number_input("Tool Wear [min]", value=None, placeholder="e.g. 5")
    product_type = st.selectbox("Product Type", ['L', 'M', 'H'])

    submitted = st.form_submit_button("Predict Maintenance")

# Prediction Logic
if submitted:
    # Encode product type manually
    type_encoded = {'L': 0, 'M': 1, 'H': 2}[product_type]

    # Feature engineering: calculate power
    power = torque * (rotation_speed * 2 * np.pi / 60)

    # Build feature array
    features = np.array([[air_temp, process_temp, rotation_speed, torque, tool_wear, type_encoded, power]])
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)[0][0]
    maintenance_needed = prediction > 0.5

    # Output
    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Probability of Machine Failure**: {prediction:.2f}")
    if maintenance_needed:
        st.error("⚠️ Maintenance Required!")
    else:
        st.success("✅ Machine is Operating Normally.")
