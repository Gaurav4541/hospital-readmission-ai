import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("xgb_readmission_model.pkl")

# Page config
st.set_page_config(
    page_title="AI Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 AI Hospital Readmission Risk Prediction")

st.write(
    "Enter patient details below to predict the risk of 30-day hospital readmission."
)

# =========================
# INPUTS
# =========================

age = st.slider("Age", 1, 100, 50)

time_in_hospital = st.slider(
    "Time in Hospital",
    1,
    20,
    5
)

num_lab_procedures = st.slider(
    "Number of Lab Procedures",
    1,
    150,
    40
)

num_medications = st.slider(
    "Number of Medications",
    1,
    100,
    10
)

number_diagnoses = st.slider(
    "Number of Diagnoses",
    1,
    20,
    5
)

number_inpatient = st.slider(
    "Previous Inpatient Visits",
    0,
    20,
    0
)

number_emergency = st.slider(
    "Emergency Visits",
    0,
    20,
    0
)

number_outpatient = st.slider(
    "Outpatient Visits",
    0,
    20,
    0
)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🔍 Predict Readmission Risk"):

    # Get feature names from trained model
    feature_names = model.get_booster().feature_names

    # Create all columns with default 0
    data_dict = {col: 0 for col in feature_names}

    # Fill user inputs
    if 'age_num' in data_dict:
        data_dict['age_num'] = age

    if 'time_in_hospital' in data_dict:
        data_dict['time_in_hospital'] = time_in_hospital

    if 'num_lab_procedures' in data_dict:
        data_dict['num_lab_procedures'] = num_lab_procedures

    if 'num_medications' in data_dict:
        data_dict['num_medications'] = num_medications

    if 'number_diagnoses' in data_dict:
        data_dict['number_diagnoses'] = number_diagnoses

    if 'number_inpatient' in data_dict:
        data_dict['number_inpatient'] = number_inpatient

    if 'number_emergency' in data_dict:
        data_dict['number_emergency'] = number_emergency

    if 'number_outpatient' in data_dict:
        data_dict['number_outpatient'] = number_outpatient

    # Convert to dataframe
    data = pd.DataFrame([data_dict])

    # Predict probability
    probability = model.predict_proba(data)[0][1]

    # Custom healthcare threshold
    prediction = 1 if probability >= 0.15 else 0

    # =========================
    # OUTPUT
    # =========================

    st.subheader(f"📊 Risk Probability: {probability:.2f}")

    # Risk meter
    st.progress(float(probability))

    # Risk category
    if probability < 0.15:
        st.success("✅ LOW RISK of Hospital Readmission")

    elif probability < 0.35:
        st.warning("⚠ MEDIUM RISK of Hospital Readmission")

    else:
        st.error("🚨 HIGH RISK of Hospital Readmission")

    # Patient summary
    st.markdown("---")
    st.subheader("📋 Patient Summary")

    st.write(f"👤 Age: {age}")
    st.write(f"🏥 Time in Hospital: {time_in_hospital} days")
    st.write(f"🧪 Lab Procedures: {num_lab_procedures}")
    st.write(f"💊 Medications: {num_medications}")
    st.write(f"📄 Diagnoses: {number_diagnoses}")
    st.write(f"🚑 Previous Inpatient Visits: {number_inpatient}")
    st.write(f"⚡ Emergency Visits: {number_emergency}")
    st.write(f"🩺 Outpatient Visits: {number_outpatient}")