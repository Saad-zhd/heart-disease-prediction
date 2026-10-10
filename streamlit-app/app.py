"""
Heart Disease Risk Prediction - Web Application
------------------------------------------------
A Streamlit web application that allows patients and clinicians to interact with
the trained Support Vector Classifier (SVC) model developed for Assignment 2.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the training notebook.
    Fitting on the same category lists reproduces the exact deterministic codes.
    """
    encoders = {}
    encoders["ChestPain"] = LabelEncoder().fit(["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    encoders["Gender"] = LabelEncoder().fit(["Female", "Male"])
    encoders["ExerciseAngina"] = LabelEncoder().fit(["No", "Yes"])
    encoders["MajorVessels"] = LabelEncoder().fit(["Zero", "One", "Two", "Three", "Four"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# UI Content
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This diagnostic decision-support system is built from the **Heart Disease Prediction System**
    machine learning model. Select patient clinical attributes below, and the trained Support Vector
    Classifier (SVC) will predict the risk of coronary heart disease.

    *Model accuracy on held-out test data: **81%** (UCI Heart Disease benchmark).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        index=0,
        help="Type of chest discomfort experienced by the patient."
    )
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        index=1
    )

with col2:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["No", "Yes"],
        index=0,
        help="Angina provoked during exercise / physical exertion."
    )
    major_vessels = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        ["Zero", "One", "Two", "Three", "Four"],
        index=0,
        help="Number of major coronary blood vessels visualized."
    )

if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "Gender": [gender],
        "ExerciseAngina": [exercise_angina],
        "MajorVessels": [major_vessels],
    })

    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "MajorVessels": encoders["MajorVessels"].transform(user_input["MajorVessels"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE DETECTED (HIGH RISK)")
        st.warning("The model indicates clinical risk indicators consistent with coronary artery disease. Further cardiology evaluation is recommended.")
    else:
        st.success("### ✅ NO HEART DISEASE DETECTED (LOW RISK)")
        st.info("The model indicates low probability of obstructive coronary artery disease based on the provided clinical features.")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for educational purposes (AIC354 Machine Learning Fundamentals - COMSATS University). "
    "This tool is an academic demo and does not replace professional clinical evaluation."
)
