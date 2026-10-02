"""Minimal diabetes risk screener app for public hosting."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Check", page_icon="🩺", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load("diabetes_model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("Model file not found. Run `python train_model.py` first, then restart the app.")
    st.stop()

model = bundle["model"]
threshold = bundle["threshold"]

st.markdown(
    """
    <style>
        .block-container { padding-top: 0.5rem; padding-bottom: 1rem; }
        .stApp {
            background: radial-gradient(circle at top, #12213d 0%, #0b1120 45%, #0a0f1a 100%);
            color: #f8fafc;
        }
        .nav-wrap {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.6rem 0.2rem 0.8rem 0.2rem; margin-bottom: 0.75rem;
            border-bottom: 1px solid rgba(148,163,184,0.2);
        }
        .nav-brand { font-size: 1.1rem; font-weight: 700; color: #f8fafc; }
        .nav-links { display: flex; gap: 1rem; align-items: center; }
        .nav-links a {
            color: #cbd5e1; text-decoration: none; font-size: 0.9rem;
        }
        .nav-links a:hover { color: #ffffff; }
        div[data-testid="stForm"] > div {
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 14px;
            padding: 0.9rem;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.3);
        }
        .stMetric { border-radius: 10px; }
        .stCaption { color: #cbd5e1; }
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nav-wrap">
        <div class="nav-brand">🩺 Diabetes Check</div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#risk">Check</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Diabetes Check")
st.caption("Simple screening for early awareness.")
st.write("A lightweight tool to estimate risk based on common health indicators.")

with st.form("risk_form"):
    st.write("Enter the details below to estimate risk.")

    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=100, value=35)
    pregnancies = col2.number_input("Pregnancies", min_value=0, max_value=20, value=0)

    col1, col2 = st.columns(2)
    height_cm = col1.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=168.0, step=0.5)
    weight_kg = col2.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=68.0, step=0.5)

    family = st.selectbox(
        "Family history of diabetes",
        ["Not sure", "None known", "One close relative", "Several close relatives"],
    )

    col1, col2 = st.columns(2)
    glucose = col1.number_input("Glucose (mg/dL)", min_value=40, max_value=300, value=100)
    bp = col2.number_input("Blood pressure (mm Hg)", min_value=30, max_value=140, value=72)

    col1, col2 = st.columns(2)
    skin = col1.number_input("Skin fold (mm)", min_value=0, max_value=100, value=0)
    insulin = col2.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=0)

    submitted = st.form_submit_button("Check risk", type="primary", use_container_width=True)

if submitted:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    dpf = {"Not sure": np.nan, "None known": 0.2, "One close relative": 0.5, "Several close relatives": 1.0}[family]

    row = pd.DataFrame(
        [{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": np.nan if skin == 0 else skin,
            "Insulin": np.nan if insulin == 0 else insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }],
        columns=bundle["features"],
    )

    score = float(model.predict_proba(row)[0, 1])
    elevated = score >= threshold

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Risk score", f"{score:.0%}")
    col2.metric("BMI", f"{bmi:.1f}")

    if elevated:
        st.warning(
            "This suggests an elevated risk signal. It is not a diagnosis. Please speak with a clinician if needed."
        )
    else:
        st.success(
            "No elevated risk signal was detected. This does not rule out diabetes if symptoms are present."
        )

    st.caption(f"Flagged at {threshold:.0%}+")

st.caption("Educational tool only. Not medical advice.")
