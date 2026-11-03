"""Streamlit app: graduate-admission prediction from a statsmodels logistic regression (RFE-selected features)."""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
FEATURES = ["GRE Score", "University Rating", "CGPA"]
THRESHOLD = 0.6  # decision threshold on the predicted admission probability

# The model was fitted on standardised features (each column has mean ~0 and std ~1 in model.model.exog), so raw
# inputs must be standardised with the SAME training means and standard deviations before predicting.
# Export them from the training notebook into scaler.json: {"mean": {feature: value}, "std": {feature: value}}.
SCALER_PATH = HERE / "scaler.json"


@st.cache_resource
def load_model():
    with open(HERE / "logreg_rfe_model.pkl", "rb") as f:
        return pickle.load(f)


def load_scaler() -> dict | None:
    if not SCALER_PATH.exists():
        return None
    return json.loads(SCALER_PATH.read_text())


def prepare_input(gre: float, rating: int, cgpa: float, scaler: dict) -> pd.DataFrame:
    raw = {"GRE Score": gre, "University Rating": rating, "CGPA": cgpa}
    row = {"const": 1.0}
    for f in FEATURES:
        row[f] = (raw[f] - scaler["mean"][f]) / scaler["std"][f]
    return pd.DataFrame([row], columns=["const", *FEATURES])


st.title("📘 Admission Prediction App")
st.markdown(
    "Predicts whether a student is likely to be **admitted** from GRE score, university rating and CGPA."
)

gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=3)
cgpa = st.slider("CGPA", min_value=6.0, max_value=10.0, step=0.1, value=8.5)

scaler = load_scaler()
if scaler is None:
    st.error(
        "scaler.json is missing. The model expects standardised inputs; without the training means and standard "
        "deviations every prediction would be 'Admit' with probability 1.0. See README."
    )
elif st.button("Predict Admission"):
    prob = float(load_model().predict(prepare_input(gre, rating, cgpa, scaler))[0])
    label = "Admit" if prob >= THRESHOLD else "Reject"
    st.subheader(f"🎯 Result: **{label}**")
    st.write(f"📊 Probability of admission: **{prob:.4f}** (threshold {THRESHOLD})")
    if label == "Admit":
        st.success("Based on the input, admission is likely.")
    else:
        st.error("Based on the input, admission is unlikely.")
