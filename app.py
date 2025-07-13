import streamlit as st
import pickle
import pandas as pd

# Load trained statsmodels model
with open("logreg_rfe_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define feature names used in training
feature_names = ['const', 'GRE Score', 'University Rating', 'CGPA']

# Prepare input for prediction
def prepare_input(gre, rating, cgpa):
    data = [[1.0, gre, rating, cgpa]]
    return pd.DataFrame(data, columns=feature_names)

# Set up Streamlit UI
st.title("📘 Admission Prediction App")
st.markdown("This app predicts whether a student will be **admitted** or **rejected** based on GRE Score, University Rating, and CGPA.")

# Input fields
gre = st.number_input("GRE Score", min_value=200, max_value=340, value=320)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=3)
cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, step=0.1, value=8.5)

# Predict button
if st.button("Predict Admission"):
    input_df = prepare_input(gre, rating, cgpa)
    prob = model.predict(input_df)[0]
    label = "Admit" if prob >= 0.6 else "Reject"
    
    st.subheader(f"🎯 Result: **{label}**")
    st.write(f"📊 Probability of Admission: **{prob:.4f}**")

    if label == "Admit":
        st.success("Congratulations! Based on the input, admission is likely.")
    else:
        st.error("Unfortunately, admission is unlikely based on the input.")
