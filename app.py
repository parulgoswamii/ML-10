
import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Application Heading
st.title("🩺 Disease Prediction System")

st.write(
    "Enter patient details and select symptoms "
    "to predict a possible disease category."
)

st.divider()

# Load Trained Model
if not os.path.exists("models/disease_model.joblib"):
    st.error("Model not found! Please run train_model.py first.")
    st.stop()

model = joblib.load("models/disease_model.joblib")

# Patient Information
st.subheader("Patient Information")

age = st.number_input(
    "Enter Patient Age",
    min_value=1,
    max_value=120,
    value=25
)

st.divider()

# Symptoms Selection
st.subheader("Select Patient Symptoms")

symptoms = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Nausea",
    "Joint_Pain",
    "Rash",
    "Sore_Throat",
    "Breathing_Difficulty",
    "Body_Pain"
]

selected_symptoms = {}

for symptom in symptoms:

    selected_symptoms[symptom] = st.checkbox(
        symptom.replace("_", " ")
    )

st.divider()

# Prediction Button
if st.button("Predict Possible Disease", use_container_width=True):

    # Prepare Input Data
    input_data = {
        "Age": age
    }

    for symptom in symptoms:
        input_data[symptom] = int(selected_symptoms[symptom])

    input_df = pd.DataFrame([input_data])

    # Make Prediction
    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    # Display Result
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Disease Category: {prediction}"
    )

    # Probability Analysis
    st.subheader("Disease Probability Analysis")

    probability_df = pd.DataFrame({
        "Disease Category": model.classes_,
        "Probability (%)": probabilities * 100
    })

    probability_df = probability_df.sort_values(
        by="Probability (%)",
        ascending=False
    )

    st.bar_chart(
        probability_df.set_index("Disease Category")
    )

    st.dataframe(
        probability_df.style.format({
            "Probability (%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    # Medical Disclaimer
    st.warning(
        "Disclaimer: This is an educational machine learning "
        "project using synthetic sample data. It is not a "
        "medical diagnosis system. Please consult a qualified "
        "healthcare professional for medical advice."
    )

st.divider()

st.caption("Disease Prediction System | Machine Learning Project")
