import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠"
)


# --------------------------------------------------
# Load Model and Feature Names
# --------------------------------------------------

model = joblib.load("alzheimers_rf_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# --------------------------------------------------
# App Title
# --------------------------------------------------

st.title("🧠 Alzheimer's Disease Prediction System")

st.write(
    "Enter patient information below to generate a machine learning-based "
    "prediction for Alzheimer's Disease."
)

st.warning(
    "This application is intended for educational and decision-support purposes "
    "only and should not be considered a clinical diagnosis."
)


# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def yes_no_input(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0


# --------------------------------------------------
# Patient Demographic Information
# --------------------------------------------------

st.header("Patient Demographic Information")

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=60,
        max_value=90,
        value=70
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:

    Ethnicity = st.selectbox(
        "Ethnicity Code",
        [0, 1, 2, 3]
    )

    EducationLevel = st.selectbox(
        "Education Level Code",
        [0, 1, 2, 3]
    )


BMI = st.number_input(
    "BMI",
    min_value=15.0,
    max_value=40.0,
    value=25.0
)


# --------------------------------------------------
# Lifestyle Information
# --------------------------------------------------

st.header("Lifestyle Information")

Smoking = yes_no_input("Smoking")

AlcoholConsumption = st.number_input(
    "Alcohol Consumption",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

PhysicalActivity = st.number_input(
    "Physical Activity",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

DietQuality = st.number_input(
    "Diet Quality",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

SleepQuality = st.number_input(
    "Sleep Quality",
    min_value=4.0,
    max_value=10.0,
    value=7.0
)


# --------------------------------------------------
# Medical History
# --------------------------------------------------

st.header("Medical History")

col1, col2 = st.columns(2)

with col1:

    FamilyHistoryAlzheimers = yes_no_input(
        "Family History of Alzheimer's"
    )

    CardiovascularDisease = yes_no_input(
        "Cardiovascular Disease"
    )

    Diabetes = yes_no_input("Diabetes")


with col2:

    Depression = yes_no_input("Depression")

    HeadInjury = yes_no_input(
        "History of Head Injury"
    )

    Hypertension = yes_no_input("Hypertension")


# --------------------------------------------------
# Blood Pressure and Cholesterol
# --------------------------------------------------

st.header("Blood Pressure and Cholesterol")

SystolicBP = st.number_input(
    "Systolic Blood Pressure",
    min_value=90,
    max_value=179,
    value=120
)

DiastolicBP = st.number_input(
    "Diastolic Blood Pressure",
    min_value=60,
    max_value=119,
    value=80
)

CholesterolTotal = st.number_input(
    "Total Cholesterol",
    min_value=150.0,
    max_value=300.0,
    value=200.0
)

CholesterolLDL = st.number_input(
    "LDL Cholesterol",
    min_value=50.0,
    max_value=200.0,
    value=100.0
)

CholesterolHDL = st.number_input(
    "HDL Cholesterol",
    min_value=20.0,
    max_value=100.0,
    value=50.0
)

CholesterolTriglycerides = st.number_input(
    "Triglycerides",
    min_value=50.0,
    max_value=400.0,
    value=150.0
)


# --------------------------------------------------
# Cognitive and Functional Assessment
# --------------------------------------------------

st.header("Cognitive and Functional Assessment")

MMSE = st.number_input(
    "MMSE Score",
    min_value=0.0,
    max_value=30.0,
    value=15.0
)

FunctionalAssessment = st.number_input(
    "Functional Assessment Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

ADL = st.number_input(
    "Activities of Daily Living (ADL) Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)


# --------------------------------------------------
# Symptoms and Behavioral Indicators
# --------------------------------------------------

st.header("Symptoms and Behavioral Indicators")

MemoryComplaints = yes_no_input("Memory Complaints")

BehavioralProblems = yes_no_input("Behavioral Problems")

Confusion = yes_no_input("Confusion")

Disorientation = yes_no_input("Disorientation")

PersonalityChanges = yes_no_input("Personality Changes")

DifficultyCompletingTasks = yes_no_input(
    "Difficulty Completing Tasks"
)

Forgetfulness = yes_no_input("Forgetfulness")


# --------------------------------------------------
# Create Input Data
# --------------------------------------------------

input_data = pd.DataFrame(
    [[
        Age,
        0 if Gender == "Male" else 1,
        Ethnicity,
        EducationLevel,
        BMI,
        Smoking,
        AlcoholConsumption,
        PhysicalActivity,
        DietQuality,
        SleepQuality,
        FamilyHistoryAlzheimers,
        CardiovascularDisease,
        Diabetes,
        Depression,
        HeadInjury,
        Hypertension,
        SystolicBP,
        DiastolicBP,
        CholesterolTotal,
        CholesterolLDL,
        CholesterolHDL,
        CholesterolTriglycerides,
        MMSE,
        FunctionalAssessment,
        MemoryComplaints,
        BehavioralProblems,
        ADL,
        Confusion,
        Disorientation,
        PersonalityChanges,
        DifficultyCompletingTasks,
        Forgetfulness
    ]],
    columns=feature_names
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("Predict Alzheimer's Risk"):

    prediction = model.predict(input_data)[0]

    prediction_probability = model.predict_proba(
        input_data
    )[0]

    if prediction == 1:

        st.error(
            "Prediction: Higher likelihood of Alzheimer's Disease"
        )

    else:

        st.success(
            "Prediction: Lower likelihood of Alzheimer's Disease"
        )

    st.subheader("Prediction Probabilities")

    st.write(
        f"Lower likelihood: {prediction_probability[0] * 100:.2f}%"
    )

    st.write(
        f"Higher likelihood: {prediction_probability[1] * 100:.2f}%"
    )