import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("pima.pkl")

st.set_page_config(page_title="Pima Diabetes Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Pima Diabetes Prediction App")
st.markdown("Predict whether a patient is **Diabetic** or **Non-Diabetic** using medical inputs.")

# Sidebar Inputs
st.sidebar.header("Patient Information")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input(
    "Glucose (mg/dL)", min_value=40, max_value=600, value=90,
    help="Fasting blood glucose, after 8–12 hours without food"
)
blood_pressure = st.sidebar.number_input(
    "Blood Pressure (Diastolic mmHg)", min_value=40, max_value=200, value=70,
    help="Resting diastolic blood pressure, measured with a cuff"
)
skin_thickness = st.sidebar.number_input(
    "Skin Thickness (mm)", min_value=0, max_value=99, value=20,
    help="Triceps skinfold thickness in millimeters — measures fat under the skin"
)
insulin = st.sidebar.number_input(
    "Insulin (µIU/mL)", min_value=0, max_value=300, value=10,
    help="Fasting insulin, after 8–12 hours without food"
)
bmi = st.sidebar.number_input(
    "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
    help="Body Mass Index, calculated from weight and height — kg/m²"
)
dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function (DPF)", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
    help="Family history score for diabetes — higher means stronger genetic risk"
)
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)

# -------------------------
# Feature Engineering
# -------------------------
N1 = 1
if (age <= 30 and glucose <= 120) or (30 < age < 48 and glucose <= 88) or (age >= 63 and glucose <= 142):
    N1 = 0

N2 = 0 if bmi <= 30 else 1
N3 = 1
if (age <= 27 and pregnancies <= 6) or (age > 60 and pregnancies > 7.5):
    N3 = 0

N4 = 1
if (glucose <= 105 and blood_pressure <= 80) or (glucose <= 105 and blood_pressure > 83):
    N4 = 0

N5 = 0 if skin_thickness <= 20 else 1
N6 = 1
if (bmi < 30 and skin_thickness <= 20) or (bmi > 33 and skin_thickness <= 20):
    N6 = 0

N7 = 1
if (glucose <= 105 and bmi <= 30) or (glucose <= 105 and bmi >= 40):
    N7 = 0

N9 = 0 if insulin <= 20 else 1
N10 = 0 if blood_pressure < 80 else 1
N11 = 0 if (pregnancies < 4 and pregnancies != 0) else 1

N0 = bmi * skin_thickness
N8 = pregnancies / age if age != 0 else 0
N13 = glucose / dpf if dpf != 0 else 0
N12 = age * dpf
N15 = 0 if N0 < 1034 else 1

# -------------------------
# Final Feature Vector
# -------------------------
feature_names = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
    'BMI','DiabetesPedigreeFunction','Age','N1','N2','N3','N6','N7',
    'N9','N10','N11','N0','N8','N13','N12','N15'
]

features = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
    N1, N2, N3, N6, N7, N9, N10, N11, N0, N8, N13, N12, N15
]], columns=feature_names)

# -------------------------
# Prediction + Plots
# -------------------------
if st.sidebar.button("Predict"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    result = "🩸 Diabetic" if prediction == 1 else "✅ Non-Diabetic"
    st.subheader("Prediction Result:")
    st.success(f"{result} (Confidence: {proba:.2f})")

        # -------------------------
    # Medical Chart: Patient vs Normal Ranges
    # -------------------------
    normal_ranges = {
        "Glucose (mg/dL)": (70, 99),              # fasting
        "Blood Pressure (Diastolic mmHg)": (60, 80),
        "Skin Thickness (mm)": (10, 40),          # typical dataset range
        "Insulin (µIU/mL)": (2, 25),              # fasting
        "BMI (kg/m²)": (18.5, 24.9),
        "DPF": (0.0, 1.0),
    }

    patient_values = {
        "Glucose (mg/dL)": glucose,
        "Blood Pressure (Diastolic mmHg)": blood_pressure,
        "Skin Thickness (mm)": skin_thickness,
        "Insulin (µIU/mL)": insulin,
        "BMI (kg/m²)": bmi,
        "DPF": dpf,
    }

    st.subheader("📊 Medical Chart: Patient vs Normal Range")

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (feature, (low, high)) in enumerate(normal_ranges.items()):
        value = patient_values[feature]

        # Plot normal range as a horizontal bar
        ax.hlines(y=i+0.2, xmin=low, xmax=high, color="lightblue", linewidth=6)

        # Plot patient value as a red/green dot slightly below the bar
        color = "green" if low <= value <= high else "red"
        ax.plot(value, i-0.1, "o", color=color, markersize=10)

    ax.set_yticks(range(len(normal_ranges)))
    ax.set_yticklabels(list(normal_ranges.keys()))
    ax.set_xlabel("Value")
    ax.set_title("Patient Values vs Normal Ranges")

    # Custom legend (placed below the chart)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lightblue", lw=6, label="Normal Range"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Patient (Normal)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Patient (Abnormal)")
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.3, -0.15), ncol=3, frameon=False)
    col1, col2, col3 = st.columns([0.005, 7, 1])  # middle col wider
    with col2:
        st.pyplot(fig)
