import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature options
Gender = {
    1: 'Male(1)',
    2: 'Female(2)'

}
BMI = {
    0: '<18.5 Kg/m2 (0)',
    1: '18.5-24 Kg/m2 (1)',
    2: '24-28 Kg/m2 (2)',
    3: '≥28 Kg/m2 (3)'
}



Duration_of_diabetes = {
    1: '<1year (1)',
    2: '1-5years (2)',
    3: '6-10years (3)',
    4: '>10years (4)'
}

Smoke = {
    0: 'NO(0)',
    1: 'YES(1)'
}

Drink = {
    0: 'NO(0)',
    1: 'YES(1)'
}

SCII = {
    0: 'NO(0)',
    1: 'YES(1)'
}

Family_history = {
    0: 'NO(0)',
    1: 'YES(1)'
}

FBG = {
    0: 'NO(0)',
    1: 'YES(1)'
}

HbAlc = {
    0: 'NO(0)',
    1: 'YES(1)'
}

BUN = {
    0: 'NO(0)',
    1: 'YES(1)'
}
Scr = {
    0: 'NO(0)',
    1: 'YES(1)'
}

ACR = {
    1: 'NAU(1)',
    2: 'MAU(2)',
    3: 'CAU(3)'
}

Proteinuria = {
    0: 'NO(0)',
    1: 'YES(1)'
}
Glucoseinurine = {
    0: 'NO(0)',
    1: 'YES(1)'
}
Carotid_plaque = {
    0: 'NO(0)',
    1: 'YES(1)'
}

TG = {
    0: 'NO(0)',
    1: 'YES(1)'
}


HDL = {
    0: 'NO(0)',
    1: 'YES(1)'
}

LDL = {
    0: 'NO(0)',
    1: 'YES(1)'
}


TC = {
    0: 'NO(0)',
    1: 'YES(1)'
}


DR = {
    0: 'NO(0)',
    1: 'YES(1)'
}

Fattyliver = {
    0: 'NO(0)',
    1: 'YES(1)'
}

DN = {
    0: 'NO(0)',
    1: 'YES(1)'
}

DPN = {
    0: 'NO(0)',
    1: 'YES(1)'
}
CAD = {
    0: 'NO(0)',
    1: 'YES(1)'
}
CVD = {
    0: 'NO(0)',
    1: 'YES(1)'
}
Hypertension = {
    0: 'NO(0)',
    1: 'YES(1)'
}
# Define feature names
feature_names = [
    "Age", "Gender", "BMI", "Duration_of_diabetes",
    "Smoke", "Drink", "SCII", "Family_history", "FBG",
    "HbAlc", "BUN", "Scr", "ACR", "Proteinuria",
    "Glucoseinurine", "Carotid_plaque", "TG", "HDL",
    "LDL", "TC", "Fattyliver", "DR", "DN", "DPN", "CAD", "CVD", "Hypertension"]

# Streamlit user interface
st.title("HUA Predictor")

# age: numerical input
Age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# sex: categorical selection
Gender = st.selectbox("Gender (0=Male, 1=Female):", options=[0, 1], format_func=lambda x: 'Male (0)' if x == 0 else 'Female (1)')
BMI = st.selectbox("BMI:", options=list(BMI.keys()), format_func=lambda x: BMI[x])
Duration_of_diabetes = st.selectbox("Duration_of_diabetes:", options=list(Duration_of_diabetes.keys()), format_func=lambda x: Duration_of_diabetes[x])
Smoke = st.selectbox("Smoke (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
Drink = st.selectbox("Drink (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
SCII = st.selectbox("SCII (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')

Family_history = st.selectbox("Family_history (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
FBG = st.selectbox("FBG (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
HbAlc = st.selectbox("HbAlc (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')

BUN = st.selectbox("BUN (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
Scr = st.selectbox("Scr (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
ACR = st.selectbox("ACR:", options=list(ACR.keys()), format_func=lambda x: ACR[x])
Proteinuria = st.selectbox("Proteinuria (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
Glucoseinurine = st.selectbox("Glucoseinurine (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')

Carotid_plaque = st.selectbox("Carotid_plaque (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
TG = st.selectbox("TG (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')

HDL = st.selectbox("HDL (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
LDL = st.selectbox("LDL (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
TC = st.selectbox("TC (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')


Fattyliver = st.selectbox("Fattyliver (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
DR = st.selectbox("DR (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
DN = st.selectbox("DN (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
DPN = st.selectbox("DPN (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
CAD = st.selectbox("CAD (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
CVD = st.selectbox("CVD (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
Hypertension = st.selectbox("Hypertension (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')


# Process inputs and make predictions
feature_values = [Age, Gender, BMI, Duration_of_diabetes,
    Smoke, Drink, SCII, Family_history, FBG,
    HbAlc, BUN, Scr, ACR, Proteinuria,
    Glucoseinurine, Carotid_plaque, TG, HDL,
    LDL, TC, Fattyliver, DR, DN, DPN, CAD, CVD, Hypertension]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")