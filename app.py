import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessing files
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Page settings
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Loan Default Risk Analysis & Prediction")
st.info(
    "Fill in the applicant's information and click **Predict Loan Default Risk** to estimate the likelihood of loan default."
)
# ==========================
# User Input Section
# ==========================

st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=100000.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
    loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36)
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.30)
    # ==========================
# Additional Information
# ==========================

st.header("Additional Information")

col3, col4 = st.columns(2)

with col3:
    education = st.selectbox(
        "Education",
        ["Bachelor's", "High School", "Master's", "PhD"]
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Auto", "Business", "Education", "Home", "Other"]
    )

with col4:

    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=0,
        value=2
    )

    has_mortgage = st.selectbox(
        "Has Mortgage?",
        ["No", "Yes"]
    )

    has_dependents = st.selectbox(
        "Has Dependents?",
        ["No", "Yes"]
    )

    has_cosigner = st.selectbox(
        "Has Co-signer?",
        ["No", "Yes"]
    )
    # ==========================
# Prediction
# ==========================

if st.button("Predict Loan Default Risk"):

    # Create input dictionary
    input_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,

        "Education_High School": 0,
        "Education_Master's": 0,
        "Education_PhD": 0,

        "EmploymentType_Part-time": 0,
        "EmploymentType_Self-employed": 0,
        "EmploymentType_Unemployed": 0,

        "MaritalStatus_Married": 0,
        "MaritalStatus_Single": 0,

        "HasMortgage_Yes": 0,
        "HasDependents_Yes": 0,

        "LoanPurpose_Business": 0,
        "LoanPurpose_Education": 0,
        "LoanPurpose_Home": 0,
        "LoanPurpose_Other": 0,

        "HasCoSigner_Yes": 0
    }

    # One-hot encoding

    if education == "High School":
        input_data["Education_High School"] = 1
    elif education == "Master's":
        input_data["Education_Master's"] = 1
    elif education == "PhD":
        input_data["Education_PhD"] = 1

    if employment_type == "Part-time":
        input_data["EmploymentType_Part-time"] = 1
    elif employment_type == "Self-employed":
        input_data["EmploymentType_Self-employed"] = 1
    elif employment_type == "Unemployed":
        input_data["EmploymentType_Unemployed"] = 1

    if marital_status == "Married":
        input_data["MaritalStatus_Married"] = 1
    elif marital_status == "Single":
        input_data["MaritalStatus_Single"] = 1

    if has_mortgage == "Yes":
        input_data["HasMortgage_Yes"] = 1

    if has_dependents == "Yes":
        input_data["HasDependents_Yes"] = 1

    if loan_purpose == "Business":
        input_data["LoanPurpose_Business"] = 1
    elif loan_purpose == "Education":
        input_data["LoanPurpose_Education"] = 1
    elif loan_purpose == "Home":
        input_data["LoanPurpose_Home"] = 1
    elif loan_purpose == "Other":
        input_data["LoanPurpose_Other"] = 1

    if has_cosigner == "Yes":
        input_data["HasCoSigner_Yes"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Match training feature order
    input_df = input_df[feature_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    st.metric(
        "Probability of Default",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("High Risk: The applicant has a high predicted probability of loan default. Additional credit assessment may be recommended.")
    else:
        st.success("Low Risk: The applicant appears to have a relatively low probability of loan default based on the provided information.")