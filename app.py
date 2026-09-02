import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# Load trained artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load('loan_decision_tree.pkl')

artifacts = load_artifacts()
model = artifacts['model']
encoders = artifacts['encoders']

st.title("🏦 Loan Approval Prediction System")
st.write("Decision Tree Classification Model")

st.markdown("---")

# User Input Form
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=1500, step=500)
    loan_amount = st.number_input("Loan Amount ($ in thousands)", min_value=10, value=150, step=10)
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 360], index=3)

with col2:
    credit_history = st.selectbox("Credit History", ["Clear History (1.0)", "Debts/Outstanding (0.0)"])
    credit_val = 1.0 if "1.0" in credit_history else 0.0
    
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Inference trigger
if st.button("Evaluate Application", type="primary"):
    # Encode categorical inputs
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_val,
        'Education': encoders['Education'].transform([education])[0],
        'Self_Employed': encoders['Self_Employed'].transform([self_employed])[0],
        'Property_Area': encoders['Property_Area'].transform([property_area])[0]
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    st.subheader("Decision Result")
    
    if prediction == 1:
        st.success("✅ **Loan Status: Approved**")
        st.info(f"Model Confidence: {probabilities[1] * 100:.1f}%")
    else:
        st.error("❌ **Loan Status: Rejected**")
        st.warning(f"Model Confidence: {probabilities[0] * 100:.1f}%")
