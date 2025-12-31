import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and the scaler
# Ensure these files are in the same folder as app.py
try:
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading files: {e}. Make sure 'loan_model.pkl' and 'scaler.pkl' are in this folder.")

st.set_page_config(page_title="Bank Loan Predictor", layout="centered")
st.title("🏦 Bank Loan Approval Predictor")

# 2. Input Section
st.subheader("Applicant Information")
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", 0, 10, 2)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", value=500000)
    loan_amount = st.number_input("Loan Amount (₹)", value=100000)

with col2:
    loan_term = st.number_input("Loan Term (Years)", 1, 20, 10)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 700)
    # Assets for calculation
    res_assets = st.number_input("Residential Asset Value (₹)", value=100000)
    comm_assets = st.number_input("Commercial Asset Value (₹)", value=0)
    lux_assets = st.number_input("Luxury Asset Value (₹)", value=0)
    bank_assets = st.number_input("Bank Asset Value (₹)", value=100000)

# 3. Prediction Button
if st.button("Predict Loan Status"):
    # A. Calculate Engineered Features
    total_assets = res_assets + comm_assets + lux_assets + bank_assets
    loan_income_ratio = loan_amount / income_annum
    assets_loan_ratio = total_assets / loan_amount if loan_amount != 0 else 0
    income_per_dependent = income_annum / (no_of_dependents + 1)
    
    # B. Handle Encoding (Matching the exact space-sensitive names)
    edu_not_grad = 1 if education == "Not Graduate" else 0
    emp_yes = 1 if self_employed == "Yes" else 0

    # C. Create Feature DataFrame in your EXACT provided order
    feature_names = ['no_of_dependents', 
        'loan_amount', 
        'loan_term', 
        'cibil_score', 
        'loan_income_ratio', 
        'education_ Not Graduate', 
        'self_employed_ Yes',      
        'total_assets', 
        'assets_loan_ratio', 
        'income_per_dependent'
        
    ]
    
    # Map inputs to the order
    data = {
        'no_of_dependents': no_of_dependents,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'loan_income_ratio': loan_income_ratio,
        'education_ Not Graduate': edu_not_grad,
        'self_employed_ Yes': emp_yes,
        'total_assets': total_assets,
        'assets_loan_ratio': assets_loan_ratio,
        'income_per_dependent': income_per_dependent
    }
    
    features_df = pd.DataFrame([data])[feature_names]

    # D. Scaling (Only for the numeric columns you scaled in the notebook)
    cols_to_scale = [
        'loan_amount', 'loan_term', 'cibil_score', 'loan_income_ratio', 
        'total_assets', 'no_of_dependents', 'assets_loan_ratio', 'income_per_dependent'
    ]
    
    try:
        features_df[cols_to_scale] = scaler.transform(features_df[cols_to_scale])
        
        # E. Final Prediction
        prediction = model.predict(features_df)
        prob = model.predict_proba(features_df)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.success(f"✅ **Loan Approved!** (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"❌ **Loan Rejected** (Confidence: {(1-prob)*100:.1f}%)")
            
    except Exception as e:

        st.error(f"Prediction Error: {e}")
