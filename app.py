import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD MODEL SAFELY ----------------
loaded_obj = pickle.load(open("customer_churn_model.pkl", "rb"))

if isinstance(loaded_obj, dict):
    model = loaded_obj.get("model") or loaded_obj.get("clf")
elif isinstance(loaded_obj, tuple):
    model = loaded_obj[0]
else:
    model = loaded_obj

# Load encoders
encoders = pickle.load(open("encoders.pkl", "rb"))

# Training feature order
expected_features = model.feature_names_in_

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

# ---------------- INPUT SECTION ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, step=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=50.0)

# ✅ AUTO-CALCULATE TotalCharges (IMPORTANT FIX)
total_charges = tenure * monthly_charges
st.info(f"Calculated Total Charges: {total_charges}")
st.caption("ℹ️ Total Charges are auto-calculated based on tenure and monthly charges.")

# ---------------- CREATE INPUT DATAFRAME ----------------
input_df = pd.DataFrame([{
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# ---------------- APPLY ENCODERS ----------------
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# ---------------- ALIGN FEATURES WITH TRAINING ----------------
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    # Probability of churn (class = 1)
    churn_proba = model.predict_proba(input_df)[0][1]

    st.write(f"🔍 Churn Probability: **{churn_proba:.2%}**")

    # ✅ Business-friendly threshold
    if churn_proba > 0.6:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")






