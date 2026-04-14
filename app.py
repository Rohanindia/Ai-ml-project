import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="ChurnShield AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0a0f1e 100%); min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b61ff, #ff6b9d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.1; }
.hero-sub { color: #6b7a99; font-size: 1.05rem; font-weight: 300; margin-top: 0.5rem; }
.hero-badge { display: inline-block; background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3); color: #00d4ff; padding: 0.25rem 0.9rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem; }
.stat-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.stat-box { flex: 1; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center; }
.stat-number { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #00d4ff; }
.stat-label { font-size: 0.75rem; color: #6b7a99; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 0.2rem; }
.result-churn { background: linear-gradient(135deg, rgba(255,59,59,0.15), rgba(255,107,107,0.05));
    border: 1px solid rgba(255,59,59,0.4); border-radius: 20px;
    padding: 2.5rem; text-align: center; margin-top: 1rem; }
.result-stay { background: linear-gradient(135deg, rgba(0,212,122,0.15), rgba(0,212,122,0.05));
    border: 1px solid rgba(0,212,122,0.4); border-radius: 20px;
    padding: 2.5rem; text-align: center; margin-top: 1rem; }
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-title-churn { font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; color: #ff6b6b; margin-bottom: 0.5rem; }
.result-title-stay { font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; color: #00d47a; margin-bottom: 0.5rem; }
.result-prob { font-size: 1rem; color: #8892a4; margin-bottom: 1.5rem; }
.prob-bar-bg { background: rgba(255,255,255,0.08); border-radius: 50px;
    height: 10px; width: 100%; margin: 1rem 0; overflow: hidden; }
.prob-bar-fill-churn { height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #ff6b6b, #ff3b3b); }
.prob-bar-fill-stay { height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #00d47a, #00b860); }
.tip-box-churn { background: rgba(255,107,107,0.1); border-left: 3px solid #ff6b6b;
    border-radius: 8px; padding: 1rem 1.2rem; text-align: left;
    color: #ccd0da; font-size: 0.9rem; margin-top: 1rem; }
.tip-box-stay { background: rgba(0,212,122,0.1); border-left: 3px solid #00d47a;
    border-radius: 8px; padding: 1rem 1.2rem; text-align: left;
    color: #ccd0da; font-size: 0.9rem; margin-top: 1rem; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(123,97,255,0.4), transparent); margin: 2rem 0; }
label { color: #8892a4 !important; font-size: 0.82rem !important;
    font-weight: 500 !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; }
div.stButton > button { width: 100%;
    background: linear-gradient(135deg, #7b61ff, #00d4ff);
    color: white; border: none; border-radius: 12px; padding: 0.9rem 2rem;
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    letter-spacing: 0.05em; cursor: pointer; text-transform: uppercase; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ================================================================
#   LOAD MODEL, SCALER, AND COLUMN ORDER
#   We load all 3 saved files from churn.py
#   feature_columns.pkl tells us the EXACT order of features
# ================================================================
@st.cache_resource
def load_model():
    model   = joblib.load("churn_model.pkl")    # trained ML model
    scaler  = joblib.load("scaler.pkl")          # data scaler
    columns = joblib.load("feature_columns.pkl") # exact column order from training
    return model, scaler, columns

model, scaler, columns = load_model()

# ── Hero ──────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2.5rem;">
    <div class="hero-badge">🛡️ AI-Powered Analytics</div>
    <div class="hero-title">ChurnShield AI</div>
    <div class="hero-sub">Predict customer churn before it happens — powered by Machine Learning</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat-box"><div class="stat-number">79.2%</div><div class="stat-label">Model Accuracy</div></div>
    <div class="stat-box"><div class="stat-number">7,043</div><div class="stat-label">Training Records</div></div>
    <div class="stat-box"><div class="stat-number">34</div><div class="stat-label">Features Analyzed</div></div>
    <div class="stat-box"><div class="stat-number">Random Forest</div><div class="stat-label">Algorithm</div></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p style="color:#7b61ff;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;">👤 Customer Profile</p>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner", ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    phone          = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7b61ff;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;">🌐 Internet & Services</p>', unsafe_allow_html=True)
    internet     = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bak   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col_right:
    st.markdown('<p style="color:#7b61ff;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;">💳 Billing & Contract</p>', unsafe_allow_html=True)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment         = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0, step=10.0)

    st.markdown('<br>', unsafe_allow_html=True)
    predict_btn = st.button("🔮  Run Churn Prediction")

    if predict_btn:
        # ── Build input dictionary with ALL features ──────────
        input_dict = {
            "SeniorCitizen"                        : [1 if senior == "Yes" else 0],
            "tenure"                               : [tenure],
            "MonthlyCharges"                       : [monthly_charges],
            "TotalCharges"                         : [total_charges],
            "gender_Male"                          : [1 if gender == "Male" else 0],
            "Partner_Yes"                          : [1 if partner == "Yes" else 0],
            "Dependents_Yes"                       : [1 if dependents == "Yes" else 0],
            "PhoneService_Yes"                     : [1 if phone == "Yes" else 0],
            "MultipleLines_No phone service"       : [1 if multiple_lines == "No phone service" else 0],
            "MultipleLines_Yes"                    : [1 if multiple_lines == "Yes" else 0],
            "InternetService_Fiber optic"          : [1 if internet == "Fiber optic" else 0],
            "InternetService_No"                   : [1 if internet == "No" else 0],
            "OnlineSecurity_No internet service"   : [1 if online_sec == "No internet service" else 0],
            "OnlineSecurity_Yes"                   : [1 if online_sec == "Yes" else 0],
            "OnlineBackup_No internet service"     : [1 if online_bak == "No internet service" else 0],
            "OnlineBackup_Yes"                     : [1 if online_bak == "Yes" else 0],
            "DeviceProtection_No internet service" : [1 if device_prot == "No internet service" else 0],
            "DeviceProtection_Yes"                 : [1 if device_prot == "Yes" else 0],
            "TechSupport_No internet service"      : [1 if tech_support == "No internet service" else 0],
            "TechSupport_Yes"                      : [1 if tech_support == "Yes" else 0],
            "StreamingTV_No internet service"      : [1 if streaming_tv == "No internet service" else 0],
            "StreamingTV_Yes"                      : [1 if streaming_tv == "Yes" else 0],
            "StreamingMovies_No internet service"  : [1 if streaming_mv == "No internet service" else 0],
            "StreamingMovies_Yes"                  : [1 if streaming_mv == "Yes" else 0],
            "Contract_One year"                    : [1 if contract == "One year" else 0],
            "Contract_Two year"                    : [1 if contract == "Two year" else 0],
            "PaperlessBilling_Yes"                 : [1 if paperless == "Yes" else 0],
            "PaymentMethod_Credit card (automatic)": [1 if payment == "Credit card (automatic)" else 0],
            "PaymentMethod_Electronic check"       : [1 if payment == "Electronic check" else 0],
            "PaymentMethod_Mailed check"           : [1 if payment == "Mailed check" else 0],
            # 4 engineered features — must match churn.py Step 3
            "ChargesPerMonth"  : [total_charges / (tenure + 1)],
            "HighSpender"      : [1 if monthly_charges > 70 else 0],
            "LongTermCustomer" : [1 if tenure > 24 else 0],
            "NewCustomer"      : [1 if tenure < 6 else 0],
        }

        # ── Convert to DataFrame ──────────────────────────────
        input_df = pd.DataFrame(input_dict)

        # ── ⭐ KEY FIX: Reorder columns to match training order ─
        # Without this line, the model crashes with "wrong order" error
        # columns variable loaded from feature_columns.pkl (saved by churn.py)
        input_df = input_df[columns]

        # ── Scale using same scaler from training ─────────────
        input_scaled = scaler.transform(input_df)

        # ── Predict ───────────────────────────────────────────
        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        churn_prob  = round(probability[1] * 100, 1)
        stay_prob   = round(probability[0] * 100, 1)

        # ── Show Result ───────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-title-churn">HIGH CHURN RISK</div>
                <div class="result-prob">This customer has a <strong style="color:#ff6b6b">{churn_prob}%</strong> probability of leaving</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill-churn" style="width:{churn_prob}%"></div></div>
                <div class="tip-box-churn">💡 <strong>Recommended Action:</strong> Offer a loyalty discount, upgrade to a longer contract, or provide a free service add-on to retain this customer.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-stay">
                <div class="result-icon">✅</div>
                <div class="result-title-stay">LOW CHURN RISK</div>
                <div class="result-prob">This customer has only a <strong style="color:#00d47a">{churn_prob}%</strong> probability of leaving</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill-stay" style="width:{stay_prob}%"></div></div>
                <div class="tip-box-stay">🎉 <strong>Great News:</strong> This customer appears satisfied and loyal. Continue providing excellent service to maintain their trust.</div>
            </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#3d4558;font-size:0.8rem;padding-bottom:2rem;">
    Built with ❤️ using Python · Scikit-learn · Streamlit &nbsp;|&nbsp; Telco Customer Churn Dataset
</div>""", unsafe_allow_html=True)