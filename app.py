#Importing all the required libraries
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import time
import joblib
import warnings
warnings.filterwarnings("ignore")
import shap
import matplotlib.pyplot as plt
import xgboost


# Page configuration
st.set_page_config(
    page_title="Approv.io - Smart Credit & Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1e3d59;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 10px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 20px 0;
    }
    .danger-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 20px 0;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'credit_data' not in st.session_state:
    st.session_state.credit_data = None
if 'loan_data' not in st.session_state:
    st.session_state.loan_data = None

def predict_credit_approval(data):
    preprocessor = joblib.load(open('preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    model = joblib.load(open('model.pkl', 'rb'))
    prediction = model.predict(final_data)
    confidence = model.predict_proba(final_data)[0].max() * 100
    return prediction[0], confidence


def scale_debt(user_debt, real_min=0, real_max=10000000, dataset_min=0, dataset_max=28):
    return ((user_debt - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min  
    
def scale_credit_score(user_score, real_min=300, real_max=900, dataset_min=0, dataset_max=67):
    return ((user_score - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min

def scale_income(user_income, real_min=0, real_max=100000000, dataset_min=0, dataset_max=100000):
    return ((user_income - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min


def predict_loan_approval(data):
    preprocessor = joblib.load(open('loan_preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    model = joblib.load(open('loan_model.pkl', 'rb'))
    prediction = model.predict(final_data)  
    confidence = model.predict_proba(final_data)[0].max() * 100
    return bool(prediction[0]), confidence

def get_loan_explaination(data):
    preprocessor = joblib.load(open('loan_preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('loan_explainer.pkl','rb'))
    shap_values = explainer(sample)
    return shap_values

def get_loan_top_features(data):
    preprocessor = joblib.load(open('loan_preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('loan_explainer.pkl','rb'))
    shap_values = explainer.shap_values(sample)

    feature_importance = pd.DataFrame({
        'Feature': sample.columns,
        'Feature Value': shap_values[0]
    }).sort_values(by='Feature Value', key=abs, ascending=False)

    return feature_importance

def get_credit_explaination(data):
    preprocessor = joblib.load(open('preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('credit_explainer.pkl','rb'))
    shap_values = explainer(sample)
    return shap_values

def get_credit_top_features(data):
    preprocessor = joblib.load(open('preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('credit_explainer.pkl','rb'))
    shap_values = explainer.shap_values(sample)

    feature_importance = pd.DataFrame({
        'Feature': sample.columns,
        'Feature Value': shap_values[0]
    }).sort_values(by='Feature Value', key=abs, ascending=False)

    return feature_importance

import streamlit as st

def home_page():
    # --- Finance-Themed CSS (Gold & Green) ---
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FFD700;
        background: linear-gradient(135deg, #0B6623, #014421);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 0 25px rgba(11, 102, 35, 0.5);
        margin-bottom: 30px;
    }
    .main-header h1 {
        font-size: 3rem;
        color: #FFD700;
        margin-bottom: 5px;
    }
    .main-header p {
        font-size: 1.2rem;
        color: #dcdcdc;
        margin: 0;
    }
    .info-box {
        background-color: #fefcf3;
        border: 2px solid #0B6623;
        border-radius: 10px;
        padding: 20px;
        color: #014421;
        text-align: center;
        transition: 0.3s ease;
        box-shadow: 0 0 15px rgba(11, 102, 35, 0.2);
    }
    .info-box:hover {
        transform: scale(1.03);
        box-shadow: 0 0 30px rgba(11, 102, 35, 0.4);
    }
    .info-box h3 {
        color: #0B6623;
        margin-bottom: 5px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0B6623, #014421);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 0 8px rgba(11, 102, 35, 0.3);
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #FFD700, #d4af37);
        color: #014421;
        transform: scale(1.05);
    }
    .metric-title {
        color: #014421;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>💳 CredLo</h1>
        <p>Smarter Credit & Loan Approvals Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 3 Info Boxes ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>⚡ Instant Decisions</h3>
            <p>Receive quick approval results powered by AI models trained on real financial data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🔒 Trusted & Secure</h3>
            <p>All your financial data is encrypted and handled with bank-level security protocols.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Transparent Insights</h3>
            <p>Every approval is backed by clear ML explainability for better financial awareness.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Navigation Buttons ---
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Credit Card Approval")
        st.write("Check your credit card eligibility and receive a detailed approval analysis instantly.")
        if st.button("Apply for Credit Card", key="credit_btn"):
            st.session_state.page = 'Credit Approval'
            st.rerun()

    with col2:
        st.markdown("### 🏦 Loan Approval")
        st.write("Submit your loan details and get real-time predictions with confidence scores.")
        if st.button("Apply for Loan", key="loan_btn"):
            st.session_state.page = 'Loan Approval'
            st.rerun()

    # --- Platform Statistics ---
    st.markdown("---")
    st.markdown("### 📈 Platform Insights")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Applications Processed", value="12,547", delta="↑ 15%")
    with col2:
        st.metric(label="Approval Rate", value="72%", delta="↑ 5%")
    with col3:
        st.metric(label="Avg. Decision Time", value="< 25s", delta="↓ 3s")
    with col4:
        st.metric(label="User Satisfaction", value="4.9 / 5", delta="↑ 0.3")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#0B6623;'>© 2025 CredLo | Empowering Financial Decisions through AI</p>",
        unsafe_allow_html=True
    )

