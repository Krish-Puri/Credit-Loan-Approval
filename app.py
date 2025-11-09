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


