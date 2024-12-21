# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Function to generate insurance data
def generate_insurance_data(num_samples=10000):
    insurance_companies = ['InsureCo', 'SafeGuard', 'TrustInsurance', 'SecureCoverage', 'LIC', 'StarHealth', 'PolicyBazaar']
    policy_types = ['Auto', 'Home', 'Health', 'Life', 'Travel']
    claim_types = ['Property Damage', 'Bodily Injury', 'Medical Expenses', 'Theft', 'Death', 'Natural Disaster']
    fraud_indicators = ['yes', 'no']
    claim_statuses = ['Pending', 'Approved', 'Denied']
    risk_levels = ['Low', 'Medium', 'High', 'Very High']
    education_levels = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD', 'Other']

    df = {
        'Insurance_Company': np.random.choice(insurance_companies, num_samples),
        'Policy_Type': np.random.choice(policy_types, num_samples),
        'Claim_Type': np.random.choice(claim_types, num_samples),
        'Claim_Amount_USD': np.random.lognormal(mean=9, sigma=1.5, size=num_samples),
        'Fraud_Indicator': np.random.choice(fraud_indicators, num_samples),
        'Claimant_Age': np.random.randint(18, 80, num_samples),
        'Claimant_Gender': np.random.choice(['Male', 'Female', 'Other'], num_samples, p=[0.48, 0.48, 0.04]),
        'Incident_Location': np.random.choice(['Urban', 'Suburban', 'Rural'], num_samples),
        'Accident_Type': np.random.choice(['Collision', 'Theft', 'Natural Disaster'], num_samples),
        'Risk_Level': np.random.choice(risk_levels, num_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'Witnesses': np.random.poisson(lam=1.5, size=num_samples),
        'Police_Involved': np.random.choice([True, False], num_samples, p=[0.7, 0.3]),
        'Injury_Reported': np.random.choice([True, False], num_samples, p=[0.55, 0.45]),
        'Vehicle_Damage': np.random.choice(['Mild', 'Moderate', 'Severe'], num_samples, p=[0.4, 0.4, 0.2]),
        'Policy_Holder_Years': np.random.randint(1, 20, num_samples),
        'Claim_Status': np.random.choice(claim_statuses, num_samples),
        'Claim_Submission_Channel': np.random.choice(['Online', 'Phone', 'Agent', 'Mail'], num_samples),
        'Prior_Claims': np.random.poisson(lam=0.8, size=num_samples),
        'Claimant_Income_USD': np.random.lognormal(mean=11, sigma=1.0, size=num_samples),
    }

    data = pd.DataFrame(df)

    # Introduce missing values
    for col in data.columns:
        if col != 'Fraud_Indicator':
            mask = np.random.rand(len(data)) < 0.1
            data.loc[mask, col] = np.nan

    # Handle extreme values
    data['Claim_Amount_USD'] = np.where(np.random.rand(len(data)) < 0.02, data['Claim_Amount_USD'] * 10, data['Claim_Amount_USD'])
    data['Claimant_Income_USD'] = np.where(np.random.rand(len(data)) < 0.02, data['Claimant_Income_USD'] * 3, data['Claimant_Income_USD'])

    return data

# Generate data
data = generate_insurance_data()
data2 = data.copy()

# Streamlit app
st.title("Insurance Fraud Detection Dashboard")

# Display raw data
st.header("Raw Data")
st.write(data.head())

# Data Preprocessing
st.header("Data Preprocessing")
# Handle missing values
numerical_cols = data.select_dtypes(include=['number']).columns
for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna('missing', inplace=True)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Encode target column
data['Fraud_Indicator'] = LabelEncoder().fit_transform(data['Fraud_Indicator'])

# Prepare train-test split
X = data.drop('Fraud_Indicator', axis=1)
y = data['Fraud_Indicator']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
st.header("Model Training and Evaluation")

# Before SMOTE
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_before = accuracy_score(y_test, y_pred)
f1_before = f1_score(y_test, y_pred, average='weighted')

st.subheader("Before SMOTE")
st.write("Accuracy:", accuracy_before)
st.write("F1 Score:", f1_before)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# After SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model_resampled = RandomForestClassifier(random_state=42)
model_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = model_resampled.predict(X_test)

accuracy_after = accuracy_score(y_test, y_pred_resampled)
f1_after = f1_score(y_test, y_pred_resampled, average='weighted')

st.subheader("After SMOTE")
st.write("Accuracy:", accuracy_after)
st.write("F1 Score:", f1_after)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_resampled))

# Comparison Table
st.header("Comparison Table")
comparison_data = {
    "Metric": ["Accuracy", "F1 Score"],
    "Before SMOTE": [accuracy_before, f1_before],
    "After SMOTE": [accuracy_after, f1_after]
}
st.dataframe(pd.DataFrame(comparison_data))

# Comparison Bar Chart
fig_comparison = px.bar(pd.DataFrame(comparison_data).melt(id_vars="Metric"),
                        x="Metric", y="value", color="variable",
                        title="Performance Comparison Before and After SMOTE")
st.plotly_chart(fig_comparison, use_container_width=True)
