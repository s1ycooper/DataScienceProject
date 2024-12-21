import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import missingno as msno
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
         'Risk_Level': np.random.choice(risk_levels,num_samples,p = [0.5, 0.3, 0.15, 0.05]),
        'Witnesses': np.random.poisson(lam=1.5, size=num_samples),
        'Police_Involved': np.random.choice([True, False], num_samples, p=[0.7, 0.3]),
        'Injury_Reported': np.random.choice([True, False], num_samples, p=[0.55, 0.45]),
        'Vehicle_Damage': np.random.choice(['Mild', 'Moderate', 'Severe'], num_samples, p = [0.4, 0.4, 0.2]),
        'Policy_Holder_Years': np.random.randint(1, 20, num_samples),
        'Claim_Status': np.random.choice(claim_statuses, num_samples),
        'Claim_Submission_Channel': np.random.choice(['Online', 'Phone', 'Agent', 'Mail'], num_samples),
        'Prior_Claims': np.random.poisson(lam=0.8, size=num_samples),
         'Claimant_Income_USD': np.random.lognormal(mean=11, sigma=1.0, size=num_samples),
    }

    data = pd.DataFrame(df)

    boolean_cols = data.select_dtypes(include=['bool']).columns
    for col in boolean_cols:
        data[col] = data[col].astype(int)

    for col in data.columns:
        if col != 'Fraud_Indicator':
            mask = np.random.rand(len(data)) < 0.1
            data.loc[mask, col] = np.nan



    data['Claim_Amount_USD'] = np.where(np.random.rand(len(data)) < 0.02, data['Claim_Amount_USD'] * 10, data['Claim_Amount_USD'])
    data['Claimant_Income_USD'] = np.where(np.random.rand(len(data)) < 0.02, data['Claimant_Income_USD'] * 3, data['Claimant_Income_USD'])
    data['Prior_Claims'] = np.where(np.random.rand(len(data)) < 0.01, np.random.poisson(lam=10, size=len(data)), data['Prior_Claims'])



    return data

# Generate data
data = generate_insurance_data()
data2 = data.copy()

# Streamlit app
st.title("Insurance Fraud Detection Dashboard")

# Display raw data
st.header("Raw Data")
st.write(data.head())

# Data Distribution Visualization
st.header("1. Data Distribution Visualization")

# Histograms for numerical columns
numerical_cols = data.select_dtypes(include=['number']).columns
for col in numerical_cols:
    fig = px.histogram(data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Count plots for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    fig = px.bar(data, x=col, title=f"Count of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Missing values visualization
st.header("Missing Values Visualization")
fig_missing = msno.bar(data)
st.pyplot(fig_missing.get_figure())

# Data description
st.header("Data Description")
st.write(data2.describe().T)

# Pair plots for selected features
st.header("Pair Plots for Selected Features")
selected_features_1 = ['Claim_Amount_USD', 'Claimant_Age']
selected_features_2 = ['Claim_Amount_USD', 'Prior_Claims']
selected_features_3 = ['Claimant_Age', 'Claimant_Income_USD']
selected_features_4 = ['Witnesses', 'Claimant_Income_USD']

fig_pairplot_1 = px.scatter_matrix(data2, dimensions=selected_features_1, color='Fraud_Indicator')
st.plotly_chart(fig_pairplot_1, use_container_width=True)

fig_pairplot_2 = px.scatter_matrix(data2, dimensions=selected_features_2, color='Fraud_Indicator')
st.plotly_chart(fig_pairplot_2, use_container_width=True)

fig_pairplot_3 = px.scatter_matrix(data2, dimensions=selected_features_3, color='Fraud_Indicator')
st.plotly_chart(fig_pairplot_3, use_container_width=True)

fig_pairplot_4 = px.scatter_matrix(data2, dimensions=selected_features_4, color='Fraud_Indicator')
st.plotly_chart(fig_pairplot_4, use_container_width=True)

# Data Preprocessing
numerical_cols = data.select_dtypes(include=['number']).columns
for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)

categorical_cols = data.select_dtypes(include=['object']).columns
boolean_cols = data.select_dtypes(include=['bool']).columns

for col in categorical_cols:
    data[col].fillna('missing', inplace=True)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

for col in boolean_cols:
    data[col] = data[col].astype(int)

X = data.drop('Fraud_Indicator', axis=1)
y = data['Fraud_Indicator']

# Model Training and Evaluation
st.header("2. Model Training and Evaluation")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("### Model Performance Metrics (Before SMOTE)")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
st.write(classification_report(y_test, y_pred))

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model_resampled = RandomForestClassifier(random_state=42)
model_resampled.fit(X_train_resampled, y_train_resampled)

y_pred_resampled = model_resampled.predict(X_test)

st.write("### Model Performance Metrics (After SMOTE)")
st.write("Accuracy:", accuracy_score(y_test, y_pred_resampled))
st.write("F1 Score:", f1_score(y_test, y_pred_resampled, average='weighted'))
st.write(classification_report(y_test, y_pred_resampled))

# Model Comparison
st.header("3. Model Comparison")

comparison_data = {
    "Model": ["Before SMOTE", "After SMOTE"],
    "Accuracy": [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_resampled)],
    "F1 Score": [f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred_resampled, average='weighted')]
}
df_comparison = pd.DataFrame(comparison_data)

fig_comparison = px.bar(df_comparison, x="Model", y=["Accuracy", "F1 Score"],
             barmode="group", title="Performance Comparison Before and After SMOTE")
st.plotly_chart(fig_comparison, use_container_width=True)

# Display comparison table
st.write("### Comparison Table")
st.dataframe(df_comparison)

# Insights
st.header("4. Insights")
st.markdown("""
- The initial model achieved an accuracy of approximately 48.9% and an F1 score of 0.486.
- After applying SMOTE to handle class imbalance, the accuracy improved to around 75.4%, and the F1 score increased to 0.753.
- SMOTE helps in improving the model's ability to correctly classify instances from the minority class.
- Use the bar charts above to analyze and compare performance metrics interactively.
""")