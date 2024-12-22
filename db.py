# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
raw_data = pd.read_csv("synthetic_insurance_data.csv")
preprocessed_data = pd.read_csv("synthetic_insurance_data2.csv")

# Page configuration
st.set_page_config(page_title="Insurance Data Comparison", layout="wide")
st.title("Synthetic Insurance Data Comparison Dashboard")

# Sidebar
st.sidebar.header("Navigation")
view = st.sidebar.radio("Select a view:", ["Value Distributions", "Evaluation Metrics"])

if view == "Value Distributions":
    st.header("Value Distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Data Distribution")
        raw_column = st.selectbox("Select column to visualize (Raw Data):", raw_data.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(raw_data[raw_column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {raw_column} (Raw Data)")
        st.pyplot(fig)

    with col2:
        st.subheader("Preprocessed Data Distribution")
        preprocessed_column = st.selectbox("Select column to visualize (Preprocessed Data):", preprocessed_data.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(preprocessed_data[preprocessed_column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {preprocessed_column} (Preprocessed Data)")
        st.pyplot(fig)

elif view == "Evaluation Metrics":
    st.header("Evaluation Metrics Comparison")

    # Hardcoded evaluation metrics
    metrics_before = {
        "Accuracy": 1.0,
        "Precision (Class 0)": 1.0,
        "Precision (Class 1)": 1.0,
        "Recall (Class 0)": 1.0,
        "Recall (Class 1)": 1.0,
        "F1-Score (Class 0)": 1.0,
        "F1-Score (Class 1)": 1.0
    }

    metrics_after = {
        "Accuracy": 0.597375,
        "Precision (Class no)": 0.59,
        "Precision (Class yes)": 0.62,
        "Recall (Class no)": 0.75,
        "Recall (Class yes)": 0.44,
        "F1-Score (Class no)": 0.66,
        "F1-Score (Class yes)": 0.51
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Metrics Before Preprocessing")
        st.table(metrics_before)

    with col2:
        st.subheader("Metrics After Preprocessing")
        st.table(metrics_after)

    # Visualization of metrics
    st.subheader("Metrics Comparison Visualization")
    metrics_df = pd.DataFrame({
        "Metric": list(metrics_before.keys()),
        "Before Preprocessing": list(metrics_before.values()),
        "After Preprocessing": list(metrics_after.values())
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.set_index("Metric").plot(kind="bar", ax=ax)
    ax.set_title("Comparison of Evaluation Metrics")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# Requirements.txt
# pandas
# streamlit
# matplotlib
# seaborn
