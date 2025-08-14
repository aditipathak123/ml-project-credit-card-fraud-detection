import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------
# Load Data
# ----------------------
pred_file = "eda/data/X_test_eval.csv"
feat_file = "eda/data/feat_imp_df.csv"

X_test_eval = pd.read_csv(pred_file)
feat_imp_df = pd.read_csv(feat_file)

st.set_page_config(page_title="💳 FraudShield Advanced Dashboard", layout="wide")
st.title("💳 FraudShield Advanced Dashboard")

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.header("Filters & Settings")
threshold = st.sidebar.slider("Set Fraud Probability Threshold:", 0.0, 1.0, 0.5)
feature_choice = st.sidebar.selectbox("Select Feature for Analysis:", X_test_eval.columns)

# Highlight high-risk transactions
high_risk = X_test_eval[X_test_eval['Fraud_Prob'] >= threshold]

# ----------------------
# Tabs
# ----------------------
tabs = st.tabs(["Predictions", "Fraud Distribution", "Feature Importances", "Feature Analysis"])

# ----------------------
# Tab 1: Predictions
# ----------------------
with tabs[0]:
    st.subheader("📊 Test Set Predictions")
    st.dataframe(X_test_eval.head(20))
    
    st.subheader(f"⚠️ High-Risk Transactions (Threshold ≥ {threshold})")
    def highlight_risk(row):
        return ['background-color: #FFCCCC' if row['Fraud_Prob'] >= threshold else '' for _ in row]
    st.dataframe(high_risk.style.apply(highlight_risk, axis=1))

    csv = high_risk.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download High-Risk Transactions CSV",
        data=csv,
        file_name='high_risk_predictions.csv',
        mime='text/csv'
    )

# ----------------------
# Tab 2: Fraud Distribution
# ----------------------
with tabs[1]:
    st.subheader("🔴 Fraud Probability Distribution")
    plt.figure(figsize=(8,4))
    bins = np.linspace(0,1,50)
    sns.histplot(X_test_eval['Fraud_Prob'], bins=bins, kde=True, color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f"Threshold={threshold}")
    plt.xlabel("Fraud Probability")
    plt.ylabel("Count")
    plt.title("Fraud Probability Distribution")
    plt.legend()
    st.pyplot(plt.gcf())

# ----------------------
# Tab 3: Feature Importances
# ----------------------
with tabs[2]:
    st.subheader("🌟 Top 15 Feature Importances")
    top15 = feat_imp_df.head(15)
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=top15, palette='viridis')
    plt.title("Top 15 Feature Importances")
    st.pyplot(plt.gcf())
    
    st.subheader("📋 Full Feature Importance Table")
    st.dataframe(feat_imp_df)

# ----------------------
# Tab 4: Feature Analysis
# ----------------------
with tabs[3]:
    st.subheader(f"📈 Distribution of {feature_choice}")
    fig, ax = plt.subplots(1,2, figsize=(15,4))
    
    sns.histplot(X_test_eval[feature_choice], bins=50, kde=True, color='blue', ax=ax[0])
    ax[0].set_title(f"{feature_choice} Histogram")
    
    sns.boxplot(x=X_test_eval[feature_choice], color='orange', ax=ax[1])
    ax[1].set_title(f"{feature_choice} Boxplot")
    
    st.pyplot(fig)
    
    st.write("Statistics:")
    st.write(X_test_eval[feature_choice].describe())
