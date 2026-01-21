import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
import os

# === Page Config ===
st.set_page_config(page_title="AI Fraud Detector", layout="wide")
st.title("🛡️ AI-Powered Insurance Fraud Detector")
st.markdown("Upload your claims CSV → Detect fraud using rules + ML → Filter, visualize & export suspicious claims")

# === File Uploader ===
uploaded_file = st.file_uploader("Upload your claims CSV file", type="csv", 
                                 help="Must contain columns: claim_id, customer_id, claim_date, claim_amount")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    
    st.success(f"Loaded {len(df)} claims successfully!")
    
    # === RULE-BASED DETECTION ===
    today = datetime.now()
    df['days_since_claim'] = (today - df['claim_date']).dt.days
    
    # 1. Too many claims in last 6 months
    recent_claims = df[df['days_since_claim'] <= 180]
    claim_counts = recent_claims['customer_id'].value_counts()
    high_frequency_customers = claim_counts[claim_counts > 3].index.tolist()
    
    # 2. High / Very high amounts
    high_amount_threshold = df['claim_amount'].quantile(0.95)
    very_high_threshold = 500000
    
    # 3. Claims too close
    df = df.sort_values(['customer_id', 'claim_date'])
    df['time_diff_days'] = df.groupby('customer_id')['claim_date'].diff().dt.days
    close_claims_indices = df[(df['time_diff_days'] <= 7) & (df['time_diff_days'] > 0)].index
    
    # 4. NEW: Duplicate claim amounts per customer
    duplicate_amounts = df.groupby('customer_id')['claim_amount'].apply(lambda x: x.duplicated().any())
    duplicate_customers = duplicate_amounts[duplicate_amounts].index.tolist()
    
    # 5. NEW: Same day-of-month pattern (at least 3 claims on same day of month)
    df['day_of_month'] = df['claim_date'].dt.day
    day_patterns = df.groupby(['customer_id', 'day_of_month']).size()
    suspicious_patterns = day_patterns[day_patterns >= 3].index.get_level_values(0).unique()
    
    # Apply rules
    df['reason'] = "OK"
    df['rule_risk'] = 0
    df['suspicious_rule'] = False
    
    for idx, row in df.iterrows():
        reasons = []
        score = 0
        
        if row['customer_id'] in high_frequency_customers:
            reasons.append("Too many claims")
            score += 30
            df.at[idx, 'suspicious_rule'] = True
        
        if row['claim_amount'] > very_high_threshold:
            reasons.append("Very high amount (over 500k)")
            score += 50
            df.at[idx, 'suspicious_rule'] = True
        elif row['claim_amount'] > high_amount_threshold:
            reasons.append("High amount")
            score += 40
            df.at[idx, 'suspicious_rule'] = True
        
        if idx in close_claims_indices:
            reasons.append("Claims too close")
            score += 30
            df.at[idx, 'suspicious_rule'] = True
        
        if row['customer_id'] in duplicate_customers:
            reasons.append("Duplicate amount")
            score += 35
            df.at[idx, 'suspicious_rule'] = True
        
        if row['customer_id'] in suspicious_patterns:
            reasons.append("Same day-of-month pattern")
            score += 35
            df.at[idx, 'suspicious_rule'] = True
        
        if reasons:
            df.at[idx, 'reason'] = " + ".join(reasons)
            df.at[idx, 'rule_risk'] = min(score, 100)
    
    # === ML: Isolation Forest (with save/load) ===
    model_path = "fraud_model.joblib"
    
    features = df[['claim_amount', 'days_since_claim']].copy()
    features['time_diff_days'] = df['time_diff_days'].fillna(999)
    
    if os.path.exists(model_path):
        st.info("Loading saved ML model...")
        iso_forest = joblib.load(model_path)
    else:
        st.info("Training new ML model...")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(features)
        joblib.dump(iso_forest, model_path)
        st.success("ML model trained & saved for future use!")
    
    df['ml_anomaly'] = iso_forest.predict(features)
    df['ml_score'] = iso_forest.decision_function(features)
    
    min_s = df['ml_score'].min()
    max_s = df['ml_score'].max()
    df['ml_risk'] = 100 * (max_s - df['ml_score']) / (max_s - min_s + 1e-10)
    df['ml_risk'] = df['ml_risk'].round(1)
    
    df['final_risk_score'] = df[['rule_risk', 'ml_risk']].max(axis=1).round(1)
    df['suspicious'] = (df['suspicious_rule']) | (df['ml_anomaly'] == -1)
    
    # === Color styling function ===
    def highlight_suspicious(row):
        return ['background-color: #ff6666' if row['suspicious'] else 'background-color: #66cc66' for _ in row]

    # === FILTERS ===
    col1, col2 = st.columns(2)
    with col1:
        min_risk = st.slider("Minimum Risk Score", 0, 100, 0, step=5)
    with col2:
        show_only_suspicious = st.checkbox("Show Suspicious Claims Only", value=False)
    
    # Filtered dataframe
    filtered_df = df.copy()
    if show_only_suspicious:
        filtered_df = filtered_df[filtered_df['suspicious']]
    filtered_df = filtered_df[filtered_df['final_risk_score'] >= min_risk]
    
    # Display table
    st.subheader("Fraud Detection Results")
    st.dataframe(filtered_df.style.apply(highlight_suspicious, axis=1), use_container_width=True)
    
    # === CHARTS ===
    st.subheader("Visual Insights")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        colors = ['red' if x else 'green' for x in df['suspicious']]
        ax_bar.bar(df.index.astype(str), df['final_risk_score'], color=colors)
        ax_bar.set_title('Fraud Risk Score by Claim')
        ax_bar.set_xlabel('Claim Index')
        ax_bar.set_ylabel('Risk Score (0-100)')
        ax_bar.tick_params(axis='x', rotation=90)
        st.pyplot(fig_bar)
    
    with col_chart2:
        suspicious_count = df['suspicious'].sum()
        normal_count = len(df) - suspicious_count
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        ax_pie.pie([suspicious_count, normal_count], labels=['Suspicious', 'Normal'], 
                   autopct='%1.1f%%', colors=['#ff6666', '#66cc66'])
        ax_pie.set_title('Suspicious vs Normal Claims')
        st.pyplot(fig_pie)
    
    # === EXPORT ===
    if suspicious_count > 0:
        csv = df[df['suspicious']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Suspicious Claims CSV",
            data=csv,
            file_name="suspicious_claims.csv",
            mime="text/csv"
        )
    else:
        st.info("No suspicious claims detected!")
