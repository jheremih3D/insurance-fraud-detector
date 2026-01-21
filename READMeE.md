# AI-Powered Insurance Fraud Detector

This is a web app built with Streamlit that detects potential fraud in insurance claims using rule-based checks + machine learning.

Features:
- Upload CSV with claims data
- Flags suspicious claims (high amounts, duplicates, patterns, etc.)
- Uses Isolation Forest ML model
- Interactive filters, charts, and export

Try it live: [Link will appear after deployment]

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
