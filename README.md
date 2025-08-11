# Financial Fraud Detection System

A comprehensive, world-class fraud detection system that uses advanced machine learning, statistical analysis, and AI techniques to identify potentially fraudulent transactions.

## Features

### 🛡️ Multi-Layer Detection
- **Unsupervised Learning**: Isolation Forest, Local Outlier Factor, Autoencoders, One-Class SVM
- **Supervised Learning**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Rule-Based Detection**: Configurable rules with weights for known fraud patterns

### 🔍 Advanced Feature Engineering
- **Statistical Features**: Benford's Law, Z-score, MAD, percentile analysis
- **Graph Features**: Centrality measures, clustering coefficients, community detection
- **NLP Features**: Sentiment analysis, keyword detection, pattern matching
- **Time Series Features**: Burstiness analysis, gap analysis, temporal patterns

### 📊 Intelligent Analysis
- **Risk Scoring**: Weighted combination of algorithm outputs
- **Explainable AI**: SHAP values and natural language explanations
- **Dynamic Thresholds**: Percentile-based and adaptive thresholds
- **Real-time Monitoring**: Continuous learning from new data

### 📈 Professional Reporting
- **Executive Summary**: High-level overview for stakeholders
- **Detailed Analysis**: Comprehensive forensic audit reports
- **Technical Reports**: Model performance and system metrics
- **Custom Reports**: User-configurable report generation

### 🌐 User-Friendly Interface
- **Streamlit Dashboard**: Interactive web-based interface
- **Data Upload**: Support for CSV, Excel files up to 10GB
- **Column Mapping**: AI-powered column matching with confirmation
- **Visualization**: Interactive charts and graphs

## Installation

### Option 1: Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

2. Create and activate the conda environment:
conda env create -f environment.yml
conda activate fraud-detection

3.Install additional packages:
pip install -r requirements.txt



Usage
Running the Application

1. Start the Streamlit application:
streamlit run app/main_dashboard.py

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)


Expected Data Format
The system expects transaction data with the following columns:

transaction_id,timestamp,amount,currency,sender_id,receiver_id,sender_account_type,receiver_account_type,sender_bank,receiver_bank,sender_location,receiver_location,transaction_type,transaction_category,merchant_id,merchant_category,ip_address,device_id,description,notes,authorization_status,chargeback_flag,fraud_flag



Data Types:
transaction_id: string/integer (unique identifier)
timestamp: datetime (ISO format: YYYY-MM-DD HH:MM:SS)
amount: float (transaction value)
currency: string (3-letter ISO code)
sender_id: string (entity initiating transaction)
receiver_id: string (entity receiving transaction)
sender_account_type: string (personal, business, corporate, etc.)
receiver_account_type: string (personal, business, corporate, etc.)
sender_bank: string (name of sender's bank)
receiver_bank: string (name of receiver's bank)
sender_location: string (country, state, city)
receiver_location: string (country, state, city)
transaction_type: string (transfer, payment, withdrawal, deposit)
transaction_category: string (retail, services, gambling, etc.)
merchant_id: string (if applicable)
merchant_category: string (if applicable)
ip_address: string (IPv4 or IPv6)
device_id: string (unique device identifier)
description: string (transaction description)
notes: string (additional notes)
authorization_status: string (approved, declined, pending)
chargeback_flag: boolean (True/False)
fraud_flag: boolean (True/False, for supervised learning)