

"""
Main Dashboard for Financial Fraud Detection System
Streamlit-based user interface for the fraud detection platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection_engine.ingestion.data_loader import DataLoader
from fraud_detection_engine.ingestion.column_mapper import ColumnMapper
from fraud_detection_engine.features.statistical_features import StatisticalFeatures
from fraud_detection_engine.features.graph_features import GraphFeatures
from fraud_detection_engine.features.nlp_features import NLPFeatures
from fraud_detection_engine.features.timeseries_features import TimeSeriesFeatures
from fraud_detection_engine.models.unsupervised import UnsupervisedModels
from fraud_detection_engine.models.supervised import SupervisedModels
from fraud_detection_engine.models.rule_based import RuleEngine
from fraud_detection_engine.analysis.risk_scorer import RiskScorer
from fraud_detection_engine.analysis.explainability import Explainability
from fraud_detection_engine.reporting.pdf_generator import PDFGenerator
from fraud_detection_engine.utils.api_utils import is_api_available

# Set page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML files"""
    config = {
        'model_params': {},
        'rule_engine': {},
        'api_keys': {}
    }
    
    try:
        # Try to load model_params.yml
        try:
            with open('config/model_params.yml', 'r') as f:
                config['model_params'] = yaml.safe_load(f)
        except FileNotFoundError:
            # Create default model_params
            config['model_params'] = {
                'unsupervised': {
                    'isolation_forest': {
                        'n_estimators': 100,
                        'contamination': 0.01,
                        'random_state': 42
                    }
                }
            }
        
        # Try to load rule_engine_config.yml
        try:
            with open('config/rule_engine_config.yml', 'r') as f:
                config['rule_engine'] = yaml.safe_load(f)
        except FileNotFoundError:
            # Create default rule_engine config
            config['rule_engine'] = {
                'rules': {
                    'high_amount': {
                        'enabled': True,
                        'threshold': 10000,
                        'weight': 0.3
                    }
                }
            }
        
        # Try to load api_keys.yml
        try:
            with open('config/api_keys.yml', 'r') as f:
                config['api_keys'] = yaml.safe_load(f)
        except FileNotFoundError:
            # Create default api_keys
            config['api_keys'] = {
                'gemini': {'api_key': 'NOT_AVAILABLE'},
                'openai': {'api_key': 'NOT_AVAILABLE'},
                'news_api': {'api_key': 'NOT_AVAILABLE'},
                'geolocation': {'api_key': 'NOT_AVAILABLE'},
                'sanctions': {'api_key': 'NOT_AVAILABLE'},
                'tax_compliance': {'api_key': 'NOT_AVAILABLE'},
                'bank_verification': {'api_key': 'NOT_AVAILABLE'},
                'identity_verification': {'api_key': 'NOT_AVAILABLE'}
            }
            
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
    
    return config

config = load_config()

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'risk_scores' not in st.session_state:
    st.session_state.risk_scores = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'explanations' not in st.session_state:
    st.session_state.explanations = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Sidebar
def render_sidebar():
    """Render the sidebar with navigation and controls"""
    st.sidebar.title("🛡️ Fraud Detection System")
    st.sidebar.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    
    # Navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Upload", 
        "Column Mapping", 
        "Analysis Settings", 
        "Run Detection", 
        "Results Dashboard", 
        "Explainability", 
        "Reports"
    ])
    
    # System status
    st.sidebar.subheader("System Status")
    if st.session_state.data is not None:
        st.sidebar.success(f"Data loaded: {len(st.session_state.data)} transactions")
    else:
        st.sidebar.warning("No data loaded")
    
    if st.session_state.processing_complete:
        st.sidebar.success("Analysis complete")
    else:
        st.sidebar.info("Analysis pending")
    
    # API Status
    st.sidebar.subheader("API Status")
    api_status = {
        'Gemini': is_api_available('gemini'),
        'OpenAI': is_api_available('openai'),
        'News API': is_api_available('news_api'),
        'Geolocation': is_api_available('geolocation'),
        'Sanctions': is_api_available('sanctions'),
        'Tax Compliance': is_api_available('tax_compliance'),
        'Bank Verification': is_api_available('bank_verification'),
        'Identity Verification': is_api_available('identity_verification')
    }
    
    for api, status in api_status.items():
        if status:
            st.sidebar.success(f"{api}: ✅ Available")
        else:
            st.sidebar.warning(f"{api}: ❌ Not Available")
    
    # Configuration
    st.sidebar.subheader("Configuration")
    if st.sidebar.button("Reload Config"):
        config = load_config()
        st.sidebar.success("Configuration reloaded")
    
    # About section
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This is a comprehensive financial fraud detection system that uses advanced machine learning, statistical analysis, and AI techniques to identify potentially fraudulent transactions.
    """)
    
    return page

# Data Upload Page
def render_data_upload():
    """Render the data upload page"""
    st.title("📁 Data Upload")
    st.markdown("""
    Upload your transaction data for analysis. The system supports CSV and Excel files.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing transaction data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display data info
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            st.write("### Data Preview")
            st.dataframe(df.head(10))
            
            # Store in session state
            st.session_state.data = df
            
            # Show column information
            st.write("### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info)
            
            # Basic statistics
            st.write("### Basic Statistics")
            st.dataframe(df.describe(include='all'))
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Sample data option
    st.markdown("---")
    st.subheader("Or Use Sample Data")
    if st.button("Load Sample Data"):
        try:
            sample_path = os.path.join('..', 'data', 'sample_transactions.csv')
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                st.session_state.data = df
                st.success(f"Sample data loaded! Shape: {df.shape}")
                st.dataframe(df.head(10))
            else:
                st.warning("Sample data file not found")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

# Column Mapping Page
def render_column_mapping():
    """Render the column mapping page"""
    st.title("🔄 Column Mapping")
    
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
    
    st.markdown("""
    Map your column headers to the expected format. The system will attempt to auto-map columns,
    but you can review and adjust as needed.
    """)
    
    # Initialize column mapper
    mapper = ColumnMapper()
    
    # Get expected columns
    expected_columns = mapper.get_expected_columns()
    
    # Get actual columns
    actual_columns = list(st.session_state.data.columns)
    
    # Auto-map columns
    if st.session_state.column_mapping is None:
        with st.spinner("Auto-mapping columns..."):
            st.session_state.column_mapping = mapper.auto_map_columns(actual_columns, expected_columns)
    
    # Display mapping
    st.write("### Column Mapping")
    
    # Create a DataFrame for the mapping
    mapping_data = []
    for actual_col in actual_columns:
        expected_col = st.session_state.column_mapping.get(actual_col, "Not mapped")
        mapping_data.append({
            "Your Column": actual_col,
            "Maps To": expected_col
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    st.dataframe(mapping_df)
    
    # Allow manual adjustment
    st.write("### Adjust Mapping")
    adjusted_mapping = {}
    
    for actual_col in actual_columns:
        current_mapping = st.session_state.column_mapping.get(actual_col, "Not mapped")
        options = ["Not mapped"] + expected_columns
        selected = st.selectbox(
            f"Map '{actual_col}' to:",
            options=options,
            index=options.index(current_mapping) if current_mapping in options else 0,
            key=f"map_{actual_col}"
        )
        
        if selected != "Not mapped":
            adjusted_mapping[actual_col] = selected
    
    # Update button
    if st.button("Update Mapping"):
        st.session_state.column_mapping = adjusted_mapping
        st.success("Column mapping updated!")
    
    # Show mapped columns preview
    if st.button("Preview Mapped Data"):
        try:
            mapped_data = mapper.apply_mapping(st.session_state.data, st.session_state.column_mapping)
            st.write("### Mapped Data Preview")
            st.dataframe(mapped_data.head(10))
        except Exception as e:
            st.error(f"Error applying mapping: {str(e)}")
    
    # Continue to analysis
    if st.button("Continue to Analysis"):
        try:
            mapped_data = mapper.apply_mapping(st.session_state.data, st.session_state.column_mapping)
            st.session_state.processed_data = mapped_data
            st.success("Data processed successfully! You can now configure analysis settings.")
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

def validate_features(df):
    """
    Validate and clean feature DataFrame:
    1. Replace infinite values with NaNs
    2. Handle extremely large values
    3. Impute missing values
    """
    df = df.copy()
    
    # 1. Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Handle extremely large values (>1e10)
    large_value_cols = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            max_val = np.nanmax(np.abs(df[col]))
            if max_val > 1e10:
                large_value_cols.append(col)
                # Apply log transformation to positive values
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
                else:
                    # Cap extreme values for columns with negatives
                    df[col] = np.clip(df[col], -1e10, 1e10)
    
    # 3. Impute missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
    
    # Log any transformations
    if large_value_cols:
        print(f"Applied transformations to large-value columns: {large_value_cols}")
    if df.isna().sum().sum() > 0:
        print(f"Imputed {df.isna().sum().sum()} missing values")
    
    return df

# Analysis Settings Page
def render_analysis_settings():
    """Render the analysis settings page"""
    st.title("⚙️ Analysis Settings")
    
    if st.session_state.processed_data is None:
        st.warning("Please process data first")
        return
    
    st.markdown("""
    Configure the fraud detection analysis settings. You can select which algorithms to run,
    adjust parameters, and set thresholds.
    """)
    
    # Feature engineering settings
    st.subheader("Feature Engineering")
    
    feature_options = {
        "Statistical Features": True,
        "Graph Features": True,
        "NLP Features": True,
        "Time Series Features": True
    }
    
    selected_features = {}
    for feature, default in feature_options.items():
        selected_features[feature] = st.checkbox(feature, value=default)
    
    # Model settings
    st.subheader("Model Selection")
    
    model_options = {
        "Unsupervised Models": True,
        "Supervised Models": True,
        "Rule-Based Models": True
    }
    
    selected_models = {}
    for model, default in model_options.items():
        selected_models[model] = st.checkbox(model, value=default)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        st.write("### Unsupervised Model Parameters")
        
        contamination = st.slider(
            "Contamination (expected fraud rate)",
            min_value=0.001,
            max_value=0.5,
            value=0.01,
            step=0.001,
            help="Expected proportion of outliers in the data"
        )
        
        st.write("### Supervised Model Parameters")
        
        if 'fraud_flag' in st.session_state.processed_data.columns:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Proportion of data to use for testing"
            )
        else:
            st.info("No fraud_flag column found. Supervised learning will use synthetic labels.")
            test_size = 0.2
        
        st.write("### Rule Engine Parameters")
        
        rule_threshold = st.slider(
            "Rule Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Threshold for rule-based detection"
        )
    
    # Risk scoring settings
    st.subheader("Risk Scoring")
    
    scoring_method = st.selectbox(
        "Scoring Method",
        ["Weighted Average", "Maximum Score", "Custom Weights"],
        help="Method to combine scores from different models"
    )
    
    if scoring_method == "Custom Weights":
        st.write("### Custom Weights")
        unsupervised_weight = st.slider("Unsupervised Weight", 0.0, 1.0, 0.4, 0.05)
        supervised_weight = st.slider("Supervised Weight", 0.0, 1.0, 0.4, 0.05)
        rule_weight = st.slider("Rule Weight", 0.0, 1.0, 0.2, 0.05)
        
        # Normalize weights
        total = unsupervised_weight + supervised_weight + rule_weight
        if total > 0:
            unsupervised_weight /= total
            supervised_weight /= total
            rule_weight /= total
        
        custom_weights = {
            "unsupervised": unsupervised_weight,
            "supervised": supervised_weight,
            "rule": rule_weight
        }
    else:
        custom_weights = None
    
    # Threshold settings
    st.subheader("Detection Thresholds")
    
    threshold_method = st.selectbox(
        "Threshold Method",
        ["Fixed", "Percentile", "Dynamic"],
        help="Method to determine fraud threshold"
    )
    
    if threshold_method == "Fixed":
        threshold = st.slider(
            "Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Transactions above this threshold will be flagged as potential fraud"
        )
    elif threshold_method == "Percentile":
        percentile = st.slider(
            "Percentile",
            min_value=90,
            max_value=100,
            value=95,
            step=1,
            help="Transactions above this percentile of risk scores will be flagged"
        )
        threshold = None
    else:  # Dynamic
        st.info("Dynamic thresholds will be calculated based on data distribution")
        threshold = None
    
    # Save settings
    if st.button("Save Settings"):
        settings = {
            "features": selected_features,
            "models": selected_models,
            "contamination": contamination,
            "test_size": test_size,
            "rule_threshold": rule_threshold,
            "scoring_method": scoring_method,
            "custom_weights": custom_weights,
            "threshold_method": threshold_method,
            "threshold": threshold,
            "percentile": percentile if threshold_method == "Percentile" else None
        }
        
        st.session_state.settings = settings
        st.success("Settings saved! You can now run the detection analysis.")

# Run Detection Page
def render_run_detection():
    """Render the run detection page"""
    st.title("🚀 Run Fraud Detection")
    
    if st.session_state.processed_data is None:
        st.warning("Please process data first")
        return
    
    if 'settings' not in st.session_state:
        st.warning("Please configure analysis settings first")
        return
    
    st.markdown("""
    Run the fraud detection analysis using the configured settings. This may take some time
    depending on the size of your dataset and selected algorithms.
    """)
    
    # Show settings summary
    st.subheader("Analysis Settings Summary")
    
    settings = st.session_state.settings
    
    st.write("#### Feature Engineering")
    for feature, enabled in settings["features"].items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        st.write(f"- {feature}: {status}")
    
    st.write("#### Models")
    for model, enabled in settings["models"].items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        st.write(f"- {model}: {status}")
    
    st.write("#### Risk Scoring")
    st.write(f"- Method: {settings['scoring_method']}")
    if settings['custom_weights']:
        st.write(f"- Unsupervised Weight: {settings['custom_weights']['unsupervised']:.2f}")
        st.write(f"- Supervised Weight: {settings['custom_weights']['supervised']:.2f}")
        st.write(f"- Rule Weight: {settings['custom_weights']['rule']:.2f}")
    
    st.write("#### Threshold")
    st.write(f"- Method: {settings['threshold_method']}")
    if settings['threshold_method'] == 'Fixed':
        st.write(f"- Value: {settings['threshold']}")
    elif settings['threshold_method'] == 'Percentile':
        st.write(f"- Percentile: {settings['percentile']}%")
    
    # Run button
    if st.button("Run Detection Analysis"):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Feature Engineering
            status_text.text("Step 1/5: Feature Engineering...")
            progress_bar.progress(10)
            
            features_df = st.session_state.processed_data.copy()
            
            # Statistical features
            if settings["features"]["Statistical Features"]:
                stat_features = StatisticalFeatures()
                features_df = stat_features.extract_features(features_df)
                progress_bar.progress(20)
            
            # Graph features
            if settings["features"]["Graph Features"]:
                graph_features = GraphFeatures()
                features_df = graph_features.extract_features(features_df)
                progress_bar.progress(30)
            
            # NLP features
            if settings["features"]["NLP Features"]:
                nlp_features = NLPFeatures()
                features_df = nlp_features.extract_features(features_df)
                progress_bar.progress(40)
            
            # Time series features
            if settings["features"]["Time Series Features"]:
                ts_features = TimeSeriesFeatures()
                features_df = ts_features.extract_features(features_df)
                progress_bar.progress(50)
            
            # Step 2: Model Training/Prediction
            status_text.text("Step 2/5: Running Models...")
            model_results = {}
            
            # Unsupervised models
            if settings["models"]["Unsupervised Models"]:
                unsupervised = UnsupervisedModels(contamination=settings["contamination"])
                unsupervised_results = unsupervised.run_models(features_df)
                model_results["unsupervised"] = unsupervised_results
                progress_bar.progress(60)
            
            # Supervised models
            if settings["models"]["Supervised Models"]:
                supervised = SupervisedModels(test_size=settings["test_size"])
                # In supervised model execution
                supervised_results = supervised.run_models(validate_features(features_df))
                model_results["supervised"] = supervised_results
                progress_bar.progress(70)
            
            # Rule-based models
            if settings["models"]["Rule-Based Models"]:
                rule_engine = RuleEngine(threshold=settings["rule_threshold"])
                rule_results = rule_engine.apply_rules(features_df)
                model_results["rule"] = rule_results
                progress_bar.progress(80)
            
            # Step 3: Risk Scoring
            status_text.text("Step 3/5: Calculating Risk Scores...")
            risk_scorer = RiskScorer(
                method=settings["scoring_method"],
                custom_weights=settings["custom_weights"]
            )
            risk_scores = risk_scorer.calculate_scores(model_results, features_df)
            progress_bar.progress(90)
            
            # Step 4: Apply Thresholds
            status_text.text("Step 4/5: Applying Thresholds...")
            if settings["threshold_method"] == "Fixed":
                threshold = settings["threshold"]
                risk_scores["is_fraud"] = risk_scores["risk_score"] > threshold
            elif settings["threshold_method"] == "Percentile":
                percentile = settings["percentile"]
                threshold = np.percentile(risk_scores["risk_score"], percentile)
                risk_scores["is_fraud"] = risk_scores["risk_score"] > threshold
            else:  # Dynamic
                # Calculate dynamic threshold based on distribution
                mean_score = risk_scores["risk_score"].mean()
                std_score = risk_scores["risk_score"].std()
                threshold = mean_score + 2 * std_score
                risk_scores["is_fraud"] = risk_scores["risk_score"] > threshold
            
            progress_bar.progress(95)
            
            # Step 5: Generate Explanations
            status_text.text("Step 5/5: Generating Explanations...")
            explainability = Explainability()
            explanations = explainability.generate_explanations(
                features_df, 
                risk_scores, 
                model_results
            )
            
            progress_bar.progress(100)
            
            # Store results in session state
            st.session_state.features_df = features_df
            st.session_state.model_results = model_results
            st.session_state.risk_scores = risk_scores
            st.session_state.explanations = explanations
            st.session_state.threshold = threshold
            st.session_state.processing_complete = True
            
            status_text.text("Analysis complete!")
            st.success(f"Fraud detection analysis complete! Found {risk_scores['is_fraud'].sum()} potentially fraudulent transactions out of {len(risk_scores)} total.")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)

# Results Dashboard Page
def render_results_dashboard():
    """Render the results dashboard page"""
    st.title("📊 Results Dashboard")
    
    if not st.session_state.processing_complete:
        st.warning("Please run the detection analysis first")
        return
    
    risk_scores = st.session_state.risk_scores
    features_df = st.session_state.features_df
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(risk_scores))
    
    with col2:
        fraud_count = risk_scores['is_fraud'].sum()
        st.metric("Fraudulent Transactions", fraud_count)
    
    with col3:
        fraud_percentage = (fraud_count / len(risk_scores)) * 100
        st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
    
    with col4:
        threshold = st.session_state.threshold
        st.metric("Detection Threshold", f"{threshold:.4f}")
    
    # Risk score distribution
    st.subheader("Risk Score Distribution")
    
    fig = px.histogram(
        risk_scores,
        x="risk_score",
        color="is_fraud",
        nbins=50,
        title="Distribution of Risk Scores",
        color_discrete_map={False: "blue", True: "red"},
        opacity=0.7
    )
    
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Threshold: {threshold:.4f}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top risky transactions
    st.subheader("Top Risky Transactions")
    
    top_risky = risk_scores[risk_scores['is_fraud']].nlargest(10, 'risk_score')
    
    if len(top_risky) > 0:
        # Merge with original data to show more details
        if 'transaction_id' in features_df.columns:
            top_risky_details = top_risky.merge(
                features_df[['transaction_id', 'amount', 'timestamp', 'sender_id', 'receiver_id']],
                left_index=True,
                right_index=True
            )
            
            st.dataframe(
                top_risky_details[[
                    'transaction_id', 'amount', 'timestamp', 'sender_id', 
                    'receiver_id', 'risk_score'
                ]].sort_values('risk_score', ascending=False)
            )
        else:
            st.dataframe(top_risky[['risk_score']].sort_values('risk_score', ascending=False))
    else:
        st.info("No fraudulent transactions detected")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if 'supervised' in st.session_state.model_results and 'feature_importance' in st.session_state.model_results['supervised']:
        feature_importance = st.session_state.model_results['supervised']['feature_importance']
        
        # Get top 20 features
        top_features = feature_importance.head(20)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Feature Importance",
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available. Run supervised models to see feature importance.")
    
    # Model performance
    st.subheader("Model Performance")
    
    if 'supervised' in st.session_state.model_results and 'performance' in st.session_state.model_results['supervised']:
        performance = st.session_state.model_results['supervised']['performance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Classification Report")
            st.dataframe(performance['classification_report'])
        
        with col2:
            st.write("##### Confusion Matrix")
            cm = performance['confusion_matrix']
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Fraud", "Fraud"],
                y=["Not Fraud", "Fraud"],
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model performance metrics not available. Run supervised models with labeled data to see performance metrics.")
    
    # Time series analysis
    if 'timestamp' in features_df.columns:
        st.subheader("Fraud Over Time")
        
        # Create a copy for time series analysis
        ts_data = features_df.copy()
        ts_data['risk_score'] = risk_scores['risk_score']
        ts_data['is_fraud'] = risk_scores['is_fraud']
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(ts_data['timestamp']):
            ts_data['timestamp'] = pd.to_datetime(ts_data['timestamp'])
        
        # Extract date components
        ts_data['date'] = ts_data['timestamp'].dt.date
        ts_data['hour'] = ts_data['timestamp'].dt.hour
        
        # Fraud by date
        fraud_by_date = ts_data.groupby('date').agg({
            'is_fraud': ['sum', 'count'],
            'risk_score': 'mean'
        }).reset_index()
        
        fraud_by_date.columns = ['date', 'fraud_count', 'total_count', 'avg_risk_score']
        fraud_by_date['fraud_rate'] = fraud_by_date['fraud_count'] / fraud_by_date['total_count']
        
        fig = px.line(
            fraud_by_date,
            x='date',
            y=['fraud_count', 'fraud_rate'],
            title="Fraud Count and Rate Over Time",
            labels={'value': 'Count / Rate', 'date': 'Date'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by hour of day
        fraud_by_hour = ts_data.groupby('hour').agg({
            'is_fraud': ['sum', 'count'],
            'risk_score': 'mean'
        }).reset_index()
        
        fraud_by_hour.columns = ['hour', 'fraud_count', 'total_count', 'avg_risk_score']
        fraud_by_hour['fraud_rate'] = fraud_by_hour['fraud_count'] / fraud_by_hour['total_count']
        
        fig = px.bar(
            fraud_by_hour,
            x='hour',
            y='fraud_count',
            title="Fraud Count by Hour of Day",
            color='avg_risk_score',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Explainability Page
def render_explainability():
    """Render the explainability page"""
    st.title("🔍 Explainability")
    
    if not st.session_state.processing_complete:
        st.warning("Please run the detection analysis first")
        return
    
    risk_scores = st.session_state.risk_scores
    explanations = st.session_state.explanations
    
    # Transaction selector
    st.subheader("Transaction Analysis")
    
    # Get fraudulent transactions
    fraud_transactions = risk_scores[risk_scores['is_fraud']]
    
    if len(fraud_transactions) == 0:
        st.info("No fraudulent transactions detected")
        return
    
    # Create a selector for transaction
    if 'transaction_id' in st.session_state.features_df.columns:
        # Merge with features to get transaction IDs
        fraud_with_ids = fraud_transactions.merge(
            st.session_state.features_df[['transaction_id']],
            left_index=True,
            right_index=True
        )
        
        transaction_options = fraud_with_ids['transaction_id'].tolist()
        selected_transaction = st.selectbox("Select a transaction to analyze", transaction_options)
        
        # Get the index of the selected transaction
        selected_index = fraud_with_ids[fraud_with_ids['transaction_id'] == selected_transaction].index[0]
    else:
        # Use index as identifier
        transaction_options = fraud_transactions.index.tolist()
        selected_transaction = st.selectbox("Select a transaction to analyze", transaction_options)
        selected_index = selected_transaction
    
    # Display transaction details
    st.write("### Transaction Details")
    
    if 'transaction_id' in st.session_state.features_df.columns:
        transaction_data = st.session_state.features_df.loc[selected_index]
        
        # Display key fields
        key_fields = ['transaction_id', 'amount', 'timestamp', 'sender_id', 'receiver_id', 
                     'transaction_type', 'transaction_category', 'description']
        
        for field in key_fields:
            if field in transaction_data:
                st.write(f"**{field.replace('_', ' ').title()}:** {transaction_data[field]}")
    
    # Risk score
    risk_score = risk_scores.loc[selected_index, 'risk_score']
    st.metric("Risk Score", f"{risk_score:.4f}")
    
    # Explanation
    st.write("### Fraud Explanation")
    
    if selected_index in explanations:
        explanation = explanations[selected_index]
        
        # Show top contributing factors
        if 'top_factors' in explanation:
            st.write("#### Top Contributing Factors")
            
            factors_df = pd.DataFrame(explanation['top_factors'])
            factors_df.columns = ['Feature', 'Contribution']
            
            # Create a bar chart
            fig = px.bar(
                factors_df,
                x='Contribution',
                y='Feature',
                orientation='h',
                title="Feature Contributions to Risk Score",
                color='Contribution',
                color_continuous_scale='RdBu'
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Show rule violations
        if 'rule_violations' in explanation and len(explanation['rule_violations']) > 0:
            st.write("#### Rule Violations")
            
            for rule in explanation['rule_violations']:
                st.write(f"- {rule}")
        
        # Show model predictions
        if 'model_predictions' in explanation:
            st.write("#### Model Predictions")
            
            model_df = pd.DataFrame([
                {'Model': model, 'Score': score} 
                for model, score in explanation['model_predictions'].items()
            ])
            
            st.dataframe(model_df)
        
        # Show natural language explanation
        if 'text_explanation' in explanation:
            st.write("#### Explanation Summary")
            st.write(explanation['text_explanation'])
    else:
        st.info("No explanation available for this transaction")
    
    # What-if analysis
    st.write("### What-If Analysis")
    
    st.write("Adjust feature values to see how they affect the risk score:")
    
    # Get top features for this transaction
    if selected_index in explanations and 'top_factors' in explanations[selected_index]:
        top_features = [factor[0] for factor in explanations[selected_index]['top_factors'][:5]]
        
        # Create sliders for top features
        adjustments = {}
        for feature in top_features:
            if feature in st.session_state.features_df.columns:
                current_value = st.session_state.features_df.loc[selected_index, feature]
                
                # Determine appropriate range
                min_val = st.session_state.features_df[feature].min()
                max_val = st.session_state.features_df[feature].max()
                
                # Handle categorical features
                if st.session_state.features_df[feature].dtype == 'object':
                    options = st.session_state.features_df[feature].unique().tolist()
                    adjusted_value = st.selectbox(
                        f"Adjust {feature}",
                        options=options,
                        index=options.index(current_value) if current_value in options else 0
                    )
                else:
                    adjusted_value = st.slider(
                        f"Adjust {feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(current_value)
                    )
                
                adjustments[feature] = adjusted_value
        
        # Recalculate risk score button
        if st.button("Recalculate Risk Score"):
            # Create a copy of the original data
            modified_data = st.session_state.features_df.copy().loc[[selected_index]]
            
            # Apply adjustments
            for feature, value in adjustments.items():
                modified_data.loc[selected_index, feature] = value
            
            # Recalculate features
            modified_features = modified_data.copy()
            
            # Statistical features
            if st.session_state.settings["features"]["Statistical Features"]:
                stat_features = StatisticalFeatures()
                modified_features = stat_features.extract_features(modified_features)
            
            # Graph features
            if st.session_state.settings["features"]["Graph Features"]:
                graph_features = GraphFeatures()
                modified_features = graph_features.extract_features(modified_features)
            
            # NLP features
            if st.session_state.settings["features"]["NLP Features"]:
                nlp_features = NLPFeatures()
                modified_features = nlp_features.extract_features(modified_features)
            
            # Time series features
            if st.session_state.settings["features"]["Time Series Features"]:
                ts_features = TimeSeriesFeatures()
                modified_features = ts_features.extract_features(modified_features)
            
            # Get model predictions
            model_predictions = {}
            
            # Unsupervised models
            if 'unsupervised' in st.session_state.model_results:
                for model_name, model in st.session_state.model_results['unsupervised']['models'].items():
                    try:
                        # Get features used by this model
                        if 'feature_names' in st.session_state.model_results['unsupervised']:
                            feature_names = st.session_state.model_results['unsupervised']['feature_names']
                            model_features = modified_features[feature_names[model_name]]
                            prediction = model.decision_function(model_features)[0]
                            model_predictions[f"unsupervised_{model_name}"] = prediction
                    except Exception as e:
                        st.warning(f"Error predicting with {model_name}: {str(e)}")
            
            # Supervised models
            if 'supervised' in st.session_state.model_results:
                for model_name, model in st.session_state.model_results['supervised']['models'].items():
                    try:
                        # Get features used by this model
                        if 'feature_names' in st.session_state.model_results['supervised']:
                            feature_names = st.session_state.model_results['supervised']['feature_names']
                            model_features = modified_features[feature_names[model_name]]
                            prediction = model.predict_proba(model_features)[0, 1]
                            model_predictions[f"supervised_{model_name}"] = prediction
                    except Exception as e:
                        st.warning(f"Error predicting with {model_name}: {str(e)}")
            
            # Rule-based models
            if 'rule' in st.session_state.model_results:
                rule_score = 0.0
                violated_rules = []
                
                for rule_name, rule_func in st.session_state.model_results['rule']['rules'].items():
                    try:
                        if rule_func(modified_data.iloc[0]):
                            rule_score += 1.0 / len(st.session_state.model_results['rule']['rules'])
                            violated_rules.append(rule_name)
                    except Exception as e:
                        st.warning(f"Error applying rule {rule_name}: {str(e)}")
                
                model_predictions["rule"] = rule_score
            
            # Calculate new risk score
            risk_scorer = RiskScorer(
                method=st.session_state.settings["scoring_method"],
                custom_weights=st.session_state.settings["custom_weights"]
            )
            
            new_risk_score = risk_scorer.calculate_scores(
                {"unsupervised": model_predictions, "supervised": model_predictions, "rule": model_predictions},
                modified_features
            )
            
            new_risk_score = new_risk_score.loc[selected_index, 'risk_score']
            
            # Display comparison
            st.write("#### Risk Score Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original Risk Score", f"{risk_score:.4f}")
            
            with col2:
                st.metric("New Risk Score", f"{new_risk_score:.4f}", 
                         delta=f"{new_risk_score - risk_score:.4f}")
            
            # Show if it would still be flagged as fraud
            threshold = st.session_state.threshold
            is_fraud_original = risk_score > threshold
            is_fraud_new = new_risk_score > threshold
            
            st.write(f"Original classification: {'Fraud' if is_fraud_original else 'Not Fraud'}")
            st.write(f"New classification: {'Fraud' if is_fraud_new else 'Not Fraud'}")
            
            if is_fraud_original != is_fraud_new:
                st.warning("Classification changed with the adjustments!")
            else:
                st.info("Classification remains the same")

# Reports Page
def render_reports():
    """Render the reports page"""
    st.title("📄 Reports")
    
    if not st.session_state.processing_complete:
        st.warning("Please run the detection analysis first")
        return
    
    risk_scores = st.session_state.risk_scores
    features_df = st.session_state.features_df
    
    st.markdown("""
    Generate comprehensive reports for auditors and stakeholders. Reports can be downloaded
    in PDF format and include detailed analysis of fraudulent transactions.
    """)
    
    # Report options
    st.subheader("Report Options")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Executive Summary", "Detailed Fraud Analysis", "Technical Report", "Custom Report"]
    )
    
    include_charts = st.checkbox("Include Charts and Visualizations", value=True)
    include_explanations = st.checkbox("Include Detailed Explanations", value=True)
    include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    # Transaction selection
    st.subheader("Transaction Selection")
    
    fraud_transactions = risk_scores[risk_scores['is_fraud']]
    
    if len(fraud_transactions) > 0:
        max_transactions = min(len(fraud_transactions), 100)  # Limit to 100 for performance
        num_transactions = st.slider(
            "Number of Fraudulent Transactions to Include",
            min_value=1,
            max_value=max_transactions,
            value=min(10, max_transactions),
            step=1
        )
        
        # Sort by risk score and select top N
        top_fraud = fraud_transactions.nlargest(num_transactions, 'risk_score')
    else:
        st.info("No fraudulent transactions detected")
        return
    
    # Report generation options
    st.subheader("Report Generation")
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                # Initialize PDF generator
                pdf_generator = PDFGenerator()
                
                # Generate report based on type
                if report_type == "Executive Summary":
                    report_path = pdf_generator.generate_executive_summary(
                        features_df,
                        risk_scores,
                        top_fraud,
                        include_charts=include_charts,
                        include_recommendations=include_recommendations
                    )
                elif report_type == "Detailed Fraud Analysis":
                    report_path = pdf_generator.generate_detailed_fraud_analysis(
                        features_df,
                        risk_scores,
                        top_fraud,
                        st.session_state.explanations,
                        include_charts=include_charts,
                        include_explanations=include_explanations,
                        include_recommendations=include_recommendations
                    )
                elif report_type == "Technical Report":
                    report_path = pdf_generator.generate_technical_report(
                        features_df,
                        risk_scores,
                        st.session_state.model_results,
                        include_charts=include_charts
                    )
                else:  # Custom Report
                    report_path = pdf_generator.generate_custom_report(
                        features_df,
                        risk_scores,
                        top_fraud,
                        st.session_state.explanations,
                        st.session_state.model_results,
                        include_charts=include_charts,
                        include_explanations=include_explanations,
                        include_recommendations=include_recommendations
                    )
                
                # Display success message
                st.success(f"Report generated successfully!")
                
                # Provide download button
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download Report",
                        data=file,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf"
                    )
                
                # Show report preview
                st.subheader("Report Preview")
                st.info("Report preview is not available in this interface. Please download the PDF to view the full report.")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.exception(e)

# Main function
def main():
    """Main function to run the Streamlit app"""
    page = render_sidebar()
    
    if page == "Data Upload":
        render_data_upload()
    elif page == "Column Mapping":
        render_column_mapping()
    elif page == "Analysis Settings":
        render_analysis_settings()
    elif page == "Run Detection":
        render_run_detection()
    elif page == "Results Dashboard":
        render_results_dashboard()
    elif page == "Explainability":
        render_explainability()
    elif page == "Reports":
        render_reports()

if __name__ == "__main__":
    main()