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
import tempfile
import logging
import traceback
import hashlib
import zipfile
import atexit
from datetime import datetime, timedelta
import yaml
from io import BytesIO, StringIO
import plotly.express as px
import plotly.graph_objects as go
import chardet

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
from fraud_detection_engine.utils.cache_utils import AnalysisCache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to create a unique hash for the file content
def get_file_hash(file_content):
    """
    Calculate SHA256 hash of file content for unique identification
    
    Args:
        file_content (bytes): Content of the file
        
    Returns:
        str: SHA256 hash string
    """
    return hashlib.sha256(file_content).hexdigest()

# Function to ensure cache directory exists
def ensure_cache_directory():
    """Ensure cache directory exists"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "..", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    except Exception as e:
        logger.error(f"Error creating cache directory: {str(e)}")
        return None

# Initialize cache with permanent settings
cache_dir = ensure_cache_directory()
if cache_dir:
    cache = AnalysisCache(cache_dir=cache_dir, max_cache_size=0, compression=True, cache_expiry_days=0)
else:
    st.error("Failed to create cache directory. Caching will be disabled.")
    cache = None

# Function to check if cache is available
def is_cache_available():
    """Check if cache is available"""
    return cache is not None

# Function to safely execute cache operations
def safe_cache_operation(operation, *args, **kwargs):
    """
    Safely execute cache operations with error handling
    
    Args:
        operation: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the operation or None if failed
    """
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.error(f"Cache operation failed: {str(e)}")
        st.error(f"Cache operation failed: {str(e)}")
        return None

# Function to validate and load cache if available
def validate_and_load_cache():
    """Validate and load cache if available"""
    if st.session_state.file_hash and is_cache_available() and cache.is_cache_valid(st.session_state.file_hash):
        cache_step = cache.get_cache_step(st.session_state.file_hash)
        if cache_step:
            st.session_state.cache_available = True
            st.session_state.cache_step_name = cache_step
            return True
    return False

# Function to display cache file information
def display_cache_info():
    """Display detailed cache information"""
    if st.session_state.file_hash and is_cache_available():
        st.write("### Cache File Information")
        
        cache_path = cache._get_cache_path(st.session_state.file_hash)
        metadata_path = cache._get_metadata_path(st.session_state.file_hash)
        
        if cache_path.exists():
            st.write(f"Cache file: {cache_path}")
            st.write(f"Cache size: {cache_path.stat().st_size / 1024:.2f} KB")
            st.write(f"Last modified: {datetime.fromtimestamp(cache_path.stat().st_mtime)}")
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    st.write("### Cache Metadata")
                    st.json(metadata)
                except Exception as e:
                    st.warning(f"Could not read metadata: {str(e)}")
        else:
            st.write("No cache file exists for this file")

# Function to render cache management options
def render_cache_management():
    """Render cache management options in the sidebar"""
    st.sidebar.subheader("Cache Management")
    
    # Show cache statistics
    if is_cache_available():
        cache_stats = cache.get_cache_stats()
        if cache_stats:
            st.sidebar.write(f"Cache files: {cache_stats.get('total_files', 0)}")
            st.sidebar.write(f"Cache size: {cache_stats.get('total_size_mb', 0):.2f} MB")
            st.sidebar.write(f"Processing complete: {cache_stats.get('processing_complete_count', 0)}")
        
        # Clear cache button
        if st.sidebar.button("Clear All Cache"):
            if st.sidebar.checkbox("Confirm Clear Cache", key="confirm_clear"):
                cache.clear_cache()
                st.sidebar.success("All cache cleared")
                st.session_state.cache_loaded = False
                st.session_state.cache_available = False
                st.session_state.cache_step_name = None
                # Reset current step to Data Upload
                st.session_state.current_step = "Data Upload"
                # Reset analysis progress
                st.session_state.analysis_progress = 0
                st.session_state.analysis_status = "Ready"
                st.session_state.analysis_start_time = None
                st.session_state.analysis_elapsed_time = 0
                st.session_state.analysis_log = []
                st.experimental_rerun()
        
        # Export cache button
        if st.sidebar.button("Export Cache"):
            export_dir = cache.export_cache()
            if export_dir:
                st.sidebar.success(f"Cache exported to: {export_dir}")
            else:
                st.sidebar.error("Failed to export cache")
        
        # Import cache section
        st.sidebar.subheader("Import Cache")
        uploaded_cache = st.sidebar.file_uploader(
            "Upload cache file",
            type=['zip'],
            help="Upload a zip file containing cache files"
        )
        
        if uploaded_cache is not None:
            # Create temporary directory for import
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                temp_path = os.path.join(temp_dir, uploaded_cache.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_cache.getvalue())
                
                # Extract zip file
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Import cache
                if cache.import_cache(temp_dir):
                    st.sidebar.success("Cache imported successfully")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Failed to import cache")

# Function to cleanup cache on application exit
def cleanup_cache():
    """Cleanup cache on application exit"""
    try:
        if cache:
            # Save any pending cache operations
            cache._save_cache_index()
            cache._save_cache_metadata()
            logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error(f"Error during cache cleanup: {str(e)}")

# Register cleanup function
atexit.register(cleanup_cache)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML files"""
    config = {}
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to config files relative to script location
        model_params_path = os.path.join(script_dir, '..', 'config', 'model_params.yml')
        rule_engine_path = os.path.join(script_dir, '..', 'config', 'rule_engine_config.yml')
        api_keys_path = os.path.join(script_dir, '..', 'config', 'api_keys.yml')
        
        with open(model_params_path, 'r') as f:
            config['model_params'] = yaml.safe_load(f)
        with open(rule_engine_path, 'r') as f:
            config['rule_engine'] = yaml.safe_load(f)
        with open(api_keys_path, 'r') as f:
            config['api_keys'] = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        config = {
            'model_params': {},
            'rule_engine': {},
            'api_keys': {}
        }
    return config

config = load_config()

# Initialize session state variables
def init_session_state():
    """Initialize session state variables"""
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
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'Data Upload'
    if 'file_hash' not in st.session_state:
        st.session_state.file_hash = None
    if 'cache_loaded' not in st.session_state:
        st.session_state.cache_loaded = False
    if 'settings' not in st.session_state:
        st.session_state.settings = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'cache_step' not in st.session_state:
        st.session_state.cache_step = None
    if 'cache_available' not in st.session_state:
        st.session_state.cache_available = False
    if 'cache_step_name' not in st.session_state:
        st.session_state.cache_step_name = None
    if 'feature_extractors' not in st.session_state:
        st.session_state.feature_extractors = {}
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = 0
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = "Ready"
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None
    if 'analysis_elapsed_time' not in st.session_state:
        st.session_state.analysis_elapsed_time = 0
    if 'analysis_log' not in st.session_state:
        st.session_state.analysis_log = []
    # New cache-related state variables
    if 'cache_file_path' not in st.session_state:
        st.session_state.cache_file_path = None
    if 'cache_metadata' not in st.session_state:
        st.session_state.cache_metadata = {}
    if 'auto_resume' not in st.session_state:
        st.session_state.auto_resume = False
    if 'resume_step' not in st.session_state:
        st.session_state.resume_step = None
    # Feature extraction state variables
    if 'statistical_features' not in st.session_state:
        st.session_state.statistical_features = None
    if 'graph_features' not in st.session_state:
        st.session_state.graph_features = None
    if 'nlp_features' not in st.session_state:
        st.session_state.nlp_features = None
    if 'timeseries_features' not in st.session_state:
        st.session_state.timeseries_features = None
    # Model state variables
    if 'unsupervised_models' not in st.session_state:
        st.session_state.unsupervised_models = None
    if 'supervised_models' not in st.session_state:
        st.session_state.supervised_models = None
    if 'rule_engine' not in st.session_state:
        st.session_state.rule_engine = None

# Initialize session state
init_session_state()

# Function to save current state to cache
def save_state_to_cache(file_hash, session_state, step_name):
    """Save current analysis state to cache"""
    if not is_cache_available():
        logger.warning("Cache not available. Cannot save state.")
        return
        
    try:
        if file_hash is None:
            logger.warning("No file hash available. Cannot save to cache.")
            return
        
        # Prepare state to save
        state = {
            'current_step': session_state.current_step,
            'data': session_state.data,
            'processed_data': session_state.processed_data,
            'features_df': session_state.features_df,
            'risk_scores': session_state.risk_scores,
            'explanations': session_state.explanations,
            'model_results': session_state.model_results,
            'column_mapping': session_state.column_mapping,
            'settings': session_state.settings,
            'processing_complete': session_state.processing_complete,
            'original_data': session_state.original_data,
            'feature_extractors': session_state.feature_extractors,
            'analysis_progress': session_state.analysis_progress,
            'analysis_status': session_state.analysis_status,
            'analysis_start_time': session_state.analysis_start_time,
            'analysis_elapsed_time': session_state.analysis_elapsed_time,
            'analysis_log': session_state.analysis_log,
            # Cache metadata
            'cache_metadata': session_state.cache_metadata,
            # Feature extraction states
            'statistical_features': session_state.statistical_features,
            'graph_features': session_state.graph_features,
            'nlp_features': session_state.nlp_features,
            'timeseries_features': session_state.timeseries_features,
            # Model states
            'unsupervised_models': session_state.unsupervised_models,
            'supervised_models': session_state.supervised_models,
            'rule_engine': session_state.rule_engine
        }
        
        # Save state to cache
        cache.save_state(file_hash, state, step_name)
        logger.info(f"Saved state to cache at step: {step_name}")
        
        # Update cache metadata
        session_state.cache_metadata = {
            'last_saved': datetime.now().isoformat(),
            'step_name': step_name,
            'file_hash': file_hash,
            'file_size': session_state.uploaded_file.size if session_state.uploaded_file else 0
        }
        
    except Exception as e:
        logger.error(f"Error saving state to cache: {str(e)}")
        st.error(f"Error saving state to cache: {str(e)}")

# Function to load state from cache
def load_state_from_cache(file_hash):
    """Load analysis state from cache"""
    if not is_cache_available():
        return False
        
    try:
        # Try to load state from cache
        state = cache.load_state(file_hash)
        
        if state is not None:
            # Restore session state
            for key, value in state.items():
                if key in st.session_state:
                    st.session_state[key] = value
            
            # Restore cache metadata if available
            if 'cache_metadata' in state:
                st.session_state.cache_metadata = state['cache_metadata']
            
            st.session_state.cache_loaded = True
            st.session_state.current_step = state.get('current_step', 'Data Upload')
            logger.info("Loaded state from cache")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error loading state from cache: {str(e)}")
        st.error(f"Error loading state from cache: {str(e)}")
        return False

# Function to get the current step from cache
def get_cache_step(file_hash):
    """Get the current step from cache"""
    if not is_cache_available():
        return None
        
    try:
        return cache.get_cache_step(file_hash)
    except Exception as e:
        logger.error(f"Error getting cache step: {str(e)}")
        return None

# Function to check if cache is valid
def is_cache_valid(file_hash):
    """Check if cache exists and is valid for a file"""
    if not is_cache_available():
        return False
        
    try:
        return cache.is_cache_valid(file_hash)
    except Exception as e:
        logger.error(f"Error checking cache validity: {str(e)}")
        return False

# Function to log analysis progress
def log_analysis_progress(message, progress=None):
    """Log analysis progress"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Add to log
    st.session_state.analysis_log.append(log_entry)
    
    # Update progress if provided
    if progress is not None:
        st.session_state.analysis_progress = progress
    
    # Update status
    st.session_state.analysis_status = message
    
    # Update elapsed time
    if st.session_state.analysis_start_time:
        st.session_state.analysis_elapsed_time = (datetime.now() - st.session_state.analysis_start_time).total_seconds()

# Function to display analysis progress
def display_analysis_progress():
    """Display analysis progress in a sidebar"""
    if st.session_state.analysis_start_time:
        st.sidebar.subheader("Analysis Progress")
        
        # Progress bar
        progress = st.session_state.analysis_progress
        st.sidebar.progress(progress / 100)
        
        # Status
        st.sidebar.write(f"**Status:** {st.session_state.analysis_status}")
        
        # Elapsed time
        elapsed = st.session_state.analysis_elapsed_time
        if elapsed > 0:
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            st.sidebar.write(f"**Elapsed Time:** {time_str}")
        
        # Show recent log entries
        st.sidebar.subheader("Recent Activity")
        log_height = min(10, len(st.session_state.analysis_log))
        for log_entry in st.session_state.analysis_log[-log_height:]:
            st.sidebar.write(log_entry)
        
        # Show cache status
        st.sidebar.subheader("Cache Status")
        if st.session_state.cache_loaded:
            st.sidebar.success("Analysis loaded from cache")
            if st.session_state.cache_step_name:
                st.sidebar.info(f"Current step: {st.session_state.cache_step_name}")
            # Show cache file info
            if st.session_state.file_hash:
                cache_file_path = cache._get_cache_path(st.session_state.file_hash)
                st.sidebar.info(f"Cache file: {cache_file_path.name}")
        elif st.session_state.cache_available:
            st.sidebar.warning("Cache available but not loaded")
            if st.session_state.file_hash:
                cache_file_path = cache._get_cache_path(st.session_state.file_hash)
                st.sidebar.info(f"Cache file: {cache_file_path.name}")
        else:
            st.sidebar.info("No cache available")

# Sidebar
def render_sidebar():
    """Render the sidebar with navigation and controls"""
    st.sidebar.title("🛡️ Fraud Detection System")
    st.sidebar.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    
    # Navigation
    st.sidebar.subheader("Navigation")
    
    # Get the list of pages
    pages = [
        "Data Upload", 
        "Column Mapping", 
        "Analysis Settings", 
        "Run Detection", 
        "Results Dashboard", 
        "Explainability", 
        "Reports"
    ]
    
    # If we have a current step in session state, set the index of the radio button
    if 'current_step' in st.session_state and st.session_state.current_step in pages:
        index = pages.index(st.session_state.current_step)
    else:
        index = 0
    
    page = st.sidebar.radio("Go to", pages, index=index, key="navigation_radio")
    
    # Update the current step if the user changes the page
    if page != st.session_state.get('current_step', 'Data Upload'):
        st.session_state.current_step = page
    
    # Display analysis progress
    display_analysis_progress()
    
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
    
    return page

# Helper function to detect file encoding
def detect_file_encoding(file_content):
    """Detect the encoding of file content"""
    try:
        result = chardet.detect(file_content)
        return result['encoding']
    except Exception as e:
        logger.error(f"Error detecting file encoding: {str(e)}")
        return None

# Helper function to read CSV with multiple encoding attempts
def read_csv_with_encodings(file_content, filename):
    """Read CSV file trying multiple encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    # First try with detected encoding
    detected_encoding = detect_file_encoding(file_content)
    if detected_encoding:
        try:
            st.write(f"Trying detected encoding: {detected_encoding}")
            return pd.read_csv(StringIO(file_content.decode(detected_encoding)))
        except Exception as e:
            st.warning(f"Failed with detected encoding {detected_encoding}: {str(e)}")
    
    # Try common encodings
    for encoding in encodings:
        try:
            st.write(f"Trying encoding: {encoding}")
            return pd.read_csv(StringIO(file_content.decode(encoding)))
        except UnicodeDecodeError:
            st.warning(f"Failed with {encoding} encoding: Unicode decode error")
        except Exception as e:
            st.warning(f"Failed with {encoding} encoding: {str(e)}")
    
    return None

# Helper function to read Excel with multiple engines
def read_excel_with_engines(file_content, filename):
    """Read Excel file trying multiple engines"""
    # Try default engine first
    try:
        st.write("Trying default Excel engine")
        return pd.read_excel(BytesIO(file_content))
    except Exception as e:
        st.warning(f"Failed with default engine: {str(e)}")
    
    # Try specific engines based on file extension
    if filename.endswith('.xlsx'):
        try:
            st.write("Trying openpyxl engine")
            return pd.read_excel(BytesIO(file_content), engine='openpyxl')
        except Exception as e:
            st.warning(f"Failed with openpyxl: {str(e)}")
    elif filename.endswith('.xls'):
        try:
            st.write("Trying xlrd engine")
            return pd.read_excel(BytesIO(file_content), engine='xlrd')
        except Exception as e:
            st.warning(f"Failed with xlrd: {str(e)}")
    
    return None

# Clean dataframe function
def clean_dataframe(df):
    """
    Clean a DataFrame by handling infinity, NaN, and extreme values
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: Cleaned DataFrame
    """
    try:
        # Replace infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle extreme values for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            # Calculate percentiles
            p1 = np.nanpercentile(df[col], 1)
            p99 = np.nanpercentile(df[col], 99)
            
            # Cap extreme values
            if not np.isnan(p1) and not np.isnan(p99):
                # Use a more conservative capping approach
                iqr = p99 - p1
                lower_bound = p1 - 1.5 * iqr
                upper_bound = p99 + 1.5 * iqr
                
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
            # Replace any remaining NaN with median
            median_val = df[col].median()
            if not np.isnan(median_val):
                df[col] = df[col].fillna(median_val)
        
        # Handle categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            # Fill NaN with mode or 'Unknown'
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
        
        return df
    except Exception as e:
        logger.error(f"Error cleaning DataFrame: {str(e)}")
        return df

# Data Upload Page
def render_data_upload():
    """Render the data upload page"""
    st.title("📁 Data Upload")
    st.markdown("""
    Upload your transaction data for analysis. The system supports CSV and Excel files.
    """)
    
    # Debug information
    st.write("### Debug Information")
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # File upload with enhanced error handling
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing transaction data",
        key="file_uploader"
    )
    
    # Debug: Show if file was uploaded
    if uploaded_file is not None:
        st.write("### File Upload Details")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Store uploaded file in session state
        st.session_state.uploaded_file = uploaded_file
        
        # Check cache first
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        # Calculate file hash for caching
        st.session_state.file_hash = get_file_hash(file_content)
        
        # Validate and load cache
        if validate_and_load_cache():
            st.success(f"This file has been processed before. Last completed step: {st.session_state.cache_step_name}")
            
            # Auto-resume if enabled
            if st.session_state.auto_resume:
                st.info("Auto-resuming from last completed step...")
                if safe_cache_operation(load_state_from_cache, st.session_state.file_hash):
                    st.success("Analysis state loaded from cache!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to load from cache")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Load from Cache"):
                    log_analysis_progress("Loading analysis state from cache...")
                    if safe_cache_operation(load_state_from_cache, st.session_state.file_hash):
                        st.success("Analysis state loaded from cache!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to load from cache")
            
            with col2:
                if st.button("Reprocess File"):
                    st.session_state.cache_loaded = False
                    st.session_state.cache_available = False
                    st.session_state.cache_step_name = None
                    st.success("File will be reprocessed")
                    # Reset analysis progress
                    st.session_state.analysis_progress = 0
                    st.session_state.analysis_status = "Ready"
                    st.session_state.analysis_start_time = None
                    st.session_state.analysis_elapsed_time = 0
                    st.session_state.analysis_log = []
            
            with col3:
                if st.button("View Cache Info"):
                    if is_cache_available():
                        cache_info = cache.get_cache_info()
                        if cache_info:
                            st.json(cache_info)
                        else:
                            st.info("No cache info available")
        
        # If cache is not loaded or user chooses to reprocess
        if not st.session_state.cache_loaded:
            # Read file content directly
            df = None
            
            if uploaded_file.name.endswith('.csv'):
                st.write("### Processing CSV file")
                df = read_csv_with_encodings(file_content, uploaded_file.name)
                
                if df is None:
                    st.error("Could not read CSV file with any encoding")
                    
                    # Show file content for debugging
                    st.write("### File Content (first 1000 bytes)")
                    try:
                        st.write(file_content[:1000])
                    except Exception as e:
                        st.error(f"Could not display file content: {str(e)}")
                    
                    # Try to create a simple CSV from the content
                    st.write("### Attempting to create CSV from content")
                    try:
                        # Try different encodings directly
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                content_str = file_content.decode(encoding, errors='ignore')
                                df = pd.read_csv(StringIO(content_str))
                                st.success(f"Successfully read CSV with {encoding} encoding")
                                break
                            except:
                                continue
                    except Exception as e:
                        st.error(f"Failed to create CSV from content: {str(e)}")
                
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                st.write("### Processing Excel file")
                df = read_excel_with_engines(file_content, uploaded_file.name)
                
                if df is None:
                    st.error("Could not read Excel file with any engine")
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
            
            # Validate the dataframe
            if df is None:
                st.error("Failed to read the uploaded file")
                return
            
            if df.empty:
                st.error("The uploaded file appears to be empty")
                return
            
            # Display data info
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            st.write("### Data Preview")
            st.dataframe(df.head(10))
            
            # Store in session state
            st.session_state.data = df
            st.session_state.original_data = df.copy()
            
            st.write("Data stored in session state")
            
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found for statistics")
                
            # Check file structure
            st.write("### File Structure Check")
            if len(df.columns) > 0:
                st.write(f"Found {len(df.columns)} columns")
                st.write("Columns:", list(df.columns))
            else:
                st.error("No columns found in the file")
                
            # Save state to cache
            if is_cache_available():
                save_state_to_cache(st.session_state.file_hash, st.session_state, "Data Upload")
    
    # Sample data option
    st.markdown("---")
    st.subheader("Or Use Sample Data")
    
    if st.button("Load Sample Data"):
        try:
            # Look for sample data in multiple locations
            possible_paths = [
                os.path.join('..', 'data', 'sample_transactions.csv'),
                os.path.join('data', 'sample_transactions.csv'),
                'sample_transactions.csv'
            ]
            
            sample_path = None
            for path in possible_paths:
                st.write(f"Checking path: {path}")
                if os.path.exists(path):
                    sample_path = path
                    st.write(f"Found sample data at: {path}")
                    break
            
            if sample_path:
                df = pd.read_csv(sample_path)
                st.session_state.data = df
                st.session_state.original_data = df.copy()
                
                # Calculate file hash for caching
                with open(sample_path, 'rb') as f:
                    file_content = f.read()
                st.session_state.file_hash = get_file_hash(file_content)
                
                st.success(f"Sample data loaded! Shape: {df.shape}")
                st.dataframe(df.head(10))
                
                # Save state to cache
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Data Upload")
            else:
                st.warning("Sample data file not found, creating new sample...")
                
                # Create sample data if it doesn't exist
                np.random.seed(42)
                
                # Create more realistic sample data
                n_samples = 1000
                start_date = datetime(2023, 1, 1)
                
                sample_data = pd.DataFrame({
                    'transaction_id': [f'TXN{i:06d}' for i in range(1, n_samples + 1)],
                    'timestamp': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
                    'amount': np.random.lognormal(8, 1.5, n_samples),  # More realistic amounts
                    'sender_id': [f'CUST{np.random.randint(1, 201):03d}' for _ in range(n_samples)],
                    'receiver_id': [f'CUST{np.random.randint(1, 201):03d}' for _ in range(n_samples)],
                    'currency': ['USD'] * n_samples,
                    'description': [f'Transaction type {np.random.choice(["PAYMENT", "TRANSFER", "PURCHASE", "WITHDRAWAL"])}' for _ in range(n_samples)],
                    'transaction_type': np.random.choice(['debit', 'credit'], n_samples),
                    'merchant_category': np.random.choice(['retail', 'services', 'utilities', 'finance'], n_samples)
                })
                
                # Ensure data directory exists
                data_dir = os.path.join('..', 'data')
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                
                sample_path = os.path.join(data_dir, 'sample_transactions.csv')
                sample_data.to_csv(sample_path, index=False)
                st.write(f"Created sample data at: {sample_path}")
                
                # Load the newly created sample data
                st.session_state.data = sample_data
                st.session_state.original_data = sample_data.copy()
                
                # Calculate file hash for caching
                st.session_state.file_hash = get_file_hash(sample_data.to_csv().encode())
                
                st.success(f"Sample data loaded! Shape: {sample_data.shape}")
                st.dataframe(sample_data.head(10))
                
                # Save state to cache
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Data Upload")
                
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.write("### Error Details")
            st.text(traceback.format_exc())
    
    # Additional troubleshooting section
    st.markdown("---")
    st.subheader("Troubleshooting File Upload Issues")
    
    with st.expander("Click here if you're having trouble uploading files"):
        st.write("""
        ### Common Issues and Solutions:
        
        1. **File not uploading at all**:
           - Try using a different browser (Chrome, Firefox, or Edge work best)
           - Clear your browser cache and cookies
           - Check if your file size is too large (try with a smaller file first)
           - Make sure the file is not open in another program
           - Try refreshing the page and uploading again
        
        2. **CSV file encoding issues**:
           - Try saving your CSV file with UTF-8 encoding
           - Use Excel to "Save As" and select "CSV UTF-8" format
           - Try converting your file to Excel format (.xlsx)
           - Check if your CSV has special characters that might cause encoding issues
        
        3. **Permission issues**:
           - Make sure the file is not read-only
           - Try moving the file to your desktop first
           - Check if your antivirus software is blocking the upload
           - Try running the application with administrator privileges
        
        4. **Browser compatibility**:
           - Update your browser to the latest version
           - Try incognito/private browsing mode
           - Disable browser extensions temporarily
           - Try a different browser
        
        5. **File format issues**:
           - Make sure your CSV file has proper headers
           - Check if your file has consistent formatting
           - Try opening the file in Excel and re-saving it
           - Remove any special characters from column names
        
        ### Alternative Methods:
        
        If the file uploader still doesn't work, you can:
        1. Use the "Load Sample Data" button above
        2. Place your CSV file in the `data` folder with the name `sample_transactions.csv`
        3. Try converting your file to Excel format (.xlsx)
        4. Use the text area below to paste your CSV data directly:
        """)
        
        # Add text area for direct CSV input
        st.write("### Direct CSV Input")
        csv_text = st.text_area("Paste your CSV data here:", height=200)
        
        if st.button("Load from Text"):
            try:
                if csv_text.strip():
                    # Use StringIO to read the CSV data
                    df = pd.read_csv(StringIO(csv_text))
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    
                    # Calculate file hash for caching
                    st.session_state.file_hash = get_file_hash(csv_text.encode())
                    
                    st.success(f"Data loaded from text! Shape: {df.shape}")
                    st.dataframe(df.head(10))
                    
                    # Save state to cache
                    if is_cache_available():
                        save_state_to_cache(st.session_state.file_hash, st.session_state, "Data Upload")
                else:
                    st.warning("Please paste some CSV data")
            except Exception as e:
                st.error(f"Error loading data from text: {str(e)}")
                st.write("### Error Details")
                st.text(traceback.format_exc())
        
        # Add cache information to the troubleshooting section
        st.write("### Cache Information")
        if st.session_state.file_hash:
            st.write(f"Current file hash: {st.session_state.file_hash[:16]}...")
            if is_cache_available():
                cache_path = cache._get_cache_path(st.session_state.file_hash)
                if cache_path.exists():
                    st.write(f"Cache file exists: {cache_path.name}")
                    st.write(f"Cache size: {cache_path.stat().st_size / 1024:.2f} KB")
                else:
                    st.write("No cache file exists for this file")
        else:
            st.write("No file hash available")

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
        
        # Save state to cache
        if is_cache_available():
            save_state_to_cache(st.session_state.file_hash, st.session_state, "Column Mapping")
    
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
            st.session_state.current_step = "Analysis Settings"
            
            # Save state to cache
            if is_cache_available():
                save_state_to_cache(st.session_state.file_hash, st.session_state, "Column Mapping")
            
            st.success("Data processed successfully! You can now configure analysis settings.")
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

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
    
    # Map the internal values to display names
    method_display_names = {
        "weighted_average": "Weighted Average",
        "maximum": "Maximum Score", 
        "custom": "Custom Weights"
    }
    
    scoring_method = st.selectbox(
        "Scoring Method",
        list(method_display_names.keys()),
        format_func=lambda x: method_display_names[x],
        help="Method to combine scores from different models"
    )
    
    if scoring_method == "custom":
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
        st.session_state.current_step = "Run Detection"
        
        # Save state to cache
        if is_cache_available():
            save_state_to_cache(st.session_state.file_hash, st.session_state, "Analysis Settings")
        
        st.success("Settings saved! You can now run the detection analysis.")

# Run Detection Page
def render_run_detection():
    """Render the run detection page"""
    st.title("🚀 Run Fraud Detection")
    
    # Check cache status
    if st.session_state.cache_loaded:
        st.info("Resuming from cached state. Some steps may be skipped.")
        if st.session_state.cache_step_name:
            st.info(f"Last completed step: {st.session_state.cache_step_name}")
    
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
    method_display_names = {
        "weighted_average": "Weighted Average",
        "maximum": "Maximum Score", 
        "custom": "Custom Weights"
    }
    st.write(f"- Method: {method_display_names.get(settings['scoring_method'], settings['scoring_method'])}")
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
        # Initialize analysis progress
        st.session_state.analysis_start_time = datetime.now()
        st.session_state.analysis_progress = 0
        st.session_state.analysis_status = "Starting analysis..."
        st.session_state.analysis_log = []
        
        log_analysis_progress("Starting fraud detection analysis...", 5)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Feature Engineering
            log_analysis_progress("Step 1/5: Feature Engineering...", 10)
            progress_bar.progress(10)
            
            features_df = st.session_state.processed_data.copy()
            
            # Clean data before feature engineering
            features_df = clean_dataframe(features_df)
            
            # Initialize feature extractors
            feature_extractors = {}
            
            # Statistical features
            if settings["features"]["Statistical Features"]:
                log_analysis_progress("Extracting statistical features...", 20)
                if st.session_state.statistical_features is None:
                    stat_features = StatisticalFeatures()
                    features_df = stat_features.extract_features(features_df)
                    # Clean after feature extraction
                    features_df = clean_dataframe(features_df)
                    st.session_state.statistical_features = stat_features
                else:
                    features_df = st.session_state.statistical_features.transform(features_df)
                
                feature_extractors['statistical'] = st.session_state.statistical_features
                progress_bar.progress(20)
                
                # Save state after statistical features
                st.session_state.features_df = features_df
                st.session_state.feature_extractors = feature_extractors
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Statistical Features")
            
            # Graph features
            if settings["features"]["Graph Features"]:
                log_analysis_progress("Extracting graph features...", 30)
                if st.session_state.graph_features is None:
                    graph_features = GraphFeatures()
                    features_df = graph_features.extract_features(features_df)
                    # Clean after feature extraction
                    features_df = clean_dataframe(features_df)
                    st.session_state.graph_features = graph_features
                else:
                    features_df = st.session_state.graph_features.transform(features_df)
                
                feature_extractors['graph'] = st.session_state.graph_features
                progress_bar.progress(30)
                
                # Save state after graph features
                st.session_state.features_df = features_df
                st.session_state.feature_extractors = feature_extractors
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Graph Features")
            
            # NLP features
            if settings["features"]["NLP Features"]:
                log_analysis_progress("Extracting NLP features...", 40)
                if st.session_state.nlp_features is None:
                    nlp_features = NLPFeatures()
                    features_df = nlp_features.extract_features(features_df)
                    # Clean after feature extraction
                    features_df = clean_dataframe(features_df)
                    st.session_state.nlp_features = nlp_features
                else:
                    features_df = st.session_state.nlp_features.transform(features_df)
                
                feature_extractors['nlp'] = st.session_state.nlp_features
                progress_bar.progress(40)
                
                # Save state after NLP features
                st.session_state.features_df = features_df
                st.session_state.feature_extractors = feature_extractors
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "NLP Features")
            
            # Time series features
            if settings["features"]["Time Series Features"]:
                log_analysis_progress("Extracting time series features...", 50)
                if st.session_state.timeseries_features is None:
                    ts_features = TimeSeriesFeatures()
                    features_df = ts_features.extract_features(features_df)
                    # Clean after feature extraction
                    features_df = clean_dataframe(features_df)
                    st.session_state.timeseries_features = ts_features
                else:
                    features_df = st.session_state.timeseries_features.transform(features_df)
                
                feature_extractors['timeseries'] = st.session_state.timeseries_features
                progress_bar.progress(50)
                
                # Save state after time series features
                st.session_state.features_df = features_df
                st.session_state.feature_extractors = feature_extractors
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Time Series Features")
            
            # Store feature extractors in session state
            st.session_state.feature_extractors = feature_extractors
            
            # Step 2: Model Training/Prediction
            log_analysis_progress("Step 2/5: Running Models...", 60)
            model_results = {}
            
            # Unsupervised models
            if settings["models"]["Unsupervised Models"]:
                log_analysis_progress("Running unsupervised models...", 65)
                if st.session_state.unsupervised_models is None:
                    unsupervised = UnsupervisedModels(contamination=settings["contamination"])
                    unsupervised_results = unsupervised.run_models(features_df)
                    st.session_state.unsupervised_models = unsupervised
                else:
                    unsupervised = st.session_state.unsupervised_models
                    unsupervised_results = unsupervised.run_models(features_df)
                
                model_results["unsupervised"] = unsupervised_results
                progress_bar.progress(65)
                
                # Save state after unsupervised models
                st.session_state.model_results = model_results
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Unsupervised Models")
            
            # Supervised models
            if settings["models"]["Supervised Models"]:
                log_analysis_progress("Running supervised models...", 75)
                if st.session_state.supervised_models is None:
                    supervised = SupervisedModels(test_size=settings["test_size"])
                    supervised_results = supervised.run_models(features_df)
                    st.session_state.supervised_models = supervised
                else:
                    supervised = st.session_state.supervised_models
                    supervised_results = supervised.run_models(features_df)
                
                model_results["supervised"] = supervised_results
                progress_bar.progress(75)
                
                # Save state after supervised models
                st.session_state.model_results = model_results
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Supervised Models")
            
            # Rule-based models
            if settings["models"]["Rule-Based Models"]:
                log_analysis_progress("Running rule-based models...", 85)
                if st.session_state.rule_engine is None:
                    rule_engine = RuleEngine(threshold=settings["rule_threshold"])
                    rule_results = rule_engine.apply_rules(features_df)
                    st.session_state.rule_engine = rule_engine
                else:
                    rule_engine = st.session_state.rule_engine
                    rule_results = rule_engine.apply_rules(features_df)
                
                model_results["rule"] = rule_results
                progress_bar.progress(85)
                
                # Save state after rule-based models
                st.session_state.model_results = model_results
                if is_cache_available():
                    save_state_to_cache(st.session_state.file_hash, st.session_state, "Rule-Based Models")
            
            # Step 3: Risk Scoring
            log_analysis_progress("Step 3/5: Calculating Risk Scores...", 90)
            
            # Initialize risk scorer with properly formatted method
            scoring_method = settings["scoring_method"]
            risk_scorer = RiskScorer(
                method=scoring_method,
                custom_weights=settings["custom_weights"]
            )
            
            risk_scores = risk_scorer.calculate_scores(model_results, features_df)
            progress_bar.progress(90)
            
            # Save state after risk scoring
            st.session_state.risk_scores = risk_scores
            if is_cache_available():
                save_state_to_cache(st.session_state.file_hash, st.session_state, "Risk Scoring")
            
            # Step 4: Apply Thresholds
            log_analysis_progress("Step 4/5: Applying Thresholds...", 95)
            if settings['threshold_method'] == "Fixed":
                threshold = settings['threshold']
                risk_scores["is_fraud"] = risk_scores["risk_score"] > threshold
            elif settings['threshold_method'] == "Percentile":
                percentile = settings['percentile']
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
            log_analysis_progress("Step 5/5: Generating Explanations...", 99)
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
            st.session_state.current_step = "Results Dashboard"
            
            # Save state to cache
            if is_cache_available():
                save_state_to_cache(st.session_state.file_hash, st.session_state, "Run Detection")
            
            log_analysis_progress("Analysis complete!", 100)
            status_text.text("Analysis complete!")
            st.success(f"Fraud detection analysis complete! Found {risk_scores['is_fraud'].sum()} potentially fraudulent transactions out of {len(risk_scores)} total.")
            
        except Exception as e:
            log_analysis_progress(f"Error during analysis: {str(e)}")
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
        # Create a display dataframe with risk scores and available details
        display_data = top_risky.copy()
        
        # Try to add transaction details from original data if available
        if hasattr(st.session_state, 'original_data') and st.session_state.original_data is not None:
            original_data = st.session_state.original_data
            
            # Get columns that exist in both dataframes
            common_cols = set(original_data.columns) & set(display_data.columns)
            
            # If there are common columns, merge them
            if common_cols:
                display_data = display_data.drop(columns=list(common_cols), errors='ignore')
                display_data = display_data.merge(
                    original_data[list(common_cols)],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
        
        # If we still don't have transaction_id, try to get it from features_df
        if 'transaction_id' not in display_data.columns and hasattr(st.session_state, 'features_df'):
            features_df = st.session_state.features_df
            if 'transaction_id' in features_df.columns:
                display_data = display_data.merge(
                    features_df[['transaction_id']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
        
        # If we still don't have transaction_id, use the index as a fallback
        if 'transaction_id' not in display_data.columns:
            display_data['transaction_id'] = display_data.index
        
        # Ensure we have the required columns for display
        display_cols = []
        
        # Always include transaction_id and risk_score
        if 'transaction_id' in display_data.columns:
            display_cols.append('transaction_id')
        display_cols.append('risk_score')
        
        # Add other available columns if they exist
        for col in ['amount', 'timestamp', 'sender_id', 'receiver_id']:
            if col in display_data.columns:
                display_cols.append(col)
        
        # Display the dataframe
        st.dataframe(
            display_data[display_cols].sort_values('risk_score', ascending=False)
        )
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
    if hasattr(st.session_state, 'original_data') and st.session_state.original_data is not None and 'timestamp' in st.session_state.original_data.columns:
        st.subheader("Fraud Over Time")
        
        # Create a copy for time series analysis
        ts_data = st.session_state.original_data.copy()
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
    elif hasattr(st.session_state, 'features_df') and st.session_state.features_df is not None and 'timestamp' in st.session_state.features_df.columns:
        st.subheader("Fraud Over Time")
        
        # Create a copy for time series analysis
        ts_data = st.session_state.features_df.copy()
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
    else:
        st.info("Timestamp data not available for time series analysis")
    
    # Display cache information
    display_cache_info()

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
    # Check if we have transaction_id in features_df
    if hasattr(st.session_state, 'features_df') and st.session_state.features_df is not None and 'transaction_id' in st.session_state.features_df.columns:
        # Merge with features to get transaction IDs
        fraud_with_ids = fraud_transactions.copy()
        fraud_with_ids = fraud_with_ids.merge(
            st.session_state.features_df[['transaction_id']],
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Create transaction options
        transaction_options = []
        for idx in fraud_with_ids.index:
            # Check if transaction_id exists and is not NaN
            if 'transaction_id' in fraud_with_ids.columns and pd.notna(fraud_with_ids.loc[idx, 'transaction_id']):
                transaction_id = fraud_with_ids.loc[idx, 'transaction_id']
                transaction_options.append(f"{transaction_id} (Index: {idx})")
            else:
                transaction_options.append(f"Index: {idx}")
        
        selected_transaction = st.selectbox("Select a transaction to analyze", transaction_options)
        
        # Parse the selected option to get the index
        if "Index:" in selected_transaction:
            selected_index = int(selected_transaction.split("Index: ")[1])
        else:
            # Extract transaction_id from the selected option
            transaction_id = selected_transaction.split(" (Index: ")[0]
            # Find the index for this transaction_id
            selected_index = fraud_with_ids[fraud_with_ids['transaction_id'] == transaction_id].index[0]
    else:
        # Use index as identifier
        transaction_options = [f"Index: {idx}" for idx in fraud_transactions.index]
        selected_transaction = st.selectbox("Select a transaction to analyze", transaction_options)
        selected_index = int(selected_transaction.split("Index: ")[1])
    
    # Display transaction details
    st.write("### Transaction Details")
    
    # Try to get transaction details from features_df
    if hasattr(st.session_state, 'features_df') and st.session_state.features_df is not None:
        try:
            transaction_data = st.session_state.features_df.loc[selected_index]
            
            # Display key fields
            key_fields = ['transaction_id', 'amount', 'timestamp', 'sender_id', 'receiver_id', 
                         'transaction_type', 'transaction_category', 'description']
            
            for field in key_fields:
                if field in transaction_data:
                    st.write(f"**{field.replace('_', ' ').title()}:** {transaction_data[field]}")
        except Exception as e:
            st.warning(f"Could not retrieve transaction details: {str(e)}")
    
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
        top_factors = [factor[0] for factor in explanations[selected_index]['top_factors'][:5]]
        
        # Create sliders for top features
        adjustments = {}
        for feature in top_factors:
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
                stat_features = st.session_state.statistical_features
                modified_features = stat_features.transform(modified_features)
            
            # Graph features
            if st.session_state.settings["features"]["Graph Features"]:
                graph_features = st.session_state.graph_features
                modified_features = graph_features.transform(modified_features)
            
            # NLP features
            if st.session_state.settings["features"]["NLP Features"]:
                nlp_features = st.session_state.nlp_features
                modified_features = nlp_features.transform(modified_features)
            
            # Time series features
            if st.session_state.settings["features"]["Time Series Features"]:
                ts_features = st.session_state.timeseries_features
                modified_features = ts_features.transform(modified_features)
            
            # Get model predictions
            model_predictions = {}
            
            # Unsupervised models
            if 'unsupervised' in st.session_state.model_results:
                for model_name, model in st.session_state.unsupervised_models.models.items():
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
                for model_name, model in st.session_state.supervised_models.models.items():
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
                
                for rule_name, rule_func in st.session_state.rule_engine.rules.items():
                    try:
                        if rule_func(modified_data.iloc[0]):
                            rule_score += 1.0 / len(st.session_state.rule_engine.rules)
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
    
    # Add cache management to sidebar
    render_cache_management()
    
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