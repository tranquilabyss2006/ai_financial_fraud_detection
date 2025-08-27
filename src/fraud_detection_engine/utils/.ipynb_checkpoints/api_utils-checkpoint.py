"""
API Utilities Module
Handles checking API availability and providing demo data
"""
import yaml
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)

def is_api_available(service_name):
    """
    Check if an API service is available
    
    Args:
        service_name (str): Name of the service (e.g., 'gemini', 'openai', 'news_api')
        
    Returns:
        bool: True if API is available, False otherwise
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', '..', 'config', 'api_keys.yml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config.get(service_name, {}).get('api_key', '')
        
        # Check if key is available and not a placeholder
        return api_key and api_key not in ["YOUR_" + service_name.upper() + "_API_KEY", "NOT_AVAILABLE"]
    except Exception as e:
        logger.warning(f"Error checking API availability for {service_name}: {str(e)}")
        return False

def get_demo_sanctions_data():
    """
    Get demo sanctions data for testing
    
    Returns:
        DataFrame: Demo sanctions data
    """
    try:
        # Create more realistic demo data
        demo_data = pd.DataFrame({
            'entity_id': ['ENT001', 'ENT002', 'ENT003', 'ENT004', 'ENT005'],
            'name': ['Global Trading Corp', 'International Investments Ltd', 'Pacific Holdings Inc', 
                    'Euro Finance Group', 'Asia Trading Company'],
            'country': ['North Korea', 'Iran', 'Syria', 'Russia', 'China'],
            'list_type': ['OFAC Sanctions', 'EU Sanctions', 'UN Sanctions', 'OFAC Sanctions', 'Special Measures'],
            'added_date': ['2023-01-01', '2023-02-15', '2023-03-10', '2023-04-05', '2023-05-20'],
            'risk_level': ['High', 'High', 'Medium', 'Medium', 'Low']
        })
        
        logger.info("Using demo sanctions data")
        return demo_data
    except Exception as e:
        logger.error(f"Error creating demo sanctions data: {str(e)}")
        return pd.DataFrame()

def get_demo_tax_compliance_data():
    """
    Get demo tax compliance data for testing
    
    Returns:
        DataFrame: Demo tax compliance data
    """
    try:
        # Create more realistic demo data
        demo_data = pd.DataFrame({
            'tax_id': ['TAX001', 'TAX002', 'TAX003', 'TAX004', 'TAX005'],
            'entity_name': ['Demo Corp 1', 'Demo Corp 2', 'Demo Corp 3', 'Demo Corp 4', 'Demo Corp 5'],
            'compliance_status': ['Compliant', 'Non-compliant', 'Under Review', 'Compliant', 'Non-compliant'],
            'last_checked': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
            'tax_jurisdiction': ['USA', 'UK', 'Germany', 'France', 'Japan'],
            'outstanding_amount': [0, 150000, 50000, 0, 75000]
        })
        
        logger.info("Using demo tax compliance data")
        return demo_data
    except Exception as e:
        logger.error(f"Error creating demo tax compliance data: {str(e)}")
        return pd.DataFrame()

def get_demo_bank_verification_data():
    """
    Get demo bank verification data for testing
    
    Returns:
        DataFrame: Demo bank verification data
    """
    try:
        # Create more realistic demo data
        demo_data = pd.DataFrame({
            'account_number': ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005'],
            'bank_name': ['Demo Bank 1', 'Demo Bank 2', 'Demo Bank 3', 'Demo Bank 4', 'Demo Bank 5'],
            'verification_status': ['Verified', 'Not Verified', 'Pending', 'Verified', 'Not Verified'],
            'last_verified': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
            'account_type': ['Checking', 'Savings', 'Business', 'Checking', 'Savings'],
            'risk_score': [0.1, 0.8, 0.5, 0.2, 0.9]
        })
        
        logger.info("Using demo bank verification data")
        return demo_data
    except Exception as e:
        logger.error(f"Error creating demo bank verification data: {str(e)}")
        return pd.DataFrame()

def get_demo_identity_verification_data():
    """
    Get demo identity verification data for testing
    
    Returns:
        DataFrame: Demo identity verification data
    """
    try:
        # Create more realistic demo data
        demo_data = pd.DataFrame({
            'id_number': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'verification_status': ['Verified', 'Not Verified', 'Pending', 'Verified', 'Not Verified'],
            'last_verified': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
            'id_type': ['Passport', 'Driver License', 'National ID', 'Passport', 'Driver License'],
            'country': ['USA', 'UK', 'Germany', 'France', 'Japan'],
            'risk_level': ['Low', 'High', 'Medium', 'Low', 'High']
        })
        
        logger.info("Using demo identity verification data")
        return demo_data
    except Exception as e:
        logger.error(f"Error creating demo identity verification data: {str(e)}")
        return pd.DataFrame()