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
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yml')
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
        # Create demo data
        demo_data = pd.DataFrame({
            'entity_id': ['ENT001', 'ENT002', 'ENT003'],
            'name': ['Demo Entity 1', 'Demo Entity 2', 'Demo Entity 3'],
            'country': ['North Korea', 'Iran', 'Syria'],
            'list_type': ['Sanctions List', 'Sanctions List', 'Sanctions List'],
            'added_date': ['2023-01-01', '2023-02-01', '2023-03-01']
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
        # Create demo data
        demo_data = pd.DataFrame({
            'tax_id': ['TAX001', 'TAX002', 'TAX003'],
            'entity_name': ['Demo Corp 1', 'Demo Corp 2', 'Demo Corp 3'],
            'compliance_status': ['Compliant', 'Non-compliant', 'Under Review'],
            'last_checked': ['2023-01-01', '2023-02-01', '2023-03-01']
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
        # Create demo data
        demo_data = pd.DataFrame({
            'account_number': ['ACC001', 'ACC002', 'ACC003'],
            'bank_name': ['Demo Bank 1', 'Demo Bank 2', 'Demo Bank 3'],
            'verification_status': ['Verified', 'Not Verified', 'Pending'],
            'last_verified': ['2023-01-01', '2023-02-01', '2023-03-01']
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
        # Create demo data
        demo_data = pd.DataFrame({
            'id_number': ['ID001', 'ID002', 'ID003'],
            'name': ['Demo Person 1', 'Demo Person 2', 'Demo Person 3'],
            'verification_status': ['Verified', 'Not Verified', 'Pending'],
            'last_verified': ['2023-01-01', '2023-02-01', '2023-03-01']
        })
        
        logger.info("Using demo identity verification data")
        return demo_data
    except Exception as e:
        logger.error(f"Error creating demo identity verification data: {str(e)}")
        return pd.DataFrame()