"""
Column Mapper Module
Handles intelligent mapping of user columns to expected format
"""
import pandas as pd
import numpy as np
import re
import logging
from collections import defaultdict
from difflib import SequenceMatcher
import yaml
import os
import pickle
import zlib
import hashlib
from typing import Dict, List, Tuple, Optional
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ColumnMapper:
    """
    Class for mapping user columns to expected format
    Uses AI-powered techniques to intelligently match columns
    """
    
    def __init__(self, config_path=None):
        """
        Initialize ColumnMapper
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.expected_columns = self._get_expected_columns()
        self.synonyms = self._load_synonyms(config_path)
        self.patterns = self._load_patterns(config_path)
        self.mapping_history = []
        
    def get_expected_columns(self):
        """
        Return the list of expected column names
        
        Returns:
            list: Expected column names
        """
        return list(self.expected_columns.keys())
    
    def _get_expected_columns(self):
        """
        Get the expected columns for the fraud detection system
        
        Returns:
            dict: Expected columns with descriptions
        """
        return {
            'transaction_id': {
                'description': 'Unique identifier for the transaction',
                'data_type': 'string',
                'required': True
            },
            'timestamp': {
                'description': 'Date and time of the transaction',
                'data_type': 'datetime',
                'required': True
            },
            'amount': {
                'description': 'Transaction amount',
                'data_type': 'float',
                'required': True
            },
            'currency': {
                'description': 'Currency code (e.g., USD, EUR)',
                'data_type': 'string',
                'required': False
            },
            'sender_id': {
                'description': 'Identifier of the sender',
                'data_type': 'string',
                'required': True
            },
            'receiver_id': {
                'description': 'Identifier of the receiver',
                'data_type': 'string',
                'required': True
            },
            'sender_account_type': {
                'description': 'Type of sender account (e.g., personal, business)',
                'data_type': 'string',
                'required': False
            },
            'receiver_account_type': {
                'description': 'Type of receiver account (e.g., personal, business)',
                'data_type': 'string',
                'required': False
            },
            'sender_bank': {
                'description': 'Name of sender bank',
                'data_type': 'string',
                'required': False
            },
            'receiver_bank': {
                'description': 'Name of receiver bank',
                'data_type': 'string',
                'required': False
            },
            'sender_location': {
                'description': 'Location of sender (country, state, city)',
                'data_type': 'string',
                'required': False
            },
            'receiver_location': {
                'description': 'Location of receiver (country, state, city)',
                'data_type': 'string',
                'required': False
            },
            'transaction_type': {
                'description': 'Type of transaction (e.g., transfer, payment)',
                'data_type': 'string',
                'required': False
            },
            'transaction_category': {
                'description': 'Category of transaction (e.g., retail, services)',
                'data_type': 'string',
                'required': False
            },
            'merchant_id': {
                'description': 'Identifier of the merchant',
                'data_type': 'string',
                'required': False
            },
            'merchant_category': {
                'description': 'Category of the merchant',
                'data_type': 'string',
                'required': False
            },
            'ip_address': {
                'description': 'IP address used for the transaction',
                'data_type': 'string',
                'required': False
            },
            'device_id': {
                'description': 'Identifier of the device used',
                'data_type': 'string',
                'required': False
            },
            'description': {
                'description': 'Description of the transaction',
                'data_type': 'string',
                'required': False
            },
            'notes': {
                'description': 'Additional notes',
                'data_type': 'string',
                'required': False
            },
            'authorization_status': {
                'description': 'Authorization status (e.g., approved, declined)',
                'data_type': 'string',
                'required': False
            },
            'chargeback_flag': {
                'description': 'Whether the transaction was charged back',
                'data_type': 'boolean',
                'required': False
            },
            'fraud_flag': {
                'description': 'Whether the transaction is fraudulent (for supervised learning)',
                'data_type': 'boolean',
                'required': False
            }
        }
    
    def _load_synonyms(self, config_path=None):
        """
        Load synonyms for column names
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            dict: Synonyms for expected columns
        """
        # Default synonyms
        synonyms = {
            'transaction_id': ['id', 'transaction_id', 'tx_id', 'trans_id', 'reference', 'ref_no', 'transaction_no'],
            'timestamp': ['timestamp', 'date', 'time', 'datetime', 'trans_date', 'trans_time', 'transaction_date', 'transaction_time'],
            'amount': ['amount', 'value', 'sum', 'total', 'transaction_amount', 'amt', 'tx_amount'],
            'currency': ['currency', 'curr', 'ccy', 'currency_code'],
            'sender_id': ['sender_id', 'from_id', 'payer_id', 'source_id', 'originator_id'],
            'receiver_id': ['receiver_id', 'to_id', 'payee_id', 'destination_id', 'beneficiary_id'],
            'sender_account_type': ['sender_account_type', 'from_account_type', 'payer_account_type', 'source_account_type'],
            'receiver_account_type': ['receiver_account_type', 'to_account_type', 'payee_account_type', 'destination_account_type'],
            'sender_bank': ['sender_bank', 'from_bank', 'payer_bank', 'source_bank'],
            'receiver_bank': ['receiver_bank', 'to_bank', 'payee_bank', 'destination_bank'],
            'sender_location': ['sender_location', 'from_location', 'payer_location', 'source_location', 'sender_country', 'from_country'],
            'receiver_location': ['receiver_location', 'to_location', 'payee_location', 'destination_location', 'receiver_country', 'to_country'],
            'transaction_type': ['transaction_type', 'trans_type', 'type', 'tx_type'],
            'transaction_category': ['transaction_category', 'trans_category', 'category', 'tx_category'],
            'merchant_id': ['merchant_id', 'merchant', 'retailer_id', 'vendor_id'],
            'merchant_category': ['merchant_category', 'merchant_type', 'retailer_category', 'vendor_category'],
            'ip_address': ['ip_address', 'ip', 'ip_addr'],
            'device_id': ['device_id', 'device', 'device_identifier'],
            'description': ['description', 'desc', 'details', 'narrative'],
            'notes': ['notes', 'note', 'comments', 'remark'],
            'authorization_status': ['authorization_status', 'auth_status', 'status', 'approval_status'],
            'chargeback_flag': ['chargeback_flag', 'chargeback', 'is_chargeback'],
            'fraud_flag': ['fraud_flag', 'fraud', 'is_fraud', 'fraudulent']
        }
        
        # Try to load from config file
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'column_synonyms' in config:
                        # Update with config synonyms
                        for col, syn_list in config['column_synonyms'].items():
                            if col in synonyms:
                                synonyms[col].extend(syn_list)
                            else:
                                synonyms[col] = syn_list
            except Exception as e:
                logger.warning(f"Error loading synonyms from config: {str(e)}")
        
        return synonyms
    
    def _load_patterns(self, config_path=None):
        """
        Load regex patterns for column names
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            dict: Regex patterns for expected columns
        """
        # Default patterns
        patterns = {
            'transaction_id': [r'transaction.?id', r'tx.?id', r'trans.?id', r'reference', r'ref.?(no|num)'],
            'timestamp': [r'timestamp', r'date.?time', r'trans.?date', r'trans.?time'],
            'amount': [r'amount', r'value', r'sum', r'total'],
            'currency': [r'currency', r'ccy', r'curr'],
            'sender_id': [r'sender.?id', r'from.?id', r'payer.?id', r'source.?id', r'originator.?id'],
            'receiver_id': [r'receiver.?id', r'to.?id', r'payee.?id', r'destination.?id', r'beneficiary.?id'],
            'sender_account_type': [r'sender.?account.?type', r'from.?account.?type', r'payer.?account.?type'],
            'receiver_account_type': [r'receiver.?account.?type', r'to.?account.?type', r'payee.?account.?type'],
            'sender_bank': [r'sender.?bank', r'from.?bank', r'payer.?bank', r'source.?bank'],
            'receiver_bank': [r'receiver.?bank', r'to.?bank', r'payee.?bank', r'destination.?bank'],
            'sender_location': [r'sender.?location', r'from.?location', r'payer.?location', r'source.?location', r'sender.?country'],
            'receiver_location': [r'receiver.?location', r'to.?location', r'payee.?location', r'destination.?location', r'receiver.?country'],
            'transaction_type': [r'transaction.?type', r'trans.?type', r'tx.?type'],
            'transaction_category': [r'transaction.?category', r'trans.?category', r'tx.?category'],
            'merchant_id': [r'merchant.?id', r'merchant', r'retailer.?id', r'vendor.?id'],
            'merchant_category': [r'merchant.?category', r'merchant.?type', r'retailer.?category'],
            'ip_address': [r'ip.?address', r'ip'],
            'device_id': [r'device.?id', r'device'],
            'description': [r'description', r'desc', r'details', r'narrative'],
            'notes': [r'notes?', r'comments?', r'remarks?'],
            'authorization_status': [r'authorization.?status', r'auth.?status', r'approval.?status'],
            'chargeback_flag': [r'chargeback.?(flag|is)'],
            'fraud_flag': [r'fraud.?(flag|is)', r'is.?fraud']
        }
        
        # Try to load from config file
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'column_patterns' in config:
                        # Update with config patterns
                        for col, pat_list in config['column_patterns'].items():
                            if col in patterns:
                                patterns[col].extend(pat_list)
                            else:
                                patterns[col] = pat_list
            except Exception as e:
                logger.warning(f"Error loading patterns from config: {str(e)}")
        
        return patterns
    
    def auto_map_columns(self, user_columns, expected_columns=None):
        """
        Automatically map user columns to expected columns
        
        Args:
            user_columns (list): List of user column names
            expected_columns (list, optional): List of expected column names
            
        Returns:
            dict: Mapping from user columns to expected columns
        """
        if expected_columns is None:
            expected_columns = list(self.expected_columns.keys())
        
        mapping = {}
        used_columns = set()
        
        # First, try exact matches
        for user_col in user_columns:
            user_col_clean = self._clean_column_name(user_col)
            if user_col_clean in expected_columns and user_col_clean not in used_columns:
                mapping[user_col] = user_col_clean
                used_columns.add(user_col_clean)
        
        # Then, try synonym matches
        for user_col in user_columns:
            if user_col in mapping:
                continue
                
            user_col_clean = self._clean_column_name(user_col)
            
            for expected_col in expected_columns:
                if expected_col in used_columns:
                    continue
                    
                if user_col_clean in self.synonyms.get(expected_col, []):
                    mapping[user_col] = expected_col
                    used_columns.add(expected_col)
                    break
        
        # Then, try pattern matches
        for user_col in user_columns:
            if user_col in mapping:
                continue
                
            user_col_clean = self._clean_column_name(user_col)
            
            for expected_col in expected_columns:
                if expected_col in used_columns:
                    continue
                    
                for pattern in self.patterns.get(expected_col, []):
                    if re.search(pattern, user_col_clean, re.IGNORECASE):
                        mapping[user_col] = expected_col
                        used_columns.add(expected_col)
                        break
                else:
                    continue
                break
        
        # Finally, try fuzzy matching for remaining columns
        for user_col in user_columns:
            if user_col in mapping:
                continue
                
            user_col_clean = self._clean_column_name(user_col)
            
            best_match = None
            best_score = 0
            
            for expected_col in expected_columns:
                if expected_col in used_columns:
                    continue
                    
                # Calculate similarity score
                score = self._calculate_similarity(user_col_clean, expected_col)
                
                if score > best_score and score > 0.6:  # Threshold for fuzzy matching
                    best_score = score
                    best_match = expected_col
            
            if best_match:
                mapping[user_col] = best_match
                used_columns.add(best_match)
        
        # Store mapping history
        self.mapping_history.append({
            'timestamp': pd.Timestamp.now(),
            'user_columns': user_columns,
            'mapping': mapping
        })
        
        return mapping
    
    def _clean_column_name(self, column_name):
        """
        Clean column name for matching
        
        Args:
            column_name (str): Column name to clean
            
        Returns:
            str: Cleaned column name
        """
        # Convert to lowercase
        cleaned = column_name.lower()
        
        # Remove special characters and spaces
        cleaned = re.sub(r'[^a-z0-9]', '_', cleaned)
        
        # Remove consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Remove leading and trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def _calculate_similarity(self, str1, str2):
        """
        Calculate similarity between two strings
        
        Args:
            str1 (str): First string
            str2 (str): Second string
            
        Returns:
            float: Similarity score (0-1)
        """
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, str1, str2).ratio()
    
    def apply_mapping(self, df, mapping):
        """
        Apply column mapping to a DataFrame
        
        Args:
            df (DataFrame): Input DataFrame
            mapping (dict): Column mapping
            
        Returns:
            DataFrame: DataFrame with mapped columns
        """
        try:
            # Create a copy of the DataFrame
            result_df = df.copy()
            
            # Create a new DataFrame with mapped columns
            mapped_df = pd.DataFrame()
            
            # Map columns
            for user_col, expected_col in mapping.items():
                if user_col in df.columns:
                    mapped_df[expected_col] = df[user_col]
            
            # Add unmapped columns with original names
            for col in df.columns:
                if col not in mapping:
                    mapped_df[col] = df[col]
            
            logger.info(f"Column mapping applied successfully. Mapped {len(mapping)} columns.")
            return mapped_df
            
        except Exception as e:
            logger.error(f"Error applying column mapping: {str(e)}")
            raise
    
    def validate_mapping(self, mapping, required_columns=None):
        """
        Validate a column mapping
        
        Args:
            mapping (dict): Column mapping
            required_columns (list, optional): List of required columns
            
        Returns:
            dict: Validation results
        """
        if required_columns is None:
            required_columns = [col for col, info in self.expected_columns.items() if info['required']]
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_required": [],
            "missing_optional": []
        }
        
        # Check for required columns
        mapped_columns = set(mapping.values())
        
        for col in required_columns:
            if col not in mapped_columns:
                validation_result["valid"] = False
                validation_result["missing_required"].append(col)
        
        # Check for optional columns
        optional_columns = [col for col in self.expected_columns if col not in required_columns]
        
        for col in optional_columns:
            if col not in mapped_columns:
                validation_result["missing_optional"].append(col)
        
        # Generate error messages
        if validation_result["missing_required"]:
            validation_result["errors"].append(
                f"Missing required columns: {', '.join(validation_result['missing_required'])}"
            )
        
        # Generate warning messages
        if validation_result["missing_optional"]:
            validation_result["warnings"].append(
                f"Missing optional columns: {', '.join(validation_result['missing_optional'])}"
            )
        
        return validation_result
    
    def save_mapping_template(self, file_path, mapping=None):
        """
        Save a mapping template to file
        
        Args:
            file_path (str): Path to save the template
            mapping (dict, optional): Mapping to save
        """
        if mapping is None:
            mapping = {}
        
        try:
            template = {
                "expected_columns": self.expected_columns,
                "synonyms": self.synonyms,
                "patterns": self.patterns,
                "current_mapping": mapping
            }
            
            with open(file_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False)
            
            logger.info(f"Mapping template saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving mapping template: {str(e)}")
            raise
    
    def load_mapping_template(self, file_path):
        """
        Load a mapping template from file
        
        Args:
            file_path (str): Path to the template file
            
        Returns:
            dict: Loaded mapping
        """
        try:
            with open(file_path, 'r') as f:
                template = yaml.safe_load(f)
            
            # Update instance variables
            if 'expected_columns' in template:
                self.expected_columns = template['expected_columns']
            
            if 'synonyms' in template:
                self.synonyms = template['synonyms']
            
            if 'patterns' in template:
                self.patterns = template['patterns']
            
            logger.info(f"Mapping template loaded from {file_path}")
            
            return template.get('current_mapping', {})
            
        except Exception as e:
            logger.error(f"Error loading mapping template: {str(e)}")
            raise
    
    def get_mapping_suggestions(self, user_columns, expected_columns=None):
        """
        Get mapping suggestions with confidence scores
        
        Args:
            user_columns (list): List of user column names
            expected_columns (list, optional): List of expected column names
            
        Returns:
            dict: Mapping suggestions with confidence scores
        """
        if expected_columns is None:
            expected_columns = list(self.expected_columns.keys())
        
        suggestions = {}
        
        for user_col in user_columns:
            user_col_clean = self._clean_column_name(user_col)
            
            col_suggestions = []
            
            # Check for exact matches
            if user_col_clean in expected_columns:
                col_suggestions.append({
                    "column": user_col_clean,
                    "confidence": 1.0,
                    "method": "exact_match"
                })
            
            # Check for synonyms
            for expected_col in expected_columns:
                if user_col_clean in self.synonyms.get(expected_col, []):
                    col_suggestions.append({
                        "column": expected_col,
                        "confidence": 0.9,
                        "method": "synonym_match"
                    })
            
            # Check for pattern matches
            for expected_col in expected_columns:
                for pattern in self.patterns.get(expected_col, []):
                    if re.search(pattern, user_col_clean, re.IGNORECASE):
                        col_suggestions.append({
                            "column": expected_col,
                            "confidence": 0.8,
                            "method": "pattern_match"
                        })
            
            # Check for fuzzy matches
            for expected_col in expected_columns:
                similarity = self._calculate_similarity(user_col_clean, expected_col)
                if similarity > 0.6:
                    col_suggestions.append({
                        "column": expected_col,
                        "confidence": similarity,
                        "method": "fuzzy_match"
                    })
            
            # Sort by confidence
            col_suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            
            suggestions[user_col] = col_suggestions
        
        return suggestions
    
    def _serialize_for_cache(self):
        """
        Serialize column mapper state for caching
        
        Returns:
            dict: Serialized state
        """
        try:
            state = {
                'expected_columns': self.expected_columns,
                'synonyms': self.synonyms,
                'patterns': self.patterns,
                'mapping_history': self.mapping_history
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error serializing column mapper state: {str(e)}")
            return None
    
    def _deserialize_from_cache(self, state):
        """
        Deserialize column mapper state from cache
        
        Args:
            state (dict): Serialized state
            
        Returns:
            bool: True if successful
        """
        try:
            self.expected_columns = state.get('expected_columns', self._get_expected_columns())
            self.synonyms = state.get('synonyms', self._load_synonyms())
            self.patterns = state.get('patterns', self._load_patterns())
            self.mapping_history = state.get('mapping_history', [])
            
            return True
            
        except Exception as e:
            logger.error(f"Error deserializing column mapper state: {str(e)}")
            return False