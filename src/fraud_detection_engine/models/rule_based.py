"""
Rule-based Models Module
Implements rule-based fraud detection
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import yaml
import os
import warnings
import logging
from typing import Dict, List, Tuple, Union, Callable

from fraud_detection_engine.utils.api_utils import is_api_available, get_demo_sanctions_data, get_demo_tax_compliance_data, get_demo_bank_verification_data, get_demo_identity_verification_data

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleEngine:
    """
    Class for rule-based fraud detection
    Implements configurable rules with weights
    """
    
    def __init__(self, config_path=None, threshold=0.7):
        """
        Initialize RuleEngine
        
        Args:
            config_path (str, optional): Path to configuration file
            threshold (float): Threshold for rule violation
        """
        self.threshold = threshold
        self.rules = {}
        self.rule_weights = {}
        self.rule_descriptions = {}
        self.fitted = False
        self.api_available = {}
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._load_default_rules()
    
    def _load_config(self, config_path=None):
        """
        Load configuration from YAML file"""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Use default config path
                config_path = 'config/rule_engine_config.yml'
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    config = {'rules': {}}
                        
            
            # Load rules
            if 'rules' in config:
                for rule_name, rule_config in config['rules'].items():
                    if rule_config.get('enabled', True):
                        self.rules[rule_name] = self._create_rule_function(rule_config)
                        self.rule_weights[rule_name] = rule_config.get('weight', 1.0)
                        self.rule_descriptions[rule_name] = rule_config.get('description', '')
            
            # Check API availability for rules that depend on external services
            self.api_available = {
                'sanctions': is_api_available('sanctions'),
                'tax_compliance': is_api_available('tax_compliance'),
                'bank_verification': is_api_available('bank_verification'),
                'identity_verification': is_api_available('identity_verification'),
                'geolocation': is_api_available('geolocation')
            }
            
            logger.info(f"API availability: {self.api_available}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self._load_default_rules()
    
    def _load_default_rules(self):
        """
        Load default rules
        """
        try:
            # Amount-based rules
            self.rules['high_amount'] = self._high_amount_rule
            self.rule_weights['high_amount'] = 0.3
            self.rule_descriptions['high_amount'] = "Transaction amount exceeds threshold"
            
            self.rules['unusual_amount_for_sender'] = self._unusual_amount_for_sender_rule
            self.rule_weights['unusual_amount_for_sender'] = 0.2
            self.rule_descriptions['unusual_amount_for_sender'] = "Amount is unusual for the sender"
            
            self.rules['unusual_amount_for_receiver'] = self._unusual_amount_for_receiver_rule
            self.rule_weights['unusual_amount_for_receiver'] = 0.2
            self.rule_descriptions['unusual_amount_for_receiver'] = "Amount is unusual for the receiver"
            
            self.rules['round_amount'] = self._round_amount_rule
            self.rule_weights['round_amount'] = 0.1
            self.rule_descriptions['round_amount'] = "Transaction amount is suspiciously round"
            
            # Frequency-based rules
            self.rules['high_frequency_sender'] = self._high_frequency_sender_rule
            self.rule_weights['high_frequency_sender'] = 0.3
            self.rule_descriptions['high_frequency_sender'] = "High transaction frequency from sender"
            
            self.rules['high_frequency_receiver'] = self._high_frequency_receiver_rule
            self.rule_weights['high_frequency_receiver'] = 0.3
            self.rule_descriptions['high_frequency_receiver'] = "High transaction frequency to receiver"
            
            self.rules['rapid_succession'] = self._rapid_succession_rule
            self.rule_weights['rapid_succession'] = 0.4
            self.rule_descriptions['rapid_succession'] = "Multiple transactions in rapid succession"
            
            # Location-based rules
            self.rules['cross_border'] = self._cross_border_rule
            self.rule_weights['cross_border'] = 0.3
            self.rule_descriptions['cross_border'] = "Transaction crosses international borders"
            
            self.rules['high_risk_country'] = self._high_risk_country_rule
            self.rule_weights['high_risk_country'] = 0.5
            self.rule_descriptions['high_risk_country'] = "Transaction involves high-risk country"
            
            self.rules['unusual_location_for_sender'] = self._unusual_location_for_sender_rule
            self.rule_weights['unusual_location_for_sender'] = 0.3
            self.rule_descriptions['unusual_location_for_sender'] = "Transaction from unusual location for sender"
            
            # Time-based rules
            self.rules['unusual_hour'] = self._unusual_hour_rule
            self.rule_weights['unusual_hour'] = 0.2
            self.rule_descriptions['unusual_hour'] = "Transaction during unusual hours"
            
            self.rules['weekend'] = self._weekend_rule
            self.rule_weights['weekend'] = 0.1
            self.rule_descriptions['weekend'] = "Transaction on weekend"
            
            # Identity-based rules
            self.rules['new_sender'] = self._new_sender_rule
            self.rule_weights['new_sender'] = 0.3
            self.rule_descriptions['new_sender'] = "First transaction from new sender"
            
            self.rules['new_receiver'] = self._new_receiver_rule
            self.rule_weights['new_receiver'] = 0.3
            self.rule_descriptions['new_receiver'] = "First transaction to new receiver"
            
            # External API-based rules
            self.rules['sanctions_check'] = self._sanctions_check_rule
            self.rule_weights['sanctions_check'] = 0.5
            self.rule_descriptions['sanctions_check'] = "Transaction involves sanctioned entities"
            
            self.rules['tax_compliance'] = self._tax_compliance_rule
            self.rule_weights['tax_compliance'] = 0.3
            self.rule_descriptions['tax_compliance'] = "Transaction involves non-compliant entities"
            
            self.rules['bank_verification'] = self._bank_verification_rule
            self.rule_weights['bank_verification'] = 0.3
            self.rule_descriptions['bank_verification'] = "Transaction involves unverified bank accounts"
            
            self.rules['identity_verification'] = self._identity_verification_rule
            self.rule_weights['identity_verification'] = 0.3
            self.rule_descriptions['identity_verification'] = "Transaction involves unverified identities"
            
            # Check API availability
            self.api_available = {
                'sanctions': is_api_available('sanctions'),
                'tax_compliance': is_api_available('tax_compliance'),
                'bank_verification': is_api_available('bank_verification'),
                'identity_verification': is_api_available('identity_verification'),
                'geolocation': is_api_available('geolocation')
            }
            
            logger.info(f"API availability: {self.api_available}")
            
        except Exception as e:
            logger.error(f"Error loading default rules: {str(e)}")
            raise
    
    def _create_rule_function(self, rule_config):
        """
        Create a rule function from configuration
        
        Args:
            rule_config (dict): Rule configuration
            
        Returns:
            function: Rule function
        """
        rule_type = rule_config.get('type')
        
        if rule_type == 'amount_threshold':
            threshold = rule_config.get('threshold', 10000)
            return lambda row: row.get('amount', 0) > threshold
        
        elif rule_type == 'sender_amount_outlier':
            std_multiplier = rule_config.get('std_multiplier', 3)
            min_transactions = rule_config.get('min_transactions', 5)
            return lambda row: self._is_sender_amount_outlier(row, std_multiplier, min_transactions)
        
        elif rule_type == 'receiver_amount_outlier':
            std_multiplier = rule_config.get('std_multiplier', 3)
            min_transactions = rule_config.get('min_transactions', 5)
            return lambda row: self._is_receiver_amount_outlier(row, std_multiplier, min_transactions)
        
        elif rule_type == 'round_amount':
            threshold = rule_config.get('threshold', 1000)
            return lambda row: row.get('amount', 0) > threshold and row.get('amount', 0) % 1000 == 0
        
        elif rule_type == 'high_frequency_sender':
            time_window = rule_config.get('time_window', '1H')
            max_transactions = rule_config.get('max_transactions', 10)
            return lambda row: self._is_high_frequency_sender(row, time_window, max_transactions)
        
        elif rule_type == 'high_frequency_receiver':
            time_window = rule_config.get('time_window', '1H')
            max_transactions = rule_config.get('max_transactions', 10)
            return lambda row: self._is_high_frequency_receiver(row, time_window, max_transactions)
        
        elif rule_type == 'rapid_succession':
            time_window = rule_config.get('time_window', '5M')
            min_transactions = rule_config.get('min_transactions', 3)
            return lambda row: self._is_rapid_succession(row, time_window, min_transactions)
        
        elif rule_type == 'cross_border':
            return lambda row: self._is_cross_border(row)
        
        elif rule_type == 'high_risk_country':
            countries = rule_config.get('countries', ['North Korea', 'Iran', 'Syria', 'Cuba'])
            return lambda row: self._is_high_risk_country(row, countries)
        
        elif rule_type == 'unusual_location_for_sender':
            return lambda row: self._is_unusual_location_for_sender(row)
        
        elif rule_type == 'unusual_hour':
            start_hour = rule_config.get('start_hour', 23)
            end_hour = rule_config.get('end_hour', 5)
            return lambda row: self._is_unusual_hour(row, start_hour, end_hour)
        
        elif rule_type == 'weekend':
            return lambda row: self._is_weekend(row)
        
        elif rule_type == 'new_sender':
            time_window = rule_config.get('time_window', '7D')
            return lambda row: self._is_new_sender(row, time_window)
        
        elif rule_type == 'new_receiver':
            time_window = rule_config.get('time_window', '7D')
            return lambda row: self._is_new_receiver(row, time_window)
        
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return lambda row: False
    
    def apply_rules(self, df):
        """
        Apply all rules to the dataframe
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            dict: Rule results
        """
        try:
            # Sort by timestamp if available
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Initialize results
            rule_results = {}
            rule_scores = {}
            violated_rules = {}
            
            # Apply each rule
            for rule_name, rule_func in self.rules.items():
                try:
                    # Apply rule to each row
                    results = []
                    for _, row in df.iterrows():
                        try:
                            result = rule_func(row)
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Error applying rule {rule_name} to row: {str(e)}")
                            results.append(False)
                    
                    rule_results[rule_name] = results
                    
                    # Calculate weighted score
                    weight = self.rule_weights.get(rule_name, 1.0)
                    rule_scores[rule_name] = [weight if result else 0 for result in results]
                    
                    # Track violated rules
                    violated_rules[rule_name] = [i for i, result in enumerate(results) if result]
                    
                except Exception as e:
                    logger.error(f"Error applying rule {rule_name}: {str(e)}")
                    rule_results[rule_name] = [False] * len(df)
                    rule_scores[rule_name] = [0] * len(df)
                    violated_rules[rule_name] = []
            
            # Calculate total rule score for each transaction
            total_scores = []
            for i in range(len(df)):
                score = sum(rule_scores[rule_name][i] for rule_name in rule_scores)
                total_scores.append(score)
            
            # Normalize scores
            max_possible_score = sum(self.rule_weights.values())
            normalized_scores = [score / max_possible_score for score in total_scores]
            
            # Determine if transaction violates rules based on threshold
            rule_violations = [score >= self.threshold for score in normalized_scores]
            
            # Get violated rule names for each transaction
            violated_rule_names = []
            for i in range(len(df)):
                names = []
                for rule_name in violated_rules:
                    if i in violated_rules[rule_name]:
                        names.append(rule_name)
                violated_rule_names.append(names)
            
            self.fitted = True
            
            return {
                'rule_results': rule_results,
                'rule_scores': rule_scores,
                'total_scores': total_scores,
                'normalized_scores': normalized_scores,
                'rule_violations': rule_violations,
                'violated_rule_names': violated_rule_names,
                'rules': self.rules,
                'rule_weights': self.rule_weights,
                'rule_descriptions': self.rule_descriptions
            }
            
        except Exception as e:
            logger.error(f"Error applying rules: {str(e)}")
            raise
    
    # Rule functions
    def _high_amount_rule(self, row):
        """Check if transaction amount is high"""
        return row.get('amount', 0) > 10000
    
    def _unusual_amount_for_sender_rule(self, row, std_multiplier=3, min_transactions=5):
        """Check if amount is unusual for the sender"""
        # This would require access to sender's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _unusual_amount_for_receiver_rule(self, row, std_multiplier=3, min_transactions=5):
        """Check if amount is unusual for the receiver"""
        # This would require access to receiver's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _round_amount_rule(self, row):
        """Check if transaction amount is suspiciously round"""
        amount = row.get('amount', 0)
        return amount > 1000 and amount % 1000 == 0
    
    def _high_frequency_sender_rule(self, row, time_window='1H', max_transactions=10):
        """Check if sender has high transaction frequency"""
        # This would require access to sender's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _high_frequency_receiver_rule(self, row, time_window='1H', max_transactions=10):
        """Check if receiver has high transaction frequency"""
        # This would require access to receiver's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _rapid_succession_rule(self, row, time_window='5M', min_transactions=3):
        """Check if there are multiple transactions in rapid succession"""
        # This would require access to transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _cross_border_rule(self, row):
        """Check if transaction crosses international borders"""
        sender_location = row.get('sender_location', '')
        receiver_location = row.get('receiver_location', '')
        
        if not sender_location or not receiver_location:
            return False
        
        # Extract countries from locations
        sender_country = sender_location.split(',')[0].strip()
        receiver_country = receiver_location.split(',')[0].strip()
        
        return sender_country != receiver_country
    
    def _high_risk_country_rule(self, row, countries=None):
        """Check if transaction involves high-risk country"""
        if countries is None:
            countries = ['North Korea', 'Iran', 'Syria', 'Cuba']
        
        sender_location = row.get('sender_location', '')
        receiver_location = row.get('receiver_location', '')
        
        # Check if any high-risk country is involved
        for country in countries:
            if country in sender_location or country in receiver_location:
                return True
        
        return False
    
    def _unusual_location_for_sender_rule(self, row):
        """Check if transaction is from unusual location for sender"""
        # This would require access to sender's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _unusual_hour_rule(self, row, start_hour=23, end_hour=5):
        """Check if transaction is during unusual hours"""
        timestamp = row.get('timestamp')
        if timestamp is None:
            return False
        
        if not pd.api.types.is_datetime64_any_dtype(timestamp):
            timestamp = pd.to_datetime(timestamp)
        
        hour = timestamp.hour
        
        # Handle overnight range
        if start_hour > end_hour:
            return hour >= start_hour or hour < end_hour
        else:
            return start_hour <= hour < end_hour
    
    def _weekend_rule(self, row):
        """Check if transaction is on weekend"""
        timestamp = row.get('timestamp')
        if timestamp is None:
            return False
        
        if not pd.api.types.is_datetime64_any_dtype(timestamp):
            timestamp = pd.to_datetime(timestamp)
        
        # Saturday=5, Sunday=6
        return timestamp.dayofweek >= 5
    
    def _new_sender_rule(self, row, time_window='7D'):
        """Check if this is the first transaction from sender"""
        # This would require access to sender's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _new_receiver_rule(self, row, time_window='7D'):
        """Check if this is the first transaction to receiver"""
        # This would require access to receiver's transaction history
        # For simplicity, we'll use a placeholder
        return False
    
    def _sanctions_check_rule(self, row):
        """Check if transaction involves sanctioned entities"""
        try:
            if not self.api_available.get('sanctions', False):
                # Use demo data
                sanctions_data = get_demo_sanctions_data()
                
                # Simple demo check
                sender_id = row.get('sender_id', '')
                receiver_id = row.get('receiver_id', '')
                
                # Check if sender or receiver is in demo sanctions list
                is_sanctioned = (
                    sanctions_data['entity_id'].str.contains(sender_id, na=False).any() or
                    sanctions_data['entity_id'].str.contains(receiver_id, na=False).any()
                )
                
                return is_sanctioned
            else:
                # Here you would implement the actual API call
                # For now, return False
                return False
        except Exception as e:
            logger.warning(f"Error in sanctions check: {str(e)}")
            return False
    
    def _tax_compliance_rule(self, row):
        """Check if entities are tax compliant"""
        try:
            if not self.api_available.get('tax_compliance', False):
                # Use demo data
                tax_data = get_demo_tax_compliance_data()
                
                # Simple demo check
                sender_id = row.get('sender_id', '')
                receiver_id = row.get('receiver_id', '')
                
                # Check if sender or receiver is non-compliant
                is_non_compliant = False
                
                for _, entity in tax_data.iterrows():
                    if entity['entity_id'] in [sender_id, receiver_id] and entity['compliance_status'] == 'Non-compliant':
                        is_non_compliant = True
                        break
                
                return is_non_compliant
            else:
                # Here you would implement the actual API call
                # For now, return False
                return False
        except Exception as e:
            logger.warning(f"Error in tax compliance check: {str(e)}")
            return False
    
    def _bank_verification_rule(self, row):
        """Check if bank accounts are verified"""
        try:
            if not self.api_available.get('bank_verification', False):
                # Use demo data
                bank_data = get_demo_bank_verification_data()
                
                # Simple demo check
                sender_id = row.get('sender_id', '')
                receiver_id = row.get('receiver_id', '')
                
                # Check if sender or receiver account is not verified
                is_unverified = False
                
                for _, account in bank_data.iterrows():
                    if account['account_number'] in [sender_id, receiver_id] and account['verification_status'] == 'Not Verified':
                        is_unverified = True
                        break
                
                return is_unverified
            else:
                # Here you would implement the actual API call
                # For now, return False
                return False
        except Exception as e:
            logger.warning(f"Error in bank verification check: {str(e)}")
            return False
    
    def _identity_verification_rule(self, row):
        """Check if identities are verified"""
        try:
            if not self.api_available.get('identity_verification', False):
                # Use demo data
                identity_data = get_demo_identity_verification_data()
                
                # Simple demo check
                sender_id = row.get('sender_id', '')
                receiver_id = row.get('receiver_id', '')
                
                # Check if sender or receiver identity is not verified
                is_unverified = False
                
                for _, identity in identity_data.iterrows():
                    if identity['id_number'] in [sender_id, receiver_id] and identity['verification_status'] == 'Not Verified':
                        is_unverified = True
                        break
                
                return is_unverified
            else:
                # Here you would implement the actual API call
                # For now, return False
                return False
        except Exception as e:
            logger.warning(f"Error in identity verification check: {str(e)}")
            return False
    
    def add_rule(self, rule_name, rule_func, weight=1.0, description=''):
        """
        Add a custom rule
        
        Args:
            rule_name (str): Name of the rule
            rule_func (function): Rule function
            weight (float): Weight of the rule
            description (str): Description of the rule
        """
        self.rules[rule_name] = rule_func
        self.rule_weights[rule_name] = weight
        self.rule_descriptions[rule_name] = description
        logger.info(f"Added rule: {rule_name}")
    
    def remove_rule(self, rule_name):
        """
        Remove a rule
        
        Args:
            rule_name (str): Name of the rule to remove
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            del self.rule_weights[rule_name]
            del self.rule_descriptions[rule_name]
            logger.info(f"Removed rule: {rule_name}")
        else:
            logger.warning(f"Rule not found: {rule_name}")
    
    def update_rule_weight(self, rule_name, weight):
        """
        Update the weight of a rule
        
        Args:
            rule_name (str): Name of the rule
            weight (float): New weight
        """
        if rule_name in self.rule_weights:
            self.rule_weights[rule_name] = weight
            logger.info(f"Updated weight for rule {rule_name}: {weight}")
        else:
            logger.warning(f"Rule not found: {rule_name}")
    
    def get_rules(self):
        """
        Get all rules
        
        Returns:
            dict: Rules information
        """
        return {
            'rules': list(self.rules.keys()),
            'weights': self.rule_weights,
            'descriptions': self.rule_descriptions
        }
    
    def save_config(self, config_path):
        """
        Save current configuration to file
        
        Args:
            config_path (str): Path to save configuration
        """
        try:
            config = {
                'rules': {}
            }
            
            for rule_name in self.rules:
                config['rules'][rule_name] = {
                    'enabled': True,
                    'weight': self.rule_weights[rule_name],
                    'description': self.rule_descriptions[rule_name]
                }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise