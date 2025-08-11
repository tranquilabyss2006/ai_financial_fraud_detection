"""
Risk Scorer Module
Implements risk scoring for fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
import logging
from typing import Dict, List, Tuple, Union
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskScorer:
    """
    Class for calculating risk scores from multiple models
    Implements weighted combination of algorithm outputs
    """
    
    def __init__(self, method='weighted_average', custom_weights=None):
        """
        Initialize RiskScorer
        
        Args:
            method (str): Scoring method ('weighted_average', 'maximum', 'custom')
            custom_weights (dict, optional): Custom weights for models
        """
        # Convert method to lowercase and replace spaces with underscores
        self.method = method.lower().replace(" ", "_")
        self.custom_weights = custom_weights or {
            'unsupervised': 0.4,
            'supervised': 0.4,
            'rule': 0.2
        }
        self.scalers = {}
        self.calibrators = {}
        self.fitted = False
    
    def calculate_scores(self, model_results, df):
        """
        Calculate risk scores from model results
        
        Args:
            model_results (dict): Results from different models
            df (DataFrame): Original data
            
        Returns:
            DataFrame: Risk scores
        """
        try:
            # Initialize result DataFrame
            result_df = df.copy()
            
            # Extract scores from each model type
            unsupervised_scores = None
            supervised_scores = None
            rule_scores = None
            
            # Process unsupervised model results
            if 'unsupervised' in model_results:
                unsupervised_scores = self._process_unsupervised_results(model_results['unsupervised'])
            
            # Process supervised model results
            if 'supervised' in model_results:
                supervised_scores = self._process_supervised_results(model_results['supervised'])
            
            # Process rule-based results
            if 'rule' in model_results:
                rule_scores = self._process_rule_results(model_results['rule'])
            
            # Combine scores based on method
            if self.method == 'weighted_average':
                combined_scores = self._weighted_average_combination(
                    unsupervised_scores, supervised_scores, rule_scores
                )
            elif self.method == 'maximum':
                combined_scores = self._maximum_combination(
                    unsupervised_scores, supervised_scores, rule_scores
                )
            elif self.method == 'custom':
                combined_scores = self._custom_combination(
                    unsupervised_scores, supervised_scores, rule_scores
                )
            else:
                raise ValueError(f"Unknown scoring method: {self.method}")
            
            # Add combined scores to result
            result_df['risk_score'] = combined_scores
            
            # Add individual model scores
            if unsupervised_scores is not None:
                result_df['unsupervised_score'] = unsupervised_scores
            
            if supervised_scores is not None:
                result_df['supervised_score'] = supervised_scores
            
            if rule_scores is not None:
                result_df['rule_score'] = rule_scores
            
            # Calculate percentile rank
            result_df['risk_percentile'] = result_df['risk_score'].rank(pct=True)
            
            # Calculate risk level
            result_df['risk_level'] = pd.cut(
                result_df['risk_percentile'],
                bins=[0, 0.7, 0.9, 0.95, 1.0],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            self.fitted = True
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating risk scores: {str(e)}")
            raise
    
    def _process_unsupervised_results(self, unsupervised_results):
        """
        Process unsupervised model results
        
        Args:
            unsupervised_results (dict): Results from unsupervised models
            
        Returns:
            array: Combined unsupervised scores
        """
        try:
            # Collect scores from all unsupervised models
            all_scores = []
            model_names = []
            
            for model_name, model_data in unsupervised_results.items():
                if 'scores' in model_data:
                    scores = model_data['scores']
                    all_scores.append(scores)
                    model_names.append(model_name)
            
            if not all_scores:
                return None
            
            # Convert to numpy array
            scores_array = np.array(all_scores).T
            
            # Calculate average score
            avg_scores = np.mean(scores_array, axis=1)
            
            # Normalize to 0-1 range
            scaler = MinMaxScaler()
            normalized_scores = scaler.fit_transform(avg_scores.reshape(-1, 1)).flatten()
            
            # Store scaler
            self.scalers['unsupervised'] = scaler
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error processing unsupervised results: {str(e)}")
            raise
    
    def _process_supervised_results(self, supervised_results):
        """
        Process supervised model results
        
        Args:
            supervised_results (dict): Results from supervised models
            
        Returns:
            array: Combined supervised scores
        """
        try:
            # Collect probabilities from all supervised models
            all_probabilities = []
            model_names = []
            
            for model_name, model_data in supervised_results.items():
                if 'probabilities' in model_data:
                    probabilities = model_data['probabilities']
                    all_probabilities.append(probabilities)
                    model_names.append(model_name)
            
            if not all_probabilities:
                return None
            
            # Convert to numpy array
            probabilities_array = np.array(all_probabilities).T
            
            # Calculate average probability
            avg_probabilities = np.mean(probabilities_array, axis=1)
            
            # Apply calibration if available
            if 'supervised' in self.calibrators:
                calibrated_scores = self.calibrators['supervised'].predict_proba(avg_probabilities.reshape(-1, 1))[:, 1]
            else:
                calibrated_scores = avg_probabilities
            
            # Store calibrator
            self.calibrators['supervised'] = CalibratedClassifierCV(
                method='isotonic', cv='prefit'
            )
            
            return calibrated_scores
            
        except Exception as e:
            logger.error(f"Error processing supervised results: {str(e)}")
            raise
    
    def _process_rule_results(self, rule_results):
        """
        Process rule-based results
        
        Args:
            rule_results (dict): Results from rule engine
            
        Returns:
            array: Combined rule scores
        """
        try:
            if 'normalized_scores' in rule_results:
                rule_scores = rule_results['normalized_scores']
                # Ensure rule_scores is a numpy array
                if not isinstance(rule_scores, np.ndarray):
                    rule_scores = np.array(rule_scores)
            else:
                return None
            
            # No calibration for now to avoid the reshape issue
            calibrated_scores = rule_scores
            
            # Store calibrator
            self.calibrators['rule'] = CalibratedClassifierCV(
                method='isotonic', cv='prefit'
            )
            
            return calibrated_scores
            
        except Exception as e:
            logger.error(f"Error processing rule results: {str(e)}")
            raise
    
    def _weighted_average_combination(self, unsupervised_scores, supervised_scores, rule_scores):
        """
        Combine scores using weighted average
        
        Args:
            unsupervised_scores (array): Unsupervised model scores
            supervised_scores (array): Supervised model scores
            rule_scores (array): Rule-based scores
            
        Returns:
            array: Combined scores
        """
        try:
            # Get weights
            unsupervised_weight = self.custom_weights.get('unsupervised', 0.4)
            supervised_weight = self.custom_weights.get('supervised', 0.4)
            rule_weight = self.custom_weights.get('rule', 0.2)
            
            # Normalize weights based on available scores
            available_weights = []
            if unsupervised_scores is not None:
                available_weights.append(unsupervised_weight)
            if supervised_scores is not None:
                available_weights.append(supervised_weight)
            if rule_scores is not None:
                available_weights.append(rule_weight)
            
            total_weight = sum(available_weights)
            if total_weight > 0:
                if unsupervised_scores is not None:
                    unsupervised_weight /= total_weight
                if supervised_scores is not None:
                    supervised_weight /= total_weight
                if rule_scores is not None:
                    rule_weight /= total_weight
            
            # Determine the length of the combined scores array
            if unsupervised_scores is not None:
                length = len(unsupervised_scores)
            elif supervised_scores is not None:
                length = len(supervised_scores)
            elif rule_scores is not None:
                length = len(rule_scores)
            else:
                # If no scores available, return empty array
                return np.array([])
            
            # Initialize combined scores
            combined_scores = np.zeros(length)
            
            # Add weighted scores
            if unsupervised_scores is not None:
                # Ensure unsupervised_scores is a numpy array
                if not isinstance(unsupervised_scores, np.ndarray):
                    unsupervised_scores = np.array(unsupervised_scores)
                combined_scores += unsupervised_weight * unsupervised_scores
            
            if supervised_scores is not None:
                # Ensure supervised_scores is a numpy array
                if not isinstance(supervised_scores, np.ndarray):
                    supervised_scores = np.array(supervised_scores)
                combined_scores += supervised_weight * supervised_scores
            
            if rule_scores is not None:
                # Ensure rule_scores is a numpy array
                if not isinstance(rule_scores, np.ndarray):
                    rule_scores = np.array(rule_scores)
                combined_scores += rule_weight * rule_scores
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error in weighted average combination: {str(e)}")
            raise
    
    def _maximum_combination(self, unsupervised_scores, supervised_scores, rule_scores):
        """
        Combine scores using maximum
        
        Args:
            unsupervised_scores (array): Unsupervised model scores
            supervised_scores (array): Supervised model scores
            rule_scores (array): Rule-based scores
            
        Returns:
            array: Combined scores
        """
        try:
            # Collect all available scores
            all_scores = []
            
            if unsupervised_scores is not None:
                all_scores.append(unsupervised_scores)
            
            if supervised_scores is not None:
                all_scores.append(supervised_scores)
            
            if rule_scores is not None:
                all_scores.append(rule_scores)
            
            if not all_scores:
                return np.array([])
            
            # Take maximum score for each transaction
            combined_scores = np.max(all_scores, axis=0)
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error in maximum combination: {str(e)}")
            raise
    
    def _custom_combination(self, unsupervised_scores, supervised_scores, rule_scores):
        """
        Combine scores using custom method
        
        Args:
            unsupervised_scores (array): Unsupervised model scores
            supervised_scores (array): Supervised model scores
            rule_scores (array): Rule-based scores
            
        Returns:
            array: Combined scores
        """
        try:
            # Get custom weights
            unsupervised_weight = self.custom_weights.get('unsupervised', 0.4)
            supervised_weight = self.custom_weights.get('supervised', 0.4)
            rule_weight = self.custom_weights.get('rule', 0.2)
            
            # Apply non-linear transformation to supervised scores
            if supervised_scores is not None:
                # Emphasize high scores
                supervised_transformed = np.power(supervised_scores, 1.5)
            else:
                supervised_transformed = None
            
            # Apply non-linear transformation to rule scores
            if rule_scores is not None:
                # Emphasize high scores even more
                rule_transformed = np.power(rule_scores, 2.0)
            else:
                rule_transformed = None
            
            # Combine using weighted average with transformed scores
            return self._weighted_average_combination(
                unsupervised_scores, supervised_transformed, rule_transformed
            )
            
        except Exception as e:
            logger.error(f"Error in custom combination: {str(e)}")
            raise
    
    def update_weights(self, new_weights):
        """
        Update the weights for combining scores
        
        Args:
            new_weights (dict): New weights for models
        """
        self.custom_weights.update(new_weights)
        logger.info(f"Updated weights: {self.custom_weights}")
    
    def get_weights(self):
        """
        Get current weights
        
        Returns:
            dict: Current weights
        """
        return self.custom_weights.copy()
    
    def calibrate_scores(self, method='isotonic'):
        """
        Calibrate scores to better reflect true probabilities
        
        Args:
            method (str): Calibration method ('isotonic', 'sigmoid')
        """
        try:
            # Update calibrators
            for model_type in ['unsupervised', 'supervised', 'rule']:
                if model_type in self.calibrators:
                    self.calibrators[model_type] = CalibratedClassifierCV(
                        method=method, cv='prefit'
                    )
            
            logger.info(f"Updated calibrators with method: {method}")
            
        except Exception as e:
            logger.error(f"Error calibrating scores: {str(e)}")
            raise