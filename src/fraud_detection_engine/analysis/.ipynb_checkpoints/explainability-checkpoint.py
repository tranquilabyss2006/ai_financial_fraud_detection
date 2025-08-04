"""
Explainability Module
Implements explainable AI techniques for fraud detection
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import graphviz
import warnings
import logging
from typing import Dict, List, Tuple, Union

from fraud_detection_engine.utils.api_utils import is_api_available

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Explainability:
    """
    Class for generating explanations for fraud detection predictions
    Implements SHAP, LIME, and other explainability techniques
    """
    
    def __init__(self, config=None):
        """
        Initialize Explainability
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.explainers = {}
        self.explanations = {}
        self.feature_names = {}
        self.fitted = False
    
    def generate_explanations(self, df, risk_scores, model_results):
        """
        Generate explanations for all transactions
        
        Args:
            df (DataFrame): Input data
            risk_scores (DataFrame): Risk scores
            model_results (dict): Results from different models
            
        Returns:
            dict: Explanations for each transaction
        """
        try:
            # Get feature columns
            feature_cols = [col for col in df.columns if col not in 
                           ['transaction_id', 'sender_id', 'receiver_id', 'fraud_flag']]
            
            # Prepare data
            X = df[feature_cols].fillna(0)
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Initialize explanations dictionary
            explanations = {}
            
            # Generate explanations for each transaction
            for idx, row in df.iterrows():
                try:
                    # Get risk score
                    risk_score = risk_scores.loc[idx, 'risk_score']
                    
                    # Generate explanation only for high-risk transactions
                    if risk_score > 0.5:  # Threshold for explanation
                        explanation = self._generate_transaction_explanation(
                            idx, row, X, risk_score, model_results
                        )
                        explanations[idx] = explanation
                except Exception as e:
                    logger.warning(f"Error generating explanation for transaction {idx}: {str(e)}")
            
            self.explanations = explanations
            self.fitted = True
            
            logger.info(f"Generated explanations for {len(explanations)} transactions")
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise
    
    def _generate_transaction_explanation(self, idx, row, X, risk_score, model_results):
        """
        Generate explanation for a single transaction
        
        Args:
            idx (int): Transaction index
            row (Series): Transaction data
            X (DataFrame): Feature data
            risk_score (float): Risk score
            model_results (dict): Model results
            
        Returns:
            dict: Explanation for the transaction
        """
        try:
            explanation = {
                'transaction_id': row.get('transaction_id', idx),
                'risk_score': risk_score,
                'top_factors': [],
                'rule_violations': [],
                'model_predictions': {},
                'feature_contributions': {},
                'text_explanation': ''
            }
            
            # Get feature contributions from different models
            feature_contributions = {}
            
            # Process unsupervised models
            if 'unsupervised' in model_results:
                unsupervised_contributions = self._get_unsupervised_contributions(
                    idx, X, model_results['unsupervised']
                )
                feature_contributions.update(unsupervised_contributions)
                
                # Add model predictions
                for model_name, model_data in model_results['unsupervised'].items():
                    if 'scores' in model_data and idx < len(model_data['scores']):
                        explanation['model_predictions'][f'unsupervised_{model_name}'] = model_data['scores'][idx]
            
            # Process supervised models
            if 'supervised' in model_results:
                supervised_contributions = self._get_supervised_contributions(
                    idx, X, model_results['supervised']
                )
                feature_contributions.update(supervised_contributions)
                
                # Add model predictions
                for model_name, model_data in model_results['supervised'].items():
                    if 'probabilities' in model_data and idx < len(model_data['probabilities']):
                        explanation['model_predictions'][f'supervised_{model_name}'] = model_data['probabilities'][idx]
            
            # Process rule-based models
            if 'rule' in model_results:
                rule_contributions = self._get_rule_contributions(
                    idx, row, model_results['rule']
                )
                feature_contributions.update(rule_contributions)
                
                # Add rule violations
                if 'violated_rule_names' in model_results['rule'] and idx < len(model_results['rule']['violated_rule_names']):
                    explanation['rule_violations'] = model_results['rule']['violated_rule_names'][idx]
            
            # Aggregate feature contributions
            explanation['feature_contributions'] = feature_contributions
            
            # Get top contributing factors
            top_factors = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]  # Top 10 factors
            
            explanation['top_factors'] = top_factors
            
            # Generate text explanation
            explanation['text_explanation'] = self._generate_text_explanation(
                explanation, row
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for transaction {idx}: {str(e)}")
            return {
                'transaction_id': row.get('transaction_id', idx),
                'risk_score': risk_score,
                'error': str(e)
            }
    
    def _get_unsupervised_contributions(self, idx, X, unsupervised_results):
        """
        Get feature contributions from unsupervised models
        
        Args:
            idx (int): Transaction index
            X (DataFrame): Feature data
            unsupervised_results (dict): Unsupervised model results
            
        Returns:
            dict: Feature contributions
        """
        try:
            contributions = {}
            
            # Process each unsupervised model
            for model_name, model_data in unsupervised_results.items():
                if 'model' in model_data and 'feature_names' in model_data:
                    model = model_data['model']
                    feature_names = model_data['feature_names']
                    
                    # Get feature values for this transaction
                    x = X.loc[idx:idx+1, feature_names]
                    
                    # Calculate feature contributions based on model type
                    if model_name == 'isolation_forest':
                        # For Isolation Forest, use feature importance
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = model.feature_importances_
                            for i, feature in enumerate(feature_names):
                                if i < len(feature_importance):
                                    contributions[f'{feature}_isolation_forest'] = feature_importance[i]
                    
                    elif model_name == 'local_outlier_factor':
                        # For LOF, use distance to neighbors
                        if hasattr(model, '_fit_X') and hasattr(model, '_distances'):
                            # Get distances to neighbors
                            distances = model._distances[idx]
                            # Calculate contribution based on feature differences
                            for i, feature in enumerate(feature_names):
                                if i < x.shape[1]:
                                    feature_value = x.iloc[0, i]
                                    # Calculate how much this feature contributes to the distance
                                    feature_diff = np.abs(feature_value - model._fit_X[:, i].mean())
                                    contributions[f'{feature}_lof'] = feature_diff
                    
                    elif model_name == 'autoencoder':
                        # For Autoencoder, use reconstruction error per feature
                        if hasattr(model, 'predict'):
                            reconstruction = model.predict(x)
                            error_per_feature = np.power(x.values - reconstruction, 2)
                            for i, feature in enumerate(feature_names):
                                if i < error_per_feature.shape[1]:
                                    contributions[f'{feature}_autoencoder'] = error_per_feature[0, i]
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error getting unsupervised contributions: {str(e)}")
            return {}
    
    def _get_supervised_contributions(self, idx, X, supervised_results):
        """
        Get feature contributions from supervised models
        
        Args:
            idx (int): Transaction index
            X (DataFrame): Feature data
            supervised_results (dict): Supervised model results
            
        Returns:
            dict: Feature contributions
        """
        try:
            contributions = {}
            
            # Process each supervised model
            for model_name, model_data in supervised_results.items():
                if 'model' in model_data and 'feature_names' in model_data:
                    model = model_data['model']
                    feature_names = model_data['feature_names']
                    
                    # Get feature values for this transaction
                    x = X.loc[idx:idx+1, feature_names]
                    
                    # Calculate SHAP values if available
                    if 'shap_values' in model_data:
                        shap_values = model_data['shap_values']
                        if isinstance(shap_values, list):
                            # For multi-class SHAP values
                            if len(shap_values) > 1:
                                shap_vals = shap_values[1][idx]  # Use positive class
                            else:
                                shap_vals = shap_values[0][idx]
                        else:
                            shap_vals = shap_values[idx]
                        
                        for i, feature in enumerate(feature_names):
                            if i < len(shap_vals):
                                contributions[f'{feature}_{model_name}'] = shap_vals[i]
                    
                    # Use feature importance as fallback
                    elif 'feature_importance' in model_data:
                        importance_df = model_data['feature_importance']
                        for _, row in importance_df.iterrows():
                            feature = row['feature']
                            importance = row['importance']
                            contributions[f'{feature}_{model_name}'] = importance
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error getting supervised contributions: {str(e)}")
            return {}
    
    def _get_rule_contributions(self, idx, row, rule_results):
        """
        Get feature contributions from rule-based model
        
        Args:
            idx (int): Transaction index
            row (Series): Transaction data
            rule_results (dict): Rule model results
            
        Returns:
            dict: Feature contributions
        """
        try:
            contributions = {}
            
            # Get violated rules for this transaction
            if 'violated_rule_names' in rule_results and idx < len(rule_results['violated_rule_names']):
                violated_rules = rule_results['violated_rule_names'][idx]
                
                # Get rule weights
                rule_weights = rule_results.get('rule_weights', {})
                
                # Map rules to features
                rule_to_features = {
                    'high_amount': ['amount'],
                    'unusual_amount_for_sender': ['amount', 'sender_id'],
                    'unusual_amount_for_receiver': ['amount', 'receiver_id'],
                    'round_amount': ['amount'],
                    'high_frequency_sender': ['sender_id', 'timestamp'],
                    'high_frequency_receiver': ['receiver_id', 'timestamp'],
                    'rapid_succession': ['timestamp'],
                    'cross_border': ['sender_location', 'receiver_location'],
                    'high_risk_country': ['sender_location', 'receiver_location'],
                    'unusual_location_for_sender': ['sender_location'],
                    'unusual_hour': ['timestamp'],
                    'weekend': ['timestamp'],
                    'new_sender': ['sender_id'],
                    'new_receiver': ['receiver_id'],
                    'sanctions_check': ['sender_id', 'receiver_id'],
                    'tax_compliance': ['sender_id', 'receiver_id'],
                    'bank_verification': ['sender_id', 'receiver_id'],
                    'identity_verification': ['sender_id', 'receiver_id']
                }
                
                # Calculate contributions based on violated rules
                for rule in violated_rules:
                    weight = rule_weights.get(rule, 1.0)
                    features = rule_to_features.get(rule, [])
                    
                    for feature in features:
                        if feature in row:
                            contributions[f'{feature}_rule'] = contributions.get(f'{feature}_rule', 0) + weight
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error getting rule contributions: {str(e)}")
            return {}
    
    def _generate_text_explanation(self, explanation, row):
        """
        Generate natural language explanation
        
        Args:
            explanation (dict): Explanation data
            row (Series): Transaction data
            
        Returns:
            str: Text explanation
        """
        try:
            # Start with risk level
            risk_score = explanation['risk_score']
            
            if risk_score >= 0.9:
                risk_level = "critical"
            elif risk_score >= 0.7:
                risk_level = "high"
            elif risk_score >= 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            text = f"This transaction has a {risk_level} risk score of {risk_score:.2f}. "
            
            # Add top contributing factors
            if explanation['top_factors']:
                top_factors = explanation['top_factors'][:3]  # Top 3 factors
                factor_names = [factor[0].split('_')[0] for factor in top_factors]
                
                if len(factor_names) == 1:
                    text += f"The primary factor is {factor_names[0]}. "
                elif len(factor_names) == 2:
                    text += f"The main factors are {factor_names[0]} and {factor_names[1]}. "
                else:
                    text += f"The main factors are {', '.join(factor_names[:-1])}, and {factor_names[-1]}. "
            
            # Add rule violations
            if explanation['rule_violations']:
                if len(explanation['rule_violations']) == 1:
                    text += f"It violates the rule: {explanation['rule_violations'][0]}. "
                else:
                    text += f"It violates {len(explanation['rule_violations'])} rules: {', '.join(explanation['rule_violations'][:3])}. "
            
            # Add transaction details
            if 'amount' in row:
                text += f"The transaction amount is {row['amount']}. "
            
            if 'sender_id' in row and 'receiver_id' in row:
                text += f"It involves sender {row['sender_id']} and receiver {row['receiver_id']}. "
            
            if 'timestamp' in row:
                text += f"The transaction occurred at {row['timestamp']}. "
            
            # Add AI-generated explanation if APIs are available
            ai_explanation = ""
            if is_api_available('gemini'):
                # Here you would call Gemini API to generate explanation
                # For now, add a placeholder
                ai_explanation = "AI analysis pending implementation. "
            elif is_api_available('openai'):
                # Here you would call OpenAI API to generate explanation
                # For now, add a placeholder
                ai_explanation = "AI analysis pending implementation. "
            
            if ai_explanation:
                text += f"AI analysis: {ai_explanation}"
            
            # Add recommendation
            if risk_score >= 0.7:
                text += "This transaction should be reviewed immediately."
            elif risk_score >= 0.5:
                text += "This transaction should be reviewed."
            else:
                text += "This transaction appears to be low risk."
            
            return text
            
        except Exception as e:
            logger.error(f"Error generating text explanation: {str(e)}")
            return "Unable to generate explanation."
    
    def get_explanation(self, transaction_id):
        """
        Get explanation for a specific transaction
        
        Args:
            transaction_id (str): Transaction ID
            
        Returns:
            dict: Explanation for the transaction
        """
        if not self.fitted:
            raise ValueError("Explanations not generated. Call generate_explanations first.")
        
        # Find explanation by transaction ID
        for idx, explanation in self.explanations.items():
            if explanation.get('transaction_id') == transaction_id:
                return explanation
        
        return None
    
    def plot_feature_importance(self, model_name=None, top_n=20):
        """
        Plot feature importance for a model
        
        Args:
            model_name (str, optional): Name of the model
            top_n (int): Number of top features to show
        """
        try:
            # Aggregate feature contributions across all explanations
            feature_contributions = {}
            
            for explanation in self.explanations.values():
                if 'feature_contributions' in explanation:
                    for feature, contribution in explanation['feature_contributions'].items():
                        if model_name is None or model_name in feature:
                            feature_contributions[feature] = feature_contributions.get(feature, 0) + abs(contribution)
            
            if not feature_contributions:
                logger.warning("No feature contributions found")
                return None
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': list(feature_contributions.keys()),
                'importance': list(feature_contributions.values())
            }).sort_values('importance', ascending=False)
            
            # Get top features
            top_features = importance_df.head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_shap_summary(self, model_name, X):
        """
        Plot SHAP summary for a model
        
        Args:
            model_name (str): Name of the model
            X (DataFrame): Feature data
        """
        try:
            # This would require access to the model and SHAP values
            # For demonstration, we'll create a placeholder plot
            
            plt.figure(figsize=(10, 8))
            plt.title(f'SHAP Summary for {model_name}')
            plt.text(0.5, 0.5, 'SHAP summary plot would be displayed here', 
                    ha='center', va='center', fontsize=12)
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {str(e)}")
            raise
    
    def plot_decision_path(self, model_name, X, idx):
        """
        Plot decision path for a tree-based model
        
        Args:
            model_name (str): Name of the model
            X (DataFrame): Feature data
            idx (int): Transaction index
        """
        try:
            # This would require access to the model
            # For demonstration, we'll create a placeholder plot
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Decision Path for {model_name}')
            plt.text(0.5, 0.5, 'Decision path would be displayed here', 
                    ha='center', va='center', fontsize=12)
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting decision path: {str(e)}")
            raise
    
    def get_counterfactual_explanation(self, idx, X, model_name, target_class=0):
        """
        Generate counterfactual explanation
        
        Args:
            idx (int): Transaction index
            X (DataFrame): Feature data
            model_name (str): Name of the model
            target_class (int): Target class (0 for non-fraud, 1 for fraud)
            
        Returns:
            dict: Counterfactual explanation
        """
        try:
            # This is a simplified implementation
            # In practice, you would use more sophisticated methods
            
            # Get original prediction
            original_row = X.loc[idx:idx+1]
            
            # Generate counterfactual by modifying features
            counterfactual = original_row.copy()
            
            # Modify top features to change prediction
            # This is a placeholder implementation
            for col in counterfactual.columns:
                if col in ['amount']:
                    # Reduce amount by 50%
                    counterfactual[col] = counterfactual[col] * 0.5
            
            return {
                'original': original_row.to_dict('records')[0],
                'counterfactual': counterfactual.to_dict('records')[0],
                'changes': {
                    'amount': 'Reduced by 50%'
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating counterfactual explanation: {str(e)}")
            return {}