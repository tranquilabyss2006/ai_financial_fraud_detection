"""
Test cases for models module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection_engine.models.unsupervised import UnsupervisedModels
from fraud_detection_engine.models.supervised import SupervisedModels
from fraud_detection_engine.models.rule_based import RuleEngine

class TestUnsupervisedModels:
    """Test cases for UnsupervisedModels class"""
    
    def setup_method(self):
        """Setup test data"""
        self.models = UnsupervisedModels(contamination=0.1)
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100),
            'feature5': np.random.normal(0, 1, 100)
        })
        
        # Add some outliers
        self.test_data.loc[0:4, 'feature1'] = 10  # Outliers
        self.test_data.loc[5:9, 'feature2'] = -10  # Outliers
    
    def test_run_models(self):
        """Test running models"""
        results = self.models.run_models(self.test_data)
        
        # Check that results are returned
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that each model has required keys
        for model_name, model_results in results.items():
            assert 'predictions' in model_results
            assert 'scores' in model_results
            assert 'model' in model_results
            assert 'feature_names' in model_results
    
    def test_isolation_forest(self):
        """Test Isolation Forest model"""
        results = self.models.run_models(self.test_data)
        
        if 'isolation_forest' in results:
            model_results = results['isolation_forest']
            
            # Check predictions
            predictions = model_results['predictions']
            assert len(predictions) == len(self.test_data)
            assert set(predictions) == {-1, 1}  # -1 for outliers, 1 for inliers
            
            # Check scores
            scores = model_results['scores']
            assert len(scores) == len(self.test_data)
            assert all(0 <= score <= 1 for score in scores)  # Normalized scores
    
    def test_local_outlier_factor(self):
        """Test Local Outlier Factor model"""
        results = self.models.run_models(self.test_data)
        
        if 'local_outlier_factor' in results:
            model_results = results['local_outlier_factor']
            
            # Check predictions
            predictions = model_results['predictions']
            assert len(predictions) == len(self.test_data)
            assert set(predictions) == {-1, 1}  # -1 for outliers, 1 for inliers
            
            # Check scores
            scores = model_results['scores']
            assert len(scores) == len(self.test_data)
            assert all(0 <= score <= 1 for score in scores)  # Normalized scores
    
    def test_autoencoder(self):
        """Test Autoencoder model"""
        results = self.models.run_models(self.test_data)
        
        if 'autoencoder' in results:
            model_results = results['autoencoder']
            
            # Check predictions
            predictions = model_results['predictions']
            assert len(predictions) == len(self.test_data)
            assert set(predictions) == {-1, 1}  # -1 for outliers, 1 for inliers
            
            # Check scores
            scores = model_results['scores']
            assert len(scores) == len(self.test_data)
            assert all(0 <= score <= 1 for score in scores)  # Normalized scores
    
    def test_predict(self):
        """Test making predictions"""
        # First fit models
        self.models.run_models(self.test_data)
        
        # Create new data
        new_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10),
            'feature2': np.random.normal(0, 1, 10),
            'feature3': np.random.normal(0, 1, 10),
            'feature4': np.random.normal(0, 1, 10),
            'feature5': np.random.normal(0, 1, 10)
        })
        
        # Test prediction with Isolation Forest
        if 'isolation_forest' in self.models.models:
            result = self.models.predict(new_data, 'isolation_forest')
            
            assert 'predictions' in result
            assert 'scores' in result
            assert len(result['predictions']) == len(new_data)
            assert len(result['scores']) == len(new_data)
    
    def test_get_feature_importance(self):
        """Test getting feature importance"""
        # First fit models
        self.models.run_models(self.test_data)
        
        # Test with Isolation Forest
        if 'isolation_forest' in self.models.models:
            importance = self.models.get_feature_importance('isolation_forest')
            
            assert isinstance(importance, pd.DataFrame)
            assert 'feature' in importance.columns
            assert 'importance' in importance.columns
            assert len(importance) == len(self.test_data.columns)

class TestSupervisedModels:
    """Test cases for SupervisedModels class"""
    
    def setup_method(self):
        """Setup test data"""
        self.models = SupervisedModels(test_size=0.3, random_state=42)
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100),
            'feature5': np.random.normal(0, 1, 100),
            'fraud_flag': np.random.choice([0, 1], 100, p=[0.9, 0.1])  # Imbalanced classes
        })
    
    def test_run_models(self):
        """Test running models"""
        results = self.models.run_models(self.test_data)
        
        # Check that results are returned
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that each model has required keys
        for model_name, model_results in results.items():
            assert 'model' in model_results
            assert 'performance' in model_results
            assert 'feature_importance' in model_results
            assert 'feature_names' in model_results
    
    def test_random_forest(self):
        """Test Random Forest model"""
        results = self.models.run_models(self.test_data)
        
        if 'random_forest' in results:
            model_results = results['random_forest']
            
            # Check model
            model = model_results['model']
            assert model is not None
            
            # Check performance
            performance = model_results['performance']
            assert 'accuracy' in performance
            assert 'precision' in performance
            assert 'recall' in performance
            assert 'f1' in performance
            assert 'roc_auc' in performance
            
            # Check feature importance
            importance = model_results['feature_importance']
            assert isinstance(importance, pd.DataFrame)
            assert 'feature' in importance.columns
            assert 'importance' in importance.columns
    
    def test_xgboost(self):
        """Test XGBoost model"""
        results = self.models.run_models(self.test_data)
        
        if 'xgboost' in results:
            model_results = results['xgboost']
            
            # Check model
            model = model_results['model']
            assert model is not None
            
            # Check performance
            performance = model_results['performance']
            assert 'accuracy' in performance
            assert 'precision' in performance
            assert 'recall' in performance
            assert 'f1' in performance
            assert 'roc_auc' in performance
            
            # Check feature importance
            importance = model_results['feature_importance']
            assert isinstance(importance, pd.DataFrame)
            assert 'feature' in importance.columns
            assert 'importance' in importance.columns
    
    def test_predict(self):
        """Test making predictions"""
        # First fit models
        self.models.run_models(self.test_data)
        
        # Create new data
        new_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10),
            'feature2': np.random.normal(0, 1, 10),
            'feature3': np.random.normal(0, 1, 10),
            'feature4': np.random.normal(0, 1, 10),
            'feature5': np.random.normal(0, 1, 10)
        })
        
        # Test prediction with Random Forest
        if 'random_forest' in self.models.models:
            result = self.models.predict(new_data, 'random_forest')
            
            assert 'predictions' in result
            assert 'probabilities' in result
            assert len(result['predictions']) == len(new_data)
            assert len(result['probabilities']) == len(new_data)
    
    def test_get_feature_importance(self):
        """Test getting feature importance"""
        # First fit models
        self.models.run_models(self.test_data)
        
        # Test with Random Forest
        if 'random_forest' in self.models.models:
            importance = self.models.get_feature_importance('random_forest')
            
            assert isinstance(importance, pd.DataFrame)
            assert 'feature' in importance.columns
            assert 'importance' in importance.columns
    
    def test_get_performance(self):
        """Test getting performance metrics"""
        # First fit models
        self.models.run_models(self.test_data)
        
        # Test with Random Forest
        if 'random_forest' in self.models.models:
            performance = self.models.get_performance('random_forest')
            
            assert isinstance(performance, dict)
            assert 'accuracy' in performance
            assert 'precision' in performance
            assert 'recall' in performance
            assert 'f1' in performance
            assert 'roc_auc' in performance

class TestRuleEngine:
    """Test cases for RuleEngine class"""
    
    def setup_method(self):
        """Setup test data"""
        self.rule_engine = RuleEngine(threshold=0.5)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'transaction_id': range(1, 11),
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'amount': [100, 200, 10000, 5000, 15000, 300, 400, 12000, 8000, 20000],
            'sender_id': [f'sender_{i}' for i in range(10)],
            'receiver_id': [f'receiver_{i}' for i in range(10)],
            'sender_location': ['USA'] * 5 + ['Canada'] * 5,
            'receiver_location': ['USA'] * 8 + ['North Korea'] * 2,
            'description': [
                'Normal payment',
                'Regular transfer',
                'Large amount',
                'Big transaction',
                'Huge payment',
                'Small amount',
                'Normal transfer',
                'Large transaction',
                'Big payment',
                'Massive amount'
            ]
        })
    
    def test_apply_rules(self):
        """Test applying rules"""
        results = self.rule_engine.apply_rules(self.test_data)
        
        # Check that results are returned
        assert isinstance(results, dict)
        
        # Check required keys
        assert 'rule_results' in results
        assert 'rule_scores' in results
        assert 'total_scores' in results
        assert 'normalized_scores' in results
        assert 'rule_violations' in results
        assert 'violated_rule_names' in results
        assert 'rules' in results
        assert 'rule_weights' in results
        assert 'rule_descriptions' in results
    
    def test_high_amount_rule(self):
        """Test high amount rule"""
        # Apply rules
        results = self.rule_engine.apply_rules(self.test_data)
        
        # Check rule results
        if 'high_amount' in results['rule_results']:
            rule_results = results['rule_results']['high_amount']
            
            # Should flag transactions with amount > 10000
            expected_flags = [amount > 10000 for amount in self.test_data['amount']]
            assert rule_results == expected_flags
    
    def test_cross_border_rule(self):
        """Test cross border rule"""
        # Apply rules
        results = self.rule_engine.apply_rules(self.test_data)
        
        # Check rule results
        if 'cross_border' in results['rule_results']:
            rule_results = results['rule_results']['cross_border']
            
            # Should flag transactions with different sender and receiver countries
            expected_flags = []
            for i in range(len(self.test_data)):
                sender_country = self.test_data.loc[i, 'sender_location'].split(',')[0]
                receiver_country = self.test_data.loc[i, 'receiver_location'].split(',')[0]
                expected_flags.append(sender_country != receiver_country)
            
            assert rule_results == expected_flags
    
    def test_high_risk_country_rule(self):
        """Test high risk country rule"""
        # Apply rules
        results = self.rule_engine.apply_rules(self.test_data)
        
        # Check rule results
        if 'high_risk_country' in results['rule_results']:
            rule_results = results['rule_results']['high_risk_country']
            
            # Should flag transactions involving North Korea
            expected_flags = ['North Korea' in location for location in self.test_data['receiver_location']]
            assert rule_results == expected_flags
    
    def test_rule_violations(self):
        """Test rule violations"""
        # Apply rules
        results = self.rule_engine.apply_rules(self.test_data)
        
        # Check that violations are detected
        violations = results['rule_violations']
        assert len(violations) == len(self.test_data)
        
        # Check that some transactions are flagged
        assert any(violations)  # At least one transaction should be flagged
    
    def test_add_rule(self):
        """Test adding custom rule"""
        # Add custom rule
        def custom_rule(row):
            return row.get('amount', 0) > 5000
        
        self.rule_engine.add_rule('custom_rule', custom_rule, weight=0.5, description='Custom rule')
        
        # Check that rule is added
        assert 'custom_rule' in self.rule_engine.rules
        assert self.rule_engine.rule_weights['custom_rule'] == 0.5
        assert self.rule_engine.rule_descriptions['custom_rule'] == 'Custom rule'
    
    def test_remove_rule(self):
        """Test removing rule"""
        # Remove a rule
        if 'high_amount' in self.rule_engine.rules:
            self.rule_engine.remove_rule('high_amount')
            
            # Check that rule is removed
            assert 'high_amount' not in self.rule_engine.rules
            assert 'high_amount' not in self.rule_engine.rule_weights
            assert 'high_amount' not in self.rule_engine.rule_descriptions
    
    def test_update_rule_weight(self):
        """Test updating rule weight"""
        # Update rule weight
        self.rule_engine.update_rule_weight('high_amount', 0.5)
        
        # Check that weight is updated
        assert self.rule_engine.rule_weights['high_amount'] == 0.5
    
    def test_get_rules(self):
        """Test getting rules"""
        rules_info = self.rule_engine.get_rules()
        
        # Check structure
        assert 'rules' in rules_info
        assert 'weights' in rules_info
        assert 'descriptions' in rules_info
        
        # Check that rules are returned
        assert len(rules_info['rules']) > 0
        assert len(rules_info['weights']) > 0
        assert len(rules_info['descriptions']) > 0