"""
Test cases for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection_engine.features.statistical_features import StatisticalFeatures
from fraud_detection_engine.features.graph_features import GraphFeatures
from fraud_detection_engine.features.nlp_features import NLPFeatures
from fraud_detection_engine.features.timeseries_features import TimeSeriesFeatures

class TestStatisticalFeatures:
    """Test cases for StatisticalFeatures class"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_extractor = StatisticalFeatures()
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'transaction_id': range(1, 101),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'amount': np.random.lognormal(5, 1, 100),
            'sender_id': [f'sender_{i % 10}' for i in range(100)],
            'receiver_id': [f'receiver_{i % 10}' for i in range(100)]
        })
    
    def test_extract_features(self):
        """Test feature extraction"""
        result_df = self.feature_extractor.extract_features(self.test_data)
        
        # Check that features are added
        original_cols = set(self.test_data.columns)
        result_cols = set(result_df.columns)
        new_cols = result_cols - original_cols
        
        assert len(new_cols) > 0
        
        # Check specific features
        assert 'amount_zscore' in result_df.columns
        assert 'amount_mad_zscore' in result_df.columns
        assert 'amount_percentile_rank' in result_df.columns
        assert 'benford_chi_square' in result_df.columns
    
    def test_benford_features(self):
        """Test Benford's Law features"""
        result_df = self.feature_extractor._extract_benford_features(self.test_data)
        
        assert 'benford_chi_square' in result_df.columns
        assert 'benford_p_value' in result_df.columns
        
        # Check that deviation features are added
        for i in range(1, 6):
            assert f'benford_deviation_{i}' in result_df.columns
    
    def test_zscore_features(self):
        """Test Z-score features"""
        result_df = self.feature_extractor._extract_zscore_features(self.test_data)
        
        assert 'amount_zscore' in result_df.columns
        assert 'amount_zscore_outlier' in result_df.columns
        assert 'sender_amount_zscore' in result_df.columns
        assert 'receiver_amount_zscore' in result_df.columns
    
    def test_mad_features(self):
        """Test MAD features"""
        result_df = self.feature_extractor._extract_mad_features(self.test_data)
        
        assert 'amount_mad_zscore' in result_df.columns
        assert 'amount_mad_outlier' in result_df.columns
        assert 'sender_amount_mad_zscore' in result_df.columns
        assert 'receiver_amount_mad_zscore' in result_df.columns
    
    def test_fit_transform(self):
        """Test fit_transform method"""
        result_df = self.feature_extractor.fit_transform(self.test_data)
        
        # Check that features are added
        original_cols = set(self.test_data.columns)
        result_cols = set(result_df.columns)
        new_cols = result_cols - original_cols
        
        assert len(new_cols) > 0
        assert self.feature_extractor.fitted == True
    
    def test_transform(self):
        """Test transform method"""
        # First fit the extractor
        self.feature_extractor.fit_transform(self.test_data)
        
        # Create new data
        new_data = pd.DataFrame({
            'transaction_id': range(101, 111),
            'timestamp': pd.date_range('2023-04-11', periods=10, freq='D'),
            'amount': np.random.lognormal(5, 1, 10),
            'sender_id': [f'sender_{i}' for i in range(10)],
            'receiver_id': [f'receiver_{i}' for i in range(10)]
        })
        
        # Transform new data
        result_df = self.feature_extractor.transform(new_data)
        
        # Check that features are added
        original_cols = set(new_data.columns)
        result_cols = set(result_df.columns)
        new_cols = result_cols - original_cols
        
        assert len(new_cols) > 0

class TestGraphFeatures:
    """Test cases for GraphFeatures class"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_extractor = GraphFeatures()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'transaction_id': range(1, 21),
            'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
            'amount': np.random.uniform(100, 1000, 20),
            'sender_id': [f'sender_{i % 5}' for i in range(20)],
            'receiver_id': [f'receiver_{i % 5}' for i in range(20)]
        })
    
    def test_extract_features(self):
        """Test feature extraction"""
        result_df = self.feature_extractor.extract_features(self.test_data)
        
        # Check that features are added
        original_cols = set(self.test_data.columns)
        result_cols = set(result_df.columns)
        new_cols = result_cols - original_cols
        
        assert len(new_cols) > 0
        
        # Check specific features
        assert 'sender_degree_centrality' in result_df.columns
        assert 'receiver_degree_centrality' in result_df.columns
        assert 'sender_clustering_coefficient' in result_df.columns
        assert 'receiver_clustering_coefficient' in result_df.columns
    
    def test_build_graphs(self):
        """Test graph building"""
        self.feature_extractor._build_graphs(self.test_data)
        
        # Check that graphs are built
        assert self.feature_extractor.graph is not None
        assert self.feature_extractor.sender_graph is not None
        assert self.feature_extractor.receiver_graph is not None
        assert self.feature_extractor.bipartite_graph is not None
    
    def test_centrality_features(self):
        """Test centrality features"""
        result_df = self.feature_extractor._extract_centrality_features(self.test_data)
        
        assert 'sender_degree_centrality' in result_df.columns
        assert 'receiver_degree_centrality' in result_df.columns
        assert 'sender_betweenness_centrality' in result_df.columns
        assert 'receiver_betweenness_centrality' in result_df.columns
        assert 'sender_pagerank' in result_df.columns
        assert 'receiver_pagerank' in result_df.columns
    
    def test_clustering_features(self):
        """Test clustering features"""
        result_df = self.feature_extractor._extract_clustering_features(self.test_data)
        
        assert 'sender_clustering_coefficient' in result_df.columns
        assert 'receiver_clustering_coefficient' in result_df.columns
        assert 'sender_avg_neighbor_degree' in result_df.columns
        assert 'receiver_avg_neighbor_degree' in result_df.columns

class TestNLPFeatures:
    """Test cases for NLPFeatures class"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_extractor = NLPFeatures()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'transaction_id': range(1, 11),
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'amount': np.random.uniform(100, 1000, 10),
            'sender_id': [f'sender_{i}' for i in range(10)],
            'receiver_id': [f'receiver_{i}' for i in range(10)],
            'description': [
                'Payment for services',
                'URGENT: Send money immediately',
                'Transfer funds',
                'Suspicious transaction',
                'Regular payment',
                'HURRY: Quick transfer needed',
                'Business expense',
                'CONFIDENTIAL: Private transfer',
                'Normal transaction',
                'Send $1000 ASAP'
            ],
            'notes': [
                'This is a normal transaction',
                'Please send money urgently',
                'Business related transfer',
                'This looks suspicious',
                'Regular monthly payment',
                'Need this done quickly',
                'Office supplies',
                'Keep this confidential',
                'Standard transaction',
                'Emergency funds needed'
            ]
        })
    
    def test_extract_features(self):
        """Test feature extraction"""
        result_df = self.feature_extractor.extract_features(self.test_data)
        
        # Check that features are added
        original_cols = set(self.test_data.columns)
        result_cols = set(result_df.columns)
        new_cols = result_cols - original_cols
        
        assert len(new_cols) > 0
        
        # Check specific features
        assert 'description_char_count' in result_df.columns
        assert 'description_sentiment_compound' in result_df.columns
        assert 'description_fraud_keyword_count' in result_df.columns
        assert 'description_money_pattern_count' in result_df.columns
    
    def test_basic_text_features(self):
        """Test basic text features"""
        result_df = self.feature_extractor._extract_basic_text_features(self.test_data)
        
        assert 'description_char_count' in result_df.columns
        assert 'description_word_count' in result_df.columns
        assert 'description_sentence_count' in result_df.columns
        assert 'description_avg_word_length' in result_df.columns
        assert 'description_punctuation_count' in result_df.columns
    
    def test_sentiment_features(self):
        """Test sentiment features"""
        result_df = self.feature_extractor._extract_sentiment_features(self.test_data)
        
        assert 'description_sentiment_neg' in result_df.columns
        assert 'description_sentiment_neu' in result_df.columns
        assert 'description_sentiment_pos' in result_df.columns
        assert 'description_sentiment_compound' in result_df.columns
        assert 'description_textblob_polarity' in result_df.columns
        assert 'description_textblob_subjectivity' in result_df.columns
    
    def test_keyword_features(self):
        """Test keyword features"""
        result_df = self.feature_extractor._extract_keyword_features(self.test_data)
        
        assert 'description_fraud_keyword_count' in result_df.columns
        assert 'description_has_fraud_keywords' in result_df.columns
        assert 'description_urgency_keyword_count' in result_df.columns
        assert 'description_secrecy_keyword_count' in result_df.columns
        assert 'description_money_keyword_count' in result_df.columns

class TestTimeSeriesFeatures:
    """Test cases for TimeSeriesFeatures class"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_extractor = TimeSeriesFeatures()
        
        # Create test data with timestamps
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=100, freq='H')
        
        self.test_data = pd.DataFrame({
            'transaction_id': range(1, 101),
            'timestamp': timestamps,
            'amount': np.random.lognormal(5, 1, 100),
            'sender_id': [f'sender_{i % 10}' for i in range(100)],
            'receiver_id': [f'receiver_{i % 10}' for i in range(100)]
        })
    
    def test_extract_features(self):
        """Test feature extraction"""
        result_df = self.feature_extractor.extract_features(self.test_data)
        
        # Check that features are added
        original_cols = set(self.test_data.columns)
        result_cols = set(result_df.columns)
        new_cols = result_cols - original_cols
        
        assert len(new_cols) > 0
        
        # Check specific features
        assert 'hour' in result_df.columns
        assert 'dayofweek' in result_df.columns
        assert 'is_weekend' in result_df.columns
        assert 'is_business_hours' in result_df.columns
        assert 'transaction_frequency_1H' in result_df.columns
    
    def test_temporal_features(self):
        """Test temporal features"""
        result_df = self.feature_extractor._extract_temporal_features(self.test_data)
        
        assert 'hour' in result_df.columns
        assert 'day' in result_df.columns
        assert 'month' in result_df.columns
        assert 'year' in result_df.columns
        assert 'dayofweek' in result_df.columns
        assert 'is_weekend' in result_df.columns
        assert 'is_business_hours' in result_df.columns
    
    def test_frequency_features(self):
        """Test frequency features"""
        result_df = self.feature_extractor._extract_frequency_features(self.test_data)
        
        assert 'transaction_frequency_1H' in result_df.columns
        assert 'transaction_frequency_6H' in result_df.columns
        assert 'transaction_frequency_24H' in result_df.columns
        assert 'sender_frequency_1H' in result_df.columns
        assert 'receiver_frequency_1H' in result_df.columns
    
    def test_burstiness_features(self):
        """Test burstiness features"""
        result_df = self.feature_extractor._extract_burstiness_features(self.test_data)
        
        assert 'burstiness_coefficient' in result_df.columns
        assert 'local_burstiness_10' in result_df.columns
        assert 'is_in_burst' in result_df.columns
        assert 'burst_duration' in result_df.columns