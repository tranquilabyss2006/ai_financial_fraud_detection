"""
Test cases for data ingestion module
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection_engine.ingestion.data_loader import DataLoader
from fraud_detection_engine.ingestion.column_mapper import ColumnMapper

class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def setup_method(self):
        """Setup test data"""
        self.loader = DataLoader()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'transaction_id': range(1, 101),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'amount': np.random.uniform(10, 1000, 100),
            'sender_id': [f'sender_{i}' for i in range(1, 101)],
            'receiver_id': [f'receiver_{i}' for i in range(1, 101)],
            'currency': ['USD'] * 100,
            'description': [f'Transaction {i}' for i in range(1, 101)]
        })
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
    
    def teardown_method(self):
        """Clean up test data"""
        os.unlink(self.temp_csv.name)
    
    def test_load_csv(self):
        """Test loading CSV file"""
        df = self.loader.load_data(self.temp_csv.name)
        
        assert len(df) == 100
        assert 'transaction_id' in df.columns
        assert 'timestamp' in df.columns
        assert 'amount' in df.columns
        assert 'sender_id' in df.columns
        assert 'receiver_id' in df.columns
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        # Create data with missing values
        test_data_with_missing = self.test_data.copy()
        test_data_with_missing.loc[0, 'amount'] = np.nan
        test_data_with_missing.loc[1, 'sender_id'] = np.nan
        
        # Preprocess data
        processed_data = self.loader._preprocess_data(test_data_with_missing)
        
        # Check that missing values are handled
        assert not processed_data['amount'].isna().any()
        assert not processed_data['sender_id'].isna().any()
        
        # Check that timestamp is converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(processed_data['timestamp'])
    
    def test_validate_data(self):
        """Test data validation"""
        validation_result = self.loader.validate_data()
        
        assert validation_result['valid'] == True
        assert len(validation_result['errors']) == 0
    
    def test_get_data_info(self):
        """Test getting data information"""
        self.loader.load_data(self.temp_csv.name)
        info = self.loader.get_data_info()
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'missing_values' in info
        assert 'memory_usage' in info
        assert 'numeric_stats' in info
        assert 'categorical_stats' in info
    
    def test_get_sample(self):
        """Test getting data sample"""
        self.loader.load_data(self.temp_csv.name)
        sample = self.loader.get_sample(n=5)
        
        assert len(sample) == 5
    
    def test_save_data(self):
        """Test saving data"""
        self.loader.load_data(self.temp_csv.name)
        
        # Create temporary file for saving
        temp_save = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        temp_save.close()
        
        try:
            self.loader.save_data(temp_save.name)
            
            # Load saved data and verify
            loaded_data = pd.read_csv(temp_save.name)
            assert len(loaded_data) == 100
        finally:
            os.unlink(temp_save.name)

class TestColumnMapper:
    """Test cases for ColumnMapper class"""
    
    def setup_method(self):
        """Setup test data"""
        self.mapper = ColumnMapper()
        
        # Test user columns
        self.user_columns = [
            'tx_id', 'trans_date', 'amt', 'from_id', 'to_id',
            'sender_location', 'receiver_location', 'tx_type'
        ]
    
    def test_get_expected_columns(self):
        """Test getting expected columns"""
        expected_columns = self.mapper.get_expected_columns()
        
        assert 'transaction_id' in expected_columns
        assert 'timestamp' in expected_columns
        assert 'amount' in expected_columns
        assert 'sender_id' in expected_columns
        assert 'receiver_id' in expected_columns
    
    def test_auto_map_columns(self):
        """Test automatic column mapping"""
        mapping = self.mapper.auto_map_columns(self.user_columns)
        
        assert 'tx_id' in mapping
        assert mapping['tx_id'] == 'transaction_id'
        
        assert 'trans_date' in mapping
        assert mapping['trans_date'] == 'timestamp'
        
        assert 'amt' in mapping
        assert mapping['amt'] == 'amount'
        
        assert 'from_id' in mapping
        assert mapping['from_id'] == 'sender_id'
        
        assert 'to_id' in mapping
        assert mapping['to_id'] == 'receiver_id'
    
    def test_apply_mapping(self):
        """Test applying column mapping"""
        # Create test dataframe
        test_df = pd.DataFrame({
            'tx_id': range(1, 6),
            'trans_date': pd.date_range('2023-01-01', periods=5),
            'amt': [100, 200, 300, 400, 500],
            'from_id': ['s1', 's2', 's3', 's4', 's5'],
            'to_id': ['r1', 'r2', 'r3', 'r4', 'r5']
        })
        
        # Get mapping
        mapping = self.mapper.auto_map_columns(test_df.columns.tolist())
        
        # Apply mapping
        mapped_df = self.mapper.apply_mapping(test_df, mapping)
        
        # Check that mapped columns exist
        assert 'transaction_id' in mapped_df.columns
        assert 'timestamp' in mapped_df.columns
        assert 'amount' in mapped_df.columns
        assert 'sender_id' in mapped_df.columns
        assert 'receiver_id' in mapped_df.columns
    
    def test_validate_mapping(self):
        """Test validating mapping"""
        mapping = self.mapper.auto_map_columns(self.user_columns)
        validation = self.mapper.validate_mapping(mapping)
        
        assert validation['valid'] == True
        assert len(validation['errors']) == 0
    
    def test_get_mapping_suggestions(self):
        """Test getting mapping suggestions"""
        suggestions = self.mapper.get_mapping_suggestions(self.user_columns)
        
        assert 'tx_id' in suggestions
        assert len(suggestions['tx_id']) > 0
        
        # Check that top suggestion is correct
        top_suggestion = suggestions['tx_id'][0]
        assert top_suggestion['column'] == 'transaction_id'
        assert top_suggestion['confidence'] > 0.8