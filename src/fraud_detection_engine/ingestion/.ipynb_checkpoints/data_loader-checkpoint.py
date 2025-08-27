"""
Data Loader Module
Handles loading and preprocessing of transaction data
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import logging
from datetime import datetime
import chardet
import warnings
import pickle
import zlib
import hashlib
from typing import Dict, List, Tuple, Union
warnings.filterwarnings('ignore')
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DataLoader:
    """
    Class for loading and preprocessing transaction data
    Supports CSV, Excel, and Parquet files
    Uses Dask for handling large files
    """
    
    def __init__(self, use_dask=True, chunk_size=100000):
        """
        Initialize DataLoader
        
        Args:
            use_dask (bool): Whether to use Dask for large files
            chunk_size (int): Chunk size for Dask processing
        """
        self.use_dask = use_dask
        self.chunk_size = chunk_size
        self.data = None
        self.original_data = None
        self.file_hash = None
        self.file_metadata = {}
        
    def load_data(self, file_path, file_type=None, **kwargs):
        """
        Load data from file
        
        Args:
            file_path (str): Path to the data file
            file_type (str, optional): Type of file (csv, excel, parquet)
            **kwargs: Additional parameters for pandas/dask read functions
            
        Returns:
            DataFrame: Loaded data
        """
        try:
            # Determine file type if not provided
            if file_type is None:
                file_type = os.path.splitext(file_path)[1][1:].lower()
            
            # Get file size to decide whether to use Dask
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            # Calculate file hash for caching
            with open(file_path, 'rb') as f:
                file_content = f.read()
                self.file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Store file metadata
            self.file_metadata = {
                'file_path': file_path,
                'file_type': file_type,
                'file_size_mb': file_size,
                'file_hash': self.file_hash,
                'loaded_at': datetime.now().isoformat()
            }
            
            # Use Dask for large files
            if self.use_dask and file_size > 100:  # Use Dask for files > 100MB
                logger.info(f"Loading large file ({file_size:.2f} MB) using Dask")
                return self._load_with_dask(file_path, file_type, **kwargs)
            else:
                logger.info(f"Loading file ({file_size:.2f} MB) using pandas")
                return self._load_with_pandas(file_path, file_type, **kwargs)
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_with_pandas(self, file_path, file_type, **kwargs):
        """
        Load data using pandas
        
        Args:
            file_path (str): Path to the data file
            file_type (str): Type of file
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            DataFrame: Loaded data
        """
        try:
            # Detect encoding for CSV files
            if file_type == 'csv':
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read())
                encoding = result['encoding']
                kwargs.setdefault('encoding', encoding)
            
            # Load data based on file type
            if file_type in ['csv', 'txt']:
                df = pd.read_csv(file_path, **kwargs)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Store original data
            self.original_data = df.copy()
            
            # Basic preprocessing
            df = self._preprocess_data(df)
            
            self.data = df
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data with pandas: {str(e)}")
            raise
    
    def _load_with_dask(self, file_path, file_type, **kwargs):
        """
        Load data using Dask
        
        Args:
            file_path (str): Path to the data file
            file_type (str): Type of file
            **kwargs: Additional parameters for dask read functions
            
        Returns:
            DataFrame: Loaded data
        """
        try:
            # Load data based on file type
            if file_type in ['csv', 'txt']:
                df = dd.read_csv(file_path, blocksize=self.chunk_size, **kwargs)
            elif file_type == 'parquet':
                df = dd.read_parquet(file_path, **kwargs)
            else:
                # For Excel files, we need to use pandas
                logger.warning("Dask does not support Excel files directly. Using pandas instead.")
                return self._load_with_pandas(file_path, file_type, **kwargs)
            
            # Store original data (compute a sample for inspection)
            self.original_data = df.head(1000)
            
            # Basic preprocessing
            df = df.map_partitions(self._preprocess_data)
            
            self.data = df
            logger.info(f"Data loaded successfully with Dask. Partitions: {df.npartitions}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data with Dask: {str(e)}")
            raise
    
    def _preprocess_data(self, df):
        """
        Basic preprocessing of the data
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Preprocessed data
        """
        try:
            # Log preprocessing start
            logger.info(f"Starting preprocessing for data with shape: {df.shape}")
            
            # Remove duplicate rows
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
            
            # Handle missing values
            for col in df.columns:
                # For numeric columns, fill with median
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        # If median is NaN, use mean
                        mean_val = df[col].mean()
                        if pd.isna(mean_val):
                            # If mean is also NaN, use 0
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna(mean_val)
                    else:
                        df[col] = df[col].fillna(median_val)
                # For categorical columns, fill with mode or 'Unknown'
                elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
            
            # Convert timestamp columns to datetime
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            for col in timestamp_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
            
            # Convert amount columns to numeric
            amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower() or 'sum' in col.lower()]
            for col in amount_cols:
                try:
                    # Remove currency symbols and commas
                    if pd.api.types.is_string_dtype(df[col]):
                        df[col] = df[col].str.replace('[\$,€,£,¥]', '', regex=True)
                        df[col] = df[col].str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {str(e)}")
            
            # Convert ID columns to string
            id_cols = [col for col in df.columns if 'id' in col.lower()]
            for col in id_cols:
                try:
                    df[col] = df[col].astype(str)
                    logger.info(f"Converted {col} to string")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to string: {str(e)}")
            
            # Convert boolean columns
            bool_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2 and all(
                        val.lower() in ['true', 'false', 'yes', 'no', 'y', 'n', '0', '1'] 
                        for val in unique_vals
                    ):
                        bool_cols.append(col)
            
            for col in bool_cols:
                try:
                    df[col] = df[col].replace({
                        'true': True, 'yes': True, 'y': True, '1': True,
                        'false': False, 'no': False, 'n': False, '0': False
                    })
                    logger.info(f"Converted {col} to boolean")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to boolean: {str(e)}")
            
            # Convert categorical columns with many unique values to 'category' dtype
            cat_cols = []
            for col in df.columns:
                if (pd.api.types.is_string_dtype(df[col]) and 
                    df[col].nunique() > 10 and 
                    df[col].nunique() < len(df) * 0.5):
                    cat_cols.append(col)
            
            for col in cat_cols:
                try:
                    df[col] = df[col].astype('category')
                    logger.info(f"Converted {col} to category")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to category: {str(e)}")
            
            logger.info("Data preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def get_data_info(self):
        """
        Get information about the loaded data
        
        Returns:
            dict: Data information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        try:
            info = {
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": dict(self.data.dtypes),
                "missing_values": dict(self.data.isnull().sum()),
                "memory_usage": dict(self.data.memory_usage(deep=True)),
                "file_hash": self.file_hash,
                "file_metadata": self.file_metadata
            }
            
            # Add statistics for numeric columns
            numeric_stats = {}
            for col in self.data.select_dtypes(include=[np.number]).columns:
                try:
                    numeric_stats[col] = {
                        "min": float(self.data[col].min()),
                        "max": float(self.data[col].max()),
                        "mean": float(self.data[col].mean()),
                        "median": float(self.data[col].median()),
                        "std": float(self.data[col].std())
                    }
                except Exception as e:
                    logger.warning(f"Error calculating stats for {col}: {str(e)}")
                    numeric_stats[col] = {"error": str(e)}
            
            info["numeric_stats"] = numeric_stats
            
            # Add unique counts for categorical columns
            categorical_stats = {}
            for col in self.data.select_dtypes(include=['object', 'category']).columns:
                try:
                    categorical_stats[col] = {
                        "unique_count": int(self.data[col].nunique()),
                        "top_values": self.data[col].value_counts().head(5).to_dict()
                    }
                except Exception as e:
                    logger.warning(f"Error calculating categorical stats for {col}: {str(e)}")
                    categorical_stats[col] = {"error": str(e)}
            
            info["categorical_stats"] = categorical_stats
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            return {"error": str(e)}
    
    def get_sample(self, n=5):
        """
        Get a sample of the data
        
        Args:
            n (int): Number of rows to return
            
        Returns:
            DataFrame: Sample data
        """
        if self.data is None:
            return None
        
        try:
            if isinstance(self.data, dd.DataFrame):
                return self.data.head(n)
            else:
                return self.data.sample(n) if len(self.data) > n else self.data
        except Exception as e:
            logger.error(f"Error getting sample: {str(e)}")
            return None
    
    def validate_data(self, required_columns=None):
        """
        Validate the data against required columns
        
        Args:
            required_columns (list, optional): List of required columns
            
        Returns:
            dict: Validation results
        """
        if self.data is None:
            return {"valid": False, "error": "No data loaded"}
        
        try:
            if required_columns is None:
                required_columns = ['transaction_id', 'timestamp', 'amount', 'sender_id', 'receiver_id']
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Check for duplicate transaction IDs
            if 'transaction_id' in self.data.columns:
                duplicate_ids = self.data['transaction_id'].duplicated().sum()
                if duplicate_ids > 0:
                    validation_result["warnings"].append(f"Found {duplicate_ids} duplicate transaction IDs")
            
            # Check for missing values in critical columns
            critical_columns = ['transaction_id', 'timestamp', 'amount']
            for col in critical_columns:
                if col in self.data.columns:
                    missing_count = self.data[col].isnull().sum()
                    if missing_count > 0:
                        validation_result["warnings"].append(f"Found {missing_count} missing values in {col}")
            
            # Check for negative amounts
            if 'amount' in self.data.columns:
                negative_amounts = (self.data['amount'] < 0).sum()
                if negative_amounts > 0:
                    validation_result["warnings"].append(f"Found {negative_amounts} transactions with negative amounts")
            
            # Check for future timestamps
            if 'timestamp' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
                future_dates = (self.data['timestamp'] > datetime.now()).sum()
                if future_dates > 0:
                    validation_result["warnings"].append(f"Found {future_dates} transactions with future timestamps")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def save_data(self, file_path, file_type=None, **kwargs):
        """
        Save the processed data to a file
        
        Args:
            file_path (str): Path to save the file
            file_type (str, optional): Type of file (csv, excel, parquet)
            **kwargs: Additional parameters for pandas save functions
        """
        if self.data is None:
            logger.error("No data to save")
            return
        
        try:
            # Determine file type if not provided
            if file_type is None:
                file_type = os.path.splitext(file_path)[1][1:].lower()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save data based on file type
            if file_type == 'csv':
                self.data.to_csv(file_path, index=False, **kwargs)
            elif file_type in ['xlsx', 'xls']:
                self.data.to_excel(file_path, index=False, **kwargs)
            elif file_type == 'parquet':
                self.data.to_parquet(file_path, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Data saved successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def _serialize_for_cache(self):
        """
        Serialize data loader state for caching
        
        Returns:
            dict: Serialized state
        """
        try:
            state = {
                'data': self.data,
                'original_data': self.original_data,
                'file_hash': self.file_hash,
                'file_metadata': self.file_metadata,
                'use_dask': self.use_dask,
                'chunk_size': self.chunk_size
            }
            
            # Handle Dask DataFrames
            if isinstance(self.data, dd.DataFrame):
                # For Dask DataFrames, compute and store as pandas
                state['data'] = self.data.compute()
                state['is_dask'] = True
            else:
                state['is_dask'] = False
            
            return state
            
        except Exception as e:
            logger.error(f"Error serializing data loader state: {str(e)}")
            return None
    
    def _deserialize_from_cache(self, state):
        """
        Deserialize data loader state from cache
        
        Args:
            state (dict): Serialized state
            
        Returns:
            bool: True if successful
        """
        try:
            self.data = state.get('data')
            self.original_data = state.get('original_data')
            self.file_hash = state.get('file_hash')
            self.file_metadata = state.get('file_metadata', {})
            self.use_dask = state.get('use_dask', True)
            self.chunk_size = state.get('chunk_size', 100000)
            
            # Convert back to Dask if needed
            if state.get('is_dask', False) and not isinstance(self.data, dd.DataFrame):
                import dask.dataframe as dd
                self.data = dd.from_pandas(self.data, npartitions=max(1, len(self.data) // 10000))
            
            return True
            
        except Exception as e:
            logger.error(f"Error deserializing data loader state: {str(e)}")
            return False
    
    def get_file_metadata(self):
        """
        Get file metadata
        
        Returns:
            dict: File metadata
        """
        return self.file_metadata
    
    def get_memory_usage(self):
        """
        Get memory usage information
        
        Returns:
            dict: Memory usage information
        """
        try:
            if self.data is None:
                return {"error": "No data loaded"}
            
            if isinstance(self.data, dd.DataFrame):
                # For Dask DataFrames, get memory usage of a sample
                sample = self.data.head(1000)
                memory_usage = sample.memory_usage(deep=True).sum()
                return {
                    "sample_memory_usage_mb": memory_usage / (1024 * 1024),
                    "estimated_total_memory_mb": (memory_usage / 1000) * (len(self.data) / 1000),
                    "partitions": self.data.npartitions
                }
            else:
                memory_usage = self.data.memory_usage(deep=True).sum()
                return {
                    "memory_usage_mb": memory_usage / (1024 * 1024),
                    "shape": self.data.shape
                }
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return {"error": str(e)}