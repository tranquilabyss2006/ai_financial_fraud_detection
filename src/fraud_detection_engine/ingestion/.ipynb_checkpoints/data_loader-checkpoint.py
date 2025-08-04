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
                    df[col] = df[col].fillna(median_val)
                # For categorical columns, fill with mode or 'Unknown'
                elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
            
            # Convert timestamp columns to datetime
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert {col} to datetime")
            
            # Convert amount columns to numeric
            for col in df.columns:
                if 'amount' in col.lower() or 'value' in col.lower() or 'sum' in col.lower():
                    try:
                        # Remove currency symbols and commas
                        if pd.api.types.is_string_dtype(df[col]):
                            df[col] = df[col].str.replace('[\$,€,£,¥]', '', regex=True)
                            df[col] = df[col].str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert {col} to numeric")
            
            # Convert ID columns to string
            for col in df.columns:
                if 'id' in col.lower():
                    df[col] = df[col].astype(str)
            
            # Convert boolean columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2 and all(val.lower() in ['true', 'false', 'yes', 'no', 'y', 'n', '0', '1'] for val in unique_vals):
                        df[col] = df[col].replace({
                            'true': True, 'yes': True, 'y': True, '1': True,
                            'false': False, 'no': False, 'n': False, '0': False
                        })
            
            # Convert categorical columns with many unique values to 'category' dtype
            for col in df.columns:
                if (pd.api.types.is_string_dtype(df[col]) and 
                    df[col].nunique() > 10 and 
                    df[col].nunique() < len(df) * 0.5):
                    df[col] = df[col].astype('category')
            
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
        
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": dict(self.data.dtypes),
            "missing_values": dict(self.data.isnull().sum()),
            "memory_usage": dict(self.data.memory_usage(deep=True))
        }
        
        # Add statistics for numeric columns
        numeric_stats = {}
        for col in self.data.select_dtypes(include=[np.number]).columns:
            numeric_stats[col] = {
                "min": self.data[col].min(),
                "max": self.data[col].max(),
                "mean": self.data[col].mean(),
                "median": self.data[col].median(),
                "std": self.data[col].std()
            }
        
        info["numeric_stats"] = numeric_stats
        
        # Add unique counts for categorical columns
        categorical_stats = {}
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            categorical_stats[col] = {
                "unique_count": self.data[col].nunique(),
                "top_values": self.data[col].value_counts().head(5).to_dict()
            }
        
        info["categorical_stats"] = categorical_stats
        
        return info
    
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
        
        if isinstance(self.data, dd.DataFrame):
            return self.data.head(n)
        else:
            return self.data.sample(n) if len(self.data) > n else self.data
    
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