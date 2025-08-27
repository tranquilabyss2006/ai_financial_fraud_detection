"""
Cache Utilities Module
Handles caching of analysis state and results to avoid reprocessing
"""
import os
import hashlib
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import shutil
import json
import pickle
import zlib
import base64
from typing import Dict, Any, Optional
import tempfile
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class AnalysisCache:
    """
    Class for handling caching of analysis state and results
    """
    
    def __init__(self, cache_dir="../cache", max_cache_size=0, compression=True, cache_expiry_days=0):
        """
        Initialize AnalysisCache
        
        Args:
            cache_dir (str): Directory to store cache files
            max_cache_size (int): Maximum number of cache files to keep (0 for unlimited)
            compression (bool): Whether to compress cache files
            cache_expiry_days (int): Number of days after which cache expires (0 for never)
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.compression = compression
        self.cache_expiry_days = cache_expiry_days
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache index
        self.cache_index = self._load_cache_index()
        
        # Initialize cache metadata
        self.cache_metadata = self._load_cache_metadata()
        
        # Clean old cache files if needed (only if limits are set)
        if self.max_cache_size > 0:
            self._clean_old_cache()
        if self.cache_expiry_days > 0:
            self._clean_expired_cache()
    
    def _load_cache_index(self):
        """Load the cache index from file"""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache index: {str(e)}")
        return {}
    
    def _save_cache_index(self):
        """Save the cache index to file"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
    
    def _load_cache_metadata(self):
        """Load the cache metadata from file"""
        try:
            if self.cache_metadata_file.exists():
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
        return {}
    
    def _save_cache_metadata(self):
        """Save the cache metadata to file"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _get_file_hash(self, file_content):
        """
        Calculate hash of file content for unique identification
        
        Args:
            file_content (bytes): Content of the file
            
        Returns:
            str: Hash string
        """
        try:
            # Use SHA-256 for better uniqueness
            return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            # Fallback to simpler hash if SHA-256 fails
            return str(hash(file_content))
    
    def _get_cache_path(self, file_hash):
        """
        Get path to cache file for a given hash
        
        Args:
            file_hash (str): File hash
            
        Returns:
            Path: Path to cache file
        """
        return self.cache_dir / f"cache_{file_hash}.joblib"
    
    def _get_metadata_path(self, file_hash):
        """
        Get path to cache metadata file for a given hash
        
        Args:
            file_hash (str): File hash
            
        Returns:
            Path: Path to cache metadata file
        """
        return self.cache_dir / f"cache_{file_hash}_metadata.json"
    
    def _clean_old_cache(self):
        """
        Clean old cache files if cache size exceeds limit
        Only runs if max_cache_size > 0
        """
        try:
            # Skip cleaning if max_cache_size is 0 (unlimited)
            if self.max_cache_size <= 0:
                return
                
            cache_files = list(self.cache_dir.glob("cache_*.joblib"))
            
            if len(cache_files) > self.max_cache_size:
                # Sort by last accessed time (oldest first)
                cache_files.sort(key=lambda x: self.cache_index.get(x.stem, {}).get('last_accessed', 0))
                
                # Remove oldest files
                for i in range(len(cache_files) - self.max_cache_size):
                    cache_file = cache_files[i]
                    try:
                        cache_file.unlink()
                        # Remove metadata file if it exists
                        metadata_file = self.cache_dir / f"{cache_file.stem}_metadata.json"
                        if metadata_file.exists():
                            metadata_file.unlink()
                        
                        # Remove from index
                        if cache_file.stem in self.cache_index:
                            del self.cache_index[cache_file.stem]
                        
                        logger.info(f"Removed old cache file: {cache_file}")
                    except Exception as e:
                        logger.error(f"Error removing cache file {cache_file}: {str(e)}")
                
                # Save updated index
                self._save_cache_index()
        except Exception as e:
            logger.error(f"Error cleaning old cache: {str(e)}")
    
    def _clean_expired_cache(self):
        """
        Clean expired cache files based on expiry days
        Only runs if cache_expiry_days > 0
        """
        try:
            # Skip cleaning if cache_expiry_days is 0 (never expire)
            if self.cache_expiry_days <= 0:
                return
                
            current_time = datetime.now()
            expired_files = []
            
            for cache_key, cache_data in self.cache_index.items():
                if 'cached_at' in cache_data:
                    cache_time = datetime.fromisoformat(cache_data['cached_at'])
                    age_days = (current_time - cache_time).days
                    
                    if age_days > self.cache_expiry_days:
                        expired_files.append(cache_key)
            
            # Remove expired files
            for cache_key in expired_files:
                try:
                    cache_file = self.cache_dir / f"{cache_key}.joblib"
                    metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
                    
                    if cache_file.exists():
                        cache_file.unlink()
                    
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    # Remove from index
                    if cache_key in self.cache_index:
                        del self.cache_index[cache_key]
                    
                    # Remove from metadata
                    if cache_key.replace("cache_", "") in self.cache_metadata:
                        del self.cache_metadata[cache_key.replace("cache_", "")]
                    
                    logger.info(f"Removed expired cache file: {cache_file}")
                except Exception as e:
                    logger.error(f"Error removing expired cache file {cache_key}: {str(e)}")
            
            # Save updated index and metadata if any files were removed
            if expired_files:
                self._save_cache_index()
                self._save_cache_metadata()
        except Exception as e:
            logger.error(f"Error cleaning expired cache: {str(e)}")
    
    def _serialize_state(self, state):
        """
        Serialize state for caching with optional compression
        
        Args:
            state (dict): State to serialize
            
        Returns:
            bytes: Serialized state
        """
        try:
            # Create a copy of the state to avoid modifying the original
            serialized_state = {}
            
            for key, value in state.items():
                if value is None:
                    serialized_state[key] = None
                elif isinstance(value, (pd.DataFrame, pd.Series)):
                    # Handle pandas objects
                    serialized_state[key] = {
                        '_type': 'pandas',
                        '_value': pickle.dumps(value),
                        '_shape': value.shape if hasattr(value, 'shape') else None,
                        '_dtypes': str(value.dtypes) if hasattr(value, 'dtypes') else None,
                        '_columns': list(value.columns) if hasattr(value, 'columns') else None
                    }
                elif isinstance(value, np.ndarray):
                    # Handle numpy arrays
                    serialized_state[key] = {
                        '_type': 'numpy',
                        '_value': pickle.dumps(value),
                        '_shape': value.shape,
                        '_dtype': str(value.dtype)
                    }
                elif hasattr(value, '__module__') and (
                    'sklearn' in value.__module__ or 
                    'tensorflow' in value.__module__ or
                    'xgboost' in value.__module__ or
                    'lightgbm' in value.__module__
                ):
                    # Handle ML models
                    serialized_state[key] = {
                        '_type': 'ml_model',
                        '_value': pickle.dumps(value),
                        '_module': value.__module__,
                        '_class': value.__class__.__name__
                    }
                elif hasattr(value, '_serialize_for_cache'):
                    # Custom objects with serialization method
                    try:
                        serialized_value = value._serialize_for_cache()
                        serialized_state[key] = {
                            '_type': 'custom',
                            '_value': serialized_value,
                            '_class': value.__class__.__name__,
                            '_module': getattr(value, '__module__', 'unknown')
                        }
                    except Exception as e:
                        logger.warning(f"Error serializing custom object {key}: {str(e)}")
                        serialized_state[key] = value
                elif isinstance(value, (dict, list, tuple, set, str, int, float, bool)):
                    # Handle basic Python types
                    serialized_state[key] = value
                else:
                    # Handle other types with pickle
                    try:
                        serialized_state[key] = {
                            '_type': 'pickle',
                            '_value': pickle.dumps(value),
                            '_class': value.__class__.__name__,
                            '_module': getattr(value, '__module__', 'unknown')
                        }
                    except Exception as e:
                        logger.warning(f"Error pickling object {key}: {str(e)}")
                        serialized_state[key] = str(value)  # Fallback to string representation
            
            # Serialize the entire state
            serialized = joblib.dumps(serialized_state)
            
            # Compress if enabled
            if self.compression:
                serialized = zlib.compress(serialized)
            
            return serialized
        except Exception as e:
            logger.error(f"Error serializing state: {str(e)}")
            return None
    
    def _deserialize_state(self, serialized_state):
        """
        Deserialize state from cache
        
        Args:
            serialized_state (bytes): Serialized state
            
        Returns:
            dict: Deserialized state
        """
        try:
            # Decompress if needed
            if self.compression:
                serialized_state = zlib.decompress(serialized_state)
            
            # Deserialize the state
            state = joblib.loads(serialized_state)
            
            # Convert special types back
            deserialized_state = {}
            for key, value in state.items():
                if value is None:
                    deserialized_state[key] = None
                elif isinstance(value, dict) and '_type' in value:
                    if value['_type'] == 'pandas':
                        try:
                            deserialized_value = pickle.loads(value['_value'])
                            # Verify shape and dtypes if available
                            if '_shape' in value and hasattr(deserialized_value, 'shape'):
                                if deserialized_value.shape != value['_shape']:
                                    logger.warning(f"Shape mismatch for {key}: expected {value['_shape']}, got {deserialized_value.shape}")
                            if '_columns' in value and hasattr(deserialized_value, 'columns'):
                                if list(deserialized_value.columns) != value['_columns']:
                                    logger.warning(f"Columns mismatch for {key}")
                            deserialized_state[key] = deserialized_value
                        except Exception as e:
                            logger.error(f"Error deserializing pandas object {key}: {str(e)}")
                            deserialized_state[key] = None
                    
                    elif value['_type'] == 'numpy':
                        try:
                            deserialized_value = pickle.loads(value['_value'])
                            # Verify shape and dtype if available
                            if '_shape' in value and deserialized_value.shape != value['_shape']:
                                logger.warning(f"Shape mismatch for {key}: expected {value['_shape']}, got {deserialized_value.shape}")
                            if '_dtype' in value and str(deserialized_value.dtype) != value['_dtype']:
                                logger.warning(f"Dtype mismatch for {key}: expected {value['_dtype']}, got {deserialized_value.dtype}")
                            deserialized_state[key] = deserialized_value
                        except Exception as e:
                            logger.error(f"Error deserializing numpy array {key}: {str(e)}")
                            deserialized_state[key] = None
                    
                    elif value['_type'] == 'ml_model':
                        try:
                            deserialized_state[key] = pickle.loads(value['_value'])
                        except Exception as e:
                            logger.error(f"Error deserializing ML model {key}: {str(e)}")
                            deserialized_state[key] = None
                    
                    elif value['_type'] == 'custom':
                        try:
                            deserialized_state[key] = value['_value']
                        except Exception as e:
                            logger.error(f"Error deserializing custom object {key}: {str(e)}")
                            deserialized_state[key] = None
                    
                    elif value['_type'] == 'pickle':
                        try:
                            deserialized_state[key] = pickle.loads(value['_value'])
                        except Exception as e:
                            logger.error(f"Error unpickling object {key}: {str(e)}")
                            deserialized_state[key] = None
                    
                    else:
                        deserialized_state[key] = value['_value']
                else:
                    deserialized_state[key] = value
            
            return deserialized_state
        except Exception as e:
            logger.error(f"Error deserializing state: {str(e)}")
            return None
    
    def save_state(self, file_content_or_hash, state, step_name="unknown"):
        """
        Save analysis state to cache
        
        Args:
            file_content_or_hash (bytes or str): Content of the uploaded file or file hash
            state (dict): Analysis state to save
            step_name (str): Name of the current step
            
        Returns:
            str: Cache file hash or None if failed
        """
        try:
            # Get file hash
            if isinstance(file_content_or_hash, bytes):
                file_hash = self._get_file_hash(file_content_or_hash)
            else:
                file_hash = file_content_or_hash
            
            # Get cache path
            cache_path = self._get_cache_path(file_hash)
            metadata_path = self._get_metadata_path(file_hash)
            
            # Add timestamp and step to state
            state['cached_at'] = datetime.now().isoformat()
            state['step_name'] = step_name
            state['file_hash'] = file_hash
            
            # Serialize state
            serialized_state = self._serialize_state(state)
            if serialized_state is None:
                logger.error("Failed to serialize state")
                return None
            
            # Save state to cache
            with open(cache_path, 'wb') as f:
                f.write(serialized_state)
            
            # Create and save metadata
            metadata = {
                'file_hash': file_hash,
                'step_name': step_name,
                'cached_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'file_size': len(file_content_or_hash) if isinstance(file_content_or_hash, bytes) else 0,
                'cache_size': len(serialized_state),
                'cache_path': str(cache_path),
                'cache_metadata_path': str(metadata_path),
                'state_keys': list(state.keys()),
                'data_shape': state.get('data', {}).shape if isinstance(state.get('data'), pd.DataFrame) else None,
                'processing_complete': state.get('processing_complete', False),
                'analysis_progress': state.get('analysis_progress', 0),
                'analysis_status': state.get('analysis_status', 'Ready')
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update cache index
            self.cache_index[f"cache_{file_hash}"] = {
                'file_hash': file_hash,
                'step_name': step_name,
                'cached_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'file_size': len(file_content_or_hash) if isinstance(file_content_or_hash, bytes) else 0,
                'cache_size': len(serialized_state),
                'cache_path': str(cache_path)
            }
            self._save_cache_index()
            
            # Update cache metadata
            self.cache_metadata[file_hash] = metadata
            self._save_cache_metadata()
            
            logger.info(f"Saved analysis state to cache: {cache_path} (step: {step_name})")
            return file_hash
            
        except Exception as e:
            logger.error(f"Error saving state to cache: {str(e)}")
            return None
    
    def load_state(self, file_content_or_hash):
        """
        Load analysis state from cache
        
        Args:
            file_content_or_hash (bytes or str): Content of the uploaded file or file hash
            
        Returns:
            dict: Cached analysis state or None if not found
        """
        try:
            # Get file hash
            if isinstance(file_content_or_hash, bytes):
                file_hash = self._get_file_hash(file_content_or_hash)
            else:
                file_hash = file_content_or_hash
            
            # Get cache path
            cache_path = self._get_cache_path(file_hash)
            metadata_path = self._get_metadata_path(file_hash)
            
            # Check if cache exists
            if not cache_path.exists():
                logger.info("No cache found for this file")
                return None
            
            # Check if cache is expired
            if f"cache_{file_hash}" in self.cache_index:
                cache_data = self.cache_index[f"cache_{file_hash}"]
                if 'cached_at' in cache_data:
                    cache_time = datetime.fromisoformat(cache_data['cached_at'])
                    if (datetime.now() - cache_time).days > self.cache_expiry_days and self.cache_expiry_days > 0:
                        logger.info("Cache has expired")
                        return None
            
            # Load serialized state from cache
            with open(cache_path, 'rb') as f:
                serialized_state = f.read()
            
            # Deserialize state
            state = self._deserialize_state(serialized_state)
            if state is None:
                logger.error("Failed to deserialize state")
                return None
            
            # Load metadata if available
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    state['cache_metadata'] = metadata
                except Exception as e:
                    logger.warning(f"Error loading metadata: {str(e)}")
            
            # Update last accessed time in index
            if f"cache_{file_hash}" in self.cache_index:
                self.cache_index[f"cache_{file_hash}"]["last_accessed"] = datetime.now().isoformat()
                self._save_cache_index()
            
            # Update last accessed time in metadata
            if file_hash in self.cache_metadata:
                self.cache_metadata[file_hash]['last_accessed'] = datetime.now().isoformat()
                self._save_cache_metadata()
            
            logger.info(f"Loaded analysis state from cache: {cache_path} (step: {state.get('step_name', 'unknown')})")
            return state
            
        except Exception as e:
            logger.error(f"Error loading state from cache: {str(e)}")
            return None
    
    def clear_cache(self):
        """
        Clear all cache files
        """
        try:
            cleared_count = 0
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Error deleting cache file {cache_file}: {str(e)}")
            
            # Clear metadata files
            for metadata_file in self.cache_dir.glob("cache_*_metadata.json"):
                try:
                    metadata_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Error deleting metadata file {metadata_file}: {str(e)}")
            
            # Clear cache index
            self.cache_index = {}
            self._save_cache_index()
            
            # Clear cache metadata
            self.cache_metadata = {}
            self._save_cache_metadata()
            
            logger.info(f"Cleared {cleared_count} cache files")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def get_cache_info(self):
        """
        Get information about cache files
        
        Returns:
            list: List of cache file information
        """
        cache_info = []
        
        try:
            for cache_key, cache_data in self.cache_index.items():
                try:
                    info = {
                        'file_hash': cache_data.get('file_hash', 'unknown'),
                        'step_name': cache_data.get('step_name', 'unknown'),
                        'cached_at': cache_data.get('cached_at', 'unknown'),
                        'last_accessed': cache_data.get('last_accessed', 'unknown'),
                        'file_size': cache_data.get('file_size', 0),
                        'cache_size': cache_data.get('cache_size', 0),
                        'cache_path': cache_data.get('cache_path', 'unknown')
                    }
                    
                    # Calculate age in days
                    if 'cached_at' in cache_data:
                        cache_time = datetime.fromisoformat(cache_data['cached_at'])
                        age_days = (datetime.now() - cache_time).days
                        info['age_days'] = age_days
                    
                    # Add metadata if available
                    if cache_data.get('file_hash') in self.cache_metadata:
                        metadata = self.cache_metadata[cache_data['file_hash']]
                        info.update({
                            'processing_complete': metadata.get('processing_complete', False),
                            'analysis_progress': metadata.get('analysis_progress', 0),
                            'analysis_status': metadata.get('analysis_status', 'Ready'),
                            'state_keys': metadata.get('state_keys', []),
                            'data_shape': metadata.get('data_shape')
                        })
                    
                    cache_info.append(info)
                except Exception as e:
                    logger.error(f"Error getting cache info for {cache_key}: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
        
        return cache_info
    
    def get_cache_step(self, file_content_or_hash):
        """
        Get the step name for a cached file
        
        Args:
            file_content_or_hash (bytes or str): Content of the uploaded file or file hash
            
        Returns:
            str: Step name or None if not cached
        """
        try:
            # Get file hash
            if isinstance(file_content_or_hash, bytes):
                file_hash = self._get_file_hash(file_content_or_hash)
            else:
                file_hash = file_content_or_hash
            
            # Check if file is in cache index
            cache_key = f"cache_{file_hash}"
            if cache_key in self.cache_index:
                return self.cache_index[cache_key]['step_name']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache step: {str(e)}")
            return None
    
    def is_cache_valid(self, file_content_or_hash):
        """
        Check if cache exists and is valid for a file
        
        Args:
            file_content_or_hash (bytes or str): Content of the uploaded file or file hash
            
        Returns:
            bool: True if cache is valid
        """
        try:
            # Get file hash
            if isinstance(file_content_or_hash, bytes):
                file_hash = self._get_file_hash(file_content_or_hash)
            else:
                file_hash = file_content_or_hash
            
            # Check if file is in cache index
            cache_key = f"cache_{file_hash}"
            if cache_key not in self.cache_index:
                return False
            
            # Check if cache file exists
            cache_path = self._get_cache_path(file_hash)
            if not cache_path.exists():
                return False
            
            # Check if cache is expired
            cache_data = self.cache_index[cache_key]
            if 'cached_at' in cache_data:
                cache_time = datetime.fromisoformat(cache_data['cached_at'])
                if (datetime.now() - cache_time).days > self.cache_expiry_days and self.cache_expiry_days > 0:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {str(e)}")
            return False
    
    def get_cache_size(self):
        """
        Get total size of all cache files in MB
        
        Returns:
            float: Total cache size in MB
        """
        try:
            total_size = 0
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                try:
                    total_size += cache_file.stat().st_size
                except Exception as e:
                    logger.error(f"Error getting size of {cache_file}: {str(e)}")
            
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting cache size: {str(e)}")
            return 0
    
    def get_cache_stats(self):
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("cache_*.joblib"))
            total_size = self.get_cache_size()
            
            # Calculate age statistics
            ages = []
            processing_complete_count = 0
            total_progress = 0
            
            for cache_key, cache_data in self.cache_index.items():
                if 'cached_at' in cache_data:
                    cache_time = datetime.fromisoformat(cache_data['cached_at'])
                    age_days = (datetime.now() - cache_time).days
                    ages.append(age_days)
                
                # Check if processing is complete
                if cache_data.get('file_hash') in self.cache_metadata:
                    metadata = self.cache_metadata[cache_data['file_hash']]
                    if metadata.get('processing_complete', False):
                        processing_complete_count += 1
                    
                    total_progress += metadata.get('analysis_progress', 0)
            
            stats = {
                'total_files': len(cache_files),
                'total_size_mb': total_size,
                'max_cache_size': self.max_cache_size,
                'compression_enabled': self.compression,
                'cache_expiry_days': self.cache_expiry_days,
                'processing_complete_count': processing_complete_count
            }
            
            if ages:
                stats['oldest_cache_days'] = max(ages)
                stats['newest_cache_days'] = min(ages)
                stats['average_age_days'] = sum(ages) / len(ages)
            
            if processing_complete_count > 0:
                stats['average_progress'] = total_progress / processing_complete_count
            
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def get_file_info(self, file_hash):
        """
        Get information about a cached file
        
        Args:
            file_hash (str): File hash
            
        Returns:
            dict: File information or None if not found
        """
        try:
            if file_hash in self.cache_metadata:
                return self.cache_metadata[file_hash]
            return None
        except Exception as e:
            logger.error(f"Error getting file info for {file_hash}: {str(e)}")
            return None
    
    def update_file_metadata(self, file_hash, metadata_updates):
        """
        Update metadata for a cached file
        
        Args:
            file_hash (str): File hash
            metadata_updates (dict): Metadata updates
            
        Returns:
            bool: True if successful
        """
        try:
            if file_hash not in self.cache_metadata:
                logger.warning(f"No metadata found for file hash: {file_hash}")
                return False
            
            # Update metadata
            self.cache_metadata[file_hash].update(metadata_updates)
            
            # Save updated metadata
            self._save_cache_metadata()
            
            logger.info(f"Updated metadata for file hash: {file_hash}")
            return True
        except Exception as e:
            logger.error(f"Error updating file metadata for {file_hash}: {str(e)}")
            return False
    
    def export_cache(self, export_dir=None):
        """
        Export all cache files to a directory
        
        Args:
            export_dir (str, optional): Directory to export to
            
        Returns:
            str: Path to exported directory or None if failed
        """
        try:
            if export_dir is None:
                export_dir = tempfile.mkdtemp(prefix="cache_export_")
            
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy cache files
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                try:
                    import shutil
                    shutil.copy2(cache_file, export_dir)
                except Exception as e:
                    logger.error(f"Error copying cache file {cache_file}: {str(e)}")
            
            # Copy metadata files
            for metadata_file in self.cache_dir.glob("cache_*_metadata.json"):
                try:
                    import shutil
                    shutil.copy2(metadata_file, export_dir)
                except Exception as e:
                    logger.error(f"Error copying metadata file {metadata_file}: {str(e)}")
            
            # Copy index file
            if self.cache_index_file.exists():
                try:
                    import shutil
                    shutil.copy2(self.cache_index_file, export_dir)
                except Exception as e:
                    logger.error(f"Error copying index file: {str(e)}")
            
            logger.info(f"Exported cache to: {export_dir}")
            return str(export_dir)
        except Exception as e:
            logger.error(f"Error exporting cache: {str(e)}")
            return None
    
    def import_cache(self, import_dir):
        """
        Import cache files from a directory
        
        Args:
            import_dir (str): Directory to import from
            
        Returns:
            bool: True if successful
        """
        try:
            import_dir = Path(import_dir)
            
            if not import_dir.exists():
                logger.error(f"Import directory does not exist: {import_dir}")
                return False
            
            # Import cache files
            for cache_file in import_dir.glob("cache_*.joblib"):
                try:
                    import shutil
                    shutil.copy2(cache_file, self.cache_dir)
                except Exception as e:
                    logger.error(f"Error importing cache file {cache_file}: {str(e)}")
            
            # Import metadata files
            for metadata_file in import_dir.glob("cache_*_metadata.json"):
                try:
                    import shutil
                    shutil.copy2(metadata_file, self.cache_dir)
                except Exception as e:
                    logger.error(f"Error importing metadata file {metadata_file}: {str(e)}")
            
            # Import index file
            index_file = import_dir / "cache_index.json"
            if index_file.exists():
                try:
                    import shutil
                    shutil.copy2(index_file, self.cache_dir)
                    # Reload index
                    self.cache_index = self._load_cache_index()
                except Exception as e:
                    logger.error(f"Error importing index file: {str(e)}")
            
            # Reload metadata
            self.cache_metadata = self._load_cache_metadata()
            
            logger.info(f"Imported cache from: {import_dir}")
            return True
        except Exception as e:
            logger.error(f"Error importing cache: {str(e)}")
            return False