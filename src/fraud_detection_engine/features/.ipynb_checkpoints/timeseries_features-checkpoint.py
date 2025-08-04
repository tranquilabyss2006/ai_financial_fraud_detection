"""
Time Series Features Module
Implements time series feature extraction techniques for fraud detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import warnings
import logging
from typing import Dict, List, Tuple, Union

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesFeatures:
    """
    Class for extracting time series features from transaction data
    Implements techniques like burstiness analysis, gap analysis, etc.
    """
    
    def __init__(self, config=None):
        """
        Initialize TimeSeriesFeatures
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.feature_names = []
        self.fitted = False
        
    def extract_features(self, df):
        """
        Extract all time series features from the dataframe
        
        Args:
            df (DataFrame): Input transaction data
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Ensure timestamp is in datetime format
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                result_df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract different types of time series features
            result_df = self._extract_temporal_features(result_df)
            result_df = self._extract_frequency_features(result_df)
            result_df = self._extract_burstiness_features(result_df)
            result_df = self._extract_gap_features(result_df)
            result_df = self._extract_seasonal_features(result_df)
            result_df = self._extract_autocorrelation_features(result_df)
            
            # Store feature names
            self.feature_names = [col for col in result_df.columns if col not in df.columns]
            
            logger.info(f"Extracted {len(self.feature_names)} time series features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting time series features: {str(e)}")
            raise
    
    def _extract_temporal_features(self, df):
        """
        Extract temporal features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with temporal features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                logger.warning("Timestamp column not found. Skipping temporal features.")
                return result_df
            
            # Extract time components
            result_df['hour'] = result_df['timestamp'].dt.hour
            result_df['day'] = result_df['timestamp'].dt.day
            result_df['month'] = result_df['timestamp'].dt.month
            result_df['year'] = result_df['timestamp'].dt.year
            result_df['dayofweek'] = result_df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
            result_df['dayofyear'] = result_df['timestamp'].dt.dayofyear
            result_df['weekofyear'] = result_df['timestamp'].dt.isocalendar().week
            result_df['quarter'] = result_df['timestamp'].dt.quarter
            
            # Time-based flags
            result_df['is_weekend'] = (result_df['dayofweek'] >= 5).astype(int)
            result_df['is_month_start'] = result_df['timestamp'].dt.is_month_start.astype(int)
            result_df['is_month_end'] = result_df['timestamp'].dt.is_month_end.astype(int)
            result_df['is_quarter_start'] = result_df['timestamp'].dt.is_quarter_start.astype(int)
            result_df['is_quarter_end'] = result_df['timestamp'].dt.is_quarter_end.astype(int)
            result_df['is_year_start'] = result_df['timestamp'].dt.is_year_start.astype(int)
            result_df['is_year_end'] = result_df['timestamp'].dt.is_year_end.astype(int)
            
            # Time of day flags
            result_df['is_night'] = ((result_df['hour'] >= 22) | (result_df['hour'] < 6)).astype(int)
            result_df['is_morning'] = ((result_df['hour'] >= 6) & (result_df['hour'] < 12)).astype(int)
            result_df['is_afternoon'] = ((result_df['hour'] >= 12) & (result_df['hour'] < 18)).astype(int)
            result_df['is_evening'] = ((result_df['hour'] >= 18) & (result_df['hour'] < 22)).astype(int)
            
            # Business hours flag (9 AM to 5 PM, Monday to Friday)
            result_df['is_business_hours'] = (
                (result_df['hour'] >= 9) & 
                (result_df['hour'] < 17) & 
                (result_df['dayofweek'] < 5)
            ).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            return df
    
    def _extract_frequency_features(self, df):
        """
        Extract frequency-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with frequency features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                logger.warning("Timestamp column not found. Skipping frequency features.")
                return result_df
            
            # Sort by timestamp
            df_sorted = result_df.sort_values('timestamp')
            
            # Time windows for frequency analysis
            time_windows = ['1H', '6H', '24H', '7D', '30D']
            
            for window in time_windows:
                # Calculate transaction frequency in time windows before each transaction
                window_counts = []
                
                for _, row in df_sorted.iterrows():
                    timestamp = row['timestamp']
                    
                    # Get transactions in time window before current transaction
                    window_start = timestamp - pd.Timedelta(window)
                    window_end = timestamp
                    
                    window_transactions = df_sorted[
                        (df_sorted['timestamp'] >= window_start) &
                        (df_sorted['timestamp'] < window_end)
                    ]
                    
                    # Count transactions in window
                    count = len(window_transactions)
                    window_counts.append(count)
                
                result_df[f'transaction_frequency_{window}'] = window_counts
            
            # Calculate frequency by sender
            if 'sender_id' in df.columns:
                for window in ['1H', '6H', '24H']:
                    sender_window_counts = []
                    
                    for _, row in df_sorted.iterrows():
                        sender = row['sender_id']
                        timestamp = row['timestamp']
                        
                        # Get transactions from same sender in time window
                        window_start = timestamp - pd.Timedelta(window)
                        window_end = timestamp
                        
                        window_transactions = df_sorted[
                            (df_sorted['timestamp'] >= window_start) &
                            (df_sorted['timestamp'] < window_end) &
                            (df_sorted['sender_id'] == sender)
                        ]
                        
                        # Count transactions in window
                        count = len(window_transactions)
                        sender_window_counts.append(count)
                    
                    result_df[f'sender_frequency_{window}'] = sender_window_counts
            
            # Calculate frequency by receiver
            if 'receiver_id' in df.columns:
                for window in ['1H', '6H', '24H']:
                    receiver_window_counts = []
                    
                    for _, row in df_sorted.iterrows():
                        receiver = row['receiver_id']
                        timestamp = row['timestamp']
                        
                        # Get transactions to same receiver in time window
                        window_start = timestamp - pd.Timedelta(window)
                        window_end = timestamp
                        
                        window_transactions = df_sorted[
                            (df_sorted['timestamp'] >= window_start) &
                            (df_sorted['timestamp'] < window_end) &
                            (df_sorted['receiver_id'] == receiver)
                        ]
                        
                        # Count transactions in window
                        count = len(window_transactions)
                        receiver_window_counts.append(count)
                    
                    result_df[f'receiver_frequency_{window}'] = receiver_window_counts
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting frequency features: {str(e)}")
            return df
    
    def _extract_burstiness_features(self, df):
        """
        Extract burstiness analysis features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with burstiness features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                logger.warning("Timestamp column not found. Skipping burstiness features.")
                return result_df
            
            # Sort by timestamp
            df_sorted = result_df.sort_values('timestamp')
            
            # Calculate inter-transaction times
            inter_times = df_sorted['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Calculate burstiness coefficient
            if len(inter_times) > 1 and inter_times.std() > 0:
                burstiness = (inter_times.std() - inter_times.mean()) / (inter_times.std() + inter_times.mean())
            else:
                burstiness = 0
            
            result_df['burstiness_coefficient'] = burstiness
            
            # Calculate local burstiness (in sliding windows)
            window_sizes = [10, 50, 100]  # Number of transactions
            
            for window_size in window_sizes:
                local_burstiness = []
                
                for i in range(len(df_sorted)):
                    # Get window around current transaction
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(df_sorted), i + window_size // 2 + 1)
                    
                    window_inter_times = inter_times.iloc[start_idx:end_idx]
                    
                    # Calculate burstiness for window
                    if len(window_inter_times) > 1 and window_inter_times.std() > 0:
                        local_b = (window_inter_times.std() - window_inter_times.mean()) / (
                            window_inter_times.std() + window_inter_times.mean()
                        )
                    else:
                        local_b = 0
                    
                    local_burstiness.append(local_b)
                
                result_df[f'local_burstiness_{window_size}'] = local_burstiness
            
            # Detect burst periods
            # A burst is defined as a period with inter-transaction times significantly lower than average
            avg_inter_time = inter_times.mean()
            std_inter_time = inter_times.std()
            
            # Threshold for burst detection (2 standard deviations below mean)
            burst_threshold = max(0, avg_inter_time - 2 * std_inter_time)
            
            # Flag transactions in bursts
            result_df['is_in_burst'] = (inter_times < burst_threshold).astype(int)
            
            # Calculate burst duration (consecutive transactions in burst)
            burst_durations = []
            current_duration = 0
            
            for is_burst in result_df['is_in_burst']:
                if is_burst:
                    current_duration += 1
                else:
                    burst_durations.append(current_duration)
                    current_duration = 0
            
            # Add the last duration if we're still in a burst
            burst_durations.append(current_duration)
            
            # Map burst durations back to transactions
            burst_duration_map = []
            idx = 0
            for i, is_burst in enumerate(result_df['is_in_burst']):
                if is_burst:
                    burst_duration_map.append(burst_durations[idx])
                else:
                    burst_duration_map.append(0)
                    if i > 0 and not result_df['is_in_burst'].iloc[i-1]:
                        idx += 1
            
            result_df['burst_duration'] = burst_duration_map
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting burstiness features: {str(e)}")
            return df
    
    def _extract_gap_features(self, df):
        """
        Extract gap analysis features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with gap features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                logger.warning("Timestamp column not found. Skipping gap features.")
                return result_df
            
            # Sort by timestamp
            df_sorted = result_df.sort_values('timestamp')
            
            # Calculate inter-transaction times
            inter_times = df_sorted['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Time since last transaction
            result_df['time_since_last_transaction'] = inter_times
            
            # Time until next transaction
            time_until_next = df_sorted['timestamp'].diff(-1).dt.total_seconds().abs().fillna(0)
            result_df['time_until_next_transaction'] = time_until_next
            
            # Detect gaps (unusually long inter-transaction times)
            if len(inter_times) > 1 and inter_times.std() > 0:
                # Threshold for gap detection (2 standard deviations above mean)
                gap_threshold = inter_times.mean() + 2 * inter_times.std()
                
                # Flag transactions after gaps
                result_df['is_after_gap'] = (inter_times > gap_threshold).astype(int)
                
                # Calculate gap sizes
                result_df['gap_size'] = np.where(inter_times > gap_threshold, inter_times, 0)
            else:
                result_df['is_after_gap'] = 0
                result_df['gap_size'] = 0
            
            # Calculate gap statistics by sender
            if 'sender_id' in df.columns:
                sender_gaps = []
                sender_gap_stats = {}
                
                # Calculate average gap for each sender
                for sender in df_sorted['sender_id'].unique():
                    sender_transactions = df_sorted[df_sorted['sender_id'] == sender].sort_values('timestamp')
                    if len(sender_transactions) > 1:
                        sender_inter_times = sender_transactions['timestamp'].diff().dt.total_seconds().fillna(0)
                        sender_gap_stats[sender] = {
                            'mean': sender_inter_times.mean(),
                            'std': sender_inter_times.std()
                        }
                
                # Calculate gap features for each transaction
                for _, row in df_sorted.iterrows():
                    sender = row['sender_id']
                    
                    if sender in sender_gap_stats:
                        stats = sender_gap_stats[sender]
                        
                        # Time since last transaction for this sender
                        sender_transactions = df_sorted[
                            (df_sorted['sender_id'] == sender) &
                            (df_sorted['timestamp'] < row['timestamp'])
                        ]
                        
                        if len(sender_transactions) > 0:
                            last_sender_time = sender_transactions['timestamp'].max()
                            sender_inter_time = (row['timestamp'] - last_sender_time).total_seconds()
                        else:
                            sender_inter_time = float('inf')
                        
                        # Calculate Z-score for this gap
                        if stats['std'] > 0:
                            gap_z_score = (sender_inter_time - stats['mean']) / stats['std']
                        else:
                            gap_z_score = 0
                        
                        sender_gaps.append({
                            'sender_time_since_last': sender_inter_time,
                            'sender_gap_z_score': gap_z_score
                        })
                    else:
                        sender_gaps.append({
                            'sender_time_since_last': float('inf'),
                            'sender_gap_z_score': 0
                        })
                
                # Add to result dataframe
                gap_df = pd.DataFrame(sender_gaps)
                result_df['sender_time_since_last'] = gap_df['sender_time_since_last']
                result_df['sender_gap_z_score'] = gap_df['sender_gap_z_score']
            
            # Calculate gap statistics by receiver
            if 'receiver_id' in df.columns:
                receiver_gaps = []
                receiver_gap_stats = {}
                
                # Calculate average gap for each receiver
                for receiver in df_sorted['receiver_id'].unique():
                    receiver_transactions = df_sorted[df_sorted['receiver_id'] == receiver].sort_values('timestamp')
                    if len(receiver_transactions) > 1:
                        receiver_inter_times = receiver_transactions['timestamp'].diff().dt.total_seconds().fillna(0)
                        receiver_gap_stats[receiver] = {
                            'mean': receiver_inter_times.mean(),
                            'std': receiver_inter_times.std()
                        }
                
                # Calculate gap features for each transaction
                for _, row in df_sorted.iterrows():
                    receiver = row['receiver_id']
                    
                    if receiver in receiver_gap_stats:
                        stats = receiver_gap_stats[receiver]
                        
                        # Time since last transaction to this receiver
                        receiver_transactions = df_sorted[
                            (df_sorted['receiver_id'] == receiver) &
                            (df_sorted['timestamp'] < row['timestamp'])
                        ]
                        
                        if len(receiver_transactions) > 0:
                            last_receiver_time = receiver_transactions['timestamp'].max()
                            receiver_inter_time = (row['timestamp'] - last_receiver_time).total_seconds()
                        else:
                            receiver_inter_time = float('inf')
                        
                        # Calculate Z-score for this gap
                        if stats['std'] > 0:
                            gap_z_score = (receiver_inter_time - stats['mean']) / stats['std']
                        else:
                            gap_z_score = 0
                        
                        receiver_gaps.append({
                            'receiver_time_since_last': receiver_inter_time,
                            'receiver_gap_z_score': gap_z_score
                        })
                    else:
                        receiver_gaps.append({
                            'receiver_time_since_last': float('inf'),
                            'receiver_gap_z_score': 0
                        })
                
                # Add to result dataframe
                gap_df = pd.DataFrame(receiver_gaps)
                result_df['receiver_time_since_last'] = gap_df['receiver_time_since_last']
                result_df['receiver_gap_z_score'] = gap_df['receiver_gap_z_score']
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting gap features: {str(e)}")
            return df
    
    def _extract_seasonal_features(self, df):
        """
        Extract seasonal features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with seasonal features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                logger.warning("Timestamp column not found. Skipping seasonal features.")
                return result_df
            
            # Sort by timestamp
            df_sorted = result_df.sort_values('timestamp')
            
            # Create a time series of transaction counts
            # Resample to different frequencies
            frequencies = ['1H', '6H', '1D', '1W']
            
            for freq in frequencies:
                # Count transactions per time period
                ts = df_sorted.set_index('timestamp').resample(freq).size()
                
                if len(ts) > 10:  # Need enough data points for decomposition
                    try:
                        # Perform seasonal decomposition
                        decomposition = seasonal_decompose(ts, model='additive', period=min(24, len(ts)//2))
                        
                        # Extract components
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                        residual = decomposition.resid
                        
                        # Map back to original dataframe
                        trend_map = {}
                        seasonal_map = {}
                        residual_map = {}
                        
                        for timestamp in df_sorted['timestamp']:
                            # Find the closest time period
                            period_start = timestamp.floor(freq)
                            
                            if period_start in trend.index:
                                trend_map[timestamp] = trend[period_start]
                                seasonal_map[timestamp] = seasonal[period_start]
                                residual_map[timestamp] = residual[period_start]
                            else:
                                trend_map[timestamp] = 0
                                seasonal_map[timestamp] = 0
                                residual_map[timestamp] = 0
                        
                        # Add to result dataframe
                        result_df[f'trend_{freq}'] = df_sorted['timestamp'].map(trend_map).fillna(0)
                        result_df[f'seasonal_{freq}'] = df_sorted['timestamp'].map(seasonal_map).fillna(0)
                        result_df[f'residual_{freq}'] = df_sorted['timestamp'].map(residual_map).fillna(0)
                        
                        # Calculate anomaly score based on residual
                        residual_mean = residual.mean()
                        residual_std = residual.std()
                        
                        if residual_std > 0:
                            anomaly_scores = (df_sorted['timestamp'].map(residual_map) - residual_mean) / residual_std
                            result_df[f'seasonal_anomaly_{freq}'] = anomaly_scores.fillna(0)
                        else:
                            result_df[f'seasonal_anomaly_{freq}'] = 0
                            
                    except Exception as e:
                        logger.warning(f"Error in seasonal decomposition for {freq}: {str(e)}")
            
            # Calculate periodicity features
            # Check for daily, weekly, and monthly patterns
            periodicity_features = {}
            
            # Daily pattern (hour of day)
            if 'hour' in result_df.columns:
                hourly_counts = result_df.groupby('hour').size()
                # Calculate entropy to measure uniformity
                hourly_probs = hourly_counts / hourly_counts.sum()
                daily_entropy = -sum(p * np.log2(p) for p in hourly_probs if p > 0)
                max_entropy = np.log2(24)  # Maximum possible entropy for 24 hours
                daily_uniformity = daily_entropy / max_entropy if max_entropy > 0 else 0
                
                periodicity_features['daily_pattern_strength'] = 1 - daily_uniformity  # Higher means stronger pattern
                
                # Calculate peak hour
                peak_hour = hourly_counts.idxmax()
                periodicity_features['peak_hour'] = peak_hour
            
            # Weekly pattern (day of week)
            if 'dayofweek' in result_df.columns:
                weekly_counts = result_df.groupby('dayofweek').size()
                # Calculate entropy to measure uniformity
                weekly_probs = weekly_counts / weekly_counts.sum()
                weekly_entropy = -sum(p * np.log2(p) for p in weekly_probs if p > 0)
                max_entropy = np.log2(7)  # Maximum possible entropy for 7 days
                weekly_uniformity = weekly_entropy / max_entropy if max_entropy > 0 else 0
                
                periodicity_features['weekly_pattern_strength'] = 1 - weekly_uniformity  # Higher means stronger pattern
                
                # Calculate peak day
                peak_day = weekly_counts.idxmax()
                periodicity_features['peak_day'] = peak_day
            
            # Add periodicity features to all rows
            for feature, value in periodicity_features.items():
                result_df[feature] = value
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting seasonal features: {str(e)}")
            return df
    
    def _extract_autocorrelation_features(self, df):
        """
        Extract autocorrelation features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with autocorrelation features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                logger.warning("Timestamp column not found. Skipping autocorrelation features.")
                return result_df
            
            # Sort by timestamp
            df_sorted = result_df.sort_values('timestamp')
            
            # Create time series of transaction counts
            # Resample to different frequencies
            frequencies = ['1H', '6H', '1D']
            
            for freq in frequencies:
                # Count transactions per time period
                ts = df_sorted.set_index('timestamp').resample(freq).size()
                
                if len(ts) > 10:  # Need enough data points for autocorrelation
                    try:
                        # Calculate autocorrelation function
                        nlags = min(10, len(ts) // 2)
                        autocorr = acf(ts, nlags=nlags, fft=True)
                        
                        # Add autocorrelation features
                        for i in range(1, min(6, len(autocorr))):  # First 5 lags
                            result_df[f'autocorr_{freq}_lag_{i}'] = autocorr[i]
                        
                        # Calculate partial autocorrelation
                        pacf_values = pacf(ts, nlags=nlags)
                        
                        # Add partial autocorrelation features
                        for i in range(1, min(6, len(pacf_values))):  # First 5 lags
                            result_df[f'pacf_{freq}_lag_{i}'] = pacf_values[i]
                        
                        # Detect periodicity in autocorrelation
                        # Look for peaks in autocorrelation
                        peaks, _ = find_peaks(autocorr[1:], height=0.2)  # Ignore lag 0
                        
                        if len(peaks) > 0:
                            # Find the most significant peak
                            peak_lag = peaks[0] + 1  # +1 because we ignored lag 0
                            result_df[f'periodicity_{freq}'] = peak_lag
                            result_df[f'periodicity_strength_{freq}'] = autocorr[peak_lag]
                        else:
                            result_df[f'periodicity_{freq}'] = 0
                            result_df[f'periodicity_strength_{freq}'] = 0
                            
                    except Exception as e:
                        logger.warning(f"Error in autocorrelation analysis for {freq}: {str(e)}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting autocorrelation features: {str(e)}")
            return df
    
    def fit_transform(self, df):
        """
        Fit the feature extractor and transform the data
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Transformed data with features
        """
        # Extract features
        result_df = self.extract_features(df)
        
        # Get feature columns
        feature_cols = [col for col in result_df.columns if col not in df.columns]
        
        if len(feature_cols) > 0:
            self.fitted = True
        
        return result_df
    
    def transform(self, df):
        """
        Transform new data using fitted feature extractor
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Transformed data with features
        """
        if not self.fitted:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        # Extract features
        result_df = self.extract_features(df)
        
        return result_df