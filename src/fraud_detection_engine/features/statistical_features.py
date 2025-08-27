"""
Statistical Features Module
Implements various statistical feature extraction techniques for fraud detection
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import logging
from typing import Dict, List, Tuple, Union
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class StatisticalFeatures:
    """
    Class for extracting statistical features from transaction data
    Implements techniques like Benford's Law, Z-score, MAD, etc.
    """
    
    def __init__(self, config=None):
        """
        Initialize StatisticalFeatures
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.pca = None
        self.fitted = False
        
    def extract_features(self, df):
        """
        Extract all statistical features from the dataframe
        
        Args:
            df (DataFrame): Input transaction data
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Extract different types of statistical features
            result_df = self._extract_benford_features(result_df)
            result_df = self._extract_zscore_features(result_df)
            result_df = self._extract_mad_features(result_df)
            result_df = self._extract_percentile_features(result_df)
            result_df = self._extract_distribution_features(result_df)
            result_df = self._extract_mahalanobis_features(result_df)
            result_df = self._extract_grubbs_features(result_df)
            result_df = self._extract_entropy_features(result_df)
            result_df = self._extract_correlation_features(result_df)
            
            # Store feature names
            self.feature_names = [col for col in result_df.columns if col not in df.columns]
            
            logger.info(f"Extracted {len(self.feature_names)} statistical features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting statistical features: {str(e)}")
            raise
    
    def _extract_benford_features(self, df):
        """
        Extract Benford's Law features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with Benford's Law features
        """
        try:
            # Apply Benford's Law to amount column if it exists
            if 'amount' in df.columns:
                # Get first digit of amounts
                amounts = df['amount'].abs()
                first_digits = amounts.astype(str).str[0].replace('n', '0').astype(int)
                first_digits = first_digits[first_digits >= 1]  # Exclude 0
                
                if len(first_digits) > 0:
                    # Calculate actual distribution
                    actual_dist = first_digits.value_counts(normalize=True).sort_index()
                    
                    # Expected Benford's distribution
                    benford_dist = pd.Series([np.log10(1 + 1/d) for d in range(1, 10)], index=range(1, 10))
                    
                    # Calculate Chi-square statistic
                    chi_square = 0
                    for digit in range(1, 10):
                        expected_count = benford_dist[digit] * len(first_digits)
                        actual_count = actual_dist.get(digit, 0)
                        if expected_count > 0:
                            chi_square += (actual_count - expected_count) ** 2 / expected_count
                    
                    # Add features
                    result_df = df.copy()
                    result_df['benford_chi_square'] = chi_square
                    result_df['benford_p_value'] = 1 - stats.chi2.cdf(chi_square, 8)  # 8 degrees of freedom
                    
                    # Calculate deviation for each digit
                    for i in range(1, 6):  # Just first 5 digits to avoid too many features
                        actual_pct = actual_dist.get(i, 0)
                        expected_pct = benford_dist[i]
                        result_df[f'benford_deviation_{i}'] = abs(actual_pct - expected_pct)
                    
                    return result_df
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error extracting Benford's Law features: {str(e)}")
            return df.copy()
    
    def _extract_zscore_features(self, df):
        """
        Extract Z-score based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with Z-score features
        """
        try:
            result_df = df.copy()
            
            # Apply to amount column if it exists
            if 'amount' in df.columns:
                amounts = df['amount']
                
                # Calculate Z-scores
                mean_amount = amounts.mean()
                std_amount = amounts.std()
                
                if std_amount > 0:
                    z_scores = (amounts - mean_amount) / std_amount
                    result_df['amount_zscore'] = z_scores
                    
                    # Flag extreme values
                    result_df['amount_zscore_outlier'] = (np.abs(z_scores) > 3).astype(int)
                else:
                    result_df['amount_zscore'] = 0
                    result_df['amount_zscore_outlier'] = 0
            
            # Apply Z-score to sender and receiver if ID columns exist
            if 'sender_id' in df.columns and 'amount' in df.columns:
                # Calculate average amount per sender
                sender_avg = df.groupby('sender_id')['amount'].mean()
                sender_std = df.groupby('sender_id')['amount'].std()
                
                # Calculate Z-scores for each transaction relative to sender's history
                sender_zscores = []
                for _, row in df.iterrows():
                    sender_id = row['sender_id']
                    amount = row['amount']
                    
                    if sender_id in sender_avg and sender_id in sender_std and sender_std[sender_id] > 0:
                        z_score = (amount - sender_avg[sender_id]) / sender_std[sender_id]
                    else:
                        z_score = 0
                    
                    sender_zscores.append(z_score)
                
                result_df['sender_amount_zscore'] = sender_zscores
                result_df['sender_amount_zscore_outlier'] = (np.abs(sender_zscores) > 3).astype(int)
            
            if 'receiver_id' in df.columns and 'amount' in df.columns:
                # Calculate average amount per receiver
                receiver_avg = df.groupby('receiver_id')['amount'].mean()
                receiver_std = df.groupby('receiver_id')['amount'].std()
                
                # Calculate Z-scores for each transaction relative to receiver's history
                receiver_zscores = []
                for _, row in df.iterrows():
                    receiver_id = row['receiver_id']
                    amount = row['amount']
                    
                    if receiver_id in receiver_avg and receiver_id in receiver_std and receiver_std[receiver_id] > 0:
                        z_score = (amount - receiver_avg[receiver_id]) / receiver_std[receiver_id]
                    else:
                        z_score = 0
                    
                    receiver_zscores.append(z_score)
                
                result_df['receiver_amount_zscore'] = receiver_zscores
                result_df['receiver_amount_zscore_outlier'] = (np.abs(receiver_zscores) > 3).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting Z-score features: {str(e)}")
            return df.copy()
    
    def _extract_mad_features(self, df):
        """
        Extract Median Absolute Deviation (MAD) features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with MAD features
        """
        try:
            result_df = df.copy()
            
            # Apply to amount column if it exists
            if 'amount' in df.columns:
                amounts = df['amount']
                
                # Calculate MAD
                median_amount = amounts.median()
                abs_dev = np.abs(amounts - median_amount)
                mad = abs_dev.median()
                
                if mad > 0:
                    # Calculate modified Z-scores using MAD
                    modified_z_scores = 0.6745 * abs_dev / mad
                    result_df['amount_mad_zscore'] = modified_z_scores
                    
                    # Flag extreme values
                    result_df['amount_mad_outlier'] = (modified_z_scores > 3.5).astype(int)
                else:
                    result_df['amount_mad_zscore'] = 0
                    result_df['amount_mad_outlier'] = 0
            
            # Apply MAD to sender and receiver if ID columns exist
            if 'sender_id' in df.columns and 'amount' in df.columns:
                # Calculate MAD per sender
                sender_mad = df.groupby('sender_id')['amount'].apply(lambda x: np.median(np.abs(x - x.median())))
                sender_median = df.groupby('sender_id')['amount'].median()
                
                # Calculate MAD-based Z-scores for each transaction
                sender_mad_zscores = []
                for _, row in df.iterrows():
                    sender_id = row['sender_id']
                    amount = row['amount']
                    
                    if sender_id in sender_mad and sender_id in sender_median and sender_mad[sender_id] > 0:
                        abs_dev = abs(amount - sender_median[sender_id])
                        mad_z_score = 0.6745 * abs_dev / sender_mad[sender_id]
                    else:
                        mad_z_score = 0
                    
                    sender_mad_zscores.append(mad_z_score)
                
                result_df['sender_amount_mad_zscore'] = sender_mad_zscores
                result_df['sender_amount_mad_outlier'] = (np.array(sender_mad_zscores) > 3.5).astype(int)
            
            if 'receiver_id' in df.columns and 'amount' in df.columns:
                # Calculate MAD per receiver
                receiver_mad = df.groupby('receiver_id')['amount'].apply(lambda x: np.median(np.abs(x - x.median())))
                receiver_median = df.groupby('receiver_id')['amount'].median()
                
                # Calculate MAD-based Z-scores for each transaction
                receiver_mad_zscores = []
                for _, row in df.iterrows():
                    receiver_id = row['receiver_id']
                    amount = row['amount']
                    
                    if receiver_id in receiver_mad and receiver_id in receiver_median and receiver_mad[receiver_id] > 0:
                        abs_dev = abs(amount - receiver_median[receiver_id])
                        mad_z_score = 0.6745 * abs_dev / receiver_mad[receiver_id]
                    else:
                        mad_z_score = 0
                    
                    receiver_mad_zscores.append(mad_z_score)
                
                result_df['receiver_amount_mad_zscore'] = receiver_mad_zscores
                result_df['receiver_amount_mad_outlier'] = (np.array(receiver_mad_zscores) > 3.5).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting MAD features: {str(e)}")
            return df.copy()
    
    def _extract_percentile_features(self, df):
        """
        Extract percentile-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with percentile features
        """
        try:
            result_df = df.copy()
            
            # Apply to amount column if it exists
            if 'amount' in df.columns:
                amounts = df['amount']
                
                # Calculate percentiles
                percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(amounts, percentiles)
                
                # Add features for each percentile
                for i, p in enumerate(percentiles):
                    result_df[f'amount_above_{p}th_percentile'] = (amounts > percentile_values[i]).astype(int)
                
                # Calculate percentile rank for each amount
                result_df['amount_percentile_rank'] = amounts.rank(pct=True)
            
            # Apply percentiles to sender and receiver if ID columns exist
            if 'sender_id' in df.columns and 'amount' in df.columns:
                # Calculate percentile rank within each sender's transactions
                sender_percentile_ranks = df.groupby('sender_id')['amount'].rank(pct=True)
                result_df['sender_amount_percentile_rank'] = sender_percentile_ranks
                
                # Flag if amount is in top 5% for sender
                result_df['sender_amount_top_5pct'] = (sender_percentile_ranks > 0.95).astype(int)
            
            if 'receiver_id' in df.columns and 'amount' in df.columns:
                # Calculate percentile rank within each receiver's transactions
                receiver_percentile_ranks = df.groupby('receiver_id')['amount'].rank(pct=True)
                result_df['receiver_amount_percentile_rank'] = receiver_percentile_ranks
                
                # Flag if amount is in top 5% for receiver
                result_df['receiver_amount_top_5pct'] = (receiver_percentile_ranks > 0.95).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting percentile features: {str(e)}")
            return df.copy()
    
    def _extract_distribution_features(self, df):
        """
        Extract distribution-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with distribution features
        """
        try:
            result_df = df.copy()
            
            # Apply to amount column if it exists
            if 'amount' in df.columns:
                amounts = df['amount']
                
                # Skewness and kurtosis
                result_df['amount_skewness'] = stats.skew(amounts)
                result_df['amount_kurtosis'] = stats.kurtosis(amounts)
                
                # Normality tests
                _, normality_p = stats.normaltest(amounts)
                result_df['amount_normality_p'] = normality_p
                
                # Shapiro-Wilk test (for smaller samples)
                if len(amounts) <= 5000:
                    _, shapiro_p = stats.shapiro(amounts)
                    result_df['amount_shapiro_p'] = shapiro_p
                
                # Kolmogorov-Smirnov test against normal distribution
                _, ks_p = stats.kstest(amounts, 'norm', args=(amounts.mean(), amounts.std()))
                result_df['amount_ks_p'] = ks_p
                
                # Anderson-Darling test
                ad_result = stats.anderson(amounts)
                result_df['amount_ad_statistic'] = ad_result.statistic
                
                # Jarque-Bera test
                jb_stat, jb_p = stats.jarque_bera(amounts)
                result_df['amount_jb_statistic'] = jb_stat
                result_df['amount_jb_p'] = jb_p
            
            # Apply distribution features to sender and receiver if ID columns exist
            if 'sender_id' in df.columns and 'amount' in df.columns:
                # Calculate distribution features per sender
                sender_stats = df.groupby('sender_id')['amount'].agg([
                    ('skewness', lambda x: stats.skew(x) if len(x) >= 3 else 0),
                    ('kurtosis', lambda x: stats.kurtosis(x) if len(x) >= 4 else 0),
                    ('variance', 'var'),
                    ('range', lambda x: x.max() - x.min() if len(x) > 0 else 0)
                ])
                
                # Map back to each transaction
                for stat in ['skewness', 'kurtosis', 'variance', 'range']:
                    result_df[f'sender_amount_{stat}'] = df['sender_id'].map(sender_stats[stat]).fillna(0)
            
            if 'receiver_id' in df.columns and 'amount' in df.columns:
                # Calculate distribution features per receiver
                receiver_stats = df.groupby('receiver_id')['amount'].agg([
                    ('skewness', lambda x: stats.skew(x) if len(x) >= 3 else 0),
                    ('kurtosis', lambda x: stats.kurtosis(x) if len(x) >= 4 else 0),
                    ('variance', 'var'),
                    ('range', lambda x: x.max() - x.min() if len(x) > 0 else 0)
                ])
                
                # Map back to each transaction
                for stat in ['skewness', 'kurtosis', 'variance', 'range']:
                    result_df[f'receiver_amount_{stat}'] = df['receiver_id'].map(receiver_stats[stat]).fillna(0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting distribution features: {str(e)}")
            return df.copy()
    
    def _extract_mahalanobis_features(self, df):
        """
        Extract Mahalanobis distance features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with Mahalanobis features
        """
        try:
            result_df = df.copy()
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                logger.warning("Not enough numeric columns for Mahalanobis distance calculation")
                return result_df.copy()
            
            # Prepare data
            X = df[numeric_cols].fillna(0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(X, rowvar=False)
            
            # Check if covariance matrix is invertible
            if np.linalg.det(cov_matrix) == 0:
                logger.warning("Covariance matrix is singular, using pseudo-inverse")
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
            else:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Calculate mean vector
            mean_vector = np.mean(X, axis=0)
            
            # Calculate Mahalanobis distances
            mahalanobis_distances = []
            for i in range(len(X)):
                mahalanobis_distances.append(
                    mahalanobis(X.iloc[i], mean_vector, inv_cov_matrix)
                )
            
            result_df['mahalanobis_distance'] = mahalanobis_distances
            
            # Flag outliers based on Mahalanobis distance
            # Using chi-square distribution with degrees of freedom = number of features
            threshold = stats.chi2.ppf(0.975, df=len(numeric_cols))
            result_df['mahalanobis_outlier'] = (np.array(mahalanobis_distances) > threshold).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting Mahalanobis features: {str(e)}")
            return df.copy()
    
    def _extract_grubbs_features(self, df):
        """
        Extract Grubbs' test features for outliers
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with Grubbs' test features
        """
        try:
            result_df = df.copy()
            
            # Apply to amount column if it exists
            if 'amount' in df.columns:
                amounts = df['amount'].values
                
                # Calculate Grubbs' test statistic
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                
                if std_amount > 0:
                    # Calculate absolute deviations
                    abs_deviations = np.abs(amounts - mean_amount)
                    max_deviation = np.max(abs_deviations)
                    
                    # Grubbs' test statistic
                    grubbs_stat = max_deviation / std_amount
                    result_df['grubbs_statistic'] = grubbs_stat
                    
                    # Calculate critical value for two-sided test
                    n = len(amounts)
                    t_critical = stats.t.ppf(1 - 0.025 / (2 * n), n - 2)
                    critical_value = (n - 1) * t_critical / np.sqrt(n * (n - 2 + t_critical**2))
                    
                    # Flag outliers
                    result_df['grubbs_outlier'] = (grubbs_stat > critical_value).astype(int)
                else:
                    result_df['grubbs_statistic'] = 0
                    result_df['grubbs_outlier'] = 0
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting Grubbs' test features: {str(e)}")
            return df.copy()
    
    def _extract_entropy_features(self, df):
        """
        Extract entropy-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with entropy features
        """
        try:
            result_df = df.copy()
            
            # Calculate entropy for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            for col in categorical_cols:
                # Calculate probability distribution
                value_counts = df[col].value_counts(normalize=True)
                
                # Calculate Shannon entropy
                entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
                result_df[f'{col}_entropy'] = entropy
                
                # Calculate normalized entropy (0 to 1)
                max_entropy = np.log2(len(value_counts))
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                else:
                    normalized_entropy = 0
                result_df[f'{col}_normalized_entropy'] = normalized_entropy
            
            # Calculate entropy for numeric columns (after binning)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                # Bin the data
                try:
                    binned = pd.cut(df[col], bins=10, duplicates='drop')
                    
                    # Calculate probability distribution
                    value_counts = binned.value_counts(normalize=True)
                    
                    # Calculate Shannon entropy
                    entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
                    result_df[f'{col}_binned_entropy'] = entropy
                    
                    # Calculate normalized entropy (0 to 1)
                    max_entropy = np.log2(len(value_counts))
                    if max_entropy > 0:
                        normalized_entropy = entropy / max_entropy
                    else:
                        normalized_entropy = 0
                    result_df[f'{col}_binned_normalized_entropy'] = normalized_entropy
                except:
                    pass
            
            # Calculate transaction entropy for sender and receiver
            if 'sender_id' in df.columns:
                # Calculate entropy of transaction amounts per sender
                sender_entropy = df.groupby('sender_id')['amount'].apply(
                    lambda x: self._calculate_series_entropy(x)
                )
                result_df['sender_amount_entropy'] = df['sender_id'].map(sender_entropy).fillna(0)
            
            if 'receiver_id' in df.columns:
                # Calculate entropy of transaction amounts per receiver
                receiver_entropy = df.groupby('receiver_id')['amount'].apply(
                    lambda x: self._calculate_series_entropy(x)
                )
                result_df['receiver_amount_entropy'] = df['receiver_id'].map(receiver_entropy).fillna(0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting entropy features: {str(e)}")
            return df.copy()
    
    def _calculate_series_entropy(self, series):
        """
        Calculate entropy of a pandas Series
        
        Args:
            series (Series): Input series
            
        Returns:
            float: Entropy value
        """
        try:
            # Bin the data
            binned = pd.cut(series, bins=min(10, len(series)), duplicates='drop')
            
            # Calculate probability distribution
            value_counts = binned.value_counts(normalize=True)
            
            # Calculate Shannon entropy
            entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
            return entropy
        except:
            return 0
    
    def _extract_correlation_features(self, df):
        """
        Extract correlation-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with correlation features
        """
        try:
            result_df = df.copy()
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                logger.warning("Not enough numeric columns for correlation calculation")
                return result_df.copy()
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Find highest correlation for each variable
            max_corr = {}
            for col in numeric_cols:
                # Get correlations with other variables
                corrs = corr_matrix[col].drop(col)
                max_corr[col] = corrs.max()
            
            # Add features
            for col in numeric_cols:
                result_df[f'{col}_max_correlation'] = max_corr[col]
            
            # Calculate average correlation
            avg_corr = {}
            for col in numeric_cols:
                # Get correlations with other variables
                corrs = corr_matrix[col].drop(col)
                avg_corr[col] = corrs.mean()
            
            # Add features
            for col in numeric_cols:
                result_df[f'{col}_avg_correlation'] = avg_corr[col]
            
            # Calculate variance inflation factor (VIF) for multicollinearity
            for i, col in enumerate(numeric_cols):
                # Prepare data for VIF calculation
                X = df[numeric_cols].copy()
                y = X[col]
                X = X.drop(col, axis=1)
                
                # Fit linear regression
                try:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R-squared
                    r_squared = model.score(X, y)
                    
                    # Calculate VIF
                    if r_squared < 1:
                        vif = 1 / (1 - r_squared)
                    else:
                        vif = float('inf')
                    
                    result_df[f'{col}_vif'] = vif
                except:
                    result_df[f'{col}_vif'] = 1
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting correlation features: {str(e)}")
            return df.copy()
    
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
            # Fit scaler
            self.scaler.fit(result_df[feature_cols])
            
            # Apply PCA if we have enough features
            if len(feature_cols) >= 10:
                self.pca = PCA(n_components=min(10, len(feature_cols)))
                self.pca.fit(result_df[feature_cols])
                
                # Add PCA components
                pca_components = self.pca.transform(result_df[feature_cols])
                for i in range(pca_components.shape[1]):
                    result_df[f'stat_pca_{i+1}'] = pca_components[:, i]
            
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
        
        # Get feature columns
        feature_cols = [col for col in result_df.columns if col not in df.columns]
        
        if len(feature_cols) > 0:
            # Transform features using fitted scaler
            result_df[feature_cols] = self.scaler.transform(result_df[feature_cols])
            
            # Apply PCA if it was fitted
            if self.pca is not None:
                pca_components = self.pca.transform(result_df[feature_cols])
                for i in range(pca_components.shape[1]):
                    result_df[f'stat_pca_{i+1}'] = pca_components[:, i]
        
        return result_df
    
    def _serialize_for_cache(self):
        """
        Serialize statistical features state for caching
        
        Returns:
            dict: Serialized state
        """
        try:
            state = {
                'config': self.config,
                'feature_names': self.feature_names,
                'fitted': self.fitted
            }
            
            # Handle scaler
            if self.scaler is not None:
                state['scaler'] = self.scaler
            
            # Handle PCA
            if self.pca is not None:
                state['pca'] = self.pca
            
            return state
            
        except Exception as e:
            logger.error(f"Error serializing statistical features state: {str(e)}")
            return None
    
    def _deserialize_from_cache(self, state):
        """
        Deserialize statistical features state from cache
        
        Args:
            state (dict): Serialized state
            
        Returns:
            bool: True if successful
        """
        try:
            self.config = state.get('config', {})
            self.feature_names = state.get('feature_names', [])
            self.fitted = state.get('fitted', False)
            
            # Handle scaler
            if 'scaler' in state:
                self.scaler = state['scaler']
            else:
                self.scaler = StandardScaler()
            
            # Handle PCA
            if 'pca' in state:
                self.pca = state['pca']
            else:
                self.pca = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error deserializing statistical features state: {str(e)}")
            return False