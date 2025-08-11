"""
Unsupervised Models Module
Implements unsupervised learning models for fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import logging
from typing import Dict, List, Tuple, Union
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnsupervisedModels:
    """
    Class for unsupervised fraud detection models
    Implements Isolation Forest, Local Outlier Factor, Autoencoders, etc.
    """
    
    def __init__(self, contamination=0.01, random_state=42):
        """
        Initialize UnsupervisedModels
        
        Args:
            contamination (float): Expected proportion of outliers
            random_state (int): Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.feature_names = {}
        self.scalers = {}
        self.imputers = {}
        self.fitted = False
        
    def run_models(self, df):
        """
        Run all unsupervised models
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            dict: Model results
        """
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                logger.warning("Not enough numeric columns for unsupervised models")
                return {}
            
            # Prepare data
            X = df[numeric_cols].copy()
            
            # Clean the data - handle infinity and very large values
            X = self._clean_data(X)
            
            # Scale data
            X_scaled, scaler, imputer = self._scale_data(X)
            
            # Store scaler and imputer
            self.scalers['global'] = scaler
            self.imputers['global'] = imputer
            
            # Run different models
            results = {}
            
            # Isolation Forest
            try:
                if_model = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_samples='auto'
                )
                if_predictions = if_model.fit_predict(X_scaled)
                if_scores = if_model.decision_function(X_scaled)
                
                # Normalize scores to 0-1 range
                if_scores_normalized = self._normalize_scores(if_scores)
                
                results['isolation_forest'] = {
                    'predictions': if_predictions,
                    'scores': if_scores_normalized,
                    'model': if_model,
                    'feature_names': numeric_cols
                }
                
                self.models['isolation_forest'] = if_model
                self.feature_names['isolation_forest'] = numeric_cols
                
                logger.info("Isolation Forest model completed")
            except Exception as e:
                logger.error(f"Error running Isolation Forest: {str(e)}")
            
            # Local Outlier Factor
            try:
                lof_model = LocalOutlierFactor(
                    contamination=self.contamination,
                    n_neighbors=min(20, len(X_scaled) - 1),  # Ensure n_neighbors < n_samples
                    novelty=True,
                    n_jobs=-1
                )
                lof_predictions = lof_model.fit_predict(X_scaled)
                lof_scores = lof_model.decision_function(X_scaled)
                
                # Normalize scores to 0-1 range
                lof_scores_normalized = self._normalize_scores(lof_scores)
                
                results['local_outlier_factor'] = {
                    'predictions': lof_predictions,
                    'scores': lof_scores_normalized,
                    'model': lof_model,
                    'feature_names': numeric_cols
                }
                
                self.models['local_outlier_factor'] = lof_model
                self.feature_names['local_outlier_factor'] = numeric_cols
                
                logger.info("Local Outlier Factor model completed")
            except Exception as e:
                logger.error(f"Error running Local Outlier Factor: {str(e)}")
            
            # One-Class SVM
            try:
                ocsvm_model = OneClassSVM(
                    nu=self.contamination,
                    kernel='rbf',
                    gamma='scale'
                )
                ocsvm_predictions = ocsvm_model.fit_predict(X_scaled)
                ocsvm_scores = ocsvm_model.decision_function(X_scaled)
                
                # Normalize scores to 0-1 range
                ocsvm_scores_normalized = self._normalize_scores(ocsvm_scores)
                
                results['one_class_svm'] = {
                    'predictions': ocsvm_predictions,
                    'scores': ocsvm_scores_normalized,
                    'model': ocsvm_model,
                    'feature_names': numeric_cols
                }
                
                self.models['one_class_svm'] = ocsvm_model
                self.feature_names['one_class_svm'] = numeric_cols
                
                logger.info("One-Class SVM model completed")
            except Exception as e:
                logger.error(f"Error running One-Class SVM: {str(e)}")
            
            # DBSCAN Clustering
            try:
                # Use a subset of data for DBSCAN if it's too large
                if len(X_scaled) > 10000:
                    X_subset = X_scaled[np.random.choice(len(X_scaled), 10000, replace=False)]
                else:
                    X_subset = X_scaled
                
                dbscan_model = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan_model.fit_predict(X_subset)
                
                # Convert to outlier predictions (-1 is outlier in DBSCAN)
                dbscan_predictions = np.where(dbscan_labels == -1, -1, 1)
                
                # Calculate distance to nearest core point as anomaly score
                dbscan_scores = np.zeros(len(X_subset))
                for i in range(len(X_subset)):
                    if dbscan_labels[i] == -1:  # Outlier
                        # Find distance to nearest core point
                        core_points = np.where(dbscan_labels != -1)[0]
                        if len(core_points) > 0:
                            distances = np.linalg.norm(X_subset[i] - X_subset[core_points], axis=1)
                            dbscan_scores[i] = distances.min()
                        else:
                            dbscan_scores[i] = np.max(np.linalg.norm(X_subset - X_subset.mean(axis=0), axis=1))
                
                # Normalize scores to 0-1 range
                dbscan_scores_normalized = self._normalize_scores(dbscan_scores)
                
                # For the full dataset, use the model to predict
                if len(X_scaled) > 10000:
                    full_predictions = dbscan_model.fit_predict(X_scaled)
                    full_predictions = np.where(full_predictions == -1, -1, 1)
                    
                    # Calculate scores for full dataset
                    full_scores = np.zeros(len(X_scaled))
                    for i in range(len(X_scaled)):
                        if full_predictions[i] == -1:  # Outlier
                            # Find distance to nearest core point
                            core_points = np.where(full_predictions != -1)[0]
                            if len(core_points) > 0:
                                distances = np.linalg.norm(X_scaled[i] - X_scaled[core_points], axis=1)
                                full_scores[i] = distances.min()
                            else:
                                full_scores[i] = np.max(np.linalg.norm(X_scaled - X_scaled.mean(axis=0), axis=1))
                    
                    # Normalize scores
                    full_scores_normalized = self._normalize_scores(full_scores)
                else:
                    full_predictions = dbscan_predictions
                    full_scores_normalized = dbscan_scores_normalized
                
                results['dbscan'] = {
                    'predictions': full_predictions,
                    'scores': full_scores_normalized,
                    'model': dbscan_model,
                    'feature_names': numeric_cols
                }
                
                self.models['dbscan'] = dbscan_model
                self.feature_names['dbscan'] = numeric_cols
                
                logger.info("DBSCAN model completed")
            except Exception as e:
                logger.error(f"Error running DBSCAN: {str(e)}")
            
            # K-Means Clustering
            try:
                # Determine optimal number of clusters
                silhouette_scores = []
                k_range = range(2, min(11, len(X_scaled) // 2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)
                    silhouette_scores.append(silhouette_score(X_scaled, labels))
                
                # Find optimal k
                optimal_k = k_range[np.argmax(silhouette_scores)]
                
                # Fit K-Means with optimal k
                kmeans_model = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
                kmeans_labels = kmeans_model.fit_predict(X_scaled)
                
                # Calculate distance to nearest cluster center as anomaly score
                distances = kmeans_model.transform(X_scaled)
                min_distances = distances.min(axis=1)
                
                # Normalize scores to 0-1 range
                kmeans_scores_normalized = self._normalize_scores(min_distances)
                
                # Convert to outlier predictions (top contamination% are outliers)
                threshold = np.percentile(kmeans_scores_normalized, (1 - self.contamination) * 100)
                kmeans_predictions = np.where(kmeans_scores_normalized > threshold, -1, 1)
                
                results['kmeans'] = {
                    'predictions': kmeans_predictions,
                    'scores': kmeans_scores_normalized,
                    'model': kmeans_model,
                    'feature_names': numeric_cols,
                    'optimal_k': optimal_k,
                    'silhouette_scores': silhouette_scores
                }
                
                self.models['kmeans'] = kmeans_model
                self.feature_names['kmeans'] = numeric_cols
                
                logger.info(f"K-Means model completed with {optimal_k} clusters")
            except Exception as e:
                logger.error(f"Error running K-Means: {str(e)}")
            
            # Autoencoder
            try:
                # Build autoencoder model
                input_dim = X_scaled.shape[1]
                encoding_dim = max(2, input_dim // 2)  # At least 2 dimensions
                
                autoencoder = self._build_autoencoder(input_dim, encoding_dim)
                
                # Train autoencoder
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = autoencoder.fit(
                    X_scaled, X_scaled,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Get reconstruction errors
                reconstructions = autoencoder.predict(X_scaled)
                mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                
                # Handle any infinity or NaN values in MSE
                mse = np.nan_to_num(mse)
                mse = np.clip(mse, 0, np.finfo(np.float64).max)  # Clip to valid range
                
                # Normalize scores to 0-1 range
                mse_normalized = self._normalize_scores(mse)
                
                # Convert to outlier predictions (top contamination% are outliers)
                threshold = np.percentile(mse_normalized, (1 - self.contamination) * 100)
                autoencoder_predictions = np.where(mse_normalized > threshold, -1, 1)
                
                results['autoencoder'] = {
                    'predictions': autoencoder_predictions,
                    'scores': mse_normalized,
                    'model': autoencoder,
                    'feature_names': numeric_cols,
                    'history': history.history
                }
                
                self.models['autoencoder'] = autoencoder
                self.feature_names['autoencoder'] = numeric_cols
                
                logger.info("Autoencoder model completed")
            except Exception as e:
                logger.error(f"Error running Autoencoder: {str(e)}")
            
            # PCA-based anomaly detection
            try:
                # Fit PCA
                pca = PCA(n_components=min(10, X_scaled.shape[1] - 1), random_state=self.random_state)
                pca_transformed = pca.fit_transform(X_scaled)
                
                # Calculate reconstruction error
                X_reconstructed = pca.inverse_transform(pca_transformed)
                pca_errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
                
                # Handle any infinity or NaN values in errors
                pca_errors = np.nan_to_num(pca_errors)
                pca_errors = np.clip(pca_errors, 0, np.finfo(np.float64).max)  # Clip to valid range
                
                # Normalize scores to 0-1 range
                pca_errors_normalized = self._normalize_scores(pca_errors)
                
                # Convert to outlier predictions (top contamination% are outliers)
                threshold = np.percentile(pca_errors_normalized, (1 - self.contamination) * 100)
                pca_predictions = np.where(pca_errors_normalized > threshold, -1, 1)
                
                results['pca'] = {
                    'predictions': pca_predictions,
                    'scores': pca_errors_normalized,
                    'model': pca,
                    'feature_names': numeric_cols,
                    'explained_variance': pca.explained_variance_ratio_
                }
                
                self.models['pca'] = pca
                self.feature_names['pca'] = numeric_cols
                
                logger.info("PCA-based anomaly detection completed")
            except Exception as e:
                logger.error(f"Error running PCA-based anomaly detection: {str(e)}")
            
            self.fitted = True
            return results
            
        except Exception as e:
            logger.error(f"Error running unsupervised models: {str(e)}")
            raise
    
    def _clean_data(self, X):
        """
        Clean data by handling infinity, NaN, and extreme values
        
        Args:
            X (DataFrame): Input data
            
        Returns:
            DataFrame: Cleaned data
        """
        try:
            # Replace infinity with NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Replace extremely large values with a more reasonable maximum
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    # Calculate 99th percentile as a reasonable maximum
                    percentile_99 = np.nanpercentile(X[col], 99)
                    if not np.isnan(percentile_99):
                        # Cap values at 10 times the 99th percentile
                        max_val = percentile_99 * 10
                        X[col] = np.where(X[col] > max_val, max_val, X[col])
                        
                        # Similarly, handle extremely negative values
                        percentile_1 = np.nanpercentile(X[col], 1)
                        if not np.isnan(percentile_1):
                            min_val = percentile_1 * 10
                            X[col] = np.where(X[col] < min_val, min_val, X[col])
            
            return X
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return X
    
    def _scale_data(self, X):
        """
        Scale data using RobustScaler (more resistant to outliers)
        
        Args:
            X (DataFrame): Input data
            
        Returns:
            tuple: (scaled data, scaler, imputer)
        """
        try:
            # First impute missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Use RobustScaler instead of StandardScaler to handle outliers better
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Check for any remaining infinity or NaN values
            X_scaled = np.nan_to_num(X_scaled)
            X_scaled = np.clip(X_scaled, -1e10, 1e10)  # Clip to reasonable range
            
            return X_scaled, scaler, imputer
            
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            # Fallback to simple normalization
            X_normalized = (X - X.min()) / (X.max() - X.min())
            return X_normalized.values, None, None
    
    def _normalize_scores(self, scores):
        """
        Normalize scores to 0-1 range
        
        Args:
            scores (array): Input scores
            
        Returns:
            array: Normalized scores
        """
        try:
            # Handle NaN and infinity
            scores = np.nan_to_num(scores)
            scores = np.clip(scores, -1e10, 1e10)  # Clip to reasonable range
            
            # Min-max normalization
            min_score = scores.min()
            max_score = scores.max()
            
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error normalizing scores: {str(e)}")
            return np.zeros_like(scores)
    
    def _build_autoencoder(self, input_dim, encoding_dim):
        """
        Build autoencoder model
        
        Args:
            input_dim (int): Input dimension
            encoding_dim (int): Encoding dimension
            
        Returns:
            Model: Autoencoder model
        """
        try:
            # Encoder
            input_layer = layers.Input(shape=(input_dim,))
            encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
            encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
            
            # Decoder
            decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
            decoded = layers.Dense(input_dim, activation='linear')(decoded)
            
            # Autoencoder model
            autoencoder = models.Model(input_layer, decoded)
            
            # Compile model
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            
            return autoencoder
            
        except Exception as e:
            logger.error(f"Error building autoencoder: {str(e)}")
            raise
    
    def predict(self, df, model_name):
        """
        Make predictions using a fitted model
        
        Args:
            df (DataFrame): Input data
            model_name (str): Name of the model to use
            
        Returns:
            dict: Predictions and scores
        """
        if not self.fitted:
            raise ValueError("Models not fitted. Call run_models first.")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        try:
            # Get feature names for this model
            feature_names = self.feature_names[model_name]
            
            # Prepare data
            X = df[feature_names].copy()
            
            # Clean the data
            X = self._clean_data(X)
            
            # Scale data using fitted scaler and imputer
            if 'global' in self.scalers and 'global' in self.imputers:
                X_imputed = self.imputers['global'].transform(X)
                X_scaled = self.scalers['global'].transform(X_imputed)
            else:
                # Fallback to simple normalization
                X_scaled = (X - X.min()) / (X.max() - X.min())
                X_scaled = np.nan_to_num(X_scaled)
                X_scaled = np.clip(X_scaled, -1e10, 1e10)
            
            # Get model
            model = self.models[model_name]
            
            # Make predictions
            if model_name == 'isolation_forest':
                predictions = model.predict(X_scaled)
                scores = model.decision_function(X_scaled)
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(scores)
                
            elif model_name == 'local_outlier_factor':
                predictions = model.predict(X_scaled)
                scores = model.decision_function(X_scaled)
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(scores)
                
            elif model_name == 'one_class_svm':
                predictions = model.predict(X_scaled)
                scores = model.decision_function(X_scaled)
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(scores)
                
            elif model_name == 'dbscan':
                predictions = model.fit_predict(X_scaled)
                # Convert to outlier predictions (-1 is outlier in DBSCAN)
                predictions = np.where(predictions == -1, -1, 1)
                
                # Calculate distance to nearest core point as anomaly score
                scores = np.zeros(len(X_scaled))
                for i in range(len(X_scaled)):
                    if predictions[i] == -1:  # Outlier
                        # Find distance to nearest core point
                        core_points = np.where(predictions != -1)[0]
                        if len(core_points) > 0:
                            distances = np.linalg.norm(X_scaled[i] - X_scaled[core_points], axis=1)
                            scores[i] = distances.min()
                        else:
                            scores[i] = np.max(np.linalg.norm(X_scaled - X_scaled.mean(axis=0), axis=1))
                
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(scores)
                
            elif model_name == 'kmeans':
                # Transform data
                distances = model.transform(X_scaled)
                min_distances = distances.min(axis=1)
                
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(min_distances)
                
                # Convert to outlier predictions (top contamination% are outliers)
                threshold = np.percentile(scores_normalized, (1 - self.contamination) * 100)
                predictions = np.where(scores_normalized > threshold, -1, 1)
                
            elif model_name == 'autoencoder':
                # Get reconstructions
                reconstructions = model.predict(X_scaled)
                mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                
                # Handle any infinity or NaN values in MSE
                mse = np.nan_to_num(mse)
                mse = np.clip(mse, 0, np.finfo(np.float64).max)
                
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(mse)
                
                # Convert to outlier predictions (top contamination% are outliers)
                threshold = np.percentile(scores_normalized, (1 - self.contamination) * 100)
                predictions = np.where(scores_normalized > threshold, -1, 1)
                
            elif model_name == 'pca':
                # Transform data
                pca_transformed = model.transform(X_scaled)
                
                # Calculate reconstruction error
                X_reconstructed = model.inverse_transform(pca_transformed)
                mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
                
                # Handle any infinity or NaN values in errors
                mse = np.nan_to_num(mse)
                mse = np.clip(mse, 0, np.finfo(np.float64).max)
                
                # Normalize scores to 0-1 range
                scores_normalized = self._normalize_scores(mse)
                
                # Convert to outlier predictions (top contamination% are outliers)
                threshold = np.percentile(scores_normalized, (1 - self.contamination) * 100)
                predictions = np.where(scores_normalized > threshold, -1, 1)
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            return {
                'predictions': predictions,
                'scores': scores_normalized
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {str(e)}")
            raise
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for a model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            DataFrame: Feature importance
        """
        if not self.fitted:
            raise ValueError("Models not fitted. Call run_models first.")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        try:
            # Get feature names
            feature_names = self.feature_names[model_name]
            
            if model_name == 'isolation_forest':
                # Get feature importance from the model
                importance = self.models[model_name].feature_importances_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
            elif model_name == 'pca':
                # Get explained variance ratio
                explained_variance = self.models[model_name].explained_variance_ratio_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': [f'PC{i+1}' for i in range(len(explained_variance))],
                    'importance': explained_variance
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
            else:
                # For other models, return equal importance
                importance = np.ones(len(feature_names)) / len(feature_names)
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
        except Exception as e:
            logger.error(f"Error getting feature importance for {model_name}: {str(e)}")
            raise