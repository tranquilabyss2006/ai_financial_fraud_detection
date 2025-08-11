"""
Supervised Models Module
Implements supervised learning models for fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from typing import Dict, List, Tuple, Union
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisedModels:
    """
    Class for supervised fraud detection models
    Implements Random Forest, XGBoost, Logistic Regression, etc.
    """
    
    def __init__(self, test_size=0.2, random_state=42, handle_imbalance=True):
        """
        Initialize SupervisedModels
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed
            handle_imbalance (bool): Whether to handle class imbalance
        """
        self.test_size = test_size
        self.random_state = random_state
        self.handle_imbalance = handle_imbalance
        self.models = {}
        self.feature_names = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        self.resamplers = {}
        self.performance = {}
        self.feature_importance = {}
        self.shap_values = {}
        self.fitted = False
        
    def run_models(self, df):
        """
        Run all supervised models
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            dict: Model results
        """
        try:
            # Check if fraud_flag column exists
            if 'fraud_flag' not in df.columns:
                logger.warning("No fraud_flag column found. Using synthetic labels for demonstration.")
                # Create synthetic labels based on statistical outliers
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    # Use Z-score to identify outliers as potential fraud
                    X = df[numeric_cols].fillna(0)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Calculate Z-scores
                    z_scores = np.abs(X_scaled).max(axis=1)
                    
                    # Label top 5% as fraud
                    threshold = np.percentile(z_scores, 95)
                    y = (z_scores > threshold).astype(int)
                    
                    # Add synthetic labels to dataframe
                    df = df.copy()
                    df['fraud_flag'] = y
                else:
                    raise ValueError("No numeric columns found for creating synthetic labels")
            else:
                y = df['fraud_flag'].astype(int)
            
            # Get feature columns (exclude target and ID columns)
            exclude_cols = ['fraud_flag', 'transaction_id', 'sender_id', 'receiver_id']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Get numeric and categorical features
            numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Prepare data
            X = df[feature_cols].copy()
            
            # Clean data before processing
            X = self._clean_data(X)
            
            # Handle categorical features
            for col in categorical_features:
                if col in X.columns:
                    # Label encode categorical features
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Scale numeric features using RobustScaler
            if numeric_features:
                scaler = RobustScaler()
                X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
                X_test[numeric_features] = scaler.transform(X_test[numeric_features])
                self.scalers['numeric'] = scaler
            
            # Handle class imbalance
            if self.handle_imbalance and y_train.sum() / len(y_train) < 0.1:  # If minority class < 10%
                # Try different resampling techniques
                resampling_methods = {
                    'smote': SMOTE(random_state=self.random_state),
                    'adasyn': ADASYN(random_state=self.random_state),
                    'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
                    'smote_enn': SMOTEENN(random_state=self.random_state),
                    'smote_tomek': SMOTETomek(random_state=self.random_state)
                }
                
                # Evaluate each method using cross-validation
                best_method = None
                best_score = 0
                
                for method_name, resampler in resampling_methods.items():
                    try:
                        # Apply resampling
                        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
                        
                        # Evaluate with a simple model
                        rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
                        scores = cross_val_score(rf, X_resampled, y_resampled, cv=3, scoring='f1')
                        avg_score = scores.mean()
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_method = method_name
                            self.resamplers['best'] = resampler
                            X_train, y_train = X_resampled, y_resampled
                    except Exception as e:
                        logger.warning(f"Error with {method_name}: {str(e)}")
                
                if best_method:
                    logger.info(f"Using {best_method} for handling class imbalance")
                else:
                    logger.warning("No suitable resampling method found")
            
            # Store feature names
            all_feature_names = feature_cols
            
            # Run different models
            results = {}
            
            # Random Forest
            try:
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_proba = rf_model.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                rf_performance = self._calculate_performance_metrics(y_test, rf_pred, rf_proba)
                
                # Get feature importance
                rf_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(rf_model)
                rf_shap_values = explainer.shap_values(X_test)
                
                results['random_forest'] = {
                    'model': rf_model,
                    'predictions': rf_pred,
                    'probabilities': rf_proba,
                    'performance': rf_performance,
                    'feature_importance': rf_importance,
                    'shap_values': rf_shap_values,
                    'feature_names': all_feature_names
                }
                
                self.models['random_forest'] = rf_model
                self.feature_names['random_forest'] = all_feature_names
                self.performance['random_forest'] = rf_performance
                self.feature_importance['random_forest'] = rf_importance
                self.shap_values['random_forest'] = rf_shap_values
                
                logger.info("Random Forest model completed")
            except Exception as e:
                logger.error(f"Error running Random Forest: {str(e)}")
            
            # XGBoost
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_test)
                xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                xgb_performance = self._calculate_performance_metrics(y_test, xgb_pred, xgb_proba)
                
                # Get feature importance
                xgb_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                xgb_shap_values = explainer.shap_values(X_test)
                
                results['xgboost'] = {
                    'model': xgb_model,
                    'predictions': xgb_pred,
                    'probabilities': xgb_proba,
                    'performance': xgb_performance,
                    'feature_importance': xgb_importance,
                    'shap_values': xgb_shap_values,
                    'feature_names': all_feature_names
                }
                
                self.models['xgboost'] = xgb_model
                self.feature_names['xgboost'] = all_feature_names
                self.performance['xgboost'] = xgb_performance
                self.feature_importance['xgboost'] = xgb_importance
                self.shap_values['xgboost'] = xgb_shap_values
                
                logger.info("XGBoost model completed")
            except Exception as e:
                logger.error(f"Error running XGBoost: {str(e)}")
            
            # LightGBM
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                    random_state=self.random_state
                )
                
                lgb_model.fit(X_train, y_train)
                lgb_pred = lgb_model.predict(X_test)
                lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                lgb_performance = self._calculate_performance_metrics(y_test, lgb_pred, lgb_proba)
                
                # Get feature importance
                lgb_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': lgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(lgb_model)
                lgb_shap_values = explainer.shap_values(X_test)
                
                results['lightgbm'] = {
                    'model': lgb_model,
                    'predictions': lgb_pred,
                    'probabilities': lgb_proba,
                    'performance': lgb_performance,
                    'feature_importance': lgb_importance,
                    'shap_values': lgb_shap_values,
                    'feature_names': all_feature_names
                }
                
                self.models['lightgbm'] = lgb_model
                self.feature_names['lightgbm'] = all_feature_names
                self.performance['lightgbm'] = lgb_performance
                self.feature_importance['lightgbm'] = lgb_importance
                self.shap_values['lightgbm'] = lgb_shap_values
                
                logger.info("LightGBM model completed")
            except Exception as e:
                logger.error(f"Error running LightGBM: {str(e)}")
            
            # Logistic Regression
            try:
                lr_model = LogisticRegression(
                    C=1.0,
                    class_weight='balanced',
                    random_state=self.random_state,
                    max_iter=1000
                )
                
                lr_model.fit(X_train, y_train)
                lr_pred = lr_model.predict(X_test)
                lr_proba = lr_model.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                lr_performance = self._calculate_performance_metrics(y_test, lr_pred, lr_proba)
                
                # Get feature importance (coefficients)
                lr_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': np.abs(lr_model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP values
                explainer = shap.LinearExplainer(lr_model, X_train)
                lr_shap_values = explainer.shap_values(X_test)
                
                results['logistic_regression'] = {
                    'model': lr_model,
                    'predictions': lr_pred,
                    'probabilities': lr_proba,
                    'performance': lr_performance,
                    'feature_importance': lr_importance,
                    'shap_values': lr_shap_values,
                    'feature_names': all_feature_names
                }
                
                self.models['logistic_regression'] = lr_model
                self.feature_names['logistic_regression'] = all_feature_names
                self.performance['logistic_regression'] = lr_performance
                self.feature_importance['logistic_regression'] = lr_importance
                self.shap_values['logistic_regression'] = lr_shap_values
                
                logger.info("Logistic Regression model completed")
            except Exception as e:
                logger.error(f"Error running Logistic Regression: {str(e)}")
            
            # Gradient Boosting
            try:
                gb_model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.random_state
                )
                
                gb_model.fit(X_train, y_train)
                gb_pred = gb_model.predict(X_test)
                gb_proba = gb_model.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                gb_performance = self._calculate_performance_metrics(y_test, gb_pred, gb_proba)
                
                # Get feature importance
                gb_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': gb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(gb_model)
                gb_shap_values = explainer.shap_values(X_test)
                
                results['gradient_boosting'] = {
                    'model': gb_model,
                    'predictions': gb_pred,
                    'probabilities': gb_proba,
                    'performance': gb_performance,
                    'feature_importance': gb_importance,
                    'shap_values': gb_shap_values,
                    'feature_names': all_feature_names
                }
                
                self.models['gradient_boosting'] = gb_model
                self.feature_names['gradient_boosting'] = all_feature_names
                self.performance['gradient_boosting'] = gb_performance
                self.feature_importance['gradient_boosting'] = gb_importance
                self.shap_values['gradient_boosting'] = gb_shap_values
                
                logger.info("Gradient Boosting model completed")
            except Exception as e:
                logger.error(f"Error running Gradient Boosting: {str(e)}")
            
            # Neural Network
            try:
                nn_model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size=32,
                    learning_rate='adaptive',
                    max_iter=200,
                    random_state=self.random_state
                )
                
                nn_model.fit(X_train, y_train)
                nn_pred = nn_model.predict(X_test)
                nn_proba = nn_model.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                nn_performance = self._calculate_performance_metrics(y_test, nn_pred, nn_proba)
                
                # For neural networks, we don't have direct feature importance
                # Use permutation importance instead
                from sklearn.inspection import permutation_importance
                
                perm_importance = permutation_importance(
                    nn_model, X_test, y_test, n_repeats=5, random_state=self.random_state
                )
                
                nn_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': perm_importance.importances_mean
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP values (using KernelExplainer for black-box models)
                explainer = shap.KernelExplainer(nn_model.predict_proba, X_train[:100])  # Use subset for speed
                nn_shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed
                
                results['neural_network'] = {
                    'model': nn_model,
                    'predictions': nn_pred,
                    'probabilities': nn_proba,
                    'performance': nn_performance,
                    'feature_importance': nn_importance,
                    'shap_values': nn_shap_values,
                    'feature_names': all_feature_names
                }
                
                self.models['neural_network'] = nn_model
                self.feature_names['neural_network'] = all_feature_names
                self.performance['neural_network'] = nn_performance
                self.feature_importance['neural_network'] = nn_importance
                self.shap_values['neural_network'] = nn_shap_values
                
                logger.info("Neural Network model completed")
            except Exception as e:
                logger.error(f"Error running Neural Network: {str(e)}")
            
            # Ensemble model (combine predictions from all models)
            try:
                # Get predictions from all models
                all_predictions = []
                all_probabilities = []
                
                for model_name in results:
                    all_predictions.append(results[model_name]['predictions'])
                    all_probabilities.append(results[model_name]['probabilities'])
                
                # Majority voting for predictions
                ensemble_pred = np.array(all_predictions).mean(axis=0) > 0.5
                
                # Average probabilities
                ensemble_proba = np.array(all_probabilities).mean(axis=0)
                
                # Calculate performance metrics
                ensemble_performance = self._calculate_performance_metrics(y_test, ensemble_pred, ensemble_proba)
                
                # For ensemble, average feature importances
                ensemble_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': np.zeros(len(all_feature_names))
                })
                
                for model_name in results:
                    if model_name in self.feature_importance:
                        model_importance = self.feature_importance[model_name]
                        for i, feature in enumerate(all_feature_names):
                            if feature in model_importance['feature'].values:
                                idx = model_importance[model_importance['feature'] == feature].index[0]
                                ensemble_importance.loc[i, 'importance'] += model_importance.loc[idx, 'importance']
                
                # Normalize importance
                ensemble_importance['importance'] = ensemble_importance['importance'] / len(results)
                ensemble_importance = ensemble_importance.sort_values('importance', ascending=False)
                
                results['ensemble'] = {
                    'predictions': ensemble_pred,
                    'probabilities': ensemble_proba,
                    'performance': ensemble_performance,
                    'feature_importance': ensemble_importance,
                    'feature_names': all_feature_names
                }
                
                self.performance['ensemble'] = ensemble_performance
                self.feature_importance['ensemble'] = ensemble_importance
                
                logger.info("Ensemble model completed")
            except Exception as e:
                logger.error(f"Error running Ensemble model: {str(e)}")
            
            self.fitted = True
            return results
            
        except Exception as e:
            logger.error(f"Error running supervised models: {str(e)}")
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
    
    def _calculate_performance_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate performance metrics for a model
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_proba (array): Predicted probabilities
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # ROC AUC
            roc_auc = roc_auc_score(y_true, y_proba)
            
            # Average precision
            avg_precision = average_precision_score(y_true, y_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'confusion_matrix': cm,
                'classification_report': class_report,
                'precision_curve': precision_curve,
                'recall_curve': recall_curve,
                'fpr': fpr,
                'tpr': tpr
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def predict(self, df, model_name):
        """
        Make predictions using a fitted model
        
        Args:
            df (DataFrame): Input data
            model_name (str): Name of the model to use
            
        Returns:
            dict: Predictions and probabilities
        """
        if not self.fitted:
            raise ValueError("Models not fitted. Call run_models first.")
        
        if model_name not in self.models and model_name != 'ensemble':
            raise ValueError(f"Model {model_name} not found.")
        
        try:
            # Get feature names for this model
            if model_name == 'ensemble':
                # For ensemble, use all feature names
                feature_names = []
                for name in self.feature_names:
                    feature_names.extend(self.feature_names[name])
                feature_names = list(set(feature_names))  # Remove duplicates
            else:
                feature_names = self.feature_names[model_name]
            
            # Prepare data
            X = df[feature_names].copy()
            
            # Clean the data
            X = self._clean_data(X)
            
            # Handle categorical features
            for col in X.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X[col] = X[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            
            # Scale numeric features if scaler exists
            if 'numeric' in self.scalers:
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_features:
                    X[numeric_features] = self.scalers['numeric'].transform(X[numeric_features])
            
            # Make predictions
            if model_name == 'ensemble':
                # Get predictions from all models
                all_predictions = []
                all_probabilities = []
                
                for name in self.models:
                    model = self.models[name]
                    pred = model.predict(X)
                    proba = model.predict_proba(X)[:, 1]
                    all_predictions.append(pred)
                    all_probabilities.append(proba)
                
                # Majority voting for predictions
                predictions = np.array(all_predictions).mean(axis=0) > 0.5
                
                # Average probabilities
                probabilities = np.array(all_probabilities).mean(axis=0)
                
            else:
                # Get model
                model = self.models[model_name]
                
                # Make predictions
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities
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
        
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance for {model_name} not found.")
        
        return self.feature_importance[model_name]
    
    def get_performance(self, model_name):
        """
        Get performance metrics for a model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Performance metrics
        """
        if not self.fitted:
            raise ValueError("Models not fitted. Call run_models first.")
        
        if model_name not in self.performance:
            raise ValueError(f"Performance for {model_name} not found.")
        
        return self.performance[model_name]
    
    def get_shap_values(self, model_name):
        """
        Get SHAP values for a model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            array: SHAP values
        """
        if not self.fitted:
            raise ValueError("Models not fitted. Call run_models first.")
        
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values for {model_name} not found.")
        
        return self.shap_values[model_name]
    
    def plot_feature_importance(self, model_name, top_n=20):
        """
        Plot feature importance for a model
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to show
        """
        try:
            # Get feature importance
            importance_df = self.get_feature_importance(model_name)
            
            # Get top features
            top_features = importance_df.head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance for {model_name}: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, model_name):
        """
        Plot confusion matrix for a model
        
        Args:
            model_name (str): Name of the model
        """
        try:
            # Get performance metrics
            performance = self.get_performance(model_name)
            cm = performance['confusion_matrix']
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Fraud', 'Fraud'], 
                       yticklabels=['Not Fraud', 'Fraud'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix for {model_name}: {str(e)}")
            raise
    
    def plot_roc_curve(self, model_name):
        """
        Plot ROC curve for a model
        
        Args:
            model_name (str): Name of the model
        """
        try:
            # Get performance metrics
            performance = self.get_performance(model_name)
            fpr = performance['fpr']
            tpr = performance['tpr']
            roc_auc = performance['roc_auc']
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve for {model_name}: {str(e)}")
            raise
    
    def plot_precision_recall_curve(self, model_name):
        """
        Plot precision-recall curve for a model
        
        Args:
            model_name (str): Name of the model
        """
        try:
            # Get performance metrics
            performance = self.get_performance(model_name)
            precision_curve = performance['precision_curve']
            recall_curve = performance['recall_curve']
            avg_precision = performance['avg_precision']
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, label=f'{model_name} (AP = {avg_precision:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curve for {model_name}: {str(e)}")
            raise