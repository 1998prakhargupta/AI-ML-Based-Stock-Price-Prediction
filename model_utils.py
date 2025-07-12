"""
Machine learning utilities for stock prediction models.
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plotting functions will be disabled")

# Constants
PLOTTING_WARNING = "Plotting not available - matplotlib/seaborn not installed"

from app_config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDataProcessor:
    """Data processing utilities for machine learning models."""
    
    def __init__(self):
        self.config = Config()
        self.scalers = {}
        self.feature_names = []
        
    def prepare_features(self, data, target_col='Close', feature_selection=True):
        """
        Prepare features for machine learning models.
        
        Args:
            data (pd.DataFrame): Raw data
            target_col (str): Target column name
            feature_selection (bool): Whether to perform feature selection
            
        Returns:
            tuple: (X, y, feature_names)
        """
        logger.info("Preparing features for ML models")
        
        # Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        # Handle missing values
        numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill')
        
        # Separate features and target
        if target_col not in numeric_data.columns:
            available_targets = [col for col in numeric_data.columns if 'close' in col.lower()]
            if available_targets:
                target_col = available_targets[0]
                logger.info(f"Using {target_col} as target column")
            else:
                raise ValueError("No suitable target column found")
        
        features_data = numeric_data.drop(columns=[target_col])
        target_data = numeric_data[target_col]
        
        # Feature selection if requested
        if feature_selection and len(features_data.columns) > 100:
            features_data = self._select_features(features_data, target_data)
        
        self.feature_names = features_data.columns.tolist()
        logger.info(f"Prepared {len(features_data.columns)} features for training")
        
        return features_data, target_data, self.feature_names
    
    def _select_features(self, features_df, target_series, max_features=100):
        """Select top features based on correlation with target."""
        logger.info(f"Selecting top {max_features} features")
        
        # Calculate correlation with target
        correlations = features_df.corrwith(target_series).abs().sort_values(ascending=False)
        
        # Select top features
        top_features = correlations.head(max_features).index.tolist()
        
        logger.info(f"Selected {len(top_features)} features")
        return features_df[top_features]
    
    def scale_features(self, train_features, test_features=None, method='standard'):
        """
        Scale features using specified method.
        
        Args:
            train_features (pd.DataFrame): Training features
            test_features (pd.DataFrame): Test features (optional)
            method (str): Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            tuple: (train_features_scaled, test_features_scaled)
        """
        logger.info(f"Scaling features using {method} scaler")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        train_features_scaled = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index
        )
        
        self.scalers[method] = scaler
        
        test_features_scaled = None
        if test_features is not None:
            test_features_scaled = pd.DataFrame(
                scaler.transform(test_features),
                columns=test_features.columns,
                index=test_features.index
            )
        
        return train_features_scaled, test_features_scaled
    
    def create_sequences(self, data, sequence_length=60, target_col='Close'):
        """
        Create sequences for time series models (LSTM, GRU).
        
        Args:
            data (pd.DataFrame): Input data
            sequence_length (int): Length of input sequences
            target_col (str): Target column name
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Prepare features
        features_data, target_data, _ = self.prepare_features(data, target_col)
        
        # Scale features
        features_scaled, _ = self.scale_features(features_data, method='minmax')
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(features_scaled)):
            sequences.append(features_scaled.iloc[i-sequence_length:i].values)
            targets.append(target_data.iloc[i])
        
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        logger.info(f"Created {len(sequences_array)} sequences")
        return sequences_array, targets_array
    
    def split_time_series(self, features_data, target_data, test_size=0.2, validation_size=0.2):
        """
        Split time series data maintaining temporal order.
        
        Args:
            features_data (pd.DataFrame): Features
            target_data (pd.Series): Target
            test_size (float): Proportion for test set
            validation_size (float): Proportion for validation set
            
        Returns:
            tuple: (train_features, val_features, test_features, train_target, val_target, test_target)
        """
        n_samples = len(features_data)
        
        # Calculate split indices
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))
        
        # Split data
        train_features = features_data.iloc[:val_start]
        val_features = features_data.iloc[val_start:test_start]
        test_features = features_data.iloc[test_start:]
        
        train_target = target_data.iloc[:val_start]
        val_target = target_data.iloc[val_start:test_start]
        test_target = target_data.iloc[test_start:]
        
        logger.info(f"Data split - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
        
        return train_features, val_features, test_features, train_target, val_target, test_target

class ModelEvaluator:
    """Model evaluation utilities."""
    
    def __init__(self):
        self.config = Config()
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Max_Error': np.max(np.abs(y_true - y_pred))
        }
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, title="Predictions vs Actual", save_path=None):
        """
        Create comprehensive prediction plots.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            title (str): Plot title
            save_path (str): Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning(PLOTTING_WARNING)
            return
            
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0, 0].plot(y_true, label='Actual', alpha=0.8, linewidth=1.5)
        axes[0, 0].plot(y_pred, label='Predicted', alpha=0.8, linewidth=1.5)
        axes[0, 0].set_title(title)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=10)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1, 0].plot(residuals, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Residuals')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list): List of feature names
            top_n (int): Number of top features to plot
            save_path (str): Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning(PLOTTING_WARNING)
            return
            
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, results_dict, save_path=None):
        """
        Compare multiple model results.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics as values
            save_path (str): Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning(PLOTTING_WARNING)
            return
            
        metrics_df = pd.DataFrame(results_dict).T
        
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MSE comparison
        axes[0, 0].bar(metrics_df.index, metrics_df['MSE'])
        axes[0, 0].set_title('Mean Squared Error')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R2 comparison
        axes[0, 1].bar(metrics_df.index, metrics_df['R2'])
        axes[0, 1].set_title('R² Score')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(metrics_df.index, metrics_df['MAE'])
        axes[1, 0].set_title('Mean Absolute Error')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(metrics_df.index, metrics_df['MAPE'])
        axes[1, 1].set_title('Mean Absolute Percentage Error')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()

class ModelManager:
    """Model saving and loading utilities."""
    
    def __init__(self):
        self.config = Config()
        self.model_save_path = self._initialize_model_path()
    
    def _initialize_model_path(self) -> str:
        """
        Initialize and create model save directory.
        
        Returns:
            str: Model save path
        """
        try:
            model_path = self.config.get_model_save_path()
            os.makedirs(model_path, exist_ok=True)
            logger.info(f"Model save path initialized: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to initialize model path: {e}")
            # Fallback to current directory
            fallback_path = os.path.join(os.getcwd(), "models")
            os.makedirs(fallback_path, exist_ok=True)
            logger.warning(f"Using fallback model path: {fallback_path}")
            return fallback_path
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model object
            model_name (str): Name for the model
            metadata (dict): Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.model_save_path, filename)
        
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            dict: Model data dictionary
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info(f"Model loaded: {filepath}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def list_saved_models(self):
        """
        List all saved models.
        
        Returns:
            list: List of model file paths
        """
        model_files = [f for f in os.listdir(self.model_save_path) if f.endswith('.pkl')]
        return [os.path.join(self.model_save_path, f) for f in model_files]

# Utility functions
def create_time_features(data, datetime_col=None):
    """
    Create time-based features from datetime index or column.
    
    Args:
        data (pd.DataFrame): Input data
        datetime_col (str): Name of datetime column (if not using index)
        
    Returns:
        pd.DataFrame: Data with added time features
    """
    df = data.copy()
    
    if datetime_col:
        dt = pd.to_datetime(df[datetime_col])
    else:
        dt = df.index
    
    df['hour'] = dt.hour
    df['day_of_week'] = dt.dayofweek
    df['day_of_month'] = dt.day
    df['month'] = dt.month
    df['quarter'] = dt.quarter
    df['is_weekend'] = dt.dayofweek.isin([5, 6]).astype(int)
    df['is_month_end'] = dt.is_month_end.astype(int)
    df['is_month_start'] = dt.is_month_start.astype(int)
    
    return df

def calculate_technical_indicators(data, price_col='Close'):
    """
    Calculate basic technical indicators.
    
    Args:
        data (pd.DataFrame): OHLCV data
        price_col (str): Price column to use
        
    Returns:
        pd.DataFrame: Data with added technical indicators
    """
    df = data.copy()
    
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'SMA_{window}'] = df[price_col].rolling(window=window).mean()
        df[f'EMA_{window}'] = df[price_col].ewm(span=window).mean()
    
    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # MACD
    ema_12 = df[price_col].ewm(span=12).mean()
    ema_26 = df[price_col].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    return df

# Additional data validation utilities for ML models

def validate_data_for_ml(data, target_column='Close', min_samples=100):
    """
    Comprehensive data validation for machine learning.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Name of target column
        min_samples (int): Minimum required samples
        
    Returns:
        dict: Validation results with status and recommendations
    """
    validation_results = {
        'status': 'valid',
        'warnings': [],
        'errors': [],
        'recommendations': [],
        'data_quality_score': 0,
        'metadata': {}
    }
    
    try:
        # Basic structure validation
        if data is None or data.empty:
            validation_results['status'] = 'error'
            validation_results['errors'].append("Data is None or empty")
            return validation_results
        
        # Sample size validation
        if len(data) < min_samples:
            validation_results['warnings'].append(f"Dataset has only {len(data)} samples, minimum recommended: {min_samples}")
        
        # Target column validation
        if target_column not in data.columns:
            close_cols = [col for col in data.columns if 'close' in col.lower()]
            if close_cols:
                validation_results['recommendations'].append(f"Target '{target_column}' not found, consider using: {close_cols[0]}")
            else:
                validation_results['errors'].append(f"Target column '{target_column}' not found and no suitable alternative")
        else:
            # Target quality validation
            target_null_ratio = data[target_column].isna().sum() / len(data)
            if target_null_ratio > 0.1:
                validation_results['warnings'].append(f"Target column has {target_null_ratio*100:.1f}% missing values")
            
            # Target variance validation
            if data[target_column].var() == 0:
                validation_results['errors'].append("Target column has zero variance")
        
        # Feature validation
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        
        if len(numeric_features) == 0:
            validation_results['errors'].append("No numeric features found")
        else:
            # Check for constant features
            constant_features = []
            for col in numeric_features:
                if data[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                validation_results['recommendations'].append(f"Remove {len(constant_features)} constant features: {constant_features[:5]}")
            
            # Check for high missing value features
            high_missing_features = []
            for col in numeric_features:
                missing_ratio = data[col].isna().sum() / len(data)
                if missing_ratio > 0.5:
                    high_missing_features.append((col, missing_ratio))
            
            if high_missing_features:
                validation_results['warnings'].append(f"{len(high_missing_features)} features have >50% missing values")
        
        # Data quality score calculation
        total_cells = len(data) * len(data.columns)
        non_null_cells = data.notna().sum().sum()
        data_quality_score = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        validation_results['data_quality_score'] = data_quality_score
        validation_results['metadata'] = {
            'total_samples': len(data),
            'total_features': len(data.columns),
            'numeric_features': len(numeric_features),
            'data_quality_score': data_quality_score,
            'date_range': (data.index.min(), data.index.max()) if hasattr(data.index, 'min') else ('N/A', 'N/A')
        }
        
        # Overall status determination
        if validation_results['errors']:
            validation_results['status'] = 'error'
        elif validation_results['warnings']:
            validation_results['status'] = 'warning'
        
        logger.info(f"Data validation completed: {validation_results['status']}")
        
    except Exception as e:
        validation_results['status'] = 'error'
        validation_results['errors'].append(f"Validation failed: {str(e)}")
        logger.error(f"Data validation error: {str(e)}")
    
    return validation_results

def safe_train_test_split(features_data, target_data, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Safely split data with validation and error handling.
    
    Args:
        features_data (pd.DataFrame): Features
        target_data (pd.Series): Target
        test_size (float): Test set proportion
        validation_size (float): Validation set proportion
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: Split datasets or None if split fails
    """
    try:
        # Validate inputs
        if len(features_data) != len(target_data):
            raise ValueError(f"Features ({len(features_data)}) and target ({len(target_data)}) length mismatch")
        
        if len(features_data) < 10:
            logger.warning(f"Very small dataset for splitting: {len(features_data)} samples")
        
        # Calculate split indices for time series (no shuffle)
        n_samples = len(features_data)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))
        
        # Ensure minimum samples in each split
        min_samples_per_split = 5
        if val_start < min_samples_per_split or (test_start - val_start) < min_samples_per_split or (n_samples - test_start) < min_samples_per_split:
            logger.warning("Insufficient data for proper train/val/test split, using simple train/test split")
            # Simple train/test split
            split_idx = int(n_samples * 0.8)
            train_features = features_data.iloc[:split_idx]
            test_features = features_data.iloc[split_idx:]
            train_target = target_data.iloc[:split_idx]
            test_target = target_data.iloc[split_idx:]
            
            logger.info(f"Simple split: Train {len(train_features)}, Test {len(test_features)}")
            return train_features, None, test_features, train_target, None, test_target
        
        # Standard three-way split
        train_features = features_data.iloc[:val_start]
        val_features = features_data.iloc[val_start:test_start]
        test_features = features_data.iloc[test_start:]
        
        train_target = target_data.iloc[:val_start]
        val_target = target_data.iloc[val_start:test_start]
        test_target = target_data.iloc[test_start:]
        
        logger.info(f"Data split completed - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
        
        return train_features, val_features, test_features, train_target, val_target, test_target
        
    except Exception as e:
        logger.error(f"Data splitting failed: {str(e)}")
        return None

def basic_ml_data_validation(data, target_column='Close'):
    """
    Basic data validation for machine learning.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Name of target column
        
    Returns:
        dict: Simple validation results
    """
    results = {'valid': True, 'messages': []}
    
    if data is None or data.empty:
        results['valid'] = False
        results['messages'].append("Data is empty")
        return results
    
    if target_column not in data.columns:
        results['valid'] = False
        results['messages'].append(f"Target column '{target_column}' not found")
    
    if len(data) < 50:
        results['messages'].append(f"Small dataset: {len(data)} samples")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        results['messages'].append("Few numeric columns available")
    
    logger.info(f"Basic validation: {'✅ Valid' if results['valid'] else '❌ Invalid'}")
    return results

def safe_feature_preparation(data, target_col='Close', max_features=100):
    """
    Safely prepare features with validation.
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target column
        max_features (int): Maximum number of features
        
    Returns:
        tuple: (features, target) or (None, None) if preparation fails
    """
    try:
        # Basic validation
        validation = basic_ml_data_validation(data, target_col)
        if not validation['valid']:
            logger.error(f"Data validation failed: {validation['messages']}")
            return None, None
        
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Limit features if too many
        if len(feature_cols) > max_features:
            # Simple feature selection based on correlation with target
            correlations = data[feature_cols].corrwith(data[target_col]).abs()
            top_features = correlations.nlargest(max_features).index.tolist()
            feature_cols = top_features
        
        # Prepare final data
        features = data[feature_cols].copy()
        target = data[target_col].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        target = target.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with remaining NaN
        valid_mask = features.notna().all(axis=1) & target.notna()
        features = features[valid_mask]
        target = target[valid_mask]
        
        logger.info(f"Feature preparation completed: {features.shape}, target: {len(target)}")
        return features, target
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {str(e)}")
        return None, None
