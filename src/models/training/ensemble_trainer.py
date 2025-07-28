"""
Enterprise Ensemble Model Trainer
=================================

Comprehensive ensemble training system with all major ML/DL models for stock price prediction.
Includes Random Forest, Gradient Boosting, XGBoost, LightGBM, Bi-LSTM, GRU, Transformer,
Ridge, Lasso, Elastic Net, SVR, ARIMA, Prophet, and seasonal decomposition models.

Features:
- Multi-model ensemble training
- Advanced model selection and validation
- Cost-aware optimization
- Real-time model adaptation
- Enterprise-grade error handling
- Comprehensive logging and monitoring
"""

import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pickle
import joblib
from pathlib import Path
import yaml
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention, LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention, TransformerBlock
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Deep learning models will be skipped.")

# Gradient Boosting Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    XGB_AVAILABLE = True
    LGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    LGB_AVAILABLE = False
    print("Warning: XGBoost/LightGBM not available.")

# Time Series Libraries
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: Statsmodels not available. ARIMA models will be skipped.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available.")

# Project imports
from ..evaluation.cost_evaluator import CostEvaluator
from ...utils.app_config import Config
from ...utils.file_management_utils import SafeFileManager

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    enabled: bool = True
    cost_weight: float = 0.1
    
@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    models: List[ModelConfig]
    ensemble_method: str = "weighted_average"  # weighted_average, stacking, voting
    validation_split: float = 0.2
    time_series_cv_splits: int = 5
    cost_optimization: bool = True
    save_individual_models: bool = True
    model_selection_metric: str = "r2_score"

@dataclass
class TrainingResult:
    """Training result container."""
    model_name: str
    model: Any
    performance_metrics: Dict[str, float]
    training_time: float
    validation_score: float
    cost_efficiency: float
    feature_importance: Optional[Dict[str, float]] = None

class TransformerLayer(tf.keras.layers.Layer):
    """Custom Transformer layer for time series prediction."""
    
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class EnterpriseEnsembleTrainer:
    """
    Enterprise-grade ensemble trainer for stock price prediction.
    
    Supports all major ML/DL models:
    - Traditional ML: Random Forest, Gradient Boosting, SVR, Linear models
    - Deep Learning: Bi-LSTM, GRU, Transformer
    - Gradient Boosting: XGBoost, LightGBM
    - Time Series: ARIMA, Prophet, Seasonal Decomposition
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ensemble trainer."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.trained_models = {}
        self.ensemble_model = None
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        self.cost_evaluator = CostEvaluator()
        self.file_manager = SafeFileManager()
        
        # Create directories
        self.model_save_dir = Path("models/experiments")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Enterprise Ensemble Trainer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> EnsembleConfig:
        """Load training configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return self._dict_to_ensemble_config(config_dict)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> EnsembleConfig:
        """Get default ensemble configuration."""
        models = [
            # Traditional ML Models
            ModelConfig(
                name="RandomForest",
                model_type="sklearn",
                hyperparameters={
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                    "n_jobs": -1
                }
            ),
            ModelConfig(
                name="GradientBoosting",
                model_type="sklearn",
                hyperparameters={
                    "n_estimators": 150,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "random_state": 42
                }
            ),
            ModelConfig(
                name="Ridge",
                model_type="sklearn",
                hyperparameters={
                    "alpha": 1.0,
                    "random_state": 42
                }
            ),
            ModelConfig(
                name="Lasso",
                model_type="sklearn",
                hyperparameters={
                    "alpha": 0.1,
                    "random_state": 42
                }
            ),
            ModelConfig(
                name="ElasticNet",
                model_type="sklearn",
                hyperparameters={
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 42
                }
            ),
            ModelConfig(
                name="SVR_RBF",
                model_type="sklearn",
                hyperparameters={
                    "kernel": "rbf",
                    "C": 1.0,
                    "gamma": "scale"
                }
            ),
            ModelConfig(
                name="SVR_Linear",
                model_type="sklearn",
                hyperparameters={
                    "kernel": "linear",
                    "C": 1.0
                }
            ),
            ModelConfig(
                name="SVR_Poly",
                model_type="sklearn",
                hyperparameters={
                    "kernel": "poly",
                    "degree": 3,
                    "C": 1.0,
                    "gamma": "scale"
                }
            )
        ]
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models.append(ModelConfig(
                name="XGBoost",
                model_type="xgboost",
                hyperparameters={
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1
                }
            ))
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models.append(ModelConfig(
                name="LightGBM",
                model_type="lightgbm",
                hyperparameters={
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1
                }
            ))
        
        # Add Deep Learning models if TensorFlow is available
        if TF_AVAILABLE:
            models.extend([
                ModelConfig(
                    name="BiLSTM",
                    model_type="tensorflow",
                    hyperparameters={
                        "lstm_units": 128,
                        "dropout": 0.2,
                        "epochs": 100,
                        "batch_size": 32,
                        "sequence_length": 60
                    }
                ),
                ModelConfig(
                    name="GRU",
                    model_type="tensorflow",
                    hyperparameters={
                        "gru_units": 128,
                        "dropout": 0.2,
                        "epochs": 100,
                        "batch_size": 32,
                        "sequence_length": 60
                    }
                ),
                ModelConfig(
                    name="Transformer",
                    model_type="tensorflow",
                    hyperparameters={
                        "d_model": 128,
                        "num_heads": 8,
                        "dff": 512,
                        "num_layers": 4,
                        "dropout": 0.1,
                        "epochs": 100,
                        "batch_size": 32,
                        "sequence_length": 60
                    }
                )
            ])
        
        # Add Time Series models if available
        if STATSMODELS_AVAILABLE:
            models.append(ModelConfig(
                name="ARIMA",
                model_type="arima",
                hyperparameters={
                    "order": (5, 1, 0),
                    "seasonal_order": (1, 1, 1, 12)
                }
            ))
        
        if PROPHET_AVAILABLE:
            models.append(ModelConfig(
                name="Prophet",
                model_type="prophet",
                hyperparameters={
                    "daily_seasonality": True,
                    "weekly_seasonality": True,
                    "yearly_seasonality": True,
                    "changepoint_prior_scale": 0.05
                }
            ))
        
        return EnsembleConfig(
            models=models,
            ensemble_method="weighted_average",
            validation_split=0.2,
            time_series_cv_splits=5,
            cost_optimization=True,
            save_individual_models=True
        )
    
    def train_ensemble_models(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        cost_features: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all ensemble models and create final ensemble.
        
        Args:
            data: Training data with features and target
            target_column: Name of target column
            feature_columns: List of feature columns (if None, uses all except target)
            cost_features: Cost-related feature columns
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary containing trained models and performance metrics
        """
        logger.info("Starting ensemble model training...")
        
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].fillna(0)
        y = data[target_column].fillna(method='ffill')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train individual models
        training_results = []
        model_predictions = {}
        
        for model_config in self.config.models:
            if not model_config.enabled:
                continue
                
            try:
                logger.info(f"Training {model_config.name}...")
                result = self._train_individual_model(
                    model_config, X_train, X_test, y_train, y_test,
                    X_train_scaled, X_test_scaled, cost_features
                )
                training_results.append(result)
                model_predictions[model_config.name] = result.model.predict(
                    X_test_scaled if model_config.model_type in ['sklearn', 'xgboost', 'lightgbm'] 
                    else X_test
                )
                logger.info(f"{model_config.name} training completed. R² Score: {result.validation_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_config.name}: {str(e)}")
                continue
        
        # Create ensemble
        ensemble_predictions = self._create_ensemble(model_predictions, training_results, y_test)
        
        # Evaluate ensemble
        ensemble_metrics = self._calculate_metrics(y_test, ensemble_predictions)
        
        # Store results
        best_model = max(training_results, key=lambda x: x.validation_score)
        
        results = {
            'individual_models': {result.model_name: result for result in training_results},
            'ensemble_predictions': ensemble_predictions,
            'ensemble_metrics': ensemble_metrics,
            'best_individual_model': best_model,
            'feature_importance': self._calculate_ensemble_feature_importance(training_results),
            'training_summary': self._create_training_summary(training_results, ensemble_metrics)
        }
        
        # Save models if configured
        if self.config.save_individual_models:
            self._save_trained_models(results)
        
        logger.info("Ensemble training completed successfully!")
        return results
    
    def _train_individual_model(
        self,
        model_config: ModelConfig,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        cost_features: Optional[List[str]]
    ) -> TrainingResult:
        """Train an individual model."""
        start_time = datetime.now()
        
        if model_config.model_type == "sklearn":
            model = self._create_sklearn_model(model_config)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
        elif model_config.model_type == "xgboost" and XGB_AVAILABLE:
            model = xgb.XGBRegressor(**model_config.hyperparameters)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
        elif model_config.model_type == "lightgbm" and LGB_AVAILABLE:
            model = lgb.LGBMRegressor(**model_config.hyperparameters)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
        elif model_config.model_type == "tensorflow" and TF_AVAILABLE:
            model, predictions = self._train_tensorflow_model(
                model_config, X_train_scaled, X_test_scaled, y_train, y_test
            )
            
        elif model_config.model_type == "arima" and STATSMODELS_AVAILABLE:
            model, predictions = self._train_arima_model(
                model_config, y_train, len(y_test)
            )
            
        elif model_config.model_type == "prophet" and PROPHET_AVAILABLE:
            model, predictions = self._train_prophet_model(
                model_config, y_train, len(y_test)
            )
            
        else:
            raise ValueError(f"Model type {model_config.model_type} not supported or dependencies not available")
        
        # Calculate metrics
        training_time = (datetime.now() - start_time).total_seconds()
        metrics = self._calculate_metrics(y_test, predictions)
        
        # Calculate cost efficiency if cost features available
        cost_efficiency = 1.0
        if cost_features and self.config.cost_optimization:
            cost_efficiency = self._calculate_cost_efficiency(
                model, X_test, y_test, predictions, cost_features
            )
        
        # Get feature importance if available
        feature_importance = self._get_feature_importance(model, X_train.columns)
        
        return TrainingResult(
            model_name=model_config.name,
            model=model,
            performance_metrics=metrics,
            training_time=training_time,
            validation_score=metrics['r2_score'],
            cost_efficiency=cost_efficiency,
            feature_importance=feature_importance
        )
    
    def _create_sklearn_model(self, model_config: ModelConfig):
        """Create sklearn model based on configuration."""
        model_map = {
            "RandomForest": RandomForestRegressor,
            "GradientBoosting": GradientBoostingRegressor,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "SVR_RBF": lambda **kwargs: SVR(kernel='rbf', **{k: v for k, v in kwargs.items() if k != 'kernel'}),
            "SVR_Linear": lambda **kwargs: SVR(kernel='linear', **{k: v for k, v in kwargs.items() if k != 'kernel'}),
            "SVR_Poly": lambda **kwargs: SVR(kernel='poly', **{k: v for k, v in kwargs.items() if k != 'kernel'})
        }
        
        model_class = model_map.get(model_config.name)
        if model_class is None:
            raise ValueError(f"Unknown sklearn model: {model_config.name}")
        
        return model_class(**model_config.hyperparameters)
    
    def _train_tensorflow_model(
        self,
        model_config: ModelConfig,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[Any, np.ndarray]:
        """Train TensorFlow/Keras model."""
        sequence_length = model_config.hyperparameters.get('sequence_length', 60)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train.values, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test.values, sequence_length)
        
        # Build model
        if model_config.name == "BiLSTM":
            model = self._build_bilstm_model(X_train_seq.shape, model_config.hyperparameters)
        elif model_config.name == "GRU":
            model = self._build_gru_model(X_train_seq.shape, model_config.hyperparameters)
        elif model_config.name == "Transformer":
            model = self._build_transformer_model(X_train_seq.shape, model_config.hyperparameters)
        else:
            raise ValueError(f"Unknown TensorFlow model: {model_config.name}")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=model_config.hyperparameters.get('epochs', 100),
            batch_size=model_config.hyperparameters.get('batch_size', 32),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Make predictions
        predictions = model.predict(X_test_seq).flatten()
        
        return model, predictions
    
    def _build_bilstm_model(self, input_shape: Tuple, hyperparams: Dict) -> Model:
        """Build Bidirectional LSTM model."""
        model = Sequential([
            Input(shape=(input_shape[1], input_shape[2])),
            tf.keras.layers.Bidirectional(LSTM(
                hyperparams.get('lstm_units', 128),
                return_sequences=True,
                dropout=hyperparams.get('dropout', 0.2)
            )),
            tf.keras.layers.Bidirectional(LSTM(
                hyperparams.get('lstm_units', 128) // 2,
                dropout=hyperparams.get('dropout', 0.2)
            )),
            Dropout(hyperparams.get('dropout', 0.2)),
            Dense(64, activation='relu'),
            Dropout(hyperparams.get('dropout', 0.2)),
            Dense(1)
        ])
        return model
    
    def _build_gru_model(self, input_shape: Tuple, hyperparams: Dict) -> Model:
        """Build GRU model."""
        model = Sequential([
            Input(shape=(input_shape[1], input_shape[2])),
            GRU(
                hyperparams.get('gru_units', 128),
                return_sequences=True,
                dropout=hyperparams.get('dropout', 0.2)
            ),
            GRU(
                hyperparams.get('gru_units', 128) // 2,
                dropout=hyperparams.get('dropout', 0.2)
            ),
            Dropout(hyperparams.get('dropout', 0.2)),
            Dense(64, activation='relu'),
            Dropout(hyperparams.get('dropout', 0.2)),
            Dense(1)
        ])
        return model
    
    def _build_transformer_model(self, input_shape: Tuple, hyperparams: Dict) -> Model:
        """Build Transformer model."""
        inputs = Input(shape=(input_shape[1], input_shape[2]))
        
        # Embedding layer
        x = Dense(hyperparams.get('d_model', 128))(inputs)
        
        # Transformer layers
        for _ in range(hyperparams.get('num_layers', 4)):
            x = TransformerLayer(
                d_model=hyperparams.get('d_model', 128),
                num_heads=hyperparams.get('num_heads', 8),
                dff=hyperparams.get('dff', 512),
                rate=hyperparams.get('dropout', 0.1)
            )(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(hyperparams.get('dropout', 0.1))(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _train_arima_model(
        self,
        model_config: ModelConfig,
        y_train: pd.Series,
        forecast_steps: int
    ) -> Tuple[Any, np.ndarray]:
        """Train ARIMA model."""
        model = ARIMA(
            y_train,
            order=model_config.hyperparameters.get('order', (5, 1, 0))
        )
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        return fitted_model, forecast
    
    def _train_prophet_model(
        self,
        model_config: ModelConfig,
        y_train: pd.Series,
        forecast_steps: int
    ) -> Tuple[Any, np.ndarray]:
        """Train Prophet model."""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(y_train), freq='D'),
            'y': y_train.values
        })
        
        # Create and fit model
        model = Prophet(**model_config.hyperparameters)
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_steps)
        
        # Generate forecasts
        forecast = model.predict(future)
        predictions = forecast['yhat'].tail(forecast_steps).values
        
        return model, predictions
    
    def _create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def _create_ensemble(
        self,
        model_predictions: Dict[str, np.ndarray],
        training_results: List[TrainingResult],
        y_test: pd.Series
    ) -> np.ndarray:
        """Create ensemble predictions."""
        if self.config.ensemble_method == "weighted_average":
            return self._weighted_average_ensemble(model_predictions, training_results)
        elif self.config.ensemble_method == "voting":
            return self._voting_ensemble(model_predictions)
        elif self.config.ensemble_method == "stacking":
            return self._stacking_ensemble(model_predictions, y_test)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")
    
    def _weighted_average_ensemble(
        self,
        model_predictions: Dict[str, np.ndarray],
        training_results: List[TrainingResult]
    ) -> np.ndarray:
        """Create weighted average ensemble based on model performance."""
        # Calculate weights based on validation scores and cost efficiency
        weights = {}
        total_weight = 0
        
        for result in training_results:
            if result.model_name in model_predictions:
                weight = result.validation_score * result.cost_efficiency
                weights[result.model_name] = weight
                total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        # Calculate weighted predictions
        ensemble_pred = np.zeros_like(list(model_predictions.values())[0])
        
        for model_name, predictions in model_predictions.items():
            if model_name in weights:
                ensemble_pred += weights[model_name] * predictions
        
        return ensemble_pred
    
    def _voting_ensemble(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create simple voting ensemble (equal weights)."""
        predictions_array = np.array(list(model_predictions.values()))
        return np.mean(predictions_array, axis=0)
    
    def _stacking_ensemble(
        self,
        model_predictions: Dict[str, np.ndarray],
        y_test: pd.Series
    ) -> np.ndarray:
        """Create stacking ensemble with meta-learner."""
        # Use Ridge regression as meta-learner
        meta_learner = Ridge(alpha=1.0)
        
        # Prepare meta-features (predictions from base models)
        meta_features = np.column_stack(list(model_predictions.values()))
        
        # Train meta-learner
        meta_learner.fit(meta_features, y_test)
        
        # Generate ensemble predictions
        ensemble_predictions = meta_learner.predict(meta_features)
        
        return ensemble_predictions
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def _calculate_cost_efficiency(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
        cost_features: List[str]
    ) -> float:
        """Calculate cost efficiency score."""
        try:
            # Use cost evaluator to calculate cost-adjusted performance
            evaluation_result = self.cost_evaluator.evaluate_with_costs(
                y_true=y_test.values,
                y_pred=predictions,
                X_features=X_test,
                cost_features=cost_features
            )
            
            return evaluation_result.get('cost_adjusted_score', 1.0)
        except Exception as e:
            logger.warning(f"Could not calculate cost efficiency: {e}")
            return 1.0
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(zip(feature_names, np.abs(model.coef_)))
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def _calculate_ensemble_feature_importance(
        self,
        training_results: List[TrainingResult]
    ) -> Dict[str, float]:
        """Calculate ensemble feature importance."""
        ensemble_importance = {}
        total_models = 0
        
        for result in training_results:
            if result.feature_importance:
                total_models += 1
                for feature, importance in result.feature_importance.items():
                    if feature not in ensemble_importance:
                        ensemble_importance[feature] = 0
                    ensemble_importance[feature] += importance
        
        # Average importance across models
        if total_models > 0:
            for feature in ensemble_importance:
                ensemble_importance[feature] /= total_models
        
        return ensemble_importance
    
    def _create_training_summary(
        self,
        training_results: List[TrainingResult],
        ensemble_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create comprehensive training summary."""
        return {
            'total_models_trained': len(training_results),
            'best_individual_model': max(training_results, key=lambda x: x.validation_score).model_name,
            'ensemble_performance': ensemble_metrics,
            'individual_model_performance': {
                result.model_name: {
                    'r2_score': result.validation_score,
                    'training_time': result.training_time,
                    'cost_efficiency': result.cost_efficiency
                }
                for result in training_results
            },
            'training_timestamp': datetime.now().isoformat()
        }
    
    def _save_trained_models(self, results: Dict[str, Any]):
        """Save trained models to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.model_save_dir / f"ensemble_{timestamp}"
            save_dir.mkdir(exist_ok=True)
            
            # Save individual models
            for model_name, result in results['individual_models'].items():
                model_path = save_dir / f"{model_name}.pkl"
                joblib.dump(result.model, model_path)
            
            # Save ensemble metadata
            metadata = {
                'ensemble_metrics': results['ensemble_metrics'],
                'feature_importance': results['feature_importance'],
                'training_summary': results['training_summary']
            }
            
            metadata_path = save_dir / "ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Models saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble."""
        if not self.trained_models:
            raise ValueError("No trained models available. Train the ensemble first.")
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.trained_models.items():
            try:
                if hasattr(model, 'predict'):
                    predictions[model_name] = model.predict(X_scaled)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
        
        # Create ensemble prediction
        if predictions:
            return self._voting_ensemble(predictions)
        else:
            raise ValueError("No valid predictions available")
    
    def _dict_to_ensemble_config(self, config_dict: Dict[str, Any]) -> EnsembleConfig:
        """Convert dictionary to EnsembleConfig object."""
        models = []
        for model_data in config_dict.get('models', []):
            models.append(ModelConfig(**model_data))
        
        return EnsembleConfig(
            models=models,
            ensemble_method=config_dict.get('ensemble_method', 'weighted_average'),
            validation_split=config_dict.get('validation_split', 0.2),
            time_series_cv_splits=config_dict.get('time_series_cv_splits', 5),
            cost_optimization=config_dict.get('cost_optimization', True),
            save_individual_models=config_dict.get('save_individual_models', True)
        )

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = EnterpriseEnsembleTrainer()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'feature_1': np.random.randn(1000).cumsum(),
        'feature_2': np.random.randn(1000).cumsum(),
        'feature_3': np.random.randn(1000).cumsum(),
        'target': np.random.randn(1000).cumsum()
    }, index=dates)
    
    # Train ensemble
    results = trainer.train_ensemble_models(
        data=data,
        target_column='target',
        test_size=0.2
    )
    
    print("Ensemble training completed!")
    print(f"Best model: {results['best_individual_model'].model_name}")
    print(f"Ensemble R² Score: {results['ensemble_metrics']['r2_score']:.4f}")
