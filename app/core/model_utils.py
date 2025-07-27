"""
Model Utilities Module
=====================

Comprehensive model management utilities for the Stock Price Predictor project.
Handles model training, evaluation, and management operations.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine learning dependencies with fallback
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error
    )
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, ML functions will be disabled")

# Import project utilities
from src.utils.app_config import Config
from src.utils.file_management_utils import SafeFileManager, SaveStrategy
from src.utils.reproducibility_utils import ReproducibilityManager

# Setup logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    Provides standardized evaluation metrics and analysis.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.metrics_functions = {
            'MSE': mean_squared_error,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error,
            'R2': r2_score,
            'MAPE': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        }
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model predictions with comprehensive metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for model evaluation")
            return {}
        
        try:
            metrics = {}
            
            for metric_name, metric_func in self.metrics_functions.items():
                try:
                    metrics[metric_name] = float(metric_func(y_true, y_pred))
                except Exception as e:
                    logger.warning(f"Error calculating {metric_name} for {model_name}: {e}")
                    metrics[metric_name] = float('nan')
            
            # Additional custom metrics
            metrics['Max_Error'] = float(np.max(np.abs(y_true - y_pred)))
            metrics['Min_Error'] = float(np.min(np.abs(y_true - y_pred)))
            metrics['Std_Error'] = float(np.std(y_true - y_pred))
            
            # Directional accuracy (for time series)
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                metrics['Directional_Accuracy'] = float(np.mean(true_direction == pred_direction))
            
            logger.info(f"Model {model_name} evaluated with {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}
    
    def compare_models(self, 
                      results_dict: Dict[str, Dict[str, float]],
                      primary_metric: str = 'R2') -> Dict[str, Any]:
        """
        Compare multiple models and provide ranking.
        
        Args:
            results_dict: Dictionary of model results
            primary_metric: Primary metric for ranking
            
        Returns:
            Comparison analysis
        """
        try:
            comparison = {
                'model_count': len(results_dict),
                'primary_metric': primary_metric,
                'rankings': {},
                'best_model': None,
                'worst_model': None,
                'metric_summary': {}
            }
            
            # Extract metrics
            all_metrics = set()
            for results in results_dict.values():
                all_metrics.update(results.keys())
            
            # Calculate rankings for each metric
            for metric in all_metrics:
                metric_values = []
                for model_name, results in results_dict.items():
                    if metric in results:
                        metric_values.append((model_name, results[metric]))
                
                if metric_values:
                    # Higher is better for R2, lower is better for error metrics
                    reverse = metric in ['R2', 'Directional_Accuracy']
                    sorted_models = sorted(metric_values, key=lambda x: x[1], reverse=reverse)
                    comparison['rankings'][metric] = [model[0] for model in sorted_models]
            
            # Determine best/worst models based on primary metric
            if primary_metric in comparison['rankings']:
                comparison['best_model'] = comparison['rankings'][primary_metric][0]
                comparison['worst_model'] = comparison['rankings'][primary_metric][-1]
            
            # Calculate metric summaries
            for metric in all_metrics:
                values = [results.get(metric, np.nan) for results in results_dict.values()]
                valid_values = [v for v in values if not np.isnan(v)]
                
                if valid_values:
                    comparison['metric_summary'][metric] = {
                        'mean': np.mean(valid_values),
                        'std': np.std(valid_values),
                        'min': np.min(valid_values),
                        'max': np.max(valid_values),
                        'range': np.max(valid_values) - np.min(valid_values)
                    }
            
            logger.info(f"Compared {len(results_dict)} models")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def cross_validate_model(self,
                           model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           cv_folds: int = 5,
                           scoring: str = 'r2') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for cross-validation")
            return {}
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            
            cv_results = {
                'cv_mean': float(np.mean(scores)),
                'cv_std': float(np.std(scores)),
                'cv_min': float(np.min(scores)),
                'cv_max': float(np.max(scores)),
                'cv_scores': scores.tolist()
            }
            
            logger.info(f"Cross-validation completed: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}


class ModelManager:
    """
    Comprehensive model management for training, saving, and loading models.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize model manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_models_save_path())
        self.evaluator = ModelEvaluator()
        self.repro_manager = ReproducibilityManager()
        
        # Model registry
        self.model_registry = {
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svr': SVR
        } if SKLEARN_AVAILABLE else {}
        
        logger.info("ModelManager initialized")
    
    def create_model(self, 
                    model_type: str, 
                    **kwargs) -> Optional[Any]:
        """
        Create a model instance with reproducible parameters.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional model parameters
            
        Returns:
            Model instance or None if not available
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for model creation")
            return None
        
        if model_type not in self.model_registry:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        try:
            # Get reproducible parameters
            repro_params = self.repro_manager.get_reproducible_model_params(model_type)
            
            # Merge with user parameters
            final_params = {**repro_params, **kwargs}
            
            # Create model
            model_class = self.model_registry[model_type]
            model = model_class(**final_params)
            
            logger.info(f"Created {model_type} model with parameters: {final_params}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {e}")
            return None
    
    def train_model(self,
                   model: Any,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   model_name: str = "model") -> Dict[str, Any]:
        """
        Train a model with comprehensive tracking.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            model_name: Name for the model
            
        Returns:
            Training results
        """
        try:
            start_time = datetime.now()
            
            # Set seeds for reproducibility
            self.repro_manager.set_all_seeds()
            
            # Train model
            model.fit(X_train, y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Make predictions for evaluation
            y_pred_train = model.predict(X_train)
            
            # Evaluate training performance
            train_metrics = self.evaluator.evaluate_model(y_train, y_pred_train, f"{model_name}_train")
            
            # Prepare results
            training_results = {
                'model': model,
                'model_name': model_name,
                'training_time': training_time,
                'training_samples': len(X_train),
                'feature_count': X_train.shape[1],
                'train_metrics': train_metrics,
                'training_timestamp': start_time.isoformat(),
                'model_type': type(model).__name__
            }
            
            # Add model-specific information
            if hasattr(model, 'feature_importances_'):
                training_results['feature_importances'] = model.feature_importances_.tolist()
            
            if hasattr(model, 'coef_'):
                training_results['coefficients'] = model.coef_.tolist()
            
            logger.info(f"Model {model_name} trained in {training_time:.2f} seconds")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return {}
    
    def evaluate_model_comprehensive(self,
                                   model: Any,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   X_train: Optional[np.ndarray] = None,
                                   y_train: Optional[np.ndarray] = None,
                                   model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test and train sets.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional)
            y_train: Training targets (optional)
            model_name: Name of the model
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            evaluation_results = {
                'model_name': model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_samples': len(X_test),
                'feature_count': X_test.shape[1]
            }
            
            # Test set evaluation
            y_pred_test = model.predict(X_test)
            test_metrics = self.evaluator.evaluate_model(y_test, y_pred_test, f"{model_name}_test")
            evaluation_results['test_metrics'] = test_metrics
            evaluation_results['test_predictions'] = y_pred_test.tolist()
            
            # Training set evaluation (if provided)
            if X_train is not None and y_train is not None:
                y_pred_train = model.predict(X_train)
                train_metrics = self.evaluator.evaluate_model(y_train, y_pred_train, f"{model_name}_train")
                evaluation_results['train_metrics'] = train_metrics
                
                # Calculate overfitting indicators
                evaluation_results['overfitting_analysis'] = self._analyze_overfitting(
                    train_metrics, test_metrics
                )
            
            # Cross-validation (if training data available)
            if X_train is not None and y_train is not None:
                cv_results = self.evaluator.cross_validate_model(
                    model, X_train, y_train, 
                    cv_folds=self.config.get_ml_cv_folds()
                )
                evaluation_results['cross_validation'] = cv_results
            
            logger.info(f"Comprehensive evaluation completed for {model_name}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation for {model_name}: {e}")
            return {}
    
    def save_model(self,
                  model: Any,
                  model_name: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  model_type: str = "production") -> str:
        """
        Save model with metadata and versioning.
        
        Args:
            model: Model to save
            model_name: Name for the model
            metadata: Additional metadata
            model_type: Type of model (production, experimental, checkpoint)
            
        Returns:
            Path to saved model
        """
        try:
            # Determine save path based on model type
            if model_type == "production":
                save_path = self.config.get_models_production_path()
            elif model_type == "experimental":
                save_path = self.config.get_models_experiments_path()
            else:  # checkpoint
                save_path = self.config.get_models_checkpoints_path()
            
            # Use specific file manager for model type
            model_file_manager = SafeFileManager(save_path, SaveStrategy.VERSION)
            
            # Prepare model metadata
            model_metadata = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'save_timestamp': datetime.now().isoformat(),
                'model_category': model_type,
                'scikit_learn_version': '0.24.0',  # Placeholder
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            # Save model
            model_filename = f"{model_name}.pkl"
            save_result = model_file_manager.save_file(
                model, model_filename,
                metadata=model_metadata
            )
            
            if save_result.success:
                logger.info(f"Model {model_name} saved: {save_result.filepath}")
                return save_result.filepath
            else:
                logger.error(f"Failed to save model {model_name}: {save_result.error_message}")
                return ""
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return ""
    
    def load_model(self,
                  model_path: str,
                  model_type: str = "production") -> Tuple[Any, Dict[str, Any]]:
        """
        Load model with metadata.
        
        Args:
            model_path: Path to model file
            model_type: Type of model directory
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            # Determine base path
            if model_type == "production":
                base_path = self.config.get_models_production_path()
            elif model_type == "experimental":
                base_path = self.config.get_models_experiments_path()
            else:  # checkpoint
                base_path = self.config.get_models_checkpoints_path()
            
            model_file_manager = SafeFileManager(base_path)
            
            # Load model and metadata
            model, metadata = model_file_manager.load_file(model_path, 'pickle')
            
            logger.info(f"Model loaded from: {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def create_ensemble_model(self,
                            models: Dict[str, Any],
                            ensemble_method: str = "average") -> 'EnsembleModel':
        """
        Create ensemble model from multiple models.
        
        Args:
            models: Dictionary of models
            ensemble_method: Method for ensemble (average, weighted_average, voting)
            
        Returns:
            Ensemble model instance
        """
        return EnsembleModel(models, ensemble_method)
    
    def _analyze_overfitting(self,
                           train_metrics: Dict[str, float],
                           test_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze overfitting based on train/test performance gap.
        
        Args:
            train_metrics: Training metrics
            test_metrics: Test metrics
            
        Returns:
            Overfitting analysis
        """
        analysis = {
            'overfitting_detected': False,
            'performance_gaps': {},
            'severity': 'none'
        }
        
        try:
            # Calculate performance gaps
            for metric in ['R2', 'MSE', 'MAE', 'RMSE']:
                if metric in train_metrics and metric in test_metrics:
                    train_val = train_metrics[metric]
                    test_val = test_metrics[metric]
                    
                    if metric == 'R2':
                        # For R2, training should be similar or lower than test
                        gap = train_val - test_val
                        analysis['performance_gaps'][metric] = gap
                        if gap > 0.1:  # Significant gap
                            analysis['overfitting_detected'] = True
                    else:
                        # For error metrics, training should be similar or higher than test
                        gap = test_val - train_val
                        analysis['performance_gaps'][metric] = gap
                        if gap > train_val * 0.2:  # 20% worse on test
                            analysis['overfitting_detected'] = True
            
            # Determine severity
            if analysis['overfitting_detected']:
                gaps = list(analysis['performance_gaps'].values())
                max_gap = max(gaps) if gaps else 0
                
                if max_gap > 0.3:
                    analysis['severity'] = 'severe'
                elif max_gap > 0.15:
                    analysis['severity'] = 'moderate'
                else:
                    analysis['severity'] = 'mild'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing overfitting: {e}")
            return analysis


class EnsembleModel:
    """
    Ensemble model combining multiple base models.
    """
    
    def __init__(self, models: Dict[str, Any], method: str = "average"):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of base models
            method: Ensemble method
        """
        self.models = models
        self.method = method
        self.weights = None
        
        logger.info(f"EnsembleModel created with {len(models)} models using {method} method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        try:
            predictions = {}
            
            # Get predictions from all models
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
            
            # Combine predictions based on method
            if self.method == "average":
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
            elif self.method == "weighted_average" and self.weights:
                weighted_preds = []
                for name, pred in predictions.items():
                    weight = self.weights.get(name, 1.0)
                    weighted_preds.append(pred * weight)
                ensemble_pred = np.sum(weighted_preds, axis=0) / sum(self.weights.values())
            else:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return np.array([])
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for weighted ensemble.
        
        Args:
            weights: Dictionary of model weights
        """
        self.weights = weights
        logger.info(f"Ensemble weights set: {weights}")


# Convenience functions
def quick_model_comparison(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         model_types: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Quick comparison of multiple model types.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_types: List of model types to compare
        
    Returns:
        Comparison results
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("Scikit-learn not available for model comparison")
        return {}
    
    if model_types is None:
        model_types = ['linear_regression', 'ridge', 'random_forest']
    
    manager = ModelManager()
    evaluator = ModelEvaluator()
    results = {}
    
    for model_type in model_types:
        try:
            # Create and train model
            model = manager.create_model(model_type)
            if model is None:
                continue
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = evaluator.evaluate_model(y_test, y_pred, model_type)
            results[model_type] = metrics
            
        except Exception as e:
            logger.error(f"Error in quick comparison for {model_type}: {e}")
    
    return results


logger.info("Model utilities loaded successfully")
