"""
Cost-Aware Model Trainer Module
===============================

Extends existing model training capabilities with cost-awareness.
Integrates transaction cost considerations into model training and optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass

# Import existing model utilities
try:
    from src.models.model_utils import ModelManager, ModelEvaluator
    from src.utils.config_manager import Config
except ImportError:
    # Fallback for testing
    class ModelManager:
        def __init__(self, config=None): pass
        def create_model(self, model_type, **kwargs): return None
        def train_model(self, model, X, y, model_name): return {}
        def evaluate_model_comprehensive(self, model, X_test, y_test, X_train=None, y_train=None, model_name="model"): return {}
    
    class ModelEvaluator:
        def __init__(self): pass
        def evaluate_model(self, y_true, y_pred, model_name): return {}
    
    class Config:
        def __init__(self): pass
        def get(self, key, default=None): return default

from .cost_integration_mixin import CostIntegrationMixin

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CostAwareTrainingConfig:
    """Configuration for cost-aware training."""
    enable_cost_features: bool = True
    enable_cost_weighting: bool = True
    enable_cost_regularization: bool = False
    cost_feature_weight: float = 1.0
    cost_penalty_factor: float = 0.1
    use_cost_adjusted_metrics: bool = True
    optimize_for_cost_efficiency: bool = True
    cost_threshold_percentile: float = 75.0
    
class CostAwareTrainer(CostIntegrationMixin):
    """
    Enhanced model trainer with cost-awareness capabilities.
    
    Extends the existing ModelManager with cost-specific training features
    while maintaining full backward compatibility.
    """
    
    def __init__(self, config: Optional[Union[Config, CostAwareTrainingConfig]] = None):
        """
        Initialize cost-aware trainer.
        
        Args:
            config: Configuration object (Config or CostAwareTrainingConfig)
        """
        # Handle different config types
        if isinstance(config, CostAwareTrainingConfig):
            self.cost_config = config
            self.base_config = Config()
        else:
            self.base_config = config or Config()
            self.cost_config = CostAwareTrainingConfig()
        
        # Initialize base components
        self.model_manager = ModelManager(self.base_config)
        self.evaluator = ModelEvaluator()
        
        # Initialize mixin
        super().__init__(self.cost_config)
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Training tracking
        self.training_history = []
        self.cost_adjusted_metrics = {}
        
        self.logger.info("CostAwareTrainer initialized")
    
    def train_cost_aware_model(self, 
                              model_type: str,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: Optional[np.ndarray] = None,
                              y_test: Optional[np.ndarray] = None,
                              cost_features: Optional[List[str]] = None,
                              model_name: str = "cost_aware_model",
                              **model_kwargs) -> Dict[str, Any]:
        """
        Train a model with cost-awareness capabilities.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            cost_features: List of cost feature names (optional)
            model_name: Name for the model
            **model_kwargs: Additional model parameters
            
        Returns:
            Dict[str, Any]: Comprehensive training results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting cost-aware training for {model_type}")
        
        try:
            training_result = {
                'model_name': model_name,
                'model_type': model_type,
                'training_timestamp': start_time.isoformat(),
                'cost_aware_config': {
                    'cost_features_enabled': self.cost_config.enable_cost_features,
                    'cost_weighting_enabled': self.cost_config.enable_cost_weighting,
                    'cost_regularization_enabled': self.cost_config.enable_cost_regularization
                }
            }
            
            # Prepare training data with cost adjustments
            X_train_processed, y_train_processed, cost_info = self._prepare_cost_aware_training_data(
                X_train, y_train, cost_features
            )
            training_result['cost_preprocessing'] = cost_info
            
            # Create and configure model
            model = self.model_manager.create_model(model_type, **model_kwargs)
            if model is None:
                raise ValueError(f"Failed to create model of type {model_type}")
            
            # Apply cost-aware model configuration
            model = self._configure_cost_aware_model(model, cost_info)
            
            # Train the model
            base_training_result = self.model_manager.train_model(
                model, X_train_processed, y_train_processed, model_name
            )
            training_result.update(base_training_result)
            
            # Cost-aware evaluation
            if X_test is not None and y_test is not None:
                evaluation_result = self._evaluate_cost_aware_model(
                    model, X_test, y_test, X_train_processed, y_train_processed, 
                    cost_features, model_name
                )
                training_result['cost_aware_evaluation'] = evaluation_result
            
            # Calculate cost-adjusted metrics
            if self.cost_config.use_cost_adjusted_metrics:
                cost_metrics = self._calculate_cost_adjusted_metrics(
                    training_result, cost_info
                )
                training_result['cost_adjusted_metrics'] = cost_metrics
                self.cost_adjusted_metrics[model_name] = cost_metrics
            
            # Store training history
            training_time = (datetime.now() - start_time).total_seconds()
            training_result['total_training_time'] = training_time
            
            self.training_history.append({
                'model_name': model_name,
                'timestamp': start_time.isoformat(),
                'training_time': training_time,
                'cost_aware': True,
                'success': True
            })
            
            self.logger.info(f"Cost-aware training completed for {model_name} in {training_time:.2f}s")
            
            return training_result
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Cost-aware training failed for {model_name}: {e}"
            self.logger.error(error_msg)
            
            self.training_history.append({
                'model_name': model_name,
                'timestamp': start_time.isoformat(),
                'training_time': training_time,
                'cost_aware': True,
                'success': False,
                'error': str(e)
            })
            
            return {
                'model_name': model_name,
                'success': False,
                'error': error_msg,
                'training_time': training_time
            }
    
    def _prepare_cost_aware_training_data(self, 
                                        X: np.ndarray, 
                                        y: np.ndarray, 
                                        cost_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare training data with cost-aware adjustments.
        
        Args:
            X: Training features
            y: Training targets
            cost_features: List of cost feature names
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: Processed X, y, and cost info
        """
        cost_info = {
            'cost_features_used': [],
            'cost_weighting_applied': False,
            'sample_weights': None,
            'cost_adjustments': {}
        }
        
        try:
            X_processed = X.copy()
            y_processed = y.copy()
            
            # Identify cost features if not provided
            if cost_features is None and hasattr(X, 'columns'):
                cost_features = self._identify_cost_features(X.columns)
            cost_info['cost_features_used'] = cost_features or []
            
            # Apply cost feature weighting
            if self.cost_config.enable_cost_features and cost_features:
                X_processed = self._apply_cost_feature_weighting(X_processed, cost_features)
                cost_info['cost_adjustments']['feature_weighting'] = self.cost_config.cost_feature_weight
            
            # Apply cost-based sample weighting
            if self.cost_config.enable_cost_weighting:
                sample_weights = self._calculate_cost_based_weights(X_processed, y_processed, cost_features)
                cost_info['sample_weights'] = sample_weights
                cost_info['cost_weighting_applied'] = True
            
            # Cost-based target adjustment (optional)
            if self.cost_config.optimize_for_cost_efficiency:
                y_processed = self._adjust_targets_for_cost_efficiency(
                    y_processed, X_processed, cost_features
                )
                cost_info['cost_adjustments']['target_adjustment'] = True
                
        except Exception as e:
            self.logger.error(f"Error in cost-aware data preparation: {e}")
            X_processed = X
            y_processed = y
            cost_info['error'] = str(e)
        
        return X_processed, y_processed, cost_info
    
    def _apply_cost_feature_weighting(self, X: np.ndarray, cost_features: List[str]) -> np.ndarray:
        """
        Apply weighting to cost features to emphasize their importance.
        
        Args:
            X: Feature matrix
            cost_features: List of cost feature names
            
        Returns:
            np.ndarray: Weighted feature matrix
        """
        X_weighted = X.copy()
        
        try:
            if hasattr(X, 'columns'):  # DataFrame
                for feature in cost_features:
                    if feature in X.columns:
                        col_idx = X.columns.get_loc(feature)
                        X_weighted.iloc[:, col_idx] *= self.cost_config.cost_feature_weight
            else:  # NumPy array - assume cost features are marked by indices
                # For numpy arrays, we can't identify cost features by name
                # Apply uniform weighting to all features (conservative approach)
                pass
                
        except Exception as e:
            self.logger.error(f"Error applying cost feature weighting: {e}")
        
        return X_weighted
    
    def _calculate_cost_based_weights(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray, 
                                    cost_features: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """
        Calculate sample weights based on cost considerations.
        
        Args:
            X: Feature matrix
            y: Target vector
            cost_features: List of cost feature names
            
        Returns:
            Optional[np.ndarray]: Sample weights
        """
        try:
            if cost_features is None or len(cost_features) == 0:
                return None
            
            # Create cost-based weights
            weights = np.ones(len(X))
            
            if hasattr(X, 'columns'):  # DataFrame
                cost_columns = [col for col in cost_features if col in X.columns]
                if cost_columns:
                    # Calculate average cost for each sample
                    avg_costs = X[cost_columns].mean(axis=1)
                    
                    # Higher weights for high-cost samples (to learn cost patterns better)
                    cost_threshold = np.percentile(avg_costs, self.cost_config.cost_threshold_percentile)
                    high_cost_mask = avg_costs > cost_threshold
                    
                    weights[high_cost_mask] *= 1.5  # Increase weight for high-cost samples
                    weights[~high_cost_mask] *= 0.8  # Decrease weight for low-cost samples
            
            # Normalize weights
            weights = weights / np.mean(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating cost-based weights: {e}")
            return None
    
    def _adjust_targets_for_cost_efficiency(self, 
                                          y: np.ndarray, 
                                          X: np.ndarray, 
                                          cost_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Adjust training targets to account for cost efficiency.
        
        Args:
            y: Original targets
            X: Feature matrix
            cost_features: List of cost feature names
            
        Returns:
            np.ndarray: Adjusted targets
        """
        try:
            if cost_features is None or len(cost_features) == 0:
                return y
            
            y_adjusted = y.copy()
            
            if hasattr(X, 'columns'):  # DataFrame
                cost_columns = [col for col in cost_features if col in X.columns]
                if cost_columns:
                    # Calculate cost penalty
                    avg_costs = X[cost_columns].mean(axis=1)
                    cost_penalty = avg_costs * self.cost_config.cost_penalty_factor
                    
                    # Adjust targets by subtracting cost penalty
                    # This encourages the model to account for costs in predictions
                    y_adjusted = y - cost_penalty
            
            return y_adjusted
            
        except Exception as e:
            self.logger.error(f"Error adjusting targets for cost efficiency: {e}")
            return y
    
    def _configure_cost_aware_model(self, model: Any, cost_info: Dict[str, Any]) -> Any:
        """
        Configure model with cost-aware settings.
        
        Args:
            model: ML model instance
            cost_info: Cost information from preprocessing
            
        Returns:
            Any: Configured model
        """
        try:
            # Apply sample weights if available and model supports it
            if hasattr(model, 'fit') and cost_info.get('sample_weights') is not None:
                # Store weights for use during training
                # Note: This requires modification during actual fit() call
                model._cost_sample_weights = cost_info['sample_weights']
            
            # Apply cost-aware regularization if enabled
            if self.cost_config.enable_cost_regularization:
                if hasattr(model, 'set_params'):
                    # Increase regularization for models that support it
                    current_params = model.get_params()
                    
                    # Adjust regularization parameters
                    if 'alpha' in current_params:  # Ridge, Lasso
                        current_alpha = current_params.get('alpha', 1.0)
                        model.set_params(alpha=current_alpha * (1 + self.cost_config.cost_penalty_factor))
                    elif 'C' in current_params:  # SVM
                        current_C = current_params.get('C', 1.0)
                        model.set_params(C=current_C * (1 - self.cost_config.cost_penalty_factor))
                    elif 'reg_alpha' in current_params:  # XGBoost
                        current_reg = current_params.get('reg_alpha', 0.0)
                        model.set_params(reg_alpha=current_reg + self.cost_config.cost_penalty_factor)
                        
        except Exception as e:
            self.logger.error(f"Error configuring cost-aware model: {e}")
        
        return model
    
    def _evaluate_cost_aware_model(self, 
                                 model: Any,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 X_train: Optional[np.ndarray] = None,
                                 y_train: Optional[np.ndarray] = None,
                                 cost_features: Optional[List[str]] = None,
                                 model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate model with cost-aware metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional)
            y_train: Training targets (optional)
            cost_features: List of cost feature names
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Cost-aware evaluation results
        """
        evaluation_result = {}
        
        try:
            # Standard evaluation
            standard_eval = self.model_manager.evaluate_model_comprehensive(
                model, X_test, y_test, X_train, y_train, model_name
            )
            evaluation_result['standard_metrics'] = standard_eval
            
            # Cost-specific evaluation
            y_pred = model.predict(X_test)
            
            # Calculate cost-adjusted performance metrics
            cost_metrics = self._calculate_cost_performance_metrics(
                y_test, y_pred, X_test, cost_features
            )
            evaluation_result['cost_performance'] = cost_metrics
            
            # Cost feature importance analysis
            if cost_features and hasattr(model, 'feature_importances_'):
                cost_importance = self._analyze_cost_feature_importance(
                    model, X_test, cost_features
                )
                evaluation_result['cost_feature_importance'] = cost_importance
            
            # Trading simulation with costs
            trading_simulation = self._simulate_trading_with_costs(
                y_test, y_pred, X_test, cost_features
            )
            evaluation_result['trading_simulation'] = trading_simulation
            
        except Exception as e:
            self.logger.error(f"Error in cost-aware evaluation: {e}")
            evaluation_result['error'] = str(e)
        
        return evaluation_result
    
    def _calculate_cost_performance_metrics(self, 
                                          y_true: np.ndarray, 
                                          y_pred: np.ndarray,
                                          X: np.ndarray,
                                          cost_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate performance metrics adjusted for costs.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X: Feature matrix
            cost_features: List of cost feature names
            
        Returns:
            Dict[str, float]: Cost-adjusted metrics
        """
        metrics = {}
        
        try:
            # Standard metrics (for comparison)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Cost-adjusted metrics
            if cost_features and hasattr(X, 'columns'):
                cost_columns = [col for col in cost_features if col in X.columns]
                if cost_columns:
                    avg_costs = X[cost_columns].mean(axis=1)
                    
                    # Cost-adjusted MAE (penalize errors when costs are high)
                    cost_weights = 1 + (avg_costs / avg_costs.mean())
                    weighted_errors = np.abs(y_true - y_pred) * cost_weights
                    metrics['cost_adjusted_mae'] = np.mean(weighted_errors)
                    
                    # Cost efficiency ratio
                    prediction_accuracy = 1 / (1 + np.abs(y_true - y_pred))
                    cost_efficiency = prediction_accuracy / (avg_costs + 1e-8)
                    metrics['cost_efficiency_ratio'] = np.mean(cost_efficiency)
                    
                    # Cost-adjusted R²
                    total_cost_weighted_error = np.sum(weighted_errors)
                    mean_weighted_error = np.mean(np.abs(y_true - np.mean(y_true)) * cost_weights)
                    metrics['cost_adjusted_r2'] = 1 - (total_cost_weighted_error / (mean_weighted_error * len(y_true)))
                    
        except Exception as e:
            self.logger.error(f"Error calculating cost performance metrics: {e}")
        
        return metrics
    
    def _analyze_cost_feature_importance(self, 
                                       model: Any,
                                       X: np.ndarray,
                                       cost_features: List[str]) -> Dict[str, Any]:
        """
        Analyze importance of cost features in the trained model.
        
        Args:
            model: Trained model
            X: Feature matrix
            cost_features: List of cost feature names
            
        Returns:
            Dict[str, Any]: Cost feature importance analysis
        """
        importance_analysis = {}
        
        try:
            if hasattr(model, 'feature_importances_') and hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
                importances = model.feature_importances_
                
                # Extract cost feature importances
                cost_importances = {}
                for feature in cost_features:
                    if feature in feature_names:
                        idx = feature_names.index(feature)
                        cost_importances[feature] = importances[idx]
                
                # Calculate cost feature statistics
                if cost_importances:
                    total_cost_importance = sum(cost_importances.values())
                    total_importance = sum(importances)
                    
                    importance_analysis = {
                        'cost_feature_importances': cost_importances,
                        'cost_importance_ratio': total_cost_importance / total_importance if total_importance > 0 else 0,
                        'top_cost_features': sorted(cost_importances.items(), key=lambda x: x[1], reverse=True)[:5],
                        'cost_feature_count': len(cost_importances),
                        'average_cost_importance': total_cost_importance / len(cost_importances) if cost_importances else 0
                    }
            
        except Exception as e:
            self.logger.error(f"Error analyzing cost feature importance: {e}")
        
        return importance_analysis
    
    def _simulate_trading_with_costs(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   X: np.ndarray,
                                   cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simulate trading performance accounting for transaction costs.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            X: Feature matrix
            cost_features: List of cost feature names
            
        Returns:
            Dict[str, Any]: Trading simulation results
        """
        simulation_result = {}
        
        try:
            # Generate trading signals based on predictions
            signals = np.where(y_pred > 0, 1, -1)  # Simple long/short strategy
            
            # Calculate returns without costs
            strategy_returns = signals * y_true
            cumulative_returns_no_cost = np.cumprod(1 + strategy_returns) - 1
            
            simulation_result['returns_no_cost'] = {
                'total_return': cumulative_returns_no_cost[-1],
                'annualized_return': np.mean(strategy_returns) * 252,  # Assuming daily data
                'volatility': np.std(strategy_returns) * np.sqrt(252),
                'sharpe_ratio': np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
            }
            
            # Calculate returns with costs
            if cost_features and hasattr(X, 'columns'):
                cost_columns = [col for col in cost_features if col in X.columns]
                if cost_columns:
                    avg_costs = X[cost_columns].mean(axis=1)
                    
                    # Apply costs when there's a position change
                    position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
                    trading_costs = position_changes * avg_costs.values
                    
                    # Adjust returns for costs
                    net_returns = strategy_returns - trading_costs
                    cumulative_returns_with_cost = np.cumprod(1 + net_returns) - 1
                    
                    simulation_result['returns_with_cost'] = {
                        'total_return': cumulative_returns_with_cost[-1],
                        'annualized_return': np.mean(net_returns) * 252,
                        'volatility': np.std(net_returns) * np.sqrt(252),
                        'sharpe_ratio': np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252),
                        'total_costs': np.sum(trading_costs),
                        'average_cost_per_trade': np.mean(trading_costs[trading_costs > 0]) if np.any(trading_costs > 0) else 0
                    }
                    
                    # Cost impact analysis
                    cost_impact = (cumulative_returns_no_cost[-1] - cumulative_returns_with_cost[-1])
                    simulation_result['cost_impact'] = {
                        'absolute_impact': cost_impact,
                        'relative_impact': cost_impact / abs(cumulative_returns_no_cost[-1]) if cumulative_returns_no_cost[-1] != 0 else 0,
                        'cost_drag_bps': cost_impact * 10000  # Convert to basis points
                    }
            
        except Exception as e:
            self.logger.error(f"Error in trading simulation: {e}")
            simulation_result['error'] = str(e)
        
        return simulation_result
    
    def _calculate_cost_adjusted_metrics(self, 
                                       training_result: Dict[str, Any], 
                                       cost_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final cost-adjusted metrics for the training session.
        
        Args:
            training_result: Training results
            cost_info: Cost information
            
        Returns:
            Dict[str, Any]: Cost-adjusted metrics
        """
        cost_metrics = {
            'cost_integration_score': 0.0,
            'cost_feature_utilization': 0.0,
            'cost_aware_performance': {}
        }
        
        try:
            # Cost integration score (how well costs are integrated)
            integration_factors = []
            
            if cost_info.get('cost_features_used'):
                integration_factors.append(0.3)  # Cost features used
            if cost_info.get('cost_weighting_applied'):
                integration_factors.append(0.3)  # Cost weighting applied
            if 'cost_aware_evaluation' in training_result:
                integration_factors.append(0.4)  # Cost evaluation performed
            
            cost_metrics['cost_integration_score'] = sum(integration_factors)
            
            # Cost feature utilization
            if cost_info.get('cost_features_used'):
                cost_metrics['cost_feature_utilization'] = len(cost_info['cost_features_used']) / 10.0  # Normalize to max 10 features
            
            # Extract cost-aware performance if available
            if 'cost_aware_evaluation' in training_result:
                cost_eval = training_result['cost_aware_evaluation']
                if 'cost_performance' in cost_eval:
                    cost_metrics['cost_aware_performance'] = cost_eval['cost_performance']
                
                if 'trading_simulation' in cost_eval:
                    cost_metrics['trading_performance'] = cost_eval['trading_simulation']
            
        except Exception as e:
            self.logger.error(f"Error calculating cost-adjusted metrics: {e}")
        
        return cost_metrics
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()
    
    def get_cost_adjusted_metrics(self) -> Dict[str, Any]:
        """Get cost-adjusted metrics for all trained models."""
        return self.cost_adjusted_metrics.copy()
    
    def compare_cost_aware_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple cost-aware models.
        
        Args:
            model_results: Dictionary of model training results
            
        Returns:
            Dict[str, Any]: Comparison analysis
        """
        comparison = {
            'model_count': len(model_results),
            'cost_integration_ranking': {},
            'performance_ranking': {},
            'best_cost_model': None,
            'recommendations': []
        }
        
        try:
            cost_scores = {}
            performance_scores = {}
            
            for model_name, result in model_results.items():
                # Extract cost integration score
                if 'cost_adjusted_metrics' in result:
                    cost_scores[model_name] = result['cost_adjusted_metrics'].get('cost_integration_score', 0.0)
                
                # Extract performance score (use cost-adjusted R² if available)
                if 'cost_aware_evaluation' in result:
                    cost_perf = result['cost_aware_evaluation'].get('cost_performance', {})
                    performance_scores[model_name] = cost_perf.get('cost_adjusted_r2', 0.0)
                elif 'test_metrics' in result:
                    performance_scores[model_name] = result['test_metrics'].get('R2', 0.0)
            
            # Rank models
            if cost_scores:
                comparison['cost_integration_ranking'] = sorted(cost_scores.items(), key=lambda x: x[1], reverse=True)
            if performance_scores:
                comparison['performance_ranking'] = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Determine best cost-aware model
            if cost_scores and performance_scores:
                # Combined score: 60% performance + 40% cost integration
                combined_scores = {}
                for model_name in model_results.keys():
                    perf_score = performance_scores.get(model_name, 0.0)
                    cost_score = cost_scores.get(model_name, 0.0)
                    combined_scores[model_name] = 0.6 * perf_score + 0.4 * cost_score
                
                comparison['best_cost_model'] = max(combined_scores.items(), key=lambda x: x[1])[0]
                comparison['combined_ranking'] = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Generate recommendations
            if comparison['best_cost_model']:
                comparison['recommendations'].append(f"Best cost-aware model: {comparison['best_cost_model']}")
            
            if len(cost_scores) > 1:
                avg_cost_score = np.mean(list(cost_scores.values()))
                high_cost_models = [name for name, score in cost_scores.items() if score > avg_cost_score]
                if high_cost_models:
                    comparison['recommendations'].append(f"Models with good cost integration: {', '.join(high_cost_models)}")
            
        except Exception as e:
            self.logger.error(f"Error comparing cost-aware models: {e}")
            comparison['error'] = str(e)
        
        return comparison