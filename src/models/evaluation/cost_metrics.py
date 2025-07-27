"""
Cost Metrics Module
==================

Provides specialized cost-related performance metrics for model evaluation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

class CostMetrics:
    """
    Specialized metrics for evaluating model performance with cost considerations.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize cost metrics calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
    
    def calculate_cost_enhanced_metrics(self, 
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      X_features: Optional[pd.DataFrame] = None,
                                      cost_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate cost-enhanced performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X_features: Feature DataFrame
            cost_features: List of cost feature names
            
        Returns:
            Dict[str, float]: Cost-enhanced metrics
        """
        metrics = {}
        
        try:
            # Standard metrics first
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Cost-adjusted metrics
            if X_features is not None and cost_features:
                cost_adjusted_metrics = self._calculate_cost_adjusted_metrics(
                    y_true, y_pred, X_features, cost_features
                )
                metrics.update(cost_adjusted_metrics)
            
            # Prediction quality metrics
            quality_metrics = self._calculate_prediction_quality_metrics(y_true, y_pred)
            metrics.update(quality_metrics)
            
            # Directional accuracy
            if len(y_true) > 1:
                directional_metrics = self._calculate_directional_metrics(y_true, y_pred)
                metrics.update(directional_metrics)
            
        except Exception as e:
            self.logger.error(f"Error calculating cost-enhanced metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_cost_adjusted_metrics(self, 
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       X_features: pd.DataFrame,
                                       cost_features: List[str]) -> Dict[str, float]:
        """Calculate metrics adjusted for transaction costs."""
        cost_metrics = {}
        
        try:
            # Extract cost information
            available_cost_features = [f for f in cost_features if f in X_features.columns]
            if not available_cost_features:
                return cost_metrics
            
            avg_costs = X_features[available_cost_features].mean(axis=1)
            
            # Cost-weighted error metrics
            cost_weights = 1 + (avg_costs / (avg_costs.mean() + 1e-8))
            
            # Cost-weighted MAE
            weighted_errors = np.abs(y_true - y_pred) * cost_weights
            cost_metrics['cost_weighted_mae'] = np.mean(weighted_errors)
            
            # Cost-weighted MSE
            weighted_squared_errors = np.square(y_true - y_pred) * cost_weights
            cost_metrics['cost_weighted_mse'] = np.mean(weighted_squared_errors)
            cost_metrics['cost_weighted_rmse'] = np.sqrt(cost_metrics['cost_weighted_mse'])
            
            # Cost-adjusted R²
            ss_res_weighted = np.sum(weighted_squared_errors)
            ss_tot_weighted = np.sum(np.square(y_true - np.mean(y_true)) * cost_weights)
            if ss_tot_weighted > 0:
                cost_metrics['cost_adjusted_r2'] = 1 - (ss_res_weighted / ss_tot_weighted)
            else:
                cost_metrics['cost_adjusted_r2'] = 0.0
            
            # Cost efficiency ratio
            prediction_quality = 1 / (1 + np.abs(y_true - y_pred))
            cost_efficiency = prediction_quality / (avg_costs + 1e-8)
            cost_metrics['cost_efficiency_ratio'] = np.mean(cost_efficiency)
            
            # Cost-return correlation
            abs_errors = np.abs(y_true - y_pred)
            cost_error_corr = np.corrcoef(avg_costs, abs_errors)[0, 1]
            cost_metrics['cost_error_correlation'] = cost_error_corr if not np.isnan(cost_error_corr) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating cost-adjusted metrics: {e}")
        
        return cost_metrics
    
    def _calculate_prediction_quality_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction quality metrics."""
        quality_metrics = {}
        
        try:
            # Mean Absolute Percentage Error
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                quality_metrics['mape'] = mape
            
            # Symmetric MAPE
            smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
            quality_metrics['smape'] = smape
            
            # Maximum error
            quality_metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            
            # Median absolute error
            quality_metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
            
            # Explained variance
            var_y = np.var(y_true)
            if var_y > 0:
                quality_metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / var_y
            else:
                quality_metrics['explained_variance'] = 0.0
            
            # Prediction intervals (assuming normal distribution)
            residuals = y_true - y_pred
            std_residuals = np.std(residuals)
            quality_metrics['prediction_std'] = std_residuals
            
            # Coverage probabilities (what percentage of predictions are within 1, 2 std)
            within_1std = np.mean(np.abs(residuals) <= std_residuals)
            within_2std = np.mean(np.abs(residuals) <= 2 * std_residuals)
            quality_metrics['coverage_1std'] = within_1std
            quality_metrics['coverage_2std'] = within_2std
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction quality metrics: {e}")
        
        return quality_metrics
    
    def _calculate_directional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate directional accuracy metrics."""
        directional_metrics = {}
        
        try:
            # Basic directional accuracy
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            directional_accuracy = np.mean(true_direction == pred_direction)
            directional_metrics['directional_accuracy'] = directional_accuracy
            
            # Up/down movement accuracy
            if len(y_true) > 1:
                true_changes = np.diff(y_true)
                pred_changes = np.diff(y_pred)
                
                true_up = true_changes > 0
                pred_up = pred_changes > 0
                
                change_accuracy = np.mean(true_up == pred_up)
                directional_metrics['change_direction_accuracy'] = change_accuracy
                
                # Magnitude-weighted directional accuracy
                magnitude_weights = np.abs(true_changes)
                magnitude_weights = magnitude_weights / (np.sum(magnitude_weights) + 1e-8)
                
                weighted_accuracy = np.sum((true_up == pred_up) * magnitude_weights)
                directional_metrics['magnitude_weighted_direction_accuracy'] = weighted_accuracy
            
            # Hit rate for different thresholds
            for threshold in [0.01, 0.02, 0.05]:  # 1%, 2%, 5%
                significant_moves = np.abs(y_true) > threshold
                if np.any(significant_moves):
                    significant_accuracy = np.mean(
                        (true_direction == pred_direction)[significant_moves]
                    )
                    directional_metrics[f'directional_accuracy_{int(threshold*100)}pct'] = significant_accuracy
            
        except Exception as e:
            self.logger.error(f"Error calculating directional metrics: {e}")
        
        return directional_metrics
    
    def calculate_regime_specific_metrics(self, 
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        market_regime: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate metrics specific to different market regimes.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            market_regime: Market regime indicators (optional)
            
        Returns:
            Dict[str, Any]: Regime-specific metrics
        """
        regime_metrics = {}
        
        try:
            if market_regime is None:
                # Create simple regime based on volatility
                rolling_vol = pd.Series(y_true).rolling(window=20).std()
                high_vol_threshold = rolling_vol.quantile(0.7)
                market_regime = (rolling_vol > high_vol_threshold).astype(int)
                regime_metrics['regime_method'] = 'volatility_based'
            else:
                regime_metrics['regime_method'] = 'provided'
            
            # Calculate metrics for each regime
            unique_regimes = np.unique(market_regime[~np.isnan(market_regime)])
            
            for regime in unique_regimes:
                regime_mask = market_regime == regime
                if np.any(regime_mask):
                    regime_y_true = y_true[regime_mask]
                    regime_y_pred = y_pred[regime_mask]
                    
                    regime_name = f'regime_{int(regime)}'
                    regime_metrics[regime_name] = {
                        'count': np.sum(regime_mask),
                        'mae': np.mean(np.abs(regime_y_true - regime_y_pred)),
                        'mse': np.mean(np.square(regime_y_true - regime_y_pred)),
                        'r2': self._safe_r2_score(regime_y_true, regime_y_pred),
                        'directional_accuracy': np.mean(np.sign(regime_y_true) == np.sign(regime_y_pred))
                    }
            
        except Exception as e:
            self.logger.error(f"Error calculating regime-specific metrics: {e}")
            regime_metrics['error'] = str(e)
        
        return regime_metrics
    
    def _safe_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Safely calculate R² score."""
        try:
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
        except:
            # Fallback calculation
            ss_res = np.sum(np.square(y_true - y_pred))
            ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def calculate_robustness_metrics(self, 
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        Calculate robustness metrics by adding noise to predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            noise_levels: List of noise levels to test
            
        Returns:
            Dict[str, Any]: Robustness metrics
        """
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1]  # 1%, 5%, 10% noise
        
        robustness_metrics = {}
        
        try:
            base_mae = np.mean(np.abs(y_true - y_pred))
            base_r2 = self._safe_r2_score(y_true, y_pred)
            
            robustness_metrics['base_metrics'] = {
                'mae': base_mae,
                'r2': base_r2
            }
            
            for noise_level in noise_levels:
                # Add noise to predictions
                noise = np.random.normal(0, noise_level * np.std(y_pred), len(y_pred))
                noisy_pred = y_pred + noise
                
                # Calculate degraded metrics
                noisy_mae = np.mean(np.abs(y_true - noisy_pred))
                noisy_r2 = self._safe_r2_score(y_true, noisy_pred)
                
                # Calculate robustness scores (lower degradation = higher robustness)
                mae_degradation = (noisy_mae - base_mae) / base_mae if base_mae > 0 else 0
                r2_degradation = (base_r2 - noisy_r2) / abs(base_r2) if base_r2 != 0 else 0
                
                robustness_metrics[f'noise_{int(noise_level*100)}pct'] = {
                    'mae_degradation': mae_degradation,
                    'r2_degradation': r2_degradation,
                    'robustness_score': 1 / (1 + mae_degradation + r2_degradation)
                }
            
        except Exception as e:
            self.logger.error(f"Error calculating robustness metrics: {e}")
            robustness_metrics['error'] = str(e)
        
        return robustness_metrics