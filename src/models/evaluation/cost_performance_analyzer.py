"""
Cost Performance Analyzer Module
===============================

Provides detailed analysis of cost impact on model performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class CostPerformanceAnalyzer:
    """
    Analyzer for detailed cost performance analysis and insights.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize cost performance analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
    
    def analyze_cost_performance(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               X_features: Optional[pd.DataFrame] = None,
                               cost_features: Optional[List[str]] = None,
                               model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive cost performance analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X_features: Feature DataFrame
            cost_features: List of cost feature names
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Comprehensive cost analysis
        """
        analysis = {
            'model_name': model_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'cost_breakdown': {},
            'performance_attribution': {},
            'cost_sensitivity': {},
            'optimization_suggestions': []
        }
        
        try:
            if X_features is not None and cost_features:
                # Cost breakdown analysis
                analysis['cost_breakdown'] = self._analyze_cost_breakdown(X_features, cost_features)
                
                # Performance attribution to costs
                analysis['performance_attribution'] = self._analyze_performance_attribution(
                    y_true, y_pred, X_features, cost_features
                )
                
                # Cost sensitivity analysis
                analysis['cost_sensitivity'] = self._analyze_cost_sensitivity(
                    y_true, y_pred, X_features, cost_features
                )
                
                # Generate optimization suggestions
                analysis['optimization_suggestions'] = self._generate_optimization_suggestions(
                    analysis['cost_breakdown'], 
                    analysis['performance_attribution'],
                    analysis['cost_sensitivity']
                )
            
            # Model performance vs cost trade-offs
            analysis['cost_efficiency_analysis'] = self._analyze_cost_efficiency(
                y_true, y_pred, X_features, cost_features
            )
            
            # Time-based cost analysis
            if X_features is not None:
                analysis['temporal_cost_analysis'] = self._analyze_temporal_costs(
                    y_true, y_pred, X_features, cost_features
                )
            
        except Exception as e:
            self.logger.error(f"Error in cost performance analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_cost_breakdown(self, X_features: pd.DataFrame, cost_features: List[str]) -> Dict[str, Any]:
        """Analyze breakdown of different cost components."""
        breakdown = {
            'total_cost_features': len(cost_features),
            'available_cost_features': 0,
            'cost_statistics': {},
            'cost_distribution': {},
            'cost_correlations': {}
        }
        
        try:
            available_features = [f for f in cost_features if f in X_features.columns]
            breakdown['available_cost_features'] = len(available_features)
            
            if not available_features:
                return breakdown
            
            cost_data = X_features[available_features]
            
            # Individual cost statistics
            for feature in available_features:
                breakdown['cost_statistics'][feature] = {
                    'mean': cost_data[feature].mean(),
                    'std': cost_data[feature].std(),
                    'min': cost_data[feature].min(),
                    'max': cost_data[feature].max(),
                    'median': cost_data[feature].median(),
                    'q25': cost_data[feature].quantile(0.25),
                    'q75': cost_data[feature].quantile(0.75),
                    'skewness': cost_data[feature].skew(),
                    'kurtosis': cost_data[feature].kurtosis()
                }
            
            # Cost distribution analysis
            total_costs = cost_data.sum(axis=1)
            breakdown['cost_distribution'] = {
                'mean_total_cost': total_costs.mean(),
                'std_total_cost': total_costs.std(),
                'cost_percentiles': {
                    'p10': total_costs.quantile(0.1),
                    'p25': total_costs.quantile(0.25),
                    'p50': total_costs.quantile(0.5),
                    'p75': total_costs.quantile(0.75),
                    'p90': total_costs.quantile(0.9)
                },
                'high_cost_threshold': total_costs.quantile(0.8),
                'low_cost_threshold': total_costs.quantile(0.2)
            }
            
            # Cost correlations
            if len(available_features) > 1:
                corr_matrix = cost_data.corr()
                breakdown['cost_correlations'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'highly_correlated_pairs': self._find_high_correlations(corr_matrix, 0.7),
                    'average_correlation': corr_matrix.abs().mean().mean()
                }
            
        except Exception as e:
            self.logger.error(f"Error in cost breakdown analysis: {e}")
            breakdown['error'] = str(e)
        
        return breakdown
    
    def _analyze_performance_attribution(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       X_features: pd.DataFrame,
                                       cost_features: List[str]) -> Dict[str, Any]:
        """Analyze how much of performance can be attributed to cost considerations."""
        attribution = {
            'cost_impact_on_errors': {},
            'cost_regime_performance': {},
            'cost_feature_contribution': {}
        }
        
        try:
            available_features = [f for f in cost_features if f in X_features.columns]
            if not available_features:
                return attribution
            
            errors = np.abs(y_true - y_pred)
            cost_data = X_features[available_features]
            
            # Cost impact on prediction errors
            total_costs = cost_data.sum(axis=1)
            cost_error_corr = np.corrcoef(total_costs, errors)[0, 1]
            attribution['cost_impact_on_errors'] = {
                'cost_error_correlation': cost_error_corr if not np.isnan(cost_error_corr) else 0.0,
                'high_cost_error_ratio': self._calculate_cost_error_ratio(total_costs, errors, 'high'),
                'low_cost_error_ratio': self._calculate_cost_error_ratio(total_costs, errors, 'low')
            }
            
            # Performance in different cost regimes
            high_cost_threshold = total_costs.quantile(0.75)
            low_cost_threshold = total_costs.quantile(0.25)
            
            high_cost_mask = total_costs > high_cost_threshold
            low_cost_mask = total_costs < low_cost_threshold
            medium_cost_mask = ~(high_cost_mask | low_cost_mask)
            
            for regime_name, mask in [('high_cost', high_cost_mask), 
                                    ('medium_cost', medium_cost_mask), 
                                    ('low_cost', low_cost_mask)]:
                if np.any(mask):
                    regime_errors = errors[mask]
                    regime_mae = np.mean(regime_errors)
                    regime_mse = np.mean(regime_errors ** 2)
                    
                    attribution['cost_regime_performance'][regime_name] = {
                        'sample_count': np.sum(mask),
                        'mae': regime_mae,
                        'mse': regime_mse,
                        'rmse': np.sqrt(regime_mse)
                    }
            
            # Individual cost feature contribution to errors
            for feature in available_features:
                feature_cost_corr = np.corrcoef(X_features[feature], errors)[0, 1]
                attribution['cost_feature_contribution'][feature] = {
                    'error_correlation': feature_cost_corr if not np.isnan(feature_cost_corr) else 0.0,
                    'relative_importance': abs(feature_cost_corr) if not np.isnan(feature_cost_corr) else 0.0
                }
            
        except Exception as e:
            self.logger.error(f"Error in performance attribution analysis: {e}")
            attribution['error'] = str(e)
        
        return attribution
    
    def _analyze_cost_sensitivity(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                X_features: pd.DataFrame,
                                cost_features: List[str]) -> Dict[str, Any]:
        """Analyze sensitivity of model performance to cost changes."""
        sensitivity = {
            'cost_elasticity': {},
            'marginal_cost_impact': {},
            'cost_threshold_analysis': {}
        }
        
        try:
            available_features = [f for f in cost_features if f in X_features.columns]
            if not available_features:
                return sensitivity
            
            errors = np.abs(y_true - y_pred)
            
            # Cost elasticity analysis
            for feature in available_features:
                elasticity = self._calculate_cost_elasticity(X_features[feature], errors)
                sensitivity['cost_elasticity'][feature] = elasticity
            
            # Marginal cost impact
            total_costs = X_features[available_features].sum(axis=1)
            sensitivity['marginal_cost_impact'] = self._calculate_marginal_impact(total_costs, errors)
            
            # Cost threshold analysis
            sensitivity['cost_threshold_analysis'] = self._analyze_cost_thresholds(total_costs, errors)
            
        except Exception as e:
            self.logger.error(f"Error in cost sensitivity analysis: {e}")
            sensitivity['error'] = str(e)
        
        return sensitivity
    
    def _analyze_cost_efficiency(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               X_features: Optional[pd.DataFrame] = None,
                               cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze cost efficiency of the model."""
        efficiency = {
            'cost_effectiveness_ratio': 0.0,
            'cost_adjusted_performance': {},
            'efficiency_benchmarks': {}
        }
        
        try:
            # Basic performance metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            
            efficiency['base_performance'] = {
                'mae': mae,
                'mse': mse,
                'r2': r2
            }
            
            if X_features is not None and cost_features:
                available_features = [f for f in cost_features if f in X_features.columns]
                if available_features:
                    avg_cost = X_features[available_features].mean(axis=1).mean()
                    
                    # Cost effectiveness ratio (performance per unit cost)
                    if avg_cost > 0:
                        efficiency['cost_effectiveness_ratio'] = max(r2, 0) / avg_cost
                    
                    # Cost-adjusted performance metrics
                    cost_penalty = avg_cost * 0.1  # 10% penalty factor
                    efficiency['cost_adjusted_performance'] = {
                        'adjusted_r2': max(r2 - cost_penalty, 0),
                        'adjusted_mae': mae * (1 + cost_penalty),
                        'cost_penalty': cost_penalty
                    }
            
            # Efficiency benchmarks
            efficiency['efficiency_benchmarks'] = {
                'excellent_threshold': 0.8,  # R² > 0.8 with low costs
                'good_threshold': 0.6,       # R² > 0.6 with moderate costs
                'acceptable_threshold': 0.4, # R² > 0.4 with high costs
                'current_rating': self._rate_efficiency(efficiency)
            }
            
        except Exception as e:
            self.logger.error(f"Error in cost efficiency analysis: {e}")
            efficiency['error'] = str(e)
        
        return efficiency
    
    def _analyze_temporal_costs(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              X_features: pd.DataFrame,
                              cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze cost patterns over time."""
        temporal = {
            'cost_trends': {},
            'volatility_analysis': {},
            'seasonal_patterns': {}
        }
        
        try:
            if cost_features is None:
                return temporal
            
            available_features = [f for f in cost_features if f in X_features.columns]
            if not available_features:
                return temporal
            
            cost_data = X_features[available_features]
            total_costs = cost_data.sum(axis=1)
            
            # Cost trends
            time_index = np.arange(len(total_costs))
            cost_trend = np.polyfit(time_index, total_costs, 1)[0]  # Linear trend
            
            temporal['cost_trends'] = {
                'linear_trend': cost_trend,
                'trend_direction': 'increasing' if cost_trend > 0 else 'decreasing',
                'trend_magnitude': abs(cost_trend),
                'cost_volatility': np.std(total_costs)
            }
            
            # Volatility analysis (rolling windows)
            if len(total_costs) > 20:
                rolling_std = pd.Series(total_costs).rolling(window=20).std()
                temporal['volatility_analysis'] = {
                    'average_volatility': rolling_std.mean(),
                    'volatility_trend': np.polyfit(time_index[19:], rolling_std.dropna(), 1)[0],
                    'max_volatility_period': rolling_std.idxmax(),
                    'min_volatility_period': rolling_std.idxmin()
                }
            
            # Simple seasonal analysis (if enough data)
            if len(total_costs) > 50:
                # Divide into periods and look for patterns
                period_length = len(total_costs) // 5
                period_means = []
                
                for i in range(5):
                    start_idx = i * period_length
                    end_idx = (i + 1) * period_length if i < 4 else len(total_costs)
                    period_mean = np.mean(total_costs[start_idx:end_idx])
                    period_means.append(period_mean)
                
                temporal['seasonal_patterns'] = {
                    'period_means': period_means,
                    'seasonal_variation': np.std(period_means),
                    'highest_cost_period': np.argmax(period_means),
                    'lowest_cost_period': np.argmin(period_means)
                }
            
        except Exception as e:
            self.logger.error(f"Error in temporal cost analysis: {e}")
            temporal['error'] = str(e)
        
        return temporal
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Find highly correlated feature pairs."""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return high_corr_pairs
    
    def _calculate_cost_error_ratio(self, costs: pd.Series, errors: np.ndarray, regime: str) -> float:
        """Calculate error ratio for cost regimes."""
        try:
            if regime == 'high':
                threshold = costs.quantile(0.75)
                mask = costs > threshold
            else:  # low
                threshold = costs.quantile(0.25)
                mask = costs < threshold
            
            if np.any(mask):
                regime_errors = errors[mask]
                overall_error = np.mean(errors)
                regime_error = np.mean(regime_errors)
                return regime_error / overall_error if overall_error > 0 else 1.0
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_cost_elasticity(self, costs: pd.Series, errors: np.ndarray) -> Dict[str, float]:
        """Calculate elasticity of errors with respect to costs."""
        elasticity = {'elasticity': 0.0, 'significance': 'low'}
        
        try:
            # Simple elasticity: % change in errors / % change in costs
            cost_changes = costs.pct_change().dropna()
            error_changes = pd.Series(errors).pct_change().dropna()
            
            # Align the series
            min_len = min(len(cost_changes), len(error_changes))
            cost_changes = cost_changes.iloc[:min_len]
            error_changes = error_changes.iloc[:min_len]
            
            # Remove infinite and zero values
            valid_mask = np.isfinite(cost_changes) & np.isfinite(error_changes) & (cost_changes != 0)
            
            if np.any(valid_mask):
                elasticity_values = error_changes[valid_mask] / cost_changes[valid_mask]
                avg_elasticity = np.mean(elasticity_values)
                elasticity['elasticity'] = avg_elasticity
                
                # Determine significance
                if abs(avg_elasticity) > 1.0:
                    elasticity['significance'] = 'high'
                elif abs(avg_elasticity) > 0.5:
                    elasticity['significance'] = 'medium'
                else:
                    elasticity['significance'] = 'low'
            
        except Exception as e:
            self.logger.error(f"Error calculating cost elasticity: {e}")
        
        return elasticity
    
    def _calculate_marginal_impact(self, costs: pd.Series, errors: np.ndarray) -> Dict[str, float]:
        """Calculate marginal impact of cost increases on errors."""
        marginal = {'marginal_error_increase': 0.0, 'cost_threshold': 0.0}
        
        try:
            # Sort by costs and calculate marginal changes
            sorted_indices = costs.argsort()
            sorted_costs = costs.iloc[sorted_indices]
            sorted_errors = errors[sorted_indices]
            
            # Calculate derivatives (discrete approximation)
            cost_diffs = np.diff(sorted_costs)
            error_diffs = np.diff(sorted_errors)
            
            # Marginal impact where cost differences are positive
            positive_cost_diffs = cost_diffs > 0
            if np.any(positive_cost_diffs):
                marginal_impacts = error_diffs[positive_cost_diffs] / cost_diffs[positive_cost_diffs]
                marginal['marginal_error_increase'] = np.mean(marginal_impacts)
                
                # Find cost threshold where marginal impact becomes significant
                significant_impact_mask = marginal_impacts > np.percentile(marginal_impacts, 75)
                if np.any(significant_impact_mask):
                    threshold_indices = np.where(positive_cost_diffs)[0][significant_impact_mask]
                    marginal['cost_threshold'] = sorted_costs.iloc[threshold_indices[0]] if len(threshold_indices) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating marginal impact: {e}")
        
        return marginal
    
    def _analyze_cost_thresholds(self, costs: pd.Series, errors: np.ndarray) -> Dict[str, Any]:
        """Analyze performance at different cost thresholds."""
        thresholds = {
            'performance_by_threshold': {},
            'optimal_cost_range': {},
            'cost_efficiency_curve': []
        }
        
        try:
            # Define cost percentile thresholds
            percentiles = [10, 25, 50, 75, 90]
            
            for p in percentiles:
                threshold = costs.quantile(p / 100)
                mask = costs <= threshold
                
                if np.any(mask):
                    threshold_errors = errors[mask]
                    thresholds['performance_by_threshold'][f'p{p}'] = {
                        'threshold_value': threshold,
                        'sample_count': np.sum(mask),
                        'mean_error': np.mean(threshold_errors),
                        'std_error': np.std(threshold_errors)
                    }
            
            # Find optimal cost range (lowest errors per unit cost)
            cost_bins = pd.qcut(costs, q=5, duplicates='drop')
            bin_performance = []
            
            for bin_label in cost_bins.cat.categories:
                bin_mask = cost_bins == bin_label
                if np.any(bin_mask):
                    bin_errors = errors[bin_mask]
                    bin_costs = costs[bin_mask]
                    
                    efficiency = np.mean(bin_errors) / (np.mean(bin_costs) + 1e-8)
                    bin_performance.append({
                        'cost_range': str(bin_label),
                        'avg_cost': np.mean(bin_costs),
                        'avg_error': np.mean(bin_errors),
                        'efficiency': efficiency
                    })
            
            if bin_performance:
                best_bin = min(bin_performance, key=lambda x: x['efficiency'])
                thresholds['optimal_cost_range'] = best_bin
                thresholds['cost_efficiency_curve'] = bin_performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing cost thresholds: {e}")
        
        return thresholds
    
    def _rate_efficiency(self, efficiency_data: Dict[str, Any]) -> str:
        """Rate the overall efficiency of the model."""
        try:
            base_r2 = efficiency_data.get('base_performance', {}).get('r2', 0.0)
            cost_effectiveness = efficiency_data.get('cost_effectiveness_ratio', 0.0)
            
            # Simple rating based on R² and cost effectiveness
            if base_r2 > 0.8 and cost_effectiveness > 1.0:
                return 'excellent'
            elif base_r2 > 0.6 and cost_effectiveness > 0.5:
                return 'good'
            elif base_r2 > 0.4 and cost_effectiveness > 0.1:
                return 'acceptable'
            else:
                return 'poor'
                
        except Exception:
            return 'unknown'
    
    def _generate_optimization_suggestions(self,
                                         cost_breakdown: Dict[str, Any],
                                         performance_attribution: Dict[str, Any],
                                         cost_sensitivity: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        try:
            # Cost correlation suggestions
            if 'cost_correlations' in cost_breakdown:
                high_corr_pairs = cost_breakdown['cost_correlations'].get('highly_correlated_pairs', [])
                if len(high_corr_pairs) > 2:
                    suggestions.append("Consider removing highly correlated cost features to reduce redundancy")
            
            # Performance attribution suggestions
            if 'cost_impact_on_errors' in performance_attribution:
                cost_error_corr = performance_attribution['cost_impact_on_errors'].get('cost_error_correlation', 0)
                if abs(cost_error_corr) > 0.3:
                    suggestions.append("Strong cost-error correlation detected - consider cost-weighted training")
            
            # Cost sensitivity suggestions
            if 'cost_elasticity' in cost_sensitivity:
                high_elasticity_features = [
                    feature for feature, data in cost_sensitivity['cost_elasticity'].items()
                    if data.get('significance') == 'high'
                ]
                if high_elasticity_features:
                    suggestions.append(f"High cost sensitivity in {', '.join(high_elasticity_features)} - monitor these features closely")
            
            # General suggestions
            if not suggestions:
                suggestions.append("Consider implementing dynamic cost thresholds for better performance")
                suggestions.append("Monitor cost trends for potential optimization opportunities")
            
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
            suggestions.append("Analysis incomplete - manual review recommended")
        
        return suggestions