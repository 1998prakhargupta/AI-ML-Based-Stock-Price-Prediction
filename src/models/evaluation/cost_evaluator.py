"""
Cost-Enhanced Evaluator Module
==============================

Extends existing model evaluation with cost-awareness.
Provides cost-adjusted metrics and cost-specific evaluation capabilities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass

# Import existing evaluator
try:
    from src.models.model_utils import ModelEvaluator
except ImportError:
    # Fallback for testing
    class ModelEvaluator:
        def __init__(self): pass
        def evaluate_model(self, y_true, y_pred, model_name): return {}
        def compare_models(self, results_dict, primary_metric='R2'): return {}

from .cost_metrics import CostMetrics
from .cost_performance_analyzer import CostPerformanceAnalyzer

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CostEvaluationConfig:
    """Configuration for cost-enhanced evaluation."""
    enable_cost_metrics: bool = True
    enable_trading_simulation: bool = True
    enable_cost_analysis: bool = True
    cost_adjustment_factor: float = 1.0
    trading_frequency: str = 'daily'  # daily, hourly, etc.
    benchmark_cost_bps: float = 5.0  # Benchmark cost in basis points
    risk_free_rate: float = 0.02  # Annual risk-free rate
    
class CostEvaluator:
    """
    Enhanced model evaluator with comprehensive cost-awareness.
    
    Extends the existing ModelEvaluator with cost-specific metrics,
    trading simulations, and cost impact analysis.
    """
    
    def __init__(self, config: Optional[CostEvaluationConfig] = None):
        """
        Initialize cost-enhanced evaluator.
        
        Args:
            config: Cost evaluation configuration
        """
        self.config = config or CostEvaluationConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize components
        self.base_evaluator = ModelEvaluator()
        self.cost_metrics = CostMetrics(self.config)
        self.performance_analyzer = CostPerformanceAnalyzer(self.config)
        
        # Evaluation tracking
        self.evaluation_history = []
        self.cost_benchmarks = {}
        
        self.logger.info("CostEvaluator initialized")
    
    def evaluate_with_costs(self, 
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           X_features: Optional[pd.DataFrame] = None,
                           cost_features: Optional[List[str]] = None,
                           model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation including cost considerations.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X_features: Feature DataFrame (optional)
            cost_features: List of cost feature names (optional)
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting cost-enhanced evaluation for {model_name}")
        
        try:
            evaluation_result = {
                'model_name': model_name,
                'evaluation_timestamp': start_time.isoformat(),
                'config': {
                    'cost_metrics_enabled': self.config.enable_cost_metrics,
                    'trading_simulation_enabled': self.config.enable_trading_simulation,
                    'cost_analysis_enabled': self.config.enable_cost_analysis
                }
            }
            
            # Standard evaluation metrics
            standard_metrics = self.base_evaluator.evaluate_model(y_true, y_pred, model_name)
            evaluation_result['standard_metrics'] = standard_metrics
            
            # Cost-enhanced metrics
            if self.config.enable_cost_metrics:
                cost_metrics = self.cost_metrics.calculate_cost_enhanced_metrics(
                    y_true, y_pred, X_features, cost_features
                )
                evaluation_result['cost_metrics'] = cost_metrics
            
            # Trading simulation with costs
            if self.config.enable_trading_simulation:
                trading_results = self._simulate_trading_performance(
                    y_true, y_pred, X_features, cost_features
                )
                evaluation_result['trading_simulation'] = trading_results
            
            # Cost performance analysis
            if self.config.enable_cost_analysis:
                cost_analysis = self.performance_analyzer.analyze_cost_performance(
                    y_true, y_pred, X_features, cost_features, model_name
                )
                evaluation_result['cost_analysis'] = cost_analysis
            
            # Cost-adjusted scoring
            cost_adjusted_score = self._calculate_cost_adjusted_score(evaluation_result)
            evaluation_result['cost_adjusted_score'] = cost_adjusted_score
            
            # Benchmark comparison
            benchmark_comparison = self._compare_with_benchmarks(evaluation_result)
            evaluation_result['benchmark_comparison'] = benchmark_comparison
            
            # Store evaluation
            evaluation_time = (datetime.now() - start_time).total_seconds()
            evaluation_result['evaluation_time'] = evaluation_time
            
            self.evaluation_history.append({
                'model_name': model_name,
                'timestamp': start_time.isoformat(),
                'evaluation_time': evaluation_time,
                'cost_adjusted_score': cost_adjusted_score
            })
            
            self.logger.info(f"Cost-enhanced evaluation completed for {model_name} in {evaluation_time:.2f}s")
            
            return evaluation_result
            
        except Exception as e:
            evaluation_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Cost-enhanced evaluation failed for {model_name}: {e}"
            self.logger.error(error_msg)
            
            return {
                'model_name': model_name,
                'success': False,
                'error': error_msg,
                'evaluation_time': evaluation_time
            }
    
    def _simulate_trading_performance(self, 
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    X_features: Optional[pd.DataFrame] = None,
                                    cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simulate trading performance with realistic cost considerations.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            X_features: Feature DataFrame
            cost_features: List of cost feature names
            
        Returns:
            Dict[str, Any]: Trading simulation results
        """
        simulation_result = {
            'strategy_type': 'long_short',
            'trading_frequency': self.config.trading_frequency,
            'without_costs': {},
            'with_costs': {},
            'cost_impact': {}
        }
        
        try:
            # Generate trading signals
            signals = self._generate_trading_signals(y_pred)
            
            # Calculate returns without costs
            strategy_returns = signals * y_true
            simulation_result['without_costs'] = self._calculate_strategy_metrics(strategy_returns)
            
            # Calculate returns with costs
            if X_features is not None and cost_features:
                transaction_costs = self._extract_transaction_costs(X_features, cost_features)
                net_returns = self._apply_transaction_costs(strategy_returns, signals, transaction_costs)
                simulation_result['with_costs'] = self._calculate_strategy_metrics(net_returns)
                
                # Cost impact analysis
                cost_impact = self._analyze_cost_impact(
                    simulation_result['without_costs'], 
                    simulation_result['with_costs']
                )
                simulation_result['cost_impact'] = cost_impact
            else:
                # Use benchmark costs if no cost features available
                benchmark_costs = np.full(len(y_true), self.config.benchmark_cost_bps / 10000)
                net_returns = self._apply_transaction_costs(strategy_returns, signals, benchmark_costs)
                simulation_result['with_costs'] = self._calculate_strategy_metrics(net_returns)
                simulation_result['cost_impact'] = {'note': 'Using benchmark costs'}
            
            # Risk-adjusted metrics
            simulation_result['risk_adjusted'] = self._calculate_risk_adjusted_metrics(
                simulation_result.get('with_costs', {})
            )
            
        except Exception as e:
            self.logger.error(f"Error in trading simulation: {e}")
            simulation_result['error'] = str(e)
        
        return simulation_result
    
    def _generate_trading_signals(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from predictions.
        
        Args:
            y_pred: Predicted returns
            
        Returns:
            np.ndarray: Trading signals (-1, 0, 1)
        """
        # Simple threshold-based strategy
        signals = np.zeros_like(y_pred)
        
        # Long signal for positive predictions above threshold
        long_threshold = np.percentile(y_pred, 75)
        signals[y_pred > long_threshold] = 1
        
        # Short signal for negative predictions below threshold
        short_threshold = np.percentile(y_pred, 25)
        signals[y_pred < short_threshold] = -1
        
        return signals
    
    def _extract_transaction_costs(self, 
                                 X_features: pd.DataFrame, 
                                 cost_features: List[str]) -> np.ndarray:
        """
        Extract transaction costs from features.
        
        Args:
            X_features: Feature DataFrame
            cost_features: List of cost feature names
            
        Returns:
            np.ndarray: Transaction costs
        """
        try:
            # Use available cost features
            available_cost_features = [f for f in cost_features if f in X_features.columns]
            
            if available_cost_features:
                # Use average of available cost features
                costs = X_features[available_cost_features].mean(axis=1).values
            else:
                # Fallback to benchmark
                costs = np.full(len(X_features), self.config.benchmark_cost_bps / 10000)
            
            # Ensure costs are reasonable (clip extreme values)
            costs = np.clip(costs, 0, 0.01)  # Max 1% cost
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Error extracting transaction costs: {e}")
            return np.full(len(X_features), self.config.benchmark_cost_bps / 10000)
    
    def _apply_transaction_costs(self, 
                               returns: np.ndarray, 
                               signals: np.ndarray, 
                               costs: np.ndarray) -> np.ndarray:
        """
        Apply transaction costs to strategy returns.
        
        Args:
            returns: Strategy returns
            signals: Trading signals
            costs: Transaction costs
            
        Returns:
            np.ndarray: Net returns after costs
        """
        try:
            # Calculate position changes
            padded_signals = np.concatenate([[0], signals])
            position_changes = np.abs(np.diff(padded_signals))
            
            # Apply costs only when positions change
            cost_impact = position_changes * costs
            
            # Net returns = gross returns - transaction costs
            net_returns = returns - cost_impact
            
            return net_returns
            
        except Exception as e:
            self.logger.error(f"Error applying transaction costs: {e}")
            return returns
    
    def _calculate_strategy_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive strategy performance metrics.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dict[str, float]: Strategy metrics
        """
        metrics = {}
        
        try:
            if len(returns) == 0:
                return metrics
            
            # Basic metrics
            metrics['total_return'] = np.prod(1 + returns) - 1
            metrics['annualized_return'] = np.mean(returns) * 252  # Assuming daily data
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
            
            # Risk-adjusted metrics
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annualized_return'] - self.config.risk_free_rate) / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Drawdown metrics
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            metrics['max_drawdown'] = np.min(drawdown)
            metrics['avg_drawdown'] = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0.0
            
            # Additional metrics
            metrics['win_rate'] = np.mean(returns > 0)
            metrics['avg_win'] = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
            metrics['avg_loss'] = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0.0
            
            if metrics['avg_loss'] != 0:
                metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss'])
            else:
                metrics['profit_factor'] = float('inf') if metrics['avg_win'] > 0 else 0.0
            
            # Calmar ratio (annual return / max drawdown)
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = abs(metrics['annualized_return'] / metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = float('inf') if metrics['annualized_return'] > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating strategy metrics: {e}")
        
        return metrics
    
    def _analyze_cost_impact(self, without_costs: Dict[str, float], with_costs: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze the impact of transaction costs on strategy performance.
        
        Args:
            without_costs: Metrics without costs
            with_costs: Metrics with costs
            
        Returns:
            Dict[str, Any]: Cost impact analysis
        """
        impact_analysis = {}
        
        try:
            # Calculate absolute and relative impacts
            for metric in ['total_return', 'annualized_return', 'sharpe_ratio']:
                if metric in without_costs and metric in with_costs:
                    without_val = without_costs[metric]
                    with_val = with_costs[metric]
                    
                    absolute_impact = without_val - with_val
                    relative_impact = (absolute_impact / abs(without_val)) if without_val != 0 else 0.0
                    
                    impact_analysis[f'{metric}_impact'] = {
                        'absolute': absolute_impact,
                        'relative': relative_impact,
                        'impact_bps': absolute_impact * 10000  # Convert to basis points
                    }
            
            # Overall cost drag
            if 'annualized_return' in without_costs and 'annualized_return' in with_costs:
                annual_drag = without_costs['annualized_return'] - with_costs['annualized_return']
                impact_analysis['annual_cost_drag_bps'] = annual_drag * 10000
            
            # Cost efficiency assessment
            if 'sharpe_ratio' in without_costs and 'sharpe_ratio' in with_costs:
                sharpe_degradation = (without_costs['sharpe_ratio'] - with_costs['sharpe_ratio']) / abs(without_costs['sharpe_ratio']) if without_costs['sharpe_ratio'] != 0 else 0
                impact_analysis['sharpe_degradation'] = sharpe_degradation
                
                if sharpe_degradation < 0.1:
                    impact_analysis['cost_efficiency'] = 'excellent'
                elif sharpe_degradation < 0.2:
                    impact_analysis['cost_efficiency'] = 'good'
                elif sharpe_degradation < 0.4:
                    impact_analysis['cost_efficiency'] = 'fair'
                else:
                    impact_analysis['cost_efficiency'] = 'poor'
            
        except Exception as e:
            self.logger.error(f"Error analyzing cost impact: {e}")
        
        return impact_analysis
    
    def _calculate_risk_adjusted_metrics(self, strategy_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate additional risk-adjusted metrics.
        
        Args:
            strategy_metrics: Strategy performance metrics
            
        Returns:
            Dict[str, float]: Risk-adjusted metrics
        """
        risk_metrics = {}
        
        try:
            # Information ratio (assuming benchmark return = risk-free rate)
            if 'annualized_return' in strategy_metrics and 'volatility' in strategy_metrics:
                excess_return = strategy_metrics['annualized_return'] - self.config.risk_free_rate
                if strategy_metrics['volatility'] > 0:
                    risk_metrics['information_ratio'] = excess_return / strategy_metrics['volatility']
            
            # Sortino ratio (downside deviation)
            # Note: This would require return series, approximating with volatility
            if 'annualized_return' in strategy_metrics and 'volatility' in strategy_metrics:
                downside_vol = strategy_metrics['volatility'] * 0.7  # Approximation
                excess_return = strategy_metrics['annualized_return'] - self.config.risk_free_rate
                if downside_vol > 0:
                    risk_metrics['sortino_ratio'] = excess_return / downside_vol
            
            # Risk-adjusted return
            if 'total_return' in strategy_metrics and 'max_drawdown' in strategy_metrics:
                if strategy_metrics['max_drawdown'] != 0:
                    risk_metrics['risk_adjusted_return'] = strategy_metrics['total_return'] / abs(strategy_metrics['max_drawdown'])
                    
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted metrics: {e}")
        
        return risk_metrics
    
    def _calculate_cost_adjusted_score(self, evaluation_result: Dict[str, Any]) -> float:
        """
        Calculate a single cost-adjusted score for model comparison.
        
        Args:
            evaluation_result: Complete evaluation results
            
        Returns:
            float: Cost-adjusted score (0-1, higher is better)
        """
        try:
            score_components = []
            
            # Standard metrics component (30%)
            if 'standard_metrics' in evaluation_result:
                r2_score = evaluation_result['standard_metrics'].get('R2', 0.0)
                score_components.append(('standard_r2', max(r2_score, 0.0), 0.3))
            
            # Cost metrics component (25%)
            if 'cost_metrics' in evaluation_result:
                cost_metrics = evaluation_result['cost_metrics']
                if 'cost_adjusted_r2' in cost_metrics:
                    cost_r2 = max(cost_metrics['cost_adjusted_r2'], 0.0)
                    score_components.append(('cost_r2', cost_r2, 0.25))
            
            # Trading performance component (30%)
            if 'trading_simulation' in evaluation_result:
                trading_sim = evaluation_result['trading_simulation']
                if 'with_costs' in trading_sim:
                    sharpe = trading_sim['with_costs'].get('sharpe_ratio', 0.0)
                    # Normalize Sharpe ratio to 0-1 scale
                    normalized_sharpe = max(min(sharpe / 2.0, 1.0), 0.0)
                    score_components.append(('trading_sharpe', normalized_sharpe, 0.3))
            
            # Cost efficiency component (15%)
            if 'trading_simulation' in evaluation_result:
                cost_impact = evaluation_result['trading_simulation'].get('cost_impact', {})
                efficiency = cost_impact.get('cost_efficiency', 'poor')
                efficiency_scores = {'excellent': 1.0, 'good': 0.75, 'fair': 0.5, 'poor': 0.25}
                efficiency_score = efficiency_scores.get(efficiency, 0.25)
                score_components.append(('cost_efficiency', efficiency_score, 0.15))
            
            # Calculate weighted average
            if score_components:
                total_score = sum(score * weight for _, score, weight in score_components)
                total_weight = sum(weight for _, _, weight in score_components)
                final_score = total_score / total_weight if total_weight > 0 else 0.0
            else:
                final_score = 0.0
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculating cost-adjusted score: {e}")
            return 0.0
    
    def _compare_with_benchmarks(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare evaluation results with established benchmarks.
        
        Args:
            evaluation_result: Evaluation results
            
        Returns:
            Dict[str, Any]: Benchmark comparison
        """
        comparison = {
            'benchmarks': {},
            'performance_vs_benchmarks': {},
            'recommendations': []
        }
        
        try:
            # Define benchmarks
            benchmarks = {
                'market_neutral': {
                    'expected_sharpe': 1.0,
                    'expected_annual_return': 0.08,
                    'max_acceptable_drawdown': -0.15
                },
                'cost_efficient': {
                    'max_cost_drag_bps': 100,  # 1% annual cost drag
                    'min_sharpe_after_costs': 0.8,
                    'max_sharpe_degradation': 0.2
                }
            }
            
            comparison['benchmarks'] = benchmarks
            
            # Compare trading performance
            if 'trading_simulation' in evaluation_result:
                trading_sim = evaluation_result['trading_simulation']
                
                if 'with_costs' in trading_sim:
                    with_costs = trading_sim['with_costs']
                    
                    # Sharpe ratio comparison
                    actual_sharpe = with_costs.get('sharpe_ratio', 0.0)
                    benchmark_sharpe = benchmarks['market_neutral']['expected_sharpe']
                    comparison['performance_vs_benchmarks']['sharpe_vs_benchmark'] = actual_sharpe / benchmark_sharpe
                    
                    # Annual return comparison
                    actual_return = with_costs.get('annualized_return', 0.0)
                    benchmark_return = benchmarks['market_neutral']['expected_annual_return']
                    comparison['performance_vs_benchmarks']['return_vs_benchmark'] = actual_return / benchmark_return
                    
                    # Drawdown comparison
                    actual_drawdown = with_costs.get('max_drawdown', 0.0)
                    benchmark_drawdown = benchmarks['market_neutral']['max_acceptable_drawdown']
                    comparison['performance_vs_benchmarks']['drawdown_vs_benchmark'] = actual_drawdown / benchmark_drawdown
                
                # Cost efficiency comparison
                if 'cost_impact' in trading_sim:
                    cost_impact = trading_sim['cost_impact']
                    
                    # Cost drag
                    actual_drag = cost_impact.get('annual_cost_drag_bps', 0.0)
                    benchmark_drag = benchmarks['cost_efficient']['max_cost_drag_bps']
                    comparison['performance_vs_benchmarks']['cost_drag_vs_benchmark'] = actual_drag / benchmark_drag
                    
                    # Sharpe degradation
                    actual_degradation = cost_impact.get('sharpe_degradation', 0.0)
                    benchmark_degradation = benchmarks['cost_efficient']['max_sharpe_degradation']
                    comparison['performance_vs_benchmarks']['degradation_vs_benchmark'] = actual_degradation / benchmark_degradation
            
            # Generate recommendations
            perf_vs_bench = comparison['performance_vs_benchmarks']
            
            if perf_vs_bench.get('sharpe_vs_benchmark', 0) < 0.8:
                comparison['recommendations'].append("Consider improving risk-adjusted returns")
            
            if perf_vs_bench.get('cost_drag_vs_benchmark', 0) > 1.0:
                comparison['recommendations'].append("High cost drag - optimize transaction costs")
            
            if perf_vs_bench.get('degradation_vs_benchmark', 0) > 1.0:
                comparison['recommendations'].append("Significant performance degradation due to costs")
            
            if perf_vs_bench.get('return_vs_benchmark', 0) > 1.2:
                comparison['recommendations'].append("Strong performance - consider increasing position sizes")
                
        except Exception as e:
            self.logger.error(f"Error comparing with benchmarks: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def compare_cost_aware_models(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models using cost-aware metrics.
        
        Args:
            evaluation_results: Dictionary of model evaluation results
            
        Returns:
            Dict[str, Any]: Model comparison analysis
        """
        comparison = {
            'model_count': len(evaluation_results),
            'rankings': {},
            'best_models': {},
            'summary_stats': {},
            'recommendations': []
        }
        
        try:
            if not evaluation_results:
                return comparison
            
            # Extract key metrics for comparison
            model_scores = {}
            cost_scores = {}
            trading_scores = {}
            
            for model_name, results in evaluation_results.items():
                # Cost-adjusted score
                model_scores[model_name] = results.get('cost_adjusted_score', 0.0)
                
                # Cost efficiency
                if 'trading_simulation' in results:
                    cost_impact = results['trading_simulation'].get('cost_impact', {})
                    efficiency = cost_impact.get('cost_efficiency', 'poor')
                    efficiency_values = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
                    cost_scores[model_name] = efficiency_values.get(efficiency, 1)
                
                # Trading performance
                if 'trading_simulation' in results and 'with_costs' in results['trading_simulation']:
                    trading_perf = results['trading_simulation']['with_costs']
                    sharpe = trading_perf.get('sharpe_ratio', 0.0)
                    trading_scores[model_name] = max(sharpe, 0.0)
            
            # Create rankings
            if model_scores:
                comparison['rankings']['overall'] = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                comparison['best_models']['overall'] = comparison['rankings']['overall'][0][0]
            
            if cost_scores:
                comparison['rankings']['cost_efficiency'] = sorted(cost_scores.items(), key=lambda x: x[1], reverse=True)
                comparison['best_models']['cost_efficiency'] = comparison['rankings']['cost_efficiency'][0][0]
            
            if trading_scores:
                comparison['rankings']['trading_performance'] = sorted(trading_scores.items(), key=lambda x: x[1], reverse=True)
                comparison['best_models']['trading_performance'] = comparison['rankings']['trading_performance'][0][0]
            
            # Summary statistics
            if model_scores:
                scores = list(model_scores.values())
                comparison['summary_stats']['cost_adjusted_scores'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'range': np.max(scores) - np.min(scores)
                }
            
            # Generate recommendations
            if comparison['best_models'].get('overall'):
                comparison['recommendations'].append(f"Best overall model: {comparison['best_models']['overall']}")
            
            if comparison['best_models'].get('cost_efficiency'):
                comparison['recommendations'].append(f"Most cost-efficient model: {comparison['best_models']['cost_efficiency']}")
            
            # Check for consistent performance
            if len(set(comparison['best_models'].values())) == 1:
                comparison['recommendations'].append("Consistent best performer across all metrics")
            else:
                comparison['recommendations'].append("Consider trade-offs between different performance aspects")
                
        except Exception as e:
            self.logger.error(f"Error comparing cost-aware models: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed.
        
        Returns:
            Dict[str, Any]: Evaluation summary
        """
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'evaluation_history': self.evaluation_history.copy(),
            'average_evaluation_time': 0.0,
            'top_models': [],
            'cost_benchmarks': self.cost_benchmarks.copy()
        }
        
        try:
            if self.evaluation_history:
                eval_times = [e.get('evaluation_time', 0) for e in self.evaluation_history]
                summary['average_evaluation_time'] = np.mean(eval_times)
                
                # Get top models by cost-adjusted score
                scored_models = [(e['model_name'], e.get('cost_adjusted_score', 0)) 
                               for e in self.evaluation_history if 'cost_adjusted_score' in e]
                if scored_models:
                    summary['top_models'] = sorted(scored_models, key=lambda x: x[1], reverse=True)[:5]
                    
        except Exception as e:
            self.logger.error(f"Error creating evaluation summary: {e}")
        
        return summary