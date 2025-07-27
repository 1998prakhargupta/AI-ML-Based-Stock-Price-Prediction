"""
Quality Checker
==============

Performs quality checks on transaction cost calculations to ensure
reliability and accuracy of the overall system.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
import statistics
from collections import defaultdict, deque

# Import existing components
from ..models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration
)
from .result_validator import ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for the calculation system."""
    accuracy_score: float = 0.0          # 0-1 score based on validation results
    consistency_score: float = 0.0       # 0-1 score based on result consistency
    reliability_score: float = 0.0       # 0-1 score based on error rates
    performance_score: float = 0.0       # 0-1 score based on performance metrics
    overall_quality_score: float = 0.0   # Weighted average of all scores
    
    # Detailed metrics
    total_calculations: int = 0
    successful_calculations: int = 0
    validation_pass_rate: float = 0.0
    average_confidence: float = 0.0
    calculation_variance: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy_score': self.accuracy_score,
            'consistency_score': self.consistency_score,
            'reliability_score': self.reliability_score,
            'performance_score': self.performance_score,
            'overall_quality_score': self.overall_quality_score,
            'total_calculations': self.total_calculations,
            'successful_calculations': self.successful_calculations,
            'validation_pass_rate': self.validation_pass_rate,
            'average_confidence': self.average_confidence,
            'calculation_variance': self.calculation_variance,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class QualityBenchmark:
    """Quality benchmark for comparison."""
    name: str
    expected_accuracy: float
    expected_consistency: float
    expected_reliability: float
    expected_performance: float
    tolerance: float = 0.05  # 5% tolerance
    
    def evaluate(self, metrics: QualityMetrics) -> Dict[str, bool]:
        """Evaluate if metrics meet benchmark."""
        return {
            'accuracy_met': metrics.accuracy_score >= (self.expected_accuracy - self.tolerance),
            'consistency_met': metrics.consistency_score >= (self.expected_consistency - self.tolerance),
            'reliability_met': metrics.reliability_score >= (self.expected_reliability - self.tolerance),
            'performance_met': metrics.performance_score >= (self.expected_performance - self.tolerance),
            'overall_met': metrics.overall_quality_score >= (
                (self.expected_accuracy + self.expected_consistency + 
                 self.expected_reliability + self.expected_performance) / 4 - self.tolerance
            )
        }


class QualityChecker:
    """
    Comprehensive quality checker for transaction cost calculations.
    
    Features:
    - Multi-dimensional quality assessment
    - Trend analysis and quality degradation detection
    - Benchmark comparison
    - Quality improvement recommendations
    - Historical quality tracking
    """
    
    def __init__(
        self,
        quality_window_size: int = 1000,  # Number of recent calculations to analyze
        benchmark_name: str = "production",
        enable_trend_analysis: bool = True
    ):
        """
        Initialize quality checker.
        
        Args:
            quality_window_size: Size of rolling window for quality analysis
            benchmark_name: Name of quality benchmark to use
            enable_trend_analysis: Whether to track quality trends
        """
        self.quality_window_size = quality_window_size
        self.benchmark_name = benchmark_name
        self.enable_trend_analysis = enable_trend_analysis
        
        # Quality tracking
        self.calculation_history: deque = deque(maxlen=quality_window_size)
        self.validation_history: deque = deque(maxlen=quality_window_size)
        self.performance_history: deque = deque(maxlen=quality_window_size)
        
        # Quality metrics over time
        self.quality_metrics_history: deque = deque(maxlen=100)  # Last 100 quality assessments
        
        # Benchmarks
        self.benchmarks = self._setup_benchmarks()
        
        # Quality trend tracking
        self.trend_data = {
            'accuracy_trend': deque(maxlen=50),
            'consistency_trend': deque(maxlen=50),
            'reliability_trend': deque(maxlen=50),
            'performance_trend': deque(maxlen=50)
        }
        
        logger.info(f"Quality checker initialized (benchmark: {benchmark_name})")
    
    def _setup_benchmarks(self) -> Dict[str, QualityBenchmark]:
        """Setup quality benchmarks."""
        return {
            'development': QualityBenchmark(
                name='development',
                expected_accuracy=0.80,
                expected_consistency=0.75,
                expected_reliability=0.85,
                expected_performance=0.70,
                tolerance=0.10
            ),
            'staging': QualityBenchmark(
                name='staging',
                expected_accuracy=0.90,
                expected_consistency=0.85,
                expected_reliability=0.92,
                expected_performance=0.85,
                tolerance=0.05
            ),
            'production': QualityBenchmark(
                name='production',
                expected_accuracy=0.95,
                expected_consistency=0.92,
                expected_reliability=0.98,
                expected_performance=0.90,
                tolerance=0.03
            )
        }
    
    def record_calculation(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        calculation_time: float,
        success: bool,
        validation_result: Optional[ValidationResult] = None,
        error: Optional[str] = None
    ):
        """Record a calculation for quality tracking."""
        record = {
            'timestamp': datetime.now(),
            'symbol': request.symbol,
            'notional_value': float(request.notional_value),
            'calculation_time': calculation_time,
            'success': success,
            'error': error,
            'total_cost': float(cost_breakdown.total_cost) if cost_breakdown else 0.0,
            'confidence_level': getattr(cost_breakdown, 'confidence_level', None),
            'validation_passed': validation_result.is_valid if validation_result else None,
            'validation_issues': len(validation_result.issues) if validation_result else 0
        }
        
        self.calculation_history.append(record)
        
        if validation_result:
            self.validation_history.append({
                'timestamp': datetime.now(),
                'is_valid': validation_result.is_valid,
                'issues': validation_result.issues,
                'warnings': validation_result.warnings
            })
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'calculation_time': calculation_time,
            'success': success
        })
    
    def assess_quality(self) -> QualityMetrics:
        """Assess current system quality."""
        if not self.calculation_history:
            return QualityMetrics()
        
        # Calculate individual quality scores
        accuracy_score = self._calculate_accuracy_score()
        consistency_score = self._calculate_consistency_score()
        reliability_score = self._calculate_reliability_score()
        performance_score = self._calculate_performance_score()
        
        # Calculate overall quality score (weighted average)
        weights = {'accuracy': 0.3, 'consistency': 0.25, 'reliability': 0.25, 'performance': 0.2}
        overall_score = (
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency'] +
            reliability_score * weights['reliability'] +
            performance_score * weights['performance']
        )
        
        # Additional metrics
        total_calcs = len(self.calculation_history)
        successful_calcs = sum(1 for calc in self.calculation_history if calc['success'])
        
        validation_passes = sum(1 for val in self.validation_history if val['is_valid'])
        validation_pass_rate = validation_passes / len(self.validation_history) if self.validation_history else 0.0
        
        confidences = [calc['confidence_level'] for calc in self.calculation_history 
                      if calc['confidence_level'] is not None]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Calculate cost variance (consistency indicator)
        costs = [calc['total_cost'] for calc in self.calculation_history if calc['total_cost'] > 0]
        cost_variance = statistics.variance(costs) if len(costs) > 1 else 0.0
        
        metrics = QualityMetrics(
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            reliability_score=reliability_score,
            performance_score=performance_score,
            overall_quality_score=overall_score,
            total_calculations=total_calcs,
            successful_calculations=successful_calcs,
            validation_pass_rate=validation_pass_rate,
            average_confidence=avg_confidence,
            calculation_variance=cost_variance
        )
        
        # Store for trend analysis
        self.quality_metrics_history.append(metrics)
        
        if self.enable_trend_analysis:
            self._update_trend_data(metrics)
        
        return metrics
    
    def _calculate_accuracy_score(self) -> float:
        """Calculate accuracy score based on validation results."""
        if not self.validation_history:
            return 0.0
        
        # Base score from validation pass rate
        valid_results = sum(1 for val in self.validation_history if val['is_valid'])
        base_score = valid_results / len(self.validation_history)
        
        # Penalty for warnings and errors
        total_issues = sum(len(val['issues']) for val in self.validation_history)
        total_warnings = sum(len(val['warnings']) for val in self.validation_history)
        
        avg_issues_per_calc = total_issues / len(self.validation_history)
        avg_warnings_per_calc = total_warnings / len(self.validation_history)
        
        # Reduce score based on issues (errors more than warnings)
        issue_penalty = min(0.3, avg_issues_per_calc * 0.1)
        warning_penalty = min(0.1, avg_warnings_per_calc * 0.05)
        
        accuracy_score = max(0.0, base_score - issue_penalty - warning_penalty)
        return accuracy_score
    
    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score based on result variance."""
        if len(self.calculation_history) < 10:
            return 0.0
        
        # Group calculations by symbol and analyze consistency
        symbol_costs = defaultdict(list)
        
        for calc in self.calculation_history:
            if calc['success'] and calc['total_cost'] > 0:
                # Normalize by notional value for fair comparison
                normalized_cost = calc['total_cost'] / calc['notional_value']
                symbol_costs[calc['symbol']].append(normalized_cost)
        
        consistency_scores = []
        
        for symbol, costs in symbol_costs.items():
            if len(costs) >= 3:  # Need minimum samples
                # Calculate coefficient of variation
                if statistics.mean(costs) > 0:
                    cv = statistics.stdev(costs) / statistics.mean(costs)
                    # Lower CV means higher consistency
                    consistency_score = max(0.0, 1.0 - cv)
                    consistency_scores.append(consistency_score)
        
        if not consistency_scores:
            return 0.5  # Neutral score if insufficient data
        
        return statistics.mean(consistency_scores)
    
    def _calculate_reliability_score(self) -> float:
        """Calculate reliability score based on success rate and error patterns."""
        if not self.calculation_history:
            return 0.0
        
        # Base score from success rate
        successful_calcs = sum(1 for calc in self.calculation_history if calc['success'])
        success_rate = successful_calcs / len(self.calculation_history)
        
        # Check for error patterns (consecutive failures reduce score more)
        consecutive_failures = 0
        max_consecutive_failures = 0
        current_streak = 0
        
        for calc in reversed(list(self.calculation_history)):
            if not calc['success']:
                current_streak += 1
                max_consecutive_failures = max(max_consecutive_failures, current_streak)
            else:
                current_streak = 0
        
        # Penalty for consecutive failures
        streak_penalty = min(0.2, max_consecutive_failures * 0.05)
        
        reliability_score = max(0.0, success_rate - streak_penalty)
        return reliability_score
    
    def _calculate_performance_score(self) -> float:
        """Calculate performance score based on calculation times."""
        if not self.performance_history:
            return 0.0
        
        # Get recent calculation times
        recent_times = [p['calculation_time'] for p in self.performance_history 
                       if p['success']]
        
        if not recent_times:
            return 0.0
        
        avg_time = statistics.mean(recent_times)
        p95_time = self._percentile(recent_times, 0.95)
        
        # Score based on target performance (100ms average, 500ms p95)
        target_avg = 0.1  # 100ms
        target_p95 = 0.5  # 500ms
        
        avg_score = max(0.0, 1.0 - (avg_time - target_avg) / target_avg) if avg_time > target_avg else 1.0
        p95_score = max(0.0, 1.0 - (p95_time - target_p95) / target_p95) if p95_time > target_p95 else 1.0
        
        # Weighted combination
        performance_score = 0.6 * avg_score + 0.4 * p95_score
        return performance_score
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _update_trend_data(self, metrics: QualityMetrics):
        """Update trend tracking data."""
        self.trend_data['accuracy_trend'].append(metrics.accuracy_score)
        self.trend_data['consistency_trend'].append(metrics.consistency_score)
        self.trend_data['reliability_trend'].append(metrics.reliability_score)
        self.trend_data['performance_trend'].append(metrics.performance_score)
    
    def get_quality_trends(self) -> Dict[str, str]:
        """Analyze quality trends."""
        trends = {}
        
        for metric_name, trend_data in self.trend_data.items():
            if len(trend_data) < 5:
                trends[metric_name] = 'insufficient_data'
                continue
            
            # Simple trend analysis - compare recent vs older values
            recent_avg = statistics.mean(list(trend_data)[-5:])
            older_avg = statistics.mean(list(trend_data)[-10:-5]) if len(trend_data) >= 10 else recent_avg
            
            if abs(recent_avg - older_avg) < 0.02:  # Within 2%
                trends[metric_name] = 'stable'
            elif recent_avg > older_avg:
                trends[metric_name] = 'improving'
            else:
                trends[metric_name] = 'degrading'
        
        return trends
    
    def compare_to_benchmark(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """Compare current quality to benchmark."""
        benchmark_name = benchmark_name or self.benchmark_name
        
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")
        
        benchmark = self.benchmarks[benchmark_name]
        current_metrics = self.assess_quality()
        
        evaluation = benchmark.evaluate(current_metrics)
        
        return {
            'benchmark_name': benchmark_name,
            'benchmark_expectations': {
                'accuracy': benchmark.expected_accuracy,
                'consistency': benchmark.expected_consistency,
                'reliability': benchmark.expected_reliability,
                'performance': benchmark.expected_performance
            },
            'current_metrics': current_metrics.to_dict(),
            'evaluation': evaluation,
            'overall_pass': all(evaluation.values()),
            'gaps': {
                'accuracy': max(0, benchmark.expected_accuracy - current_metrics.accuracy_score),
                'consistency': max(0, benchmark.expected_consistency - current_metrics.consistency_score),
                'reliability': max(0, benchmark.expected_reliability - current_metrics.reliability_score),
                'performance': max(0, benchmark.expected_performance - current_metrics.performance_score)
            }
        }
    
    def identify_quality_issues(self) -> List[Dict[str, Any]]:
        """Identify specific quality issues and recommendations."""
        issues = []
        metrics = self.assess_quality()
        trends = self.get_quality_trends()
        
        # Accuracy issues
        if metrics.accuracy_score < 0.8:
            issues.append({
                'category': 'accuracy',
                'severity': 'high' if metrics.accuracy_score < 0.6 else 'medium',
                'issue': 'Low accuracy score',
                'current_value': metrics.accuracy_score,
                'recommendation': 'Review validation failures and improve calculation logic',
                'trend': trends.get('accuracy_trend', 'unknown')
            })
        
        # Consistency issues
        if metrics.consistency_score < 0.7:
            issues.append({
                'category': 'consistency',
                'severity': 'high' if metrics.consistency_score < 0.5 else 'medium',
                'issue': 'High result variance',
                'current_value': metrics.consistency_score,
                'recommendation': 'Investigate calculation inconsistencies and market data quality',
                'trend': trends.get('consistency_trend', 'unknown')
            })
        
        # Reliability issues
        if metrics.reliability_score < 0.9:
            issues.append({
                'category': 'reliability',
                'severity': 'high' if metrics.reliability_score < 0.8 else 'medium',
                'issue': 'Calculation failures',
                'current_value': metrics.reliability_score,
                'recommendation': 'Address error handling and input validation',
                'trend': trends.get('reliability_trend', 'unknown')
            })
        
        # Performance issues
        if metrics.performance_score < 0.7:
            issues.append({
                'category': 'performance',
                'severity': 'medium',
                'issue': 'Slow calculation times',
                'current_value': metrics.performance_score,
                'recommendation': 'Optimize algorithms and consider caching improvements',
                'trend': trends.get('performance_trend', 'unknown')
            })
        
        # Trend-based issues
        for metric_name, trend in trends.items():
            if trend == 'degrading':
                issues.append({
                    'category': 'trend',
                    'severity': 'medium',
                    'issue': f'{metric_name.replace("_trend", "")} is degrading',
                    'current_value': trend,
                    'recommendation': f'Monitor {metric_name.replace("_trend", "")} closely and investigate root causes',
                    'trend': trend
                })
        
        return issues
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report."""
        metrics = self.assess_quality()
        trends = self.get_quality_trends()
        benchmark_comparison = self.compare_to_benchmark()
        issues = self.identify_quality_issues()
        
        return {
            'quality_metrics': metrics.to_dict(),
            'quality_trends': trends,
            'benchmark_comparison': benchmark_comparison,
            'quality_issues': issues,
            'recommendations': self._generate_recommendations(issues),
            'historical_quality': [m.to_dict() for m in list(self.quality_metrics_history)[-10:]],
            'report_timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Group issues by category
        issues_by_category = defaultdict(list)
        for issue in issues:
            issues_by_category[issue['category']].append(issue)
        
        # Generate category-specific recommendations
        if 'accuracy' in issues_by_category:
            recommendations.append("Implement stricter validation rules and improve calculation algorithms")
        
        if 'consistency' in issues_by_category:
            recommendations.append("Standardize calculation parameters and improve market data normalization")
        
        if 'reliability' in issues_by_category:
            recommendations.append("Enhance error handling and implement better fallback mechanisms")
        
        if 'performance' in issues_by_category:
            recommendations.append("Optimize critical calculation paths and expand caching strategies")
        
        if 'trend' in issues_by_category:
            recommendations.append("Establish regular quality monitoring and alerting to catch degradation early")
        
        # General recommendations
        if len(issues) > 3:
            recommendations.append("Consider implementing a comprehensive quality improvement program")
        
        return recommendations
    
    def set_benchmark(self, benchmark_name: str):
        """Set the active benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")
        
        self.benchmark_name = benchmark_name
        logger.info(f"Set quality benchmark to: {benchmark_name}")
    
    def add_custom_benchmark(self, benchmark: QualityBenchmark):
        """Add a custom quality benchmark."""
        self.benchmarks[benchmark.name] = benchmark
        logger.info(f"Added custom benchmark: {benchmark.name}")
    
    def clear_history(self):
        """Clear quality tracking history."""
        self.calculation_history.clear()
        self.validation_history.clear()
        self.performance_history.clear()
        self.quality_metrics_history.clear()
        
        for trend_data in self.trend_data.values():
            trend_data.clear()
        
        logger.info("Cleared quality checker history")


logger.info("Quality checker module loaded successfully")