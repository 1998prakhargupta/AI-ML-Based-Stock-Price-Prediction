"""
Cost Analyzer
=============

Advanced cost analysis capabilities including statistical analysis,
performance attribution, and cost optimization insights.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

# Import transaction cost models
from src.trading.transaction_costs.models import (
    TransactionRequest, TransactionCostBreakdown, 
    MarketConditions, BrokerConfiguration, InstrumentType, TransactionType
)

logger = logging.getLogger(__name__)

@dataclass
class CostStatistics:
    """Statistical analysis of costs."""
    mean_cost: float
    median_cost: float
    std_cost: float
    min_cost: float
    max_cost: float
    percentile_25: float
    percentile_75: float
    cost_efficiency_ratio: float
    outlier_count: int

@dataclass
class CostAttributionResult:
    """Cost attribution analysis result."""
    cost_by_component: Dict[str, float]
    cost_by_instrument: Dict[str, float]
    cost_by_time_period: Dict[str, float]
    cost_drivers: List[str]
    optimization_opportunities: List[str]

class CostAnalyzer:
    """
    Advanced cost analyzer providing statistical analysis, cost attribution,
    and optimization insights for transaction costs.
    """
    
    def __init__(self):
        """Initialize cost analyzer."""
        # Analysis parameters
        self.outlier_threshold = 2.5  # Standard deviations for outlier detection
        self.cost_component_weights = {
            'commission': 0.3,
            'market_impact': 0.25,
            'spreads': 0.2,
            'fees': 0.15,
            'timing': 0.1
        }
        
        # Benchmark thresholds (in basis points)
        self.cost_benchmarks = {
            'equity_retail': {'low': 5, 'medium': 15, 'high': 30},
            'equity_institutional': {'low': 2, 'medium': 8, 'high': 20},
            'options': {'low': 10, 'medium': 25, 'high': 50},
            'futures': {'low': 3, 'medium': 10, 'high': 25}
        }
        
        logger.info("Cost analyzer initialized")
    
    def analyze_cost_statistics(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        notional_values: List[Decimal]
    ) -> CostStatistics:
        """
        Perform statistical analysis of transaction costs.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            notional_values: Corresponding notional values
            
        Returns:
            Statistical analysis of costs
        """
        if not cost_breakdowns or len(cost_breakdowns) != len(notional_values):
            raise ValueError("Invalid input: cost breakdowns and notional values must be non-empty and same length")
        
        # Calculate cost in basis points
        cost_bps = []
        for breakdown, notional in zip(cost_breakdowns, notional_values):
            if notional > 0:
                bps = float(breakdown.total_cost / notional * 10000)
                cost_bps.append(bps)
        
        if not cost_bps:
            return CostStatistics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        cost_array = np.array(cost_bps)
        
        # Calculate statistics
        mean_cost = np.mean(cost_array)
        median_cost = np.median(cost_array)
        std_cost = np.std(cost_array)
        min_cost = np.min(cost_array)
        max_cost = np.max(cost_array)
        p25 = np.percentile(cost_array, 25)
        p75 = np.percentile(cost_array, 75)
        
        # Calculate efficiency ratio (lower is better)
        efficiency_ratio = std_cost / mean_cost if mean_cost > 0 else 0
        
        # Detect outliers
        outlier_threshold = mean_cost + self.outlier_threshold * std_cost
        outlier_count = np.sum(cost_array > outlier_threshold)
        
        return CostStatistics(
            mean_cost=mean_cost,
            median_cost=median_cost,
            std_cost=std_cost,
            min_cost=min_cost,
            max_cost=max_cost,
            percentile_25=p25,
            percentile_75=p75,
            cost_efficiency_ratio=efficiency_ratio,
            outlier_count=outlier_count
        )
    
    def perform_cost_attribution(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown]
    ) -> CostAttributionResult:
        """
        Perform cost attribution analysis to identify cost drivers.
        
        Args:
            transactions: List of transaction requests
            cost_breakdowns: Corresponding cost breakdowns
            
        Returns:
            Cost attribution analysis result
        """
        if len(transactions) != len(cost_breakdowns):
            raise ValueError("Transactions and cost breakdowns must have same length")
        
        # Cost by component
        component_costs = {
            'commission': 0.0,
            'regulatory_fees': 0.0,
            'exchange_fees': 0.0,
            'market_impact': 0.0,
            'spreads': 0.0,
            'timing': 0.0,
            'other': 0.0
        }
        
        # Cost by instrument type
        instrument_costs = {}
        
        # Cost by time period (hour of day)
        time_period_costs = {}
        
        total_cost = 0.0
        
        for transaction, breakdown in zip(transactions, cost_breakdowns):
            transaction_total = float(breakdown.total_cost)
            total_cost += transaction_total
            
            # Component attribution
            component_costs['commission'] += float(breakdown.commission)
            component_costs['regulatory_fees'] += float(breakdown.regulatory_fees)
            component_costs['exchange_fees'] += float(breakdown.exchange_fees)
            component_costs['market_impact'] += float(breakdown.market_impact_cost)
            component_costs['spreads'] += float(breakdown.bid_ask_spread_cost)
            component_costs['timing'] += float(breakdown.timing_cost)
            
            # Calculate other costs
            other_cost = (
                float(breakdown.platform_fees) +
                float(breakdown.data_fees) +
                float(breakdown.miscellaneous_fees) +
                float(breakdown.borrowing_cost) +
                float(breakdown.overnight_financing) +
                float(breakdown.currency_conversion)
            )
            component_costs['other'] += other_cost
            
            # Instrument type attribution
            instrument_type = transaction.instrument_type.name
            if instrument_type not in instrument_costs:
                instrument_costs[instrument_type] = 0.0
            instrument_costs[instrument_type] += transaction_total
            
            # Time period attribution
            hour = transaction.timestamp.hour
            time_period = self._get_time_period(hour)
            if time_period not in time_period_costs:
                time_period_costs[time_period] = 0.0
            time_period_costs[time_period] += transaction_total
        
        # Convert to percentages
        if total_cost > 0:
            cost_by_component = {k: (v / total_cost) * 100 for k, v in component_costs.items()}
            cost_by_instrument = {k: (v / total_cost) * 100 for k, v in instrument_costs.items()}
            cost_by_time_period = {k: (v / total_cost) * 100 for k, v in time_period_costs.items()}
        else:
            cost_by_component = {k: 0.0 for k in component_costs.keys()}
            cost_by_instrument = {}
            cost_by_time_period = {}
        
        # Identify cost drivers
        cost_drivers = self._identify_cost_drivers(
            cost_by_component, cost_by_instrument, cost_by_time_period
        )
        
        # Generate optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            cost_by_component, cost_by_instrument, cost_by_time_period, transactions
        )
        
        return CostAttributionResult(
            cost_by_component=cost_by_component,
            cost_by_instrument=cost_by_instrument,
            cost_by_time_period=cost_by_time_period,
            cost_drivers=cost_drivers,
            optimization_opportunities=optimization_opportunities
        )
    
    def benchmark_costs(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown],
        account_type: str = 'retail'
    ) -> Dict[str, Any]:
        """
        Benchmark costs against industry standards.
        
        Args:
            transactions: List of transactions
            cost_breakdowns: Corresponding cost breakdowns
            account_type: Account type for benchmarking ('retail' or 'institutional')
            
        Returns:
            Benchmarking analysis results
        """
        results = {
            'overall_benchmark': {},
            'by_instrument': {},
            'recommendations': []
        }
        
        # Group by instrument type
        instrument_groups = {}
        for transaction, breakdown in zip(transactions, cost_breakdowns):
            instrument_type = transaction.instrument_type.name.lower()
            if instrument_type not in instrument_groups:
                instrument_groups[instrument_type] = {'transactions': [], 'breakdowns': []}
            instrument_groups[instrument_type]['transactions'].append(transaction)
            instrument_groups[instrument_type]['breakdowns'].append(breakdown)
        
        # Benchmark each instrument type
        for instrument_type, data in instrument_groups.items():
            if not data['transactions']:
                continue
            
            # Calculate average cost in basis points
            avg_cost_bps = self._calculate_average_cost_bps(
                data['breakdowns'], 
                [t.notional_value for t in data['transactions']]
            )
            
            # Get benchmark key
            benchmark_key = f"{instrument_type}_{account_type}"
            if benchmark_key not in self.cost_benchmarks:
                benchmark_key = f"equity_{account_type}"  # Default to equity
            
            benchmark = self.cost_benchmarks.get(benchmark_key, {'low': 5, 'medium': 15, 'high': 30})
            
            # Classify performance
            if avg_cost_bps <= benchmark['low']:
                performance = 'excellent'
                score = 100
            elif avg_cost_bps <= benchmark['medium']:
                performance = 'good'
                score = 80
            elif avg_cost_bps <= benchmark['high']:
                performance = 'average'
                score = 60
            else:
                performance = 'poor'
                score = max(40 - (avg_cost_bps - benchmark['high']), 0)
            
            results['by_instrument'][instrument_type] = {
                'average_cost_bps': avg_cost_bps,
                'benchmark_low': benchmark['low'],
                'benchmark_medium': benchmark['medium'],
                'benchmark_high': benchmark['high'],
                'performance': performance,
                'score': score,
                'transaction_count': len(data['transactions'])
            }
        
        # Overall benchmark
        if cost_breakdowns:
            overall_avg_cost_bps = self._calculate_average_cost_bps(
                cost_breakdowns,
                [t.notional_value for t in transactions]
            )
            
            # Use equity benchmark for overall assessment
            equity_benchmark = self.cost_benchmarks[f"equity_{account_type}"]
            
            if overall_avg_cost_bps <= equity_benchmark['low']:
                overall_performance = 'excellent'
                overall_score = 100
            elif overall_avg_cost_bps <= equity_benchmark['medium']:
                overall_performance = 'good'
                overall_score = 80
            elif overall_avg_cost_bps <= equity_benchmark['high']:
                overall_performance = 'average'
                overall_score = 60
            else:
                overall_performance = 'poor'
                overall_score = max(40 - (overall_avg_cost_bps - equity_benchmark['high']), 0)
            
            results['overall_benchmark'] = {
                'average_cost_bps': overall_avg_cost_bps,
                'performance': overall_performance,
                'score': overall_score,
                'total_transactions': len(transactions)
            }
        
        # Generate recommendations
        results['recommendations'] = self._generate_benchmark_recommendations(results)
        
        return results
    
    def analyze_cost_efficiency_by_size(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown],
        size_buckets: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze cost efficiency by transaction size.
        
        Args:
            transactions: List of transactions
            cost_breakdowns: Corresponding cost breakdowns
            size_buckets: Custom size buckets (default: [1K, 10K, 100K, 1M])
            
        Returns:
            Cost efficiency analysis by size
        """
        if not size_buckets:
            size_buckets = [1000, 10000, 100000, 1000000]  # Default buckets
        
        # Categorize transactions by size
        size_categories = {}
        bucket_names = [
            f"< ${size_buckets[0]:,.0f}",
            *[f"${size_buckets[i]:,.0f} - ${size_buckets[i+1]:,.0f}" 
              for i in range(len(size_buckets)-1)],
            f"> ${size_buckets[-1]:,.0f}"
        ]
        
        for name in bucket_names:
            size_categories[name] = {'transactions': [], 'breakdowns': []}
        
        # Categorize each transaction
        for transaction, breakdown in zip(transactions, cost_breakdowns):
            notional = float(transaction.notional_value)
            
            # Find appropriate bucket
            bucket_index = 0
            for i, threshold in enumerate(size_buckets):
                if notional >= threshold:
                    bucket_index = i + 1
                else:
                    break
            
            if bucket_index >= len(bucket_names):
                bucket_index = len(bucket_names) - 1
            
            bucket_name = bucket_names[bucket_index]
            size_categories[bucket_name]['transactions'].append(transaction)
            size_categories[bucket_name]['breakdowns'].append(breakdown)
        
        # Analyze each size category
        results = {}
        for category_name, data in size_categories.items():
            if not data['transactions']:
                continue
            
            # Calculate metrics
            avg_cost_bps = self._calculate_average_cost_bps(
                data['breakdowns'],
                [t.notional_value for t in data['transactions']]
            )
            
            avg_notional = np.mean([float(t.notional_value) for t in data['transactions']])
            
            # Calculate cost efficiency (lower cost per dollar is better)
            cost_efficiency = avg_cost_bps / 10000 if avg_cost_bps > 0 else 0
            
            results[category_name] = {
                'transaction_count': len(data['transactions']),
                'average_notional': avg_notional,
                'average_cost_bps': avg_cost_bps,
                'cost_efficiency': cost_efficiency,
                'total_cost': sum(float(b.total_cost) for b in data['breakdowns']),
                'total_notional': sum(float(t.notional_value) for t in data['transactions'])
            }
        
        # Identify optimal size ranges
        optimal_sizes = self._identify_optimal_transaction_sizes(results)
        
        return {
            'size_analysis': results,
            'optimal_sizes': optimal_sizes,
            'recommendations': self._generate_size_optimization_recommendations(results)
        }
    
    def detect_cost_anomalies(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        notional_values: List[Decimal],
        threshold_multiplier: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect cost anomalies and outliers.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            notional_values: Corresponding notional values
            threshold_multiplier: Multiplier for anomaly detection threshold
            
        Returns:
            Anomaly detection results
        """
        # Calculate cost in basis points
        cost_bps = []
        for breakdown, notional in zip(cost_breakdowns, notional_values):
            if notional > 0:
                bps = float(breakdown.total_cost / notional * 10000)
                cost_bps.append(bps)
            else:
                cost_bps.append(0)
        
        if not cost_bps:
            return {'anomalies': [], 'statistics': {}, 'recommendations': []}
        
        cost_array = np.array(cost_bps)
        
        # Calculate statistical measures
        mean_cost = np.mean(cost_array)
        std_cost = np.std(cost_array)
        median_cost = np.median(cost_array)
        iqr = np.percentile(cost_array, 75) - np.percentile(cost_array, 25)
        
        # Detect anomalies using multiple methods
        anomalies = []
        
        # Method 1: Standard deviation-based
        std_threshold = mean_cost + threshold_multiplier * std_cost
        std_outliers = np.where(cost_array > std_threshold)[0]
        
        # Method 2: IQR-based
        q1 = np.percentile(cost_array, 25)
        q3 = np.percentile(cost_array, 75)
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = np.where((cost_array < iqr_lower) | (cost_array > iqr_upper))[0]
        
        # Method 3: Percentile-based (top 5%)
        percentile_threshold = np.percentile(cost_array, 95)
        percentile_outliers = np.where(cost_array > percentile_threshold)[0]
        
        # Combine anomalies
        all_anomaly_indices = set(std_outliers) | set(iqr_outliers) | set(percentile_outliers)
        
        for idx in all_anomaly_indices:
            anomaly_cost = cost_bps[idx]
            severity = 'high' if anomaly_cost > mean_cost + 3 * std_cost else 'medium'
            
            anomalies.append({
                'index': int(idx),
                'cost_bps': anomaly_cost,
                'severity': severity,
                'deviation_from_mean': (anomaly_cost - mean_cost) / std_cost if std_cost > 0 else 0,
                'detection_methods': []
            })
            
            # Track which methods detected this anomaly
            if idx in std_outliers:
                anomalies[-1]['detection_methods'].append('standard_deviation')
            if idx in iqr_outliers:
                anomalies[-1]['detection_methods'].append('iqr')
            if idx in percentile_outliers:
                anomalies[-1]['detection_methods'].append('percentile')
        
        # Sort by severity
        anomalies.sort(key=lambda x: x['cost_bps'], reverse=True)
        
        statistics = {
            'total_transactions': len(cost_bps),
            'anomaly_count': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(cost_bps)) * 100 if cost_bps else 0,
            'mean_cost_bps': mean_cost,
            'std_cost_bps': std_cost,
            'median_cost_bps': median_cost,
            'max_cost_bps': np.max(cost_array),
            'min_cost_bps': np.min(cost_array)
        }
        
        recommendations = self._generate_anomaly_recommendations(anomalies, statistics)
        
        return {
            'anomalies': anomalies,
            'statistics': statistics,
            'recommendations': recommendations
        }
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour."""
        if 9 <= hour < 11:
            return 'market_open'
        elif 11 <= hour < 14:
            return 'mid_day'
        elif 14 <= hour < 16:
            return 'market_close'
        elif 16 <= hour < 20:
            return 'after_hours'
        else:
            return 'extended_hours'
    
    def _identify_cost_drivers(
        self,
        cost_by_component: Dict[str, float],
        cost_by_instrument: Dict[str, float],
        cost_by_time_period: Dict[str, float]
    ) -> List[str]:
        """Identify primary cost drivers."""
        drivers = []
        
        # Component drivers
        for component, percentage in cost_by_component.items():
            if percentage > 30:  # >30% of total cost
                drivers.append(f"{component.title()} costs ({percentage:.1f}% of total)")
        
        # Instrument drivers
        for instrument, percentage in cost_by_instrument.items():
            if percentage > 50:  # >50% of total cost
                drivers.append(f"{instrument} trading ({percentage:.1f}% of total cost)")
        
        # Time period drivers
        for period, percentage in cost_by_time_period.items():
            if percentage > 40:  # >40% of total cost
                drivers.append(f"{period.replace('_', ' ').title()} trading ({percentage:.1f}% of total cost)")
        
        return drivers
    
    def _identify_optimization_opportunities(
        self,
        cost_by_component: Dict[str, float],
        cost_by_instrument: Dict[str, float],
        cost_by_time_period: Dict[str, float],
        transactions: List[TransactionRequest]
    ) -> List[str]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # Component optimization
        if cost_by_component.get('commission', 0) > 40:
            opportunities.append("High commission costs - negotiate better rates or consider discount brokers")
        
        if cost_by_component.get('market_impact', 0) > 30:
            opportunities.append("High market impact - consider breaking large orders into smaller sizes")
        
        if cost_by_component.get('spreads', 0) > 25:
            opportunities.append("High spread costs - use limit orders or trade during high liquidity periods")
        
        # Time-based optimization
        if cost_by_time_period.get('after_hours', 0) > 20:
            opportunities.append("Significant after-hours trading costs - consider timing trades during regular hours")
        
        if cost_by_time_period.get('market_open', 0) > 35:
            opportunities.append("High market open costs - avoid trading in first 30 minutes of market open")
        
        # Volume-based optimization
        small_transactions = [t for t in transactions if float(t.notional_value) < 10000]
        if len(small_transactions) > len(transactions) * 0.3:
            opportunities.append("Many small transactions - consider batching orders to reduce per-transaction costs")
        
        return opportunities
    
    def _calculate_average_cost_bps(
        self,
        breakdowns: List[TransactionCostBreakdown],
        notionals: List[Decimal]
    ) -> float:
        """Calculate average cost in basis points."""
        if not breakdowns or len(breakdowns) != len(notionals):
            return 0.0
        
        total_cost = sum(float(b.total_cost) for b in breakdowns)
        total_notional = sum(float(n) for n in notionals)
        
        if total_notional <= 0:
            return 0.0
        
        return (total_cost / total_notional) * 10000
    
    def _generate_benchmark_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate benchmarking recommendations."""
        recommendations = []
        
        overall = results.get('overall_benchmark', {})
        if overall:
            performance = overall.get('performance', '')
            if performance == 'poor':
                recommendations.append("Overall cost performance is poor. Immediate optimization needed.")
            elif performance == 'excellent':
                recommendations.append("Excellent cost performance. Maintain current practices.")
        
        # Instrument-specific recommendations
        by_instrument = results.get('by_instrument', {})
        for instrument, data in by_instrument.items():
            if data.get('performance') == 'poor':
                recommendations.append(f"Poor cost performance in {instrument} trading - review strategy")
            elif data.get('score', 0) < 70:
                recommendations.append(f"Below-average costs in {instrument} - consider optimization")
        
        return recommendations
    
    def _identify_optimal_transaction_sizes(self, size_analysis: Dict[str, Any]) -> List[str]:
        """Identify optimal transaction size ranges."""
        if not size_analysis:
            return []
        
        # Find size ranges with lowest cost per basis point
        optimal_ranges = []
        min_cost_bps = float('inf')
        
        for size_range, data in size_analysis.items():
            cost_bps = data.get('average_cost_bps', float('inf'))
            if cost_bps < min_cost_bps and data.get('transaction_count', 0) > 0:
                min_cost_bps = cost_bps
        
        # Find all ranges within 20% of minimum
        threshold = min_cost_bps * 1.2
        for size_range, data in size_analysis.items():
            cost_bps = data.get('average_cost_bps', float('inf'))
            if cost_bps <= threshold and data.get('transaction_count', 0) > 0:
                optimal_ranges.append(f"{size_range} ({cost_bps:.1f} bps)")
        
        return optimal_ranges
    
    def _generate_size_optimization_recommendations(self, size_analysis: Dict[str, Any]) -> List[str]:
        """Generate size-based optimization recommendations."""
        recommendations = []
        
        if not size_analysis:
            return recommendations
        
        # Find highest and lowest cost size ranges
        cost_by_size = {k: v.get('average_cost_bps', 0) for k, v in size_analysis.items() 
                       if v.get('transaction_count', 0) > 0}
        
        if cost_by_size:
            highest_cost_range = max(cost_by_size, key=cost_by_size.get)
            lowest_cost_range = min(cost_by_size, key=cost_by_size.get)
            
            if cost_by_size[highest_cost_range] > cost_by_size[lowest_cost_range] * 1.5:
                recommendations.append(f"Avoid {highest_cost_range} transactions when possible")
                recommendations.append(f"Optimize for {lowest_cost_range} transaction sizes")
        
        # Check for economies of scale
        large_transactions = [k for k, v in size_analysis.items() 
                            if 'M' in k or ('K' in k and any(char.isdigit() and int(char) >= 5 for char in k))]
        
        if large_transactions:
            large_costs = [size_analysis[k].get('average_cost_bps', 0) for k in large_transactions]
            if large_costs and min(large_costs) < 10:  # Low cost for large transactions
                recommendations.append("Consider batching smaller orders into larger transactions for cost efficiency")
        
        return recommendations
    
    def _generate_anomaly_recommendations(
        self,
        anomalies: List[Dict[str, Any]],
        statistics: Dict[str, float]
    ) -> List[str]:
        """Generate anomaly-based recommendations."""
        recommendations = []
        
        anomaly_percentage = statistics.get('anomaly_percentage', 0)
        
        if anomaly_percentage > 10:
            recommendations.append("High anomaly rate detected. Review trading processes for consistency.")
        elif anomaly_percentage > 5:
            recommendations.append("Moderate anomaly rate. Monitor cost patterns for improvement opportunities.")
        
        if anomalies:
            high_severity_count = len([a for a in anomalies if a.get('severity') == 'high'])
            if high_severity_count > 0:
                recommendations.append(f"{high_severity_count} high-severity cost anomalies require immediate investigation.")
        
        # Check for systematic issues
        detection_methods = {}
        for anomaly in anomalies:
            for method in anomaly.get('detection_methods', []):
                detection_methods[method] = detection_methods.get(method, 0) + 1
        
        if detection_methods.get('standard_deviation', 0) > len(anomalies) * 0.7:
            recommendations.append("Many anomalies detected by standard deviation - check for systematic cost increases.")
        
        return recommendations


logger.info("Cost analyzer module loaded successfully")