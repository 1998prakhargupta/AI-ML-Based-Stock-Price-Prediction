"""
Cost Summary Generator
======================

Generates executive summaries and high-level cost reports for management
and decision-making purposes.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal

# Import cost reporting components
from .cost_reporter import CostReporter, CostAnalysisResult, BrokerComparisonResult
from .cost_analyzer import CostAnalyzer, CostStatistics, CostAttributionResult

# Import transaction cost models
from src.trading.transaction_costs.models import (
    TransactionRequest, TransactionCostBreakdown, 
    BrokerConfiguration
)

logger = logging.getLogger(__name__)

@dataclass
class CostSummary:
    """High-level cost summary."""
    total_cost: float
    total_notional: float
    average_cost_bps: float
    cost_efficiency_score: float
    period_start: datetime
    period_end: datetime
    transaction_count: int
    key_insights: List[str]
    recommendations: List[str]

@dataclass
class ExecutiveCostReport:
    """Executive-level cost report."""
    summary: CostSummary
    cost_trends: Dict[str, Any]
    broker_comparison: Optional[BrokerComparisonResult]
    cost_attribution: CostAttributionResult
    performance_metrics: Dict[str, float]
    action_items: List[str]

class CostSummaryGenerator:
    """
    Generates executive summaries and high-level cost reports
    for management review and strategic decision-making.
    """
    
    def __init__(self):
        """Initialize cost summary generator."""
        self.cost_reporter = CostReporter()
        self.cost_analyzer = CostAnalyzer()
        
        # Executive reporting thresholds
        self.cost_alert_thresholds = {
            'high_cost_bps': 25.0,      # Alert if costs > 25 bps
            'cost_increase': 0.15,       # Alert if costs increase > 15%
            'efficiency_drop': 0.20,     # Alert if efficiency drops > 20%
            'anomaly_rate': 0.10        # Alert if anomaly rate > 10%
        }
        
        logger.info("Cost summary generator initialized")
    
    def generate_executive_summary(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown],
        period_start: datetime,
        period_end: datetime,
        broker_configs: Optional[List[BrokerConfiguration]] = None
    ) -> ExecutiveCostReport:
        """
        Generate comprehensive executive cost report.
        
        Args:
            transactions: List of transactions for the period
            cost_breakdowns: Corresponding cost breakdowns
            period_start: Report period start date
            period_end: Report period end date
            broker_configs: Optional broker configurations for comparison
            
        Returns:
            Executive cost report
        """
        logger.info(f"Generating executive cost summary for {len(transactions)} transactions")
        
        if len(transactions) != len(cost_breakdowns):
            raise ValueError("Transactions and cost breakdowns must have same length")
        
        # Generate cost summary
        summary = self._generate_cost_summary(
            transactions, cost_breakdowns, period_start, period_end
        )
        
        # Analyze cost trends
        cost_trends = self._analyze_cost_trends(transactions, cost_breakdowns)
        
        # Broker comparison (if multiple brokers available)
        broker_comparison = None
        if broker_configs and len(broker_configs) > 1:
            # Use a representative transaction for comparison
            if transactions:
                representative_transaction = transactions[0]
                try:
                    broker_comparison = self.cost_reporter.compare_broker_costs(
                        representative_transaction, broker_configs
                    )
                except Exception as e:
                    logger.warning(f"Broker comparison failed: {e}")
        
        # Cost attribution analysis
        cost_attribution = self.cost_analyzer.perform_cost_attribution(
            transactions, cost_breakdowns
        )
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            transactions, cost_breakdowns
        )
        
        # Generate action items
        action_items = self._generate_action_items(
            summary, cost_trends, cost_attribution, performance_metrics
        )
        
        report = ExecutiveCostReport(
            summary=summary,
            cost_trends=cost_trends,
            broker_comparison=broker_comparison,
            cost_attribution=cost_attribution,
            performance_metrics=performance_metrics,
            action_items=action_items
        )
        
        logger.info("Executive cost summary generated successfully")
        return report
    
    def generate_daily_cost_summary(
        self,
        daily_transactions: Dict[datetime, List[TransactionRequest]],
        daily_cost_breakdowns: Dict[datetime, List[TransactionCostBreakdown]]
    ) -> Dict[str, Any]:
        """
        Generate daily cost summaries for operational monitoring.
        
        Args:
            daily_transactions: Transactions grouped by day
            daily_cost_breakdowns: Cost breakdowns grouped by day
            
        Returns:
            Daily cost summary data
        """
        logger.info(f"Generating daily summaries for {len(daily_transactions)} days")
        
        daily_summaries = {}
        
        for date, transactions in daily_transactions.items():
            if date not in daily_cost_breakdowns:
                continue
            
            breakdowns = daily_cost_breakdowns[date]
            if len(transactions) != len(breakdowns):
                logger.warning(f"Mismatch in transaction/breakdown count for {date}")
                continue
            
            # Calculate daily metrics
            total_cost = sum(float(b.total_cost) for b in breakdowns)
            total_notional = sum(float(t.notional_value) for t in transactions)
            
            cost_bps = (total_cost / total_notional * 10000) if total_notional > 0 else 0
            
            # Categorize costs
            cost_categories = self._categorize_daily_costs(breakdowns)
            
            # Generate alerts
            alerts = self._generate_daily_alerts(cost_bps, len(transactions), cost_categories)
            
            daily_summaries[date.strftime('%Y-%m-%d')] = {
                'transaction_count': len(transactions),
                'total_cost': total_cost,
                'total_notional': total_notional,
                'cost_bps': cost_bps,
                'cost_categories': cost_categories,
                'alerts': alerts,
                'status': self._get_daily_status(cost_bps, alerts)
            }
        
        # Calculate period statistics
        period_stats = self._calculate_period_statistics(daily_summaries)
        
        return {
            'daily_summaries': daily_summaries,
            'period_statistics': period_stats,
            'summary_metadata': {
                'days_analyzed': len(daily_summaries),
                'total_transaction_days': len([d for d in daily_summaries.values() if d['transaction_count'] > 0]),
                'generation_timestamp': datetime.now().isoformat()
            }
        }
    
    def generate_monthly_cost_report(
        self,
        monthly_data: Dict[str, Any],
        comparison_periods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate monthly cost report with period comparisons.
        
        Args:
            monthly_data: Monthly aggregated data
            comparison_periods: Previous periods for comparison
            
        Returns:
            Monthly cost report
        """
        logger.info("Generating monthly cost report")
        
        current_month = monthly_data.get('current_month', {})
        
        # Key metrics
        key_metrics = {
            'total_cost': current_month.get('total_cost', 0),
            'total_notional': current_month.get('total_notional', 0),
            'average_cost_bps': current_month.get('average_cost_bps', 0),
            'transaction_count': current_month.get('transaction_count', 0),
            'trading_days': current_month.get('trading_days', 0)
        }
        
        # Month-over-month comparison
        mom_comparison = {}
        if comparison_periods and len(comparison_periods) > 0:
            previous_month = monthly_data.get('previous_months', {}).get(comparison_periods[0], {})
            if previous_month:
                mom_comparison = self._calculate_period_comparison(current_month, previous_month)
        
        # Cost breakdown by categories
        cost_breakdown = current_month.get('cost_breakdown', {})
        
        # Performance indicators
        performance_indicators = self._calculate_monthly_performance_indicators(current_month)
        
        # Monthly insights
        insights = self._generate_monthly_insights(key_metrics, mom_comparison, performance_indicators)
        
        # Strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            key_metrics, mom_comparison, performance_indicators
        )
        
        return {
            'report_period': current_month.get('period', 'Unknown'),
            'key_metrics': key_metrics,
            'mom_comparison': mom_comparison,
            'cost_breakdown': cost_breakdown,
            'performance_indicators': performance_indicators,
            'insights': insights,
            'strategic_recommendations': strategic_recommendations,
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'comparison_periods': comparison_periods or [],
                'data_quality_score': self._assess_data_quality(current_month)
            }
        }
    
    def _generate_cost_summary(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown],
        period_start: datetime,
        period_end: datetime
    ) -> CostSummary:
        """Generate high-level cost summary."""
        # Calculate totals
        total_cost = sum(float(b.total_cost) for b in cost_breakdowns)
        total_notional = sum(float(t.notional_value) for t in transactions)
        
        # Calculate average cost in basis points
        average_cost_bps = (total_cost / total_notional * 10000) if total_notional > 0 else 0
        
        # Calculate efficiency score
        notional_values = [t.notional_value for t in transactions]
        cost_stats = self.cost_analyzer.analyze_cost_statistics(cost_breakdowns, notional_values)
        cost_efficiency_score = 100 - min(cost_stats.cost_efficiency_ratio * 50, 100)
        
        # Generate key insights
        key_insights = self._generate_key_insights(
            total_cost, average_cost_bps, len(transactions), cost_stats
        )
        
        # Generate recommendations
        recommendations = self._generate_summary_recommendations(
            average_cost_bps, cost_efficiency_score, cost_stats
        )
        
        return CostSummary(
            total_cost=total_cost,
            total_notional=total_notional,
            average_cost_bps=average_cost_bps,
            cost_efficiency_score=cost_efficiency_score,
            period_start=period_start,
            period_end=period_end,
            transaction_count=len(transactions),
            key_insights=key_insights,
            recommendations=recommendations
        )
    
    def _analyze_cost_trends(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown]
    ) -> Dict[str, Any]:
        """Analyze cost trends over time."""
        # Group transactions by day
        daily_costs = {}
        for transaction, breakdown in zip(transactions, cost_breakdowns):
            date = transaction.timestamp.date()
            if date not in daily_costs:
                daily_costs[date] = {'cost': 0, 'notional': 0, 'count': 0}
            
            daily_costs[date]['cost'] += float(breakdown.total_cost)
            daily_costs[date]['notional'] += float(transaction.notional_value)
            daily_costs[date]['count'] += 1
        
        # Calculate daily cost in basis points
        daily_cost_bps = []
        dates = sorted(daily_costs.keys())
        
        for date in dates:
            day_data = daily_costs[date]
            if day_data['notional'] > 0:
                bps = (day_data['cost'] / day_data['notional']) * 10000
                daily_cost_bps.append(bps)
        
        if len(daily_cost_bps) < 2:
            return {
                'trend_available': False,
                'reason': 'Insufficient data for trend analysis'
            }
        
        # Calculate trend metrics
        import numpy as np
        
        x = np.arange(len(daily_cost_bps))
        y = np.array(daily_cost_bps)
        
        # Linear trend
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
        else:
            correlation = 0
        
        # Trend direction
        if correlation > 0.1:
            trend_direction = 'increasing'
        elif correlation < -0.1:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # Volatility
        volatility = np.std(daily_cost_bps)
        
        return {
            'trend_available': True,
            'trend_direction': trend_direction,
            'trend_strength': abs(correlation),
            'volatility': volatility,
            'average_daily_cost_bps': np.mean(daily_cost_bps),
            'cost_range': {
                'min': np.min(daily_cost_bps),
                'max': np.max(daily_cost_bps)
            },
            'days_analyzed': len(daily_cost_bps)
        }
    
    def _calculate_performance_metrics(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown]
    ) -> Dict[str, float]:
        """Calculate key performance metrics."""
        if not transactions or not cost_breakdowns:
            return {}
        
        # Basic metrics
        total_cost = sum(float(b.total_cost) for b in cost_breakdowns)
        total_notional = sum(float(t.notional_value) for t in transactions)
        avg_cost_bps = (total_cost / total_notional * 10000) if total_notional > 0 else 0
        
        # Cost efficiency metrics
        notional_values = [t.notional_value for t in transactions]
        cost_stats = self.cost_analyzer.analyze_cost_statistics(cost_breakdowns, notional_values)
        
        # Anomaly detection
        anomaly_result = self.cost_analyzer.detect_cost_anomalies(cost_breakdowns, notional_values)
        
        return {
            'average_cost_bps': avg_cost_bps,
            'cost_volatility': cost_stats.std_cost,
            'cost_efficiency_ratio': cost_stats.cost_efficiency_ratio,
            'anomaly_rate': anomaly_result['statistics'].get('anomaly_percentage', 0) / 100,
            'cost_consistency_score': max(100 - cost_stats.cost_efficiency_ratio * 50, 0),
            'outlier_count': cost_stats.outlier_count,
            'transaction_count': len(transactions),
            'average_transaction_size': total_notional / len(transactions) if transactions else 0
        }
    
    def _generate_action_items(
        self,
        summary: CostSummary,
        cost_trends: Dict[str, Any],
        cost_attribution: CostAttributionResult,
        performance_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate executive action items."""
        action_items = []
        
        # High cost alerts
        if summary.average_cost_bps > self.cost_alert_thresholds['high_cost_bps']:
            action_items.append(f"URGENT: Average costs ({summary.average_cost_bps:.1f} bps) exceed threshold. Immediate review required.")
        
        # Trend alerts
        if cost_trends.get('trend_available') and cost_trends.get('trend_direction') == 'increasing':
            if cost_trends.get('trend_strength', 0) > 0.3:
                action_items.append("WARNING: Costs are trending upward. Investigate root causes.")
        
        # Efficiency alerts
        if summary.cost_efficiency_score < 60:
            action_items.append("Cost efficiency is below target. Review trading processes and broker arrangements.")
        
        # Anomaly alerts
        anomaly_rate = performance_metrics.get('anomaly_rate', 0)
        if anomaly_rate > self.cost_alert_thresholds['anomaly_rate']:
            action_items.append(f"High anomaly rate ({anomaly_rate*100:.1f}%). Investigate irregular cost patterns.")
        
        # Attribution-based actions
        for opportunity in cost_attribution.optimization_opportunities:
            action_items.append(f"OPTIMIZATION: {opportunity}")
        
        # Performance-based actions
        if performance_metrics.get('cost_volatility', 0) > 15:
            action_items.append("High cost volatility detected. Standardize execution processes.")
        
        return action_items
    
    def _categorize_daily_costs(self, breakdowns: List[TransactionCostBreakdown]) -> Dict[str, float]:
        """Categorize daily costs by type."""
        categories = {
            'commission': 0.0,
            'fees': 0.0,
            'market_impact': 0.0,
            'spreads': 0.0,
            'other': 0.0
        }
        
        for breakdown in breakdowns:
            categories['commission'] += float(breakdown.commission)
            categories['fees'] += float(breakdown.regulatory_fees + breakdown.exchange_fees)
            categories['market_impact'] += float(breakdown.market_impact_cost)
            categories['spreads'] += float(breakdown.bid_ask_spread_cost)
            categories['other'] += float(
                breakdown.platform_fees + breakdown.data_fees + 
                breakdown.miscellaneous_fees + breakdown.timing_cost
            )
        
        return categories
    
    def _generate_daily_alerts(
        self,
        cost_bps: float,
        transaction_count: int,
        cost_categories: Dict[str, float]
    ) -> List[str]:
        """Generate daily cost alerts."""
        alerts = []
        
        if cost_bps > 30:
            alerts.append(f"HIGH_COST: Daily costs at {cost_bps:.1f} bps")
        
        if transaction_count > 100:
            alerts.append(f"HIGH_VOLUME: {transaction_count} transactions")
        
        # Category-specific alerts
        total_cost = sum(cost_categories.values())
        if total_cost > 0:
            for category, amount in cost_categories.items():
                percentage = (amount / total_cost) * 100
                if percentage > 50:
                    alerts.append(f"CONCENTRATION: {category} is {percentage:.1f}% of costs")
        
        return alerts
    
    def _get_daily_status(self, cost_bps: float, alerts: List[str]) -> str:
        """Get daily status based on costs and alerts."""
        if alerts:
            if any('HIGH_COST' in alert or 'URGENT' in alert for alert in alerts):
                return 'CRITICAL'
            elif any('HIGH_VOLUME' in alert or 'CONCENTRATION' in alert for alert in alerts):
                return 'WARNING'
            else:
                return 'ATTENTION'
        elif cost_bps < 10:
            return 'EXCELLENT'
        elif cost_bps < 20:
            return 'GOOD'
        else:
            return 'REVIEW'
    
    def _calculate_period_statistics(self, daily_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics across the period."""
        if not daily_summaries:
            return {}
        
        # Extract daily metrics
        daily_costs = [d['cost_bps'] for d in daily_summaries.values() if d['cost_bps'] > 0]
        
        if not daily_costs:
            return {'error': 'No cost data available'}
        
        import numpy as np
        
        return {
            'average_daily_cost_bps': np.mean(daily_costs),
            'median_daily_cost_bps': np.median(daily_costs),
            'max_daily_cost_bps': np.max(daily_costs),
            'min_daily_cost_bps': np.min(daily_costs),
            'cost_volatility': np.std(daily_costs),
            'trading_days': len([d for d in daily_summaries.values() if d['transaction_count'] > 0]),
            'total_alerts': sum(len(d['alerts']) for d in daily_summaries.values()),
            'critical_days': len([d for d in daily_summaries.values() if d['status'] == 'CRITICAL']),
            'excellent_days': len([d for d in daily_summaries.values() if d['status'] == 'EXCELLENT'])
        }
    
    def _calculate_period_comparison(self, current: Dict, previous: Dict) -> Dict[str, Any]:
        """Calculate period-over-period comparison."""
        comparison = {}
        
        metrics_to_compare = ['total_cost', 'average_cost_bps', 'transaction_count']
        
        for metric in metrics_to_compare:
            current_val = current.get(metric, 0)
            previous_val = previous.get(metric, 0)
            
            if previous_val > 0:
                change_pct = ((current_val - previous_val) / previous_val) * 100
                comparison[f'{metric}_change_pct'] = change_pct
                comparison[f'{metric}_direction'] = 'up' if change_pct > 0 else 'down' if change_pct < 0 else 'flat'
            
            comparison[f'{metric}_current'] = current_val
            comparison[f'{metric}_previous'] = previous_val
        
        return comparison
    
    def _calculate_monthly_performance_indicators(self, monthly_data: Dict) -> Dict[str, float]:
        """Calculate monthly performance indicators."""
        return {
            'cost_efficiency_score': self._calculate_efficiency_score(monthly_data),
            'cost_stability_score': self._calculate_stability_score(monthly_data),
            'optimization_score': self._calculate_optimization_score(monthly_data),
            'benchmark_score': self._calculate_benchmark_score(monthly_data)
        }
    
    def _calculate_efficiency_score(self, data: Dict) -> float:
        """Calculate cost efficiency score (0-100)."""
        avg_cost_bps = data.get('average_cost_bps', 0)
        
        if avg_cost_bps <= 5:
            return 100
        elif avg_cost_bps <= 10:
            return 90
        elif avg_cost_bps <= 15:
            return 75
        elif avg_cost_bps <= 20:
            return 60
        elif avg_cost_bps <= 30:
            return 40
        else:
            return max(20 - (avg_cost_bps - 30), 0)
    
    def _calculate_stability_score(self, data: Dict) -> float:
        """Calculate cost stability score (0-100)."""
        volatility = data.get('cost_volatility', 0)
        return max(100 - volatility * 5, 0)
    
    def _calculate_optimization_score(self, data: Dict) -> float:
        """Calculate optimization score based on improvement opportunities."""
        # Simplified score based on cost categories
        cost_breakdown = data.get('cost_breakdown', {})
        total_cost = sum(cost_breakdown.values()) if cost_breakdown else 1
        
        score = 100
        
        # Penalize high commission percentage
        commission_pct = (cost_breakdown.get('commission', 0) / total_cost) * 100
        if commission_pct > 50:
            score -= (commission_pct - 50) * 0.5
        
        # Penalize high market impact
        impact_pct = (cost_breakdown.get('market_impact', 0) / total_cost) * 100
        if impact_pct > 30:
            score -= (impact_pct - 30) * 0.8
        
        return max(score, 0)
    
    def _calculate_benchmark_score(self, data: Dict) -> float:
        """Calculate benchmark score against industry standards."""
        avg_cost_bps = data.get('average_cost_bps', 0)
        
        # Industry benchmarks (simplified)
        if avg_cost_bps <= 8:  # Institutional level
            return 100
        elif avg_cost_bps <= 15:  # Good retail
            return 85
        elif avg_cost_bps <= 25:  # Average retail
            return 70
        else:  # Poor performance
            return max(50 - (avg_cost_bps - 25), 0)
    
    def _generate_key_insights(
        self,
        total_cost: float,
        average_cost_bps: float,
        transaction_count: int,
        cost_stats: CostStatistics
    ) -> List[str]:
        """Generate key insights for executive summary."""
        insights = []
        
        # Cost level insights
        if average_cost_bps < 10:
            insights.append(f"Excellent cost performance at {average_cost_bps:.1f} bps average")
        elif average_cost_bps > 25:
            insights.append(f"High cost level at {average_cost_bps:.1f} bps requires attention")
        
        # Volume insights
        if transaction_count > 1000:
            insights.append(f"High trading volume: {transaction_count:,} transactions")
        
        # Consistency insights
        if cost_stats.cost_efficiency_ratio < 0.3:
            insights.append("Consistent cost structure with low volatility")
        elif cost_stats.cost_efficiency_ratio > 0.8:
            insights.append("High cost volatility indicates process inconsistencies")
        
        # Outlier insights
        if cost_stats.outlier_count > 0:
            outlier_pct = (cost_stats.outlier_count / transaction_count) * 100
            insights.append(f"{cost_stats.outlier_count} outlier transactions ({outlier_pct:.1f}%)")
        
        return insights
    
    def _generate_summary_recommendations(
        self,
        average_cost_bps: float,
        cost_efficiency_score: float,
        cost_stats: CostStatistics
    ) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []
        
        # Cost level recommendations
        if average_cost_bps > 20:
            recommendations.append("Negotiate better broker rates or consider alternative execution venues")
        
        # Efficiency recommendations
        if cost_efficiency_score < 70:
            recommendations.append("Implement cost monitoring and optimization processes")
        
        # Consistency recommendations
        if cost_stats.cost_efficiency_ratio > 0.5:
            recommendations.append("Standardize trading processes to reduce cost volatility")
        
        # Outlier recommendations
        if cost_stats.outlier_count > 0:
            recommendations.append("Investigate and address high-cost outlier transactions")
        
        return recommendations
    
    def _generate_monthly_insights(
        self,
        key_metrics: Dict,
        mom_comparison: Dict,
        performance_indicators: Dict
    ) -> List[str]:
        """Generate monthly insights."""
        insights = []
        
        # Performance insights
        efficiency_score = performance_indicators.get('cost_efficiency_score', 0)
        if efficiency_score > 90:
            insights.append("Excellent cost performance maintained")
        elif efficiency_score < 60:
            insights.append("Cost performance below target - requires improvement")
        
        # Trend insights
        if mom_comparison:
            cost_change = mom_comparison.get('average_cost_bps_change_pct', 0)
            if abs(cost_change) > 10:
                direction = "increased" if cost_change > 0 else "decreased"
                insights.append(f"Costs {direction} by {abs(cost_change):.1f}% vs previous month")
        
        # Volume insights
        transaction_count = key_metrics.get('transaction_count', 0)
        if transaction_count > 5000:
            insights.append("High monthly trading volume")
        
        return insights
    
    def _generate_strategic_recommendations(
        self,
        key_metrics: Dict,
        mom_comparison: Dict,
        performance_indicators: Dict
    ) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []
        
        # Efficiency-based recommendations
        efficiency_score = performance_indicators.get('cost_efficiency_score', 0)
        if efficiency_score < 70:
            recommendations.append("Develop comprehensive cost optimization strategy")
        
        # Trend-based recommendations
        if mom_comparison:
            cost_change = mom_comparison.get('average_cost_bps_change_pct', 0)
            if cost_change > 15:
                recommendations.append("Investigate cost increase drivers and implement controls")
        
        # Benchmark-based recommendations
        benchmark_score = performance_indicators.get('benchmark_score', 0)
        if benchmark_score < 70:
            recommendations.append("Benchmark against industry leaders and adopt best practices")
        
        return recommendations
    
    def _assess_data_quality(self, data: Dict) -> float:
        """Assess data quality score (0-100)."""
        score = 100
        
        # Check for missing key metrics
        required_fields = ['total_cost', 'transaction_count', 'average_cost_bps']
        missing_fields = [field for field in required_fields if not data.get(field)]
        score -= len(missing_fields) * 20
        
        # Check for reasonable values
        if data.get('average_cost_bps', 0) > 100:  # Suspiciously high
            score -= 10
        
        if data.get('transaction_count', 0) == 0:  # No data
            score -= 30
        
        return max(score, 0)


logger.info("Cost summary generator module loaded successfully")