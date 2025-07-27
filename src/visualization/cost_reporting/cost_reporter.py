"""
Cost Reporter
=============

Comprehensive cost breakdown analysis, impact assessment, and broker comparison tools.
Provides detailed transaction cost analysis and generates comprehensive cost reports.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal

# Import transaction cost models
from src.trading.transaction_costs.models import (
    TransactionRequest, TransactionCostBreakdown, 
    MarketConditions, BrokerConfiguration
)
from src.trading.transaction_costs.cost_aggregator import CostAggregator, AggregationResult

# Import utilities
from src.utils.file_management_utils import SafeFileManager
from src.utils.config_manager import Config

logger = logging.getLogger(__name__)

@dataclass
class CostAnalysisResult:
    """Result of comprehensive cost analysis."""
    total_costs: Decimal
    cost_breakdown: TransactionCostBreakdown
    cost_efficiency_score: float
    cost_impact_on_returns: float
    recommendations: List[str]
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BrokerComparisonResult:
    """Result of broker comparison analysis."""
    best_broker: str
    cost_differences: Dict[str, Decimal]
    potential_savings: Decimal
    comparison_matrix: pd.DataFrame
    recommendations: List[str]

class CostReporter:
    """
    Comprehensive cost reporter providing detailed cost breakdowns,
    impact analysis, broker comparisons, and cost efficiency analysis.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cost reporter.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        self.cost_aggregator = CostAggregator()
        
        # Analysis parameters
        self.basis_points_threshold = 10.0  # Threshold for high cost warning
        self.efficiency_benchmarks = {
            'excellent': 5.0,    # < 5 bps
            'good': 10.0,        # < 10 bps  
            'average': 20.0,     # < 20 bps
            'poor': float('inf') # >= 20 bps
        }
        
        logger.info("Cost reporter initialized")
    
    def generate_comprehensive_cost_breakdown(
        self,
        request: TransactionRequest,
        broker_configs: List[BrokerConfiguration],
        market_conditions: Optional[MarketConditions] = None
    ) -> Dict[str, CostAnalysisResult]:
        """
        Generate comprehensive cost breakdown analysis for multiple brokers.
        
        Args:
            request: Transaction request
            broker_configs: List of broker configurations to analyze
            market_conditions: Current market conditions
            
        Returns:
            Dictionary of cost analysis results by broker
        """
        logger.info(f"Generating cost breakdown for {request.symbol} across {len(broker_configs)} brokers")
        
        results = {}
        
        for broker_config in broker_configs:
            try:
                # Calculate costs for this broker
                aggregation_result = self.cost_aggregator.calculate_total_cost(
                    request, broker_config, market_conditions
                )
                
                # Analyze cost efficiency
                efficiency_score = self._calculate_cost_efficiency(
                    aggregation_result.cost_breakdown, request.notional_value
                )
                
                # Calculate impact on returns
                return_impact = self._calculate_return_impact(
                    aggregation_result.cost_breakdown, request.notional_value
                )
                
                # Generate recommendations
                recommendations = self._generate_cost_recommendations(
                    aggregation_result.cost_breakdown, request, efficiency_score
                )
                
                # Create analysis result
                analysis_result = CostAnalysisResult(
                    total_costs=aggregation_result.cost_breakdown.total_cost,
                    cost_breakdown=aggregation_result.cost_breakdown,
                    cost_efficiency_score=efficiency_score,
                    cost_impact_on_returns=return_impact,
                    recommendations=recommendations,
                    analysis_metadata={
                        'broker_name': broker_config.broker_name,
                        'calculation_time': aggregation_result.calculation_time,
                        'confidence_score': aggregation_result.confidence_score,
                        'errors': aggregation_result.errors,
                        'warnings': aggregation_result.warnings
                    }
                )
                
                results[broker_config.broker_name] = analysis_result
                
            except Exception as e:
                logger.error(f"Cost analysis failed for {broker_config.broker_name}: {e}")
                # Create error result
                error_breakdown = TransactionCostBreakdown()
                error_breakdown.cost_details = {'error': str(e)}
                
                results[broker_config.broker_name] = CostAnalysisResult(
                    total_costs=Decimal('0'),
                    cost_breakdown=error_breakdown,
                    cost_efficiency_score=0.0,
                    cost_impact_on_returns=0.0,
                    recommendations=[f"Error calculating costs: {str(e)}"],
                    analysis_metadata={'error': True}
                )
        
        logger.info(f"Cost breakdown analysis completed for {len(results)} brokers")
        return results
    
    def compare_broker_costs(
        self,
        request: TransactionRequest,
        broker_configs: List[BrokerConfiguration],
        market_conditions: Optional[MarketConditions] = None
    ) -> BrokerComparisonResult:
        """
        Compare costs across different brokers and identify best options.
        
        Args:
            request: Transaction request
            broker_configs: List of broker configurations to compare
            market_conditions: Current market conditions
            
        Returns:
            Broker comparison result with recommendations
        """
        logger.info(f"Comparing broker costs for {request.symbol}")
        
        # Get cost analysis for all brokers
        cost_analyses = self.generate_comprehensive_cost_breakdown(
            request, broker_configs, market_conditions
        )
        
        if not cost_analyses:
            raise ValueError("No valid cost analyses available for comparison")
        
        # Create comparison matrix
        comparison_data = []
        for broker_name, analysis in cost_analyses.items():
            breakdown = analysis.cost_breakdown
            
            comparison_data.append({
                'Broker': broker_name,
                'Total Cost': float(breakdown.total_cost),
                'Commission': float(breakdown.commission),
                'Regulatory Fees': float(breakdown.regulatory_fees),
                'Exchange Fees': float(breakdown.exchange_fees),
                'Market Impact': float(breakdown.market_impact_cost),
                'Spread Cost': float(breakdown.bid_ask_spread_cost),
                'Timing Cost': float(breakdown.timing_cost),
                'Efficiency Score': analysis.cost_efficiency_score,
                'Return Impact (%)': analysis.cost_impact_on_returns * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best broker (lowest total cost)
        best_broker_idx = comparison_df['Total Cost'].idxmin()
        best_broker = comparison_df.loc[best_broker_idx, 'Broker']
        best_cost = Decimal(str(comparison_df.loc[best_broker_idx, 'Total Cost']))
        
        # Calculate cost differences and potential savings
        cost_differences = {}
        total_potential_savings = Decimal('0')
        
        for _, row in comparison_df.iterrows():
            if row['Broker'] != best_broker:
                cost_diff = Decimal(str(row['Total Cost'])) - best_cost
                cost_differences[row['Broker']] = cost_diff
                total_potential_savings += cost_diff
        
        # Generate comparison recommendations
        recommendations = self._generate_broker_recommendations(
            comparison_df, best_broker, cost_differences
        )
        
        result = BrokerComparisonResult(
            best_broker=best_broker,
            cost_differences=cost_differences,
            potential_savings=total_potential_savings,
            comparison_matrix=comparison_df,
            recommendations=recommendations
        )
        
        logger.info(f"Broker comparison completed. Best broker: {best_broker}")
        return result
    
    def analyze_cost_impact_on_performance(
        self,
        transactions: List[TransactionRequest],
        broker_config: BrokerConfiguration,
        expected_returns: Optional[np.ndarray] = None,
        time_horizon_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze how transaction costs impact portfolio performance over time.
        
        Args:
            transactions: List of historical transactions
            broker_config: Broker configuration for cost calculation
            expected_returns: Expected returns array (if available)
            time_horizon_days: Analysis time horizon in days
            
        Returns:
            Cost impact analysis results
        """
        logger.info(f"Analyzing cost impact for {len(transactions)} transactions")
        
        total_costs = Decimal('0')
        total_notional = Decimal('0')
        cost_by_type = {
            'commission': Decimal('0'),
            'market_impact': Decimal('0'),
            'spreads': Decimal('0'),
            'fees': Decimal('0')
        }
        
        daily_costs = {}
        
        for transaction in transactions:
            try:
                # Calculate costs for this transaction
                cost_result = self.cost_aggregator.calculate_total_cost(
                    transaction, broker_config
                )
                
                breakdown = cost_result.cost_breakdown
                transaction_cost = breakdown.total_cost
                
                # Accumulate totals
                total_costs += transaction_cost
                total_notional += transaction.notional_value
                
                # Categorize costs
                cost_by_type['commission'] += breakdown.commission
                cost_by_type['market_impact'] += breakdown.market_impact_cost
                cost_by_type['spreads'] += breakdown.bid_ask_spread_cost
                cost_by_type['fees'] += (breakdown.regulatory_fees + 
                                       breakdown.exchange_fees + 
                                       breakdown.platform_fees)
                
                # Track daily costs
                date_key = transaction.timestamp.date()
                if date_key not in daily_costs:
                    daily_costs[date_key] = Decimal('0')
                daily_costs[date_key] += transaction_cost
                
            except Exception as e:
                logger.warning(f"Failed to calculate costs for transaction {transaction.symbol}: {e}")
                continue
        
        # Calculate impact metrics
        total_cost_bps = (total_costs / total_notional * 10000) if total_notional > 0 else Decimal('0')
        
        # Estimate annual cost impact
        days_analyzed = len(daily_costs)
        if days_analyzed > 0:
            avg_daily_cost = total_costs / days_analyzed
            estimated_annual_cost = avg_daily_cost * 252  # Trading days per year
        else:
            estimated_annual_cost = Decimal('0')
        
        # Calculate cost efficiency metrics
        cost_efficiency_score = self._calculate_portfolio_cost_efficiency(
            total_cost_bps, len(transactions)
        )
        
        # Generate recommendations
        recommendations = self._generate_performance_impact_recommendations(
            total_cost_bps, cost_by_type, total_notional
        )
        
        return {
            'total_costs': float(total_costs),
            'total_notional': float(total_notional),
            'total_cost_bps': float(total_cost_bps),
            'cost_breakdown': {k: float(v) for k, v in cost_by_type.items()},
            'daily_costs': {str(k): float(v) for k, v in daily_costs.items()},
            'estimated_annual_cost': float(estimated_annual_cost),
            'cost_efficiency_score': cost_efficiency_score,
            'transactions_analyzed': len(transactions),
            'analysis_period_days': days_analyzed,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_cost_trend_analysis(
        self,
        historical_transactions: List[TransactionRequest],
        broker_config: BrokerConfiguration,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate cost trend analysis and forecasting.
        
        Args:
            historical_transactions: Historical transaction data
            broker_config: Broker configuration
            period_days: Analysis period in days
            
        Returns:
            Cost trend analysis results
        """
        logger.info(f"Generating cost trend analysis for {len(historical_transactions)} transactions")
        
        # Group transactions by time periods
        transactions_by_period = {}
        current_date = datetime.now().date()
        
        for i in range(period_days):
            period_date = current_date - timedelta(days=i)
            transactions_by_period[period_date] = []
        
        # Categorize transactions by date
        for transaction in historical_transactions:
            transaction_date = transaction.timestamp.date()
            if transaction_date in transactions_by_period:
                transactions_by_period[transaction_date].append(transaction)
        
        # Calculate daily cost metrics
        daily_metrics = []
        for date, transactions in transactions_by_period.items():
            if not transactions:
                continue
                
            daily_cost = Decimal('0')
            daily_notional = Decimal('0')
            
            for transaction in transactions:
                try:
                    cost_result = self.cost_aggregator.calculate_total_cost(
                        transaction, broker_config
                    )
                    daily_cost += cost_result.cost_breakdown.total_cost
                    daily_notional += transaction.notional_value
                except Exception as e:
                    logger.warning(f"Cost calculation failed for {transaction.symbol}: {e}")
                    continue
            
            if daily_notional > 0:
                daily_cost_bps = (daily_cost / daily_notional) * 10000
                daily_metrics.append({
                    'date': date,
                    'total_cost': float(daily_cost),
                    'total_notional': float(daily_notional),
                    'cost_bps': float(daily_cost_bps),
                    'transaction_count': len(transactions)
                })
        
        if not daily_metrics:
            return {
                'trend_analysis': 'Insufficient data for trend analysis',
                'forecast': None,
                'recommendations': ['Collect more transaction data for meaningful analysis']
            }
        
        # Create DataFrame for analysis
        df = pd.DataFrame(daily_metrics)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate trends
        cost_trend = self._calculate_cost_trend(df)
        volatility_metrics = self._calculate_cost_volatility(df)
        
        # Generate forecast
        forecast = self._generate_cost_forecast(df)
        
        # Generate trend recommendations
        recommendations = self._generate_trend_recommendations(cost_trend, volatility_metrics)
        
        return {
            'trend_analysis': {
                'average_daily_cost_bps': df['cost_bps'].mean(),
                'cost_trend_direction': cost_trend['direction'],
                'cost_trend_strength': cost_trend['strength'],
                'cost_volatility': volatility_metrics['volatility'],
                'cost_stability_score': volatility_metrics['stability_score']
            },
            'daily_metrics': daily_metrics,
            'forecast': forecast,
            'recommendations': recommendations,
            'analysis_metadata': {
                'period_days': period_days,
                'data_points': len(daily_metrics),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_cost_efficiency(self, breakdown: TransactionCostBreakdown, notional_value: Decimal) -> float:
        """Calculate cost efficiency score (0-100, higher is better)."""
        if notional_value <= 0:
            return 0.0
        
        cost_bps = (breakdown.total_cost / notional_value) * 10000
        
        # Score based on efficiency benchmarks
        if cost_bps <= self.efficiency_benchmarks['excellent']:
            return 100.0
        elif cost_bps <= self.efficiency_benchmarks['good']:
            return 80.0
        elif cost_bps <= self.efficiency_benchmarks['average']:
            return 60.0
        else:
            # Decreasing score for higher costs
            return max(40.0 - float(cost_bps - self.efficiency_benchmarks['average']), 0.0)
    
    def _calculate_return_impact(self, breakdown: TransactionCostBreakdown, notional_value: Decimal) -> float:
        """Calculate cost impact on returns as percentage."""
        if notional_value <= 0:
            return 0.0
        
        return float(breakdown.total_cost / notional_value)
    
    def _generate_cost_recommendations(
        self, 
        breakdown: TransactionCostBreakdown, 
        request: TransactionRequest,
        efficiency_score: float
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        cost_bps = (breakdown.total_cost / request.notional_value) * 10000
        
        # High cost warning
        if cost_bps > self.basis_points_threshold:
            recommendations.append(f"High transaction cost: {float(cost_bps):.1f} bps. Consider optimizing order size or timing.")
        
        # Market impact recommendations
        if breakdown.market_impact_cost > breakdown.total_cost * Decimal('0.3'):
            recommendations.append("Market impact is significant. Consider breaking large orders into smaller sizes.")
        
        # Spread cost recommendations  
        if breakdown.bid_ask_spread_cost > breakdown.total_cost * Decimal('0.4'):
            recommendations.append("Bid-ask spread cost is high. Consider using limit orders during high volatility periods.")
        
        # Commission optimization
        if breakdown.commission > breakdown.total_cost * Decimal('0.5'):
            recommendations.append("Commission costs are dominant. Consider negotiating better rates or switching brokers.")
        
        # Efficiency-based recommendations
        if efficiency_score < 60:
            recommendations.append("Low cost efficiency. Review trading strategy and broker selection.")
        elif efficiency_score > 90:
            recommendations.append("Excellent cost efficiency. Current strategy is well-optimized.")
        
        # Timing recommendations
        if request.market_timing.name != 'MARKET_HOURS':
            recommendations.append("Extended hours trading may increase costs. Consider timing trades during regular market hours.")
        
        return recommendations
    
    def _generate_broker_recommendations(
        self,
        comparison_df: pd.DataFrame,
        best_broker: str,
        cost_differences: Dict[str, Decimal]
    ) -> List[str]:
        """Generate broker comparison recommendations."""
        recommendations = []
        
        # Best broker recommendation
        best_row = comparison_df[comparison_df['Broker'] == best_broker].iloc[0]
        recommendations.append(f"Recommended broker: {best_broker} with total cost of ${best_row['Total Cost']:.4f}")
        
        # Significant cost differences
        for broker, cost_diff in cost_differences.items():
            if cost_diff > Decimal('10.0'):  # $10 difference threshold
                recommendations.append(f"Avoid {broker}: ${float(cost_diff):.2f} more expensive than best option")
        
        # Efficiency insights
        high_efficiency_brokers = comparison_df[comparison_df['Efficiency Score'] > 80]['Broker'].tolist()
        if high_efficiency_brokers:
            recommendations.append(f"High efficiency brokers: {', '.join(high_efficiency_brokers)}")
        
        # Market impact considerations
        low_impact_brokers = comparison_df[comparison_df['Market Impact'] < comparison_df['Market Impact'].median()]['Broker'].tolist()
        if low_impact_brokers and len(low_impact_brokers) < len(comparison_df):
            recommendations.append(f"Low market impact brokers: {', '.join(low_impact_brokers)}")
        
        return recommendations
    
    def _calculate_portfolio_cost_efficiency(self, total_cost_bps: Decimal, transaction_count: int) -> float:
        """Calculate portfolio-level cost efficiency score."""
        base_score = 100.0
        
        # Penalize high costs
        if total_cost_bps > 20:
            base_score -= float(total_cost_bps - 20) * 2
        
        # Penalize low transaction efficiency
        if transaction_count > 0:
            avg_cost_per_transaction = total_cost_bps / transaction_count
            if avg_cost_per_transaction > 5:
                base_score -= float(avg_cost_per_transaction - 5) * 3
        
        return max(base_score, 0.0)
    
    def _generate_performance_impact_recommendations(
        self,
        total_cost_bps: Decimal,
        cost_by_type: Dict[str, Decimal],
        total_notional: Decimal
    ) -> List[str]:
        """Generate performance impact recommendations."""
        recommendations = []
        
        # Overall cost level recommendations
        if total_cost_bps > 25:
            recommendations.append("High overall trading costs. Significant performance drag expected.")
        elif total_cost_bps < 10:
            recommendations.append("Low trading costs. Minimal performance impact.")
        
        # Cost component recommendations
        total_costs = sum(cost_by_type.values())
        if total_costs > 0:
            for cost_type, amount in cost_by_type.items():
                percentage = (amount / total_costs) * 100
                if percentage > 40:
                    recommendations.append(f"{cost_type.title()} costs are {percentage:.1f}% of total. Focus optimization efforts here.")
        
        # Volume-based recommendations
        if total_notional > 1000000:  # $1M+
            recommendations.append("High volume trading detected. Negotiate institutional rates and optimize execution.")
        
        return recommendations
    
    def _calculate_cost_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate cost trend direction and strength."""
        if len(df) < 2:
            return {'direction': 'insufficient_data', 'strength': 0.0}
        
        # Simple linear trend
        x = np.arange(len(df))
        y = df['cost_bps'].values
        
        # Calculate correlation coefficient as trend strength
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        
        # Determine direction
        if correlation > 0.1:
            direction = 'increasing'
        elif correlation < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'strength': abs(correlation),
            'correlation': correlation
        }
    
    def _calculate_cost_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate cost volatility metrics."""
        if len(df) < 2:
            return {'volatility': 0.0, 'stability_score': 100.0}
        
        cost_volatility = df['cost_bps'].std()
        
        # Stability score (inverse of volatility, 0-100 scale)
        # Lower volatility = higher stability
        stability_score = max(100.0 - cost_volatility * 5, 0.0)
        
        return {
            'volatility': cost_volatility,
            'stability_score': stability_score
        }
    
    def _generate_cost_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate simple cost forecast."""
        if len(df) < 3:
            return {'forecast_available': False, 'reason': 'Insufficient data'}
        
        # Simple moving average forecast
        recent_costs = df['cost_bps'].tail(7).mean()  # Last 7 days average
        overall_avg = df['cost_bps'].mean()
        
        # Trend-based adjustment
        trend = self._calculate_cost_trend(df)
        trend_adjustment = 0.0
        
        if trend['direction'] == 'increasing':
            trend_adjustment = recent_costs * 0.1 * trend['strength']
        elif trend['direction'] == 'decreasing':
            trend_adjustment = -recent_costs * 0.1 * trend['strength']
        
        forecast_cost_bps = recent_costs + trend_adjustment
        
        return {
            'forecast_available': True,
            'forecast_cost_bps': forecast_cost_bps,
            'confidence': min(trend['strength'] * 100, 95.0),
            'method': 'trend_adjusted_moving_average',
            'forecast_horizon_days': 7
        }
    
    def _generate_trend_recommendations(
        self,
        trend: Dict[str, Any],
        volatility: Dict[str, float]
    ) -> List[str]:
        """Generate trend-based recommendations."""
        recommendations = []
        
        # Trend recommendations
        if trend['direction'] == 'increasing' and trend['strength'] > 0.3:
            recommendations.append("Costs are trending upward. Review and optimize trading strategy.")
        elif trend['direction'] == 'decreasing' and trend['strength'] > 0.3:
            recommendations.append("Costs are trending downward. Current optimization efforts are effective.")
        
        # Volatility recommendations
        if volatility['stability_score'] < 50:
            recommendations.append("High cost volatility detected. Standardize trading processes for consistency.")
        elif volatility['stability_score'] > 80:
            recommendations.append("Cost structure is stable. Good process control.")
        
        # Combined insights
        if trend['direction'] == 'stable' and volatility['stability_score'] > 70:
            recommendations.append("Costs are stable and predictable. Maintain current approach.")
        
        return recommendations


logger.info("Cost reporter module loaded successfully")