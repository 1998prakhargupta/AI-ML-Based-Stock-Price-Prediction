"""
Cost Chart Factory
==================

Factory pattern for creating different types of cost charts
based on requirements and data availability.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum

# Import chart components
from .breakdown_charts import CostBreakdownCharts
from .impact_charts import CostImpactCharts
from .comparison_charts import CostComparisonCharts
from .dashboard import CostDashboard

# Import utilities
from src.utils.config_manager import Config

logger = logging.getLogger(__name__)

class ChartType(Enum):
    """Supported chart types."""
    BREAKDOWN_PIE = "breakdown_pie"
    BREAKDOWN_STACKED = "breakdown_stacked"
    BREAKDOWN_HEATMAP = "breakdown_heatmap"
    BREAKDOWN_WATERFALL = "breakdown_waterfall"
    BREAKDOWN_DISTRIBUTION = "breakdown_distribution"
    
    IMPACT_SCATTER = "impact_scatter"
    IMPACT_EFFICIENCY = "impact_efficiency"
    IMPACT_ATTRIBUTION = "impact_attribution"
    IMPACT_BREAKEVEN = "impact_breakeven"
    
    COMPARISON_BROKER = "comparison_broker"
    COMPARISON_TREND = "comparison_trend"
    COMPARISON_BENCHMARK = "comparison_benchmark"
    COMPARISON_SCENARIO = "comparison_scenario"
    
    DASHBOARD_COMPREHENSIVE = "dashboard_comprehensive"
    DASHBOARD_EXECUTIVE = "dashboard_executive"

class CostChartFactory:
    """
    Factory for creating cost visualization charts based on
    chart type and available data.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize chart factory.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize chart components
        self.breakdown_charts = CostBreakdownCharts(config)
        self.impact_charts = CostImpactCharts(config)
        self.comparison_charts = CostComparisonCharts(config)
        self.dashboard = CostDashboard(config)
        
        logger.info("Cost chart factory initialized")
    
    def create_chart(
        self,
        chart_type: Union[ChartType, str],
        data: Dict[str, Any],
        save_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create a chart of the specified type.
        
        Args:
            chart_type: Type of chart to create
            data: Data for chart creation
            save_name: Optional name for saved chart
            **kwargs: Additional chart-specific parameters
            
        Returns:
            Path to created chart or None if creation failed
        """
        try:
            # Convert string to enum if needed
            if isinstance(chart_type, str):
                chart_type = ChartType(chart_type)
            
            # Generate default save name if not provided
            if not save_name:
                save_name = f"cost_chart_{chart_type.value}"
            
            # Route to appropriate chart creation method
            if chart_type == ChartType.BREAKDOWN_PIE:
                return self._create_breakdown_pie(data, save_name, **kwargs)
            elif chart_type == ChartType.BREAKDOWN_STACKED:
                return self._create_breakdown_stacked(data, save_name, **kwargs)
            elif chart_type == ChartType.BREAKDOWN_HEATMAP:
                return self._create_breakdown_heatmap(data, save_name, **kwargs)
            elif chart_type == ChartType.BREAKDOWN_WATERFALL:
                return self._create_breakdown_waterfall(data, save_name, **kwargs)
            elif chart_type == ChartType.BREAKDOWN_DISTRIBUTION:
                return self._create_breakdown_distribution(data, save_name, **kwargs)
            elif chart_type == ChartType.IMPACT_SCATTER:
                return self._create_impact_scatter(data, save_name, **kwargs)
            elif chart_type == ChartType.IMPACT_EFFICIENCY:
                return self._create_impact_efficiency(data, save_name, **kwargs)
            elif chart_type == ChartType.IMPACT_ATTRIBUTION:
                return self._create_impact_attribution(data, save_name, **kwargs)
            elif chart_type == ChartType.IMPACT_BREAKEVEN:
                return self._create_impact_breakeven(data, save_name, **kwargs)
            elif chart_type == ChartType.COMPARISON_BROKER:
                return self._create_comparison_broker(data, save_name, **kwargs)
            elif chart_type == ChartType.COMPARISON_TREND:
                return self._create_comparison_trend(data, save_name, **kwargs)
            elif chart_type == ChartType.COMPARISON_BENCHMARK:
                return self._create_comparison_benchmark(data, save_name, **kwargs)
            elif chart_type == ChartType.COMPARISON_SCENARIO:
                return self._create_comparison_scenario(data, save_name, **kwargs)
            elif chart_type == ChartType.DASHBOARD_COMPREHENSIVE:
                return self._create_dashboard_comprehensive(data, save_name, **kwargs)
            elif chart_type == ChartType.DASHBOARD_EXECUTIVE:
                return self._create_dashboard_executive(data, save_name, **kwargs)
            else:
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating chart {chart_type}: {e}")
            return None
    
    def create_multiple_charts(
        self,
        chart_types: List[Union[ChartType, str]],
        data: Dict[str, Any],
        base_save_name: str = "cost_analysis"
    ) -> Dict[str, Optional[str]]:
        """
        Create multiple charts from the same data.
        
        Args:
            chart_types: List of chart types to create
            data: Data for chart creation
            base_save_name: Base name for saved charts
            
        Returns:
            Dictionary mapping chart types to file paths
        """
        chart_paths = {}
        
        for i, chart_type in enumerate(chart_types):
            save_name = f"{base_save_name}_{i+1}"
            path = self.create_chart(chart_type, data, save_name)
            chart_paths[str(chart_type)] = path
        
        return chart_paths
    
    def get_recommended_charts(self, data: Dict[str, Any]) -> List[ChartType]:
        """
        Get recommended chart types based on available data.
        
        Args:
            data: Available data for analysis
            
        Returns:
            List of recommended chart types
        """
        recommended = []
        
        # Check what data is available
        has_cost_breakdowns = bool(data.get('cost_breakdowns'))
        has_transactions = bool(data.get('transactions'))
        has_returns = bool(data.get('returns'))
        has_broker_comparison = bool(data.get('broker_comparison'))
        has_historical_data = bool(data.get('historical_data'))
        
        # Recommend charts based on available data
        if has_cost_breakdowns:
            recommended.extend([
                ChartType.BREAKDOWN_PIE,
                ChartType.BREAKDOWN_DISTRIBUTION
            ])
        
        if has_cost_breakdowns and has_transactions:
            recommended.append(ChartType.BREAKDOWN_STACKED)
            
            if has_returns:
                recommended.extend([
                    ChartType.IMPACT_SCATTER,
                    ChartType.IMPACT_ATTRIBUTION
                ])
        
        if has_broker_comparison:
            recommended.append(ChartType.COMPARISON_BROKER)
        
        if has_historical_data:
            recommended.append(ChartType.COMPARISON_TREND)
        
        # Always recommend comprehensive dashboard if we have basic data
        if has_cost_breakdowns and has_transactions:
            recommended.append(ChartType.DASHBOARD_COMPREHENSIVE)
        
        return recommended
    
    # Private methods for creating specific chart types
    def _create_breakdown_pie(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create breakdown pie chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        if not cost_breakdowns:
            return None
        return self.breakdown_charts.create_cost_component_pie_chart(cost_breakdowns, save_name)
    
    def _create_breakdown_stacked(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create breakdown stacked bar chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        transactions = data.get('transactions', [])
        if not cost_breakdowns or not transactions:
            return None
        
        labels = [f"T{i+1}" for i in range(len(transactions))]
        return self.breakdown_charts.create_cost_component_stacked_bar(cost_breakdowns, labels, save_name)
    
    def _create_breakdown_heatmap(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create breakdown heatmap."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        transactions = data.get('transactions', [])
        if not cost_breakdowns or not transactions:
            return None
        
        timestamps = [t.timestamp for t in transactions]
        return self.breakdown_charts.create_cost_heatmap_by_time_and_component(cost_breakdowns, timestamps, save_name)
    
    def _create_breakdown_waterfall(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create breakdown waterfall chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        if not cost_breakdowns:
            return None
        return self.breakdown_charts.create_cost_waterfall_chart(cost_breakdowns[0], save_name)
    
    def _create_breakdown_distribution(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create breakdown distribution chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        transactions = data.get('transactions', [])
        if not cost_breakdowns or not transactions:
            return None
        
        notional_values = [float(t.notional_value) for t in transactions]
        return self.breakdown_charts.create_cost_distribution_histogram(cost_breakdowns, notional_values, save_name)
    
    def _create_impact_scatter(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create impact scatter plot."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        returns = data.get('returns', [])
        transactions = data.get('transactions', [])
        if not cost_breakdowns or not returns or not transactions:
            return None
        
        notional_values = [float(t.notional_value) for t in transactions]
        return self.impact_charts.create_cost_vs_returns_scatter(cost_breakdowns, returns, notional_values, save_name)
    
    def _create_impact_efficiency(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create impact efficiency chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        transactions = data.get('transactions', [])
        if not cost_breakdowns or not transactions:
            return None
        
        timestamps = [t.timestamp for t in transactions]
        notional_values = [float(t.notional_value) for t in transactions]
        return self.impact_charts.create_cost_efficiency_over_time(cost_breakdowns, timestamps, notional_values, save_name)
    
    def _create_impact_attribution(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create impact attribution chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        transactions = data.get('transactions', [])
        portfolio_value = data.get('portfolio_value', 1000000)  # Default $1M
        if not cost_breakdowns or not transactions:
            return None
        
        return self.impact_charts.create_cost_impact_attribution(transactions, cost_breakdowns, portfolio_value, save_name)
    
    def _create_impact_breakeven(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create impact breakeven chart."""
        cost_breakdowns = data.get('cost_breakdowns', [])
        expected_returns = data.get('expected_returns', [])
        holding_periods = data.get('holding_periods', [])
        if not cost_breakdowns or not expected_returns or not holding_periods:
            return None
        
        return self.impact_charts.create_cost_breakeven_analysis(cost_breakdowns, expected_returns, holding_periods, save_name)
    
    def _create_comparison_broker(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create broker comparison chart."""
        broker_comparison = data.get('broker_comparison')
        if not broker_comparison:
            return None
        return self.comparison_charts.create_broker_comparison_chart(broker_comparison, save_name)
    
    def _create_comparison_trend(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create trend comparison chart."""
        historical_data = data.get('historical_data', {})
        if not historical_data:
            return None
        return self.comparison_charts.create_cost_trend_comparison(historical_data, save_name)
    
    def _create_comparison_benchmark(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create benchmark comparison chart."""
        actual_costs = data.get('actual_costs', {})
        benchmark_costs = data.get('benchmark_costs', {})
        if not actual_costs or not benchmark_costs:
            return None
        return self.comparison_charts.create_cost_efficiency_benchmark(actual_costs, benchmark_costs, save_name)
    
    def _create_comparison_scenario(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create scenario comparison chart."""
        scenarios = data.get('scenarios', {})
        base_scenario = data.get('base_scenario', '')
        if not scenarios or not base_scenario:
            return None
        return self.comparison_charts.create_cost_scenario_analysis(scenarios, base_scenario, save_name)
    
    def _create_dashboard_comprehensive(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create comprehensive dashboard."""
        charts = self.dashboard.create_comprehensive_dashboard(data, save_name)
        return str(charts) if charts else None  # Return string representation
    
    def _create_dashboard_executive(self, data: Dict[str, Any], save_name: str, **kwargs) -> Optional[str]:
        """Create executive dashboard."""
        summary_data = data.get('summary_data', {})
        return self.dashboard.create_executive_summary_dashboard(summary_data, save_name)


logger.info("Cost chart factory module loaded successfully")