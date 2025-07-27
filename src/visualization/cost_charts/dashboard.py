"""
Cost Dashboard
==============

Interactive cost dashboard combining multiple visualization components
for comprehensive cost analysis and reporting.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from src.utils.file_management_utils import SafeFileManager
from src.utils.config_manager import Config

# Import other chart components
from .breakdown_charts import CostBreakdownCharts
from .impact_charts import CostImpactCharts  
from .comparison_charts import CostComparisonCharts

logger = logging.getLogger(__name__)

class CostDashboard:
    """
    Interactive cost dashboard that combines multiple chart types
    into comprehensive cost analysis dashboards.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cost dashboard.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        
        # Initialize chart components
        self.breakdown_charts = CostBreakdownCharts(config)
        self.impact_charts = CostImpactCharts(config)
        self.comparison_charts = CostComparisonCharts(config)
        
        logger.info("Cost dashboard initialized")
    
    def create_comprehensive_dashboard(
        self,
        cost_data: Dict[str, Any],
        save_name: str = "cost_dashboard"
    ) -> Dict[str, Optional[str]]:
        """
        Create comprehensive cost dashboard with multiple chart types.
        
        Args:
            cost_data: Cost analysis data
            save_name: Base name for saved charts
            
        Returns:
            Dictionary of chart paths
        """
        logger.info("Creating comprehensive cost dashboard")
        
        chart_paths = {}
        
        try:
            # Extract data components
            cost_breakdowns = cost_data.get('cost_breakdowns', [])
            transactions = cost_data.get('transactions', [])
            
            if not cost_breakdowns or not transactions:
                logger.warning("Insufficient data for dashboard creation")
                return chart_paths
            
            # Generate breakdown charts
            if cost_breakdowns:
                pie_path = self.breakdown_charts.create_cost_component_pie_chart(
                    cost_breakdowns, f"{save_name}_breakdown_pie"
                )
                chart_paths['breakdown_pie'] = pie_path
                
                # Distribution chart
                notional_values = [float(t.notional_value) for t in transactions]
                dist_path = self.breakdown_charts.create_cost_distribution_histogram(
                    cost_breakdowns, notional_values, f"{save_name}_distribution"
                )
                chart_paths['distribution'] = dist_path
            
            # Generate impact charts
            returns = cost_data.get('returns', [])
            if returns and len(returns) == len(cost_breakdowns):
                impact_path = self.impact_charts.create_cost_vs_returns_scatter(
                    cost_breakdowns, returns, notional_values, f"{save_name}_impact"
                )
                chart_paths['impact_analysis'] = impact_path
            
            # Time series efficiency
            timestamps = [t.timestamp for t in transactions]
            efficiency_path = self.impact_charts.create_cost_efficiency_over_time(
                cost_breakdowns, timestamps, notional_values, f"{save_name}_efficiency"
            )
            chart_paths['efficiency_trend'] = efficiency_path
            
            # Broker comparison (if data available)
            broker_comparison = cost_data.get('broker_comparison')
            if broker_comparison:
                comparison_path = self.comparison_charts.create_broker_comparison_chart(
                    broker_comparison, f"{save_name}_broker_comparison"
                )
                chart_paths['broker_comparison'] = comparison_path
            
            logger.info(f"Dashboard created with {len(chart_paths)} charts")
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
        
        return chart_paths
    
    def create_executive_summary_dashboard(
        self,
        summary_data: Dict[str, Any],
        save_name: str = "executive_dashboard"
    ) -> Optional[str]:
        """
        Create executive-level dashboard with key metrics.
        
        Args:
            summary_data: Executive summary data
            save_name: Name for saved dashboard
            
        Returns:
            Path to saved dashboard or None
        """
        logger.info("Creating executive summary dashboard")
        
        try:
            # This would create a high-level executive dashboard
            # For now, return a placeholder
            logger.info("Executive dashboard created (placeholder)")
            return None
            
        except Exception as e:
            logger.error(f"Error creating executive dashboard: {e}")
            return None


logger.info("Cost dashboard module loaded successfully")