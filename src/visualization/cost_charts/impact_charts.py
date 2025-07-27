"""
Cost Impact Analysis Charts
===========================

Specialized charts for analyzing the impact of transaction costs on portfolio
performance, returns, and trading efficiency.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Plotting dependencies with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    GridSpec = None
    mdates = None
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plotting functions will be disabled")

# Import utilities
from src.utils.file_management_utils import SafeFileManager
from src.utils.config_manager import Config

# Import cost models
from src.trading.transaction_costs.models import TransactionCostBreakdown, TransactionRequest

logger = logging.getLogger(__name__)

class CostImpactCharts:
    """
    Specialized charts for analyzing cost impact on portfolio performance,
    returns attribution, and trading efficiency metrics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cost impact charts.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        
        # Chart styling
        self.colors = {
            'positive': '#27AE60',    # Green
            'negative': '#E74C3C',    # Red
            'neutral': '#3498DB',     # Blue
            'warning': '#F39C12',     # Orange
            'info': '#9B59B6'         # Purple
        }
        
        self.figure_size = (12, 8)
        self.dpi = 300
        
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        logger.info("Cost impact charts initialized")
    
    def create_cost_vs_returns_scatter(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        returns: List[float],
        notional_values: List[float],
        save_name: str = "cost_vs_returns"
    ) -> Optional[str]:
        """
        Create scatter plot showing relationship between costs and returns.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            returns: Corresponding returns (as percentages)
            notional_values: Corresponding notional values
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for scatter plot creation")
            return None
        
        try:
            # Calculate cost in basis points
            cost_bps = []
            for breakdown, notional in zip(cost_breakdowns, notional_values):
                if notional > 0:
                    bps = (float(breakdown.total_cost) / notional) * 10000
                    cost_bps.append(bps)
                else:
                    cost_bps.append(0)
            
            if len(cost_bps) != len(returns):
                logger.error("Mismatch between cost and return data lengths")
                return None
            
            # Create scatter plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Scatter plot: Cost vs Returns
            colors = [self.colors['positive'] if r > 0 else self.colors['negative'] for r in returns]
            scatter = ax1.scatter(cost_bps, returns, c=colors, alpha=0.6, s=50, edgecolors='black')
            
            # Add trend line
            if len(cost_bps) > 1:
                z = np.polyfit(cost_bps, returns, 1)
                p = np.poly1d(z)
                ax1.plot(cost_bps, p(cost_bps), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
                
                # Calculate correlation
                correlation = np.corrcoef(cost_bps, returns)[0, 1]
                ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.set_xlabel('Transaction Cost (basis points)', fontsize=12)
            ax1.set_ylabel('Return (%)', fontsize=12)
            ax1.set_title('Transaction Cost vs Return Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Histogram of cost-adjusted returns
            cost_adjusted_returns = [ret - (cost/10000)*100 for ret, cost in zip(returns, cost_bps)]
            
            ax2.hist([returns, cost_adjusted_returns], bins=20, alpha=0.7, 
                    label=['Gross Returns', 'Net Returns (Cost-Adjusted)'],
                    color=[self.colors['neutral'], self.colors['warning']])
            
            ax2.set_xlabel('Return (%)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Impact of Costs on Return Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            gross_mean = np.mean(returns)
            net_mean = np.mean(cost_adjusted_returns)
            impact = gross_mean - net_mean
            
            stats_text = f"""Return Impact:
Gross Mean: {gross_mean:.2f}%
Net Mean: {net_mean:.2f}%
Cost Impact: -{impact:.2f}%"""
            
            ax2.text(0.65, 0.85, stats_text, transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_scatter")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost vs returns scatter plot: {e}")
            return None
    
    def create_cost_efficiency_over_time(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        timestamps: List[datetime],
        notional_values: List[float],
        save_name: str = "cost_efficiency_time"
    ) -> Optional[str]:
        """
        Create chart showing cost efficiency trends over time.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            timestamps: Corresponding timestamps
            notional_values: Corresponding notional values
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for time series chart")
            return None
        
        try:
            # Prepare data
            data = []
            for breakdown, timestamp, notional in zip(cost_breakdowns, timestamps, notional_values):
                if notional > 0:
                    cost_bps = (float(breakdown.total_cost) / notional) * 10000
                    data.append({
                        'timestamp': timestamp,
                        'cost_bps': cost_bps,
                        'total_cost': float(breakdown.total_cost),
                        'notional': notional
                    })
            
            if not data:
                logger.warning("No valid data for time series chart")
                return None
            
            df = pd.DataFrame(data)
            df['date'] = df['timestamp'].dt.date
            
            # Create daily aggregations
            daily_stats = df.groupby('date').agg({
                'cost_bps': ['mean', 'std', 'min', 'max'],
                'total_cost': 'sum',
                'notional': 'sum'
            }).reset_index()
            
            daily_stats.columns = ['date', 'mean_cost_bps', 'std_cost_bps', 'min_cost_bps', 
                                  'max_cost_bps', 'total_cost', 'total_notional']
            
            # Create chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Cost efficiency over time
            ax1.plot(daily_stats['date'], daily_stats['mean_cost_bps'], 
                    marker='o', linewidth=2, color=self.colors['neutral'], label='Daily Average')
            ax1.fill_between(daily_stats['date'], 
                           daily_stats['mean_cost_bps'] - daily_stats['std_cost_bps'],
                           daily_stats['mean_cost_bps'] + daily_stats['std_cost_bps'],
                           alpha=0.3, color=self.colors['neutral'], label='±1 Std Dev')
            
            # Add efficiency benchmarks
            ax1.axhline(y=10, color=self.colors['positive'], linestyle='--', alpha=0.7, label='Good (10 bps)')
            ax1.axhline(y=20, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Average (20 bps)')
            ax1.axhline(y=30, color=self.colors['negative'], linestyle='--', alpha=0.7, label='Poor (30 bps)')
            
            ax1.set_title('Cost Efficiency Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cost (basis points)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Cost volatility
            ax2.bar(daily_stats['date'], daily_stats['std_cost_bps'], 
                   color=self.colors['warning'], alpha=0.7)
            ax2.set_title('Daily Cost Volatility', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Cost Std Dev (bps)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Cost range (min-max)
            ax3.plot(daily_stats['date'], daily_stats['min_cost_bps'], 
                    marker='v', color=self.colors['positive'], label='Daily Min')
            ax3.plot(daily_stats['date'], daily_stats['max_cost_bps'], 
                    marker='^', color=self.colors['negative'], label='Daily Max')
            ax3.fill_between(daily_stats['date'], daily_stats['min_cost_bps'], 
                           daily_stats['max_cost_bps'], alpha=0.2)
            
            ax3.set_title('Daily Cost Range', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Cost (basis points)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Volume vs cost relationship
            scatter = ax4.scatter(daily_stats['total_notional'] / 1000, daily_stats['mean_cost_bps'],
                                c=daily_stats['mean_cost_bps'], cmap='RdYlGn_r', alpha=0.7, s=60)
            
            # Add trend line
            if len(daily_stats) > 1:
                x_vol = daily_stats['total_notional'] / 1000
                y_cost = daily_stats['mean_cost_bps']
                z = np.polyfit(x_vol, y_cost, 1)
                p = np.poly1d(z)
                ax4.plot(x_vol, p(x_vol), "r--", alpha=0.8, linewidth=2)
                
                correlation = np.corrcoef(x_vol, y_cost)[0, 1]
                ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.colorbar(scatter, ax=ax4, label='Cost (bps)')
            ax4.set_title('Volume vs Cost Efficiency', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Daily Volume ($000s)', fontsize=12)
            ax4.set_ylabel('Average Cost (bps)', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_time_series")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost efficiency time series: {e}")
            return None
    
    def create_cost_impact_attribution(
        self,
        transactions: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown],
        portfolio_value: float,
        save_name: str = "cost_impact_attribution"
    ) -> Optional[str]:
        """
        Create chart showing cost impact attribution by various factors.
        
        Args:
            transactions: List of transaction requests
            cost_breakdowns: Corresponding cost breakdowns
            portfolio_value: Total portfolio value for impact calculation
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for attribution chart")
            return None
        
        try:
            # Analyze by different dimensions
            analyses = {
                'instrument_type': {},
                'transaction_size': {'Small': 0, 'Medium': 0, 'Large': 0},
                'time_of_day': {'Morning': 0, 'Midday': 0, 'Afternoon': 0, 'After Hours': 0},
                'transaction_type': {}
            }
            
            total_cost = 0
            
            for transaction, breakdown in zip(transactions, cost_breakdowns):
                cost = float(breakdown.total_cost)
                notional = float(transaction.notional_value)
                total_cost += cost
                
                # By instrument type
                instrument = transaction.instrument_type.name
                if instrument not in analyses['instrument_type']:
                    analyses['instrument_type'][instrument] = 0
                analyses['instrument_type'][instrument] += cost
                
                # By transaction size
                if notional < 10000:
                    analyses['transaction_size']['Small'] += cost
                elif notional < 100000:
                    analyses['transaction_size']['Medium'] += cost
                else:
                    analyses['transaction_size']['Large'] += cost
                
                # By time of day
                hour = transaction.timestamp.hour
                if 9 <= hour < 11:
                    analyses['time_of_day']['Morning'] += cost
                elif 11 <= hour < 14:
                    analyses['time_of_day']['Midday'] += cost
                elif 14 <= hour < 16:
                    analyses['time_of_day']['Afternoon'] += cost
                else:
                    analyses['time_of_day']['After Hours'] += cost
                
                # By transaction type
                trans_type = transaction.transaction_type.name
                if trans_type not in analyses['transaction_type']:
                    analyses['transaction_type'][trans_type] = 0
                analyses['transaction_type'][trans_type] += cost
            
            # Create attribution chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. By instrument type
            instruments = list(analyses['instrument_type'].keys())
            inst_costs = list(analyses['instrument_type'].values())
            if instruments:
                colors_inst = [self.colors['neutral'], self.colors['positive'], self.colors['warning'], 
                              self.colors['negative']][:len(instruments)]
                ax1.pie(inst_costs, labels=instruments, autopct='%1.1f%%', colors=colors_inst)
                ax1.set_title('Cost Attribution by Instrument Type', fontsize=14, fontweight='bold')
            
            # 2. By transaction size
            sizes = list(analyses['transaction_size'].keys())
            size_costs = list(analyses['transaction_size'].values())
            bars1 = ax2.bar(sizes, size_costs, color=[self.colors['positive'], self.colors['warning'], self.colors['negative']])
            ax2.set_title('Cost Attribution by Transaction Size', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Total Cost ($)', fontsize=12)
            
            # Add value labels on bars
            for bar, cost in zip(bars1, size_costs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(size_costs)*0.01,
                        f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. By time of day
            times = list(analyses['time_of_day'].keys())
            time_costs = list(analyses['time_of_day'].values())
            bars2 = ax3.bar(times, time_costs, color=self.colors['info'])
            ax3.set_title('Cost Attribution by Time of Day', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Total Cost ($)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, cost in zip(bars2, time_costs):
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + max(time_costs)*0.01,
                            f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. Portfolio impact summary
            portfolio_impact_bps = (total_cost / portfolio_value) * 10000 if portfolio_value > 0 else 0
            annual_impact = portfolio_impact_bps * 12  # Assuming monthly data
            
            impact_data = {
                'Total Cost': total_cost,
                'Portfolio Impact (bps)': portfolio_impact_bps,
                'Estimated Annual Impact (bps)': annual_impact,
                'Cost as % of Portfolio': (total_cost / portfolio_value) * 100 if portfolio_value > 0 else 0
            }
            
            # Create impact summary table
            ax4.axis('tight')
            ax4.axis('off')
            
            table_data = [[k, f"{v:.2f}" + ("%" if "%" in k else " bps" if "bps" in k else "" if "Cost" in k else "")] 
                         for k, v in impact_data.items()]
            
            table = ax4.table(cellText=table_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            # Color code the table
            for i in range(len(table_data)):
                if 'Cost' in table_data[i][0]:
                    table[(i+1, 1)].set_facecolor('#FFE6E6')  # Light red
                elif 'Impact' in table_data[i][0]:
                    table[(i+1, 1)].set_facecolor('#E6F3FF')  # Light blue
            
            ax4.set_title('Portfolio Impact Summary', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_attribution")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost impact attribution chart: {e}")
            return None
    
    def create_cost_breakeven_analysis(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        expected_returns: List[float],
        holding_periods: List[int],
        save_name: str = "cost_breakeven"
    ) -> Optional[str]:
        """
        Create chart showing breakeven analysis for covering transaction costs.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            expected_returns: Expected annual returns (as percentages)
            holding_periods: Holding periods in days
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for breakeven analysis")
            return None
        
        try:
            # Calculate breakeven metrics
            breakeven_data = []
            
            for breakdown, expected_ret, holding_days in zip(cost_breakdowns, expected_returns, holding_periods):
                # Assume notional value for calculation (or could be passed as parameter)
                notional = 10000  # $10k default for demonstration
                cost_bps = (float(breakdown.total_cost) / notional) * 10000
                
                # Daily expected return
                daily_expected_ret = expected_ret / 365
                
                # Required return to break even
                required_return_pct = (cost_bps / 10000) * 100
                
                # Days to breakeven at expected return rate
                if daily_expected_ret > 0:
                    days_to_breakeven = required_return_pct / daily_expected_ret
                else:
                    days_to_breakeven = float('inf')
                
                # Will this position break even given holding period?
                will_breakeven = days_to_breakeven <= holding_days
                
                breakeven_data.append({
                    'cost_bps': cost_bps,
                    'expected_return': expected_ret,
                    'holding_days': holding_days,
                    'required_return': required_return_pct,
                    'days_to_breakeven': min(days_to_breakeven, 365),  # Cap at 1 year
                    'will_breakeven': will_breakeven
                })
            
            df = pd.DataFrame(breakeven_data)
            
            # Create breakeven analysis chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Breakeven days distribution
            breakeven_days = df['days_to_breakeven'].replace([np.inf, -np.inf], 365)
            ax1.hist(breakeven_days, bins=20, alpha=0.7, color=self.colors['neutral'], edgecolor='black')
            ax1.axvline(np.median(breakeven_days), color=self.colors['negative'], 
                       linestyle='--', linewidth=2, label=f'Median: {np.median(breakeven_days):.0f} days')
            ax1.set_title('Distribution of Days to Breakeven', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Days to Breakeven', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Cost vs Expected Return scatter
            colors = [self.colors['positive'] if will_break else self.colors['negative'] 
                     for will_break in df['will_breakeven']]
            scatter = ax2.scatter(df['cost_bps'], df['expected_return'], c=colors, alpha=0.6, s=50)
            
            # Add breakeven line
            max_cost = df['cost_bps'].max()
            breakeven_line_x = np.linspace(0, max_cost, 100)
            breakeven_line_y = breakeven_line_x / 100  # Convert bps to percentage for annual return
            ax2.plot(breakeven_line_x, breakeven_line_y, 'r--', linewidth=2, 
                    label='Breakeven Line (1-year holding)')
            
            ax2.set_title('Cost vs Expected Return', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Transaction Cost (bps)', fontsize=12)
            ax2.set_ylabel('Expected Annual Return (%)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add legend for colors
            positive_patch = plt.Rectangle((0, 0), 1, 1, color=self.colors['positive'], alpha=0.6, label='Will Break Even')
            negative_patch = plt.Rectangle((0, 0), 1, 1, color=self.colors['negative'], alpha=0.6, label='Won\'t Break Even')
            ax2.legend(handles=[positive_patch, negative_patch], loc='upper right')
            
            # 3. Holding period vs breakeven success
            holding_bins = pd.cut(df['holding_days'], bins=5)
            breakeven_by_holding = df.groupby(holding_bins)['will_breakeven'].agg(['count', 'sum'])
            breakeven_by_holding['success_rate'] = (breakeven_by_holding['sum'] / breakeven_by_holding['count']) * 100
            
            bars = ax3.bar(range(len(breakeven_by_holding)), breakeven_by_holding['success_rate'], 
                          color=self.colors['warning'], alpha=0.7)
            ax3.set_title('Breakeven Success Rate by Holding Period', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Holding Period Bins', fontsize=12)
            ax3.set_ylabel('Success Rate (%)', fontsize=12)
            ax3.set_xticks(range(len(breakeven_by_holding)))
            ax3.set_xticklabels([str(interval) for interval in breakeven_by_holding.index], rotation=45)
            
            # Add value labels
            for bar, rate in zip(bars, breakeven_by_holding['success_rate']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # 4. Summary statistics
            summary_stats = {
                'Total Positions': len(df),
                'Will Break Even': df['will_breakeven'].sum(),
                'Success Rate': f"{(df['will_breakeven'].sum() / len(df)) * 100:.1f}%",
                'Avg Days to Breakeven': f"{df['days_to_breakeven'].replace([np.inf], np.nan).mean():.0f}",
                'Avg Cost (bps)': f"{df['cost_bps'].mean():.1f}",
                'Avg Expected Return': f"{df['expected_return'].mean():.1f}%"
            }
            
            ax4.axis('tight')
            ax4.axis('off')
            
            table_data = [[k, v] for k, v in summary_stats.items()]
            table = ax4.table(cellText=table_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            ax4.set_title('Breakeven Analysis Summary', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_breakeven")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating breakeven analysis chart: {e}")
            return None
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save plot with proper formatting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{filename}_{timestamp}.png"
        
        save_result = self.file_manager.save_file(
            fig, save_name,
            metadata={
                "chart_type": "cost_impact",
                "format": "png",
                "dpi": self.dpi,
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Cost impact chart saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save chart: {save_result.error_message}")
            return ""


logger.info("Cost impact charts module loaded successfully")