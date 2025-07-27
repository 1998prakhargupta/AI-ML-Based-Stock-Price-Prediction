"""
Cost Breakdown Charts
=====================

Specialized charts for visualizing transaction cost breakdowns by component,
time period, and various categorizations.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Plotting dependencies with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    GridSpec = None
    mpatches = None
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plotting functions will be disabled")

# Import utilities
from src.utils.file_management_utils import SafeFileManager
from src.utils.config_manager import Config

# Import cost models
from src.trading.transaction_costs.models import TransactionCostBreakdown

logger = logging.getLogger(__name__)

class CostBreakdownCharts:
    """
    Specialized charts for visualizing transaction cost breakdowns
    with detailed component analysis and categorization.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cost breakdown charts.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        
        # Chart styling
        self.colors = {
            'commission': '#2E86C1',      # Blue
            'fees': '#E74C3C',            # Red  
            'market_impact': '#F39C12',   # Orange
            'spreads': '#27AE60',         # Green
            'timing': '#9B59B6',          # Purple
            'other': '#95A5A6'            # Gray
        }
        
        self.figure_size = (12, 8)
        self.dpi = 300
        
        if PLOTTING_AVAILABLE:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        logger.info("Cost breakdown charts initialized")
    
    def create_cost_component_pie_chart(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        save_name: str = "cost_component_breakdown"
    ) -> Optional[str]:
        """
        Create pie chart showing cost breakdown by component.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for pie chart creation")
            return None
        
        try:
            # Aggregate costs by component
            component_totals = {
                'Commission': 0.0,
                'Regulatory Fees': 0.0,
                'Exchange Fees': 0.0,
                'Market Impact': 0.0,
                'Bid-Ask Spread': 0.0,
                'Timing Cost': 0.0,
                'Platform Fees': 0.0,
                'Other': 0.0
            }
            
            for breakdown in cost_breakdowns:
                component_totals['Commission'] += float(breakdown.commission)
                component_totals['Regulatory Fees'] += float(breakdown.regulatory_fees)
                component_totals['Exchange Fees'] += float(breakdown.exchange_fees)
                component_totals['Market Impact'] += float(breakdown.market_impact_cost)
                component_totals['Bid-Ask Spread'] += float(breakdown.bid_ask_spread_cost)
                component_totals['Timing Cost'] += float(breakdown.timing_cost)
                component_totals['Platform Fees'] += float(breakdown.platform_fees)
                component_totals['Other'] += float(
                    breakdown.data_fees + breakdown.miscellaneous_fees + 
                    breakdown.borrowing_cost + breakdown.overnight_financing + 
                    breakdown.currency_conversion
                )
            
            # Filter out zero components
            filtered_components = {k: v for k, v in component_totals.items() if v > 0}
            
            if not filtered_components:
                logger.warning("No cost data available for pie chart")
                return None
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            labels = list(filtered_components.keys())
            sizes = list(filtered_components.values())
            colors = [self.colors.get(label.lower().replace(' ', '_'), '#95A5A6') for label in labels]
            
            # Create pie chart with enhanced styling
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10}
            )
            
            # Enhance appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Transaction Cost Breakdown by Component', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add total cost annotation
            total_cost = sum(sizes)
            ax.text(0, -1.3, f'Total Cost: ${total_cost:,.2f}', 
                   ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_pie")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost component pie chart: {e}")
            return None
    
    def create_cost_component_stacked_bar(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        transaction_labels: List[str],
        save_name: str = "cost_component_stacked"
    ) -> Optional[str]:
        """
        Create stacked bar chart showing cost components across transactions.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            transaction_labels: Labels for each transaction
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for stacked bar chart")
            return None
        
        try:
            # Prepare data
            data = []
            for i, breakdown in enumerate(cost_breakdowns):
                label = transaction_labels[i] if i < len(transaction_labels) else f"Transaction {i+1}"
                data.append({
                    'Transaction': label,
                    'Commission': float(breakdown.commission),
                    'Fees': float(breakdown.regulatory_fees + breakdown.exchange_fees),
                    'Market Impact': float(breakdown.market_impact_cost),
                    'Spreads': float(breakdown.bid_ask_spread_cost),
                    'Timing': float(breakdown.timing_cost),
                    'Other': float(breakdown.platform_fees + breakdown.data_fees + 
                                 breakdown.miscellaneous_fees)
                })
            
            df = pd.DataFrame(data)
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.8), 8))
            
            # Define components and colors
            components = ['Commission', 'Fees', 'Market Impact', 'Spreads', 'Timing', 'Other']
            colors = [self.colors.get(comp.lower(), '#95A5A6') for comp in components]
            
            # Create stacked bars
            bottom = np.zeros(len(df))
            bars = []
            
            for i, component in enumerate(components):
                values = df[component].values
                bar = ax.bar(df['Transaction'], values, bottom=bottom, 
                           label=component, color=colors[i], alpha=0.8)
                bars.append(bar)
                bottom += values
            
            # Styling
            ax.set_title('Cost Components by Transaction', fontsize=16, fontweight='bold')
            ax.set_xlabel('Transactions', fontsize=12)
            ax.set_ylabel('Cost ($)', fontsize=12)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Rotate x-axis labels if many transactions
            if len(df) > 10:
                plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars for larger segments
            for component_bars in bars:
                for bar in component_bars:
                    height = bar.get_height()
                    if height > max(bottom) * 0.05:  # Only label if >5% of max
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               bar.get_y() + height/2.,
                               f'${height:.1f}', ha='center', va='center',
                               fontsize=8, fontweight='bold', color='white')
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_stacked")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating stacked bar chart: {e}")
            return None
    
    def create_cost_heatmap_by_time_and_component(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        timestamps: List[datetime],
        save_name: str = "cost_heatmap"
    ) -> Optional[str]:
        """
        Create heatmap showing costs by time period and component.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            timestamps: Corresponding timestamps
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for heatmap creation")
            return None
        
        try:
            # Prepare data by hour and component
            hourly_data = {}
            
            for breakdown, timestamp in zip(cost_breakdowns, timestamps):
                hour = timestamp.hour
                if hour not in hourly_data:
                    hourly_data[hour] = {
                        'Commission': 0.0,
                        'Fees': 0.0,
                        'Market Impact': 0.0,
                        'Spreads': 0.0,
                        'Timing': 0.0,
                        'Other': 0.0
                    }
                
                hourly_data[hour]['Commission'] += float(breakdown.commission)
                hourly_data[hour]['Fees'] += float(breakdown.regulatory_fees + breakdown.exchange_fees)
                hourly_data[hour]['Market Impact'] += float(breakdown.market_impact_cost)
                hourly_data[hour]['Spreads'] += float(breakdown.bid_ask_spread_cost)
                hourly_data[hour]['Timing'] += float(breakdown.timing_cost)
                hourly_data[hour]['Other'] += float(breakdown.platform_fees + breakdown.data_fees)
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(hourly_data, orient='index')
            df = df.fillna(0)
            
            if df.empty:
                logger.warning("No data available for heatmap")
                return None
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create heatmap with custom colormap
            sns.heatmap(df.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Cost ($)'})
            
            ax.set_title('Cost Components by Hour of Day', fontsize=16, fontweight='bold')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Cost Component', fontsize=12)
            
            # Format hour labels
            hour_labels = [f"{h:02d}:00" for h in df.index]
            ax.set_xticklabels(hour_labels, rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_heatmap")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost heatmap: {e}")
            return None
    
    def create_cost_waterfall_chart(
        self,
        cost_breakdown: TransactionCostBreakdown,
        save_name: str = "cost_waterfall"
    ) -> Optional[str]:
        """
        Create waterfall chart showing cost buildup for a single transaction.
        
        Args:
            cost_breakdown: Single cost breakdown to analyze
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for waterfall chart")
            return None
        
        try:
            # Define cost components
            components = [
                ('Commission', float(cost_breakdown.commission)),
                ('Regulatory Fees', float(cost_breakdown.regulatory_fees)),
                ('Exchange Fees', float(cost_breakdown.exchange_fees)),
                ('Market Impact', float(cost_breakdown.market_impact_cost)),
                ('Bid-Ask Spread', float(cost_breakdown.bid_ask_spread_cost)),
                ('Timing Cost', float(cost_breakdown.timing_cost)),
                ('Platform Fees', float(cost_breakdown.platform_fees)),
                ('Other Fees', float(cost_breakdown.data_fees + cost_breakdown.miscellaneous_fees))
            ]
            
            # Filter out zero components
            components = [(name, value) for name, value in components if value > 0]
            
            if not components:
                logger.warning("No cost components to display in waterfall chart")
                return None
            
            # Create waterfall chart
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            x_pos = np.arange(len(components) + 1)
            cumulative = 0
            colors = []
            bars = []
            
            # Starting point
            ax.bar(0, 0, color='lightgray', alpha=0.5, label='Start')
            ax.text(0, 0, '$0.00', ha='center', va='bottom', fontweight='bold')
            
            # Add each component
            for i, (name, value) in enumerate(components):
                color = self.colors.get(name.lower().replace(' ', '_'), '#95A5A6')
                
                # Create bar from cumulative to cumulative + value
                bar = ax.bar(i + 1, value, bottom=cumulative, color=color, alpha=0.8)
                bars.append(bar)
                
                # Add value label
                label_y = cumulative + value / 2
                ax.text(i + 1, label_y, f'${value:.2f}', ha='center', va='center',
                       fontweight='bold', color='white' if value > max([v for _, v in components]) * 0.1 else 'black')
                
                # Add connecting lines
                if i < len(components) - 1:
                    ax.plot([i + 1.4, i + 1.6], [cumulative + value, cumulative + value], 
                           'k--', alpha=0.5, linewidth=1)
                
                cumulative += value
            
            # Total bar
            ax.bar(len(components) + 1, cumulative, color='darkgreen', alpha=0.8, label='Total')
            ax.text(len(components) + 1, cumulative / 2, f'${cumulative:.2f}', 
                   ha='center', va='center', fontweight='bold', color='white')
            
            # Styling
            ax.set_title('Transaction Cost Waterfall Analysis', fontsize=16, fontweight='bold')
            ax.set_ylabel('Cost ($)', fontsize=12)
            
            # Set x-axis labels
            labels = ['Start'] + [name for name, _ in components] + ['Total']
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_waterfall")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating waterfall chart: {e}")
            return None
    
    def create_cost_distribution_histogram(
        self,
        cost_breakdowns: List[TransactionCostBreakdown],
        notional_values: List[float],
        save_name: str = "cost_distribution"
    ) -> Optional[str]:
        """
        Create histogram showing distribution of costs in basis points.
        
        Args:
            cost_breakdowns: List of cost breakdowns
            notional_values: Corresponding notional values
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for histogram creation")
            return None
        
        try:
            # Calculate costs in basis points
            cost_bps = []
            for breakdown, notional in zip(cost_breakdowns, notional_values):
                if notional > 0:
                    bps = (float(breakdown.total_cost) / notional) * 10000
                    cost_bps.append(bps)
            
            if not cost_bps:
                logger.warning("No valid cost data for histogram")
                return None
            
            # Create histogram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            n, bins, patches = ax1.hist(cost_bps, bins=30, alpha=0.7, color=self.colors['commission'],
                                       edgecolor='black', linewidth=0.5)
            
            # Color bars based on cost level
            for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
                if bin_val < 10:
                    patch.set_facecolor(self.colors['spreads'])  # Green for low cost
                elif bin_val < 20:
                    patch.set_facecolor(self.colors['commission'])  # Blue for medium cost
                else:
                    patch.set_facecolor(self.colors['fees'])  # Red for high cost
            
            # Add statistics
            mean_cost = np.mean(cost_bps)
            median_cost = np.median(cost_bps)
            std_cost = np.std(cost_bps)
            
            ax1.axvline(mean_cost, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cost:.1f} bps')
            ax1.axvline(median_cost, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cost:.1f} bps')
            
            ax1.set_title('Distribution of Transaction Costs', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Cost (basis points)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            box_plot = ax2.boxplot(cost_bps, patch_artist=True, labels=['Transaction Costs'])
            box_plot['boxes'][0].set_facecolor(self.colors['commission'])
            box_plot['boxes'][0].set_alpha(0.7)
            
            # Add outlier information
            q1 = np.percentile(cost_bps, 25)
            q3 = np.percentile(cost_bps, 75)
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr
            outliers = [x for x in cost_bps if x > outlier_threshold]
            
            ax2.set_title('Cost Distribution Box Plot', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Cost (basis points)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"""Statistics:
Mean: {mean_cost:.2f} bps
Median: {median_cost:.2f} bps
Std Dev: {std_cost:.2f} bps
Outliers: {len(outliers)}
Sample Size: {len(cost_bps)}"""
            
            ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, 
                    verticalalignment='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_distribution")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost distribution histogram: {e}")
            return None
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save plot with proper formatting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{filename}_{timestamp}.png"
        
        save_result = self.file_manager.save_file(
            fig, save_name,
            metadata={
                "chart_type": "cost_breakdown",
                "format": "png",
                "dpi": self.dpi,
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Cost breakdown chart saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save chart: {save_result.error_message}")
            return ""


logger.info("Cost breakdown charts module loaded successfully")