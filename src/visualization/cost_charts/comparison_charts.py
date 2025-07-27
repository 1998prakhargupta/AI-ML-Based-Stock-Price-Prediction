"""
Cost Comparison Charts
======================

Specialized charts for comparing transaction costs across different brokers,
time periods, and trading strategies.
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

# Import cost reporting components
from src.visualization.cost_reporting.cost_reporter import BrokerComparisonResult

logger = logging.getLogger(__name__)

class CostComparisonCharts:
    """
    Specialized charts for comparing transaction costs across different
    brokers, time periods, and trading scenarios.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cost comparison charts.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        
        # Chart styling
        self.colors = {
            'best': '#27AE60',       # Green
            'good': '#3498DB',       # Blue
            'average': '#F39C12',    # Orange
            'poor': '#E74C3C',       # Red
            'neutral': '#95A5A6'     # Gray
        }
        
        self.figure_size = (12, 8)
        self.dpi = 300
        
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        logger.info("Cost comparison charts initialized")
    
    def create_broker_comparison_chart(
        self,
        comparison_result: BrokerComparisonResult,
        save_name: str = "broker_comparison"
    ) -> Optional[str]:
        """
        Create comprehensive broker comparison chart.
        
        Args:
            comparison_result: Broker comparison analysis result
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for broker comparison chart")
            return None
        
        try:
            df = comparison_result.comparison_matrix
            
            if df.empty:
                logger.warning("No broker comparison data available")
                return None
            
            # Create comprehensive comparison chart
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Total cost comparison (main chart)
            ax1 = fig.add_subplot(gs[0, :])
            brokers = df['Broker'].values
            total_costs = df['Total Cost'].values
            
            # Color bars based on cost ranking
            colors = []
            for i, cost in enumerate(total_costs):
                if i == 0:  # Best broker
                    colors.append(self.colors['best'])
                elif cost <= np.percentile(total_costs, 33):
                    colors.append(self.colors['good'])
                elif cost <= np.percentile(total_costs, 66):
                    colors.append(self.colors['average'])
                else:
                    colors.append(self.colors['poor'])
            
            bars = ax1.bar(brokers, total_costs, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, cost in zip(bars, total_costs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(total_costs)*0.01,
                        f'${cost:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight best broker
            best_idx = np.argmin(total_costs)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            
            ax1.set_title('Total Transaction Cost Comparison by Broker', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Total Cost ($)', fontsize=12)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add savings annotation
            savings_text = f"Best: {comparison_result.best_broker}\nMax Savings: ${float(comparison_result.potential_savings):.4f}"
            ax1.text(0.02, 0.98, savings_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    verticalalignment='top', fontweight='bold')
            
            # 2. Cost component breakdown heatmap
            ax2 = fig.add_subplot(gs[1, :2])
            component_cols = ['Commission', 'Regulatory Fees', 'Exchange Fees', 
                            'Market Impact', 'Spread Cost', 'Timing Cost']
            
            # Create heatmap data
            heatmap_data = df[component_cols].T
            heatmap_data.columns = df['Broker']
            
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                       ax=ax2, cbar_kws={'label': 'Cost ($)'})
            ax2.set_title('Cost Component Breakdown by Broker', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Broker', fontsize=12)
            ax2.set_ylabel('Cost Component', fontsize=12)
            
            # 3. Efficiency scores
            ax3 = fig.add_subplot(gs[1, 2])
            efficiency_scores = df['Efficiency Score'].values
            
            # Create radar-like bar chart
            bars_eff = ax3.barh(brokers, efficiency_scores, color=self.colors['good'], alpha=0.7)
            
            # Color code efficiency
            for i, (bar, score) in enumerate(zip(bars_eff, efficiency_scores)):
                if score >= 90:
                    bar.set_color(self.colors['best'])
                elif score >= 75:
                    bar.set_color(self.colors['good'])
                elif score >= 60:
                    bar.set_color(self.colors['average'])
                else:
                    bar.set_color(self.colors['poor'])
                
                # Add score labels
                ax3.text(score + 1, bar.get_y() + bar.get_height()/2,
                        f'{score:.1f}', ha='left', va='center', fontweight='bold')
            
            ax3.set_title('Efficiency Scores', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Score', fontsize=12)
            ax3.set_xlim(0, 100)
            ax3.grid(True, alpha=0.3, axis='x')
            
            # 4. Return impact comparison
            ax4 = fig.add_subplot(gs[2, 0])
            return_impacts = df['Return Impact (%)'].values
            
            bars_impact = ax4.bar(brokers, return_impacts, color=self.colors['poor'], alpha=0.7)
            ax4.set_title('Return Impact', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Impact (%)', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add impact threshold line
            ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='0.1% Threshold')
            ax4.legend()
            
            # 5. Cost difference from best
            ax5 = fig.add_subplot(gs[2, 1])
            best_cost = min(total_costs)
            cost_differences = [(cost - best_cost) for cost in total_costs]
            
            bars_diff = ax5.bar(brokers, cost_differences, color=self.colors['neutral'], alpha=0.7)
            
            # Color code differences
            for bar, diff in zip(bars_diff, cost_differences):
                if diff == 0:
                    bar.set_color(self.colors['best'])
                elif diff <= 0.01:
                    bar.set_color(self.colors['good'])
                elif diff <= 0.05:
                    bar.set_color(self.colors['average'])
                else:
                    bar.set_color(self.colors['poor'])
            
            ax5.set_title('Cost Difference from Best', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Additional Cost ($)', fontsize=12)
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 6. Recommendations summary
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            
            # Display key recommendations
            recommendations_text = "\n".join([
                "Key Recommendations:",
                "",
                f"• Best Broker: {comparison_result.best_broker}",
                f"• Potential Savings: ${float(comparison_result.potential_savings):.4f}",
                "",
                "Top Issues:"
            ] + comparison_result.recommendations[:3])
            
            ax6.text(0.05, 0.95, recommendations_text, transform=ax6.transAxes,
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Add overall title
            fig.suptitle('Comprehensive Broker Cost Comparison Analysis', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_comprehensive")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating broker comparison chart: {e}")
            return None
    
    def create_cost_trend_comparison(
        self,
        historical_data: Dict[str, List[Dict]],
        save_name: str = "cost_trend_comparison"
    ) -> Optional[str]:
        """
        Create chart comparing cost trends across different periods or brokers.
        
        Args:
            historical_data: Historical cost data by category
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for trend comparison chart")
            return None
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Process data for different comparisons
            colors_list = list(self.colors.values())
            
            # 1. Time series comparison
            for i, (category, data) in enumerate(historical_data.items()):
                if not data:
                    continue
                
                df = pd.DataFrame(data)
                if 'date' in df.columns and 'cost_bps' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    color = colors_list[i % len(colors_list)]
                    ax1.plot(df['date'], df['cost_bps'], marker='o', 
                            label=category, color=color, linewidth=2)
            
            ax1.set_title('Cost Trends Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cost (basis points)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Average cost comparison
            avg_costs = {}
            for category, data in historical_data.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'cost_bps' in df.columns:
                        avg_costs[category] = df['cost_bps'].mean()
            
            if avg_costs:
                categories = list(avg_costs.keys())
                costs = list(avg_costs.values())
                
                bars = ax2.bar(categories, costs, color=colors_list[:len(categories)], alpha=0.7)
                
                # Add value labels
                for bar, cost in zip(bars, costs):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                            f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')
                
                ax2.set_title('Average Cost Comparison', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Average Cost (bps)', fontsize=12)
                ax2.tick_params(axis='x', rotation=45)
            
            # 3. Volatility comparison
            volatilities = {}
            for category, data in historical_data.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'cost_bps' in df.columns and len(df) > 1:
                        volatilities[category] = df['cost_bps'].std()
            
            if volatilities:
                categories_vol = list(volatilities.keys())
                vols = list(volatilities.values())
                
                bars_vol = ax3.bar(categories_vol, vols, color=self.colors['poor'], alpha=0.7)
                ax3.set_title('Cost Volatility Comparison', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Cost Std Dev (bps)', fontsize=12)
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, vol in zip(bars_vol, vols):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + max(vols)*0.01,
                            f'{vol:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. Cost distribution comparison
            for i, (category, data) in enumerate(historical_data.items()):
                if not data:
                    continue
                
                df = pd.DataFrame(data)
                if 'cost_bps' in df.columns:
                    color = colors_list[i % len(colors_list)]
                    ax4.hist(df['cost_bps'], bins=15, alpha=0.5, label=category, 
                            color=color, density=True)
            
            ax4.set_title('Cost Distribution Comparison', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Cost (basis points)', fontsize=12)
            ax4.set_ylabel('Density', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_trends")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cost trend comparison chart: {e}")
            return None
    
    def create_cost_efficiency_benchmark(
        self,
        actual_costs: Dict[str, float],
        benchmark_costs: Dict[str, float],
        save_name: str = "cost_efficiency_benchmark"
    ) -> Optional[str]:
        """
        Create chart comparing actual costs against benchmarks.
        
        Args:
            actual_costs: Actual costs by category (in bps)
            benchmark_costs: Benchmark costs by category (in bps)
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for benchmark chart")
            return None
        
        try:
            # Prepare data
            categories = list(set(actual_costs.keys()) & set(benchmark_costs.keys()))
            if not categories:
                logger.warning("No matching categories for benchmark comparison")
                return None
            
            actual_values = [actual_costs[cat] for cat in categories]
            benchmark_values = [benchmark_costs[cat] for cat in categories]
            differences = [actual - benchmark for actual, benchmark in zip(actual_values, benchmark_values)]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Side-by-side comparison
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, actual_values, width, label='Actual', 
                           color=self.colors['neutral'], alpha=0.8)
            bars2 = ax1.bar(x + width/2, benchmark_values, width, label='Benchmark', 
                           color=self.colors['good'], alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars1, actual_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(actual_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            for bar, value in zip(bars2, benchmark_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(benchmark_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax1.set_title('Actual vs Benchmark Costs', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cost (basis points)', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 2. Performance vs benchmark (percentage)
            performance_pct = [(actual/benchmark - 1) * 100 if benchmark > 0 else 0 
                              for actual, benchmark in zip(actual_values, benchmark_values)]
            
            colors_perf = [self.colors['best'] if pct <= 0 else 
                          self.colors['good'] if pct <= 10 else
                          self.colors['average'] if pct <= 25 else
                          self.colors['poor'] for pct in performance_pct]
            
            bars_perf = ax2.bar(categories, performance_pct, color=colors_perf, alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% Above Benchmark')
            ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='25% Above Benchmark')
            
            # Add value labels
            for bar, pct in zip(bars_perf, performance_pct):
                height = bar.get_height()
                y_pos = height + 1 if height >= 0 else height - 1
                ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{pct:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontweight='bold')
            
            ax2.set_title('Performance vs Benchmark', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Difference (%)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 3. Cost difference in absolute terms
            colors_diff = [self.colors['best'] if diff <= 0 else 
                          self.colors['poor'] for diff in differences]
            
            bars_diff = ax3.bar(categories, differences, color=colors_diff, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Add value labels
            for bar, diff in zip(bars_diff, differences):
                height = bar.get_height()
                y_pos = height + max(abs(min(differences)), max(differences)) * 0.02 if height >= 0 else height - max(abs(min(differences)), max(differences)) * 0.02
                ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{diff:+.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontweight='bold')
            
            ax3.set_title('Absolute Cost Difference', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Difference (bps)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Summary scorecard
            ax4.axis('off')
            
            # Calculate summary metrics
            total_categories = len(categories)
            outperforming = sum(1 for diff in differences if diff <= 0)
            underperforming = total_categories - outperforming
            avg_difference = np.mean(differences)
            
            # Overall score (0-100)
            negative_diffs = [abs(d) for d in differences if d < 0]
            positive_diffs = [d for d in differences if d >= 0]
            
            if negative_diffs:
                outperform_benefit = np.mean(negative_diffs)
            else:
                outperform_benefit = 0
            
            if positive_diffs:
                underperform_cost = np.mean(positive_diffs)
            else:
                underperform_cost = 0
            
            # Simple scoring
            if avg_difference <= -2:
                score = 100
                grade = 'A+'
            elif avg_difference <= 0:
                score = 85
                grade = 'A'
            elif avg_difference <= 2:
                score = 70
                grade = 'B'
            elif avg_difference <= 5:
                score = 55
                grade = 'C'
            else:
                score = 40
                grade = 'D'
            
            scorecard_text = f"""BENCHMARK SCORECARD
            
Overall Grade: {grade}
Score: {score}/100

Categories Analyzed: {total_categories}
Outperforming: {outperforming} ({outperforming/total_categories*100:.1f}%)
Underperforming: {underperforming} ({underperforming/total_categories*100:.1f}%)

Average Difference: {avg_difference:+.2f} bps
Best Category: {categories[np.argmin(differences)]}
Worst Category: {categories[np.argmax(differences)]}"""
            
            # Color based on grade
            if grade in ['A+', 'A']:
                box_color = 'lightgreen'
            elif grade == 'B':
                box_color = 'lightyellow'
            elif grade == 'C':
                box_color = 'orange'
            else:
                box_color = 'lightcoral'
            
            ax4.text(0.1, 0.9, scorecard_text, transform=ax4.transAxes,
                    verticalalignment='top', fontsize=12, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_benchmark")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating benchmark comparison chart: {e}")
            return None
    
    def create_cost_scenario_analysis(
        self,
        scenarios: Dict[str, Dict[str, float]],
        base_scenario: str,
        save_name: str = "cost_scenario_analysis"
    ) -> Optional[str]:
        """
        Create chart analyzing costs under different scenarios.
        
        Args:
            scenarios: Cost data for different scenarios
            base_scenario: Name of the base scenario for comparison
            save_name: Name for saved chart
            
        Returns:
            Path to saved chart or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for scenario analysis chart")
            return None
        
        try:
            if base_scenario not in scenarios:
                logger.error(f"Base scenario '{base_scenario}' not found in scenarios")
                return None
            
            base_data = scenarios[base_scenario]
            scenario_names = list(scenarios.keys())
            
            # Get common cost components
            all_components = set()
            for scenario_data in scenarios.values():
                all_components.update(scenario_data.keys())
            components = sorted(list(all_components))
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Scenario comparison heatmap
            heatmap_data = []
            for scenario in scenario_names:
                row = [scenarios[scenario].get(comp, 0) for comp in components]
                heatmap_data.append(row)
            
            heatmap_df = pd.DataFrame(heatmap_data, index=scenario_names, columns=components)
            
            sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       ax=ax1, cbar_kws={'label': 'Cost (bps)'})
            ax1.set_title('Cost Scenarios Heatmap', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Cost Components', fontsize=12)
            ax1.set_ylabel('Scenarios', fontsize=12)
            
            # 2. Total cost by scenario
            total_costs = [sum(scenarios[scenario].values()) for scenario in scenario_names]
            base_total = sum(base_data.values())
            
            colors_scenario = [self.colors['neutral'] if scenario == base_scenario else 
                             self.colors['best'] if total < base_total else
                             self.colors['poor'] for scenario, total in zip(scenario_names, total_costs)]
            
            bars_total = ax2.bar(scenario_names, total_costs, color=colors_scenario, alpha=0.8)
            
            # Highlight base scenario
            base_idx = scenario_names.index(base_scenario)
            bars_total[base_idx].set_edgecolor('gold')
            bars_total[base_idx].set_linewidth(3)
            
            # Add value labels
            for bar, cost in zip(bars_total, total_costs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(total_costs)*0.01,
                        f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title('Total Cost by Scenario', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Total Cost (bps)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 3. Difference from base scenario
            cost_differences = [total - base_total for total in total_costs]
            
            colors_diff = [self.colors['neutral'] if diff == 0 else 
                          self.colors['best'] if diff < 0 else
                          self.colors['poor'] for diff in cost_differences]
            
            bars_diff = ax3.bar(scenario_names, cost_differences, color=colors_diff, alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Add value labels
            for bar, diff in zip(bars_diff, cost_differences):
                height = bar.get_height()
                if height != 0:
                    y_pos = height + max(abs(min(cost_differences)), max(cost_differences)) * 0.02 if height > 0 else height - max(abs(min(cost_differences)), max(cost_differences)) * 0.02
                    ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{diff:+.1f}', ha='center', va='bottom' if height > 0 else 'top', 
                            fontweight='bold')
            
            ax3.set_title(f'Difference from {base_scenario}', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Cost Difference (bps)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Scenario ranking and analysis
            ax4.axis('off')
            
            # Rank scenarios by total cost
            scenario_ranking = sorted(zip(scenario_names, total_costs), key=lambda x: x[1])
            
            ranking_text = "SCENARIO RANKING\n(Best to Worst)\n\n"
            for i, (scenario, cost) in enumerate(scenario_ranking):
                rank = i + 1
                diff_from_best = cost - scenario_ranking[0][1]
                
                if rank == 1:
                    status = "👑 BEST"
                elif rank <= len(scenario_ranking) // 2:
                    status = "✅ GOOD"
                else:
                    status = "⚠️  COSTLY"
                
                ranking_text += f"{rank}. {scenario}: {cost:.1f} bps {status}\n"
                if diff_from_best > 0:
                    ranking_text += f"   (+{diff_from_best:.1f} vs best)\n"
                ranking_text += "\n"
            
            # Best scenario analysis
            best_scenario, best_cost = scenario_ranking[0]
            worst_scenario, worst_cost = scenario_ranking[-1]
            cost_spread = worst_cost - best_cost
            
            ranking_text += f"ANALYSIS:\n"
            ranking_text += f"Best: {best_scenario} ({best_cost:.1f} bps)\n"
            ranking_text += f"Worst: {worst_scenario} ({worst_cost:.1f} bps)\n"
            ranking_text += f"Cost Spread: {cost_spread:.1f} bps\n"
            ranking_text += f"Improvement Opportunity: {(cost_spread/worst_cost)*100:.1f}%"
            
            ax4.text(0.05, 0.95, ranking_text, transform=ax4.transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            plot_path = self._save_plot(fig, f"{save_name}_scenarios")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating scenario analysis chart: {e}")
            return None
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save plot with proper formatting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{filename}_{timestamp}.png"
        
        save_result = self.file_manager.save_file(
            fig, save_name,
            metadata={
                "chart_type": "cost_comparison",
                "format": "png",
                "dpi": self.dpi,
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Cost comparison chart saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save chart: {save_result.error_message}")
            return ""


logger.info("Cost comparison charts module loaded successfully")