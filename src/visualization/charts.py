"""
Comprehensive visualization and reporting utilities for stock prediction models.
Maintains all underlying basic logic while adding enhanced visualization and reporting capabilities.
"""

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Plotting dependencies with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    PLOTTING_AVAILABLE = True
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
except ImportError:
    plt = None
    sns = None
    GridSpec = None
    mdates = None
    Rectangle = None
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plotting functions will be disabled")

# Import safe file management
from src.utils.file_management_utils import SafeFileManager, SaveStrategy
from src.utils.app_config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveVisualizer:
    """
    Enhanced visualization utility that maintains all existing logic while adding comprehensive
    visual summaries and automated reporting capabilities.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        self.plots_dir = os.path.join(self.config.get_data_save_path(), 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Color schemes for different plot types
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4B942',
            'models': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        }
        
        # Figure settings
        self.fig_params = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        logger.info("Comprehensive visualizer initialized with enhanced capabilities")
    
    def create_model_performance_dashboard(self, 
                                         results_dict: Dict[str, Dict[str, float]], 
                                         save_name: str = "model_performance_dashboard") -> str:
        """
        Create comprehensive model performance dashboard maintaining existing evaluation logic.
        
        Args:
            results_dict: Dictionary with model names as keys and metrics as values
            save_name: Base name for saved files
            
        Returns:
            Path to saved dashboard
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - dashboard cannot be created")
            return ""
        
        logger.info("Creating comprehensive model performance dashboard")
        
        # Create the dashboard with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Convert results to DataFrame for easier plotting
        metrics_df = pd.DataFrame(results_dict).T
        
        # 1. Model Comparison Bar Charts
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric_comparison(metrics_df, 'MSE', ax1, 'Mean Squared Error')
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric_comparison(metrics_df, 'RMSE', ax2, 'Root Mean Squared Error')
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_metric_comparison(metrics_df, 'R2', ax3, 'R² Score')
        
        # 2. MAE and MAPE Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_metric_comparison(metrics_df, 'MAE', ax4, 'Mean Absolute Error')
        
        ax5 = fig.add_subplot(gs[1, 1])
        if 'MAPE' in metrics_df.columns:
            self._plot_metric_comparison(metrics_df, 'MAPE', ax5, 'Mean Absolute Percentage Error (%)')
        else:
            ax5.text(0.5, 0.5, 'MAPE not available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('MAPE Comparison')
        
        # 3. Model Performance Radar Chart
        ax6 = fig.add_subplot(gs[1, 2])
        self._create_performance_radar_chart(metrics_df, ax6)
        
        # 4. Model Ranking Matrix
        ax7 = fig.add_subplot(gs[2, :])
        self._create_model_ranking_heatmap(metrics_df, ax7)
        
        # 5. Statistical Summary
        ax8 = fig.add_subplot(gs[3, 0])
        self._create_statistical_summary_table(metrics_df, ax8)
        
        # 6. Best Model Highlight
        ax9 = fig.add_subplot(gs[3, 1])
        self._highlight_best_models(metrics_df, ax9)
        
        # 7. Performance Distribution
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_performance_distribution(metrics_df, ax10)
        
        # Add main title
        fig.suptitle('Model Performance Comprehensive Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Save the dashboard
        save_path = os.path.join(self.plots_dir, f"{save_name}.png")
        save_result = self.file_manager.save_figure(
            fig, save_path, 
            metadata={
                "visualization_type": "model_performance_dashboard",
                "models_count": len(results_dict),
                "metrics_analyzed": list(metrics_df.columns.tolist()),
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        plt.show()
        
        if save_result.success:
            logger.info(f"Model performance dashboard saved to: {save_result.final_path}")
            return save_result.final_path
        else:
            logger.warning(f"Failed to save dashboard: {save_result.error_message}")
            return ""
    
    def create_prediction_analysis_suite(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       model_predictions: Optional[Dict[str, np.ndarray]] = None,
                                       save_name: str = "prediction_analysis_suite") -> str:
        """
        Create comprehensive prediction analysis suite maintaining existing prediction logic.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - analysis suite cannot be created")
            return ""
        
        logger.info("Creating comprehensive prediction analysis suite")
        
        # Create the suite with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Main Time Series Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_series_comparison(y_true, y_pred, ax1, "Actual vs Predicted Time Series")
        
        # 2. Scatter Plot with Regression Line
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_scatter_with_regression(y_true, y_pred, ax2)
        
        # 3. Residuals Analysis
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_residuals_analysis(y_true, y_pred, ax3)
        
        # 4. Error Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_error_distribution(y_true, y_pred, ax4)
        
        # 5. Prediction Accuracy by Quantile
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_accuracy_by_quantile(y_true, y_pred, ax5)
        
        # 6. Rolling Error Analysis
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_rolling_error_analysis(y_true, y_pred, ax6)
        
        # 7. Prediction Confidence Intervals
        ax7 = fig.add_subplot(gs[1, 3])
        self._plot_prediction_intervals(y_true, y_pred, ax7)
        
        # 8. Individual Model Comparison (if available)
        if model_predictions:
            ax8 = fig.add_subplot(gs[2, :])
            self._plot_multi_model_comparison(y_true, model_predictions, ax8)
        else:
            ax8 = fig.add_subplot(gs[2, :])
            ax8.text(0.5, 0.5, 'Individual model predictions not available', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=14)
            ax8.set_title('Multi-Model Comparison')
        
        # 9. Error Metrics Summary Table
        ax9 = fig.add_subplot(gs[3, 0])
        self._create_error_metrics_table(y_true, y_pred, ax9)
        
        # 10. Performance by Time Period
        ax10 = fig.add_subplot(gs[3, 1])
        self._plot_performance_by_period(y_true, y_pred, ax10)
        
        # 11. Directional Accuracy
        ax11 = fig.add_subplot(gs[3, 2])
        self._plot_directional_accuracy(y_true, y_pred, ax11)
        
        # 12. Volatility Analysis
        ax12 = fig.add_subplot(gs[3, 3])
        self._plot_volatility_analysis(y_true, y_pred, ax12)
        
        # Add main title
        fig.suptitle('Comprehensive Prediction Analysis Suite', 
                    fontsize=24, fontweight='bold', y=0.97)
        
        # Save the analysis suite
        save_path = os.path.join(self.plots_dir, f"{save_name}.png")
        save_result = self.file_manager.save_figure(
            fig, save_path,
            metadata={
                "visualization_type": "prediction_analysis_suite",
                "data_points": len(y_true),
                "models_analyzed": len(model_predictions) if model_predictions else 1,
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        plt.show()
        
        if save_result.success:
            logger.info(f"Prediction analysis suite saved to: {save_result.final_path}")
            return save_result.final_path
        else:
            logger.warning(f"Failed to save analysis suite: {save_result.error_message}")
            return ""
    
    def create_feature_importance_analysis(self, 
                                         models_dict: Dict[str, Any],
                                         feature_names: List[str],
                                         save_name: str = "feature_importance_analysis") -> str:
        """
        Create comprehensive feature importance analysis maintaining existing model logic.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - feature analysis cannot be created")
            return ""
        
        logger.info("Creating comprehensive feature importance analysis")
        
        # Extract feature importances from models
        importance_data = {}
        for model_name, model in models_dict.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_data[model_name] = np.abs(model.coef_)
        
        if not importance_data:
            logger.warning("No feature importance data available from models")
            return ""
        
        # Create the analysis with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Top Features Comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_top_features_comparison(importance_data, feature_names, ax1)
        
        # 2. Individual Model Feature Importance
        model_names = list(importance_data.keys())
        for i, (model_name, importance) in enumerate(importance_data.items()):
            if i >= 6:  # Limit to 6 models for space
                break
            row = (i // 3) + 1
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            self._plot_individual_feature_importance(importance, feature_names, model_name, ax)
        
        # Add main title
        fig.suptitle('Comprehensive Feature Importance Analysis', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Save the analysis
        save_path = os.path.join(self.plots_dir, f"{save_name}.png")
        save_result = self.file_manager.save_figure(
            fig, save_path,
            metadata={
                "visualization_type": "feature_importance_analysis",
                "models_analyzed": len(importance_data),
                "total_features": len(feature_names),
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        plt.show()
        
        if save_result.success:
            logger.info(f"Feature importance analysis saved to: {save_result.final_path}")
            return save_result.final_path
        else:
            logger.warning(f"Failed to save feature analysis: {save_result.error_message}")
            return ""
    
    def create_correlation_heatmap_suite(self, 
                                       data: pd.DataFrame, 
                                       save_name: str = "correlation_analysis") -> str:
        """
        Create comprehensive correlation analysis maintaining existing data processing logic.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - correlation analysis cannot be created")
            return ""
        
        logger.info("Creating comprehensive correlation analysis")
        
        # Prepare numeric data maintaining existing logic
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Create the analysis with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Full Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_full_correlation_heatmap(correlation_matrix, ax1)
        
        # 2. High Correlation Pairs
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_high_correlation_pairs(correlation_matrix, ax2)
        
        # 3. Feature Categories Correlation
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_feature_category_correlation(correlation_matrix, ax3, "Equity Features")
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_feature_category_correlation(correlation_matrix, ax4, "Technical Indicators")
        
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_feature_category_correlation(correlation_matrix, ax5, "Options Features")
        
        # 4. Correlation Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_correlation_distribution(correlation_matrix, ax6)
        
        # 5. Hierarchical Clustering
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_correlation_clustering(correlation_matrix, ax7)
        
        # Add main title
        fig.suptitle('Comprehensive Correlation Analysis Suite', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Save the analysis
        save_path = os.path.join(self.plots_dir, f"{save_name}.png")
        save_result = self.file_manager.save_figure(
            fig, save_path,
            metadata={
                "visualization_type": "correlation_analysis",
                "features_analyzed": len(correlation_matrix.columns),
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        plt.show()
        
        if save_result.success:
            logger.info(f"Correlation analysis saved to: {save_result.final_path}")
            return save_result.final_path
        else:
            logger.warning(f"Failed to save correlation analysis: {save_result.error_message}")
            return ""
    
    # Helper methods for individual plot components
    def _plot_metric_comparison(self, metrics_df: pd.DataFrame, metric: str, ax, title: str):
        """Plot metric comparison bar chart."""
        if metric in metrics_df.columns:
            bars = ax.bar(metrics_df.index, metrics_df[metric], color=self.colors['models'][:len(metrics_df)])
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}' if height < 1 else f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
    
    def _create_performance_radar_chart(self, metrics_df: pd.DataFrame, ax):
        """Create radar chart for model performance."""
        # This is a placeholder - radar charts need specialized libraries
        # For now, create a simplified performance comparison
        ax.text(0.5, 0.5, 'Performance Radar Chart\n(Specialized library needed)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Model Performance Radar')
    
    def _create_model_ranking_heatmap(self, metrics_df: pd.DataFrame, ax):
        """Create model ranking heatmap."""
        # Rank models by each metric (lower is better for MSE, MAE, RMSE; higher for R2)
        ranking_df = metrics_df.copy()
        
        # Invert R2 ranking (higher is better)
        for col in ranking_df.columns:
            if col.upper() in ['R2', 'R_SQUARED']:
                ranking_df[col] = ranking_df[col].rank(ascending=False)
            else:
                ranking_df[col] = ranking_df[col].rank(ascending=True)
        
        im = ax.imshow(ranking_df.values, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(ranking_df.columns)))
        ax.set_yticks(range(len(ranking_df.index)))
        ax.set_xticklabels(ranking_df.columns)
        ax.set_yticklabels(ranking_df.index)
        ax.set_title('Model Rankings by Metric (1=Best)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(ranking_df.index)):
            for j in range(len(ranking_df.columns)):
                text = ax.text(j, i, f'{int(ranking_df.iloc[i, j])}',
                             ha="center", va="center", color="black", fontweight='bold')
    
    def _create_statistical_summary_table(self, metrics_df: pd.DataFrame, ax):
        """Create statistical summary table."""
        summary_stats = metrics_df.describe().round(6)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=summary_stats.values,
                        rowLabels=summary_stats.index,
                        colLabels=summary_stats.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.set_title('Statistical Summary', fontweight='bold')
    
    def _highlight_best_models(self, metrics_df: pd.DataFrame, ax):
        """Highlight best performing models."""
        best_models = {}
        for metric in metrics_df.columns:
            if metric.upper() in ['R2', 'R_SQUARED']:
                best_models[metric] = metrics_df[metric].idxmax()
            else:
                best_models[metric] = metrics_df[metric].idxmin()
        
        ax.axis('off')
        y_pos = 0.9
        ax.text(0.5, y_pos, 'Best Models by Metric', ha='center', fontweight='bold', 
               transform=ax.transAxes, fontsize=14)
        
        for metric, model in best_models.items():
            y_pos -= 0.15
            ax.text(0.1, y_pos, f'{metric}: {model}', transform=ax.transAxes, fontsize=10)
    
    def _plot_performance_distribution(self, metrics_df: pd.DataFrame, ax):
        """Plot performance distribution across models."""
        # Create box plot for each metric
        if len(metrics_df.columns) > 0:
            data_to_plot = [metrics_df[col].values for col in metrics_df.columns]
            ax.boxplot(data_to_plot, labels=metrics_df.columns)
            ax.set_title('Performance Distribution', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
    
    def _plot_time_series_comparison(self, y_true: np.ndarray, y_pred: np.ndarray, ax, title: str):
        """Plot time series comparison."""
        ax.plot(y_true, label='Actual', alpha=0.8, linewidth=1.5, color=self.colors['primary'])
        ax.plot(y_pred, label='Predicted', alpha=0.8, linewidth=1.5, color=self.colors['secondary'])
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_scatter_with_regression(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot scatter plot with regression line."""
        ax.scatter(y_true, y_pred, alpha=0.6, s=10, color=self.colors['accent'])
        
        # Add perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), color=self.colors['success'], alpha=0.8, linewidth=2, label='Actual Fit')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted Scatter', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot residuals analysis."""
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=10, color=self.colors['warning'])
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals Plot', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot error distribution."""
        errors = y_true - y_pred
        ax.hist(errors, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {errors.mean():.4f}')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_accuracy_by_quantile(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot accuracy by quantile."""
        # Calculate quantiles and corresponding accuracy
        quantiles = np.linspace(0, 1, 11)
        quantile_values = np.quantile(y_true, quantiles)
        accuracies = []
        
        for i in range(len(quantiles)-1):
            mask = (y_true >= quantile_values[i]) & (y_true < quantile_values[i+1])
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                accuracies.append(mae)
            else:
                accuracies.append(0)
        
        ax.bar(range(len(accuracies)), accuracies, color=self.colors['models'][:len(accuracies)])
        ax.set_xlabel('Deciles')
        ax.set_ylabel('MAE')
        ax.set_title('Accuracy by Quantile', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot rolling error analysis."""
        errors = np.abs(y_true - y_pred)
        window = min(50, len(errors) // 10)  # Adaptive window size
        
        if window > 1:
            rolling_mae = pd.Series(errors).rolling(window=window).mean()
            ax.plot(rolling_mae, color=self.colors['accent'], linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Rolling MAE')
            ax.set_title(f'Rolling Error Analysis (Window: {window})', fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for rolling analysis', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot prediction confidence intervals."""
        # Simple confidence intervals based on prediction error
        errors = y_true - y_pred
        error_std = np.std(errors)
        
        upper_bound = y_pred + 2 * error_std
        lower_bound = y_pred - 2 * error_std
        
        # Plot subset for visibility
        subset_size = min(200, len(y_true))
        indices = np.linspace(0, len(y_true)-1, subset_size, dtype=int)
        
        ax.fill_between(indices, lower_bound[indices], upper_bound[indices], 
                       alpha=0.3, color=self.colors['primary'], label='95% Confidence Interval')
        ax.plot(indices, y_true[indices], label='Actual', color='red', linewidth=2)
        ax.plot(indices, y_pred[indices], label='Predicted', color='blue', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('Prediction Confidence Intervals', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_multi_model_comparison(self, y_true: np.ndarray, model_predictions: Dict[str, np.ndarray], ax):
        """Plot multiple model predictions comparison."""
        # Plot subset for visibility
        subset_size = min(200, len(y_true))
        indices = np.linspace(0, len(y_true)-1, subset_size, dtype=int)
        
        ax.plot(indices, y_true[indices], label='Actual', color='black', linewidth=2, alpha=0.8)
        
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            if len(pred) >= len(y_true):
                aligned_pred = pred[-len(y_true):]
                color = self.colors['models'][i % len(self.colors['models'])]
                ax.plot(indices, aligned_pred[indices], label=model_name, 
                       color=color, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('Multi-Model Predictions Comparison', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _create_error_metrics_table(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Create comprehensive error metrics table."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculate comprehensive metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        metrics_data = [
            ['MSE', f'{mse:.6f}'],
            ['RMSE', f'{rmse:.6f}'],
            ['MAE', f'{mae:.6f}'],
            ['MAPE', f'{mape:.2f}%'],
            ['R²', f'{r2:.6f}']
        ]
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Error Metrics Summary', fontweight='bold')
    
    def _plot_performance_by_period(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot performance by time periods."""
        # Divide data into periods
        n_periods = min(10, len(y_true) // 20)  # Ensure sufficient data per period
        if n_periods < 2:
            ax.text(0.5, 0.5, 'Insufficient data for period analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        period_size = len(y_true) // n_periods
        period_maes = []
        period_labels = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(y_true)
            
            period_true = y_true[start_idx:end_idx]
            period_pred = y_pred[start_idx:end_idx]
            period_mae = np.mean(np.abs(period_true - period_pred))
            
            period_maes.append(period_mae)
            period_labels.append(f'P{i+1}')
        
        ax.bar(period_labels, period_maes, color=self.colors['accent'])
        ax.set_xlabel('Time Period')
        ax.set_ylabel('MAE')
        ax.set_title('Performance by Time Period', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot directional accuracy analysis."""
        if len(y_true) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for directional analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate direction changes
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        # Directional accuracy
        directional_accuracy = np.mean(true_directions == pred_directions) * 100
        
        # Create pie chart
        correct_directions = np.sum(true_directions == pred_directions)
        incorrect_directions = len(true_directions) - correct_directions
        
        ax.pie([correct_directions, incorrect_directions], 
               labels=[f'Correct ({directional_accuracy:.1f}%)', 
                      f'Incorrect ({100-directional_accuracy:.1f}%)'],
               autopct='%1.1f%%', colors=[self.colors['success'], self.colors['warning']])
        ax.set_title('Directional Accuracy', fontweight='bold')
    
    def _plot_volatility_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, ax):
        """Plot volatility analysis."""
        if len(y_true) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for volatility analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate rolling volatility
        window = min(20, len(y_true) // 10)
        if window < 2:
            ax.text(0.5, 0.5, 'Insufficient data for volatility analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        true_volatility = pd.Series(y_true).rolling(window=window).std()
        pred_volatility = pd.Series(y_pred).rolling(window=window).std()
        
        ax.plot(true_volatility, label='Actual Volatility', color=self.colors['primary'])
        ax.plot(pred_volatility, label='Predicted Volatility', color=self.colors['secondary'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.set_title(f'Volatility Analysis (Window: {window})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_top_features_comparison(self, importance_data: Dict, feature_names: List[str], ax):
        """Plot top features comparison across models."""
        # Get top 20 features across all models
        all_importance = {}
        for model_name, importance in importance_data.items():
            for i, feat_name in enumerate(feature_names):
                if i < len(importance):
                    if feat_name not in all_importance:
                        all_importance[feat_name] = {}
                    all_importance[feat_name][model_name] = importance[i]
        
        # Sort by average importance
        avg_importance = {feat: np.mean(list(vals.values())) 
                         for feat, vals in all_importance.items()}
        top_features = sorted(avg_importance.keys(), key=lambda x: avg_importance[x], reverse=True)[:20]
        
        # Create grouped bar chart
        x = np.arange(len(top_features))
        width = 0.8 / len(importance_data)
        
        for i, (model_name, importance) in enumerate(importance_data.items()):
            model_importance = []
            for feat in top_features:
                if feat in all_importance and model_name in all_importance[feat]:
                    model_importance.append(all_importance[feat][model_name])
                else:
                    model_importance.append(0)
            
            offset = (i - len(importance_data)/2) * width + width/2
            ax.bar(x + offset, model_importance, width, label=model_name, 
                  color=self.colors['models'][i % len(self.colors['models'])])
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Top 20 Features Comparison Across Models', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_individual_feature_importance(self, importance: np.ndarray, feature_names: List[str], 
                                          model_name: str, ax):
        """Plot individual model feature importance."""
        # Get top 15 features for this model
        top_indices = np.argsort(importance)[-15:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        bars = ax.barh(range(len(top_features)), top_importance, 
                      color=self.colors['accent'], alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top Features', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    def _plot_full_correlation_heatmap(self, correlation_matrix: pd.DataFrame, ax):
        """Plot full correlation heatmap."""
        # Sample correlation matrix if too large
        if len(correlation_matrix) > 50:
            # Select top 50 most variable features
            variances = correlation_matrix.var().sort_values(ascending=False)
            top_features = variances.head(50).index
            correlation_matrix = correlation_matrix.loc[top_features, top_features]
        
        im = ax.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Set ticks (reduced for readability)
        step = max(1, len(correlation_matrix) // 10)
        tick_positions = range(0, len(correlation_matrix), step)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([correlation_matrix.columns[i][:10] for i in tick_positions], 
                          rotation=45, ha='right')
        ax.set_yticklabels([correlation_matrix.index[i][:10] for i in tick_positions])
    
    def _plot_high_correlation_pairs(self, correlation_matrix: pd.DataFrame, ax):
        """Plot high correlation pairs."""
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append((correlation_matrix.columns[i], 
                                          correlation_matrix.columns[j], 
                                          corr_value))
        
        if high_corr_pairs:
            # Sort by absolute correlation
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pairs = high_corr_pairs[:15]  # Top 15 pairs
            
            pair_labels = [f"{pair[0][:10]}-{pair[1][:10]}" for pair in top_pairs]
            correlations = [pair[2] for pair in top_pairs]
            colors = ['red' if corr > 0 else 'blue' for corr in correlations]
            
            bars = ax.barh(range(len(pair_labels)), correlations, color=colors, alpha=0.7)
            ax.set_yticks(range(len(pair_labels)))
            ax.set_yticklabels(pair_labels)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('High Correlation Pairs (|r| > 0.8)', fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01 if width > 0 else width - 0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left' if width > 0 else 'right', 
                       va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No high correlation pairs found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('High Correlation Pairs')
    
    def _plot_feature_category_correlation(self, correlation_matrix: pd.DataFrame, ax, category: str):
        """Plot correlation for specific feature category."""
        # Filter features by category
        if category == "Equity Features":
            features = [col for col in correlation_matrix.columns 
                       if any(term in col.upper() for term in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']) 
                       and not any(term in col.upper() for term in ['FUT', 'CE', 'PE'])]
        elif category == "Technical Indicators":
            features = [col for col in correlation_matrix.columns 
                       if any(term in col.upper() for term in ['RSI', 'MACD', 'SMA', 'EMA', 'BB', 'ATR'])]
        elif category == "Options Features":
            features = [col for col in correlation_matrix.columns 
                       if any(term in col.upper() for term in ['CE', 'PE', 'CALL', 'PUT', 'IV', 'DELTA'])]
        else:
            features = correlation_matrix.columns[:10]  # Default to first 10
        
        if len(features) > 1:
            category_corr = correlation_matrix.loc[features, features]
            im = ax.imshow(category_corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_title(f'{category} Correlation', fontweight='bold')
            
            # Set ticks
            ax.set_xticks(range(len(features)))
            ax.set_yticks(range(len(features)))
            ax.set_xticklabels([f[:8] for f in features], rotation=45, ha='right')
            ax.set_yticklabels([f[:8] for f in features])
        else:
            ax.text(0.5, 0.5, f'Insufficient {category.lower()}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{category} Correlation')
    
    def _plot_correlation_distribution(self, correlation_matrix: pd.DataFrame, ax):
        """Plot correlation coefficient distribution."""
        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        ax.hist(correlations, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax.axvline(correlations.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {correlations.mean():.3f}')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Correlation Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_clustering(self, correlation_matrix: pd.DataFrame, ax):
        """Plot hierarchical clustering of correlations."""
        # This is a placeholder for hierarchical clustering
        # Would require scipy.cluster.hierarchy
        ax.text(0.5, 0.5, 'Hierarchical Correlation Clustering\n(Requires scipy.cluster.hierarchy)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Feature Clustering by Correlation')


logger.info("Comprehensive visualization utilities loaded successfully!")
