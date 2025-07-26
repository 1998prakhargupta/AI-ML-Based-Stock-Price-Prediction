"""
Visualization Utilities Module
=============================

Comprehensive visualization utilities for the Stock Price Predictor project.
Handles chart generation, dashboards, and interactive visualizations.
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
logger = logging.getLogger(__name__)

class ComprehensiveVisualizer:
    """
    Comprehensive visualization utilities for stock prediction models.
    Provides dashboard creation, chart generation, and reporting visualizations.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_plots_path())
        
        # Plotting settings
        self.plot_style = 'seaborn-v0_8'
        self.figure_size = (12, 8)
        self.dpi = self.config.get_plot_dpi()
        self.format = self.config.get_plot_format()
        
        # Colors and styling
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#F39C12',
            'success': '#27AE60',
            'danger': '#E74C3C',
            'warning': '#F1C40F',
            'info': '#3498DB',
            'light': '#BDC3C7',
            'dark': '#34495E'
        }
        
        self.model_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available. Visualization functions will be disabled.")
        else:
            self._setup_plotting_style()
            logger.info("ComprehensiveVisualizer initialized successfully")
    
    def _setup_plotting_style(self) -> None:
        """Setup consistent plotting style."""
        if not PLOTTING_AVAILABLE:
            return
            
        plt.style.use(self.plot_style)
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def create_model_performance_dashboard(self, 
                                         results_dict: Dict[str, Dict[str, float]],
                                         save_name: str = "model_performance_dashboard") -> Optional[str]:
        """
        Create comprehensive model performance dashboard.
        
        Args:
            results_dict: Dictionary of model results
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for dashboard creation")
            return None
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # Extract metrics
            models = list(results_dict.keys())
            metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
            
            # Prepare data
            metric_data = {metric: [] for metric in metrics}
            for model in models:
                for metric in metrics:
                    metric_data[metric].append(results_dict[model].get(metric, 0))
            
            # 1. Bar chart of all metrics (normalized)
            ax1 = fig.add_subplot(gs[0, :2])
            self._create_normalized_metrics_chart(ax1, metric_data, models)
            
            # 2. R² Score comparison
            ax2 = fig.add_subplot(gs[0, 2])
            self._create_r2_comparison(ax2, metric_data['R2'], models)
            
            # 3. Error metrics comparison
            ax3 = fig.add_subplot(gs[1, :])
            self._create_error_metrics_comparison(ax3, metric_data, models)
            
            # 4. Model ranking
            ax4 = fig.add_subplot(gs[2, :])
            self._create_model_ranking(ax4, results_dict, models)
            
            # Add title
            fig.suptitle('Model Performance Dashboard', fontsize=20, fontweight='bold')
            
            # Save plot
            plot_path = self._save_plot(fig, f"{save_name}_dashboard")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return None
    
    def create_prediction_analysis_suite(self,
                                       y_true: np.ndarray,
                                       y_pred_ensemble: np.ndarray,
                                       predictions_dict: Dict[str, np.ndarray],
                                       save_name: str = "prediction_analysis") -> Optional[str]:
        """
        Create comprehensive prediction analysis suite.
        
        Args:
            y_true: True values
            y_pred_ensemble: Ensemble predictions
            predictions_dict: Individual model predictions
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for prediction analysis")
            return None
        
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Ensemble vs Actual scatter plot
            ax1 = fig.add_subplot(gs[0, :2])
            self._create_prediction_scatter(ax1, y_true, y_pred_ensemble, "Ensemble Predictions")
            
            # 2. Residual plot
            ax2 = fig.add_subplot(gs[0, 2:])
            self._create_residual_plot(ax2, y_true, y_pred_ensemble)
            
            # 3. Individual model predictions
            ax3 = fig.add_subplot(gs[1, :])
            self._create_model_predictions_comparison(ax3, y_true, predictions_dict)
            
            # 4. Error distribution
            ax4 = fig.add_subplot(gs[2, :2])
            self._create_error_distribution(ax4, y_true, y_pred_ensemble)
            
            # 5. Prediction intervals
            ax5 = fig.add_subplot(gs[2, 2:])
            self._create_prediction_intervals(ax5, y_true, y_pred_ensemble, predictions_dict)
            
            # 6. Time series plot (if applicable)
            ax6 = fig.add_subplot(gs[3, :])
            self._create_time_series_predictions(ax6, y_true, y_pred_ensemble, predictions_dict)
            
            fig.suptitle('Prediction Analysis Suite', fontsize=20, fontweight='bold')
            
            # Save plot
            plot_path = self._save_plot(fig, f"{save_name}_suite")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating prediction analysis: {e}")
            return None
    
    def create_feature_importance_analysis(self,
                                         models_dict: Dict[str, Any],
                                         feature_names: List[str],
                                         save_name: str = "feature_importance") -> Optional[str]:
        """
        Create feature importance analysis across models.
        
        Args:
            models_dict: Dictionary of trained models
            feature_names: List of feature names
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for feature importance analysis")
            return None
        
        try:
            # Extract feature importances
            importance_data = {}
            for model_name, model in models_dict.items():
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance_data[model_name] = np.abs(model.coef_)
            
            if not importance_data:
                logger.warning("No feature importance data available")
                return None
            
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Combined feature importance heatmap
            ax1 = fig.add_subplot(gs[0, :])
            self._create_feature_importance_heatmap(ax1, importance_data, feature_names)
            
            # 2. Top features by model
            ax2 = fig.add_subplot(gs[1, :])
            self._create_top_features_comparison(ax2, importance_data, feature_names, top_n=15)
            
            # 3. Feature importance distribution
            ax3 = fig.add_subplot(gs[2, 0])
            self._create_importance_distribution(ax3, importance_data, feature_names)
            
            # 4. Feature consensus ranking
            ax4 = fig.add_subplot(gs[2, 1])
            self._create_consensus_ranking(ax4, importance_data, feature_names, top_n=10)
            
            fig.suptitle('Feature Importance Analysis', fontsize=20, fontweight='bold')
            
            # Save plot
            plot_path = self._save_plot(fig, f"{save_name}_analysis")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating feature importance analysis: {e}")
            return None
    
    def create_correlation_heatmap_suite(self,
                                       data: pd.DataFrame,
                                       save_name: str = "correlation_analysis") -> Optional[str]:
        """
        Create comprehensive correlation analysis suite.
        
        Args:
            data: DataFrame for correlation analysis
            save_name: Name for saved plot
            
        Returns:
            Path to saved plot or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available for correlation analysis")
            return None
        
        try:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                logger.warning("No numeric data available for correlation analysis")
                return None
            
            correlation_matrix = numeric_data.corr()
            
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Full correlation heatmap
            ax1 = fig.add_subplot(gs[0, :])
            self._create_correlation_heatmap(ax1, correlation_matrix, "Full Correlation Matrix")
            
            # 2. High correlation pairs
            ax2 = fig.add_subplot(gs[1, :2])
            self._create_high_correlation_analysis(ax2, correlation_matrix)
            
            # 3. Correlation distribution
            ax3 = fig.add_subplot(gs[1, 2])
            self._create_correlation_distribution(ax3, correlation_matrix)
            
            # 4. Correlation clusters
            ax4 = fig.add_subplot(gs[2, :])
            self._create_correlation_clusters(ax4, correlation_matrix)
            
            fig.suptitle('Correlation Analysis Suite', fontsize=20, fontweight='bold')
            
            # Save plot
            plot_path = self._save_plot(fig, f"{save_name}_suite")
            plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating correlation analysis: {e}")
            return None
    
    # Helper methods for chart creation
    def _create_normalized_metrics_chart(self, ax, metric_data: Dict, models: List[str]) -> None:
        """Create normalized metrics comparison chart."""
        # Normalize metrics for comparison (0-1 scale)
        normalized_data = {}
        for metric, values in metric_data.items():
            if metric == 'R2':  # R² should be higher
                normalized_data[metric] = [(v + 1) / 2 for v in values]  # Transform from [-1,1] to [0,1]
            else:  # Error metrics should be lower
                max_val = max(values) if max(values) > 0 else 1
                normalized_data[metric] = [1 - (v / max_val) for v in values]
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, (metric, values) in enumerate(normalized_data.items()):
            ax.bar(x + i * width, values, width, label=metric, color=self.model_colors[i % len(self.model_colors)])
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Normalized Performance (Higher is Better)')
        ax.set_title('Normalized Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_r2_comparison(self, ax, r2_scores: List[float], models: List[str]) -> None:
        """Create R² score comparison chart."""
        colors = [self.colors['success'] if r2 > 0.8 else 
                 self.colors['warning'] if r2 > 0.6 else 
                 self.colors['danger'] for r2 in r2_scores]
        
        bars = ax.bar(models, r2_scores, color=colors)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_ylim(0, max(1, max(r2_scores) * 1.1))
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _create_error_metrics_comparison(self, ax, metric_data: Dict, models: List[str]) -> None:
        """Create error metrics comparison chart."""
        error_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(error_metrics):
            if metric in metric_data:
                values = metric_data[metric]
                ax.bar(x + i * width, values, width, label=metric, 
                      color=self.model_colors[i % len(self.model_colors)])
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Error Value')
        ax.set_title('Error Metrics Comparison (Lower is Better)')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_model_ranking(self, ax, results_dict: Dict, models: List[str]) -> None:
        """Create model ranking based on multiple criteria."""
        # Calculate composite score (weighted combination of metrics)
        scores = []
        for model in models:
            result = results_dict[model]
            # Higher R² is better, lower errors are better
            r2_score = result.get('R2', 0) * 0.4
            mse_score = (1 / (1 + result.get('MSE', float('inf')))) * 0.3
            mae_score = (1 / (1 + result.get('MAE', float('inf')))) * 0.3
            
            composite_score = r2_score + mse_score + mae_score
            scores.append(composite_score)
        
        # Sort by score
        model_scores = list(zip(models, scores))
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        sorted_models, sorted_scores = zip(*model_scores)
        
        colors = [self.colors['success'] if i < 3 else 
                 self.colors['warning'] if i < 6 else 
                 self.colors['danger'] for i in range(len(sorted_models))]
        
        bars = ax.barh(sorted_models, sorted_scores, color=colors)
        ax.set_xlabel('Composite Performance Score')
        ax.set_title('Model Ranking (Higher is Better)')
        
        # Add rank labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                   f'#{i+1}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _create_prediction_scatter(self, ax, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        """Create prediction vs actual scatter plot."""
        ax.scatter(y_true, y_pred, alpha=0.6, color=self.colors['primary'])
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_residual_plot(self, ax, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Create residual plot."""
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, color=self.colors['secondary'])
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
    
    def _create_model_predictions_comparison(self, ax, y_true: np.ndarray, predictions_dict: Dict) -> None:
        """Create comparison of all model predictions."""
        # Sample points for visualization if too many
        n_points = min(100, len(y_true))
        indices = np.random.choice(len(y_true), n_points, replace=False)
        
        x = np.arange(n_points)
        ax.plot(x, y_true[indices], 'ko-', label='Actual', linewidth=2, markersize=4)
        
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = self.model_colors[i % len(self.model_colors)]
            ax.plot(x, predictions[indices], 'o-', label=model_name, 
                   color=color, alpha=0.7, linewidth=1, markersize=3)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Model Predictions Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _create_error_distribution(self, ax, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Create error distribution histogram."""
        errors = y_true - y_pred
        ax.hist(errors, bins=30, alpha=0.7, color=self.colors['info'], edgecolor='black')
        ax.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_prediction_intervals(self, ax, y_true: np.ndarray, y_pred: np.ndarray, predictions_dict: Dict) -> None:
        """Create prediction intervals visualization."""
        # Calculate prediction intervals from model variance
        all_predictions = np.array([pred for pred in predictions_dict.values()])
        pred_mean = np.mean(all_predictions, axis=0)
        pred_std = np.std(all_predictions, axis=0)
        
        # Sample points for visualization
        n_points = min(50, len(y_true))
        indices = np.random.choice(len(y_true), n_points, replace=False)
        
        x = np.arange(n_points)
        ax.plot(x, y_true[indices], 'ko-', label='Actual', linewidth=2)
        ax.plot(x, pred_mean[indices], 'ro-', label='Ensemble Mean', linewidth=2)
        
        # Confidence intervals
        ax.fill_between(x, 
                       (pred_mean - 2*pred_std)[indices],
                       (pred_mean + 2*pred_std)[indices],
                       alpha=0.3, color=self.colors['warning'], label='95% Interval')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_time_series_predictions(self, ax, y_true: np.ndarray, y_pred: np.ndarray, predictions_dict: Dict) -> None:
        """Create time series prediction visualization."""
        # Show recent predictions
        n_points = min(100, len(y_true))
        recent_indices = slice(-n_points, None)
        
        x = np.arange(n_points)
        ax.plot(x, y_true[recent_indices], 'k-', label='Actual', linewidth=2)
        ax.plot(x, y_pred[recent_indices], 'r-', label='Ensemble', linewidth=2)
        
        # Show individual models with transparency
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = self.model_colors[i % len(self.model_colors)]
            ax.plot(x, predictions[recent_indices], '-', 
                   color=color, alpha=0.5, linewidth=1, label=model_name)
        
        ax.set_xlabel('Time Steps (Recent)')
        ax.set_ylabel('Value')
        ax.set_title('Recent Predictions Time Series')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _create_feature_importance_heatmap(self, ax, importance_data: Dict, feature_names: List[str]) -> None:
        """Create feature importance heatmap across models."""
        # Create matrix of importances
        models = list(importance_data.keys())
        importance_matrix = np.array([importance_data[model] for model in models])
        
        # Select top features for visualization
        mean_importance = np.mean(importance_matrix, axis=0)
        top_indices = np.argsort(mean_importance)[-20:]  # Top 20 features
        
        heatmap_data = importance_matrix[:, top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title('Feature Importance Heatmap (Top 20 Features)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _create_top_features_comparison(self, ax, importance_data: Dict, feature_names: List[str], top_n: int = 15) -> None:
        """Create top features comparison across models."""
        models = list(importance_data.keys())
        
        # Get top features for each model
        all_top_features = set()
        for model in models:
            top_indices = np.argsort(importance_data[model])[-top_n:]
            top_features = [feature_names[i] for i in top_indices]
            all_top_features.update(top_features)
        
        # Create comparison matrix
        feature_list = list(all_top_features)[:top_n]  # Limit for visualization
        comparison_matrix = np.zeros((len(models), len(feature_list)))
        
        for i, model in enumerate(models):
            for j, feature in enumerate(feature_list):
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                    comparison_matrix[i, j] = importance_data[model][feature_idx]
        
        # Create grouped bar chart
        x = np.arange(len(feature_list))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            ax.bar(x + i * width, comparison_matrix[i], width, 
                  label=model, color=self.model_colors[i % len(self.model_colors)])
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title(f'Top {top_n} Features Comparison')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(feature_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_importance_distribution(self, ax, importance_data: Dict, feature_names: List[str]) -> None:
        """Create feature importance distribution."""
        all_importances = []
        for importances in importance_data.values():
            all_importances.extend(importances)
        
        ax.hist(all_importances, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Importance Distribution')
        ax.grid(True, alpha=0.3)
    
    def _create_consensus_ranking(self, ax, importance_data: Dict, feature_names: List[str], top_n: int = 10) -> None:
        """Create consensus feature ranking."""
        # Calculate mean importance across models
        importance_matrix = np.array([importance_data[model] for model in importance_data.keys()])
        mean_importance = np.mean(importance_matrix, axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_importance)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = mean_importance[top_indices]
        
        # Sort for display
        sorted_pairs = sorted(zip(top_features, top_importances), key=lambda x: x[1])
        features, importances = zip(*sorted_pairs)
        
        bars = ax.barh(features, importances, color=self.colors['success'])
        ax.set_xlabel('Mean Importance')
        ax.set_title(f'Top {top_n} Features (Consensus Ranking)')
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            ax.text(importance + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center')
        
        ax.grid(True, alpha=0.3)
    
    def _create_correlation_heatmap(self, ax, correlation_matrix: pd.DataFrame, title: str) -> None:
        """Create correlation heatmap."""
        # Select subset for visualization if too large
        if correlation_matrix.shape[0] > 50:
            # Select most variable features
            variances = correlation_matrix.var().sort_values(ascending=False)
            top_features = variances.head(50).index
            correlation_matrix = correlation_matrix.loc[top_features, top_features]
        
        im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.index)
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _create_high_correlation_analysis(self, ax, correlation_matrix: pd.DataFrame) -> None:
        """Analyze high correlation pairs."""
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Plot top correlations
        top_pairs = high_corr_pairs[:20]  # Top 20
        if top_pairs:
            pairs_labels = [f"{pair[0][:10]}\\n{pair[1][:10]}" for pair in top_pairs]
            correlations = [pair[2] for pair in top_pairs]
            
            colors = [self.colors['danger'] if abs(c) > 0.9 else 
                     self.colors['warning'] if abs(c) > 0.8 else 
                     self.colors['info'] for c in correlations]
            
            bars = ax.barh(pairs_labels, correlations, color=colors)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('High Correlation Pairs')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, corr in zip(bars, correlations):
                ax.text(corr + 0.01 if corr > 0 else corr - 0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center')
        else:
            ax.text(0.5, 0.5, 'No high correlations found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('High Correlation Pairs')
    
    def _create_correlation_distribution(self, ax, correlation_matrix: pd.DataFrame) -> None:
        """Create correlation coefficient distribution."""
        # Get upper triangle correlations (avoid duplicates and self-correlations)
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        ax.hist(correlations, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax.axvline(np.mean(correlations), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(correlations):.3f}')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Correlation Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_correlation_clusters(self, ax, correlation_matrix: pd.DataFrame) -> None:
        """Create correlation clustering visualization."""
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            distance_matrix = 1 - abs(correlation_matrix)
            condensed_distances = squareform(distance_matrix)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Create dendrogram
            dendrogram(linkage_matrix, labels=correlation_matrix.columns, 
                      ax=ax, orientation='bottom', leaf_rotation=90)
            ax.set_title('Feature Correlation Clustering')
            ax.set_ylabel('Distance')
            
        except ImportError:
            ax.text(0.5, 0.5, 'Scipy required for clustering visualization', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Correlation Clustering (Scipy Required)')
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save plot with proper formatting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{filename}_{timestamp}.{self.format}"
        
        save_result = self.file_manager.save_file(
            fig, save_name,
            metadata={
                "plot_type": "visualization",
                "format": self.format,
                "dpi": self.dpi,
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Plot saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save plot: {save_result.error_message}")
            return ""


logger.info("Comprehensive visualization utilities loaded successfully")
