"""
Automated reporting utilities for stock prediction models.
Maintains all underlying basic logic while providing comprehensive automated reporting.
"""

import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import safe file management and existing utilities
from src.utils.file_management_utils import SafeFileManager, SaveStrategy
from src.utils.app_config import Config
from src.visualization.visualization_utils import ComprehensiveVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceReport:
    """Data class for model performance report."""
    model_name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    training_time: float
    feature_count: int
    data_points: int

@dataclass
class ComprehensiveReport:
    """Data class for comprehensive analysis report."""
    report_id: str
    timestamp: str
    model_reports: List[ModelPerformanceReport]
    ensemble_metrics: Dict[str, float]
    data_summary: Dict[str, Any]
    visualizations_paths: List[str]
    recommendations: List[str]

class AutomatedReportGenerator:
    """
    Comprehensive automated report generator that maintains all existing model logic
    while providing detailed analysis and recommendations.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.file_manager = SafeFileManager(self.config.get_data_save_path())
        self.visualizer = ComprehensiveVisualizer(self.config)
        
        # Report settings
        self.reports_dir = os.path.join(self.config.get_data_save_path(), 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Templates and formatting
        self.report_template = self._load_report_template()
        
        logger.info("Automated report generator initialized")
    
    def generate_comprehensive_model_report(self,
                                          models_dict: Dict[str, Any],
                                          results_dict: Dict[str, Dict[str, float]],
                                          predictions_dict: Dict[str, np.ndarray],
                                          y_true: np.ndarray,
                                          y_pred_ensemble: np.ndarray,
                                          training_data: pd.DataFrame,
                                          feature_names: List[str],
                                          report_name: str = "comprehensive_model_analysis") -> str:
        """
        Generate comprehensive model analysis report maintaining all existing logic.
        
        Args:
            models_dict: Dictionary of trained models
            results_dict: Dictionary of model performance metrics
            predictions_dict: Dictionary of model predictions
            y_true: Actual values
            y_pred_ensemble: Ensemble predictions
            training_data: Training dataset
            feature_names: List of feature names
            report_name: Base name for report files
            
        Returns:
            Path to generated report
        """
        logger.info("Generating comprehensive model analysis report")
        
        # Generate unique report ID
        report_id = f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create model performance reports maintaining existing logic
        model_reports = []
        for model_name, metrics in results_dict.items():
            if model_name in predictions_dict:
                model_report = ModelPerformanceReport(
                    model_name=model_name,
                    metrics=metrics,
                    predictions=predictions_dict[model_name],
                    training_time=0.0,  # Would need to be tracked during training
                    feature_count=len(feature_names),
                    data_points=len(y_true)
                )
                model_reports.append(model_report)
        
        # Calculate ensemble metrics maintaining existing logic
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        ensemble_metrics = {
            'MSE': mean_squared_error(y_true, y_pred_ensemble),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_ensemble)),
            'MAE': mean_absolute_error(y_true, y_pred_ensemble),
            'R2': r2_score(y_true, y_pred_ensemble),
            'MAPE': np.mean(np.abs((y_true - y_pred_ensemble) / y_true)) * 100
        }
        
        # Generate data summary maintaining existing logic
        data_summary = self._generate_data_summary(training_data, y_true, feature_names)
        
        # Generate visualizations
        visualization_paths = self._generate_report_visualizations(
            models_dict, results_dict, predictions_dict, y_true, y_pred_ensemble,
            training_data, feature_names, report_id
        )
        
        # Generate recommendations based on analysis
        recommendations = self._generate_model_recommendations(
            results_dict, ensemble_metrics, data_summary
        )
        
        # Create comprehensive report object
        comprehensive_report = ComprehensiveReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            model_reports=model_reports,
            ensemble_metrics=ensemble_metrics,
            data_summary=data_summary,
            visualizations_paths=visualization_paths,
            recommendations=recommendations
        )
        
        # Generate report files
        report_paths = self._generate_report_files(comprehensive_report)
        
        logger.info(f"Comprehensive report generated: {report_paths['html']}")
        return report_paths['html']
    
    def generate_prediction_performance_report(self,
                                             y_true: np.ndarray,
                                             y_pred: np.ndarray,
                                             model_name: str = "Model",
                                             additional_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Generate focused prediction performance report maintaining existing evaluation logic.
        """
        logger.info(f"Generating prediction performance report for {model_name}")
        
        report_id = f"prediction_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate comprehensive metrics maintaining existing logic
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Generate prediction analysis visualization
        viz_path = self.visualizer.create_prediction_analysis_suite(
            y_true, y_pred, save_name=f"prediction_analysis_{report_id}"
        )
        
        # Generate HTML report
        html_report = self._generate_prediction_html_report(
            report_id, model_name, metrics, viz_path
        )
        
        # Save report
        report_path = os.path.join(self.reports_dir, f"{report_id}.html")
        save_result = self.file_manager.save_file(
            html_report, report_path,
            metadata={
                "report_type": "prediction_performance",
                "model_name": model_name,
                "data_points": len(y_true),
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Prediction performance report saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save prediction report: {save_result.error_message}")
            return ""
    
    def generate_feature_analysis_report(self,
                                       models_dict: Dict[str, Any],
                                       feature_names: List[str],
                                       training_data: pd.DataFrame) -> str:
        """
        Generate comprehensive feature analysis report maintaining existing feature logic.
        """
        logger.info("Generating feature analysis report")
        
        report_id = f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate feature importance analysis
        viz_path = self.visualizer.create_feature_importance_analysis(
            models_dict, feature_names, save_name=f"feature_importance_{report_id}"
        )
        
        # Generate correlation analysis
        corr_viz_path = self.visualizer.create_correlation_heatmap_suite(
            training_data, save_name=f"correlation_analysis_{report_id}"
        )
        
        # Create feature analysis
        feature_analysis = self._create_feature_analysis(models_dict, feature_names, training_data)
        
        # Generate HTML report
        html_report = self._generate_feature_html_report(
            report_id, feature_analysis, [viz_path, corr_viz_path]
        )
        
        # Save report
        report_path = os.path.join(self.reports_dir, f"{report_id}.html")
        save_result = self.file_manager.save_file(
            html_report, report_path,
            metadata={
                "report_type": "feature_analysis",
                "features_count": len(feature_names),
                "models_analyzed": len(models_dict),
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Feature analysis report saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save feature report: {save_result.error_message}")
            return ""
    
    def generate_executive_summary(self,
                                 comprehensive_report: ComprehensiveReport) -> str:
        """
        Generate executive summary maintaining key business insights from existing logic.
        """
        logger.info("Generating executive summary")
        
        # Create executive summary content
        summary_content = self._create_executive_summary_content(comprehensive_report)
        
        # Generate HTML executive summary
        html_summary = self._generate_executive_summary_html(
            comprehensive_report.report_id, summary_content
        )
        
        # Save executive summary
        summary_path = os.path.join(self.reports_dir, f"executive_summary_{comprehensive_report.report_id}.html")
        save_result = self.file_manager.save_file(
            html_summary, summary_path,
            metadata={
                "report_type": "executive_summary",
                "base_report_id": comprehensive_report.report_id,
                "creation_timestamp": datetime.now().isoformat()
            }
        )
        
        if save_result.success:
            logger.info(f"Executive summary saved: {save_result.filepath}")
            return save_result.filepath
        else:
            logger.error(f"Failed to save executive summary: {save_result.error_message}")
            return ""
    
    def _generate_data_summary(self, training_data: pd.DataFrame, y_true: np.ndarray, 
                             feature_names: List[str]) -> Dict[str, Any]:
        """Generate data summary maintaining existing data processing logic."""
        return {
            'total_records': len(training_data),
            'total_features': len(feature_names),
            'prediction_records': len(y_true),
            'date_range': {
                'start': training_data.index.min() if hasattr(training_data.index, 'min') else 'N/A',
                'end': training_data.index.max() if hasattr(training_data.index, 'max') else 'N/A'
            },
            'target_statistics': {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true)),
                'median': float(np.median(y_true))
            },
            'feature_categories': self._categorize_features(feature_names),
            'data_quality': self._assess_data_quality(training_data)
        }
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features maintaining existing categorization logic."""
        categories = {
            'equity_features': 0,
            'futures_features': 0,
            'options_features': 0,
            'technical_indicators': 0,
            'index_features': 0,
            'scaled_features': 0,
            'other_features': 0
        }
        
        for feature in feature_names:
            feature_upper = feature.upper()
            
            # Equity features
            if any(term in feature_upper for term in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']) and \
               not any(term in feature_upper for term in ['FUT', 'CE', 'PE']):
                categories['equity_features'] += 1
            # Futures features
            elif 'FUT' in feature_upper:
                categories['futures_features'] += 1
            # Options features
            elif any(term in feature_upper for term in ['CE', 'PE', 'CALL', 'PUT', 'IV', 'DELTA']):
                categories['options_features'] += 1
            # Technical indicators
            elif any(term in feature_upper for term in ['RSI', 'MACD', 'SMA', 'EMA', 'BB', 'ATR', 'ADX']):
                categories['technical_indicators'] += 1
            # Index features
            elif any(term in feature_upper for term in ['NIFTY', 'SENSEX', 'BANK']):
                categories['index_features'] += 1
            # Scaled features
            elif any(term in feature for term in ['_minmax', '_robust', '_z', '_std']):
                categories['scaled_features'] += 1
            else:
                categories['other_features'] += 1
        
        return categories
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality maintaining existing validation logic."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        return {
            'completeness': float((1 - numeric_data.isnull().sum().sum() / (len(numeric_data) * len(numeric_data.columns))) * 100),
            'missing_values': int(numeric_data.isnull().sum().sum()),
            'duplicate_rows': int(numeric_data.duplicated().sum()),
            'memory_usage_mb': float(data.memory_usage(deep=True).sum() / (1024**2)),
            'numeric_columns': len(numeric_data.columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns)
        }
    
    def _generate_report_visualizations(self, models_dict: Dict, results_dict: Dict, 
                                      predictions_dict: Dict, y_true: np.ndarray, 
                                      y_pred_ensemble: np.ndarray, training_data: pd.DataFrame,
                                      feature_names: List[str], report_id: str) -> List[str]:
        """Generate all visualizations for the report."""
        viz_paths = []
        
        # Model performance dashboard
        dashboard_path = self.visualizer.create_model_performance_dashboard(
            results_dict, save_name=f"dashboard_{report_id}"
        )
        if dashboard_path:
            viz_paths.append(dashboard_path)
        
        # Prediction analysis suite
        pred_analysis_path = self.visualizer.create_prediction_analysis_suite(
            y_true, y_pred_ensemble, predictions_dict, save_name=f"predictions_{report_id}"
        )
        if pred_analysis_path:
            viz_paths.append(pred_analysis_path)
        
        # Feature importance analysis
        feature_path = self.visualizer.create_feature_importance_analysis(
            models_dict, feature_names, save_name=f"features_{report_id}"
        )
        if feature_path:
            viz_paths.append(feature_path)
        
        # Correlation analysis
        corr_path = self.visualizer.create_correlation_heatmap_suite(
            training_data, save_name=f"correlations_{report_id}"
        )
        if corr_path:
            viz_paths.append(corr_path)
        
        return viz_paths
    
    def _generate_model_recommendations(self, results_dict: Dict, ensemble_metrics: Dict,
                                      data_summary: Dict) -> List[str]:
        """Generate model recommendations based on analysis."""
        recommendations = []
        
        # Model performance recommendations
        best_model = min(results_dict.items(), key=lambda x: x[1]['RMSE'])
        
        recommendations.append(f"Best performing individual model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.6f})")
        
        if ensemble_metrics['RMSE'] < best_model[1]['RMSE']:
            improvement = ((best_model[1]['RMSE'] - ensemble_metrics['RMSE']) / best_model[1]['RMSE']) * 100
            recommendations.append(f"Ensemble model improves over best individual by {improvement:.2f}%")
        
        # R-squared recommendations
        if ensemble_metrics['R2'] > 0.8:
            recommendations.append("Excellent model fit with R² > 0.8 - suitable for production use")
        elif ensemble_metrics['R2'] > 0.6:
            recommendations.append("Good model fit with R² > 0.6 - consider additional features for improvement")
        else:
            recommendations.append("Model fit below 0.6 - significant improvement needed before production")
        
        # Data quality recommendations
        if data_summary['data_quality']['completeness'] < 95:
            recommendations.append("Data completeness below 95% - investigate missing value patterns")
        
        if data_summary['data_quality']['duplicate_rows'] > 0:
            recommendations.append("Duplicate rows detected - clean data for better model performance")
        
        # Feature recommendations
        feature_count = data_summary['total_features']
        if feature_count > 1000:
            recommendations.append("High feature count - consider feature selection techniques")
        elif feature_count < 10:
            recommendations.append("Low feature count - consider feature engineering for better performance")
        
        return recommendations
    
    def _generate_report_files(self, report: ComprehensiveReport) -> Dict[str, str]:
        """Generate all report files."""
        report_paths = {}
        
        # Generate HTML report
        html_content = self._generate_comprehensive_html_report(report)
        html_path = os.path.join(self.reports_dir, f"{report.report_id}.html")
        
        html_save_result = self.file_manager.save_file(
            html_content, html_path,
            metadata={
                "report_type": "comprehensive_analysis",
                "models_count": len(report.model_reports),
                "creation_timestamp": report.timestamp
            }
        )
        
        if html_save_result.success:
            report_paths['html'] = html_save_result.filepath
        
        # Generate JSON report for programmatic access
        json_content = self._generate_json_report(report)
        json_path = os.path.join(self.reports_dir, f"{report.report_id}.json")
        
        json_save_result = self.file_manager.save_file(
            json_content, json_path,
            metadata={
                "report_type": "comprehensive_analysis_json",
                "creation_timestamp": report.timestamp
            }
        )
        
        if json_save_result.success:
            report_paths['json'] = json_save_result.filepath
        
        # Generate executive summary
        exec_summary_path = self.generate_executive_summary(report)
        if exec_summary_path:
            report_paths['executive_summary'] = exec_summary_path
        
        return report_paths
    
    def _create_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create detailed prediction analysis."""
        errors = y_true - y_pred
        absolute_errors = np.abs(errors)
        
        return {
            'basic_metrics': metrics,
            'error_analysis': {
                'mean_error': float(np.mean(errors)),
                'error_std': float(np.std(errors)),
                'max_absolute_error': float(np.max(absolute_errors)),
                'error_skewness': float(self._calculate_skewness(errors)),
                'error_kurtosis': float(self._calculate_kurtosis(errors))
            },
            'directional_accuracy': self._calculate_directional_accuracy(y_true, y_pred),
            'performance_by_magnitude': self._analyze_performance_by_magnitude(y_true, y_pred),
            'temporal_analysis': self._analyze_temporal_performance(y_true, y_pred)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate directional accuracy maintaining existing logic."""
        if len(y_true) < 2:
            return {'directional_accuracy': 0.0, 'up_accuracy': 0.0, 'down_accuracy': 0.0}
        
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        directional_accuracy = np.mean(true_directions == pred_directions) * 100
        
        # Separate up and down accuracy
        up_mask = true_directions == True
        down_mask = true_directions == False
        
        up_accuracy = np.mean(true_directions[up_mask] == pred_directions[up_mask]) * 100 if up_mask.sum() > 0 else 0
        down_accuracy = np.mean(true_directions[down_mask] == pred_directions[down_mask]) * 100 if down_mask.sum() > 0 else 0
        
        return {
            'directional_accuracy': float(directional_accuracy),
            'up_accuracy': float(up_accuracy),
            'down_accuracy': float(down_accuracy)
        }
    
    def _analyze_performance_by_magnitude(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze performance by prediction magnitude."""
        # Divide predictions into quartiles
        quartiles = np.percentile(y_true, [25, 50, 75])
        
        performance = {}
        for i, threshold in enumerate([0, quartiles[0], quartiles[1], quartiles[2], np.max(y_true)]):
            if i == 0:
                continue
            
            if i == 1:
                mask = y_true <= threshold
                label = 'Q1_performance'
            elif i == 2:
                mask = (y_true > quartiles[0]) & (y_true <= threshold)
                label = 'Q2_performance'
            elif i == 3:
                mask = (y_true > quartiles[1]) & (y_true <= threshold)
                label = 'Q3_performance'
            else:
                mask = y_true > quartiles[2]
                label = 'Q4_performance'
            
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                performance[label] = float(mae)
        
        return performance
    
    def _analyze_temporal_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze temporal performance patterns."""
        # Analyze performance over time periods
        n_periods = min(10, len(y_true) // 20)
        if n_periods < 2:
            return {'temporal_stability': 0.0}
        
        period_size = len(y_true) // n_periods
        period_performances = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(y_true)
            
            period_true = y_true[start_idx:end_idx]
            period_pred = y_pred[start_idx:end_idx]
            period_mae = np.mean(np.abs(period_true - period_pred))
            period_performances.append(period_mae)
        
        # Calculate temporal stability (lower std indicates more stable performance)
        temporal_stability = 1 / (1 + np.std(period_performances))
        
        return {
            'temporal_stability': float(temporal_stability),
            'performance_trend': float(np.corrcoef(range(len(period_performances)), period_performances)[0, 1])
        }
    
    def _create_feature_analysis(self, models_dict: Dict, feature_names: List[str], 
                               training_data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive feature analysis."""
        # Calculate feature statistics
        numeric_data = training_data.select_dtypes(include=[np.number])
        feature_stats = {}
        
        for feature in feature_names:
            if feature in numeric_data.columns:
                feature_stats[feature] = {
                    'mean': float(numeric_data[feature].mean()),
                    'std': float(numeric_data[feature].std()),
                    'missing_rate': float(numeric_data[feature].isnull().mean()),
                    'unique_values': int(numeric_data[feature].nunique())
                }
        
        # Extract feature importances
        importance_analysis = {}
        for model_name, model in models_dict.items():
            if hasattr(model, 'feature_importances_'):
                importance_analysis[model_name] = {
                    'top_10_features': self._get_top_features(model.feature_importances_, feature_names, 10),
                    'importance_concentration': float(np.sum(np.sort(model.feature_importances_)[-10:]) / np.sum(model.feature_importances_))
                }
        
        return {
            'feature_statistics': feature_stats,
            'importance_analysis': importance_analysis,
            'feature_categories': self._categorize_features(feature_names),
            'correlation_summary': self._create_correlation_summary(numeric_data)
        }
    
    def _get_top_features(self, importances: np.ndarray, feature_names: List[str], n: int) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        top_indices = np.argsort(importances)[-n:]
        return [(feature_names[i], float(importances[i])) for i in reversed(top_indices)]
    
    def _create_correlation_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation summary."""
        if len(data.columns) > 1:
            corr_matrix = data.corr()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], float(corr_value)))
            
            return {
                'high_correlation_pairs': high_corr_pairs[:20],  # Top 20 pairs
                'correlation_stats': {
                    'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                    'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
                    'min_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min())
                }
            }
        else:
            return {'high_correlation_pairs': [], 'correlation_stats': {}}
    
    def _load_report_template(self) -> str:
        """Load HTML report template."""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; border-bottom: 3px solid #2E86AB; padding-bottom: 20px; margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-left: 4px solid #2E86AB; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #2E86AB; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }}
                .recommendation {{ background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; }}
                .warning {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                .table th {{ background-color: #f8f9fa; font-weight: bold; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                {content}
                <div class="footer">
                    <p>Generated on {timestamp}</p>
                    <p>Automated Stock Prediction Analysis Report</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def _generate_comprehensive_html_report(self, report: ComprehensiveReport) -> str:
        """Generate comprehensive HTML report."""
        # Header section
        content = f'''
        <div class="header">
            <h1>Comprehensive Stock Prediction Analysis Report</h1>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.timestamp}</p>
        </div>
        '''
        
        # Executive Summary Section
        content += '''
        <div class="section">
            <h2>📊 Executive Summary</h2>
        '''
        
        # Ensemble metrics
        for metric, value in report.ensemble_metrics.items():
            content += f'''
            <div class="metric">
                <div class="metric-value">{value:.6f}</div>
                <div class="metric-label">{metric}</div>
            </div>
            '''
        
        content += '</div>'
        
        # Model Performance Section
        content += '''
        <div class="section">
            <h2>🤖 Model Performance Comparison</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>MSE</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>R²</th>
                        <th>MAPE (%)</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for model_report in report.model_reports:
            metrics = model_report.metrics
            content += f'''
                    <tr>
                        <td><strong>{model_report.model_name}</strong></td>
                        <td>{metrics.get('MSE', 'N/A')}</td>
                        <td>{metrics.get('RMSE', 'N/A')}</td>
                        <td>{metrics.get('MAE', 'N/A')}</td>
                        <td>{metrics.get('R2', 'N/A')}</td>
                        <td>{metrics.get('MAPE', 'N/A')}</td>
                    </tr>
            '''
        
        content += '''
                </tbody>
            </table>
        </div>
        '''
        
        # Data Summary Section
        content += f'''
        <div class="section">
            <h2>📈 Data Summary</h2>
            <div class="metric">
                <div class="metric-value">{report.data_summary['total_records']:,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.data_summary['total_features']:,}</div>
                <div class="metric-label">Total Features</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.data_summary['data_quality']['completeness']:.1f}%</div>
                <div class="metric-label">Data Completeness</div>
            </div>
        </div>
        '''
        
        # Visualizations Section
        if report.visualizations_paths:
            content += '''
            <div class="section">
                <h2>📊 Visualizations</h2>
            '''
            for viz_path in report.visualizations_paths:
                if os.path.exists(viz_path):
                    content += f'''
                    <div class="chart-container">
                        <img src="file://{viz_path}" alt="Analysis Chart">
                    </div>
                    '''
            content += '</div>'
        
        # Recommendations Section
        if report.recommendations:
            content += '''
            <div class="section">
                <h2>💡 Recommendations</h2>
            '''
            for recommendation in report.recommendations:
                if "excellent" in recommendation.lower() or "good" in recommendation.lower():
                    content += f'<div class="recommendation">✅ {recommendation}</div>'
                else:
                    content += f'<div class="warning">⚠️ {recommendation}</div>'
            
            content += '</div>'
        
        return self.report_template.format(
            title="Comprehensive Stock Prediction Analysis Report",
            content=content,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _generate_json_report(self, report: ComprehensiveReport) -> str:
        """Generate JSON report for programmatic access."""
        report_dict = {
            'report_id': report.report_id,
            'timestamp': report.timestamp,
            'ensemble_metrics': report.ensemble_metrics,
            'model_reports': [
                {
                    'model_name': mr.model_name,
                    'metrics': mr.metrics,
                    'feature_count': mr.feature_count,
                    'data_points': mr.data_points
                }
                for mr in report.model_reports
            ],
            'data_summary': report.data_summary,
            'visualizations_paths': report.visualizations_paths,
            'recommendations': report.recommendations
        }
        
        return json.dumps(report_dict, indent=2, default=str)
    
    def _generate_prediction_html_report(self, report_id: str, model_name: str, 
                                       metrics: Dict[str, float],
                                       viz_path: str) -> str:
        """Generate HTML report for prediction analysis."""
        content = f'''
        <div class="header">
            <h1>Prediction Performance Analysis</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Report ID:</strong> {report_id}</p>
        </div>
        
        <div class="section">
            <h2>📊 Performance Metrics</h2>
        '''
        
        for metric, value in metrics.items():
            content += f'''
            <div class="metric">
                <div class="metric-value">{value:.6f}</div>
                <div class="metric-label">{metric}</div>
            </div>
            '''
        
        content += '</div>'
        
        if viz_path and os.path.exists(viz_path):
            content += f'''
            <div class="section">
                <h2>📈 Analysis Charts</h2>
                <div class="chart-container">
                    <img src="file://{viz_path}" alt="Prediction Analysis">
                </div>
            </div>
            '''
        
        return self.report_template.format(
            title=f"Prediction Analysis - {model_name}",
            content=content,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _generate_feature_html_report(self, report_id: str, analysis: Dict[str, Any], 
                                    viz_paths: List[str]) -> str:
        """Generate HTML report for feature analysis."""
        content = f'''
        <div class="header">
            <h1>Feature Analysis Report</h1>
            <p><strong>Report ID:</strong> {report_id}</p>
        </div>
        
        <div class="section">
            <h2>📊 Feature Categories</h2>
        '''
        
        for category, count in analysis['feature_categories'].items():
            content += f'''
            <div class="metric">
                <div class="metric-value">{count}</div>
                <div class="metric-label">{category.replace('_', ' ').title()}</div>
            </div>
            '''
        
        content += '</div>'
        
        # Visualizations
        for viz_path in viz_paths:
            if viz_path and os.path.exists(viz_path):
                content += f'''
                <div class="section">
                    <div class="chart-container">
                        <img src="file://{viz_path}" alt="Feature Analysis Chart">
                    </div>
                </div>
                '''
        
        return self.report_template.format(
            title="Feature Analysis Report",
            content=content,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _create_executive_summary_content(self, report: ComprehensiveReport) -> Dict[str, Any]:
        """Create executive summary content."""
        best_model = min(report.model_reports, key=lambda x: x.metrics['RMSE'])
        
        return {
            'key_findings': [
                f"Best performing model: {best_model.model_name} (RMSE: {best_model.metrics['RMSE']:.6f})",
                f"Ensemble R² Score: {report.ensemble_metrics['R2']:.4f}",
                f"Total features analyzed: {report.data_summary['total_features']:,}",
                f"Data quality score: {report.data_summary['data_quality']['completeness']:.1f}%"
            ],
            'business_impact': self._assess_business_impact(report),
            'risk_assessment': self._assess_model_risks(report),
            'next_steps': self._recommend_next_steps(report)
        }
    
    def _assess_business_impact(self, report: ComprehensiveReport) -> List[str]:
        """Assess business impact of the models."""
        impact = []
        
        r2_score = report.ensemble_metrics['R2']
        if r2_score > 0.8:
            impact.append("High confidence model suitable for automated trading decisions")
        elif r2_score > 0.6:
            impact.append("Moderate confidence model suitable for assisted decision making")
        else:
            impact.append("Low confidence model requires human oversight for all decisions")
        
        mae = report.ensemble_metrics.get('MAE', 0)
        if mae > 0:
            impact.append(f"Average prediction error: ₹{mae:.2f} per share")
        
        return impact
    
    def _assess_model_risks(self, report: ComprehensiveReport) -> List[str]:
        """Assess model risks."""
        risks = []
        
        if report.data_summary['data_quality']['completeness'] < 95:
            risks.append("Data quality issues may impact model reliability")
        
        if len(report.model_reports) < 3:
            risks.append("Limited ensemble diversity may reduce robustness")
        
        return risks
    
    def _recommend_next_steps(self, report: ComprehensiveReport) -> List[str]:
        """Recommend next steps."""
        steps = []
        
        if report.ensemble_metrics['R2'] < 0.7:
            steps.append("Investigate additional features to improve model performance")
        
        if report.data_summary['data_quality']['missing_values'] > 0:
            steps.append("Implement data cleaning pipeline to handle missing values")
        
        steps.append("Set up automated model monitoring and retraining pipeline")
        steps.append("Implement risk management controls for production deployment")
        
        return steps
    
    def _generate_executive_summary_html(self, report_id: str, summary: Dict[str, Any]) -> str:
        """Generate executive summary HTML."""
        content = f'''
        <div class="header">
            <h1>Executive Summary</h1>
            <p><strong>Report ID:</strong> {report_id}</p>
        </div>
        
        <div class="section">
            <h2>🎯 Key Findings</h2>
        '''
        
        for finding in summary['key_findings']:
            content += f'<div class="recommendation">📈 {finding}</div>'
        
        content += '''
        </div>
        
        <div class="section">
            <h2>💼 Business Impact</h2>
        '''
        
        for impact in summary['business_impact']:
            content += f'<div class="metric"><div class="metric-label">💰 {impact}</div></div>'
        
        content += '''
        </div>
        
        <div class="section">
            <h2>⚠️ Risk Assessment</h2>
        '''
        
        for risk in summary['risk_assessment']:
            content += f'<div class="warning">⚠️ {risk}</div>'
        
        content += '''
        </div>
        
        <div class="section">
            <h2>🚀 Next Steps</h2>
        '''
        
        for step in summary['next_steps']:
            content += f'<div class="recommendation">✅ {step}</div>'
        
        content += '</div>'
        
        return self.report_template.format(
            title=f"Executive Summary - {report_id}",
            content=content,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )


logger.info("Automated report generator utilities loaded successfully!")
