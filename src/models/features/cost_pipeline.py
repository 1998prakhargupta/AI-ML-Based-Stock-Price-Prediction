"""
Cost-Aware Feature Pipeline Module
=================================

Integrates cost features with existing feature engineering pipeline.
Provides cost feature calculation, normalization, selection, and validation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass

# Import existing processors
try:
    from src.data.processors import TechnicalIndicatorProcessor, ProcessingResult, DataQuality
except ImportError:
    # Fallback for testing
    class TechnicalIndicatorProcessor:
        def process_all_indicators(self, df): return type('MockResult', (), {'data': df, 'success': True})()
    
    class ProcessingResult:
        def __init__(self, data, success=True, quality=None, errors=None, warnings=None, metadata=None, processing_time=0):
            self.data = data
            self.success = success
            self.quality = quality or 'good'
            self.errors = errors or []
            self.warnings = warnings or []
            self.metadata = metadata or {}
            self.processing_time = processing_time
    
    class DataQuality:
        EXCELLENT = "excellent"
        GOOD = "good"
        FAIR = "fair"
        POOR = "poor"

from .cost_features import CostFeatureGenerator, CostFeatureConfig

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CostPipelineConfig:
    """Configuration for cost-aware feature pipeline."""
    enable_cost_features: bool = True
    enable_technical_indicators: bool = True
    cost_feature_config: Optional[CostFeatureConfig] = None
    feature_selection: bool = True
    max_features: Optional[int] = None
    normalize_features: bool = True
    validate_features: bool = True
    
    def __post_init__(self):
        if self.cost_feature_config is None:
            self.cost_feature_config = CostFeatureConfig()

class CostFeaturePipeline:
    """
    Comprehensive pipeline for integrating cost features with existing feature engineering.
    
    This class combines cost features with technical indicators and provides
    feature selection, normalization, and validation capabilities.
    """
    
    def __init__(self, config: Optional[CostPipelineConfig] = None):
        """
        Initialize cost-aware feature pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or CostPipelineConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize processors
        self.cost_generator = CostFeatureGenerator(self.config.cost_feature_config) if self.config.enable_cost_features else None
        self.tech_processor = TechnicalIndicatorProcessor() if self.config.enable_technical_indicators else None
        
        # Feature tracking
        self.generated_features = []
        self.selected_features = []
        self.feature_importance = {}
        
        self.logger.info("CostFeaturePipeline initialized")
    
    def process_features(self, df: pd.DataFrame) -> ProcessingResult:
        """
        Process all features including cost and technical indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            ProcessingResult: Comprehensive processing result
        """
        start_time = datetime.now()
        self.logger.info("Starting cost-aware feature processing")
        
        try:
            result_df = df.copy()
            processing_metadata = {
                'original_shape': df.shape,
                'cost_features_enabled': self.config.enable_cost_features,
                'technical_indicators_enabled': self.config.enable_technical_indicators
            }
            
            errors = []
            warnings = []
            
            # Process technical indicators first
            if self.config.enable_technical_indicators and self.tech_processor:
                self.logger.info("Processing technical indicators")
                tech_result = self.tech_processor.process_all_indicators(result_df)
                
                if tech_result.success:
                    result_df = tech_result.data
                    processing_metadata['technical_indicators'] = tech_result.metadata
                    if tech_result.warnings:
                        warnings.extend(tech_result.warnings)
                else:
                    errors.append(f"Technical indicators processing failed: {tech_result.errors}")
                    self.logger.warning("Technical indicators processing failed")
            
            # Process cost features
            if self.config.enable_cost_features and self.cost_generator:
                self.logger.info("Processing cost features")
                try:
                    result_df = self.cost_generator.generate_all_cost_features(result_df)
                    self.generated_features.extend(self.cost_generator.get_feature_names())
                    
                    cost_feature_count = len(self.cost_generator.get_feature_names())
                    processing_metadata['cost_features_generated'] = cost_feature_count
                    
                    self.logger.info(f"Generated {cost_feature_count} cost features")
                    
                except Exception as e:
                    error_msg = f"Cost feature generation failed: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Feature validation
            if self.config.validate_features:
                validation_result = self._validate_features(result_df)
                processing_metadata['validation'] = validation_result
                if validation_result.get('issues'):
                    warnings.extend(validation_result['issues'])
            
            # Feature selection
            if self.config.feature_selection:
                selection_result = self._select_features(result_df)
                result_df = selection_result['data']
                self.selected_features = selection_result['selected_features']
                processing_metadata['feature_selection'] = selection_result['metadata']
            
            # Feature normalization
            if self.config.normalize_features:
                normalization_result = self._normalize_features(result_df)
                result_df = normalization_result['data']
                processing_metadata['normalization'] = normalization_result['metadata']
            
            # Assess final quality
            quality = self._assess_pipeline_quality(result_df, errors, warnings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_metadata['final_shape'] = result_df.shape
            processing_metadata['features_added'] = result_df.shape[1] - df.shape[1]
            processing_metadata['processing_time'] = processing_time
            
            return ProcessingResult(
                data=result_df,
                success=len(errors) == 0,
                quality=quality,
                errors=errors,
                warnings=warnings,
                metadata=processing_metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Critical error in feature processing: {e}")
            
            return ProcessingResult(
                data=pd.DataFrame(),
                success=False,
                quality=DataQuality.POOR,
                errors=[str(e)],
                warnings=[],
                metadata={'processing_time': processing_time},
                processing_time=processing_time
            )
    
    def _validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate generated features for quality and consistency.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'total_features': df.shape[1],
            'total_rows': df.shape[0],
            'issues': [],
            'stats': {}
        }
        
        try:
            # Check for excessive NaN values
            nan_stats = {}
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    nan_pct = df[col].isna().mean() * 100
                    nan_stats[col] = nan_pct
                    
                    if nan_pct > 50:
                        validation_result['issues'].append(f"Feature {col} has {nan_pct:.1f}% NaN values")
            
            validation_result['stats']['nan_percentages'] = nan_stats
            
            # Check for constant features
            constant_features = []
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    if df[col].nunique() <= 1:
                        constant_features.append(col)
            
            if constant_features:
                validation_result['issues'].append(f"Found {len(constant_features)} constant features: {constant_features[:5]}")
            validation_result['stats']['constant_features'] = constant_features
            
            # Check for infinite values
            inf_features = []
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    if np.isinf(df[col]).any():
                        inf_features.append(col)
            
            if inf_features:
                validation_result['issues'].append(f"Found infinite values in features: {inf_features[:5]}")
            validation_result['stats']['infinite_features'] = inf_features
            
            # Check feature correlation (high correlation might indicate redundancy)
            if len(df.columns) > 1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    high_corr_pairs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if corr_val > 0.95 and not np.isnan(corr_val):
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                    
                    if high_corr_pairs:
                        validation_result['issues'].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
                    validation_result['stats']['high_correlation_pairs'] = high_corr_pairs[:10]  # Limit to first 10
            
            self.logger.debug(f"Feature validation completed, found {len(validation_result['issues'])} issues")
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {e}")
            self.logger.error(f"Error in feature validation: {e}")
        
        return validation_result
    
    def _select_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Select most relevant features based on various criteria.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dict[str, Any]: Selection results
        """
        selection_result = {
            'selected_features': [],
            'removed_features': [],
            'selection_criteria': {},
            'metadata': {}
        }
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            original_features = numeric_cols.copy()
            
            # Remove constant features
            constant_features = []
            for col in numeric_cols:
                if df[col].nunique() <= 1:
                    constant_features.append(col)
            
            remaining_features = [col for col in numeric_cols if col not in constant_features]
            selection_result['removed_features'].extend(constant_features)
            selection_result['selection_criteria']['constant_removal'] = len(constant_features)
            
            # Remove features with excessive NaN values
            high_nan_features = []
            for col in remaining_features:
                nan_pct = df[col].isna().mean()
                if nan_pct > 0.8:  # Remove if more than 80% NaN
                    high_nan_features.append(col)
            
            remaining_features = [col for col in remaining_features if col not in high_nan_features]
            selection_result['removed_features'].extend(high_nan_features)
            selection_result['selection_criteria']['high_nan_removal'] = len(high_nan_features)
            
            # Remove highly correlated features
            removed_corr = self._remove_correlated_features(df[remaining_features])
            remaining_features = [col for col in remaining_features if col not in removed_corr]
            selection_result['removed_features'].extend(removed_corr)
            selection_result['selection_criteria']['correlation_removal'] = len(removed_corr)
            
            # Apply max features limit if specified
            if self.config.max_features and len(remaining_features) > self.config.max_features:
                # Prioritize cost features and then by variance
                cost_features = [col for col in remaining_features if col in self.generated_features]
                other_features = [col for col in remaining_features if col not in self.generated_features]
                
                # Calculate variance for ranking
                variance_ranking = {}
                for col in other_features:
                    variance_ranking[col] = df[col].var()
                
                # Sort other features by variance (descending)
                sorted_other = sorted(other_features, key=lambda x: variance_ranking.get(x, 0), reverse=True)
                
                # Select features: prioritize cost features, then high variance features
                max_cost_features = min(len(cost_features), self.config.max_features // 2)
                max_other_features = self.config.max_features - max_cost_features
                
                selected_features = cost_features[:max_cost_features] + sorted_other[:max_other_features]
                
                removed_by_limit = [col for col in remaining_features if col not in selected_features]
                selection_result['removed_features'].extend(removed_by_limit)
                selection_result['selection_criteria']['max_features_limit'] = len(removed_by_limit)
                
                remaining_features = selected_features
            
            selection_result['selected_features'] = remaining_features
            selection_result['metadata'] = {
                'original_feature_count': len(original_features),
                'selected_feature_count': len(remaining_features),
                'removed_feature_count': len(selection_result['removed_features']),
                'cost_features_selected': len([f for f in remaining_features if f in self.generated_features])
            }
            
            # Create filtered DataFrame
            selected_df = df[remaining_features + [col for col in df.columns if col not in numeric_cols]].copy()
            selection_result['data'] = selected_df
            
            self.logger.info(f"Feature selection: {len(original_features)} -> {len(remaining_features)} features")
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            selection_result['data'] = df
            selection_result['selected_features'] = df.columns.tolist()
        
        return selection_result
    
    def _remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold
            
        Returns:
            List[str]: Features to remove
        """
        to_remove = []
        
        try:
            if len(df.columns) <= 1:
                return to_remove
            
            corr_matrix = df.corr().abs()
            
            # Find pairs of highly correlated features
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        # Remove the feature with lower variance
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        var1, var2 = df[col1].var(), df[col2].var()
                        
                        feature_to_remove = col1 if var1 < var2 else col2
                        
                        # Prefer to keep cost features if one is a cost feature
                        if col1 in self.generated_features and col2 not in self.generated_features:
                            feature_to_remove = col2
                        elif col2 in self.generated_features and col1 not in self.generated_features:
                            feature_to_remove = col1
                        
                        if feature_to_remove not in to_remove:
                            to_remove.append(feature_to_remove)
            
        except Exception as e:
            self.logger.error(f"Error removing correlated features: {e}")
        
        return to_remove
    
    def _normalize_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Normalize features for better ML model performance.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dict[str, Any]: Normalization results
        """
        normalization_result = {
            'data': df.copy(),
            'metadata': {
                'normalization_method': 'robust_scaling',
                'normalized_features': []
            }
        }
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].nunique() > 1:  # Skip constant features
                    # Use robust scaling (median and IQR) to handle outliers
                    median_val = df[col].median()
                    q75 = df[col].quantile(0.75)
                    q25 = df[col].quantile(0.25)
                    iqr = q75 - q25
                    
                    if iqr > 0:
                        normalization_result['data'][col] = (df[col] - median_val) / iqr
                        normalization_result['metadata']['normalized_features'].append(col)
                    else:
                        # Fallback to standard scaling
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val > 0:
                            normalization_result['data'][col] = (df[col] - mean_val) / std_val
                            normalization_result['metadata']['normalized_features'].append(col)
            
            # Clip extreme values to prevent outlier issues
            for col in normalization_result['metadata']['normalized_features']:
                normalization_result['data'][col] = np.clip(
                    normalization_result['data'][col], -5, 5
                )
            
            self.logger.debug(f"Normalized {len(normalization_result['metadata']['normalized_features'])} features")
            
        except Exception as e:
            self.logger.error(f"Error in feature normalization: {e}")
            normalization_result['data'] = df
        
        return normalization_result
    
    def _assess_pipeline_quality(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> str:
        """
        Assess overall quality of the feature pipeline output.
        
        Args:
            df: Final DataFrame
            errors: List of errors
            warnings: List of warnings
            
        Returns:
            str: Quality assessment
        """
        try:
            if errors:
                return DataQuality.POOR
            
            if df.empty:
                return DataQuality.POOR
            
            # Calculate quality metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return DataQuality.POOR
            
            # Check NaN percentage
            total_cells = len(numeric_cols) * len(df)
            nan_count = df[numeric_cols].isna().sum().sum()
            nan_percentage = (nan_count / total_cells) * 100 if total_cells > 0 else 100
            
            # Check feature diversity
            feature_diversity = len(numeric_cols) / max(len(df.columns), 1)
            
            # Quality assessment
            if nan_percentage < 5 and feature_diversity > 0.5 and len(warnings) < 3:
                return DataQuality.EXCELLENT
            elif nan_percentage < 15 and feature_diversity > 0.3 and len(warnings) < 5:
                return DataQuality.GOOD
            elif nan_percentage < 30 and feature_diversity > 0.1:
                return DataQuality.FAIR
            else:
                return DataQuality.POOR
                
        except Exception as e:
            self.logger.error(f"Error assessing pipeline quality: {e}")
            return DataQuality.POOR
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive statistics about generated features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dict[str, Any]: Feature statistics
        """
        stats = {
            'total_features': len(df.columns),
            'cost_features': len([col for col in df.columns if col in self.generated_features]),
            'technical_features': len(df.columns) - len([col for col in df.columns if col in self.generated_features]),
            'feature_types': {},
            'quality_metrics': {}
        }
        
        try:
            # Feature type breakdown
            for feature in self.generated_features:
                if feature in df.columns:
                    if 'cost_avg' in feature or 'cost_ewm' in feature:
                        stats['feature_types']['historical_average'] = stats['feature_types'].get('historical_average', 0) + 1
                    elif 'vol' in feature:
                        stats['feature_types']['volatility'] = stats['feature_types'].get('volatility', 0) + 1
                    elif 'ratio' in feature or 'drag' in feature:
                        stats['feature_types']['cost_return'] = stats['feature_types'].get('cost_return', 0) + 1
                    elif 'efficiency' in feature or 'consistency' in feature:
                        stats['feature_types']['broker_efficiency'] = stats['feature_types'].get('broker_efficiency', 0) + 1
                    elif 'impact' in feature or 'volume' in feature:
                        stats['feature_types']['market_impact'] = stats['feature_types'].get('market_impact', 0) + 1
                    elif 'liquidity' in feature:
                        stats['feature_types']['liquidity'] = stats['feature_types'].get('liquidity', 0) + 1
            
            # Quality metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats['quality_metrics'] = {
                    'nan_percentage': df[numeric_cols].isna().mean().mean() * 100,
                    'infinite_count': np.isinf(df[numeric_cols]).sum().sum(),
                    'constant_features': sum(df[col].nunique() <= 1 for col in numeric_cols),
                    'high_variance_features': sum(df[col].var() > 1.0 for col in numeric_cols if not df[col].isna().all())
                }
            
            # Feature importance if available
            if self.cost_generator:
                importance_estimates = self.cost_generator.get_feature_importance_estimates(df)
                if importance_estimates:
                    stats['feature_importance'] = dict(sorted(importance_estimates.items(), key=lambda x: x[1], reverse=True)[:10])
            
        except Exception as e:
            self.logger.error(f"Error calculating feature statistics: {e}")
        
        return stats
    
    def get_cost_feature_names(self) -> List[str]:
        """Get names of generated cost features."""
        return self.generated_features.copy()
    
    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features after feature selection."""
        return self.selected_features.copy()