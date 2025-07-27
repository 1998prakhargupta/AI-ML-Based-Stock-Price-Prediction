"""
Cost Integration Mixin Module
============================

Provides mixin functionality for integrating cost awareness into existing classes.
Allows existing components to be enhanced with cost capabilities without modification.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union

# Setup logging
logger = logging.getLogger(__name__)

class CostIntegrationMixin:
    """
    Mixin class providing cost integration capabilities.
    
    Can be mixed into existing classes to add cost-awareness without
    modifying the original class structure.
    """
    
    def __init__(self, cost_config: Optional[Any] = None):
        """
        Initialize cost integration mixin.
        
        Args:
            cost_config: Cost configuration object
        """
        self.cost_config = cost_config
        self.cost_mixin_logger = logger.getChild(f"{self.__class__.__name__}_CostMixin")
        self.cost_features_cache = {}
        self.cost_statistics = {}
    
    def _identify_cost_features(self, feature_names: Union[List[str], pd.Index]) -> List[str]:
        """
        Identify cost-related features from a list of feature names.
        
        Args:
            feature_names: List or Index of feature names
            
        Returns:
            List[str]: Cost feature names
        """
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
        
        # Cache check
        cache_key = str(sorted(feature_names))
        if cache_key in self.cost_features_cache:
            return self.cost_features_cache[cache_key]
        
        cost_keywords = [
            'cost', 'commission', 'spread', 'impact', 'efficiency', 
            'liquidity', 'drag', 'ratio', 'broker', 'transaction',
            'fee', 'slippage', 'bid_ask', 'market_impact'
        ]
        
        cost_features = []
        for feature in feature_names:
            feature_lower = str(feature).lower()
            if any(keyword in feature_lower for keyword in cost_keywords):
                cost_features.append(feature)
        
        # Cache result
        self.cost_features_cache[cache_key] = cost_features
        
        self.cost_mixin_logger.debug(f"Identified {len(cost_features)} cost features from {len(feature_names)} total features")
        return cost_features
    
    def _calculate_cost_statistics(self, df: pd.DataFrame, cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate statistics for cost features.
        
        Args:
            df: DataFrame with features
            cost_features: List of cost feature names (optional)
            
        Returns:
            Dict[str, Any]: Cost statistics
        """
        if cost_features is None:
            cost_features = self._identify_cost_features(df.columns)
        
        statistics = {
            'cost_feature_count': len(cost_features),
            'cost_features': cost_features,
            'feature_stats': {},
            'summary': {}
        }
        
        try:
            if not cost_features:
                return statistics
            
            cost_df = df[cost_features]
            
            # Individual feature statistics
            for feature in cost_features:
                if feature in cost_df.columns:
                    feature_stats = {
                        'mean': cost_df[feature].mean(),
                        'std': cost_df[feature].std(),
                        'min': cost_df[feature].min(),
                        'max': cost_df[feature].max(),
                        'median': cost_df[feature].median(),
                        'null_count': cost_df[feature].isnull().sum(),
                        'null_percentage': cost_df[feature].isnull().mean() * 100
                    }
                    statistics['feature_stats'][feature] = feature_stats
            
            # Summary statistics
            if len(cost_features) > 0:
                all_cost_values = cost_df.values.flatten()
                all_cost_values = all_cost_values[~pd.isnull(all_cost_values)]
                
                if len(all_cost_values) > 0:
                    statistics['summary'] = {
                        'overall_mean': np.mean(all_cost_values),
                        'overall_std': np.std(all_cost_values),
                        'overall_median': np.median(all_cost_values),
                        'total_observations': len(all_cost_values),
                        'cost_range': np.max(all_cost_values) - np.min(all_cost_values),
                        'coefficient_of_variation': np.std(all_cost_values) / (np.mean(all_cost_values) + 1e-8)
                    }
            
            # Cache statistics
            self.cost_statistics.update(statistics)
            
        except Exception as e:
            self.cost_mixin_logger.error(f"Error calculating cost statistics: {e}")
            statistics['error'] = str(e)
        
        return statistics
    
    def _assess_cost_feature_quality(self, df: pd.DataFrame, cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Assess the quality of cost features.
        
        Args:
            df: DataFrame with features
            cost_features: List of cost feature names (optional)
            
        Returns:
            Dict[str, Any]: Quality assessment
        """
        if cost_features is None:
            cost_features = self._identify_cost_features(df.columns)
        
        quality_assessment = {
            'overall_quality': 'unknown',
            'quality_score': 0.0,
            'issues': [],
            'recommendations': [],
            'feature_quality': {}
        }
        
        try:
            if not cost_features:
                quality_assessment['overall_quality'] = 'no_cost_features'
                quality_assessment['issues'].append("No cost features identified")
                return quality_assessment
            
            quality_scores = []
            
            for feature in cost_features:
                if feature not in df.columns:
                    continue
                
                feature_quality = {
                    'completeness': 1.0 - df[feature].isnull().mean(),
                    'variance': 1.0 if df[feature].var() > 0 else 0.0,
                    'range_reasonableness': 1.0,  # Default to reasonable
                    'outlier_ratio': 0.0
                }
                
                # Check for outliers (values beyond 3 standard deviations)
                if df[feature].std() > 0:
                    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                    feature_quality['outlier_ratio'] = (z_scores > 3).mean()
                
                # Check range reasonableness (cost features should be non-negative)
                if (df[feature] < 0).any():
                    feature_quality['range_reasonableness'] = 0.5
                    quality_assessment['issues'].append(f"Feature {feature} has negative values")
                
                # Calculate composite score
                composite_score = (
                    feature_quality['completeness'] * 0.4 +
                    feature_quality['variance'] * 0.3 +
                    feature_quality['range_reasonableness'] * 0.2 +
                    (1.0 - min(feature_quality['outlier_ratio'], 0.5)) * 0.1
                )
                
                feature_quality['composite_score'] = composite_score
                quality_assessment['feature_quality'][feature] = feature_quality
                quality_scores.append(composite_score)
            
            # Overall assessment
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                quality_assessment['quality_score'] = avg_quality
                
                if avg_quality >= 0.8:
                    quality_assessment['overall_quality'] = 'excellent'
                elif avg_quality >= 0.6:
                    quality_assessment['overall_quality'] = 'good'
                elif avg_quality >= 0.4:
                    quality_assessment['overall_quality'] = 'fair'
                else:
                    quality_assessment['overall_quality'] = 'poor'
            
            # Generate recommendations
            if quality_assessment['quality_score'] < 0.6:
                quality_assessment['recommendations'].append("Consider feature engineering to improve cost feature quality")
            
            low_completeness_features = [
                f for f, q in quality_assessment['feature_quality'].items() 
                if q['completeness'] < 0.8
            ]
            if low_completeness_features:
                quality_assessment['recommendations'].append(
                    f"Address missing values in: {', '.join(low_completeness_features)}"
                )
            
            high_outlier_features = [
                f for f, q in quality_assessment['feature_quality'].items() 
                if q['outlier_ratio'] > 0.1
            ]
            if high_outlier_features:
                quality_assessment['recommendations'].append(
                    f"Consider outlier treatment for: {', '.join(high_outlier_features)}"
                )
                
        except Exception as e:
            self.cost_mixin_logger.error(f"Error assessing cost feature quality: {e}")
            quality_assessment['error'] = str(e)
        
        return quality_assessment
    
    def _generate_cost_feature_report(self, df: pd.DataFrame, cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report on cost features.
        
        Args:
            df: DataFrame with features
            cost_features: List of cost feature names (optional)
            
        Returns:
            Dict[str, Any]: Comprehensive cost feature report
        """
        try:
            if cost_features is None:
                cost_features = self._identify_cost_features(df.columns)
            
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'dataset_info': {
                    'total_features': len(df.columns),
                    'total_rows': len(df),
                    'cost_features_identified': len(cost_features)
                },
                'cost_statistics': self._calculate_cost_statistics(df, cost_features),
                'quality_assessment': self._assess_cost_feature_quality(df, cost_features),
                'feature_correlations': {},
                'recommendations': []
            }
            
            # Calculate correlations between cost features
            if len(cost_features) > 1:
                cost_df = df[cost_features]
                corr_matrix = cost_df.corr()
                
                # Find highly correlated pairs
                high_corr_pairs = []
                for i in range(len(cost_features)):
                    for j in range(i+1, len(cost_features)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'feature1': cost_features[i],
                                'feature2': cost_features[j],
                                'correlation': corr_val
                            })
                
                report['feature_correlations'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'high_correlation_pairs': high_corr_pairs,
                    'max_correlation': corr_matrix.abs().max().max(),
                    'mean_correlation': corr_matrix.abs().mean().mean()
                }
            
            # Generate overall recommendations
            if report['dataset_info']['cost_features_identified'] == 0:
                report['recommendations'].append("No cost features found - consider adding transaction cost data")
            elif report['dataset_info']['cost_features_identified'] < 5:
                report['recommendations'].append("Limited cost features - consider engineering additional cost metrics")
            
            if report['quality_assessment']['quality_score'] < 0.7:
                report['recommendations'].append("Cost feature quality needs improvement")
            
            if len(report['feature_correlations'].get('high_correlation_pairs', [])) > 0:
                report['recommendations'].append("Consider removing highly correlated cost features")
            
            # Add quality recommendations
            report['recommendations'].extend(report['quality_assessment'].get('recommendations', []))
            
            return report
            
        except Exception as e:
            self.cost_mixin_logger.error(f"Error generating cost feature report: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    def _apply_cost_aware_preprocessing(self, 
                                      df: pd.DataFrame, 
                                      cost_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply cost-aware preprocessing to the DataFrame.
        
        Args:
            df: Input DataFrame
            cost_features: List of cost feature names (optional)
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        try:
            if cost_features is None:
                cost_features = self._identify_cost_features(df.columns)
            
            if not cost_features:
                return df
            
            processed_df = df.copy()
            
            # Cost-specific preprocessing
            for feature in cost_features:
                if feature in processed_df.columns:
                    # Ensure non-negative costs
                    processed_df[feature] = np.maximum(processed_df[feature], 0)
                    
                    # Handle extreme outliers (clip at 99th percentile)
                    upper_bound = processed_df[feature].quantile(0.99)
                    processed_df[feature] = np.minimum(processed_df[feature], upper_bound)
                    
                    # Fill missing values with median
                    if processed_df[feature].isnull().any():
                        median_val = processed_df[feature].median()
                        processed_df[feature].fillna(median_val, inplace=True)
            
            self.cost_mixin_logger.debug(f"Applied cost-aware preprocessing to {len(cost_features)} features")
            return processed_df
            
        except Exception as e:
            self.cost_mixin_logger.error(f"Error in cost-aware preprocessing: {e}")
            return df
    
    def _create_cost_feature_summary(self, cost_features: List[str]) -> Dict[str, Any]:
        """
        Create a summary of cost features for reporting.
        
        Args:
            cost_features: List of cost feature names
            
        Returns:
            Dict[str, Any]: Cost feature summary
        """
        summary = {
            'total_count': len(cost_features),
            'feature_types': {
                'average_cost': 0,
                'volatility': 0,
                'ratio': 0,
                'efficiency': 0,
                'impact': 0,
                'liquidity': 0,
                'other': 0
            },
            'features_by_timeframe': {
                'short_term': [],  # 5d, 10d
                'medium_term': [], # 20d, 30d
                'long_term': [],   # 50d, 100d+
                'other': []
            }
        }
        
        try:
            for feature in cost_features:
                feature_lower = feature.lower()
                
                # Categorize by type
                if 'avg' in feature_lower or 'average' in feature_lower:
                    summary['feature_types']['average_cost'] += 1
                elif 'vol' in feature_lower or 'volatility' in feature_lower:
                    summary['feature_types']['volatility'] += 1
                elif 'ratio' in feature_lower or 'drag' in feature_lower:
                    summary['feature_types']['ratio'] += 1
                elif 'efficiency' in feature_lower or 'consistency' in feature_lower:
                    summary['feature_types']['efficiency'] += 1
                elif 'impact' in feature_lower or 'momentum' in feature_lower:
                    summary['feature_types']['impact'] += 1
                elif 'liquidity' in feature_lower or 'range' in feature_lower:
                    summary['feature_types']['liquidity'] += 1
                else:
                    summary['feature_types']['other'] += 1
                
                # Categorize by timeframe
                if any(timeframe in feature_lower for timeframe in ['5d', '10d']):
                    summary['features_by_timeframe']['short_term'].append(feature)
                elif any(timeframe in feature_lower for timeframe in ['20d', '30d']):
                    summary['features_by_timeframe']['medium_term'].append(feature)
                elif any(timeframe in feature_lower for timeframe in ['50d', '100d', '200d']):
                    summary['features_by_timeframe']['long_term'].append(feature)
                else:
                    summary['features_by_timeframe']['other'].append(feature)
            
        except Exception as e:
            self.cost_mixin_logger.error(f"Error creating cost feature summary: {e}")
        
        return summary
    
    def get_cost_integration_status(self) -> Dict[str, Any]:
        """
        Get status of cost integration in the current instance.
        
        Returns:
            Dict[str, Any]: Cost integration status
        """
        status = {
            'cost_mixin_active': True,
            'cost_config_present': self.cost_config is not None,
            'cached_cost_features': len(self.cost_features_cache),
            'cost_statistics_available': len(self.cost_statistics) > 0,
            'integration_capabilities': [
                'feature_identification',
                'quality_assessment',
                'preprocessing',
                'reporting'
            ]
        }
        
        if self.cost_config:
            status['cost_config_type'] = type(self.cost_config).__name__
            if hasattr(self.cost_config, '__dict__'):
                status['cost_config_settings'] = {
                    k: v for k, v in self.cost_config.__dict__.items() 
                    if not k.startswith('_')
                }
        
        return status