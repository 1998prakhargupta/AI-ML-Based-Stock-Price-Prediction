"""
Cost Feature Selector Module
============================

Provides advanced feature selection specifically for cost-related features.
Includes statistical tests, importance ranking, and domain-specific selection criteria.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class SelectionMethod(Enum):
    """Enum for feature selection methods."""
    VARIANCE = "variance"
    CORRELATION = "correlation" 
    MUTUAL_INFORMATION = "mutual_information"
    STATISTICAL_TESTS = "statistical_tests"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    COMBINED = "combined"

@dataclass
class FeatureSelectionConfig:
    """Configuration for cost feature selection."""
    max_features: Optional[int] = None
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    statistical_significance: float = 0.05
    use_domain_knowledge: bool = True
    selection_methods: List[SelectionMethod] = None
    
    def __post_init__(self):
        if self.selection_methods is None:
            self.selection_methods = [SelectionMethod.VARIANCE, SelectionMethod.CORRELATION, SelectionMethod.DOMAIN_KNOWLEDGE]

class CostFeatureSelector:
    """
    Advanced feature selector specifically designed for cost-related features.
    
    Provides multiple selection strategies optimized for trading cost features,
    including domain knowledge-based selection and statistical methods.
    """
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """
        Initialize cost feature selector.
        
        Args:
            config: Selection configuration
        """
        self.config = config or FeatureSelectionConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Feature tracking
        self.selected_features = []
        self.feature_scores = {}
        self.selection_metadata = {}
        
        # Domain knowledge feature priorities
        self.feature_priorities = {
            'high': [
                'cost_avg_20d', 'cost_return_ratio_20d', 'cost_vol_20d',
                'broker_efficiency_20d', 'liquidity_adj_cost_20d'
            ],
            'medium': [
                'cost_avg_10d', 'cost_return_ratio_10d', 'cost_vol_10d',
                'volume_impact_20d', 'cost_drag_20d'
            ],
            'low': [
                'cost_avg_5d', 'cost_ewm_alpha_0.1', 'cost_range_ratio'
            ]
        }
        
        self.logger.info("CostFeatureSelector initialized")
    
    def select_features(self, df: pd.DataFrame, target: Optional[pd.Series] = None, 
                       cost_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Select best cost features using multiple criteria.
        
        Args:
            df: DataFrame with features
            target: Target variable for supervised selection (optional)
            cost_features: List of cost feature names (optional)
            
        Returns:
            Dict[str, Any]: Selection results
        """
        self.logger.info("Starting cost feature selection")
        
        try:
            # Identify cost features if not provided
            if cost_features is None:
                cost_features = self._identify_cost_features(df)
            
            selection_result = {
                'selected_features': [],
                'removed_features': [],
                'feature_scores': {},
                'selection_steps': {},
                'metadata': {
                    'total_features': len(cost_features),
                    'selection_methods': [method.value for method in self.config.selection_methods]
                }
            }
            
            # Start with all cost features
            candidate_features = cost_features.copy()
            
            # Apply each selection method
            for method in self.config.selection_methods:
                if method == SelectionMethod.VARIANCE:
                    result = self._select_by_variance(df, candidate_features)
                elif method == SelectionMethod.CORRELATION:
                    result = self._select_by_correlation(df, candidate_features)
                elif method == SelectionMethod.MUTUAL_INFORMATION:
                    result = self._select_by_mutual_information(df, candidate_features, target)
                elif method == SelectionMethod.STATISTICAL_TESTS:
                    result = self._select_by_statistical_tests(df, candidate_features, target)
                elif method == SelectionMethod.DOMAIN_KNOWLEDGE:
                    result = self._select_by_domain_knowledge(df, candidate_features)
                elif method == SelectionMethod.COMBINED:
                    result = self._select_by_combined_score(df, candidate_features, target)
                else:
                    continue
                
                candidate_features = result['selected_features']
                selection_result['selection_steps'][method.value] = result
                
                self.logger.debug(f"After {method.value}: {len(candidate_features)} features remaining")
            
            # Apply max features limit
            if self.config.max_features and len(candidate_features) > self.config.max_features:
                # Use combined scoring to rank features
                combined_scores = self._calculate_combined_scores(df, candidate_features, target)
                sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                candidate_features = [f[0] for f in sorted_features[:self.config.max_features]]
                
                selection_result['selection_steps']['max_features_limit'] = {
                    'applied': True,
                    'limit': self.config.max_features,
                    'selected_features': candidate_features
                }
            
            # Final results
            selection_result['selected_features'] = candidate_features
            selection_result['removed_features'] = [f for f in cost_features if f not in candidate_features]
            selection_result['feature_scores'] = self.feature_scores
            selection_result['metadata']['selected_count'] = len(candidate_features)
            selection_result['metadata']['removed_count'] = len(selection_result['removed_features'])
            
            self.selected_features = candidate_features
            self.selection_metadata = selection_result['metadata']
            
            self.logger.info(f"Feature selection completed: {len(cost_features)} -> {len(candidate_features)} features")
            
            return selection_result
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return {
                'selected_features': cost_features if cost_features else [],
                'removed_features': [],
                'feature_scores': {},
                'selection_steps': {},
                'metadata': {'error': str(e)}
            }
    
    def _identify_cost_features(self, df: pd.DataFrame) -> List[str]:
        """
        Identify cost-related features in the DataFrame.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List[str]: Cost feature names
        """
        cost_keywords = [
            'cost', 'commission', 'spread', 'impact', 'efficiency', 
            'liquidity', 'drag', 'ratio', 'broker', 'transaction'
        ]
        
        cost_features = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in cost_keywords):
                cost_features.append(col)
        
        self.logger.debug(f"Identified {len(cost_features)} cost features")
        return cost_features
    
    def _select_by_variance(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Select features based on variance threshold.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            
        Returns:
            Dict[str, Any]: Selection results
        """
        selected_features = []
        variance_scores = {}
        
        try:
            for feature in features:
                if feature in df.columns:
                    variance = df[feature].var()
                    variance_scores[feature] = variance
                    
                    if variance > self.config.variance_threshold:
                        selected_features.append(feature)
            
            # Update global scores
            self.feature_scores.update(variance_scores)
            
        except Exception as e:
            self.logger.error(f"Error in variance-based selection: {e}")
            selected_features = features
        
        return {
            'selected_features': selected_features,
            'removed_features': [f for f in features if f not in selected_features],
            'scores': variance_scores,
            'threshold': self.config.variance_threshold
        }
    
    def _select_by_correlation(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Remove highly correlated features.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            
        Returns:
            Dict[str, Any]: Selection results
        """
        selected_features = features.copy()
        removed_pairs = []
        
        try:
            if len(features) <= 1:
                return {
                    'selected_features': selected_features,
                    'removed_features': [],
                    'removed_pairs': [],
                    'threshold': self.config.correlation_threshold
                }
            
            feature_df = df[features]
            corr_matrix = feature_df.corr().abs()
            
            # Find and remove highly correlated features
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    if features[i] in selected_features and features[j] in selected_features:
                        corr_val = corr_matrix.iloc[i, j]
                        
                        if corr_val > self.config.correlation_threshold:
                            # Remove feature with lower variance
                            var1 = feature_df[features[i]].var()
                            var2 = feature_df[features[j]].var()
                            
                            feature_to_remove = features[i] if var1 < var2 else features[j]
                            
                            # Apply domain knowledge priority
                            priority1 = self._get_feature_priority(features[i])
                            priority2 = self._get_feature_priority(features[j])
                            
                            if priority1 > priority2:
                                feature_to_remove = features[j]
                            elif priority2 > priority1:
                                feature_to_remove = features[i]
                            
                            if feature_to_remove in selected_features:
                                selected_features.remove(feature_to_remove)
                                removed_pairs.append((features[i], features[j], corr_val))
            
        except Exception as e:
            self.logger.error(f"Error in correlation-based selection: {e}")
        
        return {
            'selected_features': selected_features,
            'removed_features': [f for f in features if f not in selected_features],
            'removed_pairs': removed_pairs,
            'threshold': self.config.correlation_threshold
        }
    
    def _select_by_mutual_information(self, df: pd.DataFrame, features: List[str], 
                                    target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Select features based on mutual information with target.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            target: Target variable
            
        Returns:
            Dict[str, Any]: Selection results
        """
        if target is None:
            # Return all features if no target available
            return {
                'selected_features': features,
                'removed_features': [],
                'scores': {},
                'note': 'No target provided, skipping mutual information selection'
            }
        
        selected_features = []
        mi_scores = {}
        
        try:
            # Simple correlation-based approximation of mutual information
            for feature in features:
                if feature in df.columns:
                    # Remove NaN values for correlation calculation
                    clean_data = pd.concat([df[feature], target], axis=1).dropna()
                    
                    if len(clean_data) > 10:  # Minimum samples for reliable correlation
                        correlation = abs(clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1]))
                        mi_scores[feature] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        mi_scores[feature] = 0.0
            
            # Select top features based on mutual information
            if mi_scores:
                threshold = np.percentile(list(mi_scores.values()), 50)  # Top 50%
                selected_features = [f for f, score in mi_scores.items() if score >= threshold]
            
            # Update global scores
            self.feature_scores.update(mi_scores)
            
        except Exception as e:
            self.logger.error(f"Error in mutual information selection: {e}")
            selected_features = features
        
        return {
            'selected_features': selected_features,
            'removed_features': [f for f in features if f not in selected_features],
            'scores': mi_scores,
            'threshold': np.percentile(list(mi_scores.values()), 50) if mi_scores else 0
        }
    
    def _select_by_statistical_tests(self, df: pd.DataFrame, features: List[str], 
                                   target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Select features based on statistical significance tests.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            target: Target variable
            
        Returns:
            Dict[str, Any]: Selection results
        """
        if target is None:
            return {
                'selected_features': features,
                'removed_features': [],
                'p_values': {},
                'note': 'No target provided, skipping statistical tests'
            }
        
        selected_features = []
        p_values = {}
        
        try:
            from scipy import stats
            
            for feature in features:
                if feature in df.columns:
                    # Remove NaN values
                    clean_data = pd.concat([df[feature], target], axis=1).dropna()
                    
                    if len(clean_data) > 30:  # Minimum for meaningful test
                        # Pearson correlation test
                        _, p_value = stats.pearsonr(clean_data.iloc[:, 0], clean_data.iloc[:, 1])
                        p_values[feature] = p_value
                        
                        if p_value < self.config.statistical_significance:
                            selected_features.append(feature)
                    else:
                        p_values[feature] = 1.0  # Not significant due to insufficient data
            
        except ImportError:
            self.logger.warning("SciPy not available, skipping statistical tests")
            selected_features = features
        except Exception as e:
            self.logger.error(f"Error in statistical tests: {e}")
            selected_features = features
        
        return {
            'selected_features': selected_features,
            'removed_features': [f for f in features if f not in selected_features],
            'p_values': p_values,
            'significance_threshold': self.config.statistical_significance
        }
    
    def _select_by_domain_knowledge(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Select features based on domain knowledge about cost features.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            
        Returns:
            Dict[str, Any]: Selection results
        """
        if not self.config.use_domain_knowledge:
            return {
                'selected_features': features,
                'removed_features': [],
                'priorities': {},
                'note': 'Domain knowledge selection disabled'
            }
        
        selected_features = []
        feature_priorities = {}
        
        try:
            # Assign priorities to features
            for feature in features:
                priority = self._get_feature_priority(feature)
                feature_priorities[feature] = priority
                
                # Select high and medium priority features
                if priority >= 2:  # High: 3, Medium: 2, Low: 1
                    selected_features.append(feature)
            
            # If we have too few features, add some low priority ones
            if len(selected_features) < max(5, len(features) // 3):
                low_priority_features = [f for f in features if feature_priorities[f] == 1]
                # Sort by variance and add best low priority features
                if low_priority_features:
                    variances = {f: df[f].var() if f in df.columns else 0 for f in low_priority_features}
                    sorted_low = sorted(low_priority_features, key=lambda x: variances[x], reverse=True)
                    needed = max(5, len(features) // 3) - len(selected_features)
                    selected_features.extend(sorted_low[:needed])
            
        except Exception as e:
            self.logger.error(f"Error in domain knowledge selection: {e}")
            selected_features = features
        
        return {
            'selected_features': selected_features,
            'removed_features': [f for f in features if f not in selected_features],
            'priorities': feature_priorities,
            'priority_levels': self.feature_priorities
        }
    
    def _select_by_combined_score(self, df: pd.DataFrame, features: List[str], 
                                target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Select features using combined scoring method.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            target: Target variable
            
        Returns:
            Dict[str, Any]: Selection results
        """
        combined_scores = self._calculate_combined_scores(df, features, target)
        
        # Select top 70% of features by combined score
        threshold = np.percentile(list(combined_scores.values()), 30) if combined_scores else 0
        selected_features = [f for f, score in combined_scores.items() if score >= threshold]
        
        return {
            'selected_features': selected_features,
            'removed_features': [f for f in features if f not in selected_features],
            'combined_scores': combined_scores,
            'threshold': threshold
        }
    
    def _calculate_combined_scores(self, df: pd.DataFrame, features: List[str], 
                                 target: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate combined scores for features.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            target: Target variable
            
        Returns:
            Dict[str, float]: Combined scores
        """
        combined_scores = {}
        
        try:
            for feature in features:
                if feature not in df.columns:
                    combined_scores[feature] = 0.0
                    continue
                
                score = 0.0
                
                # Variance component (0-1 scale)
                variance = df[feature].var()
                variance_score = min(variance / (variance + 1.0), 1.0) if variance > 0 else 0.0
                score += variance_score * 0.3
                
                # Non-null percentage component
                non_null_pct = df[feature].notna().mean()
                score += non_null_pct * 0.2
                
                # Domain knowledge component
                priority = self._get_feature_priority(feature)
                priority_score = priority / 3.0  # Normalize to 0-1
                score += priority_score * 0.3
                
                # Target correlation component (if available)
                if target is not None:
                    try:
                        clean_data = pd.concat([df[feature], target], axis=1).dropna()
                        if len(clean_data) > 10:
                            correlation = abs(clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1]))
                            correlation_score = correlation if not np.isnan(correlation) else 0.0
                            score += correlation_score * 0.2
                    except:
                        pass
                
                combined_scores[feature] = score
                
        except Exception as e:
            self.logger.error(f"Error calculating combined scores: {e}")
        
        return combined_scores
    
    def _get_feature_priority(self, feature_name: str) -> int:
        """
        Get priority level for a feature based on domain knowledge.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            int: Priority level (3=high, 2=medium, 1=low)
        """
        feature_lower = feature_name.lower()
        
        # Check high priority patterns
        for high_priority in self.feature_priorities['high']:
            if high_priority.lower() in feature_lower:
                return 3
        
        # Check medium priority patterns
        for medium_priority in self.feature_priorities['medium']:
            if medium_priority.lower() in feature_lower:
                return 2
        
        # Check for general high-value patterns
        high_value_patterns = ['20d', '_avg_', 'efficiency', 'ratio', 'vol_']
        if any(pattern in feature_lower for pattern in high_value_patterns):
            return 2
        
        # Default to low priority
        return 1
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature selection results.
        
        Returns:
            Dict[str, Any]: Selection summary
        """
        return {
            'selected_features': self.selected_features.copy(),
            'feature_count': len(self.selected_features),
            'feature_scores': self.feature_scores.copy(),
            'selection_metadata': self.selection_metadata.copy(),
            'top_features': self._get_top_features(5)
        }
    
    def _get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N features by combined score.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: Top features with scores
        """
        if not self.feature_scores:
            return []
        
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def recommend_features_for_model_type(self, model_type: str) -> List[str]:
        """
        Recommend features based on model type.
        
        Args:
            model_type: Type of ML model ('linear', 'tree', 'ensemble', etc.)
            
        Returns:
            List[str]: Recommended features
        """
        if not self.selected_features:
            return []
        
        recommendations = self.selected_features.copy()
        
        try:
            if model_type.lower() in ['linear', 'ridge', 'lasso']:
                # Linear models prefer normalized, uncorrelated features
                # Remove highly correlated features more aggressively
                final_features = []
                for feature in recommendations:
                    # Prefer cost ratio and efficiency features for linear models
                    if any(pattern in feature.lower() for pattern in ['ratio', 'efficiency', 'avg', 'pct']):
                        final_features.append(feature)
                
                # Add other features if we don't have enough
                if len(final_features) < len(recommendations) // 2:
                    for feature in recommendations:
                        if feature not in final_features:
                            final_features.append(feature)
                            if len(final_features) >= len(recommendations) // 2:
                                break
                
                recommendations = final_features
            
            elif model_type.lower() in ['tree', 'forest', 'random_forest', 'gradient_boosting']:
                # Tree-based models can handle correlated features better
                # Prefer features with high variance and clear patterns
                tree_friendly = []
                for feature in recommendations:
                    if any(pattern in feature.lower() for pattern in ['vol', 'impact', 'momentum', 'regime']):
                        tree_friendly.append(feature)
                
                if len(tree_friendly) < len(recommendations) // 2:
                    tree_friendly.extend([f for f in recommendations if f not in tree_friendly])
                
                recommendations = tree_friendly
            
        except Exception as e:
            self.logger.error(f"Error in model-specific recommendations: {e}")
        
        return recommendations[:len(self.selected_features)]  # Don't exceed original selection