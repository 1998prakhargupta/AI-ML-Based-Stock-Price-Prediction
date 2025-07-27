"""
Cost Feature Configuration Module
=================================

Specialized configuration for cost feature generation and management.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class CostFeatureType(Enum):
    """Types of cost features that can be generated."""
    HISTORICAL_AVERAGE = "historical_average"
    VOLATILITY = "volatility"
    COST_TO_RETURN = "cost_to_return"
    BROKER_EFFICIENCY = "broker_efficiency"
    MARKET_IMPACT = "market_impact"
    LIQUIDITY_ADJUSTED = "liquidity_adjusted"
    SYNTHETIC = "synthetic"

class CostDataSource(Enum):
    """Sources of cost data."""
    ACTUAL = "actual"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"
    ESTIMATED = "estimated"

@dataclass
class CostFeatureConfig:
    """
    Detailed configuration for cost feature generation.
    
    Provides fine-grained control over how cost features are created,
    processed, and validated.
    """
    
    # Feature generation settings
    enabled_feature_types: List[CostFeatureType] = field(default_factory=lambda: [
        CostFeatureType.HISTORICAL_AVERAGE,
        CostFeatureType.VOLATILITY,
        CostFeatureType.COST_TO_RETURN,
        CostFeatureType.BROKER_EFFICIENCY
    ])
    
    # Lookback window settings
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ewma_alphas: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    min_periods: int = 5
    
    # Synthetic cost settings
    enable_synthetic_costs: bool = True
    synthetic_base_cost_bps: float = 5.0
    synthetic_volatility_factor: float = 0.001
    synthetic_volume_factor: float = 0.2
    synthetic_noise_factor: float = 0.1
    
    # Data source preferences
    preferred_data_source: CostDataSource = CostDataSource.HYBRID
    fallback_to_synthetic: bool = True
    
    # Feature validation settings
    validate_features: bool = True
    min_feature_variance: float = 1e-8
    max_feature_correlation: float = 0.95
    outlier_detection_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    outlier_threshold: float = 3.0
    
    # Feature naming settings
    feature_prefix: str = 'cost_'
    include_timeframe_in_name: bool = True
    include_type_in_name: bool = True
    
    # Processing settings
    normalize_features: bool = True
    normalization_method: str = 'robust'  # 'standard', 'robust', 'minmax'
    handle_missing_values: str = 'interpolate'  # 'drop', 'interpolate', 'forward_fill'
    
    # Performance settings
    batch_processing: bool = True
    batch_size: int = 1000
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Quality control settings
    quality_check_enabled: bool = True
    min_data_completeness: float = 0.8
    max_acceptable_na_ratio: float = 0.2
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate lookback windows
        if not all(w > 0 for w in self.lookback_windows):
            raise ValueError("All lookback windows must be positive")
        
        if self.min_periods <= 0:
            raise ValueError("Minimum periods must be positive")
        
        # Validate EWMA alphas
        if not all(0 < alpha <= 1 for alpha in self.ewma_alphas):
            raise ValueError("EWMA alphas must be between 0 and 1")
        
        # Validate synthetic cost parameters
        if self.synthetic_base_cost_bps <= 0:
            raise ValueError("Synthetic base cost must be positive")
        
        if self.synthetic_volatility_factor < 0:
            raise ValueError("Synthetic volatility factor must be non-negative")
        
        # Validate thresholds
        if not 0 < self.min_data_completeness <= 1:
            raise ValueError("Minimum data completeness must be between 0 and 1")
        
        if not 0 <= self.max_acceptable_na_ratio <= 1:
            raise ValueError("Max acceptable NA ratio must be between 0 and 1")
        
        if not 0 < self.max_feature_correlation <= 1:
            raise ValueError("Max feature correlation must be between 0 and 1")
    
    def get_feature_names(self) -> List[str]:
        """
        Generate expected feature names based on configuration.
        
        Returns:
            List[str]: Expected feature names
        """
        feature_names = []
        
        for feature_type in self.enabled_feature_types:
            type_name = feature_type.value if self.include_type_in_name else ""
            
            if feature_type == CostFeatureType.HISTORICAL_AVERAGE:
                # Moving averages
                for window in self.lookback_windows:
                    timeframe = f"_{window}d" if self.include_timeframe_in_name else ""
                    feature_names.append(f"{self.feature_prefix}avg{timeframe}")
                    feature_names.append(f"{self.feature_prefix}avg{timeframe}_pct")
                
                # EWMA
                for alpha in self.ewma_alphas:
                    feature_names.append(f"{self.feature_prefix}ewm_alpha_{alpha:.1f}")
            
            elif feature_type == CostFeatureType.VOLATILITY:
                for window in self.lookback_windows:
                    timeframe = f"_{window}d" if self.include_timeframe_in_name else ""
                    feature_names.append(f"{self.feature_prefix}vol{timeframe}")
                    feature_names.append(f"{self.feature_prefix}vol_rel{timeframe}")
                
                feature_names.append(f"{self.feature_prefix}vol_regime")
            
            elif feature_type == CostFeatureType.COST_TO_RETURN:
                for window in self.lookback_windows:
                    timeframe = f"_{window}d" if self.include_timeframe_in_name else ""
                    feature_names.append(f"{self.feature_prefix}return_ratio{timeframe}")
                    feature_names.append(f"{self.feature_prefix}drag{timeframe}")
                
                feature_names.append(f"{self.feature_prefix}immediate_return_ratio")
            
            elif feature_type == CostFeatureType.BROKER_EFFICIENCY:
                for window in self.lookback_windows:
                    timeframe = f"_{window}d" if self.include_timeframe_in_name else ""
                    feature_names.append(f"{self.feature_prefix}efficiency{timeframe}")
                    feature_names.append(f"{self.feature_prefix}consistency{timeframe}")
                
                feature_names.append(f"{self.feature_prefix}vs_spread")
            
            elif feature_type == CostFeatureType.MARKET_IMPACT:
                feature_names.extend([
                    f"{self.feature_prefix}per_volume",
                    f"{self.feature_prefix}volume_momentum",
                    f"{self.feature_prefix}momentum",
                    f"{self.feature_prefix}volume_momentum_ratio"
                ])
                
                for window in self.lookback_windows:
                    timeframe = f"_{window}d" if self.include_timeframe_in_name else ""
                    feature_names.append(f"{self.feature_prefix}volume_impact{timeframe}")
                    feature_names.append(f"{self.feature_prefix}volume_corr{timeframe}")
            
            elif feature_type == CostFeatureType.LIQUIDITY_ADJUSTED:
                for window in self.lookback_windows:
                    timeframe = f"_{window}d" if self.include_timeframe_in_name else ""
                    feature_names.append(f"{self.feature_prefix}liquidity_proxy{timeframe}")
                    feature_names.append(f"{self.feature_prefix}liquidity_adj{timeframe}")
                
                feature_names.extend([
                    f"{self.feature_prefix}price_range",
                    f"{self.feature_prefix}range_ratio"
                ])
        
        return feature_names
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Get feature processing configuration.
        
        Returns:
            Dict[str, Any]: Processing configuration
        """
        return {
            'normalize_features': self.normalize_features,
            'normalization_method': self.normalization_method,
            'handle_missing_values': self.handle_missing_values,
            'batch_processing': self.batch_processing,
            'batch_size': self.batch_size,
            'min_periods': self.min_periods
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """
        Get feature validation configuration.
        
        Returns:
            Dict[str, Any]: Validation configuration
        """
        return {
            'validate_features': self.validate_features,
            'min_variance': self.min_feature_variance,
            'max_correlation': self.max_feature_correlation,
            'outlier_method': self.outlier_detection_method,
            'outlier_threshold': self.outlier_threshold,
            'quality_check_enabled': self.quality_check_enabled,
            'min_completeness': self.min_data_completeness,
            'max_na_ratio': self.max_acceptable_na_ratio
        }
    
    def get_synthetic_config(self) -> Dict[str, Any]:
        """
        Get synthetic cost configuration.
        
        Returns:
            Dict[str, Any]: Synthetic cost configuration
        """
        return {
            'enable_synthetic': self.enable_synthetic_costs,
            'base_cost_bps': self.synthetic_base_cost_bps,
            'volatility_factor': self.synthetic_volatility_factor,
            'volume_factor': self.synthetic_volume_factor,
            'noise_factor': self.synthetic_noise_factor,
            'fallback_to_synthetic': self.fallback_to_synthetic
        }
    
    def is_feature_type_enabled(self, feature_type: CostFeatureType) -> bool:
        """
        Check if a specific feature type is enabled.
        
        Args:
            feature_type: Feature type to check
            
        Returns:
            bool: True if enabled
        """
        return feature_type in self.enabled_feature_types
    
    def get_cache_config(self) -> Dict[str, Any]:
        """
        Get caching configuration.
        
        Returns:
            Dict[str, Any]: Cache configuration
        """
        return {
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl_seconds,
            'cache_key_prefix': f"{self.feature_prefix}cache"
        }
    
    def create_feature_subset_config(self, feature_types: List[CostFeatureType]) -> 'CostFeatureConfig':
        """
        Create a new configuration with subset of feature types.
        
        Args:
            feature_types: Feature types to include
            
        Returns:
            CostFeatureConfig: New configuration with subset
        """
        import copy
        new_config = copy.deepcopy(self)
        new_config.enabled_feature_types = feature_types
        return new_config
    
    def merge_with_config(self, other_config: 'CostFeatureConfig') -> 'CostFeatureConfig':
        """
        Merge with another configuration (other takes precedence).
        
        Args:
            other_config: Configuration to merge with
            
        Returns:
            CostFeatureConfig: Merged configuration
        """
        import copy
        merged_config = copy.deepcopy(self)
        
        # Merge enabled feature types (union)
        merged_config.enabled_feature_types = list(set(
            self.enabled_feature_types + other_config.enabled_feature_types
        ))
        
        # Merge lookback windows (union)
        merged_config.lookback_windows = list(set(
            self.lookback_windows + other_config.lookback_windows
        ))
        merged_config.lookback_windows.sort()
        
        # Merge EWMA alphas (union)
        merged_config.ewma_alphas = list(set(
            self.ewma_alphas + other_config.ewma_alphas
        ))
        merged_config.ewma_alphas.sort()
        
        # Other settings: other config takes precedence
        merged_config.min_periods = other_config.min_periods
        merged_config.enable_synthetic_costs = other_config.enable_synthetic_costs
        merged_config.synthetic_base_cost_bps = other_config.synthetic_base_cost_bps
        merged_config.preferred_data_source = other_config.preferred_data_source
        merged_config.validate_features = other_config.validate_features
        merged_config.normalize_features = other_config.normalize_features
        merged_config.normalization_method = other_config.normalization_method
        
        return merged_config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config_dict = {}
        
        # Convert enums to strings
        config_dict['enabled_feature_types'] = [ft.value for ft in self.enabled_feature_types]
        config_dict['preferred_data_source'] = self.preferred_data_source.value
        
        # Add other fields
        for field_name in [
            'lookback_windows', 'ewma_alphas', 'min_periods',
            'enable_synthetic_costs', 'synthetic_base_cost_bps',
            'synthetic_volatility_factor', 'synthetic_volume_factor',
            'synthetic_noise_factor', 'fallback_to_synthetic',
            'validate_features', 'min_feature_variance',
            'max_feature_correlation', 'outlier_detection_method',
            'outlier_threshold', 'feature_prefix',
            'include_timeframe_in_name', 'include_type_in_name',
            'normalize_features', 'normalization_method',
            'handle_missing_values', 'batch_processing',
            'batch_size', 'enable_caching', 'cache_ttl_seconds',
            'quality_check_enabled', 'min_data_completeness',
            'max_acceptable_na_ratio'
        ]:
            config_dict[field_name] = getattr(self, field_name)
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CostFeatureConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            CostFeatureConfig: Configuration instance
        """
        # Convert string enums back
        if 'enabled_feature_types' in config_dict:
            config_dict['enabled_feature_types'] = [
                CostFeatureType(ft) for ft in config_dict['enabled_feature_types']
            ]
        
        if 'preferred_data_source' in config_dict:
            config_dict['preferred_data_source'] = CostDataSource(config_dict['preferred_data_source'])
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"CostFeatureConfig("
                f"types={len(self.enabled_feature_types)}, "
                f"windows={self.lookback_windows}, "
                f"synthetic={self.enable_synthetic_costs})")


# Factory functions for common configurations
def create_minimal_cost_feature_config() -> CostFeatureConfig:
    """Create minimal cost feature configuration."""
    return CostFeatureConfig(
        enabled_feature_types=[CostFeatureType.HISTORICAL_AVERAGE],
        lookback_windows=[20],
        ewma_alphas=[0.3],
        enable_synthetic_costs=True,
        validate_features=False,
        normalize_features=False,
        batch_processing=False,
        enable_caching=False
    )

def create_standard_cost_feature_config() -> CostFeatureConfig:
    """Create standard cost feature configuration."""
    return CostFeatureConfig(
        enabled_feature_types=[
            CostFeatureType.HISTORICAL_AVERAGE,
            CostFeatureType.VOLATILITY,
            CostFeatureType.COST_TO_RETURN
        ],
        lookback_windows=[5, 10, 20, 50],
        ewma_alphas=[0.1, 0.3, 0.5],
        enable_synthetic_costs=True,
        validate_features=True,
        normalize_features=True,
        batch_processing=True,
        enable_caching=True
    )

def create_comprehensive_cost_feature_config() -> CostFeatureConfig:
    """Create comprehensive cost feature configuration."""
    return CostFeatureConfig(
        enabled_feature_types=list(CostFeatureType),
        lookback_windows=[5, 10, 20, 50, 100],
        ewma_alphas=[0.05, 0.1, 0.2, 0.3, 0.5],
        enable_synthetic_costs=True,
        validate_features=True,
        normalize_features=True,
        normalization_method='robust',
        batch_processing=True,
        enable_caching=True,
        quality_check_enabled=True,
        outlier_detection_method='iqr'
    )