"""
Cost Integration Configuration Module
====================================

Provides configuration management for cost-aware ML integration.
Extends existing configuration system with cost-specific settings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing config manager
try:
    from src.utils.config_manager import Config
except ImportError:
    # Fallback for testing - mock config without validation
    class Config:
        def __init__(self): 
            self._config = {}
        def get(self, key, default=None): 
            return self._config.get(key, default)
        def get_data_paths(self):
            return {'data_save_path': './data', 'model_save_path': './models'}
        def get_model_save_path(self):
            return './models'

# Setup logging
logger = logging.getLogger(__name__)

class CostIntegrationLevel(Enum):
    """Levels of cost integration."""
    DISABLED = "disabled"
    BASIC = "basic"
    ADVANCED = "advanced"
    FULL = "full"

@dataclass
class CostIntegrationConfig:
    """
    Configuration class for cost-aware ML integration.
    
    Provides comprehensive configuration options for all cost-related
    capabilities while maintaining backward compatibility.
    """
    
    # Main integration settings
    integration_level: CostIntegrationLevel = CostIntegrationLevel.BASIC
    enable_cost_features: bool = True
    enable_cost_training: bool = True
    enable_cost_evaluation: bool = True
    
    # Feature generation settings
    cost_feature_lookback_windows: List[int] = None
    cost_feature_types: List[str] = None
    enable_synthetic_costs: bool = True
    synthetic_cost_base_bps: float = 5.0
    
    # Training settings
    enable_cost_weighting: bool = True
    enable_cost_regularization: bool = False
    cost_weight_factor: float = 1.0
    cost_penalty_factor: float = 0.1
    cost_threshold_percentile: float = 75.0
    
    # Evaluation settings
    enable_trading_simulation: bool = True
    enable_cost_analysis: bool = True
    trading_frequency: str = 'daily'
    benchmark_cost_bps: float = 5.0
    risk_free_rate: float = 0.02
    
    # Feature selection settings
    max_cost_features: Optional[int] = None
    cost_feature_selection_method: str = 'combined'
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # Performance settings
    cost_calculation_batch_size: int = 1000
    enable_parallel_processing: bool = True
    cache_cost_features: bool = True
    
    # Data validation settings
    validate_cost_data: bool = True
    cost_data_quality_threshold: float = 0.7
    handle_missing_costs: str = 'interpolate'  # 'interpolate', 'drop', 'synthetic'
    
    # Reporting settings
    enable_cost_reporting: bool = True
    cost_report_frequency: str = 'model'  # 'model', 'batch', 'session'
    detailed_cost_analysis: bool = True
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.cost_feature_lookback_windows is None:
            self.cost_feature_lookback_windows = [5, 10, 20, 50]
        
        if self.cost_feature_types is None:
            self.cost_feature_types = [
                'historical_average', 'volatility', 'cost_to_return',
                'broker_efficiency', 'market_impact', 'liquidity_adjusted'
            ]
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate lookback windows
        if any(w <= 0 for w in self.cost_feature_lookback_windows):
            raise ValueError("All lookback windows must be positive")
        
        # Validate percentile values
        if not 0 <= self.cost_threshold_percentile <= 100:
            raise ValueError("Cost threshold percentile must be between 0 and 100")
        
        # Validate factors
        if self.cost_weight_factor < 0:
            raise ValueError("Cost weight factor must be non-negative")
        
        if self.cost_penalty_factor < 0:
            raise ValueError("Cost penalty factor must be non-negative")
        
        # Validate thresholds
        if not 0 <= self.variance_threshold <= 1:
            raise ValueError("Variance threshold must be between 0 and 1")
        
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("Correlation threshold must be between 0 and 1")
        
        # Validate quality threshold
        if not 0 <= self.cost_data_quality_threshold <= 1:
            raise ValueError("Cost data quality threshold must be between 0 and 1")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CostIntegrationConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            CostIntegrationConfig: Configuration instance
        """
        # Handle enum conversion
        if 'integration_level' in config_dict:
            if isinstance(config_dict['integration_level'], str):
                config_dict['integration_level'] = CostIntegrationLevel(config_dict['integration_level'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'CostIntegrationConfig':
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to JSON configuration file
            
        Returns:
            CostIntegrationConfig: Configuration instance
        """
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            raise
    
    @classmethod
    def from_environment(cls, prefix: str = 'COST_ML_') -> 'CostIntegrationConfig':
        """
        Create configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            CostIntegrationConfig: Configuration instance
        """
        config_dict = {}
        
        # Map environment variables to config fields
        env_mappings = {
            f'{prefix}INTEGRATION_LEVEL': ('integration_level', str),
            f'{prefix}ENABLE_COST_FEATURES': ('enable_cost_features', bool),
            f'{prefix}ENABLE_COST_TRAINING': ('enable_cost_training', bool),
            f'{prefix}ENABLE_COST_EVALUATION': ('enable_cost_evaluation', bool),
            f'{prefix}COST_WEIGHT_FACTOR': ('cost_weight_factor', float),
            f'{prefix}COST_PENALTY_FACTOR': ('cost_penalty_factor', float),
            f'{prefix}BENCHMARK_COST_BPS': ('benchmark_cost_bps', float),
            f'{prefix}MAX_COST_FEATURES': ('max_cost_features', int),
            f'{prefix}VARIANCE_THRESHOLD': ('variance_threshold', float),
            f'{prefix}CORRELATION_THRESHOLD': ('correlation_threshold', float),
        }
        
        for env_var, (field_name, field_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if field_type == bool:
                        config_dict[field_name] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int:
                        config_dict[field_name] = int(env_value)
                    elif field_type == float:
                        config_dict[field_name] = float(env_value)
                    else:
                        config_dict[field_name] = env_value
                except ValueError as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}, error: {e}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config_dict = asdict(self)
        # Convert enum to string
        config_dict['integration_level'] = self.integration_level.value
        return config_dict
    
    def to_json_file(self, file_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
            raise
    
    def update_from_base_config(self, base_config: Config) -> None:
        """
        Update cost configuration from base configuration.
        
        Args:
            base_config: Base configuration object
        """
        try:
            # Update from base config if keys exist
            cost_config_mapping = {
                'COST_INTEGRATION_LEVEL': 'integration_level',
                'ENABLE_COST_FEATURES': 'enable_cost_features',
                'ENABLE_COST_TRAINING': 'enable_cost_training',
                'ENABLE_COST_EVALUATION': 'enable_cost_evaluation',
                'COST_WEIGHT_FACTOR': 'cost_weight_factor',
                'COST_PENALTY_FACTOR': 'cost_penalty_factor',
                'BENCHMARK_COST_BPS': 'benchmark_cost_bps',
                'MAX_COST_FEATURES': 'max_cost_features',
            }
            
            for base_key, cost_key in cost_config_mapping.items():
                value = base_config.get(base_key)
                if value is not None:
                    if cost_key == 'integration_level':
                        value = CostIntegrationLevel(value)
                    setattr(self, cost_key, value)
            
            # Re-validate after updates
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error updating from base config: {e}")
    
    def get_feature_config(self) -> Dict[str, Any]:
        """
        Get feature-specific configuration.
        
        Returns:
            Dict[str, Any]: Feature configuration
        """
        return {
            'lookback_windows': self.cost_feature_lookback_windows,
            'feature_types': self.cost_feature_types,
            'enable_synthetic': self.enable_synthetic_costs,
            'synthetic_base_bps': self.synthetic_cost_base_bps,
            'max_features': self.max_cost_features,
            'selection_method': self.cost_feature_selection_method,
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training-specific configuration.
        
        Returns:
            Dict[str, Any]: Training configuration
        """
        return {
            'enable_weighting': self.enable_cost_weighting,
            'enable_regularization': self.enable_cost_regularization,
            'weight_factor': self.cost_weight_factor,
            'penalty_factor': self.cost_penalty_factor,
            'threshold_percentile': self.cost_threshold_percentile
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation-specific configuration.
        
        Returns:
            Dict[str, Any]: Evaluation configuration
        """
        return {
            'enable_trading_simulation': self.enable_trading_simulation,
            'enable_cost_analysis': self.enable_cost_analysis,
            'trading_frequency': self.trading_frequency,
            'benchmark_cost_bps': self.benchmark_cost_bps,
            'risk_free_rate': self.risk_free_rate
        }
    
    def is_cost_integration_enabled(self) -> bool:
        """
        Check if cost integration is enabled.
        
        Returns:
            bool: True if cost integration is enabled
        """
        return self.integration_level != CostIntegrationLevel.DISABLED
    
    def get_integration_level_features(self) -> List[str]:
        """
        Get features enabled for current integration level.
        
        Returns:
            List[str]: Enabled features
        """
        features = []
        
        if self.integration_level == CostIntegrationLevel.DISABLED:
            return features
        
        if self.integration_level in [CostIntegrationLevel.BASIC, CostIntegrationLevel.ADVANCED, CostIntegrationLevel.FULL]:
            features.extend(['cost_features', 'basic_evaluation'])
        
        if self.integration_level in [CostIntegrationLevel.ADVANCED, CostIntegrationLevel.FULL]:
            features.extend(['cost_training', 'cost_analysis', 'feature_selection'])
        
        if self.integration_level == CostIntegrationLevel.FULL:
            features.extend(['advanced_evaluation', 'optimization', 'detailed_reporting'])
        
        return features
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """
        Get performance-related settings.
        
        Returns:
            Dict[str, Any]: Performance settings
        """
        return {
            'batch_size': self.cost_calculation_batch_size,
            'parallel_processing': self.enable_parallel_processing,
            'cache_features': self.cache_cost_features
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"CostIntegrationConfig(level={self.integration_level.value}, features={self.enable_cost_features}, training={self.enable_cost_training}, evaluation={self.enable_cost_evaluation})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"CostIntegrationConfig({', '.join(f'{k}={v}' for k, v in self.to_dict().items())})"


# Factory functions for common configurations
def create_basic_cost_config() -> CostIntegrationConfig:
    """Create basic cost integration configuration."""
    return CostIntegrationConfig(
        integration_level=CostIntegrationLevel.BASIC,
        enable_cost_features=True,
        enable_cost_training=False,
        enable_cost_evaluation=True,
        enable_cost_weighting=False,
        enable_cost_regularization=False,
        enable_trading_simulation=True,
        enable_cost_analysis=False
    )

def create_advanced_cost_config() -> CostIntegrationConfig:
    """Create advanced cost integration configuration."""
    return CostIntegrationConfig(
        integration_level=CostIntegrationLevel.ADVANCED,
        enable_cost_features=True,
        enable_cost_training=True,
        enable_cost_evaluation=True,
        enable_cost_weighting=True,
        enable_cost_regularization=True,
        enable_trading_simulation=True,
        enable_cost_analysis=True,
        detailed_cost_analysis=True
    )

def create_full_cost_config() -> CostIntegrationConfig:
    """Create full cost integration configuration."""
    return CostIntegrationConfig(
        integration_level=CostIntegrationLevel.FULL,
        enable_cost_features=True,
        enable_cost_training=True,
        enable_cost_evaluation=True,
        enable_cost_weighting=True,
        enable_cost_regularization=True,
        enable_trading_simulation=True,
        enable_cost_analysis=True,
        enable_parallel_processing=True,
        cache_cost_features=True,
        detailed_cost_analysis=True,
        enable_cost_reporting=True
    )

def create_disabled_cost_config() -> CostIntegrationConfig:
    """Create disabled cost integration configuration (backward compatibility)."""
    return CostIntegrationConfig(
        integration_level=CostIntegrationLevel.DISABLED,
        enable_cost_features=False,
        enable_cost_training=False,
        enable_cost_evaluation=False
    )