"""
Cost Configuration Module
=========================

Configuration management for transaction cost calculations.
Integrates with the existing configuration system to provide
specialized settings for cost modeling and broker configurations.
"""

__version__ = "1.0.0"

from .base_config import CostConfiguration
from .config_validator import CostConfigurationValidator, ValidationResult
from .config_manager import ConfigurationManager, ConfigurationPriority
from .broker_configs import (
    BrokerConfigurationManager,
    BrokerType,
    AccountTier,
    FeeStructure,
    APIConfiguration,
    DataProviderConfig
)
from .market_params import (
    MarketParameterManager,
    InstrumentClass,
    MarketSession,
    VolatilityConfiguration,
    LiquidityConfiguration,
    MarketImpactConfiguration,
    SpreadConfiguration,
    TimeBasedConfiguration
)
from .defaults import (
    get_default_configuration,
    get_development_overrides,
    get_production_overrides,
    get_testing_overrides,
    get_minimal_configuration,
    get_broker_specific_defaults
)
from .validators import (
    EnvironmentConfigurationValidator,
    RuntimeConfigurationValidator,
    ConfigurationFileValidator,
    validate_configuration_setup
)

__all__ = [
    # Base configuration
    'CostConfiguration',
    
    # Validation
    'CostConfigurationValidator',
    'ValidationResult',
    
    # Main configuration manager
    'ConfigurationManager',
    'ConfigurationPriority',
    
    # Broker configuration
    'BrokerConfigurationManager',
    'BrokerType',
    'AccountTier',
    'FeeStructure',
    'APIConfiguration',
    'DataProviderConfig',
    
    # Market parameters
    'MarketParameterManager',
    'InstrumentClass',
    'MarketSession',
    'VolatilityConfiguration',
    'LiquidityConfiguration',
    'MarketImpactConfiguration',
    'SpreadConfiguration',
    'TimeBasedConfiguration',
    
    # Default configurations
    'get_default_configuration',
    'get_development_overrides',
    'get_production_overrides',
    'get_testing_overrides',
    'get_minimal_configuration',
    'get_broker_specific_defaults',
    
    # Extended validators
    'EnvironmentConfigurationValidator',
    'RuntimeConfigurationValidator',
    'ConfigurationFileValidator',
    'validate_configuration_setup'
]