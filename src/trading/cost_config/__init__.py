"""
Cost Configuration Module
=========================

Configuration management for transaction cost calculations.
Integrates with the existing configuration system to provide
specialized settings for cost modeling and broker configurations.
"""

__version__ = "1.0.0"

from .base_config import CostConfiguration
from .config_validator import CostConfigurationValidator

__all__ = [
    'CostConfiguration',
    'CostConfigurationValidator'
]