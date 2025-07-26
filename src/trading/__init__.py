"""
Trading Module
==============

Core trading functionality including transaction cost modeling and analysis.
This module provides the infrastructure for calculating, analyzing, and optimizing
transaction costs across different brokers and trading strategies.
"""

__version__ = "1.0.0"
__author__ = "Stock Price Predictor Team"

# Import core components for easy access
try:
    from .transaction_costs.models import (
        TransactionRequest,
        TransactionCostBreakdown,
        MarketConditions,
        BrokerConfiguration
    )
    from .transaction_costs.base_cost_calculator import CostCalculatorBase
    from .transaction_costs.exceptions import (
        TransactionCostError,
        InvalidTransactionError,
        BrokerConfigurationError,
        CalculationError
    )
    from .cost_config.base_config import CostConfiguration
    
    __all__ = [
        'TransactionRequest',
        'TransactionCostBreakdown', 
        'MarketConditions',
        'BrokerConfiguration',
        'CostCalculatorBase',
        'TransactionCostError',
        'InvalidTransactionError',
        'BrokerConfigurationError',
        'CalculationError',
        'CostConfiguration'
    ]
    
except ImportError:
    # Components may not be available during initial setup
    __all__ = []