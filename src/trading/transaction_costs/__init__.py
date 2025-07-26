"""
Transaction Costs Module
========================

Core transaction cost calculation infrastructure for the Stock Price Predictor.
This module provides the foundation for calculating and analyzing transaction costs
across different brokers, instruments, and market conditions.

Key Components:
- Data models for transaction requests and cost breakdowns
- Abstract base class for cost calculators
- Market impact and slippage modeling
- Custom exception hierarchy for error handling
- Constants and configuration utilities
"""

__version__ = "1.0.0"

# Import core components
from .models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    TransactionType,
    InstrumentType,
    OrderType,
    MarketTiming
)

from .exceptions import (
    TransactionCostError,
    InvalidTransactionError,
    BrokerConfigurationError,
    CalculationError,
    DataValidationError,
    MarketDataError
)

try:
    from .base_cost_calculator import CostCalculatorBase
except ImportError:
    # May not be available during initial setup
    CostCalculatorBase = None

# Market Impact Models
try:
    from .market_impact import (
        BaseImpactModel,
        LinearImpactModel,
        SquareRootImpactModel,
        AdaptiveImpactModel,
        MarketConditionAnalyzer,
        ImpactCalibrator
    )
except ImportError:
    # Market impact models may not be available during initial setup
    pass

# Slippage Models
try:
    from .slippage import (
        BaseSlippageModel,
        DelaySlippageModel,
        SizeSlippageModel,
        ConditionSlippageModel,
        SlippageEstimator
    )
except ImportError:
    # Slippage models may not be available during initial setup
    pass

try:
    from .constants import (
        DEFAULT_COMMISSION_RATES,
        REGULATORY_FEE_RATES,
        MARKET_HOURS,
        SUPPORTED_CURRENCIES
    )
except ImportError:
    # Constants may not be available during initial setup
    pass

__all__ = [
    # Data models
    'TransactionRequest',
    'TransactionCostBreakdown',
    'MarketConditions',
    'BrokerConfiguration',
    
    # Enums
    'TransactionType',
    'InstrumentType',
    'OrderType',
    'MarketTiming',
    
    # Exceptions
    'TransactionCostError',
    'InvalidTransactionError',
    'BrokerConfigurationError',
    'CalculationError',
    'DataValidationError',
    'MarketDataError',
    
    # Base classes
    'CostCalculatorBase',
    
    # Market Impact Models
    'BaseImpactModel',
    'LinearImpactModel',
    'SquareRootImpactModel',
    'AdaptiveImpactModel',
    'MarketConditionAnalyzer',
    'ImpactCalibrator',
    
    # Slippage Models
    'BaseSlippageModel',
    'DelaySlippageModel',
    'SizeSlippageModel',
    'ConditionSlippageModel',
    'SlippageEstimator'
]