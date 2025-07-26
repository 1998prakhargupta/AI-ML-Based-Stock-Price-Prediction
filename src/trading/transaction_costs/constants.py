"""
Transaction Cost Constants
==========================

Constants and default values for transaction cost calculations.
Includes regulatory fee rates, commission structures, market parameters,
and other static values used throughout the cost calculation framework.
"""

from decimal import Decimal
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Regulatory fee rates (as of 2024 - should be updated regularly)
REGULATORY_FEE_RATES = {
    # SEC Transaction Fee (Section 31)
    'SEC_FEE_RATE': Decimal('0.0000051'),  # $5.10 per $1 million of transaction value
    
    # FINRA Trading Activity Fee (TAF)
    'FINRA_TAF_SELL': Decimal('0.000166'),  # $0.166 per $1,000 of principal (sell side only)
    'FINRA_TAF_BUY': Decimal('0.0'),  # No TAF on buy side
    
    # Options Regulatory Fee (ORF)
    'OPTIONS_ORF': Decimal('0.0455'),  # $0.0455 per contract side
    
    # FINRA Option Trading Activity Fee
    'OPTIONS_TAF': Decimal('0.0095'),  # $0.0095 per contract
}

# Default commission rates by broker type
DEFAULT_COMMISSION_RATES = {
    'discount_broker': {
        'equity_commission': Decimal('0.00'),  # Many brokers offer zero commission
        'options_commission': Decimal('0.65'),  # Per contract
        'futures_commission': Decimal('2.25'),  # Per contract
        'min_commission': Decimal('0.00'),
        'max_commission': None,
    },
    'full_service_broker': {
        'equity_commission': Decimal('25.00'),  # Flat fee or percentage
        'options_commission': Decimal('25.00'),  # Plus per contract
        'futures_commission': Decimal('25.00'),  # Plus per contract
        'min_commission': Decimal('25.00'),
        'max_commission': Decimal('250.00'),
    },
    'institutional_broker': {
        'equity_commission': Decimal('0.005'),  # Per share
        'options_commission': Decimal('0.50'),  # Per contract
        'futures_commission': Decimal('1.50'),  # Per contract
        'min_commission': Decimal('1.00'),
        'max_commission': None,
    }
}

# Market timing definitions
MARKET_HOURS = {
    'US_EQUITY': {
        'regular_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        },
        'pre_market': {
            'start': '04:00',
            'end': '09:30',
            'timezone': 'US/Eastern'
        },
        'after_hours': {
            'start': '16:00',
            'end': '20:00',
            'timezone': 'US/Eastern'
        }
    },
    'US_OPTIONS': {
        'regular_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        },
        'extended_hours': False
    },
    'US_FUTURES': {
        'regular_hours': {
            'start': '17:00',  # Sunday
            'end': '16:00',    # Friday
            'timezone': 'US/Central'
        },
        'globex_hours': {
            'start': '17:00',
            'end': '16:00',
            'timezone': 'US/Central'
        }
    }
}

# Supported currencies and their properties
SUPPORTED_CURRENCIES = {
    'USD': {
        'name': 'US Dollar',
        'symbol': '$',
        'decimal_places': 2,
        'default_market': 'US'
    },
    'EUR': {
        'name': 'Euro',
        'symbol': '€',
        'decimal_places': 2,
        'default_market': 'EU'
    },
    'GBP': {
        'name': 'British Pound',
        'symbol': '£',
        'decimal_places': 2,
        'default_market': 'UK'
    },
    'JPY': {
        'name': 'Japanese Yen',
        'symbol': '¥',
        'decimal_places': 0,
        'default_market': 'JP'
    },
    'CAD': {
        'name': 'Canadian Dollar',
        'symbol': 'C$',
        'decimal_places': 2,
        'default_market': 'CA'
    },
    'CHF': {
        'name': 'Swiss Franc',
        'symbol': 'CHF',
        'decimal_places': 2,
        'default_market': 'CH'
    },
    'AUD': {
        'name': 'Australian Dollar',
        'symbol': 'A$',
        'decimal_places': 2,
        'default_market': 'AU'
    },
    'INR': {
        'name': 'Indian Rupee',
        'symbol': '₹',
        'decimal_places': 2,
        'default_market': 'IN'
    }
}

# Market impact model parameters
MARKET_IMPACT_PARAMETERS = {
    'large_cap': {
        'temporary_impact_coefficient': Decimal('0.142'),
        'permanent_impact_coefficient': Decimal('0.314'),
        'volatility_adjustment': Decimal('1.0'),
        'adv_threshold': 1000000  # Average daily volume threshold
    },
    'mid_cap': {
        'temporary_impact_coefficient': Decimal('0.225'),
        'permanent_impact_coefficient': Decimal('0.442'),
        'volatility_adjustment': Decimal('1.2'),
        'adv_threshold': 100000
    },
    'small_cap': {
        'temporary_impact_coefficient': Decimal('0.315'),
        'permanent_impact_coefficient': Decimal('0.628'),
        'volatility_adjustment': Decimal('1.5'),
        'adv_threshold': 10000
    }
}

# Bid-ask spread estimation parameters
BID_ASK_SPREAD_ESTIMATES = {
    'large_cap': {
        'base_spread_bps': Decimal('5.0'),  # 5 basis points
        'volatility_multiplier': Decimal('0.5'),
        'volume_adjustment': Decimal('0.8')
    },
    'mid_cap': {
        'base_spread_bps': Decimal('15.0'),  # 15 basis points
        'volatility_multiplier': Decimal('0.8'),
        'volume_adjustment': Decimal('1.0')
    },
    'small_cap': {
        'base_spread_bps': Decimal('35.0'),  # 35 basis points
        'volatility_multiplier': Decimal('1.2'),
        'volume_adjustment': Decimal('1.5')
    }
}

# Volume thresholds for different calculation methods
VOLUME_THRESHOLDS = {
    'small_order': 100,      # Less than 100 shares
    'medium_order': 1000,    # 100-1000 shares
    'large_order': 10000,    # 1000-10000 shares
    'block_order': 100000    # Greater than 10000 shares
}

# Calculation confidence levels
CONFIDENCE_LEVELS = {
    'high': 0.95,     # Real-time market data, complete information
    'medium': 0.80,   # Recent market data, most information available
    'low': 0.60,      # Estimated data, limited information
    'very_low': 0.30  # Historical estimates, minimal information
}

# Default timeout and retry settings
SYSTEM_DEFAULTS = {
    'calculation_timeout_seconds': 30,
    'max_retries': 3,
    'retry_delay_seconds': 1,
    'cache_duration_seconds': 300,  # 5 minutes
    'market_data_staleness_threshold_seconds': 60,  # 1 minute
    'default_precision': 4  # Decimal places for cost calculations
}

# Exchange-specific parameters
EXCHANGE_PARAMETERS = {
    'NYSE': {
        'market_hours': MARKET_HOURS['US_EQUITY'],
        'tick_size': Decimal('0.01'),
        'board_lot': 100,
        'settlement_days': 1  # T+1
    },
    'NASDAQ': {
        'market_hours': MARKET_HOURS['US_EQUITY'],
        'tick_size': Decimal('0.01'),
        'board_lot': 100,
        'settlement_days': 1  # T+1
    },
    'CBOE': {
        'market_hours': MARKET_HOURS['US_OPTIONS'],
        'tick_size': Decimal('0.01'),
        'board_lot': 1,
        'settlement_days': 1  # T+1
    },
    'CME': {
        'market_hours': MARKET_HOURS['US_FUTURES'],
        'tick_size': Decimal('0.25'),  # Varies by contract
        'board_lot': 1,
        'settlement_days': 0  # Daily settlement
    }
}

# Error message templates
ERROR_MESSAGES = {
    'invalid_quantity': "Quantity must be a positive integer, got: {quantity}",
    'invalid_price': "Price must be positive, got: {price}",
    'invalid_symbol': "Symbol cannot be empty or None",
    'unsupported_instrument': "Instrument type {instrument_type} not supported by {calculator_name}",
    'missing_market_data': "Required market data fields missing: {fields}",
    'stale_market_data': "Market data is stale (age: {age} seconds, threshold: {threshold} seconds)",
    'broker_config_invalid': "Broker configuration invalid for {broker_name}: {errors}",
    'calculation_failed': "Cost calculation failed at step '{step}': {reason}",
    'rate_limit_exceeded': "Rate limit exceeded for {service}. Retry after {seconds} seconds."
}

# Logging configuration
LOGGING_CONFIG = {
    'cost_calculation': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    },
    'performance': {
        'level': 'DEBUG',
        'format': '%(asctime)s - PERF - %(message)s'
    },
    'audit': {
        'level': 'INFO',
        'format': '%(asctime)s - AUDIT - %(message)s'
    }
}

# Version information
VERSION_INFO = {
    'framework_version': '1.0.0',
    'data_model_version': '1.0.0',
    'regulatory_data_version': '2024.1',
    'last_updated': '2024-01-01'
}


def get_regulatory_fee_rate(fee_type: str, default: Decimal = Decimal('0.0')) -> Decimal:
    """
    Get regulatory fee rate by type.
    
    Args:
        fee_type: Type of regulatory fee
        default: Default value if fee type not found
        
    Returns:
        Decimal fee rate
    """
    return REGULATORY_FEE_RATES.get(fee_type, default)


def get_commission_structure(broker_type: str) -> Dict[str, Any]:
    """
    Get default commission structure for broker type.
    
    Args:
        broker_type: Type of broker (discount, full_service, institutional)
        
    Returns:
        Dictionary containing commission structure
    """
    return DEFAULT_COMMISSION_RATES.get(broker_type, DEFAULT_COMMISSION_RATES['discount_broker'])


def get_market_hours(market: str) -> Dict[str, Any]:
    """
    Get market hours for specified market.
    
    Args:
        market: Market identifier
        
    Returns:
        Dictionary containing market hours information
    """
    return MARKET_HOURS.get(market, MARKET_HOURS['US_EQUITY'])


def is_supported_currency(currency_code: str) -> bool:
    """
    Check if currency is supported.
    
    Args:
        currency_code: Three-letter currency code
        
    Returns:
        True if currency is supported
    """
    return currency_code.upper() in SUPPORTED_CURRENCIES


def get_volume_category(quantity: int) -> str:
    """
    Categorize order by volume.
    
    Args:
        quantity: Order quantity
        
    Returns:
        Volume category string
    """
    if quantity < VOLUME_THRESHOLDS['small_order']:
        return 'micro'
    elif quantity < VOLUME_THRESHOLDS['medium_order']:
        return 'small'
    elif quantity < VOLUME_THRESHOLDS['large_order']:
        return 'medium'
    elif quantity < VOLUME_THRESHOLDS['block_order']:
        return 'large'
    else:
        return 'block'


logger.info("Transaction cost constants loaded successfully")