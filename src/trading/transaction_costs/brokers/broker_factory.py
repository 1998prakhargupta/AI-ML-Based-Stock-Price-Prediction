"""
Broker Factory
==============

Factory pattern implementation for creating broker-specific cost calculators.
Provides a centralized way to instantiate calculators based on broker names
and configurations with proper error handling.
"""

from typing import Dict, Type, Optional, List
import logging

from src.trading.transaction_costs.base_cost_calculator import CostCalculatorBase
from src.trading.transaction_costs.exceptions import BrokerConfigurationError, raise_broker_config_error
from .zerodha_calculator import ZerodhaCalculator
from .breeze_calculator import BreezeCalculator

logger = logging.getLogger(__name__)


class BrokerFactory:
    """
    Factory for creating broker-specific cost calculators.
    
    Supports configuration-based instantiation and provides
    error handling for unsupported brokers.
    """
    
    # Registry of supported brokers and their calculator classes
    SUPPORTED_BROKERS: Dict[str, Type[CostCalculatorBase]] = {
        'zerodha': ZerodhaCalculator,
        'kite': ZerodhaCalculator,  # Alias for Zerodha
        'icici': BreezeCalculator,
        'breeze': BreezeCalculator,  # Alias for ICICI Securities
        'icici_securities': BreezeCalculator
    }
    
    # Default configurations for each broker
    DEFAULT_CONFIGS = {
        'zerodha': {
            'exchange': 'NSE',
            'calculator_name': 'Zerodha'
        },
        'icici': {
            'exchange': 'NSE',
            'calculator_name': 'ICICI_Securities'
        }
    }
    
    @classmethod
    def create_calculator(
        cls,
        broker_name: str,
        exchange: str = 'NSE',
        **kwargs
    ) -> CostCalculatorBase:
        """
        Create a cost calculator for the specified broker.
        
        Args:
            broker_name: Name of the broker (case-insensitive)
            exchange: Exchange name (NSE/BSE)
            **kwargs: Additional configuration parameters
            
        Returns:
            Initialized cost calculator instance
            
        Raises:
            BrokerConfigurationError: If broker is not supported
        """
        broker_key = broker_name.lower().strip()
        
        if not broker_key:
            raise_broker_config_error(
                broker_name,
                "Broker name cannot be empty"
            )
        
        if broker_key not in cls.SUPPORTED_BROKERS:
            raise_broker_config_error(
                broker_name,
                f"Unsupported broker '{broker_name}'. "
                f"Supported brokers: {list(cls.SUPPORTED_BROKERS.keys())}"
            )
        
        calculator_class = cls.SUPPORTED_BROKERS[broker_key]
        
        try:
            # Create calculator with exchange parameter
            calculator = calculator_class(exchange=exchange, **kwargs)
            
            logger.info(
                f"Created {calculator_class.__name__} for broker '{broker_name}' "
                f"on {exchange} exchange"
            )
            
            return calculator
            
        except Exception as e:
            raise_broker_config_error(
                broker_name,
                f"Failed to create calculator for broker '{broker_name}': {str(e)}"
            )
    
    @classmethod
    def get_supported_brokers(cls) -> List[str]:
        """
        Get list of supported broker names.
        
        Returns:
            List of supported broker names
        """
        return list(cls.SUPPORTED_BROKERS.keys())
    
    @classmethod
    def is_broker_supported(cls, broker_name: str) -> bool:
        """
        Check if a broker is supported.
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            True if broker is supported, False otherwise
        """
        return broker_name.lower().strip() in cls.SUPPORTED_BROKERS
    
    @classmethod
    def get_broker_info(cls, broker_name: str) -> Optional[Dict]:
        """
        Get information about a specific broker.
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            Broker information dictionary or None if not supported
        """
        broker_key = broker_name.lower().strip()
        
        if broker_key not in cls.SUPPORTED_BROKERS:
            return None
        
        calculator_class = cls.SUPPORTED_BROKERS[broker_key]
        
        # Create a temporary instance to get supported features
        try:
            temp_calculator = calculator_class()
            features = temp_calculator.get_supported_features()
            
            return {
                'broker_name': broker_name,
                'calculator_class': calculator_class.__name__,
                'supported_instruments': features['supported_instruments'],
                'supported_modes': features['supported_modes'],
                'default_config': cls.DEFAULT_CONFIGS.get(broker_key, {})
            }
        except Exception as e:
            logger.error(f"Error getting info for broker {broker_name}: {e}")
            return {
                'broker_name': broker_name,
                'calculator_class': calculator_class.__name__,
                'error': str(e)
            }
    
    @classmethod
    def get_all_brokers_info(cls) -> Dict[str, Dict]:
        """
        Get information about all supported brokers.
        
        Returns:
            Dictionary mapping broker names to their information
        """
        brokers_info = {}
        
        for broker_name in cls.SUPPORTED_BROKERS.keys():
            brokers_info[broker_name] = cls.get_broker_info(broker_name)
        
        return brokers_info
    
    @classmethod
    def create_calculator_from_config(cls, config: Dict) -> CostCalculatorBase:
        """
        Create calculator from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing broker details
            
        Returns:
            Initialized cost calculator instance
            
        Example config:
            {
                'broker': 'zerodha',
                'exchange': 'NSE',
                'additional_params': {...}
            }
        """
        if 'broker' not in config:
            raise_broker_config_error(
                "unknown",
                "Configuration must contain 'broker' field"
            )
        
        broker_name = config['broker']
        exchange = config.get('exchange', 'NSE')
        
        # Extract additional parameters
        additional_params = config.get('additional_params', {})
        
        return cls.create_calculator(
            broker_name=broker_name,
            exchange=exchange,
            **additional_params
        )
    
    @classmethod
    def validate_broker_config(cls, config: Dict) -> bool:
        """
        Validate broker configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            BrokerConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise_broker_config_error(
                "unknown",
                "Configuration must be a dictionary"
            )
        
        if 'broker' not in config:
            raise_broker_config_error(
                "unknown",
                "Configuration must contain 'broker' field"
            )
        
        broker_name = config['broker']
        
        if not cls.is_broker_supported(broker_name):
            raise_broker_config_error(
                broker_name,
                f"Unsupported broker '{broker_name}'"
            )
        
        exchange = config.get('exchange', 'NSE')
        if exchange not in ['NSE', 'BSE']:
            raise_broker_config_error(
                broker_name,
                f"Unsupported exchange '{exchange}'. Must be NSE or BSE"
            )
        
        return True


# Convenience functions for common broker instantiation

def create_zerodha_calculator(exchange: str = 'NSE') -> ZerodhaCalculator:
    """
    Create Zerodha calculator with default configuration.
    
    Args:
        exchange: Exchange name (NSE/BSE)
        
    Returns:
        Initialized Zerodha calculator
    """
    return BrokerFactory.create_calculator('zerodha', exchange=exchange)


def create_icici_calculator(exchange: str = 'NSE') -> BreezeCalculator:
    """
    Create ICICI Securities calculator with default configuration.
    
    Args:
        exchange: Exchange name (NSE/BSE)
        
    Returns:
        Initialized ICICI Securities calculator
    """
    return BrokerFactory.create_calculator('icici', exchange=exchange)


logger.info("Broker Factory loaded successfully")