"""
Transaction Cost Exception Classes
==================================

Custom exception hierarchy for transaction cost calculations.
Provides comprehensive error categorization and handling for all
transaction cost related operations.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TransactionCostError(Exception):
    """
    Base exception class for all transaction cost related errors.
    
    Provides a common interface for error handling across the transaction
    cost calculation framework with enhanced error context and logging.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize transaction cost error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Additional context information about the error
            original_exception: Original exception if this is a wrapper
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.original_exception = original_exception
        
        # Log the error
        logger.error(
            f"TransactionCostError: {self.error_code} - {self.message}",
            extra={'context': self.context}
        )
    
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'original_exception': str(self.original_exception) if self.original_exception else None
        }


class InvalidTransactionError(TransactionCostError):
    """
    Raised when transaction request data is invalid or incomplete.
    
    This exception is thrown when the transaction request fails validation
    or contains inconsistent/missing data required for cost calculation.
    """
    
    def __init__(
        self,
        message: str,
        invalid_fields: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize invalid transaction error.
        
        Args:
            message: Error message describing the validation failure
            invalid_fields: List of field names that failed validation
            **kwargs: Additional arguments passed to parent class
        """
        self.invalid_fields = invalid_fields or []
        
        # Add invalid fields to context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context']['invalid_fields'] = self.invalid_fields
        
        super().__init__(message, error_code="INVALID_TRANSACTION", **kwargs)


class BrokerConfigurationError(TransactionCostError):
    """
    Raised when broker configuration is invalid or missing required settings.
    
    This exception is thrown when broker configuration fails validation
    or when required broker-specific parameters are missing or invalid.
    """
    
    def __init__(
        self,
        message: str,
        broker_name: Optional[str] = None,
        config_errors: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize broker configuration error.
        
        Args:
            message: Error message describing the configuration issue
            broker_name: Name of the broker with configuration problems
            config_errors: Dictionary of configuration field errors
            **kwargs: Additional arguments passed to parent class
        """
        self.broker_name = broker_name
        self.config_errors = config_errors or {}
        
        # Add broker context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context'].update({
            'broker_name': self.broker_name,
            'config_errors': self.config_errors
        })
        
        super().__init__(message, error_code="BROKER_CONFIG_ERROR", **kwargs)


class CalculationError(TransactionCostError):
    """
    Raised when cost calculation fails due to computational errors.
    
    This exception is thrown when the actual cost calculation process
    encounters errors such as missing market data, mathematical errors,
    or other computational failures.
    """
    
    def __init__(
        self,
        message: str,
        calculation_step: Optional[str] = None,
        missing_data: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize calculation error.
        
        Args:
            message: Error message describing the calculation failure
            calculation_step: Specific step in calculation that failed
            missing_data: List of missing data elements required for calculation
            **kwargs: Additional arguments passed to parent class
        """
        self.calculation_step = calculation_step
        self.missing_data = missing_data or []
        
        # Add calculation context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context'].update({
            'calculation_step': self.calculation_step,
            'missing_data': self.missing_data
        })
        
        super().__init__(message, error_code="CALCULATION_ERROR", **kwargs)


class DataValidationError(TransactionCostError):
    """
    Raised when input data fails validation checks.
    
    This exception is thrown when data validation fails for any input
    to the transaction cost calculation system, including market data,
    configuration data, or request parameters.
    """
    
    def __init__(
        self,
        message: str,
        validation_failures: Optional[Dict[str, str]] = None,
        data_source: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize data validation error.
        
        Args:
            message: Error message describing the validation failure
            validation_failures: Dictionary of field validation failures
            data_source: Source of the data that failed validation
            **kwargs: Additional arguments passed to parent class
        """
        self.validation_failures = validation_failures or {}
        self.data_source = data_source
        
        # Add validation context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context'].update({
            'validation_failures': self.validation_failures,
            'data_source': self.data_source
        })
        
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)


class MarketDataError(TransactionCostError):
    """
    Raised when market data is unavailable, stale, or corrupted.
    
    This exception is thrown when market data required for cost calculation
    is missing, outdated, or contains invalid values that prevent accurate
    cost calculation.
    """
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        data_age_seconds: Optional[int] = None,
        required_fields: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize market data error.
        
        Args:
            message: Error message describing the market data issue
            symbol: Symbol for which market data is problematic
            data_age_seconds: Age of the stale data in seconds
            required_fields: List of required market data fields that are missing
            **kwargs: Additional arguments passed to parent class
        """
        self.symbol = symbol
        self.data_age_seconds = data_age_seconds
        self.required_fields = required_fields or []
        
        # Add market data context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context'].update({
            'symbol': self.symbol,
            'data_age_seconds': self.data_age_seconds,
            'required_fields': self.required_fields
        })
        
        super().__init__(message, error_code="MARKET_DATA_ERROR", **kwargs)


class UnsupportedInstrumentError(TransactionCostError):
    """
    Raised when attempting to calculate costs for unsupported instruments.
    
    This exception is thrown when a cost calculator encounters an instrument
    type or trading scenario that it doesn't support.
    """
    
    def __init__(
        self,
        message: str,
        instrument_type: Optional[str] = None,
        calculator_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize unsupported instrument error.
        
        Args:
            message: Error message describing the unsupported instrument
            instrument_type: Type of instrument that is unsupported
            calculator_name: Name of the calculator that doesn't support the instrument
            **kwargs: Additional arguments passed to parent class
        """
        self.instrument_type = instrument_type
        self.calculator_name = calculator_name
        
        # Add instrument context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context'].update({
            'instrument_type': self.instrument_type,
            'calculator_name': self.calculator_name
        })
        
        super().__init__(message, error_code="UNSUPPORTED_INSTRUMENT", **kwargs)


class RateLimitError(TransactionCostError):
    """
    Raised when rate limits are exceeded for data requests or calculations.
    
    This exception is thrown when the system encounters rate limiting
    from external data providers or internal throttling mechanisms.
    """
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        limit_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message describing the rate limit issue
            retry_after_seconds: Seconds to wait before retrying
            limit_type: Type of rate limit encountered
            **kwargs: Additional arguments passed to parent class
        """
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type
        
        # Add rate limit context
        if 'context' not in kwargs:
            kwargs['context'] = {}
        kwargs['context'].update({
            'retry_after_seconds': self.retry_after_seconds,
            'limit_type': self.limit_type
        })
        
        super().__init__(message, error_code="RATE_LIMIT_ERROR", **kwargs)


# Convenience functions for common error scenarios
def raise_invalid_transaction(message: str, **kwargs):
    """Raise InvalidTransactionError with standard formatting."""
    raise InvalidTransactionError(f"Invalid transaction: {message}", **kwargs)


def raise_broker_config_error(broker_name: str, message: str, **kwargs):
    """Raise BrokerConfigurationError with standard formatting."""
    raise BrokerConfigurationError(
        f"Broker configuration error for {broker_name}: {message}",
        broker_name=broker_name,
        **kwargs
    )


def raise_calculation_error(step: str, message: str, **kwargs):
    """Raise CalculationError with standard formatting."""
    raise CalculationError(
        f"Calculation failed at step '{step}': {message}",
        calculation_step=step,
        **kwargs
    )


def raise_market_data_error(symbol: str, message: str, **kwargs):
    """Raise MarketDataError with standard formatting."""
    raise MarketDataError(
        f"Market data error for {symbol}: {message}",
        symbol=symbol,
        **kwargs
    )


logger.info("Transaction cost exception classes loaded successfully")