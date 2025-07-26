"""
Abstract Base Cost Calculator
=============================

Abstract base class for all transaction cost calculators.
Defines the common interface and provides standard functionality
for validation, error handling, and logging integration.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import from our modules
from .models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    InstrumentType,
    TransactionType,
    MarketTiming
)
from .exceptions import (
    TransactionCostError,
    InvalidTransactionError,
    BrokerConfigurationError,
    CalculationError,
    DataValidationError,
    MarketDataError,
    UnsupportedInstrumentError,
    raise_invalid_transaction,
    raise_broker_config_error,
    raise_calculation_error
)
from .constants import (
    SYSTEM_DEFAULTS,
    CONFIDENCE_LEVELS,
    get_volume_category
)

logger = logging.getLogger(__name__)


class CalculationMode:
    """Enumeration of calculation modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HISTORICAL = "historical"
    SIMULATION = "simulation"


class CostCalculatorBase(ABC):
    """
    Abstract base class for all transaction cost calculators.
    
    This class provides the standard interface and common functionality
    for all cost calculation implementations. It handles validation,
    error handling, logging, and provides hooks for async operations.
    
    Subclasses must implement the core calculation logic while inheriting
    standard error handling, validation, and logging capabilities.
    """
    
    def __init__(
        self,
        calculator_name: str,
        version: str = "1.0.0",
        supported_instruments: Optional[List[InstrumentType]] = None,
        supported_modes: Optional[List[str]] = None,
        default_timeout: int = None,
        enable_caching: bool = True
    ):
        """
        Initialize the cost calculator.
        
        Args:
            calculator_name: Unique name for this calculator
            version: Version of the calculator implementation
            supported_instruments: List of supported instrument types
            supported_modes: List of supported calculation modes
            default_timeout: Default timeout for calculations in seconds
            enable_caching: Whether to enable result caching
        """
        self.calculator_name = calculator_name
        self.version = version
        self.supported_instruments = supported_instruments or [InstrumentType.EQUITY]
        self.supported_modes = supported_modes or [
            CalculationMode.REAL_TIME,
            CalculationMode.BATCH,
            CalculationMode.HISTORICAL
        ]
        self.default_timeout = default_timeout or SYSTEM_DEFAULTS['calculation_timeout_seconds']
        self.enable_caching = enable_caching
        
        # Performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._last_calculation_time = None
        
        # Cache for results (simple in-memory cache)
        self._result_cache: Dict[str, tuple] = {}  # (result, timestamp)
        self._cache_duration = SYSTEM_DEFAULTS['cache_duration_seconds']
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{calculator_name}_")
        
        logger.info(f"Initialized {calculator_name} cost calculator v{version}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    # Public interface methods
    
    def calculate_cost(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        mode: str = CalculationMode.REAL_TIME
    ) -> TransactionCostBreakdown:
        """
        Calculate transaction costs synchronously.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration
            market_conditions: Current market conditions
            mode: Calculation mode
            
        Returns:
            Detailed cost breakdown
            
        Raises:
            InvalidTransactionError: If request validation fails
            BrokerConfigurationError: If broker config is invalid
            CalculationError: If calculation fails
            UnsupportedInstrumentError: If instrument not supported
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            self._validate_request(request)
            self._validate_broker_config(broker_config)
            self._validate_mode(mode)
            self._validate_instrument_support(request.instrument_type)
            
            # Check cache if enabled
            if self.enable_caching:
                cached_result = self._get_cached_result(request, broker_config, market_conditions)
                if cached_result:
                    logger.debug(f"Returning cached result for {request.symbol}")
                    return cached_result
            
            # Perform calculation
            result = self._execute_calculation(request, broker_config, market_conditions, mode)
            
            # Cache result if enabled
            if self.enable_caching:
                self._cache_result(request, broker_config, market_conditions, result)
            
            # Update performance metrics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(calculation_time)
            
            logger.info(
                f"Cost calculation completed for {request.symbol} "
                f"({request.instrument_type.name}) in {calculation_time:.3f}s"
            )
            
            return result
            
        except TransactionCostError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise CalculationError(
                f"Unexpected error during cost calculation: {str(e)}",
                calculation_step="main_calculation",
                original_exception=e,
                context={
                    'calculator': self.calculator_name,
                    'symbol': request.symbol,
                    'instrument_type': request.instrument_type.name
                }
            )
    
    async def calculate_cost_async(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        mode: str = CalculationMode.REAL_TIME
    ) -> TransactionCostBreakdown:
        """
        Calculate transaction costs asynchronously.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration
            market_conditions: Current market conditions
            mode: Calculation mode
            
        Returns:
            Detailed cost breakdown
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Run synchronous calculation in thread pool
            result = await loop.run_in_executor(
                self._executor,
                self.calculate_cost,
                request,
                broker_config,
                market_conditions,
                mode
            )
            return result
            
        except Exception as e:
            logger.error(f"Async calculation failed: {e}")
            raise
    
    def calculate_batch_costs(
        self,
        requests: List[TransactionRequest],
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        mode: str = CalculationMode.BATCH
    ) -> List[TransactionCostBreakdown]:
        """
        Calculate costs for multiple transactions in batch.
        
        Args:
            requests: List of transaction requests
            broker_config: Broker configuration
            market_conditions: Current market conditions
            mode: Calculation mode
            
        Returns:
            List of cost breakdowns
        """
        if not requests:
            return []
        
        results = []
        start_time = datetime.now()
        
        logger.info(f"Starting batch calculation for {len(requests)} transactions")
        
        for i, request in enumerate(requests):
            try:
                result = self.calculate_cost(request, broker_config, market_conditions, mode)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(requests)} transactions")
                    
            except Exception as e:
                logger.error(f"Failed to calculate cost for transaction {i}: {e}")
                # Create error result
                error_result = TransactionCostBreakdown()
                error_result.cost_details = {'error': str(e)}
                results.append(error_result)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch calculation completed in {total_time:.2f}s ({len(requests)/total_time:.1f} calc/s)")
        
        return results
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def _calculate_commission(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """
        Calculate commission for the transaction.
        
        Args:
            request: Transaction request
            broker_config: Broker configuration
            
        Returns:
            Commission amount
        """
        pass
    
    @abstractmethod
    def _calculate_regulatory_fees(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """
        Calculate regulatory fees for the transaction.
        
        Args:
            request: Transaction request
            broker_config: Broker configuration
            
        Returns:
            Total regulatory fees
        """
        pass
    
    @abstractmethod
    def _calculate_market_impact(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> Decimal:
        """
        Calculate market impact cost.
        
        Args:
            request: Transaction request
            market_conditions: Current market conditions
            
        Returns:
            Market impact cost
        """
        pass
    
    @abstractmethod
    def _get_supported_instruments(self) -> List[InstrumentType]:
        """
        Get list of supported instrument types.
        
        Returns:
            List of supported instrument types
        """
        pass
    
    # Core calculation execution
    
    def _execute_calculation(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        mode: str
    ) -> TransactionCostBreakdown:
        """
        Execute the main cost calculation logic.
        
        This method orchestrates the calculation by calling the abstract
        methods that must be implemented by subclasses.
        """
        try:
            # Create cost breakdown
            breakdown = TransactionCostBreakdown()
            breakdown.calculator_version = f"{self.calculator_name} v{self.version}"
            
            # Calculate different cost components
            breakdown.commission = self._calculate_commission(request, broker_config)
            breakdown.regulatory_fees = self._calculate_regulatory_fees(request, broker_config)
            breakdown.exchange_fees = self._calculate_exchange_fees(request, broker_config)
            
            # Market impact calculations
            if market_conditions:
                breakdown.bid_ask_spread_cost = self._calculate_bid_ask_spread_cost(request, market_conditions)
                breakdown.market_impact_cost = self._calculate_market_impact(request, market_conditions)
                breakdown.timing_cost = self._calculate_timing_cost(request, market_conditions)
            
            # Additional costs
            breakdown.platform_fees = self._calculate_platform_fees(request, broker_config)
            breakdown.borrowing_cost = self._calculate_borrowing_cost(request, broker_config)
            
            # Set confidence level
            breakdown.confidence_level = self._determine_confidence_level(request, market_conditions)
            
            # Add calculation details
            breakdown.cost_details = self._get_calculation_details(request, broker_config, market_conditions)
            
            return breakdown
            
        except Exception as e:
            raise CalculationError(
                f"Cost calculation execution failed: {str(e)}",
                calculation_step="execute_calculation",
                original_exception=e
            )
    
    # Default implementations for optional cost components
    
    def _calculate_exchange_fees(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """Default implementation for exchange fees."""
        return Decimal('0.00')
    
    def _calculate_bid_ask_spread_cost(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Default implementation for bid-ask spread cost."""
        if market_conditions.bid_ask_spread:
            return market_conditions.bid_ask_spread * Decimal(str(request.quantity)) / Decimal('2')
        return Decimal('0.00')
    
    def _calculate_timing_cost(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions
    ) -> Decimal:
        """Default implementation for timing cost."""
        return Decimal('0.00')
    
    def _calculate_platform_fees(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """Default implementation for platform fees."""
        return broker_config.platform_fee
    
    def _calculate_borrowing_cost(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """Default implementation for borrowing cost."""
        if request.transaction_type == TransactionType.SHORT:
            # This would need to be implemented based on specific borrowing rates
            return Decimal('0.00')
        return Decimal('0.00')
    
    # Validation methods
    
    def _validate_request(self, request: TransactionRequest) -> None:
        """Validate transaction request."""
        if not isinstance(request, TransactionRequest):
            raise_invalid_transaction("Request must be a TransactionRequest instance")
        
        if request.quantity <= 0:
            raise_invalid_transaction(f"Quantity must be positive, got {request.quantity}")
        
        if request.price <= 0:
            raise_invalid_transaction(f"Price must be positive, got {request.price}")
        
        if not request.symbol or not request.symbol.strip():
            raise_invalid_transaction("Symbol cannot be empty")
    
    def _validate_broker_config(self, broker_config: BrokerConfiguration) -> None:
        """Validate broker configuration."""
        if not isinstance(broker_config, BrokerConfiguration):
            raise_broker_config_error(
                "unknown",
                "Broker config must be a BrokerConfiguration instance"
            )
        
        if not broker_config.active:
            raise_broker_config_error(
                broker_config.broker_name,
                "Broker configuration is not active"
            )
    
    def _validate_mode(self, mode: str) -> None:
        """Validate calculation mode."""
        if mode not in self.supported_modes:
            raise CalculationError(
                f"Calculation mode '{mode}' not supported. "
                f"Supported modes: {self.supported_modes}",
                calculation_step="mode_validation"
            )
    
    def _validate_instrument_support(self, instrument_type: InstrumentType) -> None:
        """Validate instrument type support."""
        if instrument_type not in self.supported_instruments:
            raise UnsupportedInstrumentError(
                f"Instrument type {instrument_type.name} not supported",
                instrument_type=instrument_type.name,
                calculator_name=self.calculator_name
            )
    
    # Helper methods
    
    def _determine_confidence_level(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> float:
        """Determine confidence level for the calculation."""
        if not market_conditions:
            return CONFIDENCE_LEVELS['low']
        
        # Check data completeness and freshness
        data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
        
        if data_age < 60:  # Fresh data
            if market_conditions.bid_price and market_conditions.ask_price:
                return CONFIDENCE_LEVELS['high']
            else:
                return CONFIDENCE_LEVELS['medium']
        elif data_age < 300:  # 5 minutes
            return CONFIDENCE_LEVELS['medium']
        else:
            return CONFIDENCE_LEVELS['low']
    
    def _get_calculation_details(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> Dict[str, Any]:
        """Get detailed calculation information."""
        return {
            'calculator': self.calculator_name,
            'version': self.version,
            'volume_category': get_volume_category(request.quantity),
            'market_timing': request.market_timing.name,
            'broker': broker_config.broker_name,
            'has_market_data': market_conditions is not None,
            'calculation_timestamp': datetime.now().isoformat()
        }
    
    # Caching methods
    
    def _get_cache_key(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> str:
        """Generate cache key for the calculation."""
        key_parts = [
            request.symbol,
            str(request.quantity),
            str(request.price),
            request.transaction_type.name,
            request.instrument_type.name,
            broker_config.broker_name
        ]
        
        if market_conditions:
            key_parts.extend([
                str(market_conditions.bid_price) if market_conditions.bid_price else "None",
                str(market_conditions.ask_price) if market_conditions.ask_price else "None"
            ])
        
        return "|".join(key_parts)
    
    def _get_cached_result(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> Optional[TransactionCostBreakdown]:
        """Get cached result if available and fresh."""
        cache_key = self._get_cache_key(request, broker_config, market_conditions)
        
        if cache_key in self._result_cache:
            result, timestamp = self._result_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < self._cache_duration:
                return result
            else:
                # Remove stale cache entry
                del self._result_cache[cache_key]
        
        return None
    
    def _cache_result(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: TransactionCostBreakdown
    ) -> None:
        """Cache calculation result."""
        cache_key = self._get_cache_key(request, broker_config, market_conditions)
        self._result_cache[cache_key] = (result, datetime.now())
        
        # Simple cache cleanup - remove old entries
        if len(self._result_cache) > 1000:
            # Remove oldest 10% of entries
            sorted_keys = sorted(
                self._result_cache.keys(),
                key=lambda k: self._result_cache[k][1]
            )
            for key in sorted_keys[:100]:
                del self._result_cache[key]
    
    def _update_performance_metrics(self, calculation_time: float) -> None:
        """Update performance tracking metrics."""
        self._calculation_count += 1
        self._total_calculation_time += calculation_time
        self._last_calculation_time = calculation_time
    
    # Public utility methods
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = (
            self._total_calculation_time / self._calculation_count
            if self._calculation_count > 0
            else 0.0
        )
        
        return {
            'calculator_name': self.calculator_name,
            'version': self.version,
            'total_calculations': self._calculation_count,
            'average_calculation_time': avg_time,
            'last_calculation_time': self._last_calculation_time,
            'cache_size': len(self._result_cache) if self.enable_caching else 0
        }
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        if self.enable_caching:
            self._result_cache.clear()
            logger.info(f"Cache cleared for {self.calculator_name}")
    
    def get_supported_features(self) -> Dict[str, Any]:
        """Get information about supported features."""
        return {
            'calculator_name': self.calculator_name,
            'version': self.version,
            'supported_instruments': [inst.name for inst in self.supported_instruments],
            'supported_modes': self.supported_modes,
            'caching_enabled': self.enable_caching,
            'async_support': True,
            'batch_support': True
        }


logger.info("Base cost calculator class loaded successfully")