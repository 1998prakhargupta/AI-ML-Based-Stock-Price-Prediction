"""
Base Slippage Model
==================

Abstract base class for all slippage calculation models.
Provides common interface and validation for slippage calculations.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Import parent classes and models
from ..models import TransactionRequest, MarketConditions, OrderType
from ..exceptions import CalculationError, MarketDataError

logger = logging.getLogger(__name__)


class BaseSlippageModel(ABC):
    """
    Abstract base class for slippage models.
    
    Slippage represents the difference between expected and actual execution prices,
    caused by factors like execution delays, order size, and market conditions.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the slippage model.
        
        Args:
            model_name: Name of the slippage model
        """
        self.model_name = model_name
        self.logger = logger.getChild(f"{self.__class__.__name__}")
        
    def calculate_slippage(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions] = None,
        execution_delay: Optional[timedelta] = None
    ) -> Decimal:
        """
        Calculate slippage cost for a transaction.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Time delay between order and execution
            
        Returns:
            Slippage cost in absolute terms
            
        Raises:
            CalculationError: If calculation fails
            MarketDataError: If required market data is missing
        """
        try:
            # Validate inputs
            self._validate_request(request)
            
            # Check if market data is required
            if self._requires_market_data() and not market_conditions:
                raise MarketDataError(
                    "Market conditions required for this slippage model",
                    symbol=request.symbol
                )
            
            # Perform the calculation
            slippage = self._calculate_slippage_internal(
                request, market_conditions, execution_delay
            )
            
            # Validate result
            if slippage < 0:
                raise CalculationError(
                    "Slippage cost cannot be negative",
                    calculation_step="slippage_validation"
                )
                
            self.logger.debug(
                f"Calculated {self.model_name} slippage: {slippage} for {request.symbol}"
            )
            
            return slippage
            
        except Exception as e:
            if isinstance(e, (CalculationError, MarketDataError)):
                raise
            
            raise CalculationError(
                f"Slippage calculation failed: {str(e)}",
                calculation_step="slippage_calculation",
                original_exception=e,
                context={
                    'model': self.model_name,
                    'symbol': request.symbol,
                    'quantity': request.quantity
                }
            )
    
    @abstractmethod
    def _calculate_slippage_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        execution_delay: Optional[timedelta]
    ) -> Decimal:
        """
        Internal slippage calculation implementation.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Execution delay
            
        Returns:
            Slippage cost
        """
        pass
    
    def _requires_market_data(self) -> bool:
        """
        Whether this model requires market data.
        
        Returns:
            True if market conditions are required
        """
        return True
    
    def _validate_request(self, request: TransactionRequest) -> None:
        """
        Validate transaction request for slippage calculation.
        
        Args:
            request: Transaction request to validate
            
        Raises:
            CalculationError: If validation fails
        """
        if request.quantity <= 0:
            raise CalculationError(
                "Quantity must be positive for slippage calculation",
                calculation_step="input_validation"
            )
            
        if request.price <= 0:
            raise CalculationError(
                "Price must be positive for slippage calculation",
                calculation_step="input_validation"
            )
    
    def _get_bid_ask_spread(self, market_conditions: MarketConditions) -> Decimal:
        """
        Get bid-ask spread from market conditions.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Bid-ask spread in absolute terms
        """
        if market_conditions.bid_ask_spread:
            return market_conditions.bid_ask_spread
        elif market_conditions.bid_price and market_conditions.ask_price:
            return market_conditions.ask_price - market_conditions.bid_price
        else:
            # Default spread assumption (0.1% of price)
            price = market_conditions.last_price or market_conditions.mid_price
            if price:
                return price * Decimal('0.001')
            return Decimal('0.01')  # Absolute fallback
    
    def _get_volatility(self, market_conditions: MarketConditions) -> Decimal:
        """
        Get volatility measure from market conditions.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Volatility measure
        """
        if market_conditions.realized_volatility:
            return market_conditions.realized_volatility
        elif market_conditions.implied_volatility:
            return market_conditions.implied_volatility
        else:
            # Default volatility assumption (20% annualized = ~1.26% daily)
            return Decimal('0.0126')
    
    def _get_order_type_multiplier(self, order_type: OrderType) -> Decimal:
        """
        Get slippage multiplier based on order type.
        
        Args:
            order_type: Type of order
            
        Returns:
            Multiplier for slippage calculation
        """
        multipliers = {
            OrderType.MARKET: Decimal('1.0'),      # Full slippage
            OrderType.LIMIT: Decimal('0.3'),       # Reduced slippage
            OrderType.STOP: Decimal('1.2'),        # Higher slippage
            OrderType.STOP_LIMIT: Decimal('0.8'),  # Moderate slippage
            OrderType.TRAILING_STOP: Decimal('1.1') # Slightly higher
        }
        return multipliers.get(order_type, Decimal('1.0'))
    
    def _calculate_delay_seconds(self, execution_delay: Optional[timedelta]) -> Decimal:
        """
        Convert execution delay to seconds for calculation.
        
        Args:
            execution_delay: Execution delay timedelta
            
        Returns:
            Delay in seconds as Decimal
        """
        if execution_delay:
            return Decimal(str(execution_delay.total_seconds()))
        else:
            # Default delay assumption (1 second for market orders)
            return Decimal('1.0')
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters for reporting.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'model_name': self.model_name,
            'requires_market_data': self._requires_market_data()
        }