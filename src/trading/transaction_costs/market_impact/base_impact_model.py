"""
Base Market Impact Model
=======================

Abstract base class for all market impact calculation models.
Provides common interface and validation for market impact calculations.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# Import parent classes and models
from ..models import TransactionRequest, MarketConditions
from ..exceptions import CalculationError, MarketDataError

logger = logging.getLogger(__name__)


class BaseImpactModel(ABC):
    """
    Abstract base class for market impact models.
    
    Market impact represents the cost of moving market prices when placing orders.
    This cost is typically proportional to order size and inversely related to liquidity.
    """
    
    def __init__(self, model_name: str, alpha: Decimal = Decimal('0.1')):
        """
        Initialize the market impact model.
        
        Args:
            model_name: Name of the impact model
            alpha: Impact coefficient (model-specific parameter)
        """
        self.model_name = model_name
        self.alpha = alpha
        self.logger = logger.getChild(f"{self.__class__.__name__}")
        
    def calculate_impact(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """
        Calculate market impact cost for a transaction.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            
        Returns:
            Market impact cost in absolute terms
            
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
                    "Market conditions required for this impact model",
                    symbol=request.symbol
                )
            
            # Perform the calculation
            impact = self._calculate_impact_internal(request, market_conditions)
            
            # Validate result
            if impact < 0:
                raise CalculationError(
                    "Market impact cannot be negative",
                    calculation_step="impact_validation"
                )
                
            self.logger.debug(
                f"Calculated {self.model_name} impact: {impact} for {request.symbol}"
            )
            
            return impact
            
        except Exception as e:
            if isinstance(e, (CalculationError, MarketDataError)):
                raise
            
            raise CalculationError(
                f"Market impact calculation failed: {str(e)}",
                calculation_step="market_impact",
                original_exception=e,
                context={
                    'model': self.model_name,
                    'symbol': request.symbol,
                    'quantity': request.quantity
                }
            )
    
    @abstractmethod
    def _calculate_impact_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> Decimal:
        """
        Internal impact calculation implementation.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            
        Returns:
            Market impact cost
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
        Validate transaction request for impact calculation.
        
        Args:
            request: Transaction request to validate
            
        Raises:
            CalculationError: If validation fails
        """
        if request.quantity <= 0:
            raise CalculationError(
                "Quantity must be positive for impact calculation",
                calculation_step="input_validation"
            )
            
        if request.price <= 0:
            raise CalculationError(
                "Price must be positive for impact calculation", 
                calculation_step="input_validation"
            )
    
    def _get_participation_rate(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions
    ) -> Decimal:
        """
        Calculate participation rate (order size / average daily volume).
        
        Args:
            request: Transaction request
            market_conditions: Market conditions with volume data
            
        Returns:
            Participation rate as decimal (0.1 = 10%)
        """
        if not market_conditions.average_daily_volume:
            # Use current volume as proxy if ADV not available
            daily_volume = market_conditions.volume or 1000000  # Default fallback
        else:
            daily_volume = market_conditions.average_daily_volume
            
        if daily_volume <= 0:
            daily_volume = 1000000  # Safe fallback
            
        participation_rate = Decimal(str(request.quantity)) / Decimal(str(daily_volume))
        
        # Cap participation rate at reasonable maximum
        return min(participation_rate, Decimal('0.5'))  # Max 50% participation
    
    def _get_volatility(self, market_conditions: MarketConditions) -> Decimal:
        """
        Get volatility measure from market conditions.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Volatility measure (defaults to reasonable value if not available)
        """
        if market_conditions.realized_volatility:
            return market_conditions.realized_volatility
        elif market_conditions.implied_volatility:
            return market_conditions.implied_volatility
        else:
            # Default volatility assumption (20% annualized = ~1.26% daily)
            return Decimal('0.0126')
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters for reporting and calibration.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'model_name': self.model_name,
            'alpha': float(self.alpha),
            'requires_market_data': self._requires_market_data()
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update model parameters.
        
        Args:
            **kwargs: Parameter updates
        """
        if 'alpha' in kwargs:
            self.alpha = Decimal(str(kwargs['alpha']))
            self.logger.info(f"Updated alpha parameter to {self.alpha}")