"""
Execution Delay Slippage Model
=============================

Calculates slippage costs due to execution delays.
Models price movement during the time between order placement and execution.
"""

from decimal import Decimal
from datetime import timedelta
from typing import Optional
import logging

from .base_slippage_model import BaseSlippageModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class DelaySlippageModel(BaseSlippageModel):
    """
    Execution delay slippage model implementation.
    
    This model calculates slippage costs caused by the time delay
    between order placement and execution. Price can move unfavorably
    during this period, especially in volatile markets.
    """
    
    def __init__(self, drift_coefficient: Decimal = Decimal('0.0')):
        """
        Initialize the delay slippage model.
        
        Args:
            drift_coefficient: Expected price drift per second (usually close to 0)
        """
        super().__init__("Execution Delay Slippage")
        self.drift_coefficient = drift_coefficient
        self.logger.info(f"Initialized Delay Slippage Model with drift={drift_coefficient}")
    
    def _calculate_slippage_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        execution_delay: Optional[timedelta]
    ) -> Decimal:
        """
        Calculate slippage due to execution delay.
        
        The model estimates price movement during execution delay:
        Slippage = √(delay_seconds) × volatility × price × quantity × direction
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Time delay between order and execution
            
        Returns:
            Slippage cost in absolute currency terms
        """
        # Get delay in seconds
        delay_seconds = self._calculate_delay_seconds(execution_delay)
        
        # Get volatility (converted to per-second basis)
        daily_volatility = self._get_volatility(market_conditions)
        # Convert daily volatility to per-second (assuming 6.5 hour trading day = 23,400 seconds)
        volatility_per_second = daily_volatility / Decimal('23400').sqrt()
        
        # Calculate expected price movement due to delay
        # Using square root of time for random walk model
        delay_volatility = volatility_per_second * delay_seconds.sqrt()
        
        # Add any drift component
        drift_component = self.drift_coefficient * delay_seconds
        
        # Total expected movement
        expected_movement = delay_volatility + abs(drift_component)
        
        # Calculate slippage cost
        # For buy orders, assume adverse movement increases cost
        # For sell orders, assume adverse movement decreases proceeds
        notional_value = Decimal(str(request.quantity)) * request.price
        slippage_cost = expected_movement * notional_value
        
        self.logger.debug(
            f"Delay slippage calculation: "
            f"delay_seconds={delay_seconds:.2f}, "
            f"daily_volatility={daily_volatility:.4f}, "
            f"volatility_per_second={volatility_per_second:.6f}, "
            f"expected_movement={expected_movement:.4f}, "
            f"slippage_cost={slippage_cost:.2f}"
        )
        
        return slippage_cost
    
    def _requires_market_data(self) -> bool:
        """
        This model requires volatility data.
        
        Returns:
            True as volatility is needed for calculation
        """
        return True
    
    def get_model_description(self) -> str:
        """
        Get detailed model description.
        
        Returns:
            Model description string
        """
        return (
            f"Execution Delay Slippage Model (drift={self.drift_coefficient}): "
            f"Models price movement during execution delay using random walk. "
            f"Slippage = √(delay) × volatility × notional."
        )