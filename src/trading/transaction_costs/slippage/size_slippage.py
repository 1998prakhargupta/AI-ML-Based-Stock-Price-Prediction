"""
Order Size Slippage Model
========================

Calculates slippage costs based on order size relative to market liquidity.
Larger orders move through multiple price levels, increasing slippage.
"""

from decimal import Decimal
from datetime import timedelta
from typing import Optional
import logging

from .base_slippage_model import BaseSlippageModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class SizeSlippageModel(BaseSlippageModel):
    """
    Order size slippage model implementation.
    
    This model calculates slippage costs that increase with order size.
    Large orders must walk through multiple price levels in the order book,
    causing progressively worse execution prices.
    """
    
    def __init__(self, size_coefficient: Decimal = Decimal('0.5')):
        """
        Initialize the size slippage model.
        
        Args:
            size_coefficient: Coefficient for size-based slippage (0.5 = square root relationship)
        """
        super().__init__("Order Size Slippage")
        self.size_coefficient = size_coefficient
        self.logger.info(f"Initialized Size Slippage Model with coefficient={size_coefficient}")
    
    def _calculate_slippage_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        execution_delay: Optional[timedelta]
    ) -> Decimal:
        """
        Calculate slippage due to order size effects.
        
        The model estimates slippage from walking the order book:
        Slippage = bid_ask_spread × (order_size / typical_size) ^ size_coefficient
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Not used in this model
            
        Returns:
            Slippage cost in absolute currency terms
        """
        # Get bid-ask spread as base slippage
        spread = self._get_bid_ask_spread(market_conditions)
        
        # Calculate size factor based on order size vs typical market size
        size_factor = self._calculate_size_factor(request, market_conditions)
        
        # Apply size coefficient (0.5 = square root, 1.0 = linear, etc.)
        size_multiplier = size_factor ** self.size_coefficient
        
        # Base slippage is half the spread (assuming mid-market execution)
        base_slippage = spread / Decimal('2')
        
        # Apply size multiplier
        size_adjusted_slippage = base_slippage * size_multiplier
        
        # Calculate total cost
        slippage_cost = size_adjusted_slippage * Decimal(str(request.quantity))
        
        self.logger.debug(
            f"Size slippage calculation: "
            f"spread={spread:.4f}, "
            f"size_factor={size_factor:.3f}, "
            f"size_multiplier={size_multiplier:.3f}, "
            f"base_slippage={base_slippage:.4f}, "
            f"slippage_cost={slippage_cost:.2f}"
        )
        
        return slippage_cost
    
    def _calculate_size_factor(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions
    ) -> Decimal:
        """
        Calculate size factor based on order size relative to market.
        
        Args:
            request: Transaction request
            market_conditions: Market conditions
            
        Returns:
            Size factor (1.0 = typical size, >1.0 = larger than typical)
        """
        # Use average order size as baseline
        # If not available, estimate from bid/ask sizes or volume
        if market_conditions.bid_size and market_conditions.ask_size:
            typical_size = (market_conditions.bid_size + market_conditions.ask_size) / 2
        elif market_conditions.volume:
            # Estimate typical order size as 1% of daily volume
            typical_size = market_conditions.volume // 100
        else:
            # Default assumption: 1000 shares typical
            typical_size = 1000
        
        # Ensure typical size is reasonable
        typical_size = max(typical_size, 100)  # Minimum 100 shares
        
        # Calculate size factor
        size_factor = Decimal(str(request.quantity)) / Decimal(str(typical_size))
        
        # Cap the size factor to prevent extreme values
        return min(size_factor, Decimal('10.0'))  # Maximum 10x typical size
    
    def _requires_market_data(self) -> bool:
        """
        This model requires spread and size data.
        
        Returns:
            True as market data is needed for calculation
        """
        return True
    
    def get_model_description(self) -> str:
        """
        Get detailed model description.
        
        Returns:
            Model description string
        """
        return (
            f"Order Size Slippage Model (coefficient={self.size_coefficient}): "
            f"Models order book impact from large orders. "
            f"Slippage = spread × (order_size / typical_size) ^ coefficient."
        )