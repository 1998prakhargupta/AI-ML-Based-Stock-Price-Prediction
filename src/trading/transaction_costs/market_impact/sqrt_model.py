"""
Square Root Market Impact Model
==============================

Implements square root relationship between order size and market impact.
More realistic model that accounts for diminishing marginal impact.

Formula: Impact = α × √(Order Size / Average Daily Volume) × Volatility
"""

from decimal import Decimal
from typing import Optional
import logging

from .base_impact_model import BaseImpactModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class SquareRootImpactModel(BaseImpactModel):
    """
    Square root market impact model implementation.
    
    This model assumes market impact grows with the square root of order size,
    which is more realistic as it accounts for diminishing marginal impact.
    Widely used in academic literature and institutional trading.
    """
    
    def __init__(self, alpha: Decimal = Decimal('0.3')):
        """
        Initialize the square root impact model.
        
        Args:
            alpha: Impact coefficient (typical range: 0.2 to 0.5)
                  Higher values indicate more impact per unit volume
        """
        super().__init__("Square Root Impact Model", alpha)
        self.logger.info(f"Initialized Square Root Impact Model with alpha={alpha}")
    
    def _calculate_impact_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> Decimal:
        """
        Calculate square root market impact.
        
        The square root model uses the formula:
        Impact = α × √(Order Size / Average Daily Volume) × Volatility × Price
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            
        Returns:
            Market impact cost in absolute currency terms
        """
        # Get participation rate (order size / daily volume)
        participation_rate = self._get_participation_rate(request, market_conditions)
        
        # Get volatility measure
        volatility = self._get_volatility(market_conditions)
        
        # Calculate square root of participation rate
        sqrt_participation = participation_rate.sqrt()
        
        # Calculate base impact as percentage of notional
        impact_percentage = self.alpha * sqrt_participation * volatility
        
        # Convert to absolute cost
        notional_value = Decimal(str(request.quantity)) * request.price
        impact_cost = impact_percentage * notional_value
        
        self.logger.debug(
            f"Square root impact calculation: "
            f"participation_rate={participation_rate:.4f}, "
            f"sqrt_participation={sqrt_participation:.4f}, "
            f"volatility={volatility:.4f}, "
            f"impact_percentage={impact_percentage:.4f}, "
            f"impact_cost={impact_cost:.2f}"
        )
        
        return impact_cost
    
    def get_model_description(self) -> str:
        """
        Get detailed model description.
        
        Returns:
            Model description string
        """
        return (
            f"Square Root Impact Model (α={self.alpha}): "
            f"Impact = α × √(Order Size / ADV) × Volatility × Notional. "
            f"More realistic model with diminishing marginal impact for large orders."
        )