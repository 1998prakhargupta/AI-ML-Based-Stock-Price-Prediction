"""
Linear Market Impact Model
=========================

Implements linear relationship between order size and market impact.
Simple model where impact is proportional to order size relative to daily volume.

Formula: Impact = α × (Order Size / Average Daily Volume) × Volatility
"""

from decimal import Decimal
from typing import Optional
import logging

from .base_impact_model import BaseImpactModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class LinearImpactModel(BaseImpactModel):
    """
    Linear market impact model implementation.
    
    This model assumes market impact grows linearly with order size.
    While simple, it tends to overestimate impact for very large orders
    and is best suited for small to medium-sized orders.
    """
    
    def __init__(self, alpha: Decimal = Decimal('0.1')):
        """
        Initialize the linear impact model.
        
        Args:
            alpha: Impact coefficient (typical range: 0.05 to 0.2)
                  Higher values indicate more impact per unit volume
        """
        super().__init__("Linear Impact Model", alpha)
        self.logger.info(f"Initialized Linear Impact Model with alpha={alpha}")
    
    def _calculate_impact_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> Decimal:
        """
        Calculate linear market impact.
        
        The linear model uses the formula:
        Impact = α × (Order Size / Average Daily Volume) × Volatility × Price
        
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
        
        # Calculate base impact as percentage of notional
        impact_percentage = self.alpha * participation_rate * volatility
        
        # Convert to absolute cost
        notional_value = Decimal(str(request.quantity)) * request.price
        impact_cost = impact_percentage * notional_value
        
        self.logger.debug(
            f"Linear impact calculation: "
            f"participation_rate={participation_rate:.4f}, "
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
            f"Linear Impact Model (α={self.alpha}): "
            f"Impact = α × (Order Size / ADV) × Volatility × Notional. "
            f"Simple linear relationship between order size and market impact."
        )