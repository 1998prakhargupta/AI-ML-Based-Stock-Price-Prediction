"""
Market Condition Slippage Model
==============================

Calculates slippage costs based on current market conditions.
Accounts for volatility, liquidity, and market stress indicators.
"""

from decimal import Decimal
from datetime import timedelta
from typing import Optional
import logging

from .base_slippage_model import BaseSlippageModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class ConditionSlippageModel(BaseSlippageModel):
    """
    Market condition slippage model implementation.
    
    This model adjusts slippage based on current market conditions:
    - High volatility increases slippage
    - Low liquidity increases slippage
    - Market stress indicators affect slippage
    """
    
    def __init__(
        self,
        base_slippage_bps: Decimal = Decimal('5.0'),
        volatility_sensitivity: Decimal = Decimal('2.0'),
        liquidity_sensitivity: Decimal = Decimal('1.5')
    ):
        """
        Initialize the market condition slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_sensitivity: Sensitivity to volatility changes
            liquidity_sensitivity: Sensitivity to liquidity changes
        """
        super().__init__("Market Condition Slippage")
        self.base_slippage_bps = base_slippage_bps
        self.volatility_sensitivity = volatility_sensitivity
        self.liquidity_sensitivity = liquidity_sensitivity
        
        self.logger.info(
            f"Initialized Condition Slippage Model: "
            f"base={base_slippage_bps}bps, "
            f"vol_sens={volatility_sensitivity}, "
            f"liq_sens={liquidity_sensitivity}"
        )
    
    def _calculate_slippage_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        execution_delay: Optional[timedelta]
    ) -> Decimal:
        """
        Calculate slippage based on market conditions.
        
        The model combines multiple market condition factors:
        Slippage = base_slippage × volatility_factor × liquidity_factor × order_type_factor
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Execution delay (affects volatility impact)
            
        Returns:
            Slippage cost in absolute currency terms
        """
        # Start with base slippage
        base_slippage = self.base_slippage_bps / Decimal('10000')  # Convert bps to decimal
        
        # Calculate condition-based adjustments
        volatility_factor = self._calculate_volatility_factor(market_conditions)
        liquidity_factor = self._calculate_liquidity_factor(market_conditions)
        order_type_factor = self._get_order_type_multiplier(request.order_type)
        timing_factor = self._calculate_timing_factor(request, execution_delay)
        
        # Combine all factors
        total_slippage_rate = (
            base_slippage * 
            volatility_factor * 
            liquidity_factor * 
            order_type_factor * 
            timing_factor
        )
        
        # Apply to notional value
        notional_value = Decimal(str(request.quantity)) * request.price
        slippage_cost = total_slippage_rate * notional_value
        
        self.logger.debug(
            f"Condition slippage calculation: "
            f"base_slippage={base_slippage:.4f}, "
            f"vol_factor={volatility_factor:.2f}, "
            f"liq_factor={liquidity_factor:.2f}, "
            f"order_factor={order_type_factor:.2f}, "
            f"timing_factor={timing_factor:.2f}, "
            f"total_rate={total_slippage_rate:.4f}, "
            f"slippage_cost={slippage_cost:.2f}"
        )
        
        return slippage_cost
    
    def _calculate_volatility_factor(self, market_conditions: MarketConditions) -> Decimal:
        """
        Calculate volatility adjustment factor.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Volatility factor (1.0 = normal, >1.0 = high volatility)
        """
        volatility = self._get_volatility(market_conditions)
        
        # Define normal volatility baseline (1.5% daily)
        normal_volatility = Decimal('0.015')
        
        # Calculate relative volatility
        relative_volatility = volatility / normal_volatility
        
        # Apply sensitivity factor
        volatility_factor = Decimal('1.0') + (relative_volatility - Decimal('1.0')) * self.volatility_sensitivity
        
        # Cap between reasonable bounds
        return max(Decimal('0.5'), min(volatility_factor, Decimal('5.0')))
    
    def _calculate_liquidity_factor(self, market_conditions: MarketConditions) -> Decimal:
        """
        Calculate liquidity adjustment factor.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Liquidity factor (1.0 = normal, >1.0 = low liquidity)
        """
        # Use bid-ask spread as primary liquidity indicator
        spread = self._get_bid_ask_spread(market_conditions)
        
        # Get price for normalization
        price = market_conditions.last_price or market_conditions.mid_price
        if not price:
            return Decimal('1.0')  # No adjustment if no price data
        
        # Calculate spread as percentage of price
        spread_pct = spread / price
        
        # Define normal spread baseline (0.1% for liquid stocks)
        normal_spread = Decimal('0.001')
        
        # Calculate relative spread
        relative_spread = spread_pct / normal_spread
        
        # Apply sensitivity factor
        liquidity_factor = Decimal('1.0') + (relative_spread - Decimal('1.0')) * self.liquidity_sensitivity
        
        # Cap between reasonable bounds
        return max(Decimal('0.8'), min(liquidity_factor, Decimal('3.0')))
    
    def _calculate_timing_factor(
        self,
        request: TransactionRequest,
        execution_delay: Optional[timedelta]
    ) -> Decimal:
        """
        Calculate timing-based adjustment factor.
        
        Args:
            request: Transaction request
            execution_delay: Execution delay
            
        Returns:
            Timing factor (1.0 = normal, >1.0 = stressed timing)
        """
        # Base factor
        factor = Decimal('1.0')
        
        # Market timing adjustment
        if request.market_timing.name in ['PRE_MARKET', 'AFTER_HOURS']:
            factor *= Decimal('1.3')  # Higher slippage outside market hours
        
        # Execution delay adjustment
        if execution_delay:
            delay_seconds = self._calculate_delay_seconds(execution_delay)
            # Increase factor for longer delays
            delay_factor = Decimal('1.0') + (delay_seconds / Decimal('60'))  # +1% per minute
            factor *= min(delay_factor, Decimal('2.0'))  # Cap at 2x
        
        return factor
    
    def _requires_market_data(self) -> bool:
        """
        This model requires comprehensive market data.
        
        Returns:
            True as market conditions are essential
        """
        return True
    
    def get_model_description(self) -> str:
        """
        Get detailed model description.
        
        Returns:
            Model description string
        """
        return (
            f"Market Condition Slippage Model "
            f"(base={self.base_slippage_bps}bps, vol_sens={self.volatility_sensitivity}): "
            f"Adapts slippage to volatility, liquidity, and timing conditions."
        )