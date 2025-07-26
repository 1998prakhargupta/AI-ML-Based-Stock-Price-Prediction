"""
Adaptive Market Impact Model
===========================

Dynamic model that adapts based on real-time market conditions.
Incorporates volatility regimes, time-of-day effects, and liquidity adjustments.
"""

from decimal import Decimal
from datetime import datetime, time
from typing import Optional, Dict, Any
import logging

from .base_impact_model import BaseImpactModel
from ..models import TransactionRequest, MarketConditions, MarketTiming

logger = logging.getLogger(__name__)


class AdaptiveImpactModel(BaseImpactModel):
    """
    Adaptive market impact model implementation.
    
    This model dynamically adjusts impact calculations based on:
    - Real-time volatility conditions
    - Time-of-day effects
    - Market regime detection (trending vs. mean-reverting)
    - Liquidity conditions
    """
    
    def __init__(
        self,
        base_alpha: Decimal = Decimal('0.2'),
        volatility_adjustment: bool = True,
        time_adjustment: bool = True,
        liquidity_adjustment: bool = True
    ):
        """
        Initialize the adaptive impact model.
        
        Args:
            base_alpha: Base impact coefficient
            volatility_adjustment: Enable volatility regime adjustments
            time_adjustment: Enable time-of-day adjustments
            liquidity_adjustment: Enable liquidity-based adjustments
        """
        super().__init__("Adaptive Impact Model", base_alpha)
        self.volatility_adjustment = volatility_adjustment
        self.time_adjustment = time_adjustment
        self.liquidity_adjustment = liquidity_adjustment
        
        # Time-of-day impact multipliers
        self.time_multipliers = {
            'market_open': Decimal('1.5'),    # 9:15-10:00
            'morning': Decimal('1.2'),        # 10:00-11:30
            'midday': Decimal('1.0'),         # 11:30-14:00
            'afternoon': Decimal('1.1'),      # 14:00-15:00
            'market_close': Decimal('1.4'),   # 15:00-15:30
            'pre_market': Decimal('2.0'),     # Higher impact
            'after_hours': Decimal('1.8')     # Higher impact
        }
        
        self.logger.info(
            f"Initialized Adaptive Impact Model with base_alpha={base_alpha}, "
            f"volatility_adj={volatility_adjustment}, "
            f"time_adj={time_adjustment}, "
            f"liquidity_adj={liquidity_adjustment}"
        )
    
    def _calculate_impact_internal(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> Decimal:
        """
        Calculate adaptive market impact with dynamic adjustments.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            
        Returns:
            Market impact cost in absolute currency terms
        """
        # Start with base square root model
        participation_rate = self._get_participation_rate(request, market_conditions)
        sqrt_participation = participation_rate.sqrt()
        
        # Get base volatility
        volatility = self._get_volatility(market_conditions)
        
        # Calculate dynamic alpha with adjustments
        adjusted_alpha = self._calculate_dynamic_alpha(request, market_conditions)
        
        # Calculate base impact
        impact_percentage = adjusted_alpha * sqrt_participation * volatility
        
        # Convert to absolute cost
        notional_value = Decimal(str(request.quantity)) * request.price
        impact_cost = impact_percentage * notional_value
        
        self.logger.debug(
            f"Adaptive impact calculation: "
            f"base_alpha={self.alpha:.3f}, "
            f"adjusted_alpha={adjusted_alpha:.3f}, "
            f"participation_rate={participation_rate:.4f}, "
            f"volatility={volatility:.4f}, "
            f"impact_cost={impact_cost:.2f}"
        )
        
        return impact_cost
    
    def _calculate_dynamic_alpha(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions]
    ) -> Decimal:
        """
        Calculate dynamically adjusted alpha coefficient.
        
        Args:
            request: Transaction request
            market_conditions: Market conditions
            
        Returns:
            Adjusted alpha coefficient
        """
        alpha = self.alpha
        adjustments = []
        
        # Volatility regime adjustment
        if self.volatility_adjustment and market_conditions:
            vol_multiplier = self._get_volatility_multiplier(market_conditions)
            alpha *= vol_multiplier
            adjustments.append(f"vol_mult={vol_multiplier:.2f}")
        
        # Time-of-day adjustment
        if self.time_adjustment:
            time_multiplier = self._get_time_multiplier(request)
            alpha *= time_multiplier
            adjustments.append(f"time_mult={time_multiplier:.2f}")
        
        # Liquidity adjustment
        if self.liquidity_adjustment and market_conditions:
            liquidity_multiplier = self._get_liquidity_multiplier(market_conditions)
            alpha *= liquidity_multiplier
            adjustments.append(f"liq_mult={liquidity_multiplier:.2f}")
        
        if adjustments:
            self.logger.debug(f"Alpha adjustments: {', '.join(adjustments)}")
        
        return alpha
    
    def _get_volatility_multiplier(self, market_conditions: MarketConditions) -> Decimal:
        """
        Calculate volatility regime multiplier.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Volatility adjustment multiplier
        """
        volatility = self._get_volatility(market_conditions)
        
        # Define volatility regimes
        if volatility > Decimal('0.03'):  # High volatility (>3% daily)
            return Decimal('1.5')
        elif volatility > Decimal('0.02'):  # Medium volatility (2-3% daily)
            return Decimal('1.2')
        elif volatility > Decimal('0.01'):  # Normal volatility (1-2% daily)
            return Decimal('1.0')
        else:  # Low volatility (<1% daily)
            return Decimal('0.8')
    
    def _get_time_multiplier(self, request: TransactionRequest) -> Decimal:
        """
        Calculate time-of-day multiplier.
        
        Args:
            request: Transaction request
            
        Returns:
            Time adjustment multiplier
        """
        if request.market_timing == MarketTiming.PRE_MARKET:
            return self.time_multipliers['pre_market']
        elif request.market_timing == MarketTiming.AFTER_HOURS:
            return self.time_multipliers['after_hours']
        
        # For market hours, use timestamp
        trade_time = request.timestamp.time()
        
        if time(9, 15) <= trade_time < time(10, 0):
            return self.time_multipliers['market_open']
        elif time(10, 0) <= trade_time < time(11, 30):
            return self.time_multipliers['morning']
        elif time(11, 30) <= trade_time < time(14, 0):
            return self.time_multipliers['midday']
        elif time(14, 0) <= trade_time < time(15, 0):
            return self.time_multipliers['afternoon']
        elif time(15, 0) <= trade_time <= time(15, 30):
            return self.time_multipliers['market_close']
        else:
            return Decimal('1.0')  # Default
    
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
    
    def _get_liquidity_multiplier(self, market_conditions: MarketConditions) -> Decimal:
        """
        Calculate liquidity-based multiplier.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Liquidity adjustment multiplier
        """
        # Use bid-ask spread as liquidity proxy
        spread = self._get_bid_ask_spread(market_conditions)
        
        # Get mid price for normalization
        mid_price = market_conditions.mid_price or market_conditions.last_price
        if not mid_price:
            return Decimal('1.0')  # No adjustment if no price data
        
        # Calculate spread as percentage of price
        spread_pct = spread / mid_price
        
        # Adjust based on spread width
        if spread_pct > Decimal('0.01'):  # Wide spread (>1%)
            return Decimal('1.4')
        elif spread_pct > Decimal('0.005'):  # Medium spread (0.5-1%)
            return Decimal('1.2')
        elif spread_pct > Decimal('0.002'):  # Normal spread (0.2-0.5%)
            return Decimal('1.0')
        else:  # Tight spread (<0.2%)
            return Decimal('0.9')
    
    def get_model_description(self) -> str:
        """
        Get detailed model description.
        
        Returns:
            Model description string
        """
        adjustments = []
        if self.volatility_adjustment:
            adjustments.append("volatility regime")
        if self.time_adjustment:
            adjustments.append("time-of-day")
        if self.liquidity_adjustment:
            adjustments.append("liquidity conditions")
        
        adj_str = ", ".join(adjustments) if adjustments else "none"
        
        return (
            f"Adaptive Impact Model (base α={self.alpha}): "
            f"Dynamic square root model with {adj_str} adjustments. "
            f"Adapts to real-time market conditions."
        )
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters including adjustment settings.
        
        Returns:
            Dictionary of model parameters
        """
        params = super().get_model_parameters()
        params.update({
            'volatility_adjustment': self.volatility_adjustment,
            'time_adjustment': self.time_adjustment,
            'liquidity_adjustment': self.liquidity_adjustment,
            'time_multipliers': {k: float(v) for k, v in self.time_multipliers.items()}
        })
        return params