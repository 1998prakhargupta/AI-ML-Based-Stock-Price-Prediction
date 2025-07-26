"""
Dynamic Spread Adjuster
=======================

Real-time spread adjustment system that dynamically adapts spread estimates
based on changing market conditions, volatility, and liquidity factors.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from collections import deque
import asyncio
import threading
import time

from .base_spread_model import (
    BaseSpreadModel, SpreadEstimate, SpreadData, 
    MarketCondition
)

logger = logging.getLogger(__name__)


@dataclass
class AdjustmentRule:
    """Rule for dynamic spread adjustment."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    adjustment_factor: float
    priority: int  # Higher priority rules applied first
    description: str


@dataclass
class AdjustmentEvent:
    """Event that triggered a spread adjustment."""
    timestamp: datetime
    symbol: str
    rule_name: str
    original_spread: Decimal
    adjusted_spread: Decimal
    adjustment_factor: float
    trigger_data: Dict[str, Any]


@dataclass
class DynamicParameters:
    """Dynamic parameters for spread adjustment."""
    volatility_threshold: float = 0.02
    volume_threshold_ratio: float = 0.5
    liquidity_threshold: float = 0.3
    stress_threshold: float = 0.7
    max_adjustment_factor: float = 3.0
    min_adjustment_factor: float = 0.1


class DynamicSpreadAdjuster(BaseSpreadModel):
    """
    Dynamic spread adjuster with real-time adaptability.
    
    Features:
    - Real-time spread adjustments
    - Rule-based adjustment system
    - Market condition adaptation
    - Volatility-based scaling
    - Liquidity-aware adjustments
    """
    
    def __init__(
        self,
        adjustment_interval: float = 1.0,  # seconds
        max_adjustment_history: int = 1000,
        enable_auto_tuning: bool = True,
        **kwargs
    ):
        """
        Initialize dynamic spread adjuster.
        
        Args:
            adjustment_interval: How often to check for adjustments
            max_adjustment_history: Maximum adjustment events to store
            enable_auto_tuning: Whether to auto-tune parameters
        """
        super().__init__(
            model_name="DynamicSpreadAdjuster",
            version="1.0.0",
            **kwargs
        )
        
        self.adjustment_interval = adjustment_interval
        self.max_adjustment_history = max_adjustment_history
        self.enable_auto_tuning = enable_auto_tuning
        
        # Dynamic parameters
        self.parameters = DynamicParameters()
        
        # Adjustment rules
        self._adjustment_rules: List[AdjustmentRule] = []
        self._initialize_default_rules()
        
        # State tracking
        self._adjustment_history: deque = deque(maxlen=max_adjustment_history)
        self._market_state: Dict[str, Dict[str, Any]] = {}
        self._active_adjustments: Dict[str, float] = {}  # symbol -> current adjustment factor
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._data_callbacks: List[Callable] = []
        
        # Performance tracking
        self._adjustment_count = 0
        self._last_adjustment_time = None
        
        logger.info("Dynamic spread adjuster initialized")
    
    def start_dynamic_adjustment(self, symbols: List[str]) -> None:
        """
        Start dynamic adjustment monitoring for symbols.
        
        Args:
            symbols: List of symbols to monitor
        """
        if self._monitoring_active:
            logger.warning("Dynamic adjustment already running")
            return
        
        self._monitoring_active = True
        
        # Initialize market state for symbols
        for symbol in symbols:
            self._market_state[symbol] = {
                'last_update': datetime.now(),
                'volatility': 0.01,
                'volume_ratio': 1.0,
                'liquidity_score': 0.8,
                'market_condition': MarketCondition.NORMAL,
                'base_spread': Decimal('0.01')
            }
            self._active_adjustments[symbol] = 1.0  # No adjustment initially
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SpreadAdjusterMonitor"
        )
        self._monitor_thread.start()
        
        logger.info(f"Started dynamic adjustment monitoring for {len(symbols)} symbols")
    
    def stop_dynamic_adjustment(self) -> None:
        """Stop dynamic adjustment monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped dynamic adjustment monitoring")
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """
        Update market data for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: New market data
        """
        if symbol not in self._market_state:
            self._market_state[symbol] = {}
        
        # Update market state
        self._market_state[symbol].update({
            'last_update': datetime.now(),
            'volatility': market_data.get('volatility', 0.01),
            'volume_ratio': market_data.get('volume_ratio', 1.0),
            'liquidity_score': market_data.get('liquidity_score', 0.8),
            'market_condition': market_data.get('market_condition', MarketCondition.NORMAL),
            'price': market_data.get('price', 100.0)
        })
        
        # Trigger immediate adjustment check
        if self._monitoring_active:
            self._check_and_apply_adjustments(symbol)
    
    def get_adjusted_spread(
        self,
        symbol: str,
        base_spread: Decimal,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[Decimal, List[str]]:
        """
        Get dynamically adjusted spread for a symbol.
        
        Args:
            symbol: Trading symbol
            base_spread: Base spread estimate
            market_data: Current market data
            
        Returns:
            Tuple of (adjusted_spread, applied_rules)
        """
        # Update market data if provided
        if market_data:
            self.update_market_data(symbol, market_data)
        
        # Get current adjustment factor
        adjustment_factor = self._active_adjustments.get(symbol, 1.0)
        
        # Apply adjustment
        adjusted_spread = base_spread * Decimal(str(adjustment_factor))
        
        # Get applied rules from recent history
        applied_rules = self._get_recent_applied_rules(symbol)
        
        return adjusted_spread, applied_rules
    
    def estimate_spread(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[SpreadData]] = None
    ) -> SpreadEstimate:
        """
        Estimate spread with dynamic adjustments.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Historical spread data
            
        Returns:
            Dynamically adjusted spread estimate
        """
        try:
            # Calculate base spread (simple approach for demo)
            base_spread = self._calculate_base_spread(symbol, market_data, historical_data)
            
            # Apply dynamic adjustments
            adjusted_spread, applied_rules = self.get_adjusted_spread(symbol, base_spread, market_data)
            
            # Calculate confidence based on adjustment stability
            confidence = self._calculate_adjustment_confidence(symbol)
            
            # Determine market condition
            market_condition = self._get_market_condition(symbol)
            
            # Convert to basis points
            mid_price = Decimal(str(market_data.get('price', 100))) if market_data else Decimal('100')
            spread_bps = (adjusted_spread / mid_price) * Decimal('10000')
            
            estimate = SpreadEstimate(
                symbol=symbol,
                estimated_spread=adjusted_spread,
                spread_bps=spread_bps,
                confidence_level=confidence,
                estimation_method="dynamic_adjustment",
                timestamp=datetime.now(),
                market_condition=market_condition,
                supporting_data={
                    'base_spread': float(base_spread),
                    'adjustment_factor': self._active_adjustments.get(symbol, 1.0),
                    'applied_rules': applied_rules,
                    'market_state': self._market_state.get(symbol, {}),
                    'adjustment_count': self._adjustment_count
                }
            )
            
            return estimate
            
        except Exception as e:
            logger.error(f"Failed to estimate adjusted spread for {symbol}: {e}")
            raise
    
    def add_adjustment_rule(self, rule: AdjustmentRule) -> None:
        """
        Add a custom adjustment rule.
        
        Args:
            rule: Adjustment rule to add
        """
        self._adjustment_rules.append(rule)
        # Sort by priority (higher priority first)
        self._adjustment_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added adjustment rule: {rule.name}")
    
    def remove_adjustment_rule(self, rule_name: str) -> bool:
        """
        Remove an adjustment rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed
        """
        original_count = len(self._adjustment_rules)
        self._adjustment_rules = [r for r in self._adjustment_rules if r.name != rule_name]
        
        removed = len(self._adjustment_rules) < original_count
        if removed:
            logger.info(f"Removed adjustment rule: {rule_name}")
        
        return removed
    
    def get_adjustment_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about spread adjustments.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Adjustment statistics
        """
        # Filter events by symbol if specified
        events = self._adjustment_history
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        
        if not events:
            return {}
        
        # Calculate statistics
        adjustment_factors = [e.adjustment_factor for e in events]
        rule_counts = {}
        
        for event in events:
            rule_counts[event.rule_name] = rule_counts.get(event.rule_name, 0) + 1
        
        return {
            'total_adjustments': len(events),
            'average_adjustment_factor': sum(adjustment_factors) / len(adjustment_factors),
            'max_adjustment_factor': max(adjustment_factors),
            'min_adjustment_factor': min(adjustment_factors),
            'most_common_rule': max(rule_counts.items(), key=lambda x: x[1])[0] if rule_counts else None,
            'rule_usage': rule_counts,
            'last_adjustment': events[-1].timestamp if events else None
        }
    
    def validate_spread_data(self, spread_data: SpreadData) -> bool:
        """Validate spread data for dynamic adjustment."""
        return spread_data is not None and spread_data.symbol is not None
    
    def get_supported_symbols(self) -> List[str]:
        """Get symbols being dynamically adjusted."""
        return list(self._market_state.keys())
    
    # Private methods
    
    def _initialize_default_rules(self) -> None:
        """Initialize default adjustment rules."""
        
        # High volatility rule
        self.add_adjustment_rule(AdjustmentRule(
            name="high_volatility",
            condition=lambda data: data.get('volatility', 0) > self.parameters.volatility_threshold,
            adjustment_factor=1.5,
            priority=10,
            description="Increase spread during high volatility"
        ))
        
        # Low liquidity rule
        self.add_adjustment_rule(AdjustmentRule(
            name="low_liquidity",
            condition=lambda data: data.get('liquidity_score', 1.0) < self.parameters.liquidity_threshold,
            adjustment_factor=2.0,
            priority=9,
            description="Increase spread during low liquidity"
        ))
        
        # Low volume rule
        self.add_adjustment_rule(AdjustmentRule(
            name="low_volume",
            condition=lambda data: data.get('volume_ratio', 1.0) < self.parameters.volume_threshold_ratio,
            adjustment_factor=1.3,
            priority=8,
            description="Increase spread during low volume"
        ))
        
        # Market stress rule
        self.add_adjustment_rule(AdjustmentRule(
            name="market_stress",
            condition=lambda data: data.get('market_condition') == MarketCondition.STRESSED,
            adjustment_factor=2.5,
            priority=11,
            description="Increase spread during market stress"
        ))
        
        # Market open/close rule
        self.add_adjustment_rule(AdjustmentRule(
            name="market_hours",
            condition=self._is_market_boundary_time,
            adjustment_factor=1.4,
            priority=7,
            description="Increase spread near market open/close"
        ))
        
        # High efficiency rule (reduce spread in good conditions)
        self.add_adjustment_rule(AdjustmentRule(
            name="high_efficiency",
            condition=lambda data: (
                data.get('volatility', 1.0) < 0.01 and
                data.get('liquidity_score', 0.0) > 0.8 and
                data.get('volume_ratio', 0.0) > 1.5
            ),
            adjustment_factor=0.8,
            priority=5,
            description="Decrease spread in highly efficient markets"
        ))
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for dynamic adjustments."""
        logger.info("Started dynamic adjustment monitoring loop")
        
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Check all monitored symbols
                for symbol in list(self._market_state.keys()):
                    self._check_and_apply_adjustments(symbol)
                
                # Auto-tune parameters if enabled
                if self.enable_auto_tuning:
                    self._auto_tune_parameters()
                
                # Sleep until next check
                elapsed = time.time() - start_time
                sleep_time = max(0, self.adjustment_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in adjustment monitoring loop: {e}")
                time.sleep(self.adjustment_interval)
    
    def _check_and_apply_adjustments(self, symbol: str) -> None:
        """Check and apply adjustments for a symbol."""
        try:
            market_data = self._market_state.get(symbol, {})
            
            # Check data freshness
            last_update = market_data.get('last_update')
            if last_update and (datetime.now() - last_update).total_seconds() > 60:
                # Stale data, skip adjustment
                return
            
            # Find applicable rules
            applicable_rules = []
            for rule in self._adjustment_rules:
                try:
                    if rule.condition(market_data):
                        applicable_rules.append(rule)
                except Exception as e:
                    logger.warning(f"Error evaluating rule {rule.name}: {e}")
            
            # Calculate combined adjustment factor
            if applicable_rules:
                # Use highest priority rule, but combine factors for multiple rules
                primary_rule = applicable_rules[0]  # Highest priority
                adjustment_factor = primary_rule.adjustment_factor
                
                # Apply additional rules with diminishing effect
                for rule in applicable_rules[1:]:
                    additional_factor = (rule.adjustment_factor - 1.0) * 0.5  # 50% of additional effect
                    adjustment_factor += additional_factor
                
                # Clamp adjustment factor
                adjustment_factor = max(
                    self.parameters.min_adjustment_factor,
                    min(self.parameters.max_adjustment_factor, adjustment_factor)
                )
                
                # Apply adjustment if it's different from current
                current_factor = self._active_adjustments.get(symbol, 1.0)
                if abs(adjustment_factor - current_factor) > 0.05:  # 5% threshold
                    self._apply_adjustment(symbol, adjustment_factor, applicable_rules)
                    
            else:
                # No rules apply, move towards neutral (1.0)
                current_factor = self._active_adjustments.get(symbol, 1.0)
                if abs(current_factor - 1.0) > 0.05:
                    # Gradual return to neutral
                    new_factor = current_factor * 0.9 + 1.0 * 0.1
                    self._apply_adjustment(symbol, new_factor, [])
                    
        except Exception as e:
            logger.error(f"Error checking adjustments for {symbol}: {e}")
    
    def _apply_adjustment(self, symbol: str, adjustment_factor: float, rules: List[AdjustmentRule]) -> None:
        """Apply adjustment to a symbol."""
        old_factor = self._active_adjustments.get(symbol, 1.0)
        self._active_adjustments[symbol] = adjustment_factor
        
        # Create adjustment event
        base_spread = self._market_state.get(symbol, {}).get('base_spread', Decimal('0.01'))
        event = AdjustmentEvent(
            timestamp=datetime.now(),
            symbol=symbol,
            rule_name=rules[0].name if rules else "neutral_return",
            original_spread=base_spread,
            adjusted_spread=base_spread * Decimal(str(adjustment_factor)),
            adjustment_factor=adjustment_factor,
            trigger_data=self._market_state.get(symbol, {}).copy()
        )
        
        self._adjustment_history.append(event)
        self._adjustment_count += 1
        self._last_adjustment_time = datetime.now()
        
        logger.debug(f"Applied adjustment to {symbol}: {old_factor:.3f} -> {adjustment_factor:.3f}")
    
    def _calculate_base_spread(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]],
        historical_data: Optional[List[SpreadData]]
    ) -> Decimal:
        """Calculate base spread before adjustments."""
        # Simple base spread calculation
        if market_data and 'bid' in market_data and 'ask' in market_data:
            bid = Decimal(str(market_data['bid']))
            ask = Decimal(str(market_data['ask']))
            return ask - bid
        
        # Fallback to historical average
        if historical_data:
            spreads = [d.spread for d in historical_data if d.spread]
            if spreads:
                return sum(spreads) / len(spreads)
        
        # Default fallback
        return Decimal('0.01')
    
    def _calculate_adjustment_confidence(self, symbol: str) -> float:
        """Calculate confidence in adjustment based on stability."""
        # Look at recent adjustment history
        recent_events = [
            e for e in self._adjustment_history
            if e.symbol == symbol and (datetime.now() - e.timestamp).total_seconds() < 300
        ]
        
        if len(recent_events) < 2:
            return 0.8  # Default confidence
        
        # Check adjustment stability
        factors = [e.adjustment_factor for e in recent_events]
        factor_std = (sum((f - sum(factors)/len(factors))**2 for f in factors) / len(factors)) ** 0.5
        
        # Lower standard deviation = higher confidence
        stability_score = max(0.5, 1.0 - factor_std)
        
        return stability_score
    
    def _get_market_condition(self, symbol: str) -> MarketCondition:
        """Get current market condition for symbol."""
        market_data = self._market_state.get(symbol, {})
        return market_data.get('market_condition', MarketCondition.NORMAL)
    
    def _get_recent_applied_rules(self, symbol: str, minutes: int = 5) -> List[str]:
        """Get rules applied to symbol in recent time."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_events = [
            e for e in self._adjustment_history
            if e.symbol == symbol and e.timestamp >= cutoff_time
        ]
        
        return list(set(e.rule_name for e in recent_events))
    
    def _is_market_boundary_time(self, data: Dict[str, Any]) -> bool:
        """Check if current time is near market open/close."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Market opens at 9:30 AM, closes at 4:00 PM (approximate)
        market_open = 9 * 60 + 30  # 9:30 AM in minutes
        market_close = 16 * 60     # 4:00 PM in minutes
        current_time = hour * 60 + minute
        
        # Within 30 minutes of open or close
        near_open = abs(current_time - market_open) <= 30
        near_close = abs(current_time - market_close) <= 30
        
        return near_open or near_close
    
    def _auto_tune_parameters(self) -> None:
        """Auto-tune adjustment parameters based on performance."""
        if len(self._adjustment_history) < 100:
            return  # Need sufficient data
        
        # Simple auto-tuning: adjust thresholds based on adjustment frequency
        recent_events = [
            e for e in self._adjustment_history
            if (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if not recent_events:
            return
        
        # If adjusting too frequently, tighten thresholds
        adjustment_rate = len(recent_events) / 60.0  # Adjustments per minute
        
        if adjustment_rate > 1.0:  # More than 1 adjustment per minute
            # Increase thresholds to reduce sensitivity
            self.parameters.volatility_threshold *= 1.1
            self.parameters.volume_threshold_ratio *= 0.9
            logger.debug("Auto-tuning: Reduced adjustment sensitivity")
        
        elif adjustment_rate < 0.1:  # Less than 1 adjustment per 10 minutes
            # Decrease thresholds to increase sensitivity
            self.parameters.volatility_threshold *= 0.95
            self.parameters.volume_threshold_ratio *= 1.05
            logger.debug("Auto-tuning: Increased adjustment sensitivity")


logger.info("Dynamic spread adjuster class loaded successfully")