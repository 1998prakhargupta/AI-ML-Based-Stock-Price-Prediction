"""
Cache Invalidator
================

Intelligent cache invalidation based on market events, data updates,
and configuration changes.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)


class InvalidationTrigger(Enum):
    """Types of cache invalidation triggers."""
    MARKET_DATA_UPDATE = auto()
    BROKER_CONFIG_CHANGE = auto()
    REGULATORY_CHANGE = auto()
    MANUAL_INVALIDATION = auto()
    TIME_BASED = auto()
    SYMBOL_EVENT = auto()
    VOLUME_THRESHOLD = auto()
    PRICE_MOVEMENT = auto()


@dataclass
class InvalidationEvent:
    """Event that triggers cache invalidation."""
    trigger: InvalidationTrigger
    symbol: Optional[str] = None
    broker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scope: str = "symbol"  # symbol, broker, global
    
    def matches_entry(self, entry_key: str, entry_metadata: Dict[str, Any]) -> bool:
        """Check if this event should invalidate a cache entry."""
        if self.scope == "global":
            return True
        
        if self.scope == "broker" and self.broker_name:
            return entry_metadata.get('broker_name') == self.broker_name
        
        if self.scope == "symbol" and self.symbol:
            return entry_metadata.get('symbol') == self.symbol
        
        return False


class InvalidationRule:
    """Rule for automatic cache invalidation."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: Callable[[Dict[str, Any]], List[str]],
        enabled: bool = True
    ):
        self.name = name
        self.condition = condition
        self.action = action
        self.enabled = enabled
        self.triggered_count = 0
        self.last_triggered = None
    
    def evaluate(self, context: Dict[str, Any]) -> List[str]:
        """Evaluate rule and return list of keys to invalidate."""
        if not self.enabled:
            return []
        
        try:
            if self.condition(context):
                self.triggered_count += 1
                self.last_triggered = datetime.now()
                return self.action(context)
        except Exception as e:
            logger.warning(f"Invalidation rule '{self.name}' evaluation failed: {e}")
        
        return []


class CacheInvalidator:
    """
    Intelligent cache invalidation system.
    
    Features:
    - Market event-based invalidation
    - Configuration change detection
    - Time-based invalidation policies
    - Custom invalidation rules
    - Batch invalidation for performance
    """
    
    def __init__(
        self,
        cache_manager,
        auto_invalidation: bool = True,
        batch_size: int = 100,
        check_interval: float = 30.0
    ):
        """
        Initialize cache invalidator.
        
        Args:
            cache_manager: Cache manager instance to invalidate
            auto_invalidation: Enable automatic invalidation
            batch_size: Number of entries to process in batch
            check_interval: Interval for checking invalidation rules
        """
        self.cache_manager = cache_manager
        self.auto_invalidation = auto_invalidation
        self.batch_size = batch_size
        self.check_interval = check_interval
        
        # Invalidation rules
        self.rules: Dict[str, InvalidationRule] = {}
        self._setup_default_rules()
        
        # Event tracking
        self.recent_events: List[InvalidationEvent] = []
        self.event_history_limit = 1000
        
        # Market data tracking
        self.last_market_data: Dict[str, datetime] = {}
        self.price_changes: Dict[str, float] = {}
        self.volume_tracking: Dict[str, int] = {}
        
        # Background processing
        self._running = True
        self._invalidation_thread = threading.Thread(target=self._invalidation_worker, daemon=True)
        self._invalidation_thread.start()
        
        # Statistics
        self.stats = {
            'total_invalidations': 0,
            'invalidations_by_trigger': defaultdict(int),
            'invalidations_by_rule': defaultdict(int),
            'last_invalidation': None
        }
        
        logger.info("Cache invalidator initialized")
    
    def _setup_default_rules(self):
        """Setup default invalidation rules."""
        
        # Rule 1: Invalidate old entries
        self.add_rule(
            "time_based_invalidation",
            condition=lambda ctx: True,  # Always check
            action=self._invalidate_old_entries
        )
        
        # Rule 2: Invalidate on significant price movement
        self.add_rule(
            "price_movement_invalidation",
            condition=lambda ctx: ctx.get('price_change_percent', 0) > 5.0,
            action=lambda ctx: self._invalidate_by_symbol(ctx.get('symbol'))
        )
        
        # Rule 3: Invalidate on high volume
        self.add_rule(
            "volume_spike_invalidation",
            condition=lambda ctx: ctx.get('volume_ratio', 1.0) > 3.0,
            action=lambda ctx: self._invalidate_by_symbol(ctx.get('symbol'))
        )
        
        # Rule 4: Invalidate during market events
        self.add_rule(
            "market_event_invalidation",
            condition=lambda ctx: ctx.get('market_event', False),
            action=lambda ctx: self._invalidate_all()
        )
    
    def add_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: Callable[[Dict[str, Any]], List[str]],
        enabled: bool = True
    ):
        """Add custom invalidation rule."""
        rule = InvalidationRule(name, condition, action, enabled)
        self.rules[name] = rule
        logger.info(f"Added invalidation rule: {name}")
    
    def remove_rule(self, name: str):
        """Remove invalidation rule."""
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed invalidation rule: {name}")
    
    def enable_rule(self, name: str):
        """Enable invalidation rule."""
        if name in self.rules:
            self.rules[name].enabled = True
            logger.info(f"Enabled invalidation rule: {name}")
    
    def disable_rule(self, name: str):
        """Disable invalidation rule."""
        if name in self.rules:
            self.rules[name].enabled = False
            logger.info(f"Disabled invalidation rule: {name}")
    
    def trigger_invalidation(
        self,
        trigger: InvalidationTrigger,
        symbol: Optional[str] = None,
        broker_name: Optional[str] = None,
        scope: str = "symbol",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Manually trigger cache invalidation."""
        event = InvalidationEvent(
            trigger=trigger,
            symbol=symbol,
            broker_name=broker_name,
            scope=scope,
            metadata=metadata or {}
        )
        
        self._process_invalidation_event(event)
    
    def invalidate_symbol(self, symbol: str, reason: str = "manual"):
        """Invalidate all cache entries for a symbol."""
        self.trigger_invalidation(
            InvalidationTrigger.MANUAL_INVALIDATION,
            symbol=symbol,
            scope="symbol",
            metadata={'reason': reason}
        )
    
    def invalidate_broker(self, broker_name: str, reason: str = "manual"):
        """Invalidate all cache entries for a broker."""
        self.trigger_invalidation(
            InvalidationTrigger.MANUAL_INVALIDATION,
            broker_name=broker_name,
            scope="broker",
            metadata={'reason': reason}
        )
    
    def invalidate_all(self, reason: str = "manual"):
        """Invalidate all cache entries."""
        self.trigger_invalidation(
            InvalidationTrigger.MANUAL_INVALIDATION,
            scope="global",
            metadata={'reason': reason}
        )
    
    def update_market_data(
        self,
        symbol: str,
        current_price: float,
        volume: int,
        previous_price: Optional[float] = None,
        average_volume: Optional[int] = None
    ):
        """Update market data and check for invalidation triggers."""
        current_time = datetime.now()
        
        # Track price changes
        if previous_price and previous_price > 0:
            price_change_percent = abs((current_price - previous_price) / previous_price) * 100
            self.price_changes[symbol] = price_change_percent
            
            # Check for significant price movement
            if price_change_percent > 5.0:  # 5% threshold
                self.trigger_invalidation(
                    InvalidationTrigger.PRICE_MOVEMENT,
                    symbol=symbol,
                    metadata={
                        'price_change_percent': price_change_percent,
                        'current_price': current_price,
                        'previous_price': previous_price
                    }
                )
        
        # Track volume spikes
        if average_volume and average_volume > 0:
            volume_ratio = volume / average_volume
            self.volume_tracking[symbol] = volume_ratio
            
            if volume_ratio > 3.0:  # 3x average volume
                self.trigger_invalidation(
                    InvalidationTrigger.VOLUME_THRESHOLD,
                    symbol=symbol,
                    metadata={
                        'volume_ratio': volume_ratio,
                        'current_volume': volume,
                        'average_volume': average_volume
                    }
                )
        
        # Update last market data timestamp
        self.last_market_data[symbol] = current_time
    
    def notify_broker_config_change(self, broker_name: str, change_type: str):
        """Notify of broker configuration changes."""
        self.trigger_invalidation(
            InvalidationTrigger.BROKER_CONFIG_CHANGE,
            broker_name=broker_name,
            scope="broker",
            metadata={
                'change_type': change_type,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def notify_regulatory_change(self, change_description: str, affected_instruments: List[str] = None):
        """Notify of regulatory changes affecting costs."""
        scope = "global" if not affected_instruments else "symbol"
        
        if affected_instruments:
            for instrument in affected_instruments:
                self.trigger_invalidation(
                    InvalidationTrigger.REGULATORY_CHANGE,
                    symbol=instrument,
                    scope="symbol",
                    metadata={
                        'change_description': change_description,
                        'regulatory_change': True
                    }
                )
        else:
            self.trigger_invalidation(
                InvalidationTrigger.REGULATORY_CHANGE,
                scope="global",
                metadata={
                    'change_description': change_description,
                    'regulatory_change': True
                }
            )
    
    def _process_invalidation_event(self, event: InvalidationEvent):
        """Process an invalidation event."""
        try:
            invalidated_count = 0
            
            if event.scope == "global":
                self.cache_manager.clear()
                invalidated_count = self.cache_manager.l1_stats.size
            elif event.scope == "broker" and event.broker_name:
                self.cache_manager.invalidate(broker_name=event.broker_name)
                invalidated_count = 1  # Approximation
            elif event.scope == "symbol" and event.symbol:
                self.cache_manager.invalidate(symbol=event.symbol)
                invalidated_count = 1  # Approximation
            
            # Update statistics
            self.stats['total_invalidations'] += invalidated_count
            self.stats['invalidations_by_trigger'][event.trigger.name] += invalidated_count
            self.stats['last_invalidation'] = datetime.now()
            
            # Add to event history
            self.recent_events.append(event)
            if len(self.recent_events) > self.event_history_limit:
                self.recent_events.pop(0)
            
            logger.info(f"Cache invalidation: {event.trigger.name} - {invalidated_count} entries")
            
        except Exception as e:
            logger.error(f"Failed to process invalidation event: {e}")
    
    def _invalidation_worker(self):
        """Background worker for automatic invalidation."""
        while self._running:
            try:
                if self.auto_invalidation:
                    self._check_invalidation_rules()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Invalidation worker error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_invalidation_rules(self):
        """Check all invalidation rules."""
        context = {
            'current_time': datetime.now(),
            'cache_size': len(self.cache_manager.l1_cache),
            'price_changes': dict(self.price_changes),
            'volume_tracking': dict(self.volume_tracking),
            'last_market_data': dict(self.last_market_data)
        }
        
        for rule_name, rule in self.rules.items():
            try:
                keys_to_invalidate = rule.evaluate(context)
                if keys_to_invalidate:
                    self.stats['invalidations_by_rule'][rule_name] += len(keys_to_invalidate)
                    logger.debug(f"Rule '{rule_name}' triggered invalidation of {len(keys_to_invalidate)} entries")
            except Exception as e:
                logger.warning(f"Rule '{rule_name}' evaluation failed: {e}")
    
    def _invalidate_old_entries(self, context: Dict[str, Any]) -> List[str]:
        """Invalidate entries older than their TTL."""
        # This is handled by the cache manager's cleanup worker
        # Return empty list as this rule is for triggering cleanup
        return []
    
    def _invalidate_by_symbol(self, symbol: Optional[str]) -> List[str]:
        """Get cache keys to invalidate for a symbol."""
        if not symbol:
            return []
        
        # This would need access to cache keys to return specific ones
        # For now, trigger symbol-based invalidation
        self.cache_manager.invalidate(symbol=symbol)
        return []
    
    def _invalidate_all(self) -> List[str]:
        """Invalidate all cache entries."""
        self.cache_manager.clear()
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        return {
            'stats': dict(self.stats),
            'rules': {
                name: {
                    'enabled': rule.enabled,
                    'triggered_count': rule.triggered_count,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for name, rule in self.rules.items()
            },
            'recent_events': [
                {
                    'trigger': event.trigger.name,
                    'symbol': event.symbol,
                    'broker_name': event.broker_name,
                    'scope': event.scope,
                    'timestamp': event.timestamp.isoformat(),
                    'metadata': event.metadata
                }
                for event in self.recent_events[-10:]  # Last 10 events
            ]
        }
    
    def get_rule_performance(self) -> Dict[str, Any]:
        """Get performance metrics for invalidation rules."""
        performance = {}
        
        for rule_name, rule in self.rules.items():
            total_invalidations = self.stats['invalidations_by_rule'][rule_name]
            
            performance[rule_name] = {
                'enabled': rule.enabled,
                'triggered_count': rule.triggered_count,
                'total_invalidations': total_invalidations,
                'avg_invalidations_per_trigger': (
                    total_invalidations / rule.triggered_count 
                    if rule.triggered_count > 0 else 0
                ),
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            }
        
        return performance
    
    def optimize_rules(self):
        """Optimize invalidation rules based on performance."""
        performance = self.get_rule_performance()
        
        for rule_name, stats in performance.items():
            # Disable rules that trigger too frequently with low impact
            if stats['triggered_count'] > 100 and stats['avg_invalidations_per_trigger'] < 1:
                logger.info(f"Disabling overly aggressive rule: {rule_name}")
                self.disable_rule(rule_name)
            
            # Enable rules that seem effective
            elif stats['avg_invalidations_per_trigger'] > 10 and not stats['enabled']:
                logger.info(f"Enabling effective rule: {rule_name}")
                self.enable_rule(rule_name)
    
    def set_symbol_sensitivity(self, symbol: str, price_threshold: float, volume_threshold: float):
        """Set symbol-specific invalidation sensitivity."""
        def condition(ctx):
            symbol_price_change = ctx.get('price_changes', {}).get(symbol, 0)
            symbol_volume_ratio = ctx.get('volume_tracking', {}).get(symbol, 1.0)
            
            return (symbol_price_change > price_threshold or 
                   symbol_volume_ratio > volume_threshold)
        
        def action(ctx):
            return self._invalidate_by_symbol(symbol)
        
        rule_name = f"symbol_sensitivity_{symbol}"
        self.add_rule(rule_name, condition, action)
        
        logger.info(f"Set sensitivity for {symbol}: price={price_threshold}%, volume={volume_threshold}x")
    
    def shutdown(self):
        """Shutdown invalidator and cleanup."""
        self._running = False
        
        if hasattr(self, '_invalidation_thread'):
            self._invalidation_thread.join(timeout=5.0)
        
        logger.info("Cache invalidator shutdown completed")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except:
            pass


logger.info("Cache invalidator module loaded successfully")