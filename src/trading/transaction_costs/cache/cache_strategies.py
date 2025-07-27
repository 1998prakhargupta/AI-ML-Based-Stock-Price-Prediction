"""
Cache Strategies
===============

Different caching strategies for optimal performance based on usage patterns.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import math
from dataclasses import dataclass
from collections import OrderedDict
from ..models import MarketConditions, TransactionRequest

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def should_cache(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> bool:
        """Determine if result should be cached."""
        pass
    
    @abstractmethod
    def get_ttl(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Get TTL for cache entry."""
        pass
    
    @abstractmethod
    def should_evict(self, entry_info: Dict[str, Any]) -> bool:
        """Determine if entry should be evicted."""
        pass


class LRUStrategy(CacheStrategy):
    """Least Recently Used caching strategy."""
    
    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
    
    def should_cache(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> bool:
        """Always cache with LRU strategy."""
        return True
    
    def get_ttl(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Return default TTL."""
        return self.default_ttl
    
    def should_evict(self, entry_info: Dict[str, Any]) -> bool:
        """Evict based on LRU (handled by OrderedDict)."""
        return entry_info.get('age_seconds', 0) > self.default_ttl


class TTLStrategy(CacheStrategy):
    """Time-To-Live based caching strategy."""
    
    def __init__(self, default_ttl: float = 300.0, max_ttl: float = 3600.0):
        self.default_ttl = default_ttl
        self.max_ttl = max_ttl
    
    def should_cache(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> bool:
        """Cache if calculation time is significant or market data is fresh."""
        if calculation_time > 0.1:  # Cache expensive calculations
            return True
        
        if market_conditions:
            data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
            return data_age < 60  # Cache if market data is fresh
        
        return True  # Default to caching
    
    def get_ttl(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Calculate TTL based on market conditions and calculation complexity."""
        ttl = self.default_ttl
        
        # Market data freshness affects TTL
        if market_conditions:
            data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
            
            if data_age < 30:  # Very fresh data
                ttl *= 2.0
            elif data_age < 60:  # Fresh data
                ttl *= 1.5
            elif data_age > 300:  # Stale data
                ttl *= 0.5
        
        # Expensive calculations get longer TTL
        if calculation_time > 1.0:
            ttl *= 2.0
        elif calculation_time > 0.5:
            ttl *= 1.5
        
        # Market volatility affects TTL
        if market_conditions and market_conditions.implied_volatility:
            vol = float(market_conditions.implied_volatility)
            if vol > 0.3:  # High volatility
                ttl *= 0.7
            elif vol > 0.2:  # Medium volatility
                ttl *= 0.85
        
        return min(ttl, self.max_ttl)
    
    def should_evict(self, entry_info: Dict[str, Any]) -> bool:
        """Evict if TTL expired."""
        return entry_info.get('is_expired', False)


class AdaptiveStrategy(CacheStrategy):
    """Adaptive caching strategy that learns from access patterns."""
    
    def __init__(
        self,
        default_ttl: float = 300.0,
        learning_rate: float = 0.1,
        min_ttl: float = 60.0,
        max_ttl: float = 3600.0
    ):
        self.default_ttl = default_ttl
        self.learning_rate = learning_rate
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        
        # Learning data
        self.access_patterns: Dict[str, Dict] = {}
        self.symbol_performance: Dict[str, Dict] = {}
        self.broker_performance: Dict[str, Dict] = {}
    
    def should_cache(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> bool:
        """Decide based on learned patterns."""
        symbol_key = request.symbol
        broker_key = request.broker_config.broker_name if hasattr(request, 'broker_config') else 'unknown'
        
        # Always cache initially
        if symbol_key not in self.symbol_performance:
            return True
        
        # Check if this symbol/broker combination benefits from caching
        symbol_stats = self.symbol_performance.get(symbol_key, {})
        hit_rate = symbol_stats.get('hit_rate', 0.0)
        
        # Cache if hit rate is good or calculation is expensive
        return hit_rate > 0.1 or calculation_time > 0.2
    
    def get_ttl(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Calculate adaptive TTL based on learned patterns."""
        symbol_key = request.symbol
        
        # Start with base TTL calculation (similar to TTL strategy)
        ttl = self._calculate_base_ttl(market_conditions, calculation_time)
        
        # Adapt based on symbol-specific patterns
        if symbol_key in self.symbol_performance:
            stats = self.symbol_performance[symbol_key]
            
            # Adjust based on access frequency
            access_freq = stats.get('access_frequency', 0.0)
            if access_freq > 0.1:  # High frequency access
                ttl *= 1.5
            elif access_freq < 0.01:  # Low frequency access
                ttl *= 0.7
            
            # Adjust based on hit rate
            hit_rate = stats.get('hit_rate', 0.0)
            if hit_rate > 0.5:  # Good hit rate
                ttl *= 1.3
            elif hit_rate < 0.2:  # Poor hit rate
                ttl *= 0.8
        
        return max(self.min_ttl, min(ttl, self.max_ttl))
    
    def should_evict(self, entry_info: Dict[str, Any]) -> bool:
        """Adaptive eviction based on access patterns."""
        age = entry_info.get('age_seconds', 0)
        access_count = entry_info.get('access_count', 0)
        last_access_age = entry_info.get('last_access_age', age)
        
        # Don't evict recently accessed entries
        if last_access_age < 60:  # Accessed in last minute
            return False
        
        # Evict if old and not accessed much
        if age > 1800 and access_count < 2:  # 30 minutes old, <2 accesses
            return True
        
        # Evict if very old
        if age > 3600:  # 1 hour old
            return True
        
        return False
    
    def record_access(
        self,
        symbol: str,
        broker_name: str,
        hit: bool,
        access_time: float
    ):
        """Record access pattern for learning."""
        # Update symbol performance
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'total_accesses': 0,
                'hits': 0,
                'hit_rate': 0.0,
                'access_frequency': 0.0,
                'last_access': datetime.now()
            }
        
        stats = self.symbol_performance[symbol]
        stats['total_accesses'] += 1
        if hit:
            stats['hits'] += 1
        
        # Update hit rate with exponential moving average
        new_hit_rate = stats['hits'] / stats['total_accesses']
        stats['hit_rate'] = (
            (1 - self.learning_rate) * stats['hit_rate'] +
            self.learning_rate * new_hit_rate
        )
        
        # Update access frequency
        time_since_last = (datetime.now() - stats['last_access']).total_seconds()
        if time_since_last > 0:
            freq = 1.0 / time_since_last
            stats['access_frequency'] = (
                (1 - self.learning_rate) * stats['access_frequency'] +
                self.learning_rate * freq
            )
        
        stats['last_access'] = datetime.now()
        
        # Update broker performance
        if broker_name not in self.broker_performance:
            self.broker_performance[broker_name] = {
                'total_accesses': 0,
                'hits': 0,
                'hit_rate': 0.0,
                'avg_access_time': 0.0
            }
        
        broker_stats = self.broker_performance[broker_name]
        broker_stats['total_accesses'] += 1
        if hit:
            broker_stats['hits'] += 1
        
        broker_stats['hit_rate'] = broker_stats['hits'] / broker_stats['total_accesses']
        
        # Update average access time
        broker_stats['avg_access_time'] = (
            (broker_stats['avg_access_time'] * (broker_stats['total_accesses'] - 1) + access_time) /
            broker_stats['total_accesses']
        )
    
    def _calculate_base_ttl(
        self,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Calculate base TTL similar to TTL strategy."""
        ttl = self.default_ttl
        
        if market_conditions:
            data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
            
            if data_age < 30:
                ttl *= 2.0
            elif data_age < 60:
                ttl *= 1.5
            elif data_age > 300:
                ttl *= 0.5
        
        if calculation_time > 1.0:
            ttl *= 2.0
        elif calculation_time > 0.5:
            ttl *= 1.5
        
        return ttl
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for analysis."""
        return {
            'symbols_tracked': len(self.symbol_performance),
            'brokers_tracked': len(self.broker_performance),
            'symbol_performance': dict(self.symbol_performance),
            'broker_performance': dict(self.broker_performance),
            'learning_rate': self.learning_rate
        }


class VolumeBasedStrategy(CacheStrategy):
    """Caching strategy based on trading volume patterns."""
    
    def __init__(
        self,
        default_ttl: float = 300.0,
        high_volume_threshold: int = 100000,
        low_volume_threshold: int = 1000
    ):
        self.default_ttl = default_ttl
        self.high_volume_threshold = high_volume_threshold
        self.low_volume_threshold = low_volume_threshold
    
    def should_cache(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> bool:
        """Cache based on volume and liquidity."""
        # Always cache large transactions
        if request.quantity > 10000:
            return True
        
        # Cache if market has good volume
        if market_conditions and market_conditions.volume:
            return market_conditions.volume > self.low_volume_threshold
        
        return True  # Default to caching
    
    def get_ttl(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Calculate TTL based on volume characteristics."""
        ttl = self.default_ttl
        
        # Large transactions get longer TTL (more expensive to recalculate)
        if request.quantity > 10000:
            ttl *= 2.0
        elif request.quantity > 5000:
            ttl *= 1.5
        
        # High volume markets are more stable for caching
        if market_conditions and market_conditions.volume:
            volume = market_conditions.volume
            
            if volume > self.high_volume_threshold:
                ttl *= 1.5  # High liquidity, stable costs
            elif volume < self.low_volume_threshold:
                ttl *= 0.7  # Low liquidity, costs change quickly
        
        # Consider average daily volume if available
        if market_conditions and market_conditions.average_daily_volume:
            adv = market_conditions.average_daily_volume
            current_vol = market_conditions.volume or 0
            
            if current_vol > adv * 2:  # Unusually high volume
                ttl *= 0.8  # Costs may be abnormal
            elif current_vol < adv * 0.5:  # Unusually low volume
                ttl *= 0.6  # Low liquidity affects costs
        
        return ttl
    
    def should_evict(self, entry_info: Dict[str, Any]) -> bool:
        """Evict based on volume-related factors."""
        age = entry_info.get('age_seconds', 0)
        
        # Get volume info from entry metadata if available
        volume = entry_info.get('volume', 0)
        
        if volume > self.high_volume_threshold:
            # High volume entries can stay longer
            return age > 900  # 15 minutes
        elif volume < self.low_volume_threshold:
            # Low volume entries expire quickly
            return age > 180  # 3 minutes
        else:
            # Normal volume
            return age > 600  # 10 minutes


class InstrumentTypeStrategy(CacheStrategy):
    """Strategy that adapts caching based on instrument type."""
    
    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        
        # Different TTL multipliers for different instrument types
        self.ttl_multipliers = {
            'EQUITY': 1.0,
            'OPTION': 0.5,    # Options prices change faster
            'FUTURE': 0.7,    # Futures are moderately volatile
            'ETF': 1.2,       # ETFs are relatively stable
            'BOND': 2.0,      # Bonds are stable
            'COMMODITY': 0.6, # Commodities can be volatile
            'CURRENCY': 0.4   # FX rates change rapidly
        }
    
    def should_cache(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> bool:
        """Cache based on instrument characteristics."""
        # Some instruments benefit more from caching
        instrument_type = request.instrument_type.name
        
        # Always cache bonds and ETFs (stable)
        if instrument_type in ['BOND', 'ETF']:
            return True
        
        # Cache currencies and options only if calculation is expensive
        if instrument_type in ['CURRENCY', 'OPTION']:
            return calculation_time > 0.1
        
        return True  # Default to caching
    
    def get_ttl(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Get TTL based on instrument type."""
        instrument_type = request.instrument_type.name
        multiplier = self.ttl_multipliers.get(instrument_type, 1.0)
        
        ttl = self.default_ttl * multiplier
        
        # Consider options expiry
        if instrument_type == 'OPTION' and market_conditions:
            if hasattr(market_conditions, 'days_to_expiry') and market_conditions.days_to_expiry:
                if market_conditions.days_to_expiry < 7:  # Near expiry
                    ttl *= 0.5
                elif market_conditions.days_to_expiry < 30:  # Medium term
                    ttl *= 0.7
        
        # Consider market timing
        if hasattr(request, 'market_timing'):
            if request.market_timing.name in ['PRE_MARKET', 'AFTER_HOURS']:
                ttl *= 0.5  # Less stable pricing outside regular hours
        
        return ttl
    
    def should_evict(self, entry_info: Dict[str, Any]) -> bool:
        """Evict based on instrument-specific factors."""
        age = entry_info.get('age_seconds', 0)
        instrument_type = entry_info.get('instrument_type', 'EQUITY')
        
        # Different eviction thresholds by instrument type
        thresholds = {
            'EQUITY': 600,    # 10 minutes
            'OPTION': 180,    # 3 minutes
            'FUTURE': 300,    # 5 minutes
            'ETF': 900,       # 15 minutes
            'BOND': 1800,     # 30 minutes
            'COMMODITY': 240, # 4 minutes
            'CURRENCY': 120   # 2 minutes
        }
        
        threshold = thresholds.get(instrument_type, 600)
        return age > threshold


logger.info("Cache strategies module loaded successfully")