"""
Intelligent Cache Manager
========================

Provides intelligent caching mechanisms for transaction cost calculations
with multiple strategies, automatic invalidation, and performance optimization.
"""

import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import logging
import threading
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum, auto

# Import existing components
from ..models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration
)
from ..exceptions import TransactionCostError
from ..constants import SYSTEM_DEFAULTS

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels with different performance characteristics."""
    L1_MEMORY = auto()      # Fast in-memory cache
    L2_PERSISTENT = auto()  # Persistent cache (disk/redis)
    L3_SHARED = auto()      # Shared cache across instances


@dataclass
class CacheKey:
    """Structured cache key for cost calculations."""
    symbol: str
    quantity: int
    transaction_type: str
    instrument_type: str
    broker_name: str
    market_data_hash: Optional[str] = None
    price_bucket: Optional[str] = None  # Bucketed price for better cache hits
    
    def to_string(self) -> str:
        """Convert to string representation."""
        parts = [
            self.symbol,
            str(self.quantity),
            self.transaction_type,
            self.instrument_type,
            self.broker_name
        ]
        
        if self.market_data_hash:
            parts.append(self.market_data_hash)
        if self.price_bucket:
            parts.append(self.price_bucket)
            
        return "|".join(parts)
    
    def to_hash(self) -> str:
        """Convert to hash string."""
        return hashlib.sha256(self.to_string().encode()).hexdigest()[:16]


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: CacheKey
    value: TransactionCostBreakdown
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: float = 300.0
    cost_calculation_time: float = 0.0
    confidence_level: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return self.age_seconds > self.ttl_seconds
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)."""
        age = self.age_seconds
        return self.access_count / max(age, 1.0)
    
    def touch(self):
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheStatistics:
    """Cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
        self.total_access_time = 0.0
        self.start_time = datetime.now()
        self._lock = threading.Lock()
    
    def record_hit(self, access_time: float = 0.0):
        """Record cache hit."""
        with self._lock:
            self.hits += 1
            self.total_access_time += access_time
    
    def record_miss(self, access_time: float = 0.0):
        """Record cache miss."""
        with self._lock:
            self.misses += 1
            self.total_access_time += access_time
    
    def record_eviction(self):
        """Record cache eviction."""
        with self._lock:
            self.evictions += 1
    
    def update_size(self, new_size: int):
        """Update cache size."""
        with self._lock:
            self.size = new_size
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate."""
        return 1.0 - self.hit_rate
    
    @property
    def average_access_time(self) -> float:
        """Calculate average access time."""
        total_accesses = self.hits + self.misses
        return self.total_access_time / total_accesses if total_accesses > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'average_access_time_ms': self.average_access_time * 1000,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class CacheManager:
    """
    Intelligent cache manager for transaction cost calculations.
    
    Features:
    - Multi-level caching (L1 memory, L2 persistent, L3 shared)
    - Intelligent cache key generation
    - Multiple eviction strategies
    - Automatic cache warming
    - Performance monitoring
    - Cache invalidation strategies
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 300.0,  # 5 minutes
        enable_l1_cache: bool = True,
        enable_l2_cache: bool = False,
        enable_l3_cache: bool = False,
        price_bucket_size: float = 0.01,  # 1% price buckets
        auto_cleanup_interval: float = 60.0  # Cleanup every minute
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of entries in L1 cache
            default_ttl: Default TTL in seconds
            enable_l1_cache: Enable L1 memory cache
            enable_l2_cache: Enable L2 persistent cache
            enable_l3_cache: Enable L3 shared cache
            price_bucket_size: Price bucketing for better cache hits (as fraction)
            auto_cleanup_interval: Automatic cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_l1_cache = enable_l1_cache
        self.enable_l2_cache = enable_l2_cache
        self.enable_l3_cache = enable_l3_cache
        self.price_bucket_size = price_bucket_size
        self.auto_cleanup_interval = auto_cleanup_interval
        
        # L1 Memory Cache (OrderedDict for LRU)
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_lock = threading.RLock()
        
        # Statistics
        self.l1_stats = CacheStatistics()
        self.l2_stats = CacheStatistics()
        self.l3_stats = CacheStatistics()
        
        # Cache warming configuration
        self.warming_enabled = True
        self.warming_patterns: List[Dict[str, Any]] = []
        
        # Automatic cleanup
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Cache manager initialized (max_size: {max_size}, ttl: {default_ttl}s)")
    
    def get(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None
    ) -> Optional[TransactionCostBreakdown]:
        """
        Get cached result for transaction cost calculation.
        
        Args:
            request: Transaction request
            broker_config: Broker configuration
            market_conditions: Market conditions
            
        Returns:
            Cached cost breakdown or None if not found
        """
        start_time = time.perf_counter()
        
        try:
            cache_key = self._generate_cache_key(request, broker_config, market_conditions)
            
            # Try L1 cache first
            if self.enable_l1_cache:
                result = self._get_from_l1(cache_key)
                if result:
                    access_time = time.perf_counter() - start_time
                    self.l1_stats.record_hit(access_time)
                    return result
            
            # Try L2 cache
            if self.enable_l2_cache:
                result = self._get_from_l2(cache_key)
                if result:
                    # Promote to L1
                    if self.enable_l1_cache:
                        self._put_to_l1(cache_key, result, market_conditions)
                    access_time = time.perf_counter() - start_time
                    self.l2_stats.record_hit(access_time)
                    return result
            
            # Try L3 cache
            if self.enable_l3_cache:
                result = self._get_from_l3(cache_key)
                if result:
                    # Promote to L1 and L2
                    if self.enable_l1_cache:
                        self._put_to_l1(cache_key, result, market_conditions)
                    if self.enable_l2_cache:
                        self._put_to_l2(cache_key, result, market_conditions)
                    access_time = time.perf_counter() - start_time
                    self.l3_stats.record_hit(access_time)
                    return result
            
            # Cache miss
            access_time = time.perf_counter() - start_time
            self.l1_stats.record_miss(access_time)
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def put(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        result: TransactionCostBreakdown,
        market_conditions: Optional[MarketConditions] = None,
        ttl_override: Optional[float] = None,
        calculation_time: float = 0.0
    ):
        """
        Store result in cache.
        
        Args:
            request: Transaction request
            broker_config: Broker configuration
            result: Cost breakdown to cache
            market_conditions: Market conditions
            ttl_override: Override default TTL
            calculation_time: Time taken for original calculation
        """
        try:
            cache_key = self._generate_cache_key(request, broker_config, market_conditions)
            ttl = ttl_override or self._determine_ttl(market_conditions, calculation_time)
            
            # Store in all enabled cache levels
            if self.enable_l1_cache:
                self._put_to_l1(cache_key, result, market_conditions, ttl, calculation_time)
            
            if self.enable_l2_cache:
                self._put_to_l2(cache_key, result, market_conditions, ttl, calculation_time)
            
            if self.enable_l3_cache:
                self._put_to_l3(cache_key, result, market_conditions, ttl, calculation_time)
                
        except Exception as e:
            logger.warning(f"Cache put error: {e}")
    
    def invalidate(
        self,
        symbol: Optional[str] = None,
        broker_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Invalidate cache entries based on criteria.
        
        Args:
            symbol: Invalidate entries for specific symbol
            broker_name: Invalidate entries for specific broker
            tags: Invalidate entries with specific tags
        """
        try:
            with self.l1_lock:
                keys_to_remove = []
                
                for key_hash, entry in self.l1_cache.items():
                    should_remove = False
                    
                    if symbol and entry.key.symbol == symbol:
                        should_remove = True
                    elif broker_name and entry.key.broker_name == broker_name:
                        should_remove = True
                    elif tags and any(tag in entry.tags for tag in tags):
                        should_remove = True
                    
                    if should_remove:
                        keys_to_remove.append(key_hash)
                
                for key_hash in keys_to_remove:
                    del self.l1_cache[key_hash]
                
                self.l1_stats.update_size(len(self.l1_cache))
                
                if keys_to_remove:
                    logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
                    
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
    
    def clear(self):
        """Clear all cache levels."""
        with self.l1_lock:
            self.l1_cache.clear()
            self.l1_stats.update_size(0)
        
        # Clear L2 and L3 if implemented
        # TODO: Implement L2/L3 clearing
        
        logger.info("Cache cleared")
    
    def warm_cache(
        self,
        symbols: List[str],
        broker_configs: List[BrokerConfiguration],
        cost_calculator: Callable,
        patterns: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Warm cache with common patterns.
        
        Args:
            symbols: List of symbols to warm
            broker_configs: List of broker configurations
            cost_calculator: Function to calculate costs for warming
            patterns: Custom warming patterns
        """
        if not self.warming_enabled:
            return
        
        patterns = patterns or self._get_default_warming_patterns()
        
        logger.info(f"Starting cache warming for {len(symbols)} symbols")
        
        warmed_count = 0
        for symbol in symbols:
            for broker_config in broker_configs:
                for pattern in patterns:
                    try:
                        request = self._create_request_from_pattern(symbol, pattern)
                        
                        # Check if already cached
                        if self.get(request, broker_config) is None:
                            # Calculate and cache
                            result = cost_calculator(request, broker_config)
                            if result:
                                self.put(request, broker_config, result)
                                warmed_count += 1
                                
                    except Exception as e:
                        logger.debug(f"Cache warming failed for {symbol}: {e}")
        
        logger.info(f"Cache warming completed. Warmed {warmed_count} entries.")
    
    def _generate_cache_key(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> CacheKey:
        """Generate structured cache key."""
        # Bucket price for better cache hits
        price_bucket = self._bucket_price(float(request.price))
        
        # Hash market data if available
        market_data_hash = None
        if market_conditions:
            market_data_hash = self._hash_market_conditions(market_conditions)
        
        return CacheKey(
            symbol=request.symbol,
            quantity=request.quantity,
            transaction_type=request.transaction_type.name,
            instrument_type=request.instrument_type.name,
            broker_name=broker_config.broker_name,
            market_data_hash=market_data_hash,
            price_bucket=price_bucket
        )
    
    def _bucket_price(self, price: float) -> str:
        """Bucket price for better cache hits."""
        bucket_size = self.price_bucket_size
        bucket = int(price / (price * bucket_size)) * bucket_size
        return f"{bucket:.4f}"
    
    def _hash_market_conditions(self, market_conditions: MarketConditions) -> str:
        """Create hash of relevant market conditions."""
        # Use only significant market data for hashing
        data_parts = []
        
        if market_conditions.bid_price:
            data_parts.append(f"bid_{float(market_conditions.bid_price):.4f}")
        if market_conditions.ask_price:
            data_parts.append(f"ask_{float(market_conditions.ask_price):.4f}")
        if market_conditions.volume:
            # Use volume buckets
            vol_bucket = self._bucket_volume(market_conditions.volume)
            data_parts.append(f"vol_{vol_bucket}")
        
        if not data_parts:
            return "no_market_data"
        
        data_string = "|".join(data_parts)
        return hashlib.md5(data_string.encode()).hexdigest()[:8]
    
    def _bucket_volume(self, volume: int) -> str:
        """Bucket volume for caching."""
        # Use logarithmic buckets for volume
        if volume <= 0:
            return "0"
        
        import math
        bucket = int(math.log10(volume))
        return f"10e{bucket}"
    
    def _determine_ttl(
        self,
        market_conditions: Optional[MarketConditions],
        calculation_time: float
    ) -> float:
        """Determine TTL based on market conditions and calculation complexity."""
        base_ttl = self.default_ttl
        
        if market_conditions:
            # Fresher market data gets longer TTL
            data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
            
            if data_age < 30:  # Very fresh
                ttl_multiplier = 1.5
            elif data_age < 60:  # Fresh
                ttl_multiplier = 1.2
            elif data_age < 300:  # Reasonable
                ttl_multiplier = 1.0
            else:  # Stale
                ttl_multiplier = 0.5
            
            base_ttl *= ttl_multiplier
        
        # Expensive calculations get longer TTL
        if calculation_time > 1.0:  # If calculation took >1 second
            base_ttl *= 1.5
        elif calculation_time > 0.5:  # If calculation took >500ms
            base_ttl *= 1.2
        
        return base_ttl
    
    def _get_from_l1(self, cache_key: CacheKey) -> Optional[TransactionCostBreakdown]:
        """Get from L1 memory cache."""
        key_hash = cache_key.to_hash()
        
        with self.l1_lock:
            if key_hash in self.l1_cache:
                entry = self.l1_cache[key_hash]
                
                if entry.is_expired:
                    del self.l1_cache[key_hash]
                    self.l1_stats.update_size(len(self.l1_cache))
                    return None
                
                # Move to end (LRU)
                entry.touch()
                self.l1_cache.move_to_end(key_hash)
                return entry.value
            
            return None
    
    def _put_to_l1(
        self,
        cache_key: CacheKey,
        result: TransactionCostBreakdown,
        market_conditions: Optional[MarketConditions],
        ttl: Optional[float] = None,
        calculation_time: float = 0.0
    ):
        """Put to L1 memory cache."""
        key_hash = cache_key.to_hash()
        ttl = ttl or self.default_ttl
        
        with self.l1_lock:
            # Evict if at capacity
            while len(self.l1_cache) >= self.max_size:
                oldest_key, _ = self.l1_cache.popitem(last=False)
                self.l1_stats.record_eviction()
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=result,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                cost_calculation_time=calculation_time,
                confidence_level=result.confidence_level or 1.0
            )
            
            # Add tags for invalidation
            entry.tags = [cache_key.symbol, cache_key.broker_name]
            if market_conditions and market_conditions.market_open:
                entry.tags.append("market_open")
            
            self.l1_cache[key_hash] = entry
            self.l1_stats.update_size(len(self.l1_cache))
    
    def _get_from_l2(self, cache_key: CacheKey) -> Optional[TransactionCostBreakdown]:
        """Get from L2 persistent cache (placeholder)."""
        # TODO: Implement L2 persistent cache (file/redis)
        return None
    
    def _put_to_l2(
        self,
        cache_key: CacheKey,
        result: TransactionCostBreakdown,
        market_conditions: Optional[MarketConditions],
        ttl: Optional[float] = None,
        calculation_time: float = 0.0
    ):
        """Put to L2 persistent cache (placeholder)."""
        # TODO: Implement L2 persistent cache (file/redis)
        pass
    
    def _get_from_l3(self, cache_key: CacheKey) -> Optional[TransactionCostBreakdown]:
        """Get from L3 shared cache (placeholder)."""
        # TODO: Implement L3 shared cache (redis/memcached)
        return None
    
    def _put_to_l3(
        self,
        cache_key: CacheKey,
        result: TransactionCostBreakdown,
        market_conditions: Optional[MarketConditions],
        ttl: Optional[float] = None,
        calculation_time: float = 0.0
    ):
        """Put to L3 shared cache (placeholder)."""
        # TODO: Implement L3 shared cache (redis/memcached)
        pass
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while self._cleanup_running:
            try:
                self._cleanup_expired_entries()
                time.sleep(self.auto_cleanup_interval)
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self.l1_lock:
            expired_keys = []
            
            for key_hash, entry in self.l1_cache.items():
                if entry.is_expired:
                    expired_keys.append(key_hash)
            
            for key_hash in expired_keys:
                del self.l1_cache[key_hash]
            
            if expired_keys:
                self.l1_stats.update_size(len(self.l1_cache))
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _get_default_warming_patterns(self) -> List[Dict[str, Any]]:
        """Get default cache warming patterns."""
        from ..models import TransactionType, InstrumentType
        
        return [
            {
                'quantities': [100, 500, 1000, 5000],
                'transaction_types': [TransactionType.BUY, TransactionType.SELL],
                'instrument_type': InstrumentType.EQUITY,
                'price_base': 100.0
            },
            {
                'quantities': [10, 50, 100],
                'transaction_types': [TransactionType.BUY, TransactionType.SELL],
                'instrument_type': InstrumentType.OPTION,
                'price_base': 10.0
            }
        ]
    
    def _create_request_from_pattern(self, symbol: str, pattern: Dict[str, Any]) -> TransactionRequest:
        """Create transaction request from warming pattern."""
        from ..models import TransactionRequest
        from decimal import Decimal
        import random
        
        quantity = random.choice(pattern['quantities'])
        transaction_type = random.choice(pattern['transaction_types'])
        price = Decimal(str(pattern['price_base']))
        
        return TransactionRequest(
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type=transaction_type,
            instrument_type=pattern['instrument_type']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'l1_cache': self.l1_stats.to_dict(),
            'l2_cache': self.l2_stats.to_dict() if self.enable_l2_cache else None,
            'l3_cache': self.l3_stats.to_dict() if self.enable_l3_cache else None,
            'configuration': {
                'max_size': self.max_size,
                'default_ttl': self.default_ttl,
                'price_bucket_size': self.price_bucket_size,
                'levels_enabled': {
                    'l1': self.enable_l1_cache,
                    'l2': self.enable_l2_cache,
                    'l3': self.enable_l3_cache
                }
            }
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self.l1_lock:
            entries_by_symbol = defaultdict(int)
            entries_by_broker = defaultdict(int)
            entries_by_age = defaultdict(int)
            
            for entry in self.l1_cache.values():
                entries_by_symbol[entry.key.symbol] += 1
                entries_by_broker[entry.key.broker_name] += 1
                
                age_bucket = int(entry.age_seconds / 60)  # Age in minutes
                entries_by_age[f"{age_bucket}min"] += 1
            
            return {
                'total_entries': len(self.l1_cache),
                'entries_by_symbol': dict(entries_by_symbol),
                'entries_by_broker': dict(entries_by_broker),
                'entries_by_age': dict(entries_by_age),
                'memory_usage_estimate': len(self.l1_cache) * 1024  # Rough estimate
            }
    
    def shutdown(self):
        """Shutdown cache manager and cleanup."""
        self._cleanup_running = False
        
        if hasattr(self, '_cleanup_thread'):
            self._cleanup_thread.join(timeout=5.0)
        
        logger.info("Cache manager shutdown completed")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except:
            pass


logger.info("Cache manager module loaded successfully")