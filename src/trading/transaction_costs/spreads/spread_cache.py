"""
Spread Cache
===========

High-performance caching system for spread estimates with TTL,
invalidation strategies, and performance optimization.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple, Set
import threading
import time
import hashlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass, asdict

from .base_spread_model import SpreadEstimate, SpreadData

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: SpreadEstimate
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_entries: int
    memory_usage_bytes: int
    average_access_time_ms: float
    oldest_entry_age_seconds: float
    newest_entry_age_seconds: float


class SpreadCache:
    """
    High-performance spread cache with intelligent eviction and monitoring.
    
    Features:
    - TTL-based expiration
    - LRU eviction policy
    - Memory usage monitoring
    - Performance statistics
    - Thread-safe operations
    - Cache warming strategies
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        default_ttl_seconds: int = 300,  # 5 minutes
        max_memory_mb: int = 100,
        cleanup_interval_seconds: int = 60,
        enable_compression: bool = False
    ):
        """
        Initialize spread cache.
        
        Args:
            max_entries: Maximum number of cache entries
            default_ttl_seconds: Default TTL for entries
            max_memory_mb: Maximum memory usage in MB
            cleanup_interval_seconds: How often to run cleanup
            enable_compression: Whether to compress cached data
        """
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_compression = enable_compression
        
        # Cache storage (OrderedDict for LRU behavior)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()  # Use regular Lock instead of RWLock
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'cleanup_runs': 0,
            'memory_usage': 0
        }
        
        # Background cleanup
        self._cleanup_thread = None
        self._cleanup_running = False
        
        # Performance tracking
        self._access_times: List[float] = []
        self._max_access_time_samples = 1000
        
        self.start_background_cleanup()
        logger.info(f"Spread cache initialized: max_entries={max_entries}, ttl={default_ttl_seconds}s")
    
    def __del__(self):
        """Cleanup resources."""
        self.stop_background_cleanup()
    
    def get(self, key: str) -> Optional[SpreadEstimate]:
        """
        Get spread estimate from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached spread estimate or None
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self._stats['total_requests'] += 1
                
                if key not in self._cache:
                    self._stats['cache_misses'] += 1
                    return None
                
                entry = self._cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self._cache[key]
                    self._stats['cache_misses'] += 1
                    return None
                
                # Update access metadata
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self._stats['cache_hits'] += 1
                
                # Track access time
                access_time = (time.time() - start_time) * 1000  # ms
                self._track_access_time(access_time)
                
                return entry.value
                
        except Exception as e:
            logger.error(f"Error getting cache entry {key}: {e}")
            return None
    
    def put(
        self,
        key: str,
        value: SpreadEstimate,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Put spread estimate in cache.
        
        Args:
            key: Cache key
            value: Spread estimate to cache
            ttl_seconds: TTL override
            
        Returns:
            True if successfully cached
        """
        try:
            ttl = ttl_seconds or self.default_ttl_seconds
            
            # Calculate entry size
            entry_size = self._calculate_entry_size(value)
            
            with self._lock:
                # Check memory limits
                if self._would_exceed_memory_limit(entry_size):
                    self._evict_for_memory()
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    access_count=1,
                    ttl_seconds=ttl,
                    size_bytes=entry_size
                )
                
                # Remove existing entry if present
                if key in self._cache:
                    old_entry = self._cache[key]
                    self._stats['memory_usage'] -= old_entry.size_bytes
                
                # Add new entry
                self._cache[key] = entry
                self._stats['memory_usage'] += entry_size
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                # Evict if necessary
                self._evict_if_needed()
                
                return True
                
        except Exception as e:
            logger.error(f"Error putting cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if entry was deleted
        """
        try:
            with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    self._stats['memory_usage'] -= entry.size_bytes
                    del self._cache[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cache entry {key}: {e}")
            return False
    
    def invalidate_symbol(self, symbol: str) -> int:
        """
        Invalidate all cache entries for a symbol.
        
        Args:
            symbol: Symbol to invalidate
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        try:
            with self._lock:
                keys_to_delete = [
                    key for key in self._cache.keys()
                    if key.startswith(f"{symbol}_") or key.endswith(f"_{symbol}")
                ]
                
                for key in keys_to_delete:
                    entry = self._cache[key]
                    self._stats['memory_usage'] -= entry.size_bytes
                    del self._cache[key]
                    count += 1
                
                logger.info(f"Invalidated {count} cache entries for symbol {symbol}")
                
        except Exception as e:
            logger.error(f"Error invalidating symbol {symbol}: {e}")
        
        return count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with self._lock:
                self._cache.clear()
                self._stats['memory_usage'] = 0
                logger.info("Cache cleared")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        try:
            with self._lock:
                total_requests = self._stats['total_requests']
                cache_hits = self._stats['cache_hits']
                
                hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
                
                # Calculate entry age statistics
                now = datetime.now()
                entry_ages = [
                    (now - entry.created_at).total_seconds()
                    for entry in self._cache.values()
                ]
                
                oldest_age = max(entry_ages) if entry_ages else 0.0
                newest_age = min(entry_ages) if entry_ages else 0.0
                
                # Calculate average access time
                avg_access_time = (
                    sum(self._access_times) / len(self._access_times)
                    if self._access_times else 0.0
                )
                
                return CacheStats(
                    total_requests=total_requests,
                    cache_hits=cache_hits,
                    cache_misses=self._stats['cache_misses'],
                    hit_rate=hit_rate,
                    total_entries=len(self._cache),
                    memory_usage_bytes=self._stats['memory_usage'],
                    average_access_time_ms=avg_access_time,
                    oldest_entry_age_seconds=oldest_age,
                    newest_entry_age_seconds=newest_age
                )
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return CacheStats(0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0)
    
    def warm_cache(
        self,
        symbols: List[str],
        data_provider: callable,
        ttl_seconds: Optional[int] = None
    ) -> int:
        """
        Warm cache with spread estimates for symbols.
        
        Args:
            symbols: List of symbols to warm
            data_provider: Function that returns SpreadEstimate for a symbol
            ttl_seconds: TTL for warmed entries
            
        Returns:
            Number of entries successfully warmed
        """
        count = 0
        
        for symbol in symbols:
            try:
                estimate = data_provider(symbol)
                if estimate:
                    key = self.create_key(symbol, estimate.estimation_method)
                    if self.put(key, estimate, ttl_seconds):
                        count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to warm cache for symbol {symbol}: {e}")
        
        logger.info(f"Cache warmed with {count}/{len(symbols)} entries")
        return count
    
    def start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="SpreadCacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Started cache cleanup thread")
    
    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        logger.info("Stopped cache cleanup thread")
    
    @staticmethod
    def create_key(symbol: str, method: str, **kwargs) -> str:
        """
        Create standardized cache key.
        
        Args:
            symbol: Trading symbol
            method: Estimation method
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        key_parts = [symbol, method]
        
        # Add sorted kwargs
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}={v}" for k, v in sorted_kwargs)
        
        key_string = "|".join(key_parts)
        
        # Hash long keys
        if len(key_string) > 100:
            return hashlib.md5(key_string.encode()).hexdigest()
        
        return key_string
    
    # Private methods
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds
    
    def _calculate_entry_size(self, value: SpreadEstimate) -> int:
        """Calculate approximate size of cache entry."""
        try:
            if self.enable_compression:
                data = pickle.dumps(value)
                return len(data)
            else:
                # Rough estimate without actual serialization
                return len(str(value)) * 2  # Unicode overhead
                
        except Exception:
            return 1024  # Default estimate
    
    def _would_exceed_memory_limit(self, additional_bytes: int) -> bool:
        """Check if adding entry would exceed memory limit."""
        return (self._stats['memory_usage'] + additional_bytes) > self.max_memory_bytes
    
    def _evict_for_memory(self) -> None:
        """Evict entries to free memory."""
        target_size = self.max_memory_bytes * 0.8  # Target 80% usage
        
        while self._stats['memory_usage'] > target_size and self._cache:
            # Remove least recently used entry
            key, entry = self._cache.popitem(last=False)
            self._stats['memory_usage'] -= entry.size_bytes
            self._stats['evictions'] += 1
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        while len(self._cache) > self.max_entries:
            # Remove least recently used entry
            key, entry = self._cache.popitem(last=False)
            self._stats['memory_usage'] -= entry.size_bytes
            self._stats['evictions'] += 1
    
    def _track_access_time(self, access_time_ms: float) -> None:
        """Track cache access time for performance monitoring."""
        self._access_times.append(access_time_ms)
        
        # Keep only recent samples
        if len(self._access_times) > self._max_access_time_samples:
            self._access_times = self._access_times[-self._max_access_time_samples//2:]
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.info("Started cache cleanup loop")
        
        while self._cleanup_running:
            try:
                self._cleanup_expired_entries()
                self._stats['cleanup_runs'] += 1
                
                time.sleep(self.cleanup_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                time.sleep(self.cleanup_interval_seconds)
    
    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from cache."""
        expired_keys = []
        
        try:
            with self._lock:
                now = datetime.now()
                
                for key, entry in self._cache.items():
                    age = (now - entry.created_at).total_seconds()
                    if age > entry.ttl_seconds:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    entry = self._cache[key]
                    self._stats['memory_usage'] -= entry.size_bytes
                    del self._cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")


# Global cache instance
_global_cache: Optional[SpreadCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> SpreadCache:
    """Get or create global spread cache instance."""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = SpreadCache()
    
    return _global_cache


def configure_global_cache(**kwargs) -> SpreadCache:
    """Configure global cache with custom parameters."""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.stop_background_cleanup()
        
        _global_cache = SpreadCache(**kwargs)
    
    return _global_cache


logger.info("Spread cache module loaded successfully")