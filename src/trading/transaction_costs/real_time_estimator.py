"""
Real-Time Cost Estimator
========================

Provides sub-second cost estimation with streaming market data integration,
asynchronous calculation pipelines, intelligent caching, and batch optimization.

This module is designed for high-frequency cost estimation scenarios where
speed and accuracy are critical.
"""

import asyncio
import time
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
import threading
import hashlib

# Import existing components
from .models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    InstrumentType,
    TransactionType
)
from .exceptions import (
    TransactionCostError,
    CalculationError,
    raise_calculation_error
)
from .constants import SYSTEM_DEFAULTS, CONFIDENCE_LEVELS
from .cost_aggregator import CostAggregator, AggregationResult

logger = logging.getLogger(__name__)


@dataclass
class EstimateRequest:
    """Request for real-time cost estimation."""
    request_id: str
    transaction_request: TransactionRequest
    broker_config: BrokerConfiguration
    market_conditions: Optional[MarketConditions] = None
    priority: int = 1  # 1=highest, 5=lowest
    callback: Optional[Callable] = None
    timeout_ms: int = 1000
    use_cache: bool = True
    
    
@dataclass
class EstimateResult:
    """Result of real-time cost estimation."""
    request_id: str
    cost_breakdown: Optional[TransactionCostBreakdown] = None
    calculation_time_ms: float = 0.0
    cache_hit: bool = False
    confidence_level: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry for cost estimates."""
    result: TransactionCostBreakdown
    timestamp: datetime
    hit_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    ttl_seconds: float = 300.0  # 5 minutes default
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class StreamingDataManager:
    """Manages streaming market data for real-time estimation."""
    
    def __init__(self):
        self.data_streams: Dict[str, MarketConditions] = {}
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self._lock = threading.RLock()
        self.last_update: Dict[str, datetime] = {}
        
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to market data updates for a symbol."""
        with self._lock:
            self.subscribers[symbol].add(callback)
            
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from market data updates."""
        with self._lock:
            self.subscribers[symbol].discard(callback)
            
    def update_market_data(self, symbol: str, market_conditions: MarketConditions):
        """Update market data and notify subscribers."""
        with self._lock:
            self.data_streams[symbol] = market_conditions
            self.last_update[symbol] = datetime.now()
            
            # Notify subscribers
            for callback in self.subscribers[symbol]:
                try:
                    callback(symbol, market_conditions)
                except Exception as e:
                    logger.warning(f"Subscriber callback failed for {symbol}: {e}")
                    
    def get_market_data(self, symbol: str) -> Optional[MarketConditions]:
        """Get latest market data for symbol."""
        with self._lock:
            return self.data_streams.get(symbol)
            
    def get_data_age(self, symbol: str) -> Optional[float]:
        """Get age of market data in seconds."""
        with self._lock:
            if symbol in self.last_update:
                return (datetime.now() - self.last_update[symbol]).total_seconds()
            return None


class RealTimeEstimator:
    """
    Real-time transaction cost estimator with sub-second performance.
    
    Features:
    - Sub-second cost estimation
    - Intelligent caching with TTL
    - Asynchronous calculation pipelines
    - Streaming market data integration
    - Batch calculation optimization
    - Priority-based request handling
    """
    
    def __init__(
        self,
        cost_aggregator: Optional[CostAggregator] = None,
        cache_size: int = 10000,
        default_cache_ttl: float = 300.0,  # 5 minutes
        max_workers: int = 8,
        enable_streaming: bool = True,
        performance_target_ms: float = 100.0  # Target: <100ms
    ):
        """
        Initialize the real-time estimator.
        
        Args:
            cost_aggregator: Cost aggregator instance
            cache_size: Maximum cache size
            default_cache_ttl: Default cache TTL in seconds
            max_workers: Maximum number of worker threads
            enable_streaming: Whether to enable streaming data
            performance_target_ms: Performance target in milliseconds
        """
        self.cost_aggregator = cost_aggregator or CostAggregator()
        self.cache_size = cache_size
        self.default_cache_ttl = default_cache_ttl
        self.max_workers = max_workers
        self.performance_target_ms = performance_target_ms
        
        # Cache management
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        
        # Request queue and processing
        self.request_queue: Dict[int, deque] = defaultdict(deque)  # Priority -> requests
        self.active_requests: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rt_estimator_")
        
        # Streaming data manager
        self.streaming_enabled = enable_streaming
        if enable_streaming:
            self.data_manager = StreamingDataManager()
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_time_ms': 0.0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'last_reset': datetime.now()
        }
        self.performance_lock = threading.Lock()
        
        # Background tasks
        self._running = True
        self._cache_cleanup_thread = threading.Thread(target=self._cache_cleanup_worker, daemon=True)
        self._cache_cleanup_thread.start()
        
        logger.info(f"Real-time estimator initialized (target: {performance_target_ms}ms)")
    
    async def estimate_cost_async(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        request_id: Optional[str] = None,
        priority: int = 1,
        timeout_ms: int = 1000,
        use_cache: bool = True
    ) -> EstimateResult:
        """
        Estimate transaction cost asynchronously.
        
        Args:
            request: Transaction request
            broker_config: Broker configuration
            market_conditions: Market conditions (will use streaming data if not provided)
            request_id: Request identifier
            priority: Request priority (1=highest, 5=lowest)
            timeout_ms: Timeout in milliseconds
            use_cache: Whether to use cache
            
        Returns:
            Cost estimate result
        """
        start_time = time.perf_counter()
        request_id = request_id or self._generate_request_id(request)
        
        try:
            # Use streaming data if available and not provided
            if not market_conditions and self.streaming_enabled:
                market_conditions = self.data_manager.get_market_data(request.symbol)
            
            # Check cache first
            if use_cache:
                cached_result = self._get_cached_result(request, broker_config, market_conditions)
                if cached_result:
                    calc_time_ms = (time.perf_counter() - start_time) * 1000
                    self._update_performance_stats(calc_time_ms, True, True)
                    
                    return EstimateResult(
                        request_id=request_id,
                        cost_breakdown=cached_result.result,
                        calculation_time_ms=calc_time_ms,
                        cache_hit=True,
                        confidence_level=cached_result.result.confidence_level or 0.8,
                        timestamp=datetime.now()
                    )
            
            # Create estimate request
            estimate_request = EstimateRequest(
                request_id=request_id,
                transaction_request=request,
                broker_config=broker_config,
                market_conditions=market_conditions,
                priority=priority,
                timeout_ms=timeout_ms,
                use_cache=use_cache
            )
            
            # Submit for processing
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_estimate_request,
                estimate_request
            )
            
            calc_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(calc_time_ms, result.error is None, False)
            
            return result
            
        except Exception as e:
            calc_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(calc_time_ms, False, False)
            
            return EstimateResult(
                request_id=request_id,
                calculation_time_ms=calc_time_ms,
                error=str(e),
                timestamp=datetime.now()
            )
    
    def estimate_cost_sync(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        use_cache: bool = True
    ) -> EstimateResult:
        """
        Estimate transaction cost synchronously.
        
        Args:
            request: Transaction request
            broker_config: Broker configuration
            market_conditions: Market conditions
            use_cache: Whether to use cache
            
        Returns:
            Cost estimate result
        """
        start_time = time.perf_counter()
        request_id = self._generate_request_id(request)
        
        try:
            # Use streaming data if available and not provided
            if not market_conditions and self.streaming_enabled:
                market_conditions = self.data_manager.get_market_data(request.symbol)
            
            # Check cache first
            if use_cache:
                cached_result = self._get_cached_result(request, broker_config, market_conditions)
                if cached_result:
                    calc_time_ms = (time.perf_counter() - start_time) * 1000
                    self._update_performance_stats(calc_time_ms, True, True)
                    
                    return EstimateResult(
                        request_id=request_id,
                        cost_breakdown=cached_result.result,
                        calculation_time_ms=calc_time_ms,
                        cache_hit=True,
                        confidence_level=cached_result.result.confidence_level or 0.8,
                        timestamp=datetime.now()
                    )
            
            # Calculate cost
            aggregation_result = self.cost_aggregator.calculate_total_cost(
                request, broker_config, market_conditions
            )
            
            # Cache result
            if use_cache:
                self._cache_result(request, broker_config, market_conditions, aggregation_result.cost_breakdown)
            
            calc_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(calc_time_ms, True, False)
            
            return EstimateResult(
                request_id=request_id,
                cost_breakdown=aggregation_result.cost_breakdown,
                calculation_time_ms=calc_time_ms,
                cache_hit=False,
                confidence_level=aggregation_result.confidence_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            calc_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(calc_time_ms, False, False)
            
            return EstimateResult(
                request_id=request_id,
                calculation_time_ms=calc_time_ms,
                error=str(e),
                timestamp=datetime.now()
            )
    
    async def estimate_batch_async(
        self,
        requests: List[Tuple[TransactionRequest, BrokerConfiguration]],
        market_conditions: Optional[Dict[str, MarketConditions]] = None,
        use_cache: bool = True,
        max_concurrent: int = 20
    ) -> List[EstimateResult]:
        """
        Estimate costs for multiple transactions in parallel.
        
        Args:
            requests: List of (transaction_request, broker_config) tuples
            market_conditions: Market conditions by symbol
            use_cache: Whether to use cache
            max_concurrent: Maximum concurrent calculations
            
        Returns:
            List of estimate results
        """
        if not requests:
            return []
        
        start_time = time.perf_counter()
        logger.info(f"Starting batch estimation for {len(requests)} requests")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def estimate_single(req_tuple):
            async with semaphore:
                transaction_req, broker_conf = req_tuple
                symbol_conditions = None
                if market_conditions and transaction_req.symbol in market_conditions:
                    symbol_conditions = market_conditions[transaction_req.symbol]
                
                return await self.estimate_cost_async(
                    transaction_req, broker_conf, symbol_conditions, use_cache=use_cache
                )
        
        # Execute all requests
        tasks = [estimate_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = EstimateResult(
                    request_id=f"batch_{i}",
                    error=str(result),
                    timestamp=datetime.now()
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Batch estimation completed in {total_time:.2f}ms "
            f"({len(requests)/(total_time/1000):.1f} calc/s)"
        )
        
        return final_results
    
    def _process_estimate_request(self, estimate_request: EstimateRequest) -> EstimateResult:
        """Process a single estimate request."""
        start_time = time.perf_counter()
        
        try:
            # Calculate cost using aggregator
            aggregation_result = self.cost_aggregator.calculate_total_cost(
                estimate_request.transaction_request,
                estimate_request.broker_config,
                estimate_request.market_conditions
            )
            
            # Cache result if enabled
            if estimate_request.use_cache:
                self._cache_result(
                    estimate_request.transaction_request,
                    estimate_request.broker_config,
                    estimate_request.market_conditions,
                    aggregation_result.cost_breakdown
                )
            
            calc_time_ms = (time.perf_counter() - start_time) * 1000
            
            return EstimateResult(
                request_id=estimate_request.request_id,
                cost_breakdown=aggregation_result.cost_breakdown,
                calculation_time_ms=calc_time_ms,
                cache_hit=False,
                confidence_level=aggregation_result.confidence_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            calc_time_ms = (time.perf_counter() - start_time) * 1000
            
            return EstimateResult(
                request_id=estimate_request.request_id,
                calculation_time_ms=calc_time_ms,
                error=str(e),
                timestamp=datetime.now()
            )
    
    def _generate_cache_key(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> str:
        """Generate cache key for the request."""
        key_parts = [
            request.symbol,
            str(request.quantity),
            str(request.price),
            request.transaction_type.name,
            request.instrument_type.name,
            broker_config.broker_name,
            broker_config.config_version
        ]
        
        if market_conditions:
            # Include market data but with reduced precision to improve cache hits
            if market_conditions.bid_price:
                key_parts.append(f"bid_{float(market_conditions.bid_price):.4f}")
            if market_conditions.ask_price:
                key_parts.append(f"ask_{float(market_conditions.ask_price):.4f}")
            if market_conditions.volume:
                key_parts.append(f"vol_{market_conditions.volume}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> Optional[CacheEntry]:
        """Get cached result if available and fresh."""
        cache_key = self._generate_cache_key(request, broker_config, market_conditions)
        
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                if not entry.is_expired:
                    entry.hit_count += 1
                    entry.last_access = datetime.now()
                    self.cache_stats['hits'] += 1
                    return entry
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
                    self.cache_stats['size'] = len(self.cache)
            
            self.cache_stats['misses'] += 1
            return None
    
    def _cache_result(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: TransactionCostBreakdown
    ):
        """Cache calculation result."""
        cache_key = self._generate_cache_key(request, broker_config, market_conditions)
        
        # Determine TTL based on market conditions
        ttl = self.default_cache_ttl
        if market_conditions:
            # Fresher data gets longer cache time
            data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
            if data_age < 60:  # Very fresh data
                ttl = 300  # 5 minutes
            elif data_age < 300:  # Reasonably fresh
                ttl = 180  # 3 minutes
            else:  # Older data
                ttl = 60   # 1 minute
        
        with self.cache_lock:
            # Evict old entries if cache is full
            if len(self.cache) >= self.cache_size:
                self._evict_cache_entries()
            
            self.cache[cache_key] = CacheEntry(
                result=result,
                timestamp=datetime.now(),
                ttl_seconds=ttl
            )
            self.cache_stats['size'] = len(self.cache)
    
    def _evict_cache_entries(self):
        """Evict old cache entries using LRU strategy."""
        if not self.cache:
            return
        
        # Sort by last access time and remove oldest 10%
        entries_to_evict = max(1, len(self.cache) // 10)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_access
        )
        
        for i in range(entries_to_evict):
            cache_key, _ = sorted_entries[i]
            del self.cache[cache_key]
            self.cache_stats['evictions'] += 1
        
        self.cache_stats['size'] = len(self.cache)
    
    def _cache_cleanup_worker(self):
        """Background worker to clean up expired cache entries."""
        while self._running:
            try:
                with self.cache_lock:
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                    
                    if expired_keys:
                        self.cache_stats['size'] = len(self.cache)
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                time.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _generate_request_id(self, request: TransactionRequest) -> str:
        """Generate unique request ID."""
        timestamp = str(int(time.time() * 1000))
        symbol_hash = hashlib.md5(request.symbol.encode()).hexdigest()[:8]
        return f"{timestamp}_{symbol_hash}"
    
    def _update_performance_stats(self, calc_time_ms: float, success: bool, cache_hit: bool):
        """Update performance statistics."""
        with self.performance_lock:
            self.performance_stats['total_requests'] += 1
            
            if success:
                self.performance_stats['successful_requests'] += 1
            else:
                self.performance_stats['failed_requests'] += 1
            
            # Update timing stats
            self.performance_stats['min_time_ms'] = min(
                self.performance_stats['min_time_ms'], calc_time_ms
            )
            self.performance_stats['max_time_ms'] = max(
                self.performance_stats['max_time_ms'], calc_time_ms
            )
            
            # Update running average
            total = self.performance_stats['total_requests']
            current_avg = self.performance_stats['average_time_ms']
            self.performance_stats['average_time_ms'] = (
                (current_avg * (total - 1)) + calc_time_ms
            ) / total
            
            # Update cache hit rate
            if cache_hit:
                self.cache_stats['hits'] += 1
            else:
                self.cache_stats['misses'] += 1
            
            total_cache_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            if total_cache_requests > 0:
                self.performance_stats['cache_hit_rate'] = (
                    self.cache_stats['hits'] / total_cache_requests
                )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.performance_lock:
            return {
                'requests': dict(self.performance_stats),
                'cache': dict(self.cache_stats),
                'target_performance_ms': self.performance_target_ms,
                'meets_target': self.performance_stats['average_time_ms'] <= self.performance_target_ms,
                'streaming_enabled': self.streaming_enabled,
                'cache_efficiency': {
                    'hit_rate_percent': self.performance_stats['cache_hit_rate'] * 100,
                    'avg_entry_age_seconds': self._get_average_cache_age(),
                    'eviction_rate': self.cache_stats.get('evictions', 0) / max(1, self.cache_stats['size'])
                }
            }
    
    def _get_average_cache_age(self) -> float:
        """Get average age of cache entries."""
        with self.cache_lock:
            if not self.cache:
                return 0.0
            
            total_age = sum(entry.age_seconds for entry in self.cache.values())
            return total_age / len(self.cache)
    
    def clear_cache(self):
        """Clear all cached results."""
        with self.cache_lock:
            self.cache.clear()
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'size': 0
            }
        logger.info("Real-time estimator cache cleared")
    
    def warm_cache(
        self,
        symbols: List[str],
        broker_configs: List[BrokerConfiguration],
        quantities: List[int] = None,
        transaction_types: List[TransactionType] = None
    ):
        """Warm up cache with common request patterns."""
        quantities = quantities or [100, 500, 1000, 5000]
        transaction_types = transaction_types or [TransactionType.BUY, TransactionType.SELL]
        
        logger.info(f"Warming cache for {len(symbols)} symbols")
        
        for symbol in symbols:
            for broker_config in broker_configs:
                for quantity in quantities:
                    for trans_type in transaction_types:
                        try:
                            request = TransactionRequest(
                                symbol=symbol,
                                quantity=quantity,
                                price=Decimal('100.00'),  # Default price
                                transaction_type=trans_type,
                                instrument_type=InstrumentType.EQUITY
                            )
                            
                            # Calculate and cache
                            self.estimate_cost_sync(request, broker_config, use_cache=True)
                            
                        except Exception as e:
                            logger.warning(f"Cache warming failed for {symbol}: {e}")
        
        logger.info(f"Cache warming completed. Cache size: {len(self.cache)}")
    
    def update_streaming_data(self, symbol: str, market_conditions: MarketConditions):
        """Update streaming market data."""
        if self.streaming_enabled:
            self.data_manager.update_market_data(symbol, market_conditions)
    
    def subscribe_to_symbol(self, symbol: str, callback: Callable):
        """Subscribe to market data updates for a symbol."""
        if self.streaming_enabled:
            self.data_manager.subscribe(symbol, callback)
    
    def shutdown(self):
        """Shutdown the estimator and cleanup resources."""
        self._running = False
        
        if hasattr(self, '_cache_cleanup_thread'):
            self._cache_cleanup_thread.join(timeout=5.0)
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=10.0)
        
        logger.info("Real-time estimator shutdown completed")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except:
            pass


logger.info("Real-time estimator module loaded successfully")