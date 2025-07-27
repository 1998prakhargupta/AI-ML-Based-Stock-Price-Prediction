"""
Performance Optimizer
====================

Optimizes transaction cost calculation performance through intelligent
resource management, parallel processing, and adaptive algorithms.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    avg_calculation_time: float = 0.0
    p95_calculation_time: float = 0.0
    p99_calculation_time: float = 0.0
    throughput_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    active_threads: int = 0
    queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'avg_calculation_time': self.avg_calculation_time,
            'p95_calculation_time': self.p95_calculation_time,
            'p99_calculation_time': self.p99_calculation_time,
            'throughput_per_second': self.throughput_per_second,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization': self.cpu_utilization,
            'active_threads': self.active_threads,
            'queue_size': self.queue_size
        }


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with rationale."""
    category: str  # cache, threading, algorithm, etc.
    action: str    # increase, decrease, enable, disable, etc.
    parameter: str # specific parameter to adjust
    current_value: Any
    recommended_value: Any
    confidence: float  # 0.0 to 1.0
    expected_improvement: str
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'category': self.category,
            'action': self.action,
            'parameter': self.parameter,
            'current_value': self.current_value,
            'recommended_value': self.recommended_value,
            'confidence': self.confidence,
            'expected_improvement': self.expected_improvement,
            'rationale': self.rationale
        }


class AdaptiveThreadPool:
    """Thread pool that adapts size based on performance metrics."""
    
    def __init__(
        self,
        initial_size: int = 4,
        min_size: int = 2,
        max_size: int = 16,
        target_utilization: float = 0.8
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.target_utilization = target_utilization
        
        self.current_size = initial_size
        self.executor = ThreadPoolExecutor(max_workers=initial_size, thread_name_prefix="adaptive_")
        
        # Performance tracking
        self.utilization_history = deque(maxlen=100)
        self.response_time_history = deque(maxlen=100)
        self.last_adjustment = datetime.now()
        
        # Adjustment settings
        self.adjustment_interval = timedelta(minutes=5)
        self.adjustment_threshold = 0.1  # 10% change threshold
        
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit task and track performance."""
        start_time = time.perf_counter()
        future = self.executor.submit(fn, *args, **kwargs)
        
        def track_completion(fut):
            try:
                response_time = time.perf_counter() - start_time
                self.response_time_history.append(response_time)
                
                # Estimate utilization based on queue size and active threads
                queue_size = getattr(self.executor._work_queue, 'qsize', lambda: 0)()
                utilization = min(1.0, queue_size / max(1, self.current_size))
                self.utilization_history.append(utilization)
                
                # Check if adjustment is needed
                if datetime.now() - self.last_adjustment > self.adjustment_interval:
                    self._consider_adjustment()
                    
            except Exception as e:
                logger.warning(f"Performance tracking error: {e}")
        
        future.add_done_callback(track_completion)
        return future
    
    def _consider_adjustment(self):
        """Consider adjusting thread pool size."""
        if len(self.utilization_history) < 10:
            return
        
        avg_utilization = statistics.mean(self.utilization_history)
        avg_response_time = statistics.mean(self.response_time_history)
        
        new_size = self.current_size
        
        # Increase threads if high utilization and slow response
        if avg_utilization > self.target_utilization + self.adjustment_threshold:
            if avg_response_time > 0.1:  # 100ms threshold
                new_size = min(self.max_size, self.current_size + 2)
        
        # Decrease threads if low utilization
        elif avg_utilization < self.target_utilization - self.adjustment_threshold:
            new_size = max(self.min_size, self.current_size - 1)
        
        if new_size != self.current_size:
            self._resize_pool(new_size)
    
    def _resize_pool(self, new_size: int):
        """Resize the thread pool."""
        try:
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_size, thread_name_prefix="adaptive_")
            self.current_size = new_size
            self.last_adjustment = datetime.now()
            
            # Shutdown old executor gracefully
            old_executor.shutdown(wait=False)
            
            logger.info(f"Adjusted thread pool size to {new_size}")
            
        except Exception as e:
            logger.error(f"Failed to resize thread pool: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get thread pool metrics."""
        avg_util = statistics.mean(self.utilization_history) if self.utilization_history else 0
        avg_response = statistics.mean(self.response_time_history) if self.response_time_history else 0
        
        return {
            'current_size': self.current_size,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'target_utilization': self.target_utilization,
            'avg_utilization': avg_util,
            'avg_response_time': avg_response,
            'last_adjustment': self.last_adjustment.isoformat()
        }
    
    def shutdown(self):
        """Shutdown the thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=10.0)


class PerformanceOptimizer:
    """
    Performance optimizer for transaction cost calculations.
    
    Features:
    - Adaptive thread pool sizing
    - Cache optimization recommendations
    - Algorithm selection based on patterns
    - Resource usage optimization
    - Performance bottleneck identification
    """
    
    def __init__(
        self,
        target_response_time_ms: float = 100.0,
        target_throughput: float = 1000.0,  # calculations per second
        optimization_interval: float = 300.0,  # 5 minutes
        enable_auto_optimization: bool = True
    ):
        """
        Initialize performance optimizer.
        
        Args:
            target_response_time_ms: Target response time in milliseconds
            target_throughput: Target throughput (calculations/second)
            optimization_interval: How often to run optimization (seconds)
            enable_auto_optimization: Enable automatic optimizations
        """
        self.target_response_time_ms = target_response_time_ms
        self.target_throughput = target_throughput
        self.optimization_interval = optimization_interval
        self.enable_auto_optimization = enable_auto_optimization
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.calculation_times = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        
        # Adaptive components
        self.adaptive_thread_pool = AdaptiveThreadPool()
        
        # Optimization state
        self.optimizations_applied = []
        self.last_optimization = datetime.now()
        
        # Background optimization
        if enable_auto_optimization:
            self._running = True
            self._optimization_thread = threading.Thread(target=self._optimization_worker, daemon=True)
            self._optimization_thread.start()
        
        logger.info("Performance optimizer initialized")
    
    def record_calculation(
        self,
        calculation_time: float,
        cache_hit: bool,
        error: bool = False,
        queue_size: int = 0
    ):
        """Record a calculation for performance tracking."""
        self.calculation_times.append(calculation_time)
        
        # Sample throughput periodically
        current_time = time.time()
        if len(self.throughput_samples) == 0 or current_time - self.throughput_samples[-1][1] > 1.0:
            # Calculate throughput over last second
            recent_calcs = [t for t in self.calculation_times if current_time - t < 1.0]
            throughput = len(recent_calcs)
            self.throughput_samples.append((throughput, current_time))
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.calculation_times:
            return PerformanceMetrics()
        
        times = list(self.calculation_times)
        
        metrics = PerformanceMetrics(
            avg_calculation_time=statistics.mean(times),
            p95_calculation_time=self._percentile(times, 0.95),
            p99_calculation_time=self._percentile(times, 0.99),
            throughput_per_second=self._calculate_current_throughput(),
            active_threads=self.adaptive_thread_pool.current_size
        )
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_current_throughput(self) -> float:
        """Calculate current throughput."""
        if not self.throughput_samples:
            return 0.0
        
        recent_samples = [s[0] for s in self.throughput_samples if time.time() - s[1] < 60]
        return statistics.mean(recent_samples) if recent_samples else 0.0
    
    def analyze_performance(self) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations."""
        recommendations = []
        current_metrics = self.get_current_metrics()
        
        # Response time analysis
        if current_metrics.avg_calculation_time * 1000 > self.target_response_time_ms:
            recommendations.extend(self._analyze_response_time(current_metrics))
        
        # Throughput analysis
        if current_metrics.throughput_per_second < self.target_throughput:
            recommendations.extend(self._analyze_throughput(current_metrics))
        
        # Cache analysis
        recommendations.extend(self._analyze_cache_performance(current_metrics))
        
        # Thread pool analysis
        recommendations.extend(self._analyze_thread_pool(current_metrics))
        
        return recommendations
    
    def _analyze_response_time(self, metrics: PerformanceMetrics) -> List[OptimizationRecommendation]:
        """Analyze response time issues."""
        recommendations = []
        
        current_ms = metrics.avg_calculation_time * 1000
        target_ms = self.target_response_time_ms
        
        if current_ms > target_ms * 2:  # Significantly slow
            # Recommend increasing cache size
            recommendations.append(OptimizationRecommendation(
                category="cache",
                action="increase",
                parameter="cache_size",
                current_value="unknown",
                recommended_value="increase by 50%",
                confidence=0.8,
                expected_improvement="20-40% response time reduction",
                rationale=f"Response time ({current_ms:.1f}ms) is {current_ms/target_ms:.1f}x target"
            ))
            
            # Recommend parallel processing
            recommendations.append(OptimizationRecommendation(
                category="threading",
                action="increase",
                parameter="max_workers",
                current_value=metrics.active_threads,
                recommended_value=min(16, metrics.active_threads * 2),
                confidence=0.7,
                expected_improvement="30-50% throughput increase",
                rationale="High response times suggest need for more parallelism"
            ))
        
        return recommendations
    
    def _analyze_throughput(self, metrics: PerformanceMetrics) -> List[OptimizationRecommendation]:
        """Analyze throughput issues."""
        recommendations = []
        
        if metrics.throughput_per_second < self.target_throughput * 0.5:
            # Recommend batch processing
            recommendations.append(OptimizationRecommendation(
                category="algorithm",
                action="enable",
                parameter="batch_processing",
                current_value=False,
                recommended_value=True,
                confidence=0.9,
                expected_improvement="2-5x throughput increase",
                rationale=f"Current throughput ({metrics.throughput_per_second:.1f}/s) is well below target"
            ))
        
        return recommendations
    
    def _analyze_cache_performance(self, metrics: PerformanceMetrics) -> List[OptimizationRecommendation]:
        """Analyze cache performance."""
        recommendations = []
        
        if metrics.cache_hit_rate < 0.5:  # Low hit rate
            recommendations.append(OptimizationRecommendation(
                category="cache",
                action="increase",
                parameter="cache_ttl",
                current_value="unknown",
                recommended_value="increase by 50%",
                confidence=0.7,
                expected_improvement="10-30% hit rate improvement",
                rationale=f"Low cache hit rate ({metrics.cache_hit_rate:.1%})"
            ))
            
            recommendations.append(OptimizationRecommendation(
                category="cache",
                action="enable",
                parameter="cache_warming",
                current_value=False,
                recommended_value=True,
                confidence=0.8,
                expected_improvement="20-40% hit rate improvement",
                rationale="Cache warming can pre-populate common patterns"
            ))
        
        return recommendations
    
    def _analyze_thread_pool(self, metrics: PerformanceMetrics) -> List[OptimizationRecommendation]:
        """Analyze thread pool configuration."""
        recommendations = []
        
        pool_metrics = self.adaptive_thread_pool.get_metrics()
        
        if pool_metrics['avg_utilization'] > 0.9:  # High utilization
            recommendations.append(OptimizationRecommendation(
                category="threading",
                action="increase",
                parameter="max_workers",
                current_value=metrics.active_threads,
                recommended_value=min(32, metrics.active_threads + 4),
                confidence=0.8,
                expected_improvement="Reduced queue times",
                rationale=f"Thread pool utilization is high ({pool_metrics['avg_utilization']:.1%})"
            ))
        
        elif pool_metrics['avg_utilization'] < 0.3:  # Low utilization
            recommendations.append(OptimizationRecommendation(
                category="threading",
                action="decrease",
                parameter="max_workers",
                current_value=metrics.active_threads,
                recommended_value=max(2, metrics.active_threads - 2),
                confidence=0.6,
                expected_improvement="Reduced resource usage",
                rationale=f"Thread pool utilization is low ({pool_metrics['avg_utilization']:.1%})"
            ))
        
        return recommendations
    
    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply an optimization recommendation."""
        try:
            applied = False
            
            if recommendation.category == "threading" and recommendation.parameter == "max_workers":
                if recommendation.action == "increase":
                    new_size = min(32, int(recommendation.recommended_value))
                    self.adaptive_thread_pool._resize_pool(new_size)
                    applied = True
                elif recommendation.action == "decrease":
                    new_size = max(2, int(recommendation.recommended_value))
                    self.adaptive_thread_pool._resize_pool(new_size)
                    applied = True
            
            # Add more optimization implementations as needed
            
            if applied:
                self.optimizations_applied.append({
                    'recommendation': recommendation.to_dict(),
                    'applied_at': datetime.now().isoformat(),
                    'success': True
                })
                logger.info(f"Applied optimization: {recommendation.category}.{recommendation.parameter}")
            
            return applied
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
            self.optimizations_applied.append({
                'recommendation': recommendation.to_dict(),
                'applied_at': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            return False
    
    def auto_optimize(self):
        """Perform automatic optimization based on current metrics."""
        if not self.enable_auto_optimization:
            return
        
        recommendations = self.analyze_performance()
        
        # Apply high-confidence recommendations automatically
        for rec in recommendations:
            if rec.confidence > 0.8:
                self.apply_optimization(rec)
    
    def _optimization_worker(self):
        """Background worker for automatic optimization."""
        while getattr(self, '_running', False):
            try:
                self.auto_optimize()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Optimization worker error: {e}")
                time.sleep(60)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        current_metrics = self.get_current_metrics()
        recommendations = self.analyze_performance()
        
        return {
            'current_metrics': current_metrics.to_dict(),
            'recommendations': [rec.to_dict() for rec in recommendations],
            'applied_optimizations': self.optimizations_applied[-10:],  # Last 10
            'thread_pool_metrics': self.adaptive_thread_pool.get_metrics(),
            'performance_targets': {
                'response_time_ms': self.target_response_time_ms,
                'throughput_per_second': self.target_throughput
            },
            'target_achievement': {
                'response_time': current_metrics.avg_calculation_time * 1000 <= self.target_response_time_ms,
                'throughput': current_metrics.throughput_per_second >= self.target_throughput
            }
        }
    
    def enable_optimization_category(self, category: str):
        """Enable automatic optimization for a category."""
        # This would be implemented to enable/disable specific types of optimizations
        logger.info(f"Enabled optimization category: {category}")
    
    def disable_optimization_category(self, category: str):
        """Disable automatic optimization for a category."""
        # This would be implemented to enable/disable specific types of optimizations
        logger.info(f"Disabled optimization category: {category}")
    
    def reset_optimizations(self):
        """Reset all applied optimizations."""
        # Reset thread pool to initial size
        self.adaptive_thread_pool._resize_pool(4)
        
        # Clear optimization history
        self.optimizations_applied.clear()
        
        logger.info("Reset all optimizations")
    
    def shutdown(self):
        """Shutdown the optimizer."""
        if hasattr(self, '_running'):
            self._running = False
        
        if hasattr(self, '_optimization_thread'):
            self._optimization_thread.join(timeout=5.0)
        
        self.adaptive_thread_pool.shutdown()
        
        logger.info("Performance optimizer shutdown completed")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except:
            pass


logger.info("Performance optimizer module loaded successfully")