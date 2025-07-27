"""
Performance Monitor
==================

Real-time monitoring of transaction cost calculation performance with
alerting and dashboard capabilities.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    alert_id: str
    severity: str  # info, warning, critical
    metric: str
    threshold: float
    current_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'severity': self.severity,
            'metric': self.metric,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Features:
    - Real-time metrics collection
    - Threshold-based alerting
    - Performance trend analysis
    - Dashboard data provision
    """
    
    def __init__(
        self,
        collection_interval: float = 10.0,  # seconds
        retention_hours: int = 24,
        alert_cooldown: float = 300.0  # 5 minutes
    ):
        """
        Initialize performance monitor.
        
        Args:
            collection_interval: How often to collect metrics
            retention_hours: How long to retain metrics
            alert_cooldown: Minimum time between same alerts
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.alert_cooldown = alert_cooldown
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=int(retention_hours * 3600 / collection_interval))
        self.real_time_metrics: Dict[str, Any] = {}
        
        # Alert management
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Performance counters
        self.counters = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Timing data
        self.response_times = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        
        # Setup default thresholds
        self._setup_default_thresholds()
        
        # Background monitoring
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Performance monitor initialized")
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            'avg_response_time_ms': {
                'warning': 200.0,
                'critical': 500.0
            },
            'error_rate': {
                'warning': 0.05,  # 5%
                'critical': 0.10  # 10%
            },
            'cache_hit_rate': {
                'warning': 0.50,  # Below 50%
                'critical': 0.30  # Below 30%
            },
            'throughput_per_second': {
                'warning': 50.0,   # Below 50/sec
                'critical': 10.0   # Below 10/sec
            }
        }
    
    def record_calculation(
        self,
        response_time: float,
        success: bool,
        cache_hit: bool,
        error_details: Optional[str] = None
    ):
        """Record a calculation for monitoring."""
        self.counters['total_calculations'] += 1
        
        if success:
            self.counters['successful_calculations'] += 1
        else:
            self.counters['failed_calculations'] += 1
        
        if cache_hit:
            self.counters['cache_hits'] += 1
        else:
            self.counters['cache_misses'] += 1
        
        self.response_times.append(response_time)
        
        # Sample throughput
        current_time = time.time()
        if len(self.throughput_samples) == 0 or current_time - self.throughput_samples[-1][1] > 1.0:
            # Calculate recent throughput
            recent_time = current_time - 10.0  # Last 10 seconds
            recent_count = sum(1 for t in self.response_times if current_time - t < 10.0)
            throughput = recent_count / 10.0
            self.throughput_samples.append((throughput, current_time))
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_time = datetime.now()
        
        # Calculate derived metrics
        total_calcs = self.counters['total_calculations']
        total_cache_requests = self.counters['cache_hits'] + self.counters['cache_misses']
        
        metrics = {
            'timestamp': current_time.isoformat(),
            'counters': dict(self.counters),
            'avg_response_time_ms': 0.0,
            'p95_response_time_ms': 0.0,
            'error_rate': 0.0,
            'cache_hit_rate': 0.0,
            'throughput_per_second': 0.0
        }
        
        # Response time metrics
        if self.response_times:
            times_ms = [t * 1000 for t in self.response_times]
            metrics['avg_response_time_ms'] = statistics.mean(times_ms)
            metrics['p95_response_time_ms'] = self._percentile(times_ms, 0.95)
        
        # Error rate
        if total_calcs > 0:
            metrics['error_rate'] = self.counters['failed_calculations'] / total_calcs
        
        # Cache hit rate
        if total_cache_requests > 0:
            metrics['cache_hit_rate'] = self.counters['cache_hits'] / total_cache_requests
        
        # Throughput
        if self.throughput_samples:
            recent_samples = [s[0] for s in self.throughput_samples if time.time() - s[1] < 60]
            if recent_samples:
                metrics['throughput_per_second'] = statistics.mean(recent_samples)
        
        self.real_time_metrics = metrics
        return metrics
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def set_threshold(self, metric: str, severity: str, value: float):
        """Set alert threshold."""
        if metric not in self.alert_thresholds:
            self.alert_thresholds[metric] = {}
        
        self.alert_thresholds[metric][severity] = value
        logger.info(f"Set {severity} threshold for {metric}: {value}")
    
    def check_alerts(self):
        """Check for alert conditions."""
        current_metrics = self.get_current_metrics()
        current_time = datetime.now()
        
        for metric, thresholds in self.alert_thresholds.items():
            if metric not in current_metrics:
                continue
            
            current_value = current_metrics[metric]
            
            for severity, threshold in thresholds.items():
                alert_id = f"{metric}_{severity}"
                
                # Check cooldown
                if alert_id in self.last_alert_times:
                    time_since_last = (current_time - self.last_alert_times[alert_id]).total_seconds()
                    if time_since_last < self.alert_cooldown:
                        continue
                
                # Check threshold condition
                condition_met = False
                if metric in ['avg_response_time_ms', 'p95_response_time_ms', 'error_rate']:
                    condition_met = current_value > threshold
                elif metric in ['cache_hit_rate', 'throughput_per_second']:
                    condition_met = current_value < threshold
                
                if condition_met:
                    self._create_alert(alert_id, severity, metric, threshold, current_value)
                elif alert_id in self.active_alerts:
                    # Clear resolved alert
                    self._clear_alert(alert_id)
    
    def _create_alert(
        self,
        alert_id: str,
        severity: str,
        metric: str,
        threshold: float,
        current_value: float
    ):
        """Create a new alert."""
        message = f"{metric} {severity}: {current_value:.2f} (threshold: {threshold:.2f})"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            severity=severity,
            metric=metric,
            threshold=threshold,
            current_value=current_value,
            message=message
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_id] = datetime.now()
        
        logger.warning(f"Performance alert: {message}")
    
    def _clear_alert(self, alert_id: str):
        """Clear a resolved alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            del self.active_alerts[alert_id]
            
            logger.info(f"Performance alert cleared: {alert_id}")
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history."""
        recent_alerts = list(self.alert_history)[-limit:]
        return [alert.to_dict() for alert in recent_alerts]
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        current_metrics = self.get_current_metrics()
        
        # Historical data (last hour)
        hour_ago = datetime.now() - timedelta(hours=1)
        historical_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > hour_ago
        ]
        
        return {
            'current_metrics': current_metrics,
            'historical_metrics': historical_metrics[-60:],  # Last 60 samples
            'active_alerts': self.get_active_alerts(),
            'recent_alert_history': self.get_alert_history(10),
            'alert_thresholds': self.alert_thresholds,
            'system_status': self._get_system_status(),
            'performance_trends': self._calculate_trends()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        metrics = self.real_time_metrics
        
        # Determine overall health
        health_score = 1.0
        
        if metrics.get('error_rate', 0) > 0.1:
            health_score *= 0.5
        elif metrics.get('error_rate', 0) > 0.05:
            health_score *= 0.8
        
        if metrics.get('avg_response_time_ms', 0) > 500:
            health_score *= 0.5
        elif metrics.get('avg_response_time_ms', 0) > 200:
            health_score *= 0.8
        
        if metrics.get('cache_hit_rate', 1.0) < 0.3:
            health_score *= 0.7
        elif metrics.get('cache_hit_rate', 1.0) < 0.5:
            health_score *= 0.9
        
        status = 'healthy'
        if health_score < 0.5:
            status = 'critical'
        elif health_score < 0.8:
            status = 'degraded'
        elif len(self.active_alerts) > 0:
            status = 'warning'
        
        return {
            'status': status,
            'health_score': health_score,
            'active_alert_count': len(self.active_alerts),
            'uptime_hours': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
        }
    
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate performance trends."""
        trends = {}
        
        if len(self.metrics_history) < 2:
            return trends
        
        # Compare recent vs older metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        older_metrics = list(self.metrics_history)[-20:-10]  # Previous 10 samples
        
        if not older_metrics:
            return trends
        
        metrics_to_trend = ['avg_response_time_ms', 'error_rate', 'cache_hit_rate', 'throughput_per_second']
        
        for metric in metrics_to_trend:
            recent_avg = statistics.mean([m.get(metric, 0) for m in recent_metrics])
            older_avg = statistics.mean([m.get(metric, 0) for m in older_metrics])
            
            if older_avg == 0:
                trends[metric] = 'stable'
                continue
            
            change_percent = ((recent_avg - older_avg) / older_avg) * 100
            
            if abs(change_percent) < 5:
                trends[metric] = 'stable'
            elif change_percent > 0:
                if metric in ['avg_response_time_ms', 'error_rate']:
                    trends[metric] = 'worsening'
                else:
                    trends[metric] = 'improving'
            else:
                if metric in ['avg_response_time_ms', 'error_rate']:
                    trends[metric] = 'improving'
                else:
                    trends[metric] = 'worsening'
        
        return trends
    
    def _monitor_worker(self):
        """Background monitoring worker."""
        while self._running:
            try:
                # Collect current metrics
                current_metrics = self.get_current_metrics()
                self.metrics_history.append(current_metrics)
                
                # Check for alerts
                self.check_alerts()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitor worker error: {e}")
                time.sleep(60)
    
    def export_metrics(self, format: str = 'json') -> Any:
        """Export metrics in specified format."""
        dashboard_data = self.get_performance_dashboard()
        
        if format == 'json':
            return dashboard_data
        elif format == 'prometheus':
            # Convert to Prometheus format
            return self._to_prometheus_format(dashboard_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _to_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Convert metrics to Prometheus format."""
        metrics = data['current_metrics']
        lines = []
        
        lines.append(f"# HELP transaction_cost_calculations_total Total number of calculations")
        lines.append(f"# TYPE transaction_cost_calculations_total counter")
        lines.append(f"transaction_cost_calculations_total {metrics['counters']['total_calculations']}")
        
        lines.append(f"# HELP transaction_cost_response_time_ms Average response time in milliseconds")
        lines.append(f"# TYPE transaction_cost_response_time_ms gauge")
        lines.append(f"transaction_cost_response_time_ms {metrics['avg_response_time_ms']}")
        
        lines.append(f"# HELP transaction_cost_error_rate Error rate")
        lines.append(f"# TYPE transaction_cost_error_rate gauge")
        lines.append(f"transaction_cost_error_rate {metrics['error_rate']}")
        
        lines.append(f"# HELP transaction_cost_cache_hit_rate Cache hit rate")
        lines.append(f"# TYPE transaction_cost_cache_hit_rate gauge")
        lines.append(f"transaction_cost_cache_hit_rate {metrics['cache_hit_rate']}")
        
        return '\n'.join(lines)
    
    def shutdown(self):
        """Shutdown the monitor."""
        self._running = False
        
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Performance monitor shutdown completed")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except:
            pass


logger.info("Performance monitor module loaded successfully")