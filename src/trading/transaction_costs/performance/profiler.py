"""
Calculation Profiler
===================

Profiles transaction cost calculations to identify performance bottlenecks
and optimization opportunities.
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)


@dataclass
class ProfileData:
    """Profile data for a calculation step."""
    step_name: str
    execution_time: float
    memory_usage: Optional[int] = None
    cpu_time: Optional[float] = None
    call_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_name': self.step_name,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_time': self.cpu_time,
            'call_count': self.call_count,
            'timestamp': self.timestamp.isoformat()
        }


class CalculationProfiler:
    """
    Profiles calculation performance to identify bottlenecks.
    
    Features:
    - Step-by-step timing
    - Memory usage tracking
    - Call frequency analysis
    - Bottleneck identification
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiles: Dict[str, List[ProfileData]] = defaultdict(list)
        self.active_profiles: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
    def start_profile(self, profile_id: str) -> str:
        """Start profiling a calculation."""
        if not self.enabled:
            return profile_id
            
        with self._lock:
            self.active_profiles[profile_id] = {
                'start_time': time.perf_counter(),
                'steps': [],
                'step_stack': []
            }
            
        return profile_id
    
    def step(self, profile_id: str, step_name: str):
        """Mark a profiling step."""
        if not self.enabled or profile_id not in self.active_profiles:
            return
            
        current_time = time.perf_counter()
        
        with self._lock:
            profile = self.active_profiles[profile_id]
            
            # End previous step if exists
            if profile['step_stack']:
                prev_step = profile['step_stack'][-1]
                step_time = current_time - prev_step['start_time']
                
                profile_data = ProfileData(
                    step_name=prev_step['name'],
                    execution_time=step_time
                )
                profile['steps'].append(profile_data)
            
            # Start new step
            profile['step_stack'].append({
                'name': step_name,
                'start_time': current_time
            })
    
    def end_profile(self, profile_id: str):
        """End profiling and store results."""
        if not self.enabled or profile_id not in self.active_profiles:
            return
            
        current_time = time.perf_counter()
        
        with self._lock:
            profile = self.active_profiles[profile_id]
            
            # End any remaining step
            if profile['step_stack']:
                step = profile['step_stack'][-1]
                step_time = current_time - step['start_time']
                
                profile_data = ProfileData(
                    step_name=step['name'],
                    execution_time=step_time
                )
                profile['steps'].append(profile_data)
            
            # Store profile results
            for step_data in profile['steps']:
                self.profiles[step_data.step_name].append(step_data)
            
            # Clean up
            del self.active_profiles[profile_id]
    
    def profile_function(self, step_name: Optional[str] = None):
        """Decorator to profile a function."""
        def decorator(func: Callable):
            nonlocal step_name
            if step_name is None:
                step_name = func.__name__
                
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.perf_counter() - start_time
                    profile_data = ProfileData(
                        step_name=step_name,
                        execution_time=execution_time
                    )
                    
                    with self._lock:
                        self.profiles[step_name].append(profile_data)
            
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get profiling statistics."""
        stats = {}
        
        with self._lock:
            for step_name, profile_list in self.profiles.items():
                if not profile_list:
                    continue
                    
                times = [p.execution_time for p in profile_list]
                
                stats[step_name] = {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'last_call': profile_list[-1].timestamp.isoformat()
                }
        
        return stats
    
    def get_bottlenecks(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        stats = self.get_statistics()
        
        for step_name, step_stats in stats.items():
            avg_time_ms = step_stats['avg_time'] * 1000
            
            if avg_time_ms > threshold_ms:
                bottlenecks.append({
                    'step_name': step_name,
                    'avg_time_ms': avg_time_ms,
                    'call_count': step_stats['call_count'],
                    'total_impact_ms': step_stats['total_time'] * 1000,
                    'severity': 'high' if avg_time_ms > 500 else 'medium'
                })
        
        # Sort by total impact
        bottlenecks.sort(key=lambda x: x['total_impact_ms'], reverse=True)
        return bottlenecks
    
    def clear_profiles(self):
        """Clear all profile data."""
        with self._lock:
            self.profiles.clear()
            self.active_profiles.clear()
        
        logger.info("Cleared profiling data")


logger.info("Calculation profiler module loaded successfully")