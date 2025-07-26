#!/usr/bin/env python3
"""
🛡️ API COMPLIANCE AND RATE LIMITING SYSTEM
Comprehensive compliance manager for all data provider APIs

This module ensures compliance with data provider terms of service,
implements rate limiting, request throttling, and usage monitoring.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import hashlib
import pickle
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.app_config import Config

# Setup logging
logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Supported data providers"""
    BREEZE_CONNECT = "breeze_connect"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    CUSTOM_API = "custom_api"

class ComplianceLevel(Enum):
    """Compliance enforcement levels"""
    STRICT = "strict"          # Enforce all limits strictly
    MODERATE = "moderate"      # Allow some flexibility
    LENIENT = "lenient"        # Minimal enforcement
    MONITORING = "monitoring"  # Monitor only, no enforcement

@dataclass
class RateLimitConfig:
    """Rate limiting configuration for a data provider"""
    provider: DataProvider
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_capacity: int = 10
    cooldown_period: float = 1.0
    max_concurrent: int = 5
    
    # Compliance settings
    respect_server_limits: bool = True
    adaptive_backoff: bool = True
    error_threshold: float = 0.1  # Max error rate before throttling
    
    # Terms of service compliance
    commercial_use_allowed: bool = False
    attribution_required: bool = True
    data_redistribution_allowed: bool = False
    cache_duration_hours: int = 24

@dataclass
class APIRequest:
    """Represents an API request for tracking"""
    provider: DataProvider
    endpoint: str
    params: Dict[str, Any]
    timestamp: datetime
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    size_estimate: int = 0
    priority: int = 1  # 1=high, 5=low

@dataclass
class APIResponse:
    """Represents an API response for tracking"""
    request: APIRequest
    success: bool
    response_time: float
    data_size: int
    error_message: Optional[str] = None
    rate_limited: bool = False
    cached: bool = False

class ComplianceManager:
    """
    Comprehensive API compliance and rate limiting manager.
    
    Features:
    - Multi-provider rate limiting
    - Terms of service compliance
    - Request monitoring and analytics
    - Adaptive throttling
    - Usage reporting
    """
    
    def __init__(self, config_file: Optional[str] = None, compliance_level: ComplianceLevel = ComplianceLevel.MODERATE):
        """Initialize compliance manager"""
        self.compliance_level = compliance_level
        self.config = Config()
        
        # Initialize tracking
        self.request_history: Dict[DataProvider, List[APIRequest]] = {provider: [] for provider in DataProvider}
        self.response_history: List[APIResponse] = []
        self.error_counts: Dict[DataProvider, int] = {provider: 0 for provider in DataProvider}
        self.last_request_time: Dict[DataProvider, float] = {provider: 0.0 for provider in DataProvider}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load configurations
        self.rate_configs = self._load_rate_configs(config_file)
        
        # Initialize monitoring
        self.start_time = datetime.now()
        self.monitoring_enabled = True
        
        # Cache for responses
        self.response_cache: Dict[str, Any] = {}
        self.cache_file = os.path.join(self.config.get_data_save_path(), "api_response_cache.pkl")
        self._load_cache()
        
        logger.info(f"✅ API Compliance Manager initialized with {compliance_level.value} level")
        
    def _load_rate_configs(self, config_file: Optional[str] = None) -> Dict[DataProvider, RateLimitConfig]:
        """Load rate limiting configurations for all providers"""
        
        # Default configurations based on known provider limits
        default_configs = {
            DataProvider.BREEZE_CONNECT: RateLimitConfig(
                provider=DataProvider.BREEZE_CONNECT,
                requests_per_second=2.0,  # Conservative estimate
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=5000,
                burst_capacity=5,
                cooldown_period=0.5,
                max_concurrent=3,
                commercial_use_allowed=True,  # With proper license
                attribution_required=False,
                data_redistribution_allowed=False,
                cache_duration_hours=1  # Real-time data
            ),
            
            DataProvider.YAHOO_FINANCE: RateLimitConfig(
                provider=DataProvider.YAHOO_FINANCE,
                requests_per_second=1.0,  # Conservative for free tier
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=2000,
                burst_capacity=5,
                cooldown_period=1.0,
                max_concurrent=2,
                commercial_use_allowed=False,  # Terms prohibit commercial use
                attribution_required=True,
                data_redistribution_allowed=False,
                cache_duration_hours=24  # Daily data
            ),
            
            DataProvider.ALPHA_VANTAGE: RateLimitConfig(
                provider=DataProvider.ALPHA_VANTAGE,
                requests_per_second=0.2,  # 5 requests per minute for free tier
                requests_per_minute=5,
                requests_per_hour=250,
                requests_per_day=500,
                burst_capacity=1,
                cooldown_period=12.0,  # 12 seconds between requests
                max_concurrent=1,
                commercial_use_allowed=True,  # With paid plan
                attribution_required=True,
                data_redistribution_allowed=False,
                cache_duration_hours=1
            ),
            
            DataProvider.CUSTOM_API: RateLimitConfig(
                provider=DataProvider.CUSTOM_API,
                requests_per_second=1.0,
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=5000,
                burst_capacity=10,
                cooldown_period=1.0,
                max_concurrent=5,
                commercial_use_allowed=True,
                attribution_required=False,
                data_redistribution_allowed=True,
                cache_duration_hours=6
            )
        }
        
        # Load custom configurations if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_configs = json.load(f)
                    
                for provider_name, config_data in custom_configs.items():
                    if provider_name in [p.value for p in DataProvider]:
                        provider = DataProvider(provider_name)
                        # Update default config with custom values
                        config_dict = default_configs[provider].__dict__.copy()
                        config_dict.update(config_data)
                        default_configs[provider] = RateLimitConfig(**config_dict)
                        
                logger.info(f"✅ Loaded custom rate configurations from {config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load custom config: {e}, using defaults")
        
        return default_configs
    
    def _load_cache(self):
        """Load response cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Filter expired entries
                current_time = datetime.now()
                self.response_cache = {
                    key: value for key, value in cache_data.items()
                    if (current_time - value.get('timestamp', current_time)).total_seconds() < 86400  # 24 hours
                }
                
                logger.info(f"✅ Loaded {len(self.response_cache)} cached responses")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {e}")
            self.response_cache = {}
    
    def _save_cache(self):
        """Save response cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.response_cache, f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache: {e}")
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'provider': request.provider.value,
            'endpoint': request.endpoint,
            'params': sorted(request.params.items()) if request.params else []
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any], provider: DataProvider) -> bool:
        """Check if cached response is still valid"""
        config = self.rate_configs[provider]
        timestamp = cache_entry.get('timestamp', datetime.min)
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return age_hours < config.cache_duration_hours
    
    def check_rate_limit(self, provider: DataProvider, endpoint: str = "", 
                        priority: int = 1) -> Dict[str, Any]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            provider: Data provider
            endpoint: API endpoint
            priority: Request priority (1=high, 5=low)
            
        Returns:
            Dict with allowed status and wait time
        """
        with self._lock:
            config = self.rate_configs[provider]
            current_time = time.time()
            
            # Get recent requests
            recent_requests = [
                req for req in self.request_history[provider]
                if (current_time - req.timestamp.timestamp()) < 3600  # Last hour
            ]
            
            # Check various time windows
            checks = []
            
            # Requests per second
            second_requests = [
                req for req in recent_requests
                if (current_time - req.timestamp.timestamp()) < 1
            ]
            if len(second_requests) >= config.requests_per_second:
                wait_time = 1 - (current_time - min(req.timestamp.timestamp() for req in second_requests))
                checks.append(('per_second', False, max(0, wait_time)))
            else:
                checks.append(('per_second', True, 0))
            
            # Requests per minute
            minute_requests = [
                req for req in recent_requests
                if (current_time - req.timestamp.timestamp()) < 60
            ]
            if len(minute_requests) >= config.requests_per_minute:
                wait_time = 60 - (current_time - min(req.timestamp.timestamp() for req in minute_requests))
                checks.append(('per_minute', False, max(0, wait_time)))
            else:
                checks.append(('per_minute', True, 0))
            
            # Requests per hour
            if len(recent_requests) >= config.requests_per_hour:
                wait_time = 3600 - (current_time - min(req.timestamp.timestamp() for req in recent_requests))
                checks.append(('per_hour', False, max(0, wait_time)))
            else:
                checks.append(('per_hour', True, 0))
            
            # Check minimum time between requests
            last_request_time = self.last_request_time[provider]
            time_since_last = current_time - last_request_time
            if time_since_last < config.cooldown_period:
                wait_time = config.cooldown_period - time_since_last
                checks.append(('cooldown', False, wait_time))
            else:
                checks.append(('cooldown', True, 0))
            
            # Overall result
            allowed = all(check[1] for check in checks)
            max_wait_time = max(check[2] for check in checks) if not allowed else 0
            
            # Apply compliance level adjustments
            if self.compliance_level == ComplianceLevel.LENIENT:
                # Allow some violations for lenient mode
                if max_wait_time < 5:  # Allow if wait is less than 5 seconds
                    allowed = True
                    max_wait_time = 0
            elif self.compliance_level == ComplianceLevel.MONITORING:
                # Always allow but log violations
                if not allowed:
                    logger.warning(f"🚨 Rate limit violation for {provider.value}: {checks}")
                allowed = True
                max_wait_time = 0
            
            result = {
                'allowed': allowed,
                'wait_time': max_wait_time,
                'checks': checks,
                'provider': provider.value,
                'endpoint': endpoint,
                'request_count': len(recent_requests),
                'compliance_level': self.compliance_level.value
            }
            
            if not allowed:
                logger.warning(f"🚫 Rate limit exceeded for {provider.value}: wait {max_wait_time:.2f}s")
            
            return result
    
    def request_permission(self, provider: DataProvider, endpoint: str, 
                          params: Dict[str, Any] = None, priority: int = 1) -> APIRequest:
        """
        Request permission to make an API call.
        
        Args:
            provider: Data provider
            endpoint: API endpoint
            params: Request parameters
            priority: Request priority
            
        Returns:
            APIRequest object if allowed
            
        Raises:
            RateLimitError: If request is not allowed
        """
        if params is None:
            params = {}
            
        # Create request object
        request = APIRequest(
            provider=provider,
            endpoint=endpoint,
            params=params,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if self._is_cache_valid(cache_entry, provider):
                logger.debug(f"📦 Cache hit for {provider.value}:{endpoint}")
                return request  # Return request but mark as cached
        
        # Check rate limits
        rate_check = self.check_rate_limit(provider, endpoint, priority)
        
        if not rate_check['allowed']:
            wait_time = rate_check['wait_time']
            
            if self.compliance_level == ComplianceLevel.STRICT:
                raise RateLimitError(
                    f"Rate limit exceeded for {provider.value}. Wait {wait_time:.2f} seconds."
                )
            elif wait_time > 0:
                logger.info(f"⏳ Rate limiting: waiting {wait_time:.2f}s for {provider.value}")
                time.sleep(wait_time)
        
        # Record request
        with self._lock:
            self.request_history[provider].append(request)
            self.last_request_time[provider] = time.time()
            
            # Cleanup old requests (keep last 1000 per provider)
            if len(self.request_history[provider]) > 1000:
                self.request_history[provider] = self.request_history[provider][-1000:]
        
        return request
    
    def record_response(self, request: APIRequest, success: bool, 
                       response_time: float, data_size: int = 0, 
                       error_message: Optional[str] = None, 
                       rate_limited: bool = False,
                       response_data: Any = None) -> APIResponse:
        """
        Record API response for monitoring and caching.
        
        Args:
            request: Original request object
            success: Whether request was successful
            response_time: Response time in seconds
            data_size: Size of response data in bytes
            error_message: Error message if failed
            rate_limited: Whether response was rate limited
            response_data: Actual response data for caching
            
        Returns:
            APIResponse object
        """
        response = APIResponse(
            request=request,
            success=success,
            response_time=response_time,
            data_size=data_size,
            error_message=error_message,
            rate_limited=rate_limited
        )
        
        with self._lock:
            self.response_history.append(response)
            
            # Update error counts
            if not success:
                self.error_counts[request.provider] += 1
            
            # Cache successful responses
            if success and response_data is not None:
                config = self.rate_configs[request.provider]
                if config.cache_duration_hours > 0:
                    cache_key = self._generate_cache_key(request)
                    self.response_cache[cache_key] = {
                        'data': response_data,
                        'timestamp': datetime.now(),
                        'response_time': response_time
                    }
                    
                    # Periodically save cache
                    if len(self.response_cache) % 10 == 0:
                        self._save_cache()
            
            # Cleanup old responses (keep last 1000)
            if len(self.response_history) > 1000:
                self.response_history = self.response_history[-1000:]
        
        # Log significant events
        if rate_limited:
            logger.warning(f"🚨 Rate limited response from {request.provider.value}")
        elif not success:
            logger.error(f"❌ Failed request to {request.provider.value}: {error_message}")
        
        return response
    
    def get_cached_response(self, request: APIRequest) -> Optional[Any]:
        """Get cached response for request if available and valid"""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if self._is_cache_valid(cache_entry, request.provider):
                logger.debug(f"📦 Returning cached response for {request.provider.value}")
                return cache_entry['data']
        
        return None
    
    def compliance_decorator(self, provider: DataProvider, endpoint: str = ""):
        """
        Decorator to enforce compliance on API functions.
        
        Usage:
            @compliance_manager.compliance_decorator(DataProvider.YAHOO_FINANCE)
            def fetch_data(symbol, start_date, end_date):
                # Your API call here
                return data
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract parameters for request tracking
                params = {
                    'args': str(args)[:100],  # Truncated for brevity
                    'kwargs': str(kwargs)[:100]
                }
                
                try:
                    # Request permission
                    request = self.request_permission(
                        provider=provider,
                        endpoint=endpoint or func.__name__,
                        params=params
                    )
                    
                    # Check for cached response
                    cached_data = self.get_cached_response(request)
                    if cached_data is not None:
                        return cached_data
                    
                    # Make actual API call
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    response_time = time.time() - start_time
                    
                    # Record successful response
                    self.record_response(
                        request=request,
                        success=True,
                        response_time=response_time,
                        data_size=len(str(result)) if result else 0,
                        response_data=result
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record failed response
                    if 'request' in locals():
                        self.record_response(
                            request=request,
                            success=False,
                            response_time=time.time() - start_time if 'start_time' in locals() else 0,
                            error_message=str(e),
                            rate_limited='rate' in str(e).lower()
                        )
                    raise
            
            return wrapper
        return decorator
    
    def get_usage_statistics(self, provider: Optional[DataProvider] = None) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        with self._lock:
            if provider:
                providers = [provider]
            else:
                providers = list(DataProvider)
            
            stats = {}
            
            for prov in providers:
                requests = self.request_history[prov]
                responses = [r for r in self.response_history if r.request.provider == prov]
                
                if not requests:
                    stats[prov.value] = {'message': 'No requests made'}
                    continue
                
                # Calculate statistics
                total_requests = len(requests)
                successful_requests = len([r for r in responses if r.success])
                failed_requests = len([r for r in responses if not r.success])
                avg_response_time = sum(r.response_time for r in responses) / len(responses) if responses else 0
                total_data_size = sum(r.data_size for r in responses)
                
                # Time-based statistics
                now = datetime.now()
                last_hour_requests = len([
                    r for r in requests
                    if (now - r.timestamp).total_seconds() < 3600
                ])
                last_day_requests = len([
                    r for r in requests
                    if (now - r.timestamp).total_seconds() < 86400
                ])
                
                # Rate limit compliance
                config = self.rate_configs[prov]
                compliance_score = min(100, max(0, 100 - (failed_requests / max(total_requests, 1)) * 100))
                
                stats[prov.value] = {
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'success_rate': successful_requests / max(total_requests, 1) * 100,
                    'avg_response_time': avg_response_time,
                    'total_data_size': total_data_size,
                    'last_hour_requests': last_hour_requests,
                    'last_day_requests': last_day_requests,
                    'compliance_score': compliance_score,
                    'rate_limits': {
                        'per_second': config.requests_per_second,
                        'per_minute': config.requests_per_minute,
                        'per_hour': config.requests_per_hour,
                        'per_day': config.requests_per_day
                    },
                    'terms_compliance': {
                        'commercial_use_allowed': config.commercial_use_allowed,
                        'attribution_required': config.attribution_required,
                        'data_redistribution_allowed': config.data_redistribution_allowed
                    }
                }
            
            return stats
    
    def generate_compliance_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive compliance report"""
        stats = self.get_usage_statistics()
        
        report_lines = [
            "🛡️ API COMPLIANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Compliance Level: {self.compliance_level.value}",
            f"Monitoring Duration: {datetime.now() - self.start_time}",
            "",
            "📊 USAGE STATISTICS:",
            "-" * 40
        ]
        
        for provider, data in stats.items():
            if 'message' in data:
                report_lines.extend([
                    f"\n{provider.upper()}:",
                    f"  {data['message']}"
                ])
                continue
            
            report_lines.extend([
                f"\n{provider.upper()}:",
                f"  Total Requests: {data['total_requests']}",
                f"  Success Rate: {data['success_rate']:.1f}%",
                f"  Avg Response Time: {data['avg_response_time']:.3f}s",
                f"  Data Transfer: {data['total_data_size']:,} bytes",
                f"  Last Hour Requests: {data['last_hour_requests']}",
                f"  Last Day Requests: {data['last_day_requests']}",
                f"  Compliance Score: {data['compliance_score']:.1f}%",
                "",
                f"  Rate Limits:",
                f"    Per Second: {data['rate_limits']['per_second']}",
                f"    Per Minute: {data['rate_limits']['per_minute']}",
                f"    Per Hour: {data['rate_limits']['per_hour']}",
                f"    Per Day: {data['rate_limits']['per_day']}",
                "",
                f"  Terms Compliance:",
                f"    Commercial Use: {'✅' if data['terms_compliance']['commercial_use_allowed'] else '❌'}",
                f"    Attribution Required: {'✅' if data['terms_compliance']['attribution_required'] else '❌'}",
                f"    Redistribution Allowed: {'✅' if data['terms_compliance']['data_redistribution_allowed'] else '❌'}"
            ])
        
        report_lines.extend([
            "",
            "🔍 COMPLIANCE RECOMMENDATIONS:",
            "-" * 40
        ])
        
        # Add recommendations based on usage patterns
        for provider, data in stats.items():
            if 'message' in data:
                continue
                
            if data['success_rate'] < 90:
                report_lines.append(f"⚠️  {provider}: Low success rate, consider reducing request frequency")
            
            if data['compliance_score'] < 80:
                report_lines.append(f"🚨 {provider}: Poor compliance, review rate limiting strategy")
            
            if data['last_hour_requests'] > data['rate_limits']['per_hour'] * 0.8:
                report_lines.append(f"⚠️  {provider}: Approaching hourly rate limit")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_content)
                logger.info(f"✅ Compliance report saved to {output_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save compliance report: {e}")
        
        return report_content
    
    def cleanup(self):
        """Cleanup resources and save state"""
        logger.info("🧹 Cleaning up compliance manager...")
        self._save_cache()
        logger.info("✅ Compliance manager cleanup complete")

class RateLimitError(Exception):
    """Raised when rate limit is exceeded"""
    pass

# Global compliance manager instance
_compliance_manager = None

def get_compliance_manager(compliance_level: ComplianceLevel = ComplianceLevel.MODERATE) -> ComplianceManager:
    """Get global compliance manager instance"""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager(compliance_level=compliance_level)
    return _compliance_manager

def compliance_decorator(provider: DataProvider, endpoint: str = ""):
    """Convenience decorator for API compliance"""
    manager = get_compliance_manager()
    return manager.compliance_decorator(provider, endpoint)

def check_terms_compliance(provider: DataProvider) -> Dict[str, bool]:
    """Check terms of service compliance for a provider"""
    manager = get_compliance_manager()
    config = manager.rate_configs[provider]
    
    return {
        'commercial_use_allowed': config.commercial_use_allowed,
        'attribution_required': config.attribution_required,
        'data_redistribution_allowed': config.data_redistribution_allowed,
        'rate_limits_configured': True,
        'monitoring_enabled': manager.monitoring_enabled
    }

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing API Compliance Manager...")
    
    # Initialize manager
    manager = ComplianceManager(compliance_level=ComplianceLevel.MODERATE)
    
    # Test rate limiting
    try:
        request = manager.request_permission(
            provider=DataProvider.YAHOO_FINANCE,
            endpoint="download",
            params={'symbol': 'AAPL', 'period': '1d'}
        )
        print(f"✅ Request approved: {request.request_id}")
        
        # Simulate response
        manager.record_response(
            request=request,
            success=True,
            response_time=0.5,
            data_size=1024
        )
        
    except RateLimitError as e:
        print(f"🚫 Rate limited: {e}")
    
    # Generate report
    report = manager.generate_compliance_report()
    print("\n" + report)
    
    # Cleanup
    manager.cleanup()
    print("🎉 Compliance manager test complete!")
