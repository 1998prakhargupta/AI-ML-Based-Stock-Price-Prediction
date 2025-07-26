#!/usr/bin/env python3
"""
🛡️ ENHANCED BREEZE UTILITIES WITH API COMPLIANCE
Breeze API utilities with comprehensive compliance, rate limiting, and monitoring

This module extends the existing Breeze functionality with strict compliance
to API terms of service, rate limiting, and usage monitoring.
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import existing modules
from enhanced_breeze_utils import EnhancedBreezeDataManager, APIResponse, MarketDataRequest
from api_compliance import (
    ComplianceManager, DataProvider, ComplianceLevel, 
    compliance_decorator, get_compliance_manager, check_terms_compliance
)
from app_config import Config
from file_management_utils import SafeFileManager, SaveStrategy

logger = logging.getLogger(__name__)

class ComplianceBreezeDataManager(EnhancedBreezeDataManager):
    """
    Enhanced Breeze Data Manager with comprehensive API compliance.
    
    Features:
    - Strict rate limiting based on Breeze API terms
    - Request monitoring and analytics
    - Terms of service compliance checking
    - Automatic throttling and backoff
    - Usage reporting and documentation
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 compliance_level: ComplianceLevel = ComplianceLevel.MODERATE):
        """
        Initialize compliance-aware Breeze data manager.
        
        Args:
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            compliance_level: Level of compliance enforcement
        """
        # Initialize parent class
        super().__init__(max_retries, retry_delay)
        
        # Initialize compliance manager
        self.compliance_manager = get_compliance_manager(compliance_level)
        self.compliance_level = compliance_level
        
        # Breeze-specific compliance settings
        self.breeze_limits = {
            'requests_per_second': 2.0,  # Conservative estimate
            'requests_per_minute': 100,
            'requests_per_hour': 1000,
            'daily_request_limit': 5000,
            'max_concurrent_requests': 3,
            'min_request_interval': 0.5,  # 500ms between requests
            'burst_allowance': 5,
            'cooldown_after_error': 2.0,  # 2 seconds after any error
            'max_data_points_per_request': 1000,
            'max_symbols_per_request': 10
        }
        
        # Terms of service compliance
        self.terms_compliance = {
            'commercial_use_allowed': True,  # With proper licensing
            'data_redistribution_prohibited': True,
            'attribution_required': False,
            'real_time_data_restrictions': True,
            'bulk_download_limitations': True,
            'api_abuse_prevention': True
        }
        
        # Enhanced monitoring
        self.request_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'cached_responses': 0,
            'total_data_points': 0,
            'session_start_time': datetime.now()
        }
        
        self.logger.info("✅ Compliance-aware Breeze Data Manager initialized")
        self._validate_terms_compliance()
    
    def _validate_terms_compliance(self):
        """Validate compliance with Breeze API terms of service"""
        compliance_check = check_terms_compliance(DataProvider.BREEZE_CONNECT)
        
        self.logger.info("🔍 Breeze API Terms of Service Compliance Check:")
        self.logger.info(f"  Commercial Use: {'✅' if compliance_check['commercial_use_allowed'] else '❌'}")
        self.logger.info(f"  Attribution Required: {'✅' if compliance_check['attribution_required'] else 'ℹ️ Not Required'}")
        self.logger.info(f"  Redistribution: {'❌ Prohibited' if not compliance_check['data_redistribution_allowed'] else '✅ Allowed'}")
        self.logger.info(f"  Rate Limiting: {'✅' if compliance_check['rate_limits_configured'] else '❌'}")
        self.logger.info(f"  Monitoring: {'✅' if compliance_check['monitoring_enabled'] else '❌'}")
        
        # Log important restrictions
        if self.terms_compliance['data_redistribution_prohibited']:
            self.logger.warning("⚠️  DATA REDISTRIBUTION PROHIBITED: Data cannot be shared with third parties")
        
        if self.terms_compliance['real_time_data_restrictions']:
            self.logger.info("ℹ️  Real-time data subject to usage restrictions")
        
        if self.terms_compliance['bulk_download_limitations']:
            self.logger.info("ℹ️  Bulk data downloads subject to rate limiting")
    
    @compliance_decorator(DataProvider.BREEZE_CONNECT, "authenticate")
    def authenticate(self) -> bool:
        """
        Authenticate with Breeze API with compliance monitoring.
        
        Returns:
            bool: True if authentication successful
        """
        try:
            self.logger.info("🔐 Authenticating with Breeze API (compliance monitored)...")
            
            # Use parent authentication with compliance tracking
            success = super().authenticate()
            
            if success:
                self.logger.info("✅ Breeze API authentication successful")
                self._log_compliance_status()
            else:
                self.logger.error("❌ Breeze API authentication failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Authentication error: {str(e)}")
            return False
    
    def _log_compliance_status(self):
        """Log current compliance status"""
        stats = self.compliance_manager.get_usage_statistics(DataProvider.BREEZE_CONNECT)
        breeze_stats = stats.get('breeze_connect', {})
        
        if 'message' not in breeze_stats:
            self.logger.info(f"📊 Current session stats:")
            self.logger.info(f"  Requests made: {breeze_stats.get('total_requests', 0)}")
            self.logger.info(f"  Success rate: {breeze_stats.get('success_rate', 0):.1f}%")
            self.logger.info(f"  Compliance score: {breeze_stats.get('compliance_score', 0):.1f}%")
    
    @compliance_decorator(DataProvider.BREEZE_CONNECT, "get_quotes")
    def get_quotes_safe(self, stock_code: str, exchange_code: str) -> APIResponse:
        """
        Get live quotes with compliance monitoring.
        
        Args:
            stock_code: Stock symbol
            exchange_code: Exchange code
            
        Returns:
            APIResponse with quote data
        """
        self.logger.debug(f"📈 Fetching quote for {stock_code} on {exchange_code}")
        
        try:
            # Validate request parameters
            self._validate_quote_request(stock_code, exchange_code)
            
            # Check compliance before making request
            self._pre_request_compliance_check('get_quotes', {
                'stock_code': stock_code,
                'exchange_code': exchange_code
            })
            
            # Make the actual request using parent method
            response = super().get_quotes_safe(stock_code, exchange_code)
            
            # Update metrics
            self._update_request_metrics(response.success, response.data)
            
            # Log compliance information
            if response.success:
                self.logger.debug(f"✅ Quote fetched successfully for {stock_code}")
            else:
                self.logger.warning(f"⚠️ Quote fetch failed for {stock_code}: {response.errors}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching quotes: {str(e)}")
            self.request_metrics['failed_requests'] += 1
            return APIResponse(
                success=False,
                data=None,
                errors=[str(e)],
                warnings=[],
                metadata={'compliance_error': True},
                response_time=0
            )
    
    def _validate_quote_request(self, stock_code: str, exchange_code: str):
        """Validate quote request parameters"""
        if not stock_code or len(stock_code.strip()) == 0:
            raise ValueError("Stock code cannot be empty")
        
        if not exchange_code or len(exchange_code.strip()) == 0:
            raise ValueError("Exchange code cannot be empty")
        
        # Check for valid exchange codes
        valid_exchanges = ['NSE', 'BSE', 'NFO', 'BFO', 'MCX']
        if exchange_code.upper() not in valid_exchanges:
            self.logger.warning(f"⚠️ Unusual exchange code: {exchange_code}")
    
    @compliance_decorator(DataProvider.BREEZE_CONNECT, "get_historical_data")
    def get_historical_data_safe(self, request: MarketDataRequest) -> APIResponse:
        """
        Get historical data with comprehensive compliance monitoring.
        
        Args:
            request: Market data request object
            
        Returns:
            APIResponse with historical data
        """
        self.logger.debug(f"📊 Fetching historical data: {request.stock_code} ({request.product_type})")
        
        try:
            # Enhanced request validation
            self._validate_historical_request(request)
            
            # Check compliance before making request
            self._pre_request_compliance_check('get_historical_data', {
                'stock_code': request.stock_code,
                'product_type': request.product_type,
                'interval': request.interval,
                'from_date': request.from_date,
                'to_date': request.to_date
            })
            
            # Estimate data points to check against limits
            estimated_points = self._estimate_data_points(request)
            if estimated_points > self.breeze_limits['max_data_points_per_request']:
                self.logger.warning(f"⚠️ Large data request: ~{estimated_points} points")
                # Consider splitting the request
                return self._handle_large_data_request(request)
            
            # Make the actual request using parent method
            response = super().get_historical_data_safe(request)
            
            # Update metrics with actual data
            self._update_request_metrics(response.success, response.data)
            
            # Enhanced response logging
            if response.success and response.data is not None:
                data_points = len(response.data)
                self.request_metrics['total_data_points'] += data_points
                self.logger.info(f"✅ Historical data fetched: {data_points} points for {request.stock_code}")
                
                # Check for data quality issues
                self._validate_response_data(response.data, request)
            else:
                self.logger.warning(f"⚠️ Historical data fetch failed: {response.errors}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data: {str(e)}")
            self.request_metrics['failed_requests'] += 1
            return APIResponse(
                success=False,
                data=None,
                errors=[str(e)],
                warnings=[],
                metadata={'compliance_error': True},
                response_time=0
            )
    
    def _validate_historical_request(self, request: MarketDataRequest):
        """Enhanced validation for historical data requests"""
        # Basic validation from parent class
        super()._validate_data_request(request)
        
        # Additional compliance-specific validation
        from_date = pd.to_datetime(request.from_date)
        to_date = pd.to_datetime(request.to_date)
        
        # Check date range limitations
        date_range_days = (to_date - from_date).days
        max_days = {
            '1minute': 7,      # 1 week for 1-minute data
            '5minute': 30,     # 1 month for 5-minute data
            '30minute': 90,    # 3 months for 30-minute data
            '1day': 365 * 5    # 5 years for daily data
        }
        
        interval_limit = max_days.get(request.interval, 365)
        if date_range_days > interval_limit:
            raise ValueError(f"Date range too large for {request.interval} data: {date_range_days} days (max: {interval_limit})")
        
        # Check if request is too recent for historical data
        if request.product_type in ['futures', 'options']:
            # Futures and options need proper expiry validation
            if not request.expiry_date:
                raise ValueError(f"{request.product_type} requests require expiry_date")
    
    def _estimate_data_points(self, request: MarketDataRequest) -> int:
        """Estimate number of data points in the request"""
        try:
            from_date = pd.to_datetime(request.from_date)
            to_date = pd.to_datetime(request.to_date)
            
            # Trading hours per day (approximately)
            trading_hours_per_day = 6.25  # 9:15 AM to 3:30 PM
            
            # Data points per hour based on interval
            points_per_hour = {
                '1minute': 60,
                '5minute': 12,
                '30minute': 2,
                '1hour': 1,
                '1day': 1/24  # For daily data
            }
            
            interval_points = points_per_hour.get(request.interval, 12)  # Default to 5-minute
            
            if request.interval == '1day':
                # For daily data, count trading days
                business_days = pd.bdate_range(from_date, to_date)
                return len(business_days)
            else:
                # For intraday data
                date_range_days = (to_date - from_date).days
                trading_days = date_range_days * 5/7  # Approximate trading days
                return int(trading_days * trading_hours_per_day * interval_points)
                
        except Exception as e:
            self.logger.warning(f"Could not estimate data points: {e}")
            return 1000  # Conservative estimate
    
    def _handle_large_data_request(self, request: MarketDataRequest) -> APIResponse:
        """Handle requests that exceed size limits by splitting them"""
        self.logger.info("🔄 Splitting large data request to comply with limits...")
        
        try:
            from_date = pd.to_datetime(request.from_date)
            to_date = pd.to_datetime(request.to_date)
            
            # Calculate chunk size based on interval
            chunk_days = {
                '1minute': 3,    # 3 days per chunk
                '5minute': 7,    # 1 week per chunk
                '30minute': 30,  # 1 month per chunk
                '1day': 90       # 3 months per chunk
            }
            
            chunk_size = chunk_days.get(request.interval, 7)
            all_data = []
            current_date = from_date
            
            while current_date < to_date:
                chunk_end = min(current_date + timedelta(days=chunk_size), to_date)
                
                # Create chunk request
                chunk_request = MarketDataRequest(
                    stock_code=request.stock_code,
                    exchange_code=request.exchange_code,
                    product_type=request.product_type,
                    interval=request.interval,
                    from_date=current_date.strftime('%Y-%m-%d'),
                    to_date=chunk_end.strftime('%Y-%m-%d'),
                    expiry_date=request.expiry_date,
                    strike_price=request.strike_price,
                    right=request.right
                )
                
                # Add delay between chunks to respect rate limits
                if len(all_data) > 0:
                    time.sleep(self.breeze_limits['min_request_interval'])
                
                # Make chunk request
                chunk_response = super().get_historical_data_safe(chunk_request)
                
                if chunk_response.success and chunk_response.data is not None:
                    all_data.append(chunk_response.data)
                    self.logger.debug(f"✅ Chunk fetched: {current_date.date()} to {chunk_end.date()}")
                else:
                    self.logger.warning(f"⚠️ Chunk failed: {current_date.date()} to {chunk_end.date()}")
                
                current_date = chunk_end + timedelta(days=1)
            
            # Combine all data
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.drop_duplicates().sort_values('datetime')
                
                self.logger.info(f"✅ Large request completed: {len(combined_data)} total points")
                
                return APIResponse(
                    success=True,
                    data=combined_data,
                    errors=[],
                    warnings=[f"Request split into {len(all_data)} chunks for compliance"],
                    metadata={'chunked_request': True, 'chunk_count': len(all_data)},
                    response_time=0
                )
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    errors=["All chunks failed"],
                    warnings=[],
                    metadata={'chunked_request_failed': True},
                    response_time=0
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error handling large request: {str(e)}")
            return APIResponse(
                success=False,
                data=None,
                errors=[str(e)],
                warnings=[],
                metadata={'large_request_error': True},
                response_time=0
            )
    
    def _validate_response_data(self, data: pd.DataFrame, request: MarketDataRequest):
        """Validate response data quality and completeness"""
        if data.empty:
            self.logger.warning("⚠️ Empty response data")
            return
        
        # Check for required columns
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.warning(f"⚠️ Missing columns in response: {missing_columns}")
        
        # Check for data anomalies
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                if data[col].isna().sum() > 0:
                    self.logger.warning(f"⚠️ NaN values found in {col}")
                
                if (data[col] <= 0).sum() > 0 and col != 'volume':  # Volume can be 0
                    self.logger.warning(f"⚠️ Non-positive values found in {col}")
        
        # Check date range completeness
        expected_start = pd.to_datetime(request.from_date)
        expected_end = pd.to_datetime(request.to_date)
        actual_start = data['datetime'].min()
        actual_end = data['datetime'].max()
        
        if actual_start > expected_start:
            self.logger.warning(f"⚠️ Data starts later than requested: {actual_start} vs {expected_start}")
        
        if actual_end < expected_end:
            self.logger.warning(f"⚠️ Data ends earlier than requested: {actual_end} vs {expected_end}")
    
    def _pre_request_compliance_check(self, endpoint: str, params: Dict[str, Any]):
        """Perform compliance checks before making request"""
        # Check current rate limit status
        rate_check = self.compliance_manager.check_rate_limit(
            provider=DataProvider.BREEZE_CONNECT,
            endpoint=endpoint
        )
        
        if not rate_check['allowed']:
            wait_time = rate_check['wait_time']
            self.logger.info(f"⏳ Rate limit reached, waiting {wait_time:.2f}s...")
            
            if self.compliance_level == ComplianceLevel.STRICT:
                time.sleep(wait_time)
            elif wait_time > 10:  # Only wait if more than 10 seconds
                time.sleep(min(wait_time, 10))  # Cap at 10 seconds
        
        # Log compliance status
        if self.request_metrics['total_requests'] % 50 == 0:  # Every 50 requests
            self._log_compliance_status()
    
    def _update_request_metrics(self, success: bool, data: Optional[pd.DataFrame]):
        """Update internal request metrics"""
        self.request_metrics['total_requests'] += 1
        
        if success:
            self.request_metrics['successful_requests'] += 1
            if data is not None:
                self.request_metrics['total_data_points'] += len(data)
        else:
            self.request_metrics['failed_requests'] += 1
    
    def get_session_compliance_report(self) -> Dict[str, Any]:
        """Get compliance report for current session"""
        session_duration = datetime.now() - self.request_metrics['session_start_time']
        
        # Get compliance manager statistics
        compliance_stats = self.compliance_manager.get_usage_statistics(DataProvider.BREEZE_CONNECT)
        breeze_stats = compliance_stats.get('breeze_connect', {})
        
        report = {
            'session_info': {
                'start_time': self.request_metrics['session_start_time'].isoformat(),
                'duration': str(session_duration),
                'compliance_level': self.compliance_level.value
            },
            'request_metrics': self.request_metrics.copy(),
            'compliance_stats': breeze_stats,
            'rate_limits': self.breeze_limits.copy(),
            'terms_compliance': self.terms_compliance.copy(),
            'recommendations': []
        }
        
        # Add recommendations based on metrics
        if self.request_metrics['total_requests'] > 0:
            success_rate = (self.request_metrics['successful_requests'] / 
                          self.request_metrics['total_requests']) * 100
            
            if success_rate < 90:
                report['recommendations'].append(
                    "Low success rate detected. Consider reducing request frequency."
                )
            
            requests_per_minute = self.request_metrics['total_requests'] / max(session_duration.total_seconds() / 60, 1)
            if requests_per_minute > self.breeze_limits['requests_per_minute'] * 0.8:
                report['recommendations'].append(
                    "High request rate detected. Consider implementing additional throttling."
                )
        
        return report
    
    def save_compliance_documentation(self, output_dir: Optional[str] = None) -> str:
        """
        Save comprehensive compliance documentation.
        
        Args:
            output_dir: Directory to save documentation
            
        Returns:
            str: Path to saved documentation
        """
        if output_dir is None:
            output_dir = self.config.get_data_save_path()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comprehensive report
        report = self.get_session_compliance_report()
        compliance_report = self.compliance_manager.generate_compliance_report()
        
        # Create documentation content
        doc_content = [
            "🛡️ BREEZE API COMPLIANCE DOCUMENTATION",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Compliance Level: {self.compliance_level.value}",
            "",
            "📋 TERMS OF SERVICE COMPLIANCE:",
            "-" * 40,
            f"Commercial Use: {'✅ Allowed' if self.terms_compliance['commercial_use_allowed'] else '❌ Prohibited'}",
            f"Data Redistribution: {'❌ Prohibited' if self.terms_compliance['data_redistribution_prohibited'] else '✅ Allowed'}",
            f"Attribution Required: {'✅ Required' if self.terms_compliance['attribution_required'] else 'ℹ️ Not Required'}",
            f"Real-time Restrictions: {'✅ Active' if self.terms_compliance['real_time_data_restrictions'] else 'ℹ️ None'}",
            "",
            "⚙️ RATE LIMITING CONFIGURATION:",
            "-" * 40,
            f"Requests per Second: {self.breeze_limits['requests_per_second']}",
            f"Requests per Minute: {self.breeze_limits['requests_per_minute']}",
            f"Requests per Hour: {self.breeze_limits['requests_per_hour']}",
            f"Daily Limit: {self.breeze_limits['daily_request_limit']}",
            f"Max Concurrent: {self.breeze_limits['max_concurrent_requests']}",
            f"Min Interval: {self.breeze_limits['min_request_interval']}s",
            "",
            "📊 SESSION METRICS:",
            "-" * 40,
            f"Total Requests: {report['request_metrics']['total_requests']}",
            f"Successful: {report['request_metrics']['successful_requests']}",
            f"Failed: {report['request_metrics']['failed_requests']}",
            f"Data Points: {report['request_metrics']['total_data_points']:,}",
            f"Session Duration: {report['session_info']['duration']}",
            "",
            "🎯 COMPLIANCE RECOMMENDATIONS:",
            "-" * 40
        ]
        
        # Add recommendations
        for rec in report['recommendations']:
            doc_content.append(f"• {rec}")
        
        if not report['recommendations']:
            doc_content.append("✅ No compliance issues detected")
        
        doc_content.extend([
            "",
            "📋 DETAILED COMPLIANCE REPORT:",
            "-" * 40,
            compliance_report
        ])
        
        # Save documentation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_filename = f"breeze_api_compliance_report_{timestamp}.md"
        doc_path = os.path.join(output_dir, doc_filename)
        
        with open(doc_path, 'w') as f:
            f.write('\n'.join(doc_content))
        
        self.logger.info(f"✅ Compliance documentation saved to {doc_path}")
        return doc_path
    
    def cleanup(self):
        """Cleanup resources and save compliance state"""
        self.logger.info("🧹 Cleaning up compliance-aware Breeze manager...")
        
        # Save final compliance report
        try:
            self.save_compliance_documentation()
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save final compliance report: {e}")
        
        # Cleanup compliance manager
        self.compliance_manager.cleanup()
        
        self.logger.info("✅ Compliance cleanup complete")

# Convenience functions for backward compatibility
def create_compliance_breeze_manager(compliance_level: ComplianceLevel = ComplianceLevel.MODERATE) -> ComplianceBreezeDataManager:
    """Create a compliance-aware Breeze data manager"""
    return ComplianceBreezeDataManager(compliance_level=compliance_level)

def check_breeze_compliance() -> Dict[str, Any]:
    """Quick compliance check for Breeze API"""
    return check_terms_compliance(DataProvider.BREEZE_CONNECT)

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Compliance-Aware Breeze Data Manager...")
    
    # Initialize manager with moderate compliance
    manager = ComplianceBreezeDataManager(compliance_level=ComplianceLevel.MODERATE)
    
    # Test authentication
    if manager.authenticate():
        print("✅ Authentication successful")
        
        # Test quote fetching
        response = manager.get_quotes_safe("TCS", "NSE")
        if response.success:
            print(f"✅ Quote fetched successfully")
        else:
            print(f"❌ Quote fetch failed: {response.errors}")
        
        # Generate compliance report
        report_path = manager.save_compliance_documentation()
        print(f"✅ Compliance report saved to: {report_path}")
        
    else:
        print("❌ Authentication failed")
    
    # Cleanup
    manager.cleanup()
    print("🎉 Compliance test complete!")
