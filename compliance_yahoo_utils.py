#!/usr/bin/env python3
"""
🛡️ COMPLIANCE-AWARE YAHOO FINANCE UTILITIES
Enhanced Yahoo Finance data utilities with API compliance and rate limiting

This module provides compliant access to Yahoo Finance data with strict
adherence to terms of service, rate limiting, and usage monitoring.
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import yfinance as yf

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import existing modules
from index_utils import IndexDataManager
from api_compliance import (
    ComplianceManager, DataProvider, ComplianceLevel, 
    compliance_decorator, get_compliance_manager, check_terms_compliance
)
from app_config import Config
from file_management_utils import SafeFileManager, SaveStrategy

logger = logging.getLogger(__name__)

class ComplianceYahooFinanceManager:
    """
    Yahoo Finance data manager with comprehensive API compliance.
    
    Features:
    - Strict adherence to Yahoo Finance Terms of Service
    - Rate limiting to prevent API abuse
    - Commercial use compliance checking
    - Request monitoring and analytics
    - Data caching to reduce API calls
    - Usage documentation and reporting
    
    Important: Yahoo Finance terms prohibit commercial use without proper licensing.
    This implementation is designed for personal, educational, and research use.
    """
    
    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.STRICT):
        """
        Initialize compliance-aware Yahoo Finance manager.
        
        Args:
            compliance_level: Level of compliance enforcement
            
        Note: 
            Yahoo Finance has strict terms of service. This manager enforces
            compliance and is suitable for educational/research use only.
        """
        self.config = Config()
        self.compliance_manager = get_compliance_manager(compliance_level)
        self.compliance_level = compliance_level
        
        # Initialize file manager
        self.file_manager = SafeFileManager(
            base_path=self.config.get_data_save_path(),
            default_strategy=SaveStrategy.TIMESTAMP
        )
        
        # Yahoo Finance specific limitations
        self.yahoo_limits = {
            'requests_per_second': 1.0,       # Conservative rate
            'requests_per_minute': 30,        # Very conservative
            'requests_per_hour': 500,         # Daily usage spread
            'daily_request_limit': 2000,      # Conservative daily limit
            'max_concurrent_requests': 2,     # Minimal concurrent requests
            'min_request_interval': 1.0,      # 1 second between requests
            'burst_allowance': 5,             # Small burst capacity
            'max_symbols_per_request': 10,    # Batch request limit
            'max_history_years': 10,          # Historical data limit
            'cooldown_after_error': 5.0       # 5 seconds after error
        }
        
        # Terms of Service compliance (CRITICAL)
        self.terms_compliance = {
            'commercial_use_prohibited': True,     # Yahoo prohibits commercial use
            'personal_use_only': True,            # Personal/educational only
            'data_redistribution_prohibited': True, # Cannot redistribute data
            'attribution_required': True,          # Must attribute Yahoo Finance
            'api_abuse_prevention': True,          # Must prevent abuse
            'rate_limiting_mandatory': True,       # Rate limiting required
            'caching_recommended': True,           # Reduce API calls via caching
            'research_use_allowed': True,          # Academic research permitted
            'real_time_restrictions': True         # Real-time data restrictions
        }
        
        # Usage tracking
        self.usage_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_responses': 0,
            'symbols_fetched': set(),
            'data_points_retrieved': 0,
            'session_start_time': datetime.now(),
            'last_request_time': None
        }
        
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("✅ Compliance-aware Yahoo Finance Manager initialized")
        
        # Validate terms compliance on initialization
        self._validate_terms_compliance()
        self._log_usage_warnings()
    
    def _validate_terms_compliance(self):
        """Validate and log Yahoo Finance terms of service compliance"""
        self.logger.info("🔍 Yahoo Finance Terms of Service Compliance Check:")
        self.logger.warning("⚠️  COMMERCIAL USE PROHIBITED: Yahoo Finance data cannot be used commercially")
        self.logger.info("✅ Personal/Educational Use: Allowed for research and learning")
        self.logger.warning("⚠️  DATA REDISTRIBUTION PROHIBITED: Cannot share or redistribute data")
        self.logger.info("✅ Attribution Required: Must credit Yahoo Finance as data source")
        self.logger.info("✅ Rate Limiting: Implemented to prevent API abuse")
        self.logger.info("✅ Caching: Enabled to reduce API calls")
        
        # Check compliance configuration
        compliance_check = check_terms_compliance(DataProvider.YAHOO_FINANCE)
        
        if not compliance_check['commercial_use_allowed']:
            self.logger.warning("🚨 COMPLIANCE ALERT: Commercial use not permitted with current terms")
        
        if compliance_check['attribution_required']:
            self.logger.info("ℹ️  Attribution: Please credit 'Data provided by Yahoo Finance'")
    
    def _log_usage_warnings(self):
        """Log important usage warnings and guidelines"""
        self.logger.warning("=" * 80)
        self.logger.warning("🚨 IMPORTANT YAHOO FINANCE USAGE GUIDELINES:")
        self.logger.warning("• This tool is for PERSONAL, EDUCATIONAL, and RESEARCH use only")
        self.logger.warning("• COMMERCIAL use of Yahoo Finance data is PROHIBITED")
        self.logger.warning("• DO NOT redistribute or share the retrieved data")
        self.logger.warning("• Rate limiting is enforced to comply with terms of service")
        self.logger.warning("• Always attribute Yahoo Finance as the data source")
        self.logger.warning("• Use responsibly to maintain access for the community")
        self.logger.warning("=" * 80)
    
    @compliance_decorator(DataProvider.YAHOO_FINANCE, "download")
    def download_symbol_data(self, symbol: str, period: str = "1y", 
                           interval: str = "1d", start: Optional[str] = None, 
                           end: Optional[str] = None) -> Dict[str, Any]:
        """
        Download data for a single symbol with compliance monitoring.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            
        Returns:
            Dict with success status, data, and metadata
            
        Note:
            This method enforces Yahoo Finance terms of service and implements
            rate limiting to prevent API abuse.
        """
        self.logger.info(f"📊 Downloading {symbol} data (period: {period}, interval: {interval})")
        
        try:
            # Validate request parameters
            self._validate_download_request(symbol, period, interval, start, end)
            
            # Pre-request compliance check
            self._pre_request_compliance_check('download', {
                'symbol': symbol,
                'period': period,
                'interval': interval
            })
            
            # Check cache first
            cache_key = self._generate_cache_key(symbol, period, interval, start, end)
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                self.usage_metrics['cached_responses'] += 1
                self.logger.info(f"📦 Using cached data for {symbol}")
                return {
                    'success': True,
                    'data': cached_data,
                    'cached': True,
                    'symbol': symbol,
                    'metadata': {'cache_hit': True}
                }
            
            # Make the actual request
            start_time = time.time()
            
            # Prepare yfinance download parameters
            download_params = {
                'tickers': symbol,
                'period': period if not (start and end) else None,
                'interval': interval,
                'start': start,
                'end': end,
                'progress': False,  # Disable progress bar
                'threads': 1,       # Single thread for rate limiting
                'group_by': 'ticker'
            }
            
            # Remove None values
            download_params = {k: v for k, v in download_params.items() if v is not None}
            
            # Download data
            data = yf.download(**download_params)
            response_time = time.time() - start_time
            
            # Process response
            if data.empty:
                self.logger.warning(f"⚠️ No data returned for {symbol}")
                self.usage_metrics['failed_requests'] += 1
                return {
                    'success': False,
                    'data': None,
                    'error': 'No data available',
                    'symbol': symbol,
                    'metadata': {'response_time': response_time}
                }
            
            # Clean and validate data
            cleaned_data = self._clean_yahoo_data(data, symbol)
            
            # Update metrics
            self.usage_metrics['successful_requests'] += 1
            self.usage_metrics['symbols_fetched'].add(symbol)
            self.usage_metrics['data_points_retrieved'] += len(cleaned_data)
            self.usage_metrics['last_request_time'] = datetime.now()
            
            # Cache the data
            self._cache_data(cache_key, cleaned_data)
            
            # Save to file with proper attribution
            self._save_data_with_attribution(cleaned_data, symbol, period, interval)
            
            self.logger.info(f"✅ Successfully downloaded {len(cleaned_data)} data points for {symbol}")
            
            return {
                'success': True,
                'data': cleaned_data,
                'cached': False,
                'symbol': symbol,
                'metadata': {
                    'response_time': response_time,
                    'data_points': len(cleaned_data),
                    'period': period,
                    'interval': interval
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading {symbol}: {str(e)}")
            self.usage_metrics['failed_requests'] += 1
            
            return {
                'success': False,
                'data': None,
                'error': str(e),
                'symbol': symbol,
                'metadata': {'error_type': type(e).__name__}
            }
    
    @compliance_decorator(DataProvider.YAHOO_FINANCE, "download_batch")
    def download_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Download data for multiple symbols with proper rate limiting.
        
        Args:
            symbols: List of stock symbols
            **kwargs: Arguments passed to download_symbol_data
            
        Returns:
            Dict with symbol as key and download result as value
        """
        self.logger.info(f"📊 Downloading data for {len(symbols)} symbols")
        
        # Validate symbol count
        if len(symbols) > self.yahoo_limits['max_symbols_per_request']:
            self.logger.warning(f"⚠️ Too many symbols requested: {len(symbols)} (max: {self.yahoo_limits['max_symbols_per_request']})")
            # Split into batches
            return self._download_symbols_in_batches(symbols, **kwargs)
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            self.logger.debug(f"Downloading {symbol} ({i+1}/{len(symbols)})")
            
            # Rate limiting between requests
            if i > 0:
                time.sleep(self.yahoo_limits['min_request_interval'])
            
            # Download individual symbol
            result = self.download_symbol_data(symbol, **kwargs)
            results[symbol] = result
            
            # Log progress
            if (i + 1) % 10 == 0:
                success_count = sum(1 for r in results.values() if r['success'])
                self.logger.info(f"Progress: {i+1}/{len(symbols)} symbols, {success_count} successful")
        
        # Summary
        successful = [s for s, r in results.items() if r['success']]
        failed = [s for s, r in results.items() if not r['success']]
        
        self.logger.info(f"✅ Batch download complete: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            self.logger.warning(f"⚠️ Failed symbols: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
        
        return results
    
    def _download_symbols_in_batches(self, symbols: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """Download symbols in smaller batches to comply with limits"""
        batch_size = self.yahoo_limits['max_symbols_per_request']
        all_results = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            self.logger.info(f"📦 Processing batch {i//batch_size + 1}: {len(batch)} symbols")
            
            # Add delay between batches
            if i > 0:
                batch_delay = self.yahoo_limits['min_request_interval'] * 5  # Longer delay between batches
                self.logger.info(f"⏳ Waiting {batch_delay}s between batches...")
                time.sleep(batch_delay)
            
            # Download batch
            batch_results = self.download_multiple_symbols(batch, **kwargs)
            all_results.update(batch_results)
        
        return all_results
    
    def _validate_download_request(self, symbol: str, period: str, interval: str, 
                                 start: Optional[str], end: Optional[str]):
        """Validate download request parameters"""
        # Validate symbol
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        
        # Validate period
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            raise ValueError(f"Invalid period: {period}. Valid options: {valid_periods}")
        
        # Validate interval
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {interval}. Valid options: {valid_intervals}")
        
        # Validate date range if provided
        if start and end:
            try:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                
                if end_date <= start_date:
                    raise ValueError("End date must be after start date")
                
                # Check if date range is too large
                date_range = (end_date - start_date).days
                max_days = {
                    '1m': 30,    # 1-minute data limited to 30 days
                    '2m': 60,    # 2-minute data limited to 60 days
                    '5m': 60,    # 5-minute data limited to 60 days
                    '15m': 730,  # 15-minute data limited to 2 years
                    '30m': 730,  # 30-minute data limited to 2 years
                    '1h': 730,   # 1-hour data limited to 2 years
                    '1d': 3650   # Daily data limited to 10 years
                }
                
                limit = max_days.get(interval, 3650)
                if date_range > limit:
                    raise ValueError(f"Date range too large for {interval} interval: {date_range} days (max: {limit})")
                
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid date format: {e}")
        
        # Warn about high-frequency data
        if interval in ['1m', '2m', '5m'] and period in ['1y', '2y', '5y', '10y', 'max']:
            self.logger.warning(f"⚠️ High-frequency data ({interval}) over long period ({period}) may hit API limits")
    
    def _generate_cache_key(self, symbol: str, period: str, interval: str, 
                          start: Optional[str], end: Optional[str]) -> str:
        """Generate cache key for request"""
        key_components = [symbol, period, interval, start or '', end or '']
        return '_'.join(str(c) for c in key_components)
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and not expired"""
        cache_file = os.path.join(self.config.get_data_save_path(), f"cache_{cache_key}.csv")
        
        if os.path.exists(cache_file):
            try:
                # Check file age
                file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
                cache_expiry_hours = 24  # Cache expires after 24 hours
                
                if file_age_hours < cache_expiry_hours:
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
                else:
                    # Remove expired cache
                    os.remove(cache_file)
            except Exception as e:
                self.logger.warning(f"⚠️ Error reading cache: {e}")
        
        return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data to disk"""
        try:
            cache_file = os.path.join(self.config.get_data_save_path(), f"cache_{cache_key}.csv")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            data.to_csv(cache_file)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cache data: {e}")
    
    def _clean_yahoo_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize Yahoo Finance data"""
        if data.empty:
            return data
        
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # If multiple symbols, get data for the specific symbol
            if symbol in data.columns.get_level_values(1):
                data = data.xs(symbol, level=1, axis=1)
            else:
                # Single symbol data, flatten columns
                data.columns = data.columns.get_level_values(0)
        
        # Standardize column names
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"⚠️ Missing columns for {symbol}: {missing_columns}")
        
        # Clean data
        data = data.dropna(subset=['close'])  # Remove rows with no closing price
        data = data[data['volume'] >= 0]      # Remove negative volume
        
        # Add metadata
        data.index.name = 'date'
        
        return data
    
    def _save_data_with_attribution(self, data: pd.DataFrame, symbol: str, 
                                  period: str, interval: str):
        """Save data to file with proper Yahoo Finance attribution"""
        try:
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{period}_{interval}_{timestamp}.csv"
            
            # Add attribution metadata
            metadata = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'data_source': 'Yahoo Finance',
                'attribution': 'Data provided by Yahoo Finance',
                'download_time': datetime.now().isoformat(),
                'terms_notice': 'Data is for personal/educational use only. Commercial use prohibited.',
                'data_points': len(data)
            }
            
            # Save using file manager
            save_result = self.file_manager.save_dataframe(
                df=data,
                filename=filename,
                metadata=metadata
            )
            
            if save_result.success:
                self.logger.debug(f"💾 Data saved: {save_result.final_filename}")
            else:
                self.logger.warning(f"⚠️ Failed to save data: {save_result.error_message}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error saving data: {e}")
    
    def _pre_request_compliance_check(self, endpoint: str, params: Dict[str, Any]):
        """Perform compliance checks before making request"""
        # Update total requests
        self.usage_metrics['total_requests'] += 1
        
        # Check rate limits
        rate_check = self.compliance_manager.check_rate_limit(
            provider=DataProvider.YAHOO_FINANCE,
            endpoint=endpoint
        )
        
        if not rate_check['allowed']:
            wait_time = rate_check['wait_time']
            self.logger.info(f"⏳ Rate limit reached, waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        
        # Enforce minimum interval between requests
        if self.usage_metrics['last_request_time']:
            time_since_last = (datetime.now() - self.usage_metrics['last_request_time']).total_seconds()
            if time_since_last < self.yahoo_limits['min_request_interval']:
                sleep_time = self.yahoo_limits['min_request_interval'] - time_since_last
                self.logger.debug(f"⏳ Enforcing minimum interval: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Log periodic compliance status
        if self.usage_metrics['total_requests'] % 25 == 0:
            self._log_usage_statistics()
    
    def _log_usage_statistics(self):
        """Log current usage statistics"""
        session_duration = datetime.now() - self.usage_metrics['session_start_time']
        
        self.logger.info("📊 Yahoo Finance Usage Statistics:")
        self.logger.info(f"  Total Requests: {self.usage_metrics['total_requests']}")
        self.logger.info(f"  Successful: {self.usage_metrics['successful_requests']}")
        self.logger.info(f"  Failed: {self.usage_metrics['failed_requests']}")
        self.logger.info(f"  Cached Responses: {self.usage_metrics['cached_responses']}")
        self.logger.info(f"  Unique Symbols: {len(self.usage_metrics['symbols_fetched'])}")
        self.logger.info(f"  Data Points: {self.usage_metrics['data_points_retrieved']:,}")
        self.logger.info(f"  Session Duration: {session_duration}")
        
        # Calculate rates
        if session_duration.total_seconds() > 0:
            requests_per_minute = self.usage_metrics['total_requests'] / (session_duration.total_seconds() / 60)
            self.logger.info(f"  Request Rate: {requests_per_minute:.2f} req/min")
            
            # Check if approaching limits
            if requests_per_minute > self.yahoo_limits['requests_per_minute'] * 0.8:
                self.logger.warning("⚠️ Approaching rate limit! Consider reducing request frequency.")
    
    def get_terms_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive terms of service compliance report"""
        session_duration = datetime.now() - self.usage_metrics['session_start_time']
        
        return {
            'terms_compliance': self.terms_compliance.copy(),
            'usage_metrics': {
                **self.usage_metrics,
                'symbols_fetched': list(self.usage_metrics['symbols_fetched']),
                'session_duration_str': str(session_duration)
            },
            'rate_limits': self.yahoo_limits.copy(),
            'compliance_level': self.compliance_level.value,
            'attribution_text': 'Data provided by Yahoo Finance',
            'usage_guidelines': [
                'Data is for personal, educational, and research use only',
                'Commercial use is prohibited without proper licensing',
                'Data cannot be redistributed or shared with third parties',
                'Always attribute Yahoo Finance as the data source',
                'Respect rate limits to maintain API access',
                'Use caching to minimize API requests'
            ],
            'recommendations': self._generate_compliance_recommendations()
        }
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations based on usage patterns"""
        recommendations = []
        
        if self.usage_metrics['total_requests'] > 0:
            success_rate = (self.usage_metrics['successful_requests'] / 
                          self.usage_metrics['total_requests']) * 100
            
            if success_rate < 90:
                recommendations.append("Low success rate detected. Consider reducing request frequency or checking symbol validity.")
            
            if self.usage_metrics['cached_responses'] == 0:
                recommendations.append("No cache hits detected. Enable data caching to reduce API calls.")
            
            session_duration = datetime.now() - self.usage_metrics['session_start_time']
            if session_duration.total_seconds() > 0:
                requests_per_hour = self.usage_metrics['total_requests'] / (session_duration.total_seconds() / 3600)
                
                if requests_per_hour > self.yahoo_limits['requests_per_hour'] * 0.8:
                    recommendations.append("High request rate detected. Implement additional throttling to stay within limits.")
        
        if len(self.usage_metrics['symbols_fetched']) > 50:
            recommendations.append("Large number of symbols fetched. Consider batch processing with longer intervals.")
        
        return recommendations
    
    def save_compliance_documentation(self, output_file: Optional[str] = None) -> str:
        """Save comprehensive compliance documentation"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.config.get_data_save_path(),
                f"yahoo_finance_compliance_report_{timestamp}.md"
            )
        
        # Generate report
        report = self.get_terms_compliance_report()
        
        doc_content = [
            "🛡️ YAHOO FINANCE API COMPLIANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Compliance Level: {report['compliance_level']}",
            "",
            "📋 TERMS OF SERVICE COMPLIANCE:",
            "-" * 40,
            f"Commercial Use: {'❌ PROHIBITED' if report['terms_compliance']['commercial_use_prohibited'] else '✅ Allowed'}",
            f"Personal Use: {'✅ ALLOWED' if report['terms_compliance']['personal_use_only'] else '❌ Prohibited'}",
            f"Data Redistribution: {'❌ PROHIBITED' if report['terms_compliance']['data_redistribution_prohibited'] else '✅ Allowed'}",
            f"Attribution Required: {'✅ REQUIRED' if report['terms_compliance']['attribution_required'] else 'ℹ️ Optional'}",
            f"Rate Limiting: {'✅ IMPLEMENTED' if report['terms_compliance']['rate_limiting_mandatory'] else '❌ Missing'}",
            f"Caching: {'✅ ENABLED' if report['terms_compliance']['caching_recommended'] else '⚠️ Disabled'}",
            "",
            "⚙️ RATE LIMITING CONFIGURATION:",
            "-" * 40,
            f"Requests per Second: {report['rate_limits']['requests_per_second']}",
            f"Requests per Minute: {report['rate_limits']['requests_per_minute']}",
            f"Requests per Hour: {report['rate_limits']['requests_per_hour']}",
            f"Daily Limit: {report['rate_limits']['daily_request_limit']}",
            f"Min Request Interval: {report['rate_limits']['min_request_interval']}s",
            f"Max Symbols per Request: {report['rate_limits']['max_symbols_per_request']}",
            "",
            "📊 SESSION USAGE STATISTICS:",
            "-" * 40,
            f"Total Requests: {report['usage_metrics']['total_requests']}",
            f"Successful Requests: {report['usage_metrics']['successful_requests']}",
            f"Failed Requests: {report['usage_metrics']['failed_requests']}",
            f"Cached Responses: {report['usage_metrics']['cached_responses']}",
            f"Unique Symbols: {len(report['usage_metrics']['symbols_fetched'])}",
            f"Data Points Retrieved: {report['usage_metrics']['data_points_retrieved']:,}",
            f"Session Duration: {report['usage_metrics']['session_duration_str']}",
            "",
            "📝 ATTRIBUTION:",
            "-" * 40,
            f"Required Attribution: {report['attribution_text']}",
            "",
            "🎯 USAGE GUIDELINES:",
            "-" * 40
        ]
        
        for guideline in report['usage_guidelines']:
            doc_content.append(f"• {guideline}")
        
        doc_content.extend([
            "",
            "💡 COMPLIANCE RECOMMENDATIONS:",
            "-" * 40
        ])
        
        if report['recommendations']:
            for rec in report['recommendations']:
                doc_content.append(f"• {rec}")
        else:
            doc_content.append("✅ No compliance issues detected")
        
        doc_content.extend([
            "",
            "⚠️  IMPORTANT NOTICES:",
            "-" * 40,
            "• Yahoo Finance data is provided for informational purposes only",
            "• Commercial use requires proper licensing from Yahoo Finance",
            "• Always verify data accuracy before making financial decisions",
            "• Respect API rate limits to maintain access for all users",
            "• This implementation is designed for educational and research use"
        ])
        
        # Save documentation
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(doc_content))
        
        self.logger.info(f"✅ Compliance documentation saved to {output_file}")
        return output_file
    
    def cleanup(self):
        """Cleanup resources and save final compliance state"""
        self.logger.info("🧹 Cleaning up Yahoo Finance compliance manager...")
        
        try:
            # Save final compliance report
            self.save_compliance_documentation()
            
            # Log final statistics
            self._log_usage_statistics()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during cleanup: {e}")
        
        self.logger.info("✅ Yahoo Finance compliance cleanup complete")

# Convenience functions
def create_compliant_yahoo_manager(compliance_level: ComplianceLevel = ComplianceLevel.STRICT) -> ComplianceYahooFinanceManager:
    """Create a compliance-aware Yahoo Finance manager"""
    return ComplianceYahooFinanceManager(compliance_level=compliance_level)

def download_symbol_compliant(symbol: str, **kwargs) -> Dict[str, Any]:
    """Download symbol data with full compliance checking"""
    manager = create_compliant_yahoo_manager()
    try:
        result = manager.download_symbol_data(symbol, **kwargs)
        return result
    finally:
        manager.cleanup()

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Compliance-Aware Yahoo Finance Manager...")
    
    # Initialize manager
    manager = ComplianceYahooFinanceManager(compliance_level=ComplianceLevel.STRICT)
    
    # Test single symbol download
    result = manager.download_symbol_data("AAPL", period="1mo", interval="1d")
    
    if result['success']:
        print(f"✅ Successfully downloaded {len(result['data'])} data points for AAPL")
    else:
        print(f"❌ Failed to download AAPL: {result['error']}")
    
    # Test multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = manager.download_multiple_symbols(symbols, period="1mo", interval="1d")
    
    successful = [s for s, r in results.items() if r['success']]
    print(f"✅ Batch download: {len(successful)}/{len(symbols)} successful")
    
    # Generate compliance report
    report_path = manager.save_compliance_documentation()
    print(f"✅ Compliance report saved to: {report_path}")
    
    # Cleanup
    manager.cleanup()
    print("🎉 Yahoo Finance compliance test complete!")
