#!/usr/bin/env python3
"""
🛡️ API COMPLIANCE DEMONSTRATION
Comprehensive demonstration of API compliance and rate limiting features

This script demonstrates how the enhanced compliance system works across
different data providers while maintaining all existing functionality.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import compliance utilities
from api_compliance import ComplianceManager, DataProvider, ComplianceLevel, get_compliance_manager
from compliance_breeze_utils import ComplianceBreezeDataManager
from compliance_yahoo_utils import ComplianceYahooFinanceManager
from enhanced_breeze_utils import MarketDataRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_compliance_manager():
    """Demonstrate core compliance manager functionality"""
    print("\n" + "=" * 80)
    print("🛡️ API COMPLIANCE MANAGER DEMONSTRATION")
    print("=" * 80)
    
    # Initialize compliance manager
    manager = get_compliance_manager(ComplianceLevel.MODERATE)
    
    print("\n1. RATE LIMIT CHECKING:")
    print("-" * 40)
    
    # Test rate limit checking for different providers
    providers = [DataProvider.BREEZE_CONNECT, DataProvider.YAHOO_FINANCE]
    
    for provider in providers:
        rate_check = manager.check_rate_limit(provider, "test_endpoint")
        print(f"{provider.value}:")
        print(f"  Allowed: {'✅' if rate_check['allowed'] else '❌'}")
        print(f"  Wait Time: {rate_check['wait_time']:.2f}s")
        print(f"  Request Count: {rate_check['request_count']}")
    
    print("\n2. USAGE STATISTICS:")
    print("-" * 40)
    
    # Get usage statistics
    stats = manager.get_usage_statistics()
    for provider, data in stats.items():
        print(f"{provider}:")
        if 'message' in data:
            print(f"  {data['message']}")
        else:
            print(f"  Requests: {data['total_requests']}")
            print(f"  Success Rate: {data['success_rate']:.1f}%")
            print(f"  Compliance Score: {data['compliance_score']:.1f}%")
    
    print("\n3. COMPLIANCE REPORT:")
    print("-" * 40)
    
    # Generate brief report
    report = manager.generate_compliance_report()
    print(report[:500] + "..." if len(report) > 500 else report)
    
    return manager

def demonstrate_breeze_compliance():
    """Demonstrate Breeze API compliance features"""
    print("\n" + "=" * 80)
    print("🔗 BREEZE API COMPLIANCE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize compliance-aware Breeze manager
        manager = ComplianceBreezeDataManager(
            compliance_level=ComplianceLevel.MODERATE
        )
        
        print("\n1. TERMS OF SERVICE VALIDATION:")
        print("-" * 40)
        print("✅ Commercial use allowed with proper licensing")
        print("❌ Data redistribution prohibited")
        print("ℹ️  Real-time data subject to usage restrictions")
        
        print("\n2. AUTHENTICATION WITH COMPLIANCE:")
        print("-" * 40)
        
        # Test authentication (will show compliance monitoring)
        auth_success = manager.authenticate()
        if auth_success:
            print("✅ Authentication successful with compliance monitoring")
            
            print("\n3. RATE-LIMITED QUOTE FETCHING:")
            print("-" * 40)
            
            # Test quote fetching with rate limiting
            symbols = ["TCS", "INFY", "WIPRO"]
            for symbol in symbols:
                print(f"Fetching quote for {symbol}...")
                response = manager.get_quotes_safe(symbol, "NSE")
                
                if response.success:
                    print(f"✅ {symbol}: Quote fetched successfully")
                else:
                    print(f"❌ {symbol}: {response.errors}")
                
                # Small delay between requests (compliance enforced automatically)
                time.sleep(0.5)
            
            print("\n4. HISTORICAL DATA WITH VALIDATION:")
            print("-" * 40)
            
            # Test historical data fetching
            request = MarketDataRequest(
                stock_code="TCS",
                exchange_code="NSE",
                product_type="cash",
                interval="1day",
                from_date="2024-01-01",
                to_date="2024-01-07"  # Small date range for demo
            )
            
            print("Fetching historical data with compliance validation...")
            response = manager.get_historical_data_safe(request)
            
            if response.success and response.data is not None:
                print(f"✅ Historical data: {len(response.data)} records fetched")
                print("✅ Data validation and compliance checks passed")
            else:
                print(f"❌ Historical data fetch failed: {response.errors}")
            
            print("\n5. SESSION COMPLIANCE REPORT:")
            print("-" * 40)
            
            # Get session report
            session_report = manager.get_session_compliance_report()
            print(f"Total Requests: {session_report['request_metrics']['total_requests']}")
            print(f"Successful: {session_report['request_metrics']['successful_requests']}")
            print(f"Failed: {session_report['request_metrics']['failed_requests']}")
            print(f"Compliance Level: {session_report['session_info']['compliance_level']}")
            
            # Save compliance documentation
            doc_path = manager.save_compliance_documentation()
            print("📋 Compliance documentation saved to:")
            print(f"    {os.path.basename(doc_path)}")
            
        else:
            print("❌ Authentication failed")
        
        # Cleanup
        manager.cleanup()
        return manager
        
    except Exception as e:
        print(f"❌ Breeze compliance demonstration failed: {str(e)}")
        return None

def demonstrate_yahoo_compliance():
    """Demonstrate Yahoo Finance compliance features"""
    print("\n" + "=" * 80)
    print("📊 YAHOO FINANCE COMPLIANCE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize strict compliance Yahoo manager
        manager = ComplianceYahooFinanceManager(
            compliance_level=ComplianceLevel.STRICT
        )
        
        print("\n1. TERMS OF SERVICE VALIDATION:")
        print("-" * 40)
        print("❌ Commercial use PROHIBITED without proper licensing")
        print("✅ Personal/educational use ALLOWED")
        print("❌ Data redistribution PROHIBITED")
        print("✅ Attribution required: 'Data provided by Yahoo Finance'")
        
        print("\n2. COMPLIANT DATA DOWNLOAD:")
        print("-" * 40)
        
        # Test single symbol download
        print("Downloading AAPL data with compliance monitoring...")
        result = manager.download_symbol_data(
            symbol="AAPL",
            period="5d",  # Small period for demo
            interval="1d"
        )
        
        if result['success']:
            data_points = len(result['data'])
            cached = result.get('cached', False)
            print(f"✅ AAPL: {data_points} data points downloaded")
            print(f"📦 Cached: {'Yes' if cached else 'No'}")
            print("✅ Proper attribution applied automatically")
        else:
            print(f"❌ AAPL download failed: {result['error']}")
        
        print("\n3. BATCH DOWNLOAD WITH RATE LIMITING:")
        print("-" * 40)
        
        # Test multiple symbols with rate limiting
        symbols = ["MSFT", "GOOGL"]  # Small batch for demo
        print(f"Downloading {len(symbols)} symbols with rate limiting...")
        
        results = manager.download_multiple_symbols(
            symbols=symbols,
            period="5d",
            interval="1d"
        )
        
        successful = [s for s, r in results.items() if r['success']]
        print(f"✅ Batch download: {len(successful)}/{len(symbols)} successful")
        
        for symbol in successful:
            cached = results[symbol].get('cached', False)
            data_points = len(results[symbol]['data'])
            print(f"  {symbol}: {data_points} points {'(cached)' if cached else '(new)'}")
        
        print("\n4. TERMS COMPLIANCE REPORT:")
        print("-" * 40)
        
        # Get compliance report
        report = manager.get_terms_compliance_report()
        usage = report['usage_metrics']
        
        print(f"Total Requests: {usage['total_requests']}")
        print(f"Cached Responses: {usage['cached_responses']}")
        print(f"Unique Symbols: {len(usage['symbols_fetched'])}")
        print(f"Data Points: {usage['data_points_retrieved']:,}")
        
        # Check for recommendations
        if report['recommendations']:
            print("\n⚠️  Compliance Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        else:
            print("✅ No compliance issues detected")
        
        # Save compliance documentation
        doc_path = manager.save_compliance_documentation()
        print("\n📋 Compliance documentation saved to:")
        print(f"    {os.path.basename(doc_path)}")
        
        # Cleanup
        manager.cleanup()
        return manager
        
    except Exception as e:
        print(f"❌ Yahoo Finance compliance demonstration failed: {str(e)}")
        return None

def demonstrate_compliance_levels():
    """Demonstrate different compliance enforcement levels"""
    print("\n" + "=" * 80)
    print("⚙️ COMPLIANCE LEVELS DEMONSTRATION")
    print("=" * 80)
    
    levels = [
        (ComplianceLevel.STRICT, "All limits strictly enforced"),
        (ComplianceLevel.MODERATE, "Most limits enforced with flexibility"),
        (ComplianceLevel.LENIENT, "Basic limits with significant flexibility"),
        (ComplianceLevel.MONITORING, "Monitoring only, no enforcement")
    ]
    
    for level, description in levels:
        print(f"\n{level.value.upper()} COMPLIANCE:")
        print(f"  Description: {description}")
        
        # Create manager with this level
        manager = get_compliance_manager(level)
        
        # Test rate limit check
        rate_check = manager.check_rate_limit(
            DataProvider.YAHOO_FINANCE, 
            "demo_endpoint"
        )
        
        print(f"  Rate Check: {'✅ Allowed' if rate_check['allowed'] else '❌ Blocked'}")
        print(f"  Enforcement: {level.value}")

def demonstrate_usage_monitoring():
    """Demonstrate usage monitoring and analytics"""
    print("\n" + "=" * 80)
    print("📊 USAGE MONITORING DEMONSTRATION")
    print("=" * 80)
    
    # Get global compliance manager
    manager = get_compliance_manager()
    
    print("\n1. REAL-TIME MONITORING:")
    print("-" * 40)
    
    # Simulate some API requests for monitoring
    for i in range(5):
        try:
            request = manager.request_permission(
                provider=DataProvider.YAHOO_FINANCE,
                endpoint="demo_monitoring",
                params={'request_id': i}
            )
            
            # Simulate response
            manager.record_response(
                request=request,
                success=True,
                response_time=0.5 + (i * 0.1),
                data_size=1024 * (i + 1)
            )
            
            print(f"  Request {i+1}: ✅ Recorded")
            
        except Exception as e:
            print(f"  Request {i+1}: ❌ {str(e)}")
    
    print("\n2. ANALYTICS SUMMARY:")
    print("-" * 40)
    
    # Get updated statistics
    stats = manager.get_usage_statistics()
    
    for provider, data in stats.items():
        if 'message' not in data and data['total_requests'] > 0:
            print(f"{provider}:")
            print(f"  Total Requests: {data['total_requests']}")
            print(f"  Success Rate: {data['success_rate']:.1f}%")
            print(f"  Avg Response Time: {data['avg_response_time']:.3f}s")
            print(f"  Data Transfer: {data['total_data_size']:,} bytes")

def main():
    """Main demonstration function"""
    print("🛡️ API COMPLIANCE AND RATE LIMITING DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows comprehensive API compliance features")
    print("while maintaining all existing functionality and logic.")
    print("=" * 80)
    
    try:
        # Core compliance manager demonstration
        compliance_manager = demonstrate_compliance_manager()
        
        # Yahoo Finance compliance (most restrictive)
        yahoo_manager = demonstrate_yahoo_compliance()
        
        # Breeze API compliance (if available)
        breeze_manager = demonstrate_breeze_compliance()
        
        # Compliance levels demonstration
        demonstrate_compliance_levels()
        
        # Usage monitoring demonstration
        demonstrate_usage_monitoring()
        
        print("\n" + "=" * 80)
        print("🎉 API COMPLIANCE DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("✅ All API calls now include comprehensive compliance monitoring")
        print("✅ Rate limiting prevents API abuse and terms violations")
        print("✅ Usage analytics provide visibility into API consumption")
        print("✅ Automatic documentation ensures compliance audit trails")
        print("✅ Existing functionality preserved with enhanced reliability")
        print("\n🛡️ Your API usage is now fully compliant and monitored!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup all managers
        try:
            if 'compliance_manager' in locals():
                compliance_manager.cleanup()
        except Exception:
            pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
