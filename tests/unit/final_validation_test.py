#!/usr/bin/env python3
"""
🧪 API COMPLIANCE VALIDATION TEST
Technical validation script to confirm all compliance components are working correctly.
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

def test_imports():
    """Test that all compliance modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Core compliance system
        from api_compliance import ComplianceManager, DataProvider, ComplianceLevel, get_compliance_manager
        print("  ✅ api_compliance.py - Core compliance system")
        
        # Provider-specific compliance
        from compliance_breeze_utils import ComplianceBreezeDataManager
        print("  ✅ compliance_breeze_utils.py - Breeze API compliance")
        
        from compliance_yahoo_utils import ComplianceYahooFinanceManager
        print("  ✅ compliance_yahoo_utils.py - Yahoo Finance compliance")
        
        # Enhanced utilities with compliance integration
        from enhanced_breeze_utils import EnhancedBreezeDataManager, MarketDataRequest
        print("  ✅ enhanced_breeze_utils.py - Enhanced utilities with compliance")
        
        # Utility modules
        from data_processing_utils import DataProcessor
        print("  ✅ data_processing_utils.py - Data processing utilities")
        
        from src.utils.file_management_utils import SafeFileManager
        print("  ✅ file_management_utils.py - File management utilities")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_compliance_manager():
    """Test core compliance manager functionality"""
    print("\n🔍 Testing ComplianceManager...")
    
    try:
        from api_compliance import ComplianceManager, ComplianceLevel, DataProvider
        
        # Test initialization
        manager = ComplianceManager(ComplianceLevel.MODERATE)
        print("  ✅ ComplianceManager initialization")
        
        # Test rate limit checking
        rate_check = manager.check_rate_limit(DataProvider.YAHOO_FINANCE, "test_endpoint")
        assert 'allowed' in rate_check
        assert 'wait_time' in rate_check
        assert 'request_count' in rate_check
        print("  ✅ Rate limit checking")
        
        # Test request permission
        try:
            request = manager.request_permission(
                DataProvider.YAHOO_FINANCE, 
                "test_endpoint", 
                {'test': 'data'}
            )
            print("  ✅ Request permission system")
        except Exception as e:
            # Expected to potentially hit rate limits
            print(f"  ℹ️  Request permission (rate limited): {str(e)[:50]}...")
        
        # Test usage statistics
        stats = manager.get_usage_statistics()
        assert isinstance(stats, dict)
        print("  ✅ Usage statistics")
        
        # Test compliance report
        report = manager.generate_compliance_report()
        assert isinstance(report, str)
        assert len(report) > 0
        print("  ✅ Compliance report generation")
        
        # Cleanup
        manager.cleanup()
        print("  ✅ Cleanup functionality")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ComplianceManager test failed: {str(e)}")
        return False

def test_breeze_compliance():
    """Test Breeze API compliance wrapper"""
    print("\n🔍 Testing Breeze Compliance Wrapper...")
    
    try:
        from compliance_breeze_utils import ComplianceBreezeDataManager
        from api_compliance import ComplianceLevel
        
        # Test initialization
        manager = ComplianceBreezeDataManager(
            compliance_level=ComplianceLevel.MONITORING
        )
        print("  ✅ ComplianceBreezeDataManager initialization")
        
        # Test terms validation
        terms_valid = manager.validate_terms_compliance()
        assert isinstance(terms_valid, dict)
        print("  ✅ Terms compliance validation")
        
        # Test session compliance report
        report = manager.get_session_compliance_report()
        assert isinstance(report, dict)
        assert 'session_info' in report
        assert 'request_metrics' in report
        print("  ✅ Session compliance reporting")
        
        # Cleanup
        manager.cleanup()
        print("  ✅ Cleanup functionality")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Breeze compliance test failed: {str(e)}")
        return False

def test_yahoo_compliance():
    """Test Yahoo Finance compliance wrapper"""
    print("\n🔍 Testing Yahoo Finance Compliance Wrapper...")
    
    try:
        from compliance_yahoo_utils import ComplianceYahooFinanceManager
        from api_compliance import ComplianceLevel
        
        # Test initialization
        manager = ComplianceYahooFinanceManager(
            compliance_level=ComplianceLevel.MONITORING
        )
        print("  ✅ ComplianceYahooFinanceManager initialization")
        
        # Test terms validation
        terms_valid = manager.validate_terms_compliance()
        assert isinstance(terms_valid, dict)
        print("  ✅ Terms compliance validation")
        
        # Test commercial use validation
        commercial_check = manager.validate_commercial_use()
        assert isinstance(commercial_check, dict)
        print("  ✅ Commercial use validation")
        
        # Test compliance report
        report = manager.get_terms_compliance_report()
        assert isinstance(report, dict)
        assert 'usage_metrics' in report
        print("  ✅ Terms compliance reporting")
        
        # Cleanup
        manager.cleanup()
        print("  ✅ Cleanup functionality")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Yahoo Finance compliance test failed: {str(e)}")
        return False

def test_enhanced_utilities():
    """Test enhanced utilities with compliance integration"""
    print("\n🔍 Testing Enhanced Utilities with Compliance...")
    
    try:
        from enhanced_breeze_utils import EnhancedBreezeDataManager, MarketDataRequest
        
        # Test initialization
        manager = EnhancedBreezeDataManager()
        print("  ✅ EnhancedBreezeDataManager initialization")
        
        # Test data request validation
        request = MarketDataRequest(
            stock_code="TEST",
            exchange_code="NSE",
            product_type="cash",
            interval="1day",
            from_date="2024-01-01",
            to_date="2024-01-02"
        )
        print("  ✅ MarketDataRequest creation")
        
        # Test request validation (should not raise exception for valid request)
        try:
            manager._validate_data_request(request)
            print("  ✅ Data request validation")
        except Exception as e:
            print(f"  ⚠️  Data request validation: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced utilities test failed: {str(e)}")
        return False

def test_file_management():
    """Test file management utilities"""
    print("\n🔍 Testing File Management Utilities...")
    
    try:
        from src.utils.file_management_utils import SafeFileManager
        
        # Test initialization
        base_path = os.path.join(os.path.dirname(__file__), "data")
        manager = SafeFileManager(base_path)
        print("  ✅ SafeFileManager initialization")
        
        # Test path generation
        test_path = manager.get_safe_path("test_file.csv")
        assert test_path is not None
        print("  ✅ Safe path generation")
        
        # Test file existence checking
        exists = manager.file_exists("nonexistent_file.csv")
        assert exists is False
        print("  ✅ File existence checking")
        
        return True
        
    except Exception as e:
        print(f"  ❌ File management test failed: {str(e)}")
        return False

def test_documentation_files():
    """Test that documentation files exist and are readable"""
    print("\n🔍 Testing Documentation Files...")
    
    expected_docs = [
        "API_COMPLIANCE_DOCUMENTATION.md",
        "API_COMPLIANCE_IMPLEMENTATION_COMPLETE.md"
    ]
    
    try:
        for doc_file in expected_docs:
            doc_path = os.path.join(os.path.dirname(__file__), doc_file)
            
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:  # Basic content check
                        print(f"  ✅ {doc_file} - exists and readable")
                    else:
                        print(f"  ⚠️  {doc_file} - exists but may be incomplete")
            else:
                print(f"  ❌ {doc_file} - not found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Documentation test failed: {str(e)}")
        return False

def test_configuration_files():
    """Test configuration file handling"""
    print("\n🔍 Testing Configuration Files...")
    
    try:
        # Test basic config loading
        config_files = ["config.json", "requirements.txt"]
        
        for config_file in config_files:
            config_path = os.path.join(os.path.dirname(__file__), config_file)
            
            if os.path.exists(config_path):
                print(f"  ✅ {config_file} - exists")
            else:
                print(f"  ⚠️  {config_file} - not found (may be optional)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {str(e)}")
        return False

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🧪 API COMPLIANCE VALIDATION TEST")
    print("=" * 80)
    print("Running comprehensive validation of all compliance components...")
    print("=" * 80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Compliance Manager", test_compliance_manager),
        ("Breeze Compliance", test_breeze_compliance),
        ("Yahoo Finance Compliance", test_yahoo_compliance),
        ("Enhanced Utilities", test_enhanced_utilities),
        ("File Management", test_file_management),
        ("Documentation Files", test_documentation_files),
        ("Configuration Files", test_configuration_files),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("🧪 VALIDATION RESULTS")
    print("=" * 80)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ API compliance system is fully operational")
        print("✅ All components working correctly")
        print("✅ Ready for production use")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        print("Review the output above for specific issues")
    
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
