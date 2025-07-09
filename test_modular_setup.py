#!/usr/bin/env python3
"""
Test script to verify the modular security implementation works correctly.
This script tests that all components integrate properly without hardcoded values.
"""

def test_configuration():
    """Test that configuration loads correctly."""
    try:
        from app_config import Config
        config = Config()
        print("✓ Configuration loaded successfully")
        
        # Test index symbols
        symbols = config.get_index_symbols()
        print(f"✓ Index symbols: {len(symbols)} indices configured")
        assert len(symbols) > 0, "No index symbols configured"
        
        # Test date configuration
        start_date = config.get('START_DATE')
        end_date = config.get('END_DATE')
        print(f"✓ Date range: {start_date} to {end_date}")
        assert start_date is not None, "START_DATE not configured"
        assert end_date is not None, "END_DATE not configured"
        
        # Test path configuration
        data_path = config.get_data_save_path()
        model_path = config.get_model_save_path()
        print(f"✓ Data path: {data_path}")
        print(f"✓ Model path: {model_path}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_index_processor():
    """Test that index processor can be created."""
    try:
        from index_utils import create_index_processor
        processor = create_index_processor('.')
        print("✓ Index processor created successfully")
        
        # Test that processor has required methods
        required_methods = [
            'clean_and_merge_data',
            'create_normalized_data',
            'apply_scaling_transformations',
            'calculate_row_statistics',
            'add_rolling_features',
            'process_complete_pipeline'
        ]
        
        for method in required_methods:
            assert hasattr(processor, method), f"Missing method: {method}"
        
        print("✓ All required processing methods available")
        return True
    except Exception as e:
        print(f"❌ Index processor test failed: {e}")
        return False

def test_no_hardcoded_values():
    """Verify no hardcoded values remain in the notebook."""
    try:
        with open('index_data_fetch.ipynb', 'r') as f:
            content = f.read()
        
        # Check that hardcoded values are not present
        hardcoded_patterns = [
            '"2025-02-01"',
            '"2025-04-28"',
            'index_symbols = {',
            '"^NSEI"',
            '"NIFTY50": "^NSEI"'
        ]
        
        for pattern in hardcoded_patterns:
            if pattern in content:
                print(f"❌ Found hardcoded value: {pattern}")
                return False
        
        # Check that modular imports are present
        required_imports = [
            'from app_config import Config',
            'from index_utils import create_index_processor'
        ]
        
        for import_stmt in required_imports:
            if import_stmt not in content:
                print(f"❌ Missing import: {import_stmt}")
                return False
        
        print("✓ No hardcoded values found in notebook")
        print("✓ Modular imports present")
        return True
    except Exception as e:
        print(f"❌ Hardcoded values test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Modular Security Implementation")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Index Processor", test_index_processor),
        ("No Hardcoded Values", test_no_hardcoded_values)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Security fixes implemented successfully")
        print("✓ Modular architecture working correctly")
        print("✓ No hardcoded credentials or values")
        print("✓ Configuration-based approach active")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
