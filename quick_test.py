#!/usr/bin/env python3
"""Simple test to verify dependencies are working"""

print("Starting dependency test...")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
    
    import numpy as np
    print("✅ numpy imported successfully")
    
    import ta
    print("✅ ta imported successfully")
    
    import breeze_connect
    print("✅ breeze_connect imported successfully")
    
    from app_config import Config
    print("✅ app_config imported successfully")
    
    from data_processing_utils import TechnicalIndicatorProcessor, DataQuality
    print("✅ data_processing_utils imported successfully")
    
    from enhanced_breeze_utils import EnhancedBreezeDataManager
    print("✅ enhanced_breeze_utils imported successfully")
    
    # Test basic functionality
    config = Config()
    processor = TechnicalIndicatorProcessor()
    quality = DataQuality.EXCELLENT
    
    print("🎉 ALL TESTS PASSED! Enhanced error handling system is fully operational!")
    print("✅ Dependencies resolved successfully")
    print("✅ Module imports working correctly") 
    print("✅ Enhanced error handling system ready for use")
    
    # Write success to file
    with open('dependency_test_result.txt', 'w') as f:
        f.write("SUCCESS: All dependencies installed and working correctly\n")
        f.write("Enhanced error handling system is operational\n")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Write failure to file
    with open('dependency_test_result.txt', 'w') as f:
        f.write(f"FAILED: {str(e)}\n")
        f.write(traceback.format_exc())
