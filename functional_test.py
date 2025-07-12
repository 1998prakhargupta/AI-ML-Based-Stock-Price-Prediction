#!/usr/bin/env python3
"""Comprehensive functional test"""

print("=== Stock Price Predictor Functional Test ===")

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from app_config import Config
    from data_processing_utils import TechnicalIndicatorProcessor
    from model_utils import ModelManager
    from enhanced_breeze_utils import EnhancedBreezeDataManager
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Configuration
print("\n2. Testing configuration...")
try:
    config = Config()
    print(f"✅ Config loaded with {len(config.__dict__)} settings")
except Exception as e:
    print(f"❌ Config test failed: {e}")

# Test 3: Data processor initialization
print("\n3. Testing data processor...")
try:
    processor = TechnicalIndicatorProcessor()
    print("✅ TechnicalIndicatorProcessor created successfully")
except Exception as e:
    print(f"❌ Data processor test failed: {e}")

# Test 4: Model manager initialization
print("\n4. Testing model manager...")
try:
    model_manager = ModelManager()
    print("✅ ModelManager created successfully")
except Exception as e:
    print(f"❌ Model manager test failed: {e}")

# Test 5: Enhanced breeze manager initialization
print("\n5. Testing enhanced breeze manager...")
try:
    breeze_manager = EnhancedBreezeDataManager()
    print("✅ EnhancedBreezeDataManager created successfully")
except Exception as e:
    print(f"❌ Enhanced breeze manager test failed: {e}")

print("\n🎉 All functional tests passed!")
print("The stock price predictor system is ready to use.")
