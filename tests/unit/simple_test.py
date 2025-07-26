#!/usr/bin/env python3
"""Simple test to validate imports"""

print("Starting import test...")

try:
    from src.utils.app_config import Config
    print("✅ app_config imported successfully")
except Exception as e:
    print(f"❌ app_config failed: {e}")

try:
    from data_processing_utils import TechnicalIndicatorProcessor
    print("✅ data_processing_utils imported successfully")
except Exception as e:
    print(f"❌ data_processing_utils failed: {e}")

try:
    from src.models.model_utils import ModelManager
    print("✅ model_utils imported successfully")
except Exception as e:
    print(f"❌ model_utils failed: {e}")

print("Import test completed!")
