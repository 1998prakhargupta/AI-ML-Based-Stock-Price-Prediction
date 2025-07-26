#!/usr/bin/env python3
"""Quick test for ModelManager"""

import sys
import traceback

print("=== ModelManager Test ===")

try:
    print("1. Importing Config...")
    from src.utils.app_config import Config
    print("✅ Config imported")
    
    print("2. Creating Config instance...")
    config = Config()
    print("✅ Config instance created")
    
    print("3. Getting data paths...")
    paths = config.get_data_paths()
    print(f"✅ Data paths: {paths}")
    
    print("4. Importing ModelManager...")
    from src.models.model_utils import ModelManager
    print("✅ ModelManager imported")
    
    print("5. Creating ModelManager instance...")
    manager = ModelManager()
    print("✅ ModelManager instance created")
    print(f"Model save path: {manager.model_save_path}")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
