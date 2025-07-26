#!/usr/bin/env python3
"""
Test script for the enhanced modular setup to verify error handling and logging.
This script tests all components to ensure they work as expected.
"""

import logging
import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup comprehensive logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_enhanced_setup.log')
        ]
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test that all enhanced utilities can be imported"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("🧪 Testing imports...")
        
        from src.utils.app_config import Config
        logger.info("✅ Config imported successfully")
        
        from data_processing_utils import (
            TechnicalIndicatorProcessor, 
            OptionsDataProcessor,
            ProcessingResult,
            ValidationError,
            ProcessingError,
            DataQuality
        )
        logger.info("✅ Data processing utilities imported successfully")
        
        from enhanced_breeze_utils import (
            EnhancedBreezeDataManager,
            OptionChainAnalyzer,
            MarketDataRequest,
            APIResponse
        )
        logger.info("✅ Enhanced Breeze utilities imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during import: {str(e)}")
        return False

def test_utility_initialization():
    """Test that enhanced utilities can be initialized"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("🧪 Testing utility initialization...")
        
        from src.utils.app_config import Config
        from data_processing_utils import TechnicalIndicatorProcessor, OptionsDataProcessor
        from enhanced_breeze_utils import EnhancedBreezeDataManager, OptionChainAnalyzer
        
        # Test Config
        config = Config()
        logger.info("✅ Config initialized successfully")
        
        # Test processors
        indicator_processor = TechnicalIndicatorProcessor()
        logger.info("✅ TechnicalIndicatorProcessor initialized successfully")
        
        options_processor = OptionsDataProcessor()
        logger.info("✅ OptionsDataProcessor initialized successfully")
        
        # Test enhanced managers (these might fail if credentials aren't available)
        try:
            data_manager = EnhancedBreezeDataManager()
            logger.info("✅ EnhancedBreezeDataManager initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ EnhancedBreezeDataManager initialization failed (expected without credentials): {str(e)}")
        
        try:
            option_analyzer = OptionChainAnalyzer()
            logger.info("✅ OptionChainAnalyzer initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ OptionChainAnalyzer initialization failed (expected without credentials): {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility initialization failed: {str(e)}")
        return False

def test_error_handling():
    """Test that error handling works as expected"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("🧪 Testing error handling...")
        
        from data_processing_utils import ValidationError, ProcessingError, ProcessingResult, DataQuality
        
        # Test custom exceptions
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            logger.info(f"✅ ValidationError caught successfully: {str(e)}")
        
        try:
            raise ProcessingError("Test processing error")
        except ProcessingError as e:
            logger.info(f"✅ ProcessingError caught successfully: {str(e)}")
        
        # Test ProcessingResult data structures
        success_result = ProcessingResult(
            data=pd.DataFrame({'test': [1, 2, 3]}),
            success=True,
            quality=DataQuality.EXCELLENT,
            errors=[],
            warnings=[],
            metadata={"test": "success"},
            processing_time=0.5
        )
        logger.info(f"✅ Success ProcessingResult created successfully: success={success_result.success}")
        
        error_result = ProcessingResult(
            data=pd.DataFrame(),
            success=False,
            quality=DataQuality.INVALID,
            errors=["Test error"],
            warnings=[],
            metadata={"error_type": "test"},
            processing_time=0.1
        )
        logger.info(f"✅ Error ProcessingResult created successfully: success={error_result.success}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {str(e)}")
        return False

def test_data_structures():
    """Test that data structures work correctly"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("🧪 Testing data structures...")
        
        from enhanced_breeze_utils import MarketDataRequest, APIResponse
        from data_processing_utils import DataQuality
        
        # Test MarketDataRequest
        request = MarketDataRequest(
            stock_code="TCS",
            exchange_code="NSE",
            product_type="cash",
            interval="5minute",
            from_date="2024-01-01T09:00:00.000Z",
            to_date="2024-01-31T15:30:00.000Z"
        )
        logger.info(f"✅ MarketDataRequest created successfully: {request.stock_code}")
        
        # Test APIResponse
        response = APIResponse(
            success=True,
            data=pd.DataFrame({"test": [1, 2, 3]}),
            errors=[],
            warnings=[],
            metadata={"timestamp": datetime.now().isoformat()},
            response_time=0.5
        )
        logger.info(f"✅ APIResponse created successfully: success={response.success}")
        
        # Test DataQuality enum
        quality = DataQuality.EXCELLENT
        logger.info(f"✅ DataQuality enum works: {quality}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data structures test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    logger = setup_logging()
    logger.info("🚀 Starting comprehensive enhanced utilities test")
    logger.info("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Utility Initialization Test", test_utility_initialization),
        ("Error Handling Test", test_error_handling),
        ("Data Structures Test", test_data_structures)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Enhanced modular setup is working correctly.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
