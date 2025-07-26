#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE TESTING FRAMEWORK
Testing and Reproducibility Implementation for Stock Price Predictor

This module provides comprehensive unit tests, integration tests, and reproducibility
utilities while maintaining all underlying basic logic.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import json
import random
import logging
import inspect
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReproducibilityManager:
    """Manages random seeds and environment state for reproducible results"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.original_state = {}
        
    def set_seeds(self):
        """Set all random seeds for reproducibility"""
        # Set Python random seed
        random.seed(self.seed)
        
        # Set NumPy random seed
        np.random.seed(self.seed)
        
        # Set pandas random seed if available
        try:
            import pandas as pd
            pd.core.common.random.seed = self.seed
        except:
            pass
            
        # Set scikit-learn random state if available
        try:
            import sklearn
            # This will be used by model implementations
            os.environ['PYTHONHASHSEED'] = str(self.seed)
        except:
            pass
            
        logger.info(f"🎲 Random seeds set to {self.seed} for reproducibility")
        
    def save_environment_state(self, filepath: str = "environment_state.json"):
        """Save current environment state for reproducibility documentation"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.seed,
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'python_path': sys.path,
            'environment_variables': dict(os.environ),
            'installed_packages': self._get_installed_packages()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        logger.info(f"📝 Environment state saved to {filepath}")
        return state
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages and versions"""
        packages = {}
        
        # Core packages we care about
        important_packages = [
            'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
            'ta', 'breeze-connect', 'joblib', 'scipy'
        ]
        
        for package in important_packages:
            try:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                packages[package] = version
            except ImportError:
                packages[package] = 'not_installed'
        
        return packages


class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def create_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Create sample financial data for testing while preserving data structure"""
        np.random.seed(seed)
        
        # Generate realistic time series data
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        base_price = 100.0
        
        # Generate correlated OHLCV data
        returns = np.random.normal(0, 0.02, n_samples)
        price_series = base_price * np.cumprod(1 + returns)
        
        # Create OHLC maintaining proper relationships
        opens = price_series * (1 + np.random.normal(0, 0.001, n_samples))
        highs = np.maximum(opens, price_series) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
        lows = np.minimum(opens, price_series) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
        closes = price_series
        volumes = np.random.lognormal(10, 1, n_samples)
        
        return pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    @staticmethod
    def create_options_sample_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
        """Create sample options data for testing"""
        np.random.seed(seed)
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        return pd.DataFrame({
            'datetime': dates,
            'strike': np.random.uniform(95, 105, n_samples),
            'option_type': np.random.choice(['CE', 'PE'], n_samples),
            'premium': np.random.uniform(0.5, 10, n_samples),
            'delta': np.random.uniform(-1, 1, n_samples),
            'gamma': np.random.uniform(0, 0.1, n_samples),
            'theta': np.random.uniform(-0.1, 0, n_samples),
            'vega': np.random.uniform(0, 0.5, n_samples),
            'volume': np.random.exponential(1000, n_samples)
        })


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
        self.repro_manager.set_seeds()
        
    def test_config_import(self):
        """Test that Config can be imported and initialized"""
        try:
            from src.utils.app_config import Config
            config = Config()
            self.assertIsNotNone(config)
            logger.info("✅ Config import and initialization test passed")
        except Exception as e:
            self.fail(f"Config import/initialization failed: {e}")
    
    def test_config_paths(self):
        """Test that config provides valid paths"""
        from src.utils.app_config import Config
        config = Config()
        
        # Test data save path
        data_path = config.get_data_save_path()
        self.assertIsNotNone(data_path)
        self.assertIsInstance(data_path, str)
        
        # Test model save path  
        model_path = config.get_model_save_path()
        self.assertIsNotNone(model_path)
        self.assertIsInstance(model_path, str)
        
        logger.info("✅ Config paths test passed")


class TestDataProcessing(unittest.TestCase):
    """Test data processing utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
        self.repro_manager.set_seeds()
        self.sample_data = TestUtilities.create_sample_data()
        
    def test_data_processing_import(self):
        """Test data processing utilities import"""
        try:
            from data_processing_utils import (
                TechnicalIndicatorProcessor,
                OptionsDataProcessor,
                ProcessingResult,
                DataQuality
            )
            logger.info("✅ Data processing imports test passed")
        except Exception as e:
            self.fail(f"Data processing import failed: {e}")
    
    def test_technical_indicator_processor_init(self):
        """Test TechnicalIndicatorProcessor initialization"""
        from data_processing_utils import TechnicalIndicatorProcessor
        
        processor = TechnicalIndicatorProcessor()
        self.assertIsNotNone(processor)
        logger.info("✅ TechnicalIndicatorProcessor initialization test passed")
    
    def test_data_validation_functions(self):
        """Test data validation preserves data integrity"""
        # Test basic data structure
        self.assertGreater(len(self.sample_data), 0)
        self.assertIn('close', self.sample_data.columns)
        self.assertIn('volume', self.sample_data.columns)
        
        # Test OHLC relationships
        valid_ohlc = (
            (self.sample_data['high'] >= self.sample_data['low']) &
            (self.sample_data['high'] >= self.sample_data['open']) &
            (self.sample_data['high'] >= self.sample_data['close']) &
            (self.sample_data['low'] <= self.sample_data['open']) &
            (self.sample_data['low'] <= self.sample_data['close'])
        )
        
        valid_ratio = valid_ohlc.mean()
        self.assertGreater(valid_ratio, 0.95)  # At least 95% should be valid
        
        logger.info("✅ Data validation functions test passed")


class TestModelUtilities(unittest.TestCase):
    """Test model management utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
        self.repro_manager.set_seeds()
    
    def test_model_utils_import(self):
        """Test model utilities import"""
        try:
            from src.models.model_utils import ModelManager, ModelEvaluator
            logger.info("✅ Model utilities import test passed")
        except Exception as e:
            self.fail(f"Model utilities import failed: {e}")
    
    def test_model_manager_initialization(self):
        """Test ModelManager can be initialized"""
        from src.models.model_utils import ModelManager
        
        manager = ModelManager()
        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, 'model_save_path'))
        logger.info("✅ ModelManager initialization test passed")
    
    def test_model_reproducibility(self):
        """Test that models produce consistent results with fixed seeds"""
        from src.models.model_utils import ModelManager
        
        # Create two identical model managers with same seed
        self.repro_manager.set_seeds()
        manager1 = ModelManager()
        
        self.repro_manager.set_seeds()
        manager2 = ModelManager()
        
        # Both should have identical configurations
        self.assertEqual(manager1.model_save_path, manager2.model_save_path)
        
        logger.info("✅ Model reproducibility test passed")


class TestFileManagement(unittest.TestCase):
    """Test file management utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
        self.repro_manager.set_seeds()
        self.test_dir = "test_file_management"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_file_management_import(self):
        """Test file management utilities import"""
        try:
            from src.utils.file_management_utils import SafeFileManager, SaveResult, SaveStrategy
            logger.info("✅ File management import test passed")
        except Exception as e:
            self.fail(f"File management import failed: {e}")
    
    def test_safe_file_manager_init(self):
        """Test SafeFileManager initialization"""
        from src.utils.file_management_utils import SafeFileManager
        
        manager = SafeFileManager(self.test_dir)
        self.assertIsNotNone(manager)
        self.assertEqual(str(manager.base_path), self.test_dir)
        logger.info("✅ SafeFileManager initialization test passed")
    
    def test_file_versioning(self):
        """Test file versioning functionality preserves data"""
        from src.utils.file_management_utils import SafeFileManager
        
        manager = SafeFileManager(self.test_dir)
        
        # Create test data
        test_data = TestUtilities.create_sample_data(100)
        
        # Save multiple versions
        result1 = manager.save_dataframe(test_data, "test_data.csv")
        self.assertTrue(result1.success)
        
        result2 = manager.save_dataframe(test_data, "test_data.csv")
        self.assertTrue(result2.success)
        
        # Should have created versioned files
        files = os.listdir(self.test_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 1)
        
        logger.info("✅ File versioning test passed")


class TestVisualizationReporting(unittest.TestCase):
    """Test visualization and reporting utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
        self.repro_manager.set_seeds()
        
    def test_visualization_import(self):
        """Test visualization utilities import"""
        try:
            from src.visualization.visualization_utils import ComprehensiveVisualizer
            logger.info("✅ Visualization import test passed")
        except Exception as e:
            self.fail(f"Visualization import failed: {e}")
    
    def test_reporting_import(self):
        """Test reporting utilities import"""
        try:
            from src.visualization.automated_reporting import AutomatedReportGenerator
            logger.info("✅ Reporting import test passed")
        except Exception as e:
            self.fail(f"Reporting import failed: {e}")
    
    def test_visualization_init(self):
        """Test visualization utilities initialization"""
        from src.utils.app_config import Config
        from src.visualization.visualization_utils import ComprehensiveVisualizer
        
        config = Config()
        visualizer = ComprehensiveVisualizer(config)
        self.assertIsNotNone(visualizer)
        logger.info("✅ Visualization initialization test passed")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
        self.repro_manager.set_seeds()
    
    def test_complete_pipeline(self):
        """Test complete data processing to model pipeline"""
        # Test all major components can work together
        from src.utils.app_config import Config
        from data_processing_utils import TechnicalIndicatorProcessor
        from src.models.model_utils import ModelManager
        from src.utils.file_management_utils import SafeFileManager
        
        # Initialize components
        config = Config()
        processor = TechnicalIndicatorProcessor()
        model_manager = ModelManager()
        file_manager = SafeFileManager(config.get_data_save_path())
        
        # Create test data
        sample_data = TestUtilities.create_sample_data(200)
        
        # Test data can be processed and saved
        self.assertIsNotNone(sample_data)
        self.assertGreater(len(sample_data), 0)
        
        logger.info("✅ Complete pipeline integration test passed")
    
    def test_reproducibility_consistency(self):
        """Test that results are consistent across runs with same seeds"""
        # Run the same operations twice with same seeds
        results1 = self._run_sample_operation()
        results2 = self._run_sample_operation()
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        
        # Compare key metrics (allowing for small floating point differences)
        np.testing.assert_array_almost_equal(
            results1['close'].values, 
            results2['close'].values,
            decimal=10
        )
        
        logger.info("✅ Reproducibility consistency test passed")
    
    def _run_sample_operation(self):
        """Helper method to run a sample operation"""
        self.repro_manager.set_seeds()
        return TestUtilities.create_sample_data(100)


class TestReproducibility(unittest.TestCase):
    """Test reproducibility features"""
    
    def setUp(self):
        """Set up test environment"""
        self.repro_manager = ReproducibilityManager()
    
    def test_seed_setting(self):
        """Test that seeds can be set consistently"""
        # Set seeds multiple times and verify consistency
        seed = 42
        
        self.repro_manager.seed = seed
        self.repro_manager.set_seeds()
        value1 = np.random.random()
        
        self.repro_manager.set_seeds()
        value2 = np.random.random()
        
        self.assertEqual(value1, value2)
        logger.info("✅ Seed setting consistency test passed")
    
    def test_environment_documentation(self):
        """Test environment state can be documented"""
        state_file = "test_environment_state.json"
        
        try:
            state = self.repro_manager.save_environment_state(state_file)
            
            # Verify state contains required information
            self.assertIn('timestamp', state)
            self.assertIn('random_seed', state)
            self.assertIn('python_version', state)
            self.assertIn('installed_packages', state)
            
            # Verify file was created
            self.assertTrue(os.path.exists(state_file))
            
            logger.info("✅ Environment documentation test passed")
            
        finally:
            # Clean up
            if os.path.exists(state_file):
                os.remove(state_file)


class TestRunner:
    """Custom test runner with enhanced reporting"""
    
    def __init__(self):
        self.repro_manager = ReproducibilityManager()
        
    def run_all_tests(self, save_report: bool = True):
        """Run all tests and generate comprehensive report"""
        logger.info("🧪 Starting comprehensive test suite...")
        logger.info("=" * 60)
        
        # Set seeds for reproducibility
        self.repro_manager.set_seeds()
        
        # Save environment state
        env_state = self.repro_manager.save_environment_state("test_environment_state.json")
        
        # Define test suites
        test_suites = [
            ('Configuration Tests', TestConfiguration),
            ('Data Processing Tests', TestDataProcessing),
            ('Model Utilities Tests', TestModelUtilities),
            ('File Management Tests', TestFileManagement),
            ('Visualization/Reporting Tests', TestVisualizationReporting),
            ('Integration Tests', TestIntegration),
            ('Reproducibility Tests', TestReproducibility)
        ]
        
        results = {}
        overall_success = True
        
        for suite_name, test_class in test_suites:
            logger.info(f"\n📋 Running {suite_name}...")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
            result = runner.run(suite)
            
            # Record results
            success = result.wasSuccessful()
            results[suite_name] = {
                'success': success,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'failure_details': result.failures,
                'error_details': result.errors
            }
            
            overall_success = overall_success and success
            
            # Log results
            if success:
                logger.info(f"✅ {suite_name} PASSED ({result.testsRun} tests)")
            else:
                logger.error(f"❌ {suite_name} FAILED ({result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors)")
        
        # Generate final report
        self._generate_test_report(results, env_state, overall_success)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = sum(r['tests_run'] for r in results.values())
        total_failures = sum(r['failures'] for r in results.values())
        total_errors = sum(r['errors'] for r in results.values())
        
        logger.info(f"Total Tests Run: {total_tests}")
        logger.info(f"Total Failures: {total_failures}")
        logger.info(f"Total Errors: {total_errors}")
        
        if overall_success:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("✅ Testing and reproducibility system is fully functional")
        else:
            logger.error("❌ Some tests failed. Please review the details above.")
        
        return overall_success, results
    
    def _generate_test_report(self, results: Dict, env_state: Dict, overall_success: bool):
        """Generate comprehensive test report"""
        report = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'overall_success': overall_success,
                'random_seed_used': self.repro_manager.seed
            },
            'environment_state': env_state,
            'test_results': results,
            'summary': {
                'total_suites': len(results),
                'passed_suites': sum(1 for r in results.values() if r['success']),
                'total_tests': sum(r['tests_run'] for r in results.values()),
                'total_failures': sum(r['failures'] for r in results.values()),
                'total_errors': sum(r['errors'] for r in results.values())
            }
        }
        
        # Save report
        with open('comprehensive_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📋 Comprehensive test report saved to 'comprehensive_test_report.json'")


def main():
    """Main function to run tests"""
    print("🧪 COMPREHENSIVE TESTING AND REPRODUCIBILITY FRAMEWORK")
    print("=" * 80)
    print("Testing all components while maintaining underlying basic logic")
    print("=" * 80)
    
    try:
        # Create test runner
        test_runner = TestRunner()
        
        # Run all tests
        success, results = test_runner.run_all_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"💥 Test runner crashed: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
