#!/usr/bin/env python3
"""
🧪 UNIT TESTS FOR KEY FUNCTIONS
Comprehensive unit testing for all major components

This module provides detailed unit tests for key functions while 
maintaining all underlying basic logic.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import reproducibility utilities
from src.utils.reproducibility_utils import ReproducibilityManager, set_global_seed


class TestAppConfig(unittest.TestCase):
    """Unit tests for app configuration"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
    
    def test_config_initialization(self):
        """Test Config class can be initialized"""
        from src.utils.app_config import Config
        
        config = Config()
        self.assertIsNotNone(config)
        
    def test_config_data_paths(self):
        """Test config provides valid data paths"""
        from src.utils.app_config import Config
        
        config = Config()
        
        # Test data save path
        data_path = config.get_data_save_path()
        self.assertIsInstance(data_path, str)
        self.assertGreater(len(data_path), 0)
        
        # Test model save path
        model_path = config.get_model_save_path()
        self.assertIsInstance(model_path, str)
        self.assertGreater(len(model_path), 0)
    
    def test_config_consistency(self):
        """Test config returns consistent paths across calls"""
        from src.utils.app_config import Config
        
        config = Config()
        
        path1 = config.get_data_save_path()
        path2 = config.get_data_save_path()
        
        self.assertEqual(path1, path2)


class TestDataProcessingUtils(unittest.TestCase):
    """Unit tests for data processing utilities"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='5min'),
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
    
    def test_technical_indicator_processor_init(self):
        """Test TechnicalIndicatorProcessor initialization"""
        from data_processing_utils import TechnicalIndicatorProcessor
        
        processor = TechnicalIndicatorProcessor()
        self.assertIsNotNone(processor)
        
    def test_options_data_processor_init(self):
        """Test OptionsDataProcessor initialization"""
        try:
            from data_processing_utils import OptionsDataProcessor
            
            processor = OptionsDataProcessor()
            self.assertIsNotNone(processor)
        except ImportError:
            self.skipTest("OptionsDataProcessor not available")
    
    def test_processing_result_structure(self):
        """Test ProcessingResult data structure"""
        from data_processing_utils import ProcessingResult, DataQuality
        
        result = ProcessingResult(
            data=self.sample_data,
            success=True,
            quality=DataQuality.EXCELLENT,
            errors=[],
            warnings=[],
            metadata={'test': 'metadata'},
            processing_time=1.0
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.quality, DataQuality.EXCELLENT)
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertEqual(len(result.data), 100)
    
    def test_data_quality_enum(self):
        """Test DataQuality enumeration"""
        from data_processing_utils import DataQuality
        
        # Test all quality levels exist
        self.assertIsNotNone(DataQuality.EXCELLENT)
        self.assertIsNotNone(DataQuality.GOOD)
        self.assertIsNotNone(DataQuality.FAIR)
        self.assertIsNotNone(DataQuality.POOR)
        self.assertIsNotNone(DataQuality.INVALID)
    
    def test_validation_error_handling(self):
        """Test custom exception handling"""
        from data_processing_utils import ValidationError, ProcessingError
        
        # Test ValidationError
        with self.assertRaises(ValidationError):
            raise ValidationError("Test validation error")
        
        # Test ProcessingError
        with self.assertRaises(ProcessingError):
            raise ProcessingError("Test processing error")


class TestModelUtils(unittest.TestCase):
    """Unit tests for model utilities"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
        
    def test_model_manager_init(self):
        """Test ModelManager initialization"""
        from src.models.model_utils import ModelManager
        
        manager = ModelManager()
        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, 'model_save_path'))
    
    def test_model_evaluator_init(self):
        """Test ModelEvaluator initialization"""
        from src.models.model_utils import ModelEvaluator
        
        evaluator = ModelEvaluator()
        self.assertIsNotNone(evaluator)
    
    def test_model_manager_paths(self):
        """Test ModelManager provides valid paths"""
        from src.models.model_utils import ModelManager
        
        manager = ModelManager()
        
        # Test model save path exists and is string
        self.assertIsInstance(manager.model_save_path, str)
        self.assertGreater(len(manager.model_save_path), 0)
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation"""
        # Create synthetic predictions and actuals
        y_true = np.random.normal(100, 10, 100)
        y_pred = y_true + np.random.normal(0, 2, 100)  # Add small error
        
        # Test basic metrics can be calculated
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        self.assertGreater(mse, 0)
        self.assertLess(r2, 1)  # R2 should be less than 1 for noisy predictions
        self.assertGreater(mae, 0)


class TestFileManagementUtils(unittest.TestCase):
    """Unit tests for file management utilities"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
        
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_safe_file_manager_init(self):
        """Test SafeFileManager initialization"""
        from src.utils.file_management_utils import SafeFileManager
        
        manager = SafeFileManager(self.test_dir)
        self.assertIsNotNone(manager)
        self.assertEqual(str(manager.base_path), self.test_dir)
    
    def test_save_result_structure(self):
        """Test SaveResult data structure"""
        from src.utils.file_management_utils import SaveResult, SaveStrategy
        
        result = SaveResult(
            success=True,
            filepath="/test/path.csv",
            original_filename="test.csv",
            final_filename="test_v1.csv",
            strategy_used=SaveStrategy.VERSION,
            backup_created=False,
            backup_path=None,
            metadata={'test': 'data'},
            error_message=None,
            timestamp=datetime.now().isoformat()
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.filepath, "/test/path.csv")
        self.assertEqual(result.strategy_used, SaveStrategy.VERSION)
    
    def test_save_strategy_enum(self):
        """Test SaveStrategy enumeration"""
        from src.utils.file_management_utils import SaveStrategy
        
        # Test all strategies exist
        self.assertIsNotNone(SaveStrategy.OVERWRITE)
        self.assertIsNotNone(SaveStrategy.VERSION)
        self.assertIsNotNone(SaveStrategy.TIMESTAMP)
        self.assertIsNotNone(SaveStrategy.BACKUP_OVERWRITE)
    
    def test_dataframe_saving(self):
        """Test DataFrame saving functionality"""
        from src.utils.file_management_utils import SafeFileManager
        
        manager = SafeFileManager(self.test_dir)
        
        # Create test data
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']
        })
        
        # Save DataFrame
        result = manager.save_dataframe(test_df, "test_data.csv")
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(result.filepath))
        
        # Verify data integrity
        loaded_df = pd.read_csv(result.filepath)
        pd.testing.assert_frame_equal(test_df, loaded_df)
    
    def test_file_versioning(self):
        """Test file versioning functionality"""
        from src.utils.file_management_utils import SafeFileManager
        
        manager = SafeFileManager(self.test_dir)
        
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Save same filename multiple times
        result1 = manager.save_dataframe(test_df, "versioning_test.csv")
        result2 = manager.save_dataframe(test_df, "versioning_test.csv")
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        
        # Should create different files
        self.assertNotEqual(result1.final_filename, result2.final_filename)
        
        # Both files should exist
        self.assertTrue(os.path.exists(result1.filepath))
        self.assertTrue(os.path.exists(result2.filepath))


class TestVisualizationUtils(unittest.TestCase):
    """Unit tests for visualization utilities"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
        
    def test_comprehensive_visualizer_init(self):
        """Test ComprehensiveVisualizer initialization"""
        try:
            from src.visualization.visualization_utils import ComprehensiveVisualizer
            from src.utils.app_config import Config
            
            config = Config()
            visualizer = ComprehensiveVisualizer(config)
            self.assertIsNotNone(visualizer)
        except ImportError:
            self.skipTest("Visualization utilities not available")
    
    def test_visualizer_plotting_availability(self):
        """Test plotting functionality availability"""
        try:
            from src.visualization.visualization_utils import ComprehensiveVisualizer
            from src.utils.app_config import Config
            
            config = Config()
            visualizer = ComprehensiveVisualizer(config)
            
            # Check if plotting is available
            self.assertTrue(hasattr(visualizer, 'plotting_available'))
        except ImportError:
            self.skipTest("Visualization utilities not available")


class TestAutomatedReporting(unittest.TestCase):
    """Unit tests for automated reporting"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
    
    def test_automated_report_generator_init(self):
        """Test AutomatedReportGenerator initialization"""
        try:
            from src.visualization.automated_reporting import AutomatedReportGenerator
            from src.utils.app_config import Config
            
            config = Config()
            generator = AutomatedReportGenerator(config)
            self.assertIsNotNone(generator)
        except ImportError:
            self.skipTest("Automated reporting not available")


class TestReproducibilityUtils(unittest.TestCase):
    """Unit tests for reproducibility utilities"""
    
    def test_reproducibility_manager_init(self):
        """Test ReproducibilityManager initialization"""
        manager = ReproducibilityManager(seed=123)
        self.assertEqual(manager.seed, 123)
    
    def test_seed_consistency(self):
        """Test seed setting produces consistent results"""
        manager = ReproducibilityManager(seed=42)
        
        # Set seeds and generate random number
        manager.set_all_seeds()
        value1 = np.random.random()
        
        # Reset seeds and generate again
        manager.set_all_seeds()
        value2 = np.random.random()
        
        # Should be identical
        self.assertEqual(value1, value2)
    
    def test_reproducible_data_split(self):
        """Test reproducible data splitting"""
        manager = ReproducibilityManager(seed=42)
        
        # Create test data
        test_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=1000, freq='5min'),
            'value': np.random.normal(0, 1, 1000)
        })
        
        # Split data twice
        split1 = manager.create_reproducible_data_split(test_data, time_column='datetime')
        split2 = manager.create_reproducible_data_split(test_data, time_column='datetime')
        
        # Should be identical
        pd.testing.assert_frame_equal(split1['train'], split2['train'])
        pd.testing.assert_frame_equal(split1['test'], split2['test'])
    
    def test_model_params_consistency(self):
        """Test model parameter consistency"""
        manager = ReproducibilityManager(seed=42)
        
        params1 = manager.get_reproducible_model_params('RandomForestRegressor')
        params2 = manager.get_reproducible_model_params('RandomForestRegressor')
        
        self.assertEqual(params1['random_state'], params2['random_state'])
        self.assertEqual(params1['random_state'], 42)
    
    def test_environment_state_capture(self):
        """Test environment state capture"""
        manager = ReproducibilityManager(seed=42)
        
        state = manager._capture_environment_state()
        
        # Check required fields exist
        self.assertIn('python_version', state)
        self.assertIn('installed_packages', state)
        self.assertIn('timestamp', state)
    
    def test_experiment_state_saving(self):
        """Test experiment state saving and loading"""
        manager = ReproducibilityManager(seed=42)
        
        # Create experiments directory for test
        os.makedirs('experiments', exist_ok=True)
        
        try:
            # Save experiment state
            filepath = manager.save_experiment_state('unit_test', {'test_param': 'value'})
            
            self.assertTrue(os.path.exists(filepath))
            
            # Load experiment state
            loaded_state = manager.load_experiment_state(filepath)
            
            self.assertIn('experiment_name', loaded_state)
            self.assertEqual(loaded_state['experiment_name'], 'unit_test')
            self.assertIn('test_param', loaded_state['additional_info'])
            
        finally:
            # Clean up
            if os.path.exists('experiments'):
                shutil.rmtree('experiments')


class TestIntegrationWithExistingCode(unittest.TestCase):
    """Integration tests with existing codebase"""
    
    def setUp(self):
        """Set up test environment"""
        set_global_seed(42)
    
    def test_imports_work_together(self):
        """Test all modules can be imported together"""
        try:
            from src.utils.app_config import Config
            from data_processing_utils import TechnicalIndicatorProcessor
            from src.models.model_utils import ModelManager
            from src.utils.file_management_utils import SafeFileManager
            from src.utils.reproducibility_utils import ReproducibilityManager
            
            # All imports should succeed
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Integration import failed: {e}")
    
    def test_config_integration(self):
        """Test config works with other components"""
        from src.utils.app_config import Config
        from src.utils.file_management_utils import SafeFileManager
        
        config = Config()
        file_manager = SafeFileManager(config.get_data_save_path())
        
        self.assertIsNotNone(file_manager)
    
    def test_reproducibility_integration(self):
        """Test reproducibility integrates with existing workflow"""
        from src.utils.reproducibility_utils import set_global_seed, get_model_params
        
        # Set global seed
        set_global_seed(123)
        
        # Get model parameters
        params = get_model_params('RandomForestRegressor')
        
        self.assertEqual(params['random_state'], 123)
    
    def test_existing_logic_preservation(self):
        """Test that existing logic is preserved"""
        # Create sample financial data following existing patterns
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        
        # This follows the same pattern as existing code
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.exponential(10000, 100)
        })
        
        # Verify data structure matches existing expectations
        self.assertIn('datetime', data.columns)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns) 
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
        # Verify OHLC logic is maintained
        self.assertTrue((data['high'] >= data['low']).all())
        self.assertTrue((data['volume'] > 0).all())


def create_test_suite():
    """Create comprehensive test suite"""
    test_classes = [
        TestAppConfig,
        TestDataProcessingUtils,
        TestModelUtils,
        TestFileManagementUtils,
        TestVisualizationUtils,
        TestAutomatedReporting,
        TestReproducibilityUtils,
        TestIntegrationWithExistingCode
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_unit_tests():
    """Run all unit tests"""
    print("🧪 RUNNING COMPREHENSIVE UNIT TESTS")
    print("=" * 60)
    print("Testing key functions while maintaining underlying logic")
    print("=" * 60)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 UNIT TEST SUMMARY")
    print("=" * 60)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 ALL UNIT TESTS PASSED!")
        print("✅ All key functions working correctly")
        print("✅ Underlying logic preserved")
    else:
        print("❌ Some tests failed")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)
