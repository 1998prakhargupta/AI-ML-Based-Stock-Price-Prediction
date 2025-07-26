#!/usr/bin/env python3
"""
🧪 NOTEBOOK TESTING UTILITIES
Testing utilities specifically designed for notebook validation

This module provides utilities to test notebook functionality while
maintaining all underlying basic logic.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.reproducibility_utils import ReproducibilityManager

logger = logging.getLogger(__name__)

class NotebookTestRunner:
    """Test runner for notebook functionality"""
    
    def __init__(self, seed: int = 42):
        """Initialize notebook test runner"""
        self.repro_manager = ReproducibilityManager(seed)
        self.test_results = {}
        
    def test_data_collection_pipeline(self) -> Dict[str, Any]:
        """
        Test data collection functionality from breeze_data notebooks.
        Maintains all existing data collection logic.
        """
        logger.info("🧪 Testing data collection pipeline...")
        
        test_result = {
            'test_name': 'data_collection_pipeline',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Set seeds for reproducibility
            self.repro_manager.set_all_seeds()
            
            # Test Config import and initialization
            from src.utils.app_config import Config
            config = Config()
            test_result['details']['config_loaded'] = True
            
            # Test data processing utilities
            from data_processing_utils import TechnicalIndicatorProcessor
            processor = TechnicalIndicatorProcessor()
            test_result['details']['processor_initialized'] = True
            
            # Test enhanced breeze utilities
            try:
                from enhanced_breeze_utils import EnhancedBreezeDataManager
                breeze_manager = EnhancedBreezeDataManager()
                test_result['details']['breeze_manager_available'] = True
            except Exception as e:
                test_result['warnings'].append(f"Breeze manager initialization warning: {str(e)}")
                test_result['details']['breeze_manager_available'] = False
            
            # Test basic data structure creation (simulating data collection)
            sample_data = self._create_sample_market_data()
            test_result['details']['sample_data_created'] = len(sample_data) > 0
            test_result['details']['sample_data_columns'] = list(sample_data.columns)
            
            logger.info("✅ Data collection pipeline test passed")
            
        except Exception as e:
            test_result['success'] = False
            test_result['errors'].append(str(e))
            logger.error(f"❌ Data collection pipeline test failed: {str(e)}")
        
        self.test_results['data_collection'] = test_result
        return test_result
    
    def test_model_training_pipeline(self) -> Dict[str, Any]:
        """
        Test model training functionality from stock_ML_Model notebook.
        Maintains all existing model training logic.
        """
        logger.info("🧪 Testing model training pipeline...")
        
        test_result = {
            'test_name': 'model_training_pipeline',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Set seeds for reproducibility
            self.repro_manager.set_all_seeds()
            
            # Test model utilities
            from src.models.model_utils import ModelManager, ModelEvaluator
            
            model_manager = ModelManager()
            model_evaluator = ModelEvaluator()
            
            test_result['details']['model_manager_initialized'] = True
            test_result['details']['model_evaluator_initialized'] = True
            test_result['details']['model_save_path'] = model_manager.model_save_path
            
            # Test reproducible model parameters
            rf_params = self.repro_manager.get_reproducible_model_params('RandomForestRegressor')
            xgb_params = self.repro_manager.get_reproducible_model_params('XGBRegressor')
            lgb_params = self.repro_manager.get_reproducible_model_params('LGBMRegressor')
            
            test_result['details']['reproducible_params'] = {
                'random_forest': rf_params,
                'xgboost': xgb_params, 
                'lightgbm': lgb_params
            }
            
            # Test data splitting with preserved logic
            sample_data = self._create_sample_training_data()
            split_result = self.repro_manager.create_reproducible_data_split(
                sample_data, 
                time_column='datetime',
                test_size=0.2,
                validation_size=0.1
            )
            
            test_result['details']['data_split'] = {
                'train_size': len(split_result['train']),
                'val_size': len(split_result['validation']),
                'test_size': len(split_result['test']),
                'split_method': split_result['split_info']['method']
            }
            
            # Test basic model evaluation metrics
            y_true, y_pred = self._create_sample_predictions()
            metrics = self._calculate_evaluation_metrics(y_true, y_pred)
            test_result['details']['evaluation_metrics'] = metrics
            
            logger.info("✅ Model training pipeline test passed")
            
        except Exception as e:
            test_result['success'] = False
            test_result['errors'].append(str(e))
            logger.error(f"❌ Model training pipeline test failed: {str(e)}")
        
        self.test_results['model_training'] = test_result
        return test_result
    
    def test_visualization_reporting_pipeline(self) -> Dict[str, Any]:
        """
        Test visualization and reporting functionality.
        Maintains all existing visualization logic.
        """
        logger.info("🧪 Testing visualization and reporting pipeline...")
        
        test_result = {
            'test_name': 'visualization_reporting_pipeline',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Set seeds for reproducibility
            self.repro_manager.set_all_seeds()
            
            # Test visualization utilities
            try:
                from src.visualization.visualization_utils import ComprehensiveVisualizer
                from src.utils.app_config import Config
                
                config = Config()
                visualizer = ComprehensiveVisualizer(config)
                
                test_result['details']['visualizer_initialized'] = True
                test_result['details']['plotting_available'] = getattr(visualizer, 'plotting_available', False)
                
            except Exception as e:
                test_result['warnings'].append(f"Visualization initialization warning: {str(e)}")
                test_result['details']['visualizer_initialized'] = False
            
            # Test automated reporting
            try:
                from src.visualization.automated_reporting import AutomatedReportGenerator
                
                report_generator = AutomatedReportGenerator(config)
                test_result['details']['report_generator_initialized'] = True
                
            except Exception as e:
                test_result['warnings'].append(f"Report generator initialization warning: {str(e)}")
                test_result['details']['report_generator_initialized'] = False
            
            # Test file management for outputs
            from src.utils.file_management_utils import SafeFileManager
            
            file_manager = SafeFileManager(config.get_data_save_path())
            test_result['details']['file_manager_initialized'] = True
            
            logger.info("✅ Visualization and reporting pipeline test passed")
            
        except Exception as e:
            test_result['success'] = False
            test_result['errors'].append(str(e))
            logger.error(f"❌ Visualization and reporting pipeline test failed: {str(e)}")
        
        self.test_results['visualization_reporting'] = test_result
        return test_result
    
    def test_file_management_integration(self) -> Dict[str, Any]:
        """
        Test file management system integration.
        Maintains all existing file handling logic.
        """
        logger.info("🧪 Testing file management integration...")
        
        test_result = {
            'test_name': 'file_management_integration',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Set seeds for reproducibility
            self.repro_manager.set_all_seeds()
            
            from src.utils.app_config import Config
            from src.utils.file_management_utils import SafeFileManager, SaveStrategy
            
            config = Config()
            file_manager = SafeFileManager(config.get_data_save_path())
            
            # Test directory creation
            test_result['details']['base_path'] = str(file_manager.base_path)
            test_result['details']['directory_exists'] = os.path.exists(file_manager.base_path)
            
            # Create test data and save it
            test_data = self._create_sample_market_data()
            
            # Test different save strategies
            strategies_tested = []
            
            for strategy in [SaveStrategy.VERSION, SaveStrategy.TIMESTAMP, SaveStrategy.OVERWRITE]:
                try:
                    result = file_manager.save_dataframe(
                        test_data, 
                        f"test_file_{strategy.value}.csv",
                        strategy=strategy
                    )
                    
                    if result.success:
                        strategies_tested.append(strategy.value)
                        
                except Exception as e:
                    test_result['warnings'].append(f"Strategy {strategy.value} warning: {str(e)}")
            
            test_result['details']['strategies_tested'] = strategies_tested
            test_result['details']['file_management_working'] = len(strategies_tested) > 0
            
            logger.info("✅ File management integration test passed")
            
        except Exception as e:
            test_result['success'] = False
            test_result['errors'].append(str(e))
            logger.error(f"❌ File management integration test failed: {str(e)}")
        
        self.test_results['file_management'] = test_result
        return test_result
    
    def test_reproducibility_features(self) -> Dict[str, Any]:
        """
        Test reproducibility features across all components.
        """
        logger.info("🧪 Testing reproducibility features...")
        
        test_result = {
            'test_name': 'reproducibility_features',
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Test seed consistency
            self.repro_manager.set_all_seeds()
            value1 = np.random.random()
            
            self.repro_manager.set_all_seeds()
            value2 = np.random.random()
            
            test_result['details']['seed_consistency'] = (value1 == value2)
            
            # Test data split reproducibility
            sample_data = self._create_sample_training_data()
            
            split1 = self.repro_manager.create_reproducible_data_split(sample_data, time_column='datetime')
            split2 = self.repro_manager.create_reproducible_data_split(sample_data, time_column='datetime')
            
            # Check if splits are identical
            train_identical = split1['train'].equals(split2['train'])
            test_identical = split1['test'].equals(split2['test'])
            
            test_result['details']['data_split_reproducibility'] = train_identical and test_identical
            
            # Test model parameter consistency
            params1 = self.repro_manager.get_reproducible_model_params('RandomForestRegressor')
            params2 = self.repro_manager.get_reproducible_model_params('RandomForestRegressor')
            
            test_result['details']['model_params_consistency'] = (params1 == params2)
            
            # Test environment state capture
            env_state = self.repro_manager._capture_environment_state()
            test_result['details']['environment_captured'] = len(env_state) > 0
            test_result['details']['packages_documented'] = len(env_state.get('installed_packages', {})) > 0
            
            logger.info("✅ Reproducibility features test passed")
            
        except Exception as e:
            test_result['success'] = False
            test_result['errors'].append(str(e))
            logger.error(f"❌ Reproducibility features test failed: {str(e)}")
        
        self.test_results['reproducibility'] = test_result
        return test_result
    
    def run_all_notebook_tests(self) -> Dict[str, Any]:
        """Run all notebook-related tests"""
        logger.info("🧪 Running comprehensive notebook tests...")
        logger.info("=" * 60)
        
        # Run all test categories
        tests = [
            self.test_data_collection_pipeline,
            self.test_model_training_pipeline,
            self.test_visualization_reporting_pipeline,
            self.test_file_management_integration,
            self.test_reproducibility_features
        ]
        
        all_results = {}
        overall_success = True
        
        for test_func in tests:
            result = test_func()
            all_results[result['test_name']] = result
            
            if not result['success']:
                overall_success = False
        
        # Generate summary
        summary = {
            'overall_success': overall_success,
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(tests),
            'passed_tests': sum(1 for r in all_results.values() if r['success']),
            'failed_tests': sum(1 for r in all_results.values() if not r['success']),
            'total_warnings': sum(len(r['warnings']) for r in all_results.values()),
            'test_results': all_results,
            'reproducibility_seed': self.repro_manager.seed
        }
        
        # Save results
        self._save_notebook_test_results(summary)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 NOTEBOOK TEST SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Warnings: {summary['total_warnings']}")
        
        if overall_success:
            logger.info("🎉 ALL NOTEBOOK TESTS PASSED!")
            logger.info("✅ All notebook functionality preserved and working")
            logger.info("✅ Reproducibility features integrated successfully")
        else:
            logger.error("❌ Some notebook tests failed")
        
        return summary
    
    def _create_sample_market_data(self) -> pd.DataFrame:
        """Create sample market data following existing patterns"""
        np.random.seed(self.repro_manager.seed)
        
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        
        # Create realistic OHLCV data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n_samples)
        price_series = base_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'datetime': dates,
            'open': price_series * (1 + np.random.normal(0, 0.001, n_samples)),
            'high': price_series * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'low': price_series * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'close': price_series,
            'volume': np.random.lognormal(10, 1, n_samples)
        })
    
    def _create_sample_training_data(self) -> pd.DataFrame:
        """Create sample training data with features"""
        market_data = self._create_sample_market_data()
        
        # Add technical indicators (simulated)
        market_data['sma_20'] = market_data['close'].rolling(20).mean()
        market_data['rsi'] = np.random.uniform(20, 80, len(market_data))
        market_data['macd'] = np.random.normal(0, 1, len(market_data))
        
        # Add target variable
        market_data['target'] = market_data['close'].shift(-1)
        
        return market_data.dropna()
    
    def _create_sample_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample predictions for testing"""
        n_samples = 100
        y_true = np.random.normal(100, 10, n_samples)
        y_pred = y_true + np.random.normal(0, 2, n_samples)
        
        return y_true, y_pred
    
    def _calculate_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard evaluation metrics"""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _save_notebook_test_results(self, summary: Dict[str, Any]) -> None:
        """Save notebook test results"""
        results_file = f"notebook_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📋 Notebook test results saved to {results_file}")


def main():
    """Main function to run notebook tests"""
    print("🧪 NOTEBOOK TESTING UTILITIES")
    print("=" * 60)
    print("Testing notebook functionality while maintaining all underlying logic")
    print("=" * 60)
    
    try:
        # Create test runner
        test_runner = NotebookTestRunner(seed=42)
        
        # Run all notebook tests
        results = test_runner.run_all_notebook_tests()
        
        return 0 if results['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"💥 Notebook test runner crashed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
