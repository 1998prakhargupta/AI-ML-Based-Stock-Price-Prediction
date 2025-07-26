#!/usr/bin/env python3
"""
Import Validation Test
=====================

Comprehensive test to validate all import statements work correctly after migration.
Tests both old symbolic link imports and new organized imports.
"""

import sys
import os
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImportValidator:
    """Validates all import patterns work correctly."""
    
    def __init__(self):
        self.results = {
            'symbolic_links': {'passed': 0, 'failed': 0, 'errors': []},
            'organized_imports': {'passed': 0, 'failed': 0, 'errors': []},
            'cross_module': {'passed': 0, 'failed': 0, 'errors': []}
        }
    
    def test_symbolic_link_imports(self) -> None:
        """Test that symbolic link imports still work for backward compatibility."""
        logger.info("🔗 Testing symbolic link imports...")
        
        tests = [
            ('app_config', 'Config'),
            ('file_management_utils', 'SafeFileManager'),
            ('visualization_utils', 'ComprehensiveVisualizer'),
            ('model_utils', 'ModelManager'),
            ('reproducibility_utils', 'ReproducibilityManager'),
            ('automated_reporting', 'AutomatedReportGenerator')
        ]
        
        for module_name, class_name in tests:
            try:
                # Test basic import
                module = __import__(module_name)
                
                # Test class access
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    logger.info(f"✅ {module_name}.{class_name} imported successfully")
                    self.results['symbolic_links']['passed'] += 1
                else:
                    error = f"Class {class_name} not found in {module_name}"
                    logger.error(f"❌ {error}")
                    self.results['symbolic_links']['failed'] += 1
                    self.results['symbolic_links']['errors'].append(error)
                    
            except Exception as e:
                error = f"Failed to import {module_name}: {e}"
                logger.error(f"❌ {error}")
                self.results['symbolic_links']['failed'] += 1
                self.results['symbolic_links']['errors'].append(error)
    
    def test_organized_imports(self) -> None:
        """Test that new organized imports work correctly."""
        logger.info("📂 Testing organized imports...")
        
        tests = [
            ('src.utils.app_config', 'Config'),
            ('src.utils.file_management_utils', 'SafeFileManager'),
            ('src.visualization.visualization_utils', 'ComprehensiveVisualizer'),
            ('src.models.model_utils', 'ModelManager'),
            ('src.utils.reproducibility_utils', 'ReproducibilityManager'),
            ('src.visualization.automated_reporting', 'AutomatedReportGenerator')
        ]
        
        for module_path, class_name in tests:
            try:
                # Test direct import
                module = __import__(module_path, fromlist=[class_name])
                
                # Test class access
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    logger.info(f"✅ {module_path}.{class_name} imported successfully")
                    self.results['organized_imports']['passed'] += 1
                else:
                    error = f"Class {class_name} not found in {module_path}"
                    logger.error(f"❌ {error}")
                    self.results['organized_imports']['failed'] += 1
                    self.results['organized_imports']['errors'].append(error)
                    
            except Exception as e:
                error = f"Failed to import {module_path}: {e}"
                logger.error(f"❌ {error}")
                self.results['organized_imports']['failed'] += 1
                self.results['organized_imports']['errors'].append(error)
    
    def test_cross_module_functionality(self) -> None:
        """Test that modules can import from each other correctly."""
        logger.info("🔄 Testing cross-module imports...")
        
        try:
            # Test Config usage
            from src.utils.app_config import Config
            config = Config()
            logger.info("✅ Config instantiation successful")
            self.results['cross_module']['passed'] += 1
            
            # Test SafeFileManager with Config
            from src.utils.file_management_utils import SafeFileManager
            file_manager = SafeFileManager(config.get_data_save_path())
            logger.info("✅ SafeFileManager with Config successful")
            self.results['cross_module']['passed'] += 1
            
            # Test ComprehensiveVisualizer with Config
            from src.visualization.visualization_utils import ComprehensiveVisualizer
            visualizer = ComprehensiveVisualizer(config)
            logger.info("✅ ComprehensiveVisualizer with Config successful")
            self.results['cross_module']['passed'] += 1
            
            # Test AutomatedReportGenerator with Config
            from src.visualization.automated_reporting import AutomatedReportGenerator
            reporter = AutomatedReportGenerator(config)
            logger.info("✅ AutomatedReportGenerator with Config successful")
            self.results['cross_module']['passed'] += 1
            
            # Test ModelManager functionality
            from src.models.model_utils import ModelManager
            model_manager = ModelManager(config)
            logger.info("✅ ModelManager with Config successful")
            self.results['cross_module']['passed'] += 1
            
        except Exception as e:
            error = f"Cross-module functionality test failed: {e}"
            logger.error(f"❌ {error}")
            self.results['cross_module']['failed'] += 1
            self.results['cross_module']['errors'].append(error)
    
    def test_specific_import_patterns(self) -> None:
        """Test specific import patterns that should work."""
        logger.info("🎯 Testing specific import patterns...")
        
        import_tests = [
            # Test from imports
            "from src.utils.app_config import Config",
            "from src.utils.file_management_utils import SafeFileManager, SaveStrategy",
            "from src.visualization.visualization_utils import ComprehensiveVisualizer",
            "from src.models.model_utils import ModelManager, ModelEvaluator",
            "from src.utils.reproducibility_utils import ReproducibilityManager",
            "from src.visualization.automated_reporting import AutomatedReportGenerator",
            
            # Test module imports
            "import src.utils.app_config as config_module",
            "import src.utils.file_management_utils as file_module",
            "import src.visualization.visualization_utils as viz_module",
            "import src.models.model_utils as model_module"
        ]
        
        for import_statement in import_tests:
            try:
                exec(import_statement)
                logger.info(f"✅ '{import_statement}' successful")
                self.results['cross_module']['passed'] += 1
            except Exception as e:
                error = f"Failed: '{import_statement}' - {e}"
                logger.error(f"❌ {error}")
                self.results['cross_module']['failed'] += 1
                self.results['cross_module']['errors'].append(error)
    
    def run_all_tests(self) -> Dict:
        """Run all validation tests."""
        logger.info("=" * 60)
        logger.info("🧪 IMPORT VALIDATION TEST SUITE")
        logger.info("=" * 60)
        
        self.test_symbolic_link_imports()
        self.test_organized_imports()
        self.test_cross_module_functionality()
        self.test_specific_import_patterns()
        
        return self.results
    
    def print_summary(self) -> None:
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("📊 VALIDATION RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total_passed = sum(category['passed'] for category in self.results.values())
        total_failed = sum(category['failed'] for category in self.results.values())
        total_tests = total_passed + total_failed
        
        for category, results in self.results.items():
            passed = results['passed']
            failed = results['failed']
            category_total = passed + failed
            
            if category_total > 0:
                success_rate = (passed / category_total) * 100
                status = "✅ PASS" if failed == 0 else f"⚠️ {failed} FAILED"
                logger.info(f"{category.upper():20} | {passed:3}/{category_total:3} ({success_rate:5.1f}%) | {status}")
                
                # Show errors if any
                if results['errors']:
                    for error in results['errors']:
                        logger.error(f"  • {error}")
        
        logger.info("-" * 60)
        if total_tests > 0:
            overall_success = (total_passed / total_tests) * 100
            status = "🎉 ALL TESTS PASSED" if total_failed == 0 else f"❌ {total_failed} TESTS FAILED"
            logger.info(f"{'OVERALL':20} | {total_passed:3}/{total_tests:3} ({overall_success:5.1f}%) | {status}")
        
        logger.info("=" * 60)


def main():
    """Run import validation."""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    validator = ImportValidator()
    results = validator.run_all_tests()
    validator.print_summary()
    
    # Return exit code based on results
    total_failed = sum(category['failed'] for category in results.values())
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
