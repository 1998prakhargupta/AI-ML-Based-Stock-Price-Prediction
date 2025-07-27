"""
Simple validation script for cost-aware ML integration.
Tests basic imports and module structure without external dependencies.
"""

import sys
import os

# Add project root to path
sys.path.append('/home/runner/work/AI-ML-Based-Stock-Price-Prediction/AI-ML-Based-Stock-Price-Prediction')

def test_imports():
    """Test that all modules can be imported."""
    try:
        # Test feature modules
        from src.models.features import CostFeatureGenerator, CostFeaturePipeline, CostFeatureSelector
        print("✓ Feature modules imported successfully")
        
        # Test training modules
        from src.models.training import CostAwareTrainer, CostIntegrationMixin
        print("✓ Training modules imported successfully")
        
        # Test evaluation modules
        from src.models.evaluation import CostEvaluator, CostMetrics, CostPerformanceAnalyzer
        print("✓ Evaluation modules imported successfully")
        
        # Test configuration modules
        from src.models.config import CostIntegrationConfig, CostFeatureConfig
        print("✓ Configuration modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_class_instantiation():
    """Test that classes can be instantiated."""
    try:
        # Test configuration classes
        from src.models.config.cost_integration import create_basic_cost_config
        from src.models.config.cost_feature_config import create_standard_cost_feature_config
        
        cost_config = create_basic_cost_config()
        feature_config = create_standard_cost_feature_config()
        print("✓ Configuration classes instantiated successfully")
        
        # Test basic feature generator (without data processing)
        from src.models.features.cost_features import CostFeatureGenerator
        generator = CostFeatureGenerator(feature_config)
        print("✓ CostFeatureGenerator instantiated successfully")
        
        # Test feature selector
        from src.models.features.cost_feature_selector import CostFeatureSelector
        selector = CostFeatureSelector()
        print("✓ CostFeatureSelector instantiated successfully")
        
        # Test cost metrics
        from src.models.evaluation.cost_metrics import CostMetrics
        metrics = CostMetrics()
        print("✓ CostMetrics instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Class instantiation test failed: {e}")
        return False

def test_configuration_methods():
    """Test configuration methods."""
    try:
        from src.models.config.cost_integration import create_basic_cost_config, create_advanced_cost_config
        from src.models.config.cost_feature_config import create_minimal_cost_feature_config
        
        # Test different config types
        basic_config = create_basic_cost_config()
        advanced_config = create_advanced_cost_config()
        minimal_feature_config = create_minimal_cost_feature_config()
        
        # Test config conversion
        config_dict = basic_config.to_dict()
        assert 'integration_level' in config_dict
        
        # Test feature config methods
        feature_names = minimal_feature_config.get_feature_names()
        assert len(feature_names) > 0
        
        print("✓ Configuration methods work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Configuration methods test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required files exist."""
    base_path = '/home/runner/work/AI-ML-Based-Stock-Price-Prediction/AI-ML-Based-Stock-Price-Prediction/src/models'
    
    required_files = [
        'features/__init__.py',
        'features/cost_features.py',
        'features/cost_pipeline.py',
        'features/cost_feature_selector.py',
        'training/__init__.py',
        'training/cost_aware_trainer.py',
        'training/cost_integration_mixin.py',
        'evaluation/__init__.py',
        'evaluation/cost_evaluator.py',
        'evaluation/cost_metrics.py',
        'evaluation/cost_performance_analyzer.py',
        'config/__init__.py',
        'config/cost_integration.py',
        'config/cost_feature_config.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def main():
    """Run all validation tests."""
    print("Validating cost-aware ML integration implementation...\n")
    
    tests = [
        test_directory_structure,
        test_imports,
        test_class_instantiation,
        test_configuration_methods
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()  # Empty line for readability
    
    print(f"Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 Implementation validation successful! Cost-aware ML integration is properly structured.")
        return True
    else:
        print("❌ Some validations failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)