"""
Simple test for cost-aware ML integration.
Tests the basic functionality without external dependencies.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append('/home/runner/work/AI-ML-Based-Stock-Price-Prediction/AI-ML-Based-Stock-Price-Prediction')

def test_cost_feature_generation():
    """Test cost feature generation functionality."""
    try:
        from src.models.features.cost_features import CostFeatureGenerator, CostFeatureConfig
        
        # Create test data
        n_samples = 100
        test_data = pd.DataFrame({
            'close': np.random.normal(100, 10, n_samples),
            'volume': np.random.normal(1000000, 100000, n_samples),
            'high': np.random.normal(105, 10, n_samples),
            'low': np.random.normal(95, 10, n_samples)
        })
        
        # Ensure high >= close >= low
        test_data['high'] = np.maximum(test_data['high'], test_data['close'])
        test_data['low'] = np.minimum(test_data['low'], test_data['close'])
        
        # Create generator
        config = CostFeatureConfig()
        generator = CostFeatureGenerator(config)
        
        # Generate features
        result_data = generator.generate_all_cost_features(test_data)
        
        # Check results
        assert len(result_data) == n_samples, f"Expected {n_samples} rows, got {len(result_data)}"
        assert len(result_data.columns) > len(test_data.columns), "No cost features were generated"
        
        feature_names = generator.get_feature_names()
        assert len(feature_names) > 0, "No feature names returned"
        
        print(f"✓ Cost feature generation test passed. Generated {len(feature_names)} features.")
        return True
        
    except Exception as e:
        print(f"✗ Cost feature generation test failed: {e}")
        return False

def test_cost_pipeline():
    """Test cost-aware feature pipeline."""
    try:
        from src.models.features.cost_pipeline import CostFeaturePipeline, CostPipelineConfig
        
        # Create test data
        n_samples = 50
        test_data = pd.DataFrame({
            'close': np.random.normal(100, 10, n_samples),
            'volume': np.random.normal(1000000, 100000, n_samples),
            'open': np.random.normal(100, 10, n_samples)
        })
        
        # Create pipeline
        config = CostPipelineConfig(enable_technical_indicators=False)  # Disable to avoid dependencies
        pipeline = CostFeaturePipeline(config)
        
        # Process features
        result = pipeline.process_features(test_data)
        
        # Check results
        assert result.success, f"Pipeline failed: {result.errors}"
        assert len(result.data) == n_samples, f"Expected {n_samples} rows, got {len(result.data)}"
        
        print(f"✓ Cost pipeline test passed. Processing time: {result.processing_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"✗ Cost pipeline test failed: {e}")
        return False

def test_cost_feature_selector():
    """Test cost feature selector."""
    try:
        from src.models.features.cost_feature_selector import CostFeatureSelector, FeatureSelectionConfig
        
        # Create test data with cost features
        n_samples = 100
        test_data = pd.DataFrame({
            'close': np.random.normal(100, 10, n_samples),
            'cost_avg_20d': np.random.normal(0.05, 0.01, n_samples),
            'cost_vol_20d': np.random.normal(0.02, 0.005, n_samples),
            'cost_return_ratio_20d': np.random.normal(0.1, 0.02, n_samples),
            'other_feature': np.random.normal(0, 1, n_samples)
        })
        
        # Create selector
        config = FeatureSelectionConfig(max_features=2)
        selector = CostFeatureSelector(config)
        
        # Select features
        result = selector.select_features(test_data)
        
        # Check results
        assert len(result['selected_features']) <= 2, "Too many features selected"
        assert len(result['selected_features']) > 0, "No features selected"
        
        print(f"✓ Cost feature selector test passed. Selected {len(result['selected_features'])} features.")
        return True
        
    except Exception as e:
        print(f"✗ Cost feature selector test failed: {e}")
        return False

def test_cost_metrics():
    """Test cost metrics calculation."""
    try:
        from src.models.evaluation.cost_metrics import CostMetrics
        
        # Create test data
        n_samples = 100
        y_true = np.random.normal(0, 1, n_samples)
        y_pred = y_true + np.random.normal(0, 0.1, n_samples)
        
        X_features = pd.DataFrame({
            'cost_avg_20d': np.random.normal(0.05, 0.01, n_samples),
            'cost_vol_20d': np.random.normal(0.02, 0.005, n_samples)
        })
        
        # Create metrics calculator
        metrics_calc = CostMetrics()
        
        # Calculate metrics
        metrics = metrics_calc.calculate_cost_enhanced_metrics(
            y_true, y_pred, X_features, ['cost_avg_20d', 'cost_vol_20d']
        )
        
        # Check results
        assert 'mse' in metrics, "Basic MSE metric missing"
        assert 'cost_weighted_mae' in metrics, "Cost-weighted MAE missing"
        assert 'cost_adjusted_r2' in metrics, "Cost-adjusted R² missing"
        
        print(f"✓ Cost metrics test passed. Calculated {len(metrics)} metrics.")
        return True
        
    except Exception as e:
        print(f"✗ Cost metrics test failed: {e}")
        return False

def test_configuration():
    """Test configuration classes."""
    try:
        from src.models.config.cost_integration import CostIntegrationConfig, create_basic_cost_config
        from src.models.config.cost_feature_config import CostFeatureConfig, create_standard_cost_feature_config
        
        # Test cost integration config
        cost_config = create_basic_cost_config()
        assert cost_config.enable_cost_features, "Cost features should be enabled in basic config"
        
        config_dict = cost_config.to_dict()
        assert 'integration_level' in config_dict, "Integration level missing from dict"
        
        # Test cost feature config
        feature_config = create_standard_cost_feature_config()
        feature_names = feature_config.get_feature_names()
        assert len(feature_names) > 0, "No feature names generated"
        
        print(f"✓ Configuration test passed. Generated {len(feature_names)} feature name templates.")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running cost-aware ML integration tests...\n")
    
    tests = [
        test_cost_feature_generation,
        test_cost_pipeline,
        test_cost_feature_selector,
        test_cost_metrics,
        test_configuration
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
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Cost-aware ML integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)