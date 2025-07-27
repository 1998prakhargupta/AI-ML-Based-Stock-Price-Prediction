"""
Cost-Aware ML Integration Example
=================================

This script demonstrates how to use the cost-aware ML features with the existing
stock price prediction pipeline. It includes fallbacks for missing dependencies.
"""

import sys
import os

# Add project root to path
sys.path.append('/home/runner/work/AI-ML-Based-Stock-Price-Prediction/AI-ML-Based-Stock-Price-Prediction')

def check_dependencies():
    """Check which dependencies are available."""
    deps = {}
    
    try:
        import numpy as np
        deps['numpy'] = True
    except ImportError:
        deps['numpy'] = False
    
    try:
        import pandas as pd
        deps['pandas'] = True
    except ImportError:
        deps['pandas'] = False
    
    try:
        import sklearn
        deps['sklearn'] = True
    except ImportError:
        deps['sklearn'] = False
    
    return deps

def demonstrate_configuration():
    """Demonstrate cost integration configuration."""
    print("=== Cost Integration Configuration ===")
    
    try:
        from src.models.config.cost_integration import (
            create_basic_cost_config, 
            create_advanced_cost_config,
            create_disabled_cost_config
        )
        
        # Show different configuration levels
        configs = {
            'Disabled': create_disabled_cost_config(),
            'Basic': create_basic_cost_config(),
            'Advanced': create_advanced_cost_config()
        }
        
        for name, config in configs.items():
            print(f"\n{name} Configuration:")
            print(f"  Integration Level: {config.integration_level.value}")
            print(f"  Cost Features: {config.enable_cost_features}")
            print(f"  Cost Training: {config.enable_cost_training}")
            print(f"  Cost Evaluation: {config.enable_cost_evaluation}")
            
            enabled_features = config.get_integration_level_features()
            print(f"  Enabled Features: {', '.join(enabled_features)}")
        
        print("\n✓ Configuration demonstration completed successfully")
        
    except Exception as e:
        print(f"✗ Configuration demonstration failed: {e}")

def demonstrate_feature_configuration():
    """Demonstrate cost feature configuration."""
    print("\n=== Cost Feature Configuration ===")
    
    try:
        from src.models.config.cost_feature_config import (
            create_minimal_cost_feature_config,
            create_standard_cost_feature_config,
            CostFeatureType
        )
        
        # Show different feature configurations
        configs = {
            'Minimal': create_minimal_cost_feature_config(),
            'Standard': create_standard_cost_feature_config()
        }
        
        for name, config in configs.items():
            print(f"\n{name} Feature Configuration:")
            print(f"  Feature Types: {[ft.value for ft in config.enabled_feature_types]}")
            print(f"  Lookback Windows: {config.lookback_windows}")
            print(f"  Synthetic Costs: {config.enable_synthetic_costs}")
            print(f"  Normalization: {config.normalize_features}")
            
            # Show expected feature names
            feature_names = config.get_feature_names()
            print(f"  Expected Features ({len(feature_names)}): {feature_names[:5]}...")
        
        print("\n✓ Feature configuration demonstration completed successfully")
        
    except Exception as e:
        print(f"✗ Feature configuration demonstration failed: {e}")

def demonstrate_integration_workflow():
    """Demonstrate the complete integration workflow."""
    print("\n=== Integration Workflow ===")
    
    # Check dependencies first
    deps = check_dependencies()
    print(f"Available dependencies: {[k for k, v in deps.items() if v]}")
    
    if not deps['numpy'] or not deps['pandas']:
        print("Note: numpy/pandas not available - showing configuration only")
        return
    
    try:
        import numpy as np
        import pandas as pd
        
        # Create sample data
        print("\n1. Creating sample trading data...")
        n_samples = 100
        data = pd.DataFrame({
            'close': np.random.normal(100, 10, n_samples),
            'volume': np.random.normal(1000000, 100000, n_samples),
            'high': np.random.normal(105, 10, n_samples),
            'low': np.random.normal(95, 10, n_samples),
            'datetime': pd.date_range('2023-01-01', periods=n_samples, freq='D')
        })
        
        # Ensure OHLC logic
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        print(f"   Created {len(data)} samples with columns: {list(data.columns)}")
        
        # 2. Generate cost features
        print("\n2. Generating cost features...")
        from src.models.features.cost_features import CostFeatureGenerator
        from src.models.config.cost_feature_config import create_standard_cost_feature_config
        
        feature_config = create_standard_cost_feature_config()
        generator = CostFeatureGenerator(feature_config)
        
        enhanced_data = generator.generate_all_cost_features(data)
        cost_features = generator.get_feature_names()
        
        print(f"   Generated {len(cost_features)} cost features")
        print(f"   Data shape: {data.shape} -> {enhanced_data.shape}")
        print(f"   Sample features: {cost_features[:5]}")
        
        # 3. Feature selection
        print("\n3. Selecting best cost features...")
        from src.models.features.cost_feature_selector import CostFeatureSelector
        
        selector = CostFeatureSelector()
        selection_result = selector.select_features(enhanced_data, cost_features=cost_features)
        
        selected_features = selection_result['selected_features']
        print(f"   Selected {len(selected_features)} features from {len(cost_features)}")
        print(f"   Selected: {selected_features[:3]}...")
        
        # 4. Cost metrics demonstration
        print("\n4. Demonstrating cost metrics...")
        from src.models.evaluation.cost_metrics import CostMetrics
        
        # Create mock predictions
        y_true = np.random.normal(0, 1, n_samples)
        y_pred = y_true + np.random.normal(0, 0.1, n_samples)
        
        metrics_calc = CostMetrics()
        metrics = metrics_calc.calculate_cost_enhanced_metrics(
            y_true, y_pred, enhanced_data, cost_features
        )
        
        print(f"   Calculated {len(metrics)} metrics:")
        for key, value in list(metrics.items())[:5]:
            if isinstance(value, (int, float)):
                print(f"     {key}: {value:.4f}")
        
        print("\n✓ Complete integration workflow demonstrated successfully")
        
    except Exception as e:
        print(f"✗ Integration workflow demonstration failed: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_usage_patterns():
    """Show common usage patterns."""
    print("\n=== Usage Patterns ===")
    
    print("\n1. Basic Cost Integration (Existing Pipeline):")
    print("""
    from src.models.config.cost_integration import create_basic_cost_config
    from src.models.features.cost_pipeline import CostFeaturePipeline
    
    # Enable cost features in existing pipeline
    config = create_basic_cost_config()
    pipeline = CostFeaturePipeline(config)
    
    # Process features as usual - cost features added automatically
    result = pipeline.process_features(your_data)
    enhanced_data = result.data
    """)
    
    print("\n2. Cost-Aware Model Training:")
    print("""
    from src.models.training.cost_aware_trainer import CostAwareTrainer
    from src.models.config.cost_integration import create_advanced_cost_config
    
    # Create cost-aware trainer
    config = create_advanced_cost_config()
    trainer = CostAwareTrainer(config)
    
    # Train with cost considerations
    result = trainer.train_cost_aware_model(
        'random_forest', X_train, y_train, X_test, y_test,
        cost_features=['cost_avg_20d', 'cost_vol_20d']
    )
    """)
    
    print("\n3. Cost-Enhanced Evaluation:")
    print("""
    from src.models.evaluation.cost_evaluator import CostEvaluator
    
    # Evaluate with cost metrics and trading simulation
    evaluator = CostEvaluator()
    eval_result = evaluator.evaluate_with_costs(
        y_true, y_pred, X_features, cost_features
    )
    
    # Get cost-adjusted score
    cost_score = eval_result['cost_adjusted_score']
    """)
    
    print("\n4. Backward Compatibility (Disable Costs):")
    print("""
    from src.models.config.cost_integration import create_disabled_cost_config
    
    # Disable all cost features for backward compatibility
    config = create_disabled_cost_config()
    # Your existing code works unchanged
    """)

def main():
    """Main demonstration function."""
    print("Cost-Aware ML Integration Demonstration")
    print("=" * 50)
    
    # Check system
    deps = check_dependencies()
    print(f"System Dependencies: {deps}")
    
    # Run demonstrations
    demonstrate_configuration()
    demonstrate_feature_configuration()
    demonstrate_integration_workflow()
    demonstrate_usage_patterns()
    
    print("\n" + "=" * 50)
    print("🎉 Demonstration completed!")
    print("\nKey Benefits of Cost-Aware ML Integration:")
    print("✓ Optional integration - fully backward compatible")
    print("✓ Automatic cost feature generation when cost data unavailable")
    print("✓ Cost-weighted training improves real-world performance")
    print("✓ Trading simulation shows realistic cost impact")
    print("✓ Configurable integration levels (disabled/basic/advanced/full)")
    print("✓ Comprehensive cost performance analysis and optimization")
    
    if not deps['numpy']:
        print("\nNote: Install numpy, pandas, and scikit-learn for full functionality:")
        print("pip install numpy pandas scikit-learn")

if __name__ == "__main__":
    main()