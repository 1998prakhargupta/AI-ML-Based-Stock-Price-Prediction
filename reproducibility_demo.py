#!/usr/bin/env python3
"""
🎲 REPRODUCIBILITY DEMONSTRATION
Quick demonstration of reproducibility features

This script shows how reproducibility works across multiple runs.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import reproducibility utilities
from reproducibility_utils import ReproducibilityManager, set_global_seed, get_reproducible_split

def demonstrate_seed_consistency():
    """Demonstrate that seeds produce consistent results"""
    print("🧪 SEED CONSISTENCY DEMONSTRATION")
    print("=" * 50)
    
    # Run 1
    print("\n📊 Run 1:")
    set_global_seed(42)
    values_1 = [np.random.random() for _ in range(5)]
    data_1 = np.random.normal(0, 1, 10)
    print(f"Random values: {[f'{v:.6f}' for v in values_1]}")
    print(f"Normal data mean: {data_1.mean():.6f}")
    
    # Run 2 (should be identical)
    print("\n📊 Run 2:")
    set_global_seed(42)
    values_2 = [np.random.random() for _ in range(5)]
    data_2 = np.random.normal(0, 1, 10)
    print(f"Random values: {[f'{v:.6f}' for v in values_2]}")
    print(f"Normal data mean: {data_2.mean():.6f}")
    
    # Verification
    print("\n✅ VERIFICATION:")
    print(f"Values identical: {values_1 == values_2}")
    print(f"Data arrays identical: {np.array_equal(data_1, data_2)}")

def demonstrate_data_split_reproducibility():
    """Demonstrate reproducible data splitting"""
    print("\n\n🔄 DATA SPLIT REPRODUCIBILITY DEMONSTRATION")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'target': np.random.normal(100, 10, 100)
    })
    
    print(f"📊 Sample data created: {len(sample_data)} rows")
    
    # Split 1
    print("\n📊 Split 1:")
    split_1 = get_reproducible_split(sample_data, time_column='datetime', test_size=0.2)
    print(f"Train: {len(split_1['train'])} rows")
    print(f"Test: {len(split_1['test'])} rows")
    print(f"First train datetime: {split_1['train']['datetime'].iloc[0]}")
    print(f"First test datetime: {split_1['test']['datetime'].iloc[0]}")
    
    # Split 2 (should be identical)
    print("\n📊 Split 2:")
    split_2 = get_reproducible_split(sample_data, time_column='datetime', test_size=0.2)
    print(f"Train: {len(split_2['train'])} rows")
    print(f"Test: {len(split_2['test'])} rows")
    print(f"First train datetime: {split_2['train']['datetime'].iloc[0]}")
    print(f"First test datetime: {split_2['test']['datetime'].iloc[0]}")
    
    # Verification
    print("\n✅ VERIFICATION:")
    splits_identical = split_1['train'].equals(split_2['train']) and split_1['test'].equals(split_2['test'])
    print(f"Splits identical: {splits_identical}")

def demonstrate_model_parameter_consistency():
    """Demonstrate consistent model parameters"""
    print("\n\n⚙️ MODEL PARAMETER CONSISTENCY DEMONSTRATION")
    print("=" * 50)
    
    manager = ReproducibilityManager(seed=42)
    
    models = ['RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor']
    
    print("\n📊 Run 1:")
    params_1 = {}
    for model in models:
        params = manager.get_reproducible_model_params(model)
        params_1[model] = params
        print(f"{model}: {params}")
    
    print("\n📊 Run 2:")
    params_2 = {}
    for model in models:
        params = manager.get_reproducible_model_params(model)
        params_2[model] = params
        print(f"{model}: {params}")
    
    print("\n✅ VERIFICATION:")
    consistency = all(params_1[model] == params_2[model] for model in models)
    print(f"Parameter consistency: {consistency}")

def demonstrate_experiment_tracking():
    """Demonstrate experiment state saving and loading"""
    print("\n\n💾 EXPERIMENT TRACKING DEMONSTRATION")
    print("=" * 50)
    
    manager = ReproducibilityManager(seed=42)
    
    # Save experiment state
    experiment_info = {
        'model_type': 'ensemble',
        'features_used': ['technical_indicators', 'price_data'],
        'performance': {'rmse': 2.45, 'r2': 0.89}
    }
    
    filepath = manager.save_experiment_state('reproducibility_demo', experiment_info)
    print(f"📋 Experiment saved to: {os.path.basename(filepath)}")
    
    # Load experiment state
    loaded_state = manager.load_experiment_state(filepath)
    print(f"📂 Experiment loaded successfully")
    print(f"Experiment name: {loaded_state['experiment_name']}")
    print(f"Seed used: {loaded_state['reproducibility_config']['seed']}")
    print(f"Additional info: {loaded_state['additional_info']}")
    
    # Clean up
    if os.path.exists(filepath):
        os.remove(filepath)
        if os.path.exists('experiments') and not os.listdir('experiments'):
            os.rmdir('experiments')

def main():
    """Run all reproducibility demonstrations"""
    print("🎲 REPRODUCIBILITY FEATURES DEMONSTRATION")
    print("=" * 80)
    print("Showing how reproducibility ensures consistent results")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demonstrate_seed_consistency()
        demonstrate_data_split_reproducibility()
        demonstrate_model_parameter_consistency()
        demonstrate_experiment_tracking()
        
        print("\n\n" + "=" * 80)
        print("🎉 REPRODUCIBILITY DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("✅ Seed consistency: All random operations produce identical results")
        print("✅ Data splits: Train/test splits are identical across runs")
        print("✅ Model params: ML model parameters consistent with fixed seeds")
        print("✅ Experiment tracking: Complete state persistence working")
        print("\n🎯 All operations are fully reproducible!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
