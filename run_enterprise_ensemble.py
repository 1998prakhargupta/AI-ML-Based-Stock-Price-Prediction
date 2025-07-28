#!/usr/bin/env python3
"""
Quick Start Script for Enterprise Ensemble System
=================================================

This script provides a simple interface to run the enterprise ensemble system
with all models and configurations.

Usage:
    python run_enterprise_ensemble.py [options]

Options:
    --demo         Run demonstration with sample data
    --config PATH  Use custom configuration file
    --models LIST  Comma-separated list of models to train
    --output DIR   Output directory for results

Examples:
    python run_enterprise_ensemble.py --demo
    python run_enterprise_ensemble.py --config config/production.yaml
    python run_enterprise_ensemble.py --models "random_forest,xgboost,lstm"
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enterprise Ensemble Model Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                              # Run demo with sample data
  %(prog)s --config config/production.yaml    # Use production config
  %(prog)s --models "rf,xgb,lstm"             # Train specific models
        """
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run demonstration with generated sample data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/ensemble-config.yaml',
        help='Path to configuration file (default: config/ensemble-config.yaml)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to train'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/experiments',
        help='Output directory for trained models (default: models/experiments)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to input data file (CSV format)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='close',
        help='Target column name (default: close)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Enterprise Ensemble Model Training System")
    print("=" * 60)
    print(f"📁 Project root: {project_root}")
    print(f"🔧 Configuration: {args.config}")
    
    if args.demo:
        print("📊 Running demonstration mode...")
        try:
            from scripts.enterprise_ensemble_demo import run_enterprise_ensemble_demo
            run_enterprise_ensemble_demo()
        except ImportError:
            print("❌ Demo script not found. Running basic ensemble instead.")
            run_basic_ensemble(args)
    else:
        print("🎯 Running production mode...")
        run_production_ensemble(args)

def run_basic_ensemble(args):
    """Run basic ensemble training."""
    try:
        from src.models.training.ensemble_trainer import EnterpriseEnsembleTrainer
        import pandas as pd
        import numpy as np
        
        print(f"📁 Configuration: {args.config}")
        print(f"📊 Output directory: {args.output}")
        
        # Initialize trainer
        trainer = EnterpriseEnsembleTrainer(config_path=args.config)
        
        # Load or generate data
        if args.data:
            print(f"📥 Loading data from: {args.data}")
            data = pd.read_csv(args.data)
        else:
            print("📊 Generating sample data...")
            # Generate sample data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=1000, freq='D')
            data = pd.DataFrame({
                'timestamp': dates,
                'close': 100 + np.random.randn(1000).cumsum(),
                'volume': np.random.randint(1000000, 10000000, 1000),
                'feature_1': np.random.randn(1000),
                'feature_2': np.random.randn(1000),
                'feature_3': np.random.randn(1000)
            })
        
        # Prepare features
        exclude_cols = ['timestamp', 'close', 'symbol'] if 'symbol' in data.columns else ['timestamp', 'close']
        feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"🎯 Target: {args.target}")
        print(f"📏 Data shape: {data.shape}")
        
        # Filter models if specified
        model_filter = None
        if args.models:
            model_filter = [m.strip() for m in args.models.split(',')]
            print(f"🎭 Training models: {model_filter}")
        
        # Train models
        results = trainer.train_ensemble_models(
            data=data,
            target_column=args.target,
            feature_columns=feature_columns,
            test_size=args.test_size,
            model_filter=model_filter
        )
        
        # Display results
        print("\n📊 Training Results:")
        print(f"✅ Best model: {results['best_individual_model'].model_name}")
        print(f"📈 Best score: {results['best_individual_model'].validation_score:.4f}")
        print(f"🎯 Ensemble score: {results['ensemble_metrics'].get('r2_score', 0):.4f}")
        
        print(f"\n💾 Models saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_production_ensemble(args):
    """Run production ensemble training."""
    print("🏭 Production mode - Full enterprise training pipeline")
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("Creating default configuration...")
        create_default_config(args.config)
    
    if args.data and not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    # Run the basic ensemble for now
    run_basic_ensemble(args)

def create_default_config(config_path):
    """Create a default configuration file."""
    import yaml
    
    default_config = {
        "ensemble": {
            "models": {
                "traditional_ml": {
                    "random_forest": {"n_estimators": 100, "random_state": 42},
                    "gradient_boosting": {"n_estimators": 100, "random_state": 42}
                },
                "gradient_boosting": {
                    "xgboost": {"n_estimators": 100, "random_state": 42}
                }
            },
            "ensemble_methods": ["weighted_average"],
            "validation": {
                "method": "time_series_split",
                "n_splits": 5
            }
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, indent=2)
    
    print(f"✅ Default configuration created: {config_path}")

if __name__ == "__main__":
    main()
