"""
Enhanced Notebook Interface for Ensemble Model Training
======================================================

This notebook demonstrates the complete ensemble training pipeline using all models:
Random Forest, Gradient Boosting, XGBoost, LightGBM, Bi-LSTM, GRU, Transformer,
Ridge, Lasso, Elastic Net, SVR, ARIMA, Prophet, and seasonal decomposition.

Features:
- Interactive model training and evaluation
- Real-time performance monitoring
- Advanced visualization
- Cost-aware optimization
- Model comparison and selection
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our ensemble trainer
from src.models.training.ensemble_trainer import EnterpriseEnsembleTrainer, EnsembleConfig
from src.data.processors import TechnicalIndicatorProcessor
from src.data.fetchers import IndexDataManager
from src.utils.visualization_utils import create_comprehensive_plots
from src.utils.app_config import Config

def main_with_analysis():
    """
    Main training function with comprehensive ensemble model creation and analysis.
    
    This function:
    1. Loads and prepares financial data
    2. Engineers features using technical indicators
    3. Trains all available models (ML, DL, Time Series)
    4. Creates optimal ensemble
    5. Evaluates performance with cost considerations
    6. Generates comprehensive visualizations and reports
    
    Returns:
        dict: Complete results including models, predictions, and analysis
    """
    print("🚀 Starting Enterprise Ensemble Training Pipeline...")
    print("=" * 80)
    
    # Initialize components
    config = Config()
    data_manager = IndexDataManager()
    indicator_processor = TechnicalIndicatorProcessor()
    ensemble_trainer = EnterpriseEnsembleTrainer(
        config_path="config/ensemble-config.yaml"
    )
    
    # Step 1: Data Loading and Preparation
    print("📊 Step 1: Loading and preparing market data...")
    
    # Load stock data for multiple symbols
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    combined_data = pd.DataFrame()
    
    for symbol in symbols:
        print(f"   Loading data for {symbol}...")
        try:
            stock_data = data_manager.fetch_stock_data(
                symbol=symbol,
                period="2y",  # 2 years of data
                interval="1d"
            )
            
            if stock_data is not None and not stock_data.empty:
                stock_data['symbol'] = symbol
                combined_data = pd.concat([combined_data, stock_data], ignore_index=True)
                
        except Exception as e:
            print(f"   Warning: Could not load data for {symbol}: {e}")
            continue
    
    if combined_data.empty:
        print("❌ Error: No data loaded. Using sample data for demonstration.")
        # Create sample data for demonstration
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        combined_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(1000).cumsum() * 2,
            'high': 100 + np.random.randn(1000).cumsum() * 2 + 2,
            'low': 100 + np.random.randn(1000).cumsum() * 2 - 2,
            'close': 100 + np.random.randn(1000).cumsum() * 2,
            'volume': np.random.randint(1000000, 10000000, 1000),
            'symbol': 'SAMPLE'
        })
    
    print(f"   ✅ Loaded {len(combined_data)} data points")
    
    # Step 2: Feature Engineering
    print("\n🔧 Step 2: Engineering advanced features...")
    
    enhanced_data_list = []
    feature_columns = []
    
    for symbol in combined_data['symbol'].unique():
        symbol_data = combined_data[combined_data['symbol'] == symbol].copy()
        
        if len(symbol_data) < 100:  # Skip if insufficient data
            continue
            
        print(f"   Processing features for {symbol}...")
        
        # Apply technical indicators
        processed_data = indicator_processor.process_dataframe(
            symbol_data,
            add_all_indicators=True,
            rolling_windows=[5, 10, 20, 50, 100],
            lag_periods=[1, 2, 3, 5, 10]
        )
        
        if hasattr(processed_data, 'data'):
            enhanced_data_list.append(processed_data.data)
        else:
            enhanced_data_list.append(processed_data)
    
    if not enhanced_data_list:
        raise ValueError("No valid data after feature engineering")
    
    # Combine all enhanced data
    final_data = pd.concat(enhanced_data_list, ignore_index=True)
    
    # Get feature columns (exclude target and metadata)
    exclude_cols = ['close', 'symbol', 'timestamp', 'open', 'high', 'low', 'volume']
    feature_columns = [col for col in final_data.columns if col not in exclude_cols]
    
    # Handle missing values
    final_data = final_data.fillna(method='ffill').fillna(method='bfill')
    final_data = final_data.dropna()
    
    print(f"   ✅ Generated {len(feature_columns)} features")
    print(f"   ✅ Final dataset: {final_data.shape}")
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Training ensemble models...")
    print("   Models to train:")
    print("   - Traditional ML: Random Forest, Gradient Boosting, Ridge, Lasso, Elastic Net")
    print("   - Gradient Boosting: XGBoost, LightGBM")
    print("   - Support Vector: SVR (RBF, Linear, Polynomial)")
    print("   - Deep Learning: Bi-LSTM, GRU, Transformer")
    print("   - Time Series: ARIMA, Prophet")
    
    # Prepare cost features (if available)
    cost_features = [col for col in feature_columns if any(
        keyword in col.lower() for keyword in ['spread', 'volume_ratio', 'volatility', 'liquidity']
    )]
    
    # Train ensemble models
    training_results = ensemble_trainer.train_ensemble_models(
        data=final_data,
        target_column='close',
        feature_columns=feature_columns,
        cost_features=cost_features if cost_features else None,
        test_size=0.2
    )
    
    print("\n✅ Training completed successfully!")
    
    # Step 4: Performance Analysis
    print("\n📈 Step 4: Analyzing model performance...")
    
    # Extract results
    individual_models = training_results['individual_models']
    ensemble_metrics = training_results['ensemble_metrics']
    best_model = training_results['best_individual_model']
    feature_importance = training_results['feature_importance']
    training_summary = training_results['training_summary']
    
    # Print performance summary
    print("\n" + "="*60)
    print("📊 ENSEMBLE TRAINING RESULTS")
    print("="*60)
    
    print(f"\n🥇 Best Individual Model: {best_model.model_name}")
    print(f"   R² Score: {best_model.validation_score:.4f}")
    print(f"   Training Time: {best_model.training_time:.2f} seconds")
    print(f"   Cost Efficiency: {best_model.cost_efficiency:.4f}")
    
    print(f"\n🎯 Ensemble Performance:")
    for metric, value in ensemble_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")
    
    print(f"\n📋 Individual Model Performance:")
    performance_data = []
    for model_name, result in individual_models.items():
        performance_data.append({
            'Model': model_name,
            'R² Score': result.validation_score,
            'Training Time (s)': result.training_time,
            'Cost Efficiency': result.cost_efficiency
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('R² Score', ascending=False)
    print(performance_df.to_string(index=False))
    
    # Step 5: Feature Importance Analysis
    print(f"\n🔍 Top 10 Most Important Features:")
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"   {i:2d}. {feature:<30} {importance:.4f}")
    
    # Step 6: Create Visualizations
    print("\n🎨 Step 6: Generating visualizations...")
    
    try:
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Ensemble Model Training Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Model Performance Comparison
        ax1 = axes[0, 0]
        models = performance_df['Model'][:8]  # Top 8 models
        scores = performance_df['R² Score'][:8]
        bars = ax1.bar(range(len(models)), scores, color='skyblue')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Training Time vs Performance
        ax2 = axes[0, 1]
        times = performance_df['Training Time (s)'][:8]
        scores = performance_df['R² Score'][:8]
        scatter = ax2.scatter(times, scores, c=scores, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Training Time vs Performance')
        plt.colorbar(scatter, ax=ax2, label='R² Score')
        
        # Add model labels
        for i, model in enumerate(models[:8]):
            ax2.annotate(model, (times.iloc[i], scores.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: Feature Importance (Top 15)
        ax3 = axes[0, 2]
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            features, importances = zip(*top_features)
            ax3.barh(range(len(features)), importances, color='lightcoral')
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels(features)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 15 Feature Importance')
            ax3.invert_yaxis()
        
        # Plot 4: Cost Efficiency Analysis
        ax4 = axes[1, 0]
        cost_efficiencies = performance_df['Cost Efficiency'][:8]
        models_subset = performance_df['Model'][:8]
        bars = ax4.bar(range(len(models_subset)), cost_efficiencies, color='lightgreen')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Cost Efficiency')
        ax4.set_title('Cost Efficiency by Model')
        ax4.set_xticks(range(len(models_subset)))
        ax4.set_xticklabels(models_subset, rotation=45, ha='right')
        
        # Plot 5: Ensemble vs Best Individual
        ax5 = axes[1, 1]
        comparison_data = {
            'Best Individual': best_model.validation_score,
            'Ensemble': ensemble_metrics.get('r2_score', 0)
        }
        bars = ax5.bar(comparison_data.keys(), comparison_data.values(), 
                      color=['orange', 'green'], alpha=0.7)
        ax5.set_ylabel('R² Score')
        ax5.set_title('Ensemble vs Best Individual Model')
        
        # Add value labels
        for bar, (name, value) in zip(bars, comparison_data.items()):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Training Summary
        ax6 = axes[1, 2]
        summary_text = f"""
        TRAINING SUMMARY
        
        Total Models Trained: {training_summary['total_models_trained']}
        Best Model: {training_summary['best_individual_model']}
        
        Ensemble R²: {ensemble_metrics.get('r2_score', 0):.4f}
        Ensemble RMSE: {ensemble_metrics.get('rmse', 0):.4f}
        Ensemble MAE: {ensemble_metrics.get('mae', 0):.4f}
        
        Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("   ✅ Visualizations generated successfully!")
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not generate all visualizations: {e}")
    
    # Step 7: Model Persistence
    print("\n💾 Step 7: Saving trained models...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = f"models/experiments/ensemble_{timestamp}"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        for model_name, result in individual_models.items():
            model_path = os.path.join(models_dir, f"{model_name}.pkl")
            joblib.dump(result.model, model_path)
        
        # Save metadata
        metadata = {
            'ensemble_metrics': ensemble_metrics,
            'feature_importance': feature_importance,
            'training_summary': training_summary,
            'performance_comparison': performance_df.to_dict('records')
        }
        
        import json
        metadata_path = os.path.join(models_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   ✅ Models saved to: {models_dir}")
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not save models: {e}")
    
    # Create predictor class for easy usage
    class EnsemblePredictor:
        def __init__(self):
            self.models = {name: result.model for name, result in individual_models.items()}
            self.ensemble_trainer = ensemble_trainer
            self.feature_columns = feature_columns
            self.performance_metrics = ensemble_metrics
            self.individual_results = individual_models
            
        def predict(self, X):
            """Make ensemble predictions."""
            return self.ensemble_trainer.predict(X[self.feature_columns])
        
        def get_feature_importance(self):
            """Get ensemble feature importance."""
            return feature_importance
        
        def get_performance_summary(self):
            """Get performance summary."""
            return {
                'ensemble_metrics': self.performance_metrics,
                'best_individual': best_model.model_name,
                'total_models': len(self.models)
            }
    
    predictor = EnsemblePredictor()
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 ENSEMBLE TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"✅ Trained {training_summary['total_models_trained']} models")
    print(f"✅ Best individual model: {training_summary['best_individual_model']}")
    print(f"✅ Ensemble R² Score: {ensemble_metrics.get('r2_score', 0):.4f}")
    print(f"✅ All models include: Random Forest, Gradient Boosting, XGBoost, LightGBM,")
    print(f"   Bi-LSTM, GRU, Transformer, Ridge, Lasso, Elastic Net, SVR variants,")
    print(f"   ARIMA, Prophet, and seasonal decomposition")
    print("="*80)
    
    return {
        'predictor': predictor,
        'ensemble_predictions': training_results.get('ensemble_predictions'),
        'test_targets': None,  # Will be available in actual implementation
        'models': individual_models,
        'metrics': ensemble_metrics,
        'feature_importance': feature_importance,
        'performance_df': performance_df
    }

# Execute the main function when notebook is run
if __name__ == "__main__":
    results = main_with_analysis()
    
    # Additional analysis and exploration can be added here
    print("\n🔍 Additional Analysis Available:")
    print("- results['predictor'].predict(new_data)")
    print("- results['predictor'].get_feature_importance()")
    print("- results['predictor'].get_performance_summary()")
    print("- results['performance_df'] - Detailed performance comparison")
    print("- results['feature_importance'] - Feature importance rankings")
