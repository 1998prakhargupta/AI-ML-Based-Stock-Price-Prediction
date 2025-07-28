#!/usr/bin/env python3
"""
Enterprise Ensemble Model Demo
=============================

This demo script showcases the complete enterprise-level ensemble modeling system
with all implemented models including:

Traditional ML: Random Forest, Gradient Boosting, Ridge, Lasso, Elastic Net
Gradient Boosting: XGBoost, LightGBM  
Support Vector: SVR (RBF, Linear, Polynomial)
Deep Learning: Bi-LSTM, GRU, Transformer
Time Series: ARIMA, Prophet, Seasonal Decomposition

Features:
- Docker containerization
- Kubernetes deployment
- Comprehensive configuration management
- Cost-aware training
- Real-time monitoring
- Professional reporting

Usage:
    python scripts/enterprise_ensemble_demo.py
    
Environment:
    docker-compose up -d  # Start full infrastructure
    kubectl apply -f k8s/ # Deploy to Kubernetes
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import yaml
import joblib
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_demo_environment():
    """Setup the demo environment with necessary directories and configurations."""
    print("🔧 Setting up demo environment...")
    
    # Create necessary directories
    directories = [
        "data/demo",
        "models/demo",
        "logs/demo", 
        "plots/demo",
        "reports/demo",
        "config/demo"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create demo configuration
    demo_config = {
        "ensemble": {
            "models": {
                "traditional_ml": {
                    "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                    "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
                    "ridge": {"alpha": 1.0, "random_state": 42},
                    "lasso": {"alpha": 1.0, "random_state": 42},
                    "elastic_net": {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42}
                },
                "gradient_boosting": {
                    "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
                    "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
                },
                "support_vector": {
                    "svr_rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                    "svr_linear": {"kernel": "linear", "C": 1.0},
                    "svr_poly": {"kernel": "poly", "degree": 3, "C": 1.0}
                },
                "deep_learning": {
                    "bi_lstm": {"units": 50, "dropout": 0.2, "epochs": 50},
                    "gru": {"units": 50, "dropout": 0.2, "epochs": 50},
                    "transformer": {"d_model": 64, "nhead": 8, "epochs": 50}
                },
                "time_series": {
                    "arima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
                    "prophet": {"yearly_seasonality": True, "weekly_seasonality": True}
                }
            },
            "ensemble_methods": ["weighted_average", "voting", "stacking"],
            "validation": {
                "method": "time_series_split",
                "n_splits": 5,
                "test_size": 0.2
            }
        },
        "training": {
            "cost_aware": True,
            "hyperparameter_tuning": True,
            "early_stopping": True,
            "feature_selection": True
        }
    }
    
    with open("config/demo/ensemble-config.yaml", "w") as f:
        yaml.dump(demo_config, f, indent=2)
    
    print("✅ Demo environment setup complete!")
    return demo_config

def generate_realistic_market_data(symbols=None, period_days=730):
    """Generate realistic market data for demonstration."""
    print(f"📊 Generating realistic market data for {period_days} days...")
    
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    all_data = []
    
    for symbol in symbols:
        print(f"   Generating data for {symbol}...")
        
        # Generate realistic price movements
        # Start with a base price
        base_price = np.random.uniform(50, 500)
        
        # Generate returns with volatility clustering
        returns = []
        volatility = 0.02  # Base volatility
        
        for i in range(len(dates)):
            # Volatility clustering
            if i > 0:
                volatility = 0.95 * volatility + 0.05 * abs(returns[-1])
            
            # Generate return with some trend and mean reversion
            trend = 0.0005 if np.random.random() > 0.3 else -0.0002
            mean_reversion = -0.1 * returns[-1] if returns else 0
            
            daily_return = np.random.normal(trend + mean_reversion, volatility)
            returns.append(daily_return)
        
        # Convert returns to prices
        returns = np.array(returns)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        opens = prices * (1 + np.random.normal(0, 0.005, len(prices)))
        
        # Highs and lows with realistic spreads
        daily_ranges = np.random.exponential(0.02, len(prices))
        highs = np.maximum(opens, prices) * (1 + daily_ranges/2)
        lows = np.minimum(opens, prices) * (1 - daily_ranges/2)
        
        # Volume with realistic patterns
        base_volume = np.random.uniform(1e6, 1e8)
        volume_multiplier = 1 + np.abs(returns) * 10  # Higher volume on big moves
        volumes = base_volume * volume_multiplier
        
        # Create DataFrame for this symbol
        symbol_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes.astype(int)
        })
        
        all_data.append(symbol_data)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Generated {len(combined_data)} data points for {len(symbols)} symbols")
    return combined_data

def create_advanced_features(data):
    """Create advanced technical indicators and features."""
    print("🔧 Creating advanced technical features...")
    
    enhanced_data = []
    
    for symbol in data['symbol'].unique():
        symbol_data = data[data['symbol'] == symbol].copy().sort_values('timestamp')
        
        if len(symbol_data) < 50:  # Skip if insufficient data
            continue
        
        print(f"   Processing features for {symbol}...")
        
        # Price-based features
        symbol_data['returns'] = symbol_data['close'].pct_change()
        symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            symbol_data[f'sma_{window}'] = symbol_data['close'].rolling(window).mean()
            symbol_data[f'ema_{window}'] = symbol_data['close'].ewm(span=window).mean()
        
        # Volatility measures
        symbol_data['volatility_10'] = symbol_data['returns'].rolling(10).std()
        symbol_data['volatility_30'] = symbol_data['returns'].rolling(30).std()
        
        # Price position indicators
        symbol_data['rsi_14'] = calculate_rsi(symbol_data['close'], 14)
        symbol_data['bb_position'] = calculate_bollinger_position(symbol_data['close'], 20)
        
        # Volume indicators
        symbol_data['volume_sma_20'] = symbol_data['volume'].rolling(20).mean()
        symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_20']
        
        # Price momentum
        for lag in [1, 5, 10, 20]:
            symbol_data[f'momentum_{lag}'] = symbol_data['close'] / symbol_data['close'].shift(lag) - 1
        
        # High-low spread
        symbol_data['hl_spread'] = (symbol_data['high'] - symbol_data['low']) / symbol_data['close']
        
        # MACD
        ema_12 = symbol_data['close'].ewm(span=12).mean()
        ema_26 = symbol_data['close'].ewm(span=26).mean()
        symbol_data['macd'] = ema_12 - ema_26
        symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
        symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
        
        # Stochastic oscillator
        low_14 = symbol_data['low'].rolling(14).min()
        high_14 = symbol_data['high'].rolling(14).max()
        symbol_data['stoch_k'] = 100 * (symbol_data['close'] - low_14) / (high_14 - low_14)
        symbol_data['stoch_d'] = symbol_data['stoch_k'].rolling(3).mean()
        
        enhanced_data.append(symbol_data)
    
    # Combine enhanced data
    final_data = pd.concat(enhanced_data, ignore_index=True)
    
    # Remove infinite and null values
    final_data = final_data.replace([np.inf, -np.inf], np.nan)
    final_data = final_data.fillna(method='ffill').fillna(method='bfill')
    final_data = final_data.dropna()
    
    print(f"✅ Created {final_data.shape[1]} features")
    return final_data

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_position(prices, window=20):
    """Calculate position within Bollinger Bands."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return (prices - lower_band) / (upper_band - lower_band)

def run_enterprise_ensemble_demo():
    """Run the complete enterprise ensemble demonstration."""
    print("🚀 Starting Enterprise Ensemble Model Demo")
    print("=" * 80)
    
    # Setup environment
    config = setup_demo_environment()
    
    # Generate market data
    market_data = generate_realistic_market_data(
        symbols=["DEMO_STOCK_1", "DEMO_STOCK_2", "DEMO_STOCK_3"],
        period_days=1000
    )
    
    # Create features
    enhanced_data = create_advanced_features(market_data)
    
    # Save demo data
    demo_data_path = "data/demo/market_data.csv"
    enhanced_data.to_csv(demo_data_path, index=False)
    print(f"📁 Demo data saved to: {demo_data_path}")
    
    # Import and initialize ensemble trainer
    try:
        from src.models.training.ensemble_trainer import EnterpriseEnsembleTrainer
        
        print("\n🤖 Initializing Enterprise Ensemble Trainer...")
        trainer = EnterpriseEnsembleTrainer(
            config_path="config/demo/ensemble-config.yaml"
        )
        
        # Prepare training data
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'volume']
        feature_columns = [col for col in enhanced_data.columns if col not in exclude_cols + ['close']]
        
        print(f"📊 Training data: {enhanced_data.shape}")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"🎯 Target: close price")
        
        # Train ensemble models
        print("\n🎯 Training all ensemble models...")
        training_results = trainer.train_ensemble_models(
            data=enhanced_data,
            target_column='close',
            feature_columns=feature_columns,
            test_size=0.2
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 ENTERPRISE ENSEMBLE TRAINING RESULTS")
        print("="*60)
        
        individual_models = training_results['individual_models']
        ensemble_metrics = training_results['ensemble_metrics']
        best_model = training_results['best_individual_model']
        
        print(f"\n🥇 Best Individual Model: {best_model.model_name}")
        print(f"   R² Score: {best_model.validation_score:.4f}")
        print(f"   Training Time: {best_model.training_time:.2f}s")
        print(f"   Cost Efficiency: {best_model.cost_efficiency:.4f}")
        
        print(f"\n🎯 Ensemble Performance:")
        for metric, value in ensemble_metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        # Model comparison table
        print(f"\n📋 All Models Performance:")
        performance_data = []
        for model_name, result in individual_models.items():
            performance_data.append({
                'Model': model_name,
                'R² Score': f"{result.validation_score:.4f}",
                'Training Time (s)': f"{result.training_time:.2f}",
                'Cost Efficiency': f"{result.cost_efficiency:.4f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('R² Score', ascending=False)
        print(performance_df.to_string(index=False))
        
        # Save results
        results_path = "reports/demo/ensemble_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'ensemble_metrics': ensemble_metrics,
                'best_model': {
                    'name': best_model.model_name,
                    'score': best_model.validation_score,
                    'training_time': best_model.training_time
                },
                'training_summary': training_results['training_summary']
            }, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_path}")
        
        # Demonstration of predictions
        print(f"\n🔮 Making sample predictions...")
        test_data = enhanced_data[feature_columns].tail(10)
        predictions = trainer.predict(test_data)
        
        print("Sample predictions (last 10 data points):")
        for i, pred in enumerate(predictions[-10:], 1):
            print(f"   {i:2d}. ${pred:.2f}")
        
        # Docker and Kubernetes demo
        print(f"\n🐋 Docker & Kubernetes Deployment Ready!")
        print("   To deploy the full enterprise infrastructure:")
        print("   1. docker-compose up -d    # Start all services")
        print("   2. kubectl apply -f k8s/   # Deploy to Kubernetes")
        print("   3. Access API at: http://localhost:8000")
        print("   4. Monitoring: http://localhost:3000 (Grafana)")
        
        print("\n✅ Enterprise Ensemble Demo completed successfully!")
        print("📊 All models trained and ready for production deployment")
        
    except ImportError as e:
        print(f"❌ Error importing ensemble trainer: {e}")
        print("Please ensure the ensemble trainer is properly implemented.")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Check the configuration and data paths.")

def main():
    """Main demo function."""
    try:
        run_enterprise_ensemble_demo()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
