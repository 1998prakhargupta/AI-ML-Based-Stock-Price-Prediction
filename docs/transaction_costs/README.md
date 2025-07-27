# Transaction Cost Modeling System

## Overview

The Transaction Cost Modeling System provides comprehensive tools for calculating and analyzing trading costs across different brokers, instruments, and market conditions. This system is designed to integrate seamlessly with ML-based stock price prediction models to provide accurate cost estimates for trading decisions.

## Quick Start

```python
from src.trading.transaction_costs.models import TransactionRequest, TransactionType, InstrumentType
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
from decimal import Decimal

# Create a transaction request
request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)

# Calculate costs using Zerodha calculator
calculator = ZerodhaCalculator()
result = calculator.calculate_cost(request)

print(f"Total cost: ₹{result.total_cost}")
print(f"Brokerage: ₹{result.brokerage}")
print(f"STT: ₹{result.stt}")
print(f"GST: ₹{result.gst}")
```

## System Features

### ✅ Core Capabilities
- **Multiple Broker Support**: Zerodha, Angel Broking (Breeze), and extensible framework for adding more
- **Comprehensive Cost Breakdown**: Brokerage, STT, exchange charges, GST, stamp duty, and SEBI fees
- **Multiple Instrument Types**: Equity, derivatives, commodities, currency, and more
- **Real-time & Batch Processing**: Synchronous and asynchronous calculation support
- **Market Conditions Integration**: Bid-ask spreads, market impact, and timing considerations
- **Performance Optimization**: Built-in caching, batch processing, and monitoring

### 🎯 Use Cases
- **Trading Decision Support**: Accurate cost estimates for strategy optimization
- **Portfolio Management**: Total cost analysis for portfolio construction
- **Algorithmic Trading**: Real-time cost calculation for automated systems
- **Research & Backtesting**: Historical cost analysis for strategy validation
- **Compliance & Reporting**: Detailed cost breakdowns for regulatory reporting

## Documentation Structure

### 📚 [Getting Started Guide](getting_started.md)
Complete setup and basic usage instructions for new users.

### 🔧 [API Documentation](api/)
- [Cost Calculators](api/cost_calculators.md) - Core calculation interfaces
- [Brokers](api/brokers.md) - Broker-specific implementations
- [Market Impact](api/market_impact.md) - Market impact modeling
- [Spreads](api/spreads.md) - Bid-ask spread estimation
- [Aggregation](api/aggregation.md) - Cost aggregation and reporting

### 👥 [User Guide](user_guide/)
- [Installation](user_guide/installation.md) - Setup and dependencies
- [Basic Usage](user_guide/basic_usage.md) - Common usage patterns
- [Broker Setup](user_guide/broker_setup.md) - Configuring broker connections
- [Cost Analysis](user_guide/cost_analysis.md) - Advanced analysis features
- [Reporting](user_guide/reporting.md) - Generating cost reports

### ⚙️ [Configuration](configuration/)
- [Configuration Reference](configuration/configuration_reference.md) - Complete settings guide
- [Environment Setup](configuration/environment_setup.md) - Environment configuration
- [Broker Configs](configuration/broker_configs.md) - Broker-specific settings
- [Performance Tuning](configuration/performance_tuning.md) - Optimization settings

### 🏗️ [Technical Documentation](technical/)
- [Architecture](technical/architecture.md) - System design and components
- [Algorithms](technical/algorithms.md) - Cost calculation methodologies
- [Performance](technical/performance.md) - Performance considerations
- [Extending](technical/extending.md) - Adding new brokers and features

### 💡 [Examples](examples/)
- [Basic Calculations](examples/basic_calculations.py) - Simple cost calculations
- [ML Integration](examples/ml_integration.py) - Integration with ML models
- [Custom Broker](examples/custom_broker.py) - Creating custom broker calculators
- [Reporting Example](examples/reporting_example.py) - Generating reports
- [Advanced Usage](examples/advanced_usage.py) - Complex scenarios

### 🔧 [Troubleshooting](troubleshooting/)
- [Common Issues](troubleshooting/common_issues.md) - Frequently encountered problems
- [Performance Issues](troubleshooting/performance_issues.md) - Performance optimization
- [Configuration Errors](troubleshooting/configuration_errors.md) - Configuration problems

## Integration with ML Models

The transaction cost system is designed to integrate seamlessly with the existing ML-based stock price prediction framework:

```python
from src.models.features.cost_features import CostFeatureExtractor
from src.models.training.cost_aware_trainer import CostAwareTrainer

# Extract cost-related features for ML models
feature_extractor = CostFeatureExtractor()
cost_features = feature_extractor.extract_features(transactions)

# Train models with cost awareness
trainer = CostAwareTrainer()
model = trainer.train_with_cost_awareness(features, targets, cost_features)
```

## System Requirements

- Python 3.8+
- Dependencies: pandas, numpy, decimal, typing, dataclasses
- Optional: asyncio for async operations, redis for caching

## Project Integration

This documentation is part of the larger AI-ML-Based-Stock-Price-Prediction project. For general project information, see:
- [Main Project README](../../README.md)
- [Project Documentation](../)
- [API Compliance Documentation](../API_COMPLIANCE_DOCUMENTATION.md)

## Support

For issues, questions, or contributions:
- Check the [troubleshooting guide](troubleshooting/)
- Review existing [issues](https://github.com/1998prakhargupta/AI-ML-Based-Stock-Price-Prediction/issues)
- Create a new issue with the `transaction-costs` label

---

*This documentation covers the comprehensive transaction cost modeling system. For the latest updates and examples, please refer to the individual documentation sections.*