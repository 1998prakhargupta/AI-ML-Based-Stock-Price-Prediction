# Broker-Specific Fee Structure Implementation

This module provides comprehensive broker-specific fee calculation for Indian brokers, supporting accurate cost computation for trading across different instruments and exchanges.

## Supported Brokers

### 1. Zerodha (Kite Connect)
- **Equity Delivery**: ₹0 (Free)
- **Equity Intraday**: 0.03% or ₹20 (whichever is lower)
- **Options**: ₹20 per order
- **Futures**: 0.03% or ₹20 (whichever is lower)
- **Currency**: 0.03% or ₹20 (whichever is lower)
- **Commodity**: 0.03% or ₹20 (whichever is lower)

### 2. ICICI Securities (Breeze Connect)
- **Equity Delivery**: ₹20 per order
- **Equity Intraday**: 0.05% or ₹20 (whichever is lower)
- **Options**: ₹20 per order
- **Futures**: 0.05% or ₹20 (whichever is lower)
- **Currency**: 0.05% or ₹20 (whichever is lower)
- **Commodity**: 0.05% or ₹20 (whichever is lower)

## Regulatory Charges

The system accurately calculates all applicable regulatory charges:

- **STT (Securities Transaction Tax)**: Varies by instrument type
- **CTT (Commodities Transaction Tax)**: 0.01% on sell side
- **GST**: 18% on brokerage and statutory charges
- **SEBI Charges**: ₹10 per crore of transaction value
- **Exchange Transaction Charges**: Varies by segment (NSE/BSE)
- **Exchange Clearing Charges**: Varies by segment
- **Stamp Duty**: 0.003% on buy side (maximum ₹1500)

## Quick Start

### Basic Usage

```python
from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory
from src.trading.transaction_costs.models import *
from decimal import Decimal

# Create calculator
calculator = BrokerFactory.create_calculator('zerodha')

# Define transaction
request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)

# Configure broker
broker_config = BrokerConfiguration(
    broker_name='Zerodha',
    active=True
)

# Calculate costs
result = calculator.calculate_cost(request, broker_config)
print(f'Total cost: ₹{result.total_cost:.2f}')
```

### Broker Comparison

```python
# Compare costs between brokers
zerodha = BrokerFactory.create_calculator('zerodha')
icici = BrokerFactory.create_calculator('icici')

zerodha_config = BrokerConfiguration(broker_name='Zerodha', active=True)
icici_config = BrokerConfiguration(broker_name='ICICI Securities', active=True)

zerodha_result = zerodha.calculate_cost(request, zerodha_config)
icici_result = icici.calculate_cost(request, icici_config)

print(f"Zerodha: ₹{zerodha_result.total_cost:.2f}")
print(f"ICICI: ₹{icici_result.total_cost:.2f}")
```

### Batch Processing

```python
# Process multiple transactions
requests = [
    TransactionRequest(...),
    TransactionRequest(...),
    # ... more requests
]

results = calculator.calculate_batch_costs(requests, broker_config)
total_cost = sum(result.total_cost for result in results)
```

### Detailed Breakdown

```python
# Get detailed cost breakdown
breakdown = calculator.get_detailed_breakdown(request, broker_config)
print(f"Commission: ₹{breakdown['brokerage']['commission']:.2f}")
print(f"STT: ₹{breakdown['regulatory_charges']['statutory_charges']['stt']:.2f}")
print(f"Stamp Duty: ₹{breakdown['regulatory_charges']['statutory_charges']['stamp_duty']:.2f}")
print(f"GST: ₹{breakdown['regulatory_charges']['taxes']['gst']:.2f}")
```

## Advanced Features

### Async Support

```python
# Asynchronous calculation
result = await calculator.calculate_cost_async(request, broker_config)
```

### Market Impact Calculation

```python
# Include market conditions for impact calculation
market_conditions = MarketConditions(
    bid_price=Decimal('2495.00'),
    ask_price=Decimal('2505.00'),
    volume=1000000
)

result = calculator.calculate_cost(request, broker_config, market_conditions)
```

### Performance Monitoring

```python
# Get performance statistics
stats = calculator.get_performance_stats()
print(f"Average calculation time: {stats['average_calculation_time']:.6f}s")
print(f"Total calculations: {stats['total_calculations']}")
```

## Supported Instruments

- **Equity**: Stocks, ETFs
- **Options**: Call and Put options
- **Futures**: Index and stock futures
- **Currency**: Currency pairs
- **Commodity**: Commodity contracts

## Supported Exchanges

- **NSE (National Stock Exchange)**
- **BSE (Bombay Stock Exchange)**

## File Structure

```
src/trading/transaction_costs/brokers/
├── __init__.py
├── broker_factory.py          # Factory for creating calculators
├── zerodha_calculator.py      # Zerodha-specific implementation
├── breeze_calculator.py       # ICICI Securities implementation
└── regulatory/
    ├── __init__.py
    ├── charges_calculator.py  # Main regulatory charges coordinator
    ├── stt_calculator.py      # Securities Transaction Tax
    ├── gst_calculator.py      # Goods and Services Tax
    └── stamp_duty_calculator.py # Stamp duty calculations
```

## Error Handling

The system provides comprehensive error handling:

```python
from src.trading.transaction_costs.exceptions import (
    BrokerConfigurationError,
    InvalidTransactionError,
    CalculationError
)

try:
    result = calculator.calculate_cost(request, broker_config)
except BrokerConfigurationError as e:
    print(f"Broker configuration error: {e}")
except InvalidTransactionError as e:
    print(f"Invalid transaction: {e}")
except CalculationError as e:
    print(f"Calculation error: {e}")
```

## Demo

Run the comprehensive demo to see all features:

```bash
python broker_calculator_demo.py
```

## Testing

The implementation includes comprehensive tests:

```bash
python -m src.trading.tests.test_broker_calculators
```

## Performance

- **Real-time calculations**: Sub-millisecond response times
- **Batch processing**: Optimized for high-volume calculations
- **Caching**: Built-in result caching for improved performance
- **Thread-safe**: Concurrent calculation support

## Validation

All calculations are validated against official broker calculators:
- [Zerodha Brokerage Calculator](https://zerodha.com/brokerage-calculator)
- [ICICI Securities Brokerage Calculator](https://www.icicidirect.com/brokerage)

## Contributing

To add support for new brokers:

1. Create a new calculator class inheriting from `CostCalculatorBase`
2. Implement the abstract methods for commission and regulatory fees
3. Add the broker to `BrokerFactory.SUPPORTED_BROKERS`
4. Add comprehensive tests for the new broker

## Notes

- All amounts are in Indian Rupees (INR)
- Rates are current as of 2024 and should be updated periodically
- The system supports both NSE and BSE exchanges
- Market impact calculations are included for institutional-level analysis