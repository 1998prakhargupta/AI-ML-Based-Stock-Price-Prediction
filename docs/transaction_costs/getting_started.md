# Getting Started with Transaction Cost Modeling

This guide will help you get started with the Transaction Cost Modeling System quickly and easily.

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- Basic understanding of Python programming
- Access to the AI-ML-Based-Stock-Price-Prediction project

## Installation

### 1. Project Setup

If you haven't already set up the main project:

```bash
git clone https://github.com/1998prakhargupta/AI-ML-Based-Stock-Price-Prediction.git
cd AI-ML-Based-Stock-Price-Prediction
```

### 2. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Or use the Makefile
make install
```

### 3. Verify Installation

Run a quick test to ensure everything is working:

```python
from src.trading.transaction_costs.models import TransactionRequest
print("Transaction cost system ready!")
```

## Your First Cost Calculation

Let's calculate the cost of buying 100 shares of Reliance Industries through Zerodha:

### Step 1: Import Required Modules

```python
from src.trading.transaction_costs.models import (
    TransactionRequest, 
    TransactionType, 
    InstrumentType
)
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
from decimal import Decimal
```

### Step 2: Create a Transaction Request

```python
# Define the transaction
request = TransactionRequest(
    symbol='RELIANCE',           # Stock symbol
    quantity=100,                # Number of shares
    price=Decimal('2500.00'),    # Price per share in INR
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)
```

### Step 3: Calculate Costs

```python
# Initialize the calculator
calculator = ZerodhaCalculator()

# Calculate the costs
result = calculator.calculate_cost(request)

# Display results
print(f"Transaction: {request.transaction_type.name} {request.quantity} shares of {request.symbol}")
print(f"Share Price: ₹{request.price}")
print(f"Total Value: ₹{request.quantity * request.price}")
print("\n--- Cost Breakdown ---")
print(f"Brokerage: ₹{result.brokerage:.2f}")
print(f"STT: ₹{result.stt:.2f}")
print(f"Exchange Charges: ₹{result.exchange_charges:.2f}")
print(f"GST: ₹{result.gst:.2f}")
print(f"SEBI Charges: ₹{result.sebi_charges:.2f}")
print(f"Stamp Duty: ₹{result.stamp_duty:.2f}")
print(f"\nTotal Cost: ₹{result.total_cost:.2f}")
print(f"Net Amount: ₹{result.net_amount:.2f}")
```

## Working with Different Brokers

The system supports multiple brokers. Here's how to use different ones:

### Zerodha

```python
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator

calculator = ZerodhaCalculator()
result = calculator.calculate_cost(request)
```

### Angel Broking (Breeze)

```python
from src.trading.transaction_costs.brokers.breeze_calculator import BreezeCalculator

calculator = BreezeCalculator()
result = calculator.calculate_cost(request)
```

### Compare Costs Across Brokers

```python
def compare_brokers(request):
    """Compare costs across different brokers."""
    brokers = {
        'Zerodha': ZerodhaCalculator(),
        'Angel Broking': BreezeCalculator()
    }
    
    results = {}
    for name, calculator in brokers.items():
        try:
            result = calculator.calculate_cost(request)
            results[name] = result.total_cost
        except Exception as e:
            results[name] = f"Error: {e}"
    
    return results

# Compare costs
comparison = compare_brokers(request)
for broker, cost in comparison.items():
    print(f"{broker}: ₹{cost}")
```

## Different Transaction Types

### Buying Stocks

```python
buy_request = TransactionRequest(
    symbol='TCS',
    quantity=50,
    price=Decimal('3200.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)
```

### Selling Stocks

```python
sell_request = TransactionRequest(
    symbol='TCS',
    quantity=50,
    price=Decimal('3250.00'),
    transaction_type=TransactionType.SELL,
    instrument_type=InstrumentType.EQUITY
)
```

### Options Trading

```python
option_request = TransactionRequest(
    symbol='NIFTY50',
    quantity=1,  # Number of lots
    price=Decimal('150.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.OPTION,
    metadata={'strike_price': 17500, 'expiry': '2024-01-25'}
)
```

## Batch Processing

For processing multiple transactions:

```python
def calculate_portfolio_costs(transactions, calculator):
    """Calculate costs for multiple transactions."""
    total_cost = Decimal('0')
    results = []
    
    for transaction in transactions:
        result = calculator.calculate_cost(transaction)
        results.append(result)
        total_cost += result.total_cost
    
    return results, total_cost

# Example usage
transactions = [buy_request, sell_request, option_request]
calculator = ZerodhaCalculator()

results, total_cost = calculate_portfolio_costs(transactions, calculator)
print(f"Total portfolio cost: ₹{total_cost}")
```

## Asynchronous Operations

For high-performance applications:

```python
import asyncio

async def calculate_cost_async(request, calculator):
    """Calculate cost asynchronously."""
    result = await calculator.calculate_cost_async(request)
    return result

async def main():
    calculator = ZerodhaCalculator()
    
    # Process multiple requests concurrently
    tasks = [
        calculate_cost_async(buy_request, calculator),
        calculate_cost_async(sell_request, calculator)
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"Total cost: ₹{result.total_cost}")

# Run async example
# asyncio.run(main())
```

## Configuration

### Basic Configuration

```python
from src.trading.cost_config.base_config import CostConfiguration

# Initialize configuration
config = CostConfiguration()

# Load default broker configurations
config.create_default_broker_configurations()

# Get broker configuration
zerodha_config = config.get_broker_configuration('Zerodha')
print(f"Zerodha equity commission: {zerodha_config.equity_commission}")
```

### Custom Configuration

```python
# Modify settings
config.update_setting('calculation.precision_decimal_places', 4)
config.update_setting('caching.enabled', True)
config.update_setting('caching.ttl_seconds', 3600)

# Save configuration
config.save_config()
```

## Error Handling

The system provides comprehensive error handling:

```python
from src.trading.transaction_costs.exceptions import (
    TransactionCostError,
    InvalidTransactionError,
    BrokerConfigurationError
)

try:
    # Invalid transaction (negative quantity)
    invalid_request = TransactionRequest(
        symbol='INVALID',
        quantity=-10,  # This will raise an error
        price=Decimal('100.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY
    )
except InvalidTransactionError as e:
    print(f"Transaction error: {e}")
except TransactionCostError as e:
    print(f"General cost error: {e}")
```

## Next Steps

Now that you've learned the basics, explore these advanced topics:

1. **[User Guide](user_guide/)** - Detailed usage patterns and configurations
2. **[API Documentation](api/)** - Complete API reference
3. **[Examples](examples/)** - Real-world usage examples
4. **[Configuration](configuration/)** - Advanced configuration options
5. **[Integration](../ML_INTEGRATION.md)** - Integrate with ML models

## Common Use Cases

### 1. Strategy Backtesting

```python
def backtest_strategy_with_costs(trades, calculator):
    """Backtest a trading strategy including transaction costs."""
    total_cost = Decimal('0')
    
    for trade in trades:
        cost_result = calculator.calculate_cost(trade)
        total_cost += cost_result.total_cost
        
        # Adjust strategy returns by transaction costs
        trade.net_pnl = trade.gross_pnl - cost_result.total_cost
    
    return total_cost

# Use in your backtesting framework
trades = load_historical_trades()
calculator = ZerodhaCalculator()
total_transaction_cost = backtest_strategy_with_costs(trades, calculator)
```

### 2. Real-time Trading

```python
def execute_trade_with_cost_check(trade_signal, max_cost_percent=0.1):
    """Execute trade only if costs are below threshold."""
    calculator = ZerodhaCalculator()
    
    # Calculate expected cost
    cost_result = calculator.calculate_cost(trade_signal)
    
    # Check if cost is within acceptable range
    cost_percent = cost_result.total_cost / trade_signal.value
    
    if cost_percent <= max_cost_percent:
        print(f"Executing trade - Cost: {cost_percent:.2%}")
        # Execute trade
        return True
    else:
        print(f"Skipping trade - Cost too high: {cost_percent:.2%}")
        return False
```

### 3. Portfolio Optimization

```python
def optimize_portfolio_with_costs(positions, calculator):
    """Optimize portfolio considering transaction costs."""
    total_cost = Decimal('0')
    optimized_positions = []
    
    for position in positions:
        # Calculate cost for this position
        cost_result = calculator.calculate_cost(position)
        
        # Only include if cost-benefit is favorable
        if position.expected_return > cost_result.total_cost:
            optimized_positions.append(position)
            total_cost += cost_result.total_cost
    
    return optimized_positions, total_cost
```

## Help and Support

If you encounter issues:

1. Check the [troubleshooting guide](troubleshooting/common_issues.md)
2. Review the [API documentation](api/) for detailed method signatures
3. Look at the [examples](examples/) for similar use cases
4. Create an issue on GitHub with the `transaction-costs` label

Happy trading! 🚀