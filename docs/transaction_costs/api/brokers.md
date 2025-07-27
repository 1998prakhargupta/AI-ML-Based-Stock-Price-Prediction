# Broker-Specific Calculators API

This documentation covers the broker-specific implementations of cost calculators, including their unique fee structures, supported instruments, and special features.

## Overview

Each broker has its own fee structure and calculation rules. The system provides dedicated calculator classes for each supported broker, all extending the `CostCalculatorBase` abstract class.

## Supported Brokers

- **Zerodha (Kite Connect)** - Popular discount broker with competitive rates
- **ICICI Securities (Breeze)** - Full-service broker with comprehensive offerings
- **Generic Broker** - Configurable calculator for custom brokers

## Zerodha Calculator

### ZerodhaCalculator

**Module**: `src.trading.transaction_costs.brokers.zerodha_calculator`

Implements Zerodha's official brokerage structure with accurate fee calculations.

#### Constructor

```python
ZerodhaCalculator(exchange: str = 'NSE')
```

**Parameters:**
- `exchange` (str): Exchange name ('NSE' or 'BSE')

#### Brokerage Structure

Zerodha follows a simple and transparent fee structure:

| Instrument Type | Fee Structure |
|----------------|---------------|
| Equity Delivery | ₹0 (Free) |
| Equity Intraday | 0.03% or ₹20 (whichever is lower) |
| Options | ₹20 per order |
| Futures | 0.03% or ₹20 (whichever is lower) |
| Currency | 0.03% or ₹20 (whichever is lower) |
| Commodity | 0.03% or ₹20 (whichever is lower) |

#### Usage Example

```python
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
from src.trading.transaction_costs.models import TransactionRequest, TransactionType, InstrumentType
from decimal import Decimal

# Initialize calculator
calculator = ZerodhaCalculator(exchange='NSE')

# Equity delivery (free brokerage)
delivery_request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY,
    order_type=OrderType.MARKET,
    metadata={'product_type': 'CNC'}  # Cash and Carry (delivery)
)

result = calculator.calculate_cost(delivery_request)
print(f"Delivery brokerage: ₹{result.brokerage}")  # ₹0.00

# Equity intraday
intraday_request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY,
    order_type=OrderType.MARKET,
    metadata={'product_type': 'MIS'}  # Margin Intraday Square-off
)

result = calculator.calculate_cost(intraday_request)
print(f"Intraday brokerage: ₹{result.brokerage}")  # min(0.03% of 2,50,000, ₹20) = ₹20

# Options trading
option_request = TransactionRequest(
    symbol='NIFTY24JAN17500CE',
    quantity=1,  # 1 lot (usually 50 shares for NIFTY)
    price=Decimal('150.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.OPTION,
    metadata={'lot_size': 50}
)

result = calculator.calculate_cost(option_request)
print(f"Option brokerage: ₹{result.brokerage}")  # ₹20.00
```

#### Special Features

##### Auto Product Type Detection

```python
# Automatic detection based on metadata
request = TransactionRequest(
    symbol='TCS',
    quantity=50,
    price=Decimal('3200.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)

# For delivery (if no product_type specified, defaults to delivery)
result_delivery = calculator.calculate_cost(request)

# For intraday (specify in metadata)
request.metadata = {'product_type': 'MIS'}
result_intraday = calculator.calculate_cost(request)

print(f"Delivery cost: ₹{result_delivery.total_cost}")
print(f"Intraday cost: ₹{result_intraday.total_cost}")
```

##### Exchange-Specific Calculations

```python
# NSE exchange
nse_calculator = ZerodhaCalculator(exchange='NSE')

# BSE exchange
bse_calculator = ZerodhaCalculator(exchange='BSE')

# Different exchanges may have different regulatory charges
nse_result = nse_calculator.calculate_cost(request)
bse_result = bse_calculator.calculate_cost(request)
```

## ICICI Securities (Breeze) Calculator

### BreezeCalculator

**Module**: `src.trading.transaction_costs.brokers.breeze_calculator`

Implements ICICI Securities' brokerage structure with comprehensive fee calculations.

#### Constructor

```python
BreezeCalculator(exchange: str = 'NSE')
```

#### Brokerage Structure

ICICI Securities has a slightly different fee structure:

| Instrument Type | Fee Structure |
|----------------|---------------|
| Equity Delivery | ₹20 per order |
| Equity Intraday | 0.05% or ₹20 (whichever is lower) |
| Options | ₹20 per order |
| Futures | 0.05% or ₹20 (whichever is lower) |
| Currency | 0.05% or ₹20 (whichever is lower) |
| Commodity | 0.05% or ₹20 (whichever is lower) |

#### Usage Example

```python
from src.trading.transaction_costs.brokers.breeze_calculator import BreezeCalculator

# Initialize calculator
calculator = BreezeCalculator(exchange='NSE')

# Equity delivery (₹20 brokerage)
delivery_request = TransactionRequest(
    symbol='INFY',
    quantity=100,
    price=Decimal('1500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY,
    metadata={'product_type': 'DELIVERY'}
)

result = calculator.calculate_cost(delivery_request)
print(f"ICICI delivery brokerage: ₹{result.brokerage}")  # ₹20.00

# Compare with Zerodha
zerodha_calc = ZerodhaCalculator()
zerodha_result = zerodha_calc.calculate_cost(delivery_request)
print(f"Zerodha delivery brokerage: ₹{zerodha_result.brokerage}")  # ₹0.00

print(f"Difference: ₹{result.brokerage - zerodha_result.brokerage}")
```

## Broker Comparison

### Side-by-Side Comparison

```python
def compare_brokers(request):
    """Compare costs across different brokers."""
    brokers = {
        'Zerodha': ZerodhaCalculator(),
        'ICICI Securities': BreezeCalculator()
    }
    
    results = {}
    for name, calculator in brokers.items():
        result = calculator.calculate_cost(request)
        results[name] = {
            'brokerage': result.brokerage,
            'total_cost': result.total_cost,
            'effective_rate': result.total_cost / request.value * 100
        }
    
    return results

# Example comparison
request = TransactionRequest(
    symbol='HDFC',
    quantity=100,
    price=Decimal('1600.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)

comparison = compare_brokers(request)
for broker, costs in comparison.items():
    print(f"{broker}:")
    print(f"  Brokerage: ₹{costs['brokerage']}")
    print(f"  Total Cost: ₹{costs['total_cost']}")
    print(f"  Effective Rate: {costs['effective_rate']:.3f}%")
    print()
```

### Best Broker for Different Scenarios

```python
def find_best_broker_for_scenario(scenarios):
    """Find the best broker for different trading scenarios."""
    brokers = {
        'Zerodha': ZerodhaCalculator(),
        'ICICI Securities': BreezeCalculator()
    }
    
    results = {}
    
    for scenario_name, request in scenarios.items():
        min_cost = float('inf')
        best_broker = None
        
        for broker_name, calculator in brokers.items():
            result = calculator.calculate_cost(request)
            if result.total_cost < min_cost:
                min_cost = result.total_cost
                best_broker = broker_name
        
        results[scenario_name] = {
            'best_broker': best_broker,
            'cost': min_cost
        }
    
    return results

# Define scenarios
scenarios = {
    'Large Delivery Trade': TransactionRequest(
        symbol='RELIANCE', quantity=1000, price=Decimal('2500.00'),
        transaction_type=TransactionType.BUY, instrument_type=InstrumentType.EQUITY,
        metadata={'product_type': 'CNC'}
    ),
    'Small Intraday Trade': TransactionRequest(
        symbol='TCS', quantity=10, price=Decimal('3200.00'),
        transaction_type=TransactionType.BUY, instrument_type=InstrumentType.EQUITY,
        metadata={'product_type': 'MIS'}
    ),
    'Options Trade': TransactionRequest(
        symbol='BANKNIFTY24JAN45000CE', quantity=1, price=Decimal('200.00'),
        transaction_type=TransactionType.BUY, instrument_type=InstrumentType.OPTION,
        metadata={'lot_size': 25}
    )
}

best_choices = find_best_broker_for_scenario(scenarios)
for scenario, choice in best_choices.items():
    print(f"{scenario}: {choice['best_broker']} (₹{choice['cost']})")
```

## Custom Broker Implementation

### Creating a Custom Broker Calculator

```python
from src.trading.transaction_costs.base_cost_calculator import CostCalculatorBase
from decimal import Decimal

class CustomBrokerCalculator(CostCalculatorBase):
    """Custom broker calculator implementation."""
    
    def __init__(self, broker_name: str, commission_rates: dict):
        super().__init__(
            calculator_name=broker_name,
            version="1.0.0",
            supported_instruments=[InstrumentType.EQUITY, InstrumentType.OPTION]
        )
        self.commission_rates = commission_rates
    
    def _calculate_commission(self, request, broker_config):
        """Calculate commission based on custom rates."""
        if request.instrument_type == InstrumentType.EQUITY:
            if request.metadata.get('product_type') == 'MIS':
                # Intraday
                rate = self.commission_rates.get('equity_intraday_rate', Decimal('0.0005'))
                max_fee = self.commission_rates.get('equity_intraday_max', Decimal('20'))
                commission = min(request.value * rate, max_fee)
            else:
                # Delivery
                commission = self.commission_rates.get('equity_delivery', Decimal('10'))
        elif request.instrument_type == InstrumentType.OPTION:
            commission = self.commission_rates.get('option_flat_fee', Decimal('25'))
        else:
            commission = Decimal('0')
        
        return commission
    
    def _calculate_regulatory_fees(self, request, broker_config):
        """Calculate regulatory fees."""
        # Use standard regulatory fee calculator
        return {
            'stt': request.value * Decimal('0.00025'),  # 0.025% for equity delivery
            'exchange_charges': request.value * Decimal('0.0000345'),
            'sebi_charges': request.value * Decimal('0.0000001')
        }
    
    def _calculate_taxes(self, request, commission, regulatory_fees):
        """Calculate taxes."""
        total_charges = commission + sum(regulatory_fees.values())
        gst = total_charges * Decimal('0.18')  # 18% GST
        
        return {
            'gst': gst,
            'stamp_duty': request.value * Decimal('0.000015')  # 0.0015%
        }

# Usage
custom_rates = {
    'equity_delivery': Decimal('15.00'),        # ₹15 per order
    'equity_intraday_rate': Decimal('0.0004'),  # 0.04%
    'equity_intraday_max': Decimal('25.00'),    # Max ₹25
    'option_flat_fee': Decimal('30.00')         # ₹30 per option order
}

custom_calculator = CustomBrokerCalculator('MyBroker', custom_rates)
result = custom_calculator.calculate_cost(request)
```

## Broker Factory

### Broker Factory Pattern

```python
from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory

class BrokerFactory:
    """Factory class for creating broker calculators."""
    
    @staticmethod
    def create_calculator(broker_name: str, **kwargs):
        """Create calculator instance for specified broker."""
        if broker_name.lower() == 'zerodha':
            return ZerodhaCalculator(**kwargs)
        elif broker_name.lower() == 'icici' or broker_name.lower() == 'breeze':
            return BreezeCalculator(**kwargs)
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")
    
    @staticmethod
    def get_supported_brokers():
        """Get list of supported brokers."""
        return ['Zerodha', 'ICICI Securities', 'Breeze']

# Usage
calculator = BrokerFactory.create_calculator('zerodha', exchange='NSE')
result = calculator.calculate_cost(request)
```

### Dynamic Broker Selection

```python
def select_best_broker_dynamically(request, broker_preferences=None):
    """Dynamically select the best broker based on preferences and cost."""
    supported_brokers = BrokerFactory.get_supported_brokers()
    
    if broker_preferences:
        # Filter by preferences
        supported_brokers = [b for b in supported_brokers if b in broker_preferences]
    
    best_broker = None
    min_cost = float('inf')
    
    for broker_name in supported_brokers:
        try:
            calculator = BrokerFactory.create_calculator(broker_name)
            result = calculator.calculate_cost(request)
            
            if result.total_cost < min_cost:
                min_cost = result.total_cost
                best_broker = broker_name
        except Exception as e:
            print(f"Error calculating for {broker_name}: {e}")
    
    return best_broker, min_cost

# Example usage
best_broker, cost = select_best_broker_dynamically(
    request, 
    broker_preferences=['Zerodha', 'ICICI Securities']
)
print(f"Best broker: {best_broker} with cost ₹{cost}")
```

## Advanced Features

### Broker-Specific Optimizations

#### Zerodha Optimizations

```python
class OptimizedZerodhaCalculator(ZerodhaCalculator):
    """Optimized Zerodha calculator with advanced features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_delivery_optimization = True
    
    def suggest_order_optimization(self, requests):
        """Suggest order optimizations for Zerodha."""
        suggestions = []
        
        for request in requests:
            if (request.instrument_type == InstrumentType.EQUITY and 
                request.metadata.get('product_type') != 'CNC'):
                
                # Suggest delivery for large orders to save on brokerage
                if request.value > Decimal('100000'):  # ₹1,00,000
                    suggestions.append({
                        'original_request': request,
                        'suggestion': 'Consider delivery instead of intraday for large orders',
                        'potential_savings': self._calculate_potential_savings(request)
                    })
        
        return suggestions
    
    def _calculate_potential_savings(self, request):
        """Calculate potential savings by switching to delivery."""
        # Calculate intraday cost
        intraday_request = request.copy()
        intraday_request.metadata['product_type'] = 'MIS'
        intraday_cost = self.calculate_cost(intraday_request)
        
        # Calculate delivery cost
        delivery_request = request.copy()
        delivery_request.metadata['product_type'] = 'CNC'
        delivery_cost = self.calculate_cost(delivery_request)
        
        return intraday_cost.total_cost - delivery_cost.total_cost

# Usage
optimizer = OptimizedZerodhaCalculator()
suggestions = optimizer.suggest_order_optimization([large_trade_request])
```

## Error Handling

### Broker-Specific Errors

```python
from src.trading.transaction_costs.exceptions import BrokerConfigurationError

try:
    # Attempt calculation with invalid broker configuration
    result = calculator.calculate_cost(request, invalid_broker_config)
except BrokerConfigurationError as e:
    print(f"Broker configuration error: {e}")
    print("Please check your broker settings")
```

### Fallback Mechanisms

```python
def calculate_with_fallback(request, primary_broker='Zerodha', fallback_broker='ICICI'):
    """Calculate costs with fallback broker."""
    try:
        primary_calc = BrokerFactory.create_calculator(primary_broker)
        return primary_calc.calculate_cost(request)
    except Exception as e:
        print(f"Primary broker {primary_broker} failed: {e}")
        print(f"Falling back to {fallback_broker}")
        
        fallback_calc = BrokerFactory.create_calculator(fallback_broker)
        return fallback_calc.calculate_cost(request)
```

## Performance Considerations

### Broker-Specific Caching

```python
# Enable broker-specific caching for better performance
calculator = ZerodhaCalculator()
calculator.configure_caching(
    cache_key_prefix='zerodha_',
    ttl_seconds=1800  # 30 minutes
)

# Batch processing for multiple transactions
requests = [request1, request2, request3]
results = calculator.calculate_batch(requests, parallel=True)
```

## Related Documentation

- [Cost Calculators Base API](cost_calculators.md) - Base calculator interface
- [Configuration Guide](../configuration/broker_configs.md) - Broker configuration details
- [Examples](../examples/) - Real-world usage examples
- [Market Impact](market_impact.md) - Market impact calculations
- [Troubleshooting](../troubleshooting/common_issues.md) - Common broker issues