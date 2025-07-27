# Cost Calculators API Reference

This section provides comprehensive documentation for all cost calculator classes and interfaces.

## Overview

The cost calculator system is built around an abstract base class `CostCalculatorBase` that defines the standard interface for all cost calculations. Specific broker implementations extend this base class to provide accurate cost calculations for their respective platforms.

## Base Calculator

### CostCalculatorBase

**Module**: `src.trading.transaction_costs.base_cost_calculator`

Abstract base class that provides the foundation for all cost calculators.

#### Constructor

```python
CostCalculatorBase(
    calculator_name: str,
    version: str = "1.0.0",
    supported_instruments: Optional[List[InstrumentType]] = None,
    supported_modes: Optional[List[str]] = None,
    default_timeout: int = None,
    enable_caching: bool = True
)
```

**Parameters:**
- `calculator_name` (str): Unique identifier for the calculator
- `version` (str): Version of the calculator implementation
- `supported_instruments` (List[InstrumentType], optional): List of supported instrument types
- `supported_modes` (List[str], optional): Supported calculation modes
- `default_timeout` (int, optional): Default timeout for calculations in seconds
- `enable_caching` (bool): Whether to enable result caching

#### Core Methods

##### calculate_cost()

```python
def calculate_cost(
    self,
    request: TransactionRequest,
    broker_config: Optional[BrokerConfiguration] = None,
    market_conditions: Optional[MarketConditions] = None,
    calculation_mode: str = CalculationMode.REAL_TIME
) -> TransactionCostBreakdown
```

Calculate transaction costs synchronously.

**Parameters:**
- `request` (TransactionRequest): Transaction details
- `broker_config` (BrokerConfiguration, optional): Broker-specific configuration
- `market_conditions` (MarketConditions, optional): Current market conditions
- `calculation_mode` (str): Calculation mode (real_time, batch, historical, simulation)

**Returns:** `TransactionCostBreakdown` - Detailed cost breakdown

**Raises:**
- `InvalidTransactionError`: Invalid transaction parameters
- `BrokerConfigurationError`: Invalid broker configuration
- `CalculationError`: Calculation failure
- `UnsupportedInstrumentError`: Unsupported instrument type

**Example:**
```python
from src.trading.transaction_costs.models import TransactionRequest, TransactionType, InstrumentType
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
from decimal import Decimal

# Create transaction request
request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)

# Calculate costs
calculator = ZerodhaCalculator()
result = calculator.calculate_cost(request)

print(f"Total cost: ₹{result.total_cost}")
print(f"Breakdown: {result.cost_breakdown}")
```

##### calculate_cost_async()

```python
async def calculate_cost_async(
    self,
    request: TransactionRequest,
    broker_config: Optional[BrokerConfiguration] = None,
    market_conditions: Optional[MarketConditions] = None,
    calculation_mode: str = CalculationMode.REAL_TIME
) -> TransactionCostBreakdown
```

Calculate transaction costs asynchronously.

**Parameters:** Same as `calculate_cost()`

**Returns:** `TransactionCostBreakdown` - Detailed cost breakdown

**Example:**
```python
import asyncio

async def calculate_multiple_costs():
    calculator = ZerodhaCalculator()
    
    requests = [request1, request2, request3]
    tasks = [calculator.calculate_cost_async(req) for req in requests]
    
    results = await asyncio.gather(*tasks)
    return results

# Usage
results = asyncio.run(calculate_multiple_costs())
```

##### calculate_batch()

```python
def calculate_batch(
    self,
    requests: List[TransactionRequest],
    broker_config: Optional[BrokerConfiguration] = None,
    market_conditions: Optional[MarketConditions] = None,
    parallel: bool = True
) -> List[TransactionCostBreakdown]
```

Calculate costs for multiple transactions in batch.

**Parameters:**
- `requests` (List[TransactionRequest]): List of transaction requests
- `broker_config` (BrokerConfiguration, optional): Broker configuration
- `market_conditions` (MarketConditions, optional): Market conditions
- `parallel` (bool): Whether to process requests in parallel

**Returns:** `List[TransactionCostBreakdown]` - List of cost breakdowns

**Example:**
```python
requests = [
    TransactionRequest(symbol='RELIANCE', quantity=100, price=Decimal('2500'), 
                      transaction_type=TransactionType.BUY, instrument_type=InstrumentType.EQUITY),
    TransactionRequest(symbol='TCS', quantity=50, price=Decimal('3200'), 
                      transaction_type=TransactionType.SELL, instrument_type=InstrumentType.EQUITY)
]

calculator = ZerodhaCalculator()
results = calculator.calculate_batch(requests)

total_cost = sum(result.total_cost for result in results)
print(f"Total portfolio cost: ₹{total_cost}")
```

#### Abstract Methods

Subclasses must implement these methods:

##### _calculate_commission()

```python
@abstractmethod
def _calculate_commission(
    self,
    request: TransactionRequest,
    broker_config: BrokerConfiguration
) -> Decimal
```

Calculate brokerage commission for the transaction.

##### _calculate_regulatory_fees()

```python
@abstractmethod
def _calculate_regulatory_fees(
    self,
    request: TransactionRequest,
    broker_config: BrokerConfiguration
) -> Dict[str, Decimal]
```

Calculate regulatory fees (STT, exchange charges, etc.).

##### _calculate_taxes()

```python
@abstractmethod
def _calculate_taxes(
    self,
    request: TransactionRequest,
    commission: Decimal,
    regulatory_fees: Dict[str, Decimal]
) -> Dict[str, Decimal]
```

Calculate applicable taxes (GST, stamp duty, etc.).

#### Utility Methods

##### validate_request()

```python
def validate_request(
    self,
    request: TransactionRequest
) -> bool
```

Validate transaction request parameters.

**Returns:** `bool` - True if valid, raises exception if invalid

##### is_instrument_supported()

```python
def is_instrument_supported(
    self,
    instrument_type: InstrumentType
) -> bool
```

Check if the instrument type is supported.

##### get_cached_result()

```python
def get_cached_result(
    self,
    cache_key: str
) -> Optional[TransactionCostBreakdown]
```

Retrieve cached calculation result.

##### cache_result()

```python
def cache_result(
    self,
    cache_key: str,
    result: TransactionCostBreakdown,
    ttl: Optional[int] = None
) -> None
```

Cache calculation result.

#### Performance Methods

##### get_performance_metrics()

```python
def get_performance_metrics(self) -> Dict[str, Any]
```

Get performance metrics for the calculator.

**Returns:**
```python
{
    'total_calculations': 1500,
    'average_calculation_time': 0.025,  # seconds
    'cache_hit_rate': 0.65,
    'error_rate': 0.02,
    'supported_instruments': ['EQUITY', 'OPTION'],
    'last_calculation_time': '2024-01-15T10:30:00Z'
}
```

##### reset_performance_metrics()

```python
def reset_performance_metrics(self) -> None
```

Reset all performance counters.

## Calculator Configuration

### Configuration Properties

Each calculator can be configured with:

```python
calculator.config = {
    'precision_decimal_places': 2,
    'enable_caching': True,
    'cache_ttl_seconds': 3600,
    'parallel_batch_size': 10,
    'timeout_seconds': 30,
    'enable_performance_tracking': True
}
```

### Performance Tuning

```python
# Enable high-performance mode
calculator.enable_high_performance_mode()

# Configure batch processing
calculator.configure_batch_processing(
    max_batch_size=100,
    parallel_workers=4,
    timeout_per_batch=60
)

# Configure caching
calculator.configure_caching(
    cache_type='redis',  # or 'memory'
    max_cache_size=10000,
    ttl_seconds=3600
)
```

## Error Handling

All calculator methods can raise these exceptions:

### TransactionCostError
Base exception for all transaction cost related errors.

### InvalidTransactionError
Raised when transaction parameters are invalid.

```python
try:
    result = calculator.calculate_cost(invalid_request)
except InvalidTransactionError as e:
    print(f"Invalid transaction: {e}")
    print(f"Error context: {e.context}")
```

### BrokerConfigurationError
Raised when broker configuration is invalid or missing.

### CalculationError
Raised when cost calculation fails.

### UnsupportedInstrumentError
Raised when trying to calculate costs for unsupported instruments.

### RateLimitError
Raised when API rate limits are exceeded.

## Best Practices

### 1. Always Handle Exceptions

```python
from src.trading.transaction_costs.exceptions import TransactionCostError

try:
    result = calculator.calculate_cost(request)
except TransactionCostError as e:
    logger.error(f"Cost calculation failed: {e}")
    # Implement fallback logic
```

### 2. Use Batch Processing for Multiple Calculations

```python
# Efficient batch processing
results = calculator.calculate_batch(requests, parallel=True)

# Instead of individual calls
# results = [calculator.calculate_cost(req) for req in requests]  # Inefficient
```

### 3. Configure Caching for Better Performance

```python
calculator = ZerodhaCalculator(enable_caching=True)
calculator.configure_caching(ttl_seconds=1800)  # 30 minutes
```

### 4. Monitor Performance

```python
# Get performance metrics
metrics = calculator.get_performance_metrics()
print(f"Average calculation time: {metrics['average_calculation_time']:.3f}s")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

### 5. Use Async for High-Throughput Applications

```python
async def process_trades(trades):
    calculator = ZerodhaCalculator()
    tasks = [calculator.calculate_cost_async(trade) for trade in trades]
    return await asyncio.gather(*tasks)
```

## Integration Examples

### With Portfolio Management

```python
class PortfolioManager:
    def __init__(self, calculator):
        self.calculator = calculator
    
    def calculate_rebalancing_cost(self, current_positions, target_positions):
        transactions = self.generate_rebalancing_transactions(
            current_positions, target_positions
        )
        
        cost_results = self.calculator.calculate_batch(transactions)
        total_cost = sum(result.total_cost for result in cost_results)
        
        return total_cost, cost_results
```

### With Risk Management

```python
class RiskManager:
    def __init__(self, calculator, max_cost_percent=0.1):
        self.calculator = calculator
        self.max_cost_percent = max_cost_percent
    
    def validate_trade(self, trade_request):
        cost_result = self.calculator.calculate_cost(trade_request)
        cost_percent = cost_result.total_cost / trade_request.value
        
        if cost_percent > self.max_cost_percent:
            raise ValueError(f"Trade cost {cost_percent:.2%} exceeds limit {self.max_cost_percent:.2%}")
        
        return cost_result
```

## Related Documentation

- [Broker-Specific Calculators](brokers.md) - Implementation details for specific brokers
- [Market Impact Models](market_impact.md) - Market impact calculation methods
- [Spread Estimation](spreads.md) - Bid-ask spread modeling
- [Data Models](../user_guide/basic_usage.md#data-models) - Core data structures
- [Configuration Reference](../configuration/configuration_reference.md) - Configuration options