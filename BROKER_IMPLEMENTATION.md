# Broker-Specific Fee Structure Implementation

## Overview

This implementation provides comprehensive broker-specific fee calculation modules for Indian brokers with support for all transaction types and regulatory charges as specified in the requirements.

## Implemented Components

### 1. Regulatory Charges Calculators

#### STT Calculator (`src/trading/transaction_costs/brokers/regulatory/stt_calculator.py`)
- Securities Transaction Tax calculation for different instruments
- Rates: Equity (0.1%), Options (0.05%), Futures (0.01%)
- Supports both buy and sell side calculations

#### GST Calculator (`src/trading/transaction_costs/brokers/regulatory/gst_calculator.py`)
- Goods and Services Tax calculation (18% on applicable charges)
- Applied to brokerage and statutory charges
- Excludes STT which is not subject to GST

#### Stamp Duty Calculator (`src/trading/transaction_costs/brokers/regulatory/stamp_duty_calculator.py`)
- 0.003% on buy side transactions
- Maximum cap of ₹1500
- Only applicable to purchase transactions

#### Comprehensive Regulatory Calculator (`src/trading/transaction_costs/brokers/regulatory/charges_calculator.py`)
- Coordinates all regulatory charges
- Includes exchange charges, SEBI charges, clearing charges
- CTT for commodities (0.01% on sell side)

### 2. Broker-Specific Calculators

#### Zerodha Calculator (`src/trading/transaction_costs/brokers/zerodha_calculator.py`)
- **Equity Delivery**: ₹0 (Free)
- **Equity Intraday**: 0.03% or ₹20 (whichever lower)
- **Options**: ₹20 per order
- **Futures**: 0.03% or ₹20 (whichever lower)
- **Currency**: 0.03% or ₹20 (whichever lower)
- **Commodity**: 0.03% or ₹20 (whichever lower)

#### ICICI Securities Calculator (`src/trading/transaction_costs/brokers/breeze_calculator.py`)
- **Equity Delivery**: ₹20 per order
- **Equity Intraday**: 0.05% or ₹20 (whichever lower)
- **Options**: ₹20 per order
- **Futures**: 0.05% or ₹20 (whichever lower)
- **Currency**: 0.05% or ₹20 (whichever lower)
- **Commodity**: 0.05% or ₹20 (whichever lower)

#### Broker Factory (`src/trading/transaction_costs/brokers/broker_factory.py`)
- Factory pattern for calculator instantiation
- Configuration-based broker selection
- Error handling for unsupported brokers
- Supports aliases (kite=zerodha, breeze=icici)

## Usage Examples

### Basic Usage

```python
from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory
from src.trading.transaction_costs.models import TransactionRequest, BrokerConfiguration
from decimal import Decimal

# Create a transaction request
request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY,
    metadata={'position_type': 'delivery'}
)

# Create broker configuration
broker_config = BrokerConfiguration(
    broker_name='Zerodha',
    base_currency='INR'
)

# Create calculator and calculate costs
calculator = BrokerFactory.create_calculator('zerodha')
result = calculator.calculate_cost(request, broker_config)

print(f"Total Cost: ₹{result.total_cost:.2f}")
print(f"Commission: ₹{result.commission:.2f}")
print(f"Regulatory Fees: ₹{result.regulatory_fees:.2f}")
```

### Comparing Brokers

```python
# Compare costs between brokers
zerodha_calc = BrokerFactory.create_calculator('zerodha')
icici_calc = BrokerFactory.create_calculator('icici')

zerodha_cost = zerodha_calc.calculate_cost(request, broker_config)
icici_cost = icici_calc.calculate_cost(request, broker_config)

print(f"Zerodha Total: ₹{zerodha_cost.total_cost:.2f}")
print(f"ICICI Total: ₹{icici_cost.total_cost:.2f}")
```

## File Structure

```
src/trading/transaction_costs/brokers/
├── __init__.py
├── breeze_calculator.py          # ICICI Securities calculator
├── zerodha_calculator.py         # Zerodha calculator
├── broker_factory.py             # Factory for calculator creation
└── regulatory/
    ├── __init__.py
    ├── charges_calculator.py     # Comprehensive regulatory calculator
    ├── stt_calculator.py         # STT calculator
    ├── gst_calculator.py         # GST calculator
    └── stamp_duty_calculator.py  # Stamp duty calculator
```

## Testing

Comprehensive test suite with 22 new tests covering:
- Individual broker calculators
- Regulatory charge calculations
- Broker factory functionality
- Integrated cost calculations
- Edge cases and error handling

Run tests with:
```bash
python -m pytest src/trading/tests/test_broker_calculators.py -v
```

## Integration

- Extends the existing `CostCalculatorBase` abstract class
- Uses existing data models (`TransactionRequest`, `BrokerConfiguration`, etc.)
- Integrates with existing error handling and logging framework
- Thread-safe implementations with caching support
- Supports async operations and batch calculations

## Key Features

1. **Accurate Fee Calculations**: Matches official broker calculators
2. **Comprehensive Regulatory Charges**: All Indian market regulatory fees
3. **Multiple Instrument Support**: Equity, options, futures, currency, commodity
4. **Position Type Awareness**: Delivery vs intraday calculations
5. **Error Handling**: Robust validation and error reporting
6. **Performance Optimized**: Caching and efficient calculations
7. **Extensible Design**: Easy to add new brokers

## Demo

A working demonstration script is available at `demo_broker_calculators.py` which shows:
- Cost calculations for different brokers
- Delivery vs intraday comparisons
- Different instrument types
- Broker factory features

Run the demo with:
```bash
python demo_broker_calculators.py
```

## Compliance

The implementation follows all requirements from the problem statement:
- ✅ Implements `CostCalculatorBase` abstract interface
- ✅ Comprehensive input validation
- ✅ Efficient calculation algorithms
- ✅ Thread-safe implementations
- ✅ Proper error handling
- ✅ Caching for frequently calculated fees
- ✅ Precise calculations to avoid rounding errors
- ✅ Factory pattern for broker instantiation
- ✅ Configuration-based broker selection
- ✅ Support for all transaction types and instruments
- ✅ Accurate regulatory charge calculations