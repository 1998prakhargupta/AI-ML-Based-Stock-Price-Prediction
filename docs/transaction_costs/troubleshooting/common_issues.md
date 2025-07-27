# Common Issues and Solutions

This guide covers the most frequently encountered issues when using the Transaction Cost Modeling System and their solutions.

## Quick Troubleshooting Checklist

Before diving into specific issues, run through this quick checklist:

- [ ] Python version 3.8+ is installed
- [ ] All required dependencies are installed (`pip install -r requirements.txt`)
- [ ] PYTHONPATH is set correctly
- [ ] Configuration files are in the correct locations
- [ ] Environment variables are set (if used)
- [ ] Log files are accessible and not full

## Installation and Setup Issues

### Issue 1: Import Errors

#### Problem
```
ModuleNotFoundError: No module named 'src.trading.transaction_costs'
```

#### Causes and Solutions

**Cause 1: PYTHONPATH not set**
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Make permanent by adding to shell profile
echo 'export PYTHONPATH="${PWD}/src:${PYTHONPATH}"' >> ~/.bashrc
source ~/.bashrc
```

**Cause 2: Running from wrong directory**
```bash
# Solution: Navigate to project root
cd /path/to/AI-ML-Based-Stock-Price-Prediction

# Verify you're in the right directory
ls src/trading/transaction_costs/
```

**Cause 3: Virtual environment not activated**
```bash
# Solution: Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### Issue 2: Dependency Installation Failures

#### Problem
```
ERROR: Could not install packages due to an EnvironmentError
```

#### Solutions

**For Permission Errors:**
```bash
# Install with user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**For Package Conflicts:**
```bash
# Create fresh virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**For Network Issues:**
```bash
# Use alternative index
pip install -r requirements.txt -i https://pypi.org/simple/

# Or install offline if packages are cached
pip install --no-index --find-links file:///local/cache -r requirements.txt
```

### Issue 3: Configuration File Errors

#### Problem
```
FileNotFoundError: Configuration file not found
```

#### Solution
```python
# Create default configuration
from src.trading.cost_config.base_config import CostConfiguration

config = CostConfiguration()
config.create_default_broker_configurations()
config.save_config()

print("✅ Default configuration created")
```

## Calculation Issues

### Issue 4: Incorrect Cost Calculations

#### Problem
Cost calculations seem wrong or inconsistent.

#### Debugging Steps

**Step 1: Verify Input Data**
```python
from decimal import Decimal

# Always use Decimal for monetary values
request = TransactionRequest(
    symbol='RELIANCE',
    quantity=100,
    price=Decimal('2500.00'),  # Use Decimal, not float
    transaction_type=TransactionType.BUY,
    instrument_type=InstrumentType.EQUITY
)

# Verify the request
print(f"Value: ₹{request.quantity * request.price}")
print(f"Type: {type(request.price)}")  # Should be <class 'decimal.Decimal'>
```

**Step 2: Enable Debug Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

calculator = ZerodhaCalculator()
result = calculator.calculate_cost(request)
```

**Step 3: Compare with Manual Calculation**
```python
def manual_zerodha_calculation(request):
    """Manual calculation for verification."""
    value = request.quantity * request.price
    
    # Zerodha equity delivery brokerage is ₹0
    if request.metadata.get('product_type') == 'CNC':
        brokerage = Decimal('0.00')
    else:
        # Intraday: 0.03% or ₹20, whichever is lower
        brokerage = min(value * Decimal('0.0003'), Decimal('20.00'))
    
    # STT: 0.025% for equity delivery
    stt = value * Decimal('0.00025')
    
    # Exchange charges: 0.00345% for NSE
    exchange_charges = value * Decimal('0.0000345')
    
    # SEBI charges: 0.00001%
    sebi_charges = value * Decimal('0.0000001')
    
    # GST: 18% on (brokerage + exchange charges + SEBI charges)
    taxable_amount = brokerage + exchange_charges + sebi_charges
    gst = taxable_amount * Decimal('0.18')
    
    # Stamp duty: 0.015% on buy transactions
    stamp_duty = value * Decimal('0.000015') if request.transaction_type == TransactionType.BUY else Decimal('0')
    
    total_cost = brokerage + stt + exchange_charges + sebi_charges + gst + stamp_duty
    
    return {
        'brokerage': brokerage,
        'stt': stt,
        'exchange_charges': exchange_charges,
        'sebi_charges': sebi_charges,
        'gst': gst,
        'stamp_duty': stamp_duty,
        'total_cost': total_cost
    }

# Compare results
manual_result = manual_zerodha_calculation(request)
system_result = calculator.calculate_cost(request)

print("Manual vs System Calculation:")
for key in manual_result:
    manual_val = manual_result[key]
    system_val = getattr(system_result, key)
    diff = abs(manual_val - system_val)
    print(f"{key}: Manual=₹{manual_val:.2f}, System=₹{system_val:.2f}, Diff=₹{diff:.2f}")
```

### Issue 5: Performance Problems

#### Problem
Calculations are slow or system becomes unresponsive.

#### Solutions

**Enable Caching:**
```python
calculator = ZerodhaCalculator(enable_caching=True)
calculator.configure_caching(ttl_seconds=3600)

# For repeated calculations with same parameters
result1 = calculator.calculate_cost(request)  # Calculated
result2 = calculator.calculate_cost(request)  # Retrieved from cache
```

**Use Batch Processing:**
```python
# Inefficient: Individual calculations
results = []
for request in requests:
    result = calculator.calculate_cost(request)
    results.append(result)

# Efficient: Batch processing
results = calculator.calculate_batch(requests, parallel=True)
```

**Enable Parallel Processing:**
```python
calculator = ZerodhaCalculator()
calculator.configure_batch_processing(
    max_batch_size=100,
    parallel_workers=4
)
```

**Monitor Performance:**
```python
# Check performance metrics
metrics = calculator.get_performance_metrics()
print(f"Average calculation time: {metrics['average_calculation_time']:.3f}s")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")

# Reset if needed
calculator.reset_performance_metrics()
```

### Issue 6: Memory Issues

#### Problem
```
MemoryError: Unable to allocate memory
```

#### Solutions

**Reduce Batch Size:**
```python
# Instead of processing 10,000 transactions at once
large_batch = list(range(10000))

# Process in smaller chunks
chunk_size = 100
for i in range(0, len(large_batch), chunk_size):
    chunk = large_batch[i:i + chunk_size]
    results = calculator.calculate_batch(chunk)
    # Process results immediately
```

**Clear Cache Periodically:**
```python
# Clear cache after processing large batches
calculator.clear_cache()

# Or configure smaller cache size
calculator.configure_caching(max_cache_size=1000)
```

**Use Generators for Large Datasets:**
```python
def process_large_dataset(file_path):
    """Process large dataset using generator."""
    def request_generator():
        with open(file_path, 'r') as f:
            for line in f:
                # Parse line and create TransactionRequest
                yield create_request_from_line(line)
    
    for batch in batch_generator(request_generator(), batch_size=100):
        results = calculator.calculate_batch(batch)
        # Process results immediately
        yield results
```

## Broker Integration Issues

### Issue 7: Broker API Errors

#### Problem
```
BrokerConfigurationError: Invalid API credentials
```

#### Solutions

**Verify Credentials:**
```python
# Check environment variables
import os
api_key = os.getenv('ZERODHA_API_KEY')
if not api_key:
    print("❌ ZERODHA_API_KEY not set")
else:
    print(f"✅ API key found: {api_key[:10]}...")

# Test credentials
try:
    calculator = ZerodhaCalculator()
    # Perform a simple calculation to test
    test_request = create_test_request()
    result = calculator.calculate_cost(test_request)
    print("✅ Broker integration working")
except Exception as e:
    print(f"❌ Broker integration failed: {e}")
```

**Update Broker Configuration:**
```python
from src.trading.cost_config.base_config import CostConfiguration

config = CostConfiguration()

# Update Zerodha configuration
config.update_broker_configuration('Zerodha', {
    'api_key': 'your_api_key',
    'api_secret': 'your_api_secret',
    'exchange': 'NSE'
})

config.save_config()
```

### Issue 8: Unsupported Instrument Types

#### Problem
```
UnsupportedInstrumentError: Currency trading not supported by this broker
```

#### Solutions

**Check Supported Instruments:**
```python
calculator = ZerodhaCalculator()
supported = calculator.supported_instruments
print(f"Supported instruments: {[inst.name for inst in supported]}")

# Check if specific instrument is supported
if InstrumentType.CURRENCY in calculator.supported_instruments:
    print("✅ Currency trading supported")
else:
    print("❌ Currency trading not supported")
```

**Use Appropriate Broker:**
```python
# Find broker that supports the instrument
from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory

def find_supporting_broker(instrument_type):
    """Find brokers that support the instrument type."""
    brokers = ['Zerodha', 'ICICI_Securities']
    supporting_brokers = []
    
    for broker_name in brokers:
        try:
            calculator = BrokerFactory.create_calculator(broker_name)
            if instrument_type in calculator.supported_instruments:
                supporting_brokers.append(broker_name)
        except Exception:
            pass
    
    return supporting_brokers

# Example usage
currency_brokers = find_supporting_broker(InstrumentType.CURRENCY)
print(f"Brokers supporting currency: {currency_brokers}")
```

## Data Validation Issues

### Issue 9: Invalid Transaction Data

#### Problem
```
InvalidTransactionError: Quantity must be positive
```

#### Solutions

**Validate Input Data:**
```python
def validate_transaction_request(request):
    """Validate transaction request before processing."""
    errors = []
    
    if request.quantity <= 0:
        errors.append("Quantity must be positive")
    
    if request.price <= 0:
        errors.append("Price must be positive")
    
    if not request.symbol or not request.symbol.strip():
        errors.append("Symbol cannot be empty")
    
    if request.symbol.lower() in ['test', 'demo', 'invalid']:
        errors.append("Invalid symbol")
    
    if errors:
        raise ValueError(f"Validation errors: {', '.join(errors)}")
    
    return True

# Use before calculation
try:
    validate_transaction_request(request)
    result = calculator.calculate_cost(request)
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

**Sanitize Input Data:**
```python
def sanitize_request(request):
    """Sanitize and normalize request data."""
    # Normalize symbol
    request.symbol = request.symbol.strip().upper()
    
    # Ensure Decimal types
    if not isinstance(request.price, Decimal):
        request.price = Decimal(str(request.price))
    
    # Validate ranges
    if request.quantity > 100000:
        print(f"⚠️  Large quantity: {request.quantity}")
    
    if request.price > Decimal('50000'):
        print(f"⚠️  High price: {request.price}")
    
    return request

# Sanitize before processing
request = sanitize_request(request)
result = calculator.calculate_cost(request)
```

### Issue 10: Market Timing Issues

#### Problem
```
MarketDataError: Transaction outside market hours
```

#### Solutions

**Disable Market Timing Validation:**
```python
# Temporarily disable validation
calculator = ZerodhaCalculator()
calculator.config['enable_market_timing_validation'] = False

result = calculator.calculate_cost(request)
```

**Adjust Market Hours:**
```python
from src.trading.cost_config.base_config import CostConfiguration

config = CostConfiguration()
config.update_setting('market_data.default_market_hours', {
    'start': '09:00',  # Earlier start
    'end': '16:00',    # Later end
    'timezone': 'Asia/Kolkata'
})
```

**Use Historical Mode:**
```python
# For historical calculations
result = calculator.calculate_cost(
    request,
    calculation_mode=CalculationMode.HISTORICAL
)
```

## Configuration Issues

### Issue 11: Configuration Not Loading

#### Problem
Configuration changes don't take effect.

#### Solutions

**Force Configuration Reload:**
```python
config = CostConfiguration()
config.reload_config()

# Or create new instance
config = CostConfiguration(force_reload=True)
```

**Check Configuration File Location:**
```python
config = CostConfiguration()
print(f"Config file location: {config.config_file_path}")
print(f"Config exists: {os.path.exists(config.config_file_path)}")

# Load from specific location
config = CostConfiguration(config_file='path/to/your/config.json')
```

**Validate Configuration:**
```python
from src.trading.cost_config.config_validator import CostConfigurationValidator

validator = CostConfigurationValidator()
result = validator.validate_full_configuration(config.get_all_settings())

if not result.is_valid:
    print("❌ Configuration errors:")
    for error in result.errors:
        print(f"  - {error}")
else:
    print("✅ Configuration is valid")
```

## Logging and Debugging Issues

### Issue 12: No Log Output

#### Problem
No log messages are appearing.

#### Solutions

**Check Log Level:**
```python
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logger = logging.getLogger('src.trading.transaction_costs')
logger.setLevel(logging.DEBUG)
```

**Check Log File Permissions:**
```bash
# Ensure log directory exists and is writable
mkdir -p logs
chmod 755 logs

# Check log file permissions
ls -la logs/transaction_costs.log
```

**Enable Console Logging:**
```python
config = CostConfiguration()
config.update_setting('logging.enable_console_logging', True)
config.save_config()
```

### Issue 13: Too Much Log Output

#### Problem
Logs are too verbose and filling up disk space.

#### Solutions

**Reduce Log Level:**
```python
config = CostConfiguration()
config.update_setting('logging.level', 'WARNING')  # Only warnings and errors
config.save_config()
```

**Configure Log Rotation:**
```python
config.update_settings({
    'logging.max_file_size': '10MB',
    'logging.backup_count': 3
})
```

**Disable Verbose Components:**
```python
# Disable performance logging
config.update_setting('performance.enable_metrics_collection', False)

# Disable profiling
config.update_setting('performance.enable_profiling', False)
```

## Network and Connectivity Issues

### Issue 14: Redis Connection Issues

#### Problem
```
ConnectionError: Could not connect to Redis server
```

#### Solutions

**Check Redis Status:**
```bash
# Check if Redis is running
redis-cli ping

# Start Redis if not running
sudo systemctl start redis-server

# Check Redis logs
sudo journalctl -u redis-server
```

**Fallback to Memory Cache:**
```python
try:
    config = CostConfiguration()
    config.update_setting('caching.backend', 'redis')
    
    # Test Redis connection
    calculator = ZerodhaCalculator()
    calculator.configure_caching(backend='redis')
    
except Exception as e:
    print(f"Redis unavailable, using memory cache: {e}")
    config.update_setting('caching.backend', 'memory')
```

**Configure Redis Timeout:**
```python
config.update_setting('caching.redis_config', {
    'host': 'localhost',
    'port': 6379,
    'socket_timeout': 10,
    'socket_connect_timeout': 10,
    'retry_on_timeout': True
})
```

## Error Recovery and Fallbacks

### Issue 15: System Recovery After Errors

#### Problem
System becomes unstable after errors.

#### Solutions

**Implement Error Recovery:**
```python
def robust_calculation(request, max_retries=3):
    """Calculate with automatic error recovery."""
    calculator = ZerodhaCalculator()
    
    for attempt in range(max_retries):
        try:
            result = calculator.calculate_cost(request)
            return result
            
        except TransactionCostError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Reset calculator state
                calculator.reset_performance_metrics()
                calculator.clear_cache()
                
                # Wait before retry
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    raise RuntimeError("All retry attempts failed")

# Usage
try:
    result = robust_calculation(request)
except Exception as e:
    print(f"❌ Calculation failed after retries: {e}")
```

**Implement Graceful Degradation:**
```python
def calculate_with_fallback(request):
    """Calculate with fallback to simpler methods."""
    try:
        # Try primary calculator
        calculator = ZerodhaCalculator()
        return calculator.calculate_cost(request)
        
    except Exception as e:
        print(f"Primary calculation failed: {e}")
        
        try:
            # Try alternative broker
            calculator = BreezeCalculator()
            return calculator.calculate_cost(request)
            
        except Exception as e2:
            print(f"Alternative calculation failed: {e2}")
            
            # Fallback to basic estimation
            return estimate_basic_cost(request)

def estimate_basic_cost(request):
    """Basic cost estimation as last resort."""
    value = request.quantity * request.price
    
    # Simple estimation: 0.1% of trade value
    estimated_cost = value * Decimal('0.001')
    
    return TransactionCostBreakdown(
        brokerage=estimated_cost * Decimal('0.5'),
        stt=estimated_cost * Decimal('0.3'),
        total_cost=estimated_cost,
        # ... other fields with estimates
    )
```

## Performance Optimization

### Issue 16: Slow Batch Processing

#### Problem
Batch calculations are slower than expected.

#### Solutions

**Optimize Batch Size:**
```python
import time

def find_optimal_batch_size(requests, max_batch_size=1000):
    """Find optimal batch size for your system."""
    calculator = ZerodhaCalculator()
    
    batch_sizes = [10, 50, 100, 500, 1000]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(requests):
            continue
            
        test_batch = requests[:batch_size]
        
        start_time = time.time()
        calculator.calculate_batch(test_batch)
        end_time = time.time()
        
        time_per_calculation = (end_time - start_time) / batch_size
        results[batch_size] = time_per_calculation
        
        print(f"Batch size {batch_size}: {time_per_calculation:.4f}s per calculation")
    
    optimal_size = min(results.items(), key=lambda x: x[1])[0]
    print(f"\nOptimal batch size: {optimal_size}")
    return optimal_size

# Find and use optimal batch size
optimal_size = find_optimal_batch_size(your_requests)
calculator.configure_batch_processing(max_batch_size=optimal_size)
```

## Getting Help

### When to Seek Additional Help

If you've tried the solutions above and still have issues:

1. **Check the GitHub Issues**: Search existing issues for similar problems
2. **Create a Detailed Bug Report**: Include error messages, configuration, and steps to reproduce
3. **Provide System Information**: Python version, OS, dependency versions
4. **Include Minimal Reproduction Case**: Simple code that demonstrates the problem

### Bug Report Template

```markdown
## Issue Description
Brief description of the problem

## Environment
- Python version: 3.x.x
- OS: Ubuntu 20.04 / Windows 10 / macOS 12.x
- Package version: x.x.x

## Configuration
```json
{
  "relevant": "configuration options"
}
```

## Reproduction Steps
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Messages
```
Full error traceback
```

## Additional Context
Any other relevant information
```

### Community Resources

- **GitHub Issues**: [Project Issues](https://github.com/1998prakhargupta/AI-ML-Based-Stock-Price-Prediction/issues)
- **Documentation**: [Full Documentation](../README.md)
- **Examples**: [Code Examples](../examples/)
- **Configuration**: [Configuration Guide](../configuration/configuration_reference.md)

---

*Remember: Most issues can be resolved by carefully checking configuration, validating input data, and following the debugging steps outlined above. When in doubt, enable debug logging to get more information about what's happening.*