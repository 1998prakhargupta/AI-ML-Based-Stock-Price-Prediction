# Configuration Reference

This document provides a complete reference for all configuration options in the Transaction Cost Modeling System.

## Overview

The system uses a hierarchical configuration structure with the following sources (in order of precedence):

1. **Runtime Parameters** - Passed directly to methods
2. **Environment Variables** - Set in shell or `.env` file
3. **Configuration Files** - JSON/YAML configuration files
4. **Default Values** - Built-in system defaults

## Configuration File Structure

### Main Configuration File

**Location**: `configs/cost_config/cost_configuration.json`

```json
{
  "version": "1.0.0",
  "system": {
    "precision_decimal_places": 2,
    "default_currency": "INR",
    "default_exchange": "NSE",
    "enable_validation": true,
    "strict_mode": false
  },
  "calculation": {
    "timeout_seconds": 30,
    "enable_parallel_processing": true,
    "max_batch_size": 1000,
    "default_calculation_mode": "real_time",
    "enable_caching": true
  },
  "caching": {
    "backend": "memory",
    "ttl_seconds": 3600,
    "max_cache_size": 10000,
    "cache_key_prefix": "tc_",
    "enable_compression": false
  },
  "performance": {
    "enable_metrics_collection": true,
    "metrics_retention_days": 30,
    "enable_profiling": false,
    "profile_sample_rate": 0.1
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "logs/transaction_costs.log",
    "max_file_size": "10MB",
    "backup_count": 5,
    "enable_console_logging": true
  },
  "brokers": {
    "default_broker": "Zerodha",
    "broker_configs": {},
    "enable_broker_discovery": true
  },
  "market_data": {
    "default_market_hours": {
      "start": "09:15",
      "end": "15:30",
      "timezone": "Asia/Kolkata"
    },
    "enable_market_timing_validation": true,
    "extended_hours_support": false
  },
  "regulatory": {
    "enable_regulatory_compliance": true,
    "stt_rates": {
      "equity_delivery": 0.00025,
      "equity_intraday": 0.00025,
      "options": 0.00525,
      "futures": 0.0002
    },
    "exchange_charges": {
      "NSE": 0.0000345,
      "BSE": 0.0000375
    },
    "sebi_charges": 0.0000001,
    "gst_rate": 0.18
  }
}
```

## Configuration Sections

### System Configuration

Controls core system behavior and defaults.

#### `system.precision_decimal_places`
- **Type**: Integer
- **Default**: `2`
- **Range**: 0-10
- **Description**: Number of decimal places for monetary calculations
- **Example**: 
  ```json
  "precision_decimal_places": 4
  ```

#### `system.default_currency`
- **Type**: String
- **Default**: `"INR"`
- **Options**: `"INR"`, `"USD"`, `"EUR"`, `"GBP"`
- **Description**: Default currency for calculations
- **Example**:
  ```json
  "default_currency": "USD"
  ```

#### `system.default_exchange`
- **Type**: String
- **Default**: `"NSE"`
- **Options**: `"NSE"`, `"BSE"`, `"NASDAQ"`, `"NYSE"`
- **Description**: Default exchange for regulatory calculations
- **Example**:
  ```json
  "default_exchange": "BSE"
  ```

#### `system.enable_validation`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable input validation for all calculations
- **Example**:
  ```json
  "enable_validation": false
  ```

#### `system.strict_mode`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable strict validation mode (raises errors for warnings)
- **Example**:
  ```json
  "strict_mode": true
  ```

### Calculation Configuration

Controls how calculations are performed.

#### `calculation.timeout_seconds`
- **Type**: Integer
- **Default**: `30`
- **Range**: 1-300
- **Description**: Maximum time allowed for a single calculation
- **Example**:
  ```json
  "timeout_seconds": 60
  ```

#### `calculation.enable_parallel_processing`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable parallel processing for batch calculations
- **Example**:
  ```json
  "enable_parallel_processing": false
  ```

#### `calculation.max_batch_size`
- **Type**: Integer
- **Default**: `1000`
- **Range**: 1-10000
- **Description**: Maximum number of transactions in a batch
- **Example**:
  ```json
  "max_batch_size": 500
  ```

#### `calculation.default_calculation_mode`
- **Type**: String
- **Default**: `"real_time"`
- **Options**: `"real_time"`, `"batch"`, `"historical"`, `"simulation"`
- **Description**: Default calculation mode for operations
- **Example**:
  ```json
  "default_calculation_mode": "simulation"
  ```

#### `calculation.enable_caching`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable result caching for improved performance
- **Example**:
  ```json
  "enable_caching": false
  ```

### Caching Configuration

Controls caching behavior for performance optimization.

#### `caching.backend`
- **Type**: String
- **Default**: `"memory"`
- **Options**: `"memory"`, `"redis"`, `"file"`, `"disabled"`
- **Description**: Caching backend to use
- **Example**:
  ```json
  "backend": "redis"
  ```

#### `caching.ttl_seconds`
- **Type**: Integer
- **Default**: `3600`
- **Range**: 60-86400
- **Description**: Time-to-live for cached results in seconds
- **Example**:
  ```json
  "ttl_seconds": 1800
  ```

#### `caching.max_cache_size`
- **Type**: Integer
- **Default**: `10000`
- **Range**: 100-100000
- **Description**: Maximum number of cached items (memory backend only)
- **Example**:
  ```json
  "max_cache_size": 50000
  ```

#### `caching.cache_key_prefix`
- **Type**: String
- **Default**: `"tc_"`
- **Description**: Prefix for all cache keys
- **Example**:
  ```json
  "cache_key_prefix": "transaction_cost_"
  ```

#### `caching.redis_config` (Redis backend only)
- **Type**: Object
- **Description**: Redis connection configuration
- **Example**:
  ```json
  "redis_config": {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": null,
    "ssl": false,
    "connection_pool_size": 10
  }
  ```

### Performance Configuration

Controls performance monitoring and optimization.

#### `performance.enable_metrics_collection`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable collection of performance metrics
- **Example**:
  ```json
  "enable_metrics_collection": false
  ```

#### `performance.metrics_retention_days`
- **Type**: Integer
- **Default**: `30`
- **Range**: 1-365
- **Description**: Number of days to retain performance metrics
- **Example**:
  ```json
  "metrics_retention_days": 7
  ```

#### `performance.enable_profiling`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable detailed performance profiling
- **Example**:
  ```json
  "enable_profiling": true
  ```

#### `performance.profile_sample_rate`
- **Type**: Float
- **Default**: `0.1`
- **Range**: 0.0-1.0
- **Description**: Fraction of operations to profile (when profiling enabled)
- **Example**:
  ```json
  "profile_sample_rate": 0.05
  ```

### Logging Configuration

Controls logging behavior and output.

#### `logging.level`
- **Type**: String
- **Default**: `"INFO"`
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Description**: Minimum logging level
- **Example**:
  ```json
  "level": "DEBUG"
  ```

#### `logging.format`
- **Type**: String
- **Default**: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- **Description**: Log message format string
- **Example**:
  ```json
  "format": "[%(levelname)s] %(asctime)s - %(message)s"
  ```

#### `logging.file_path`
- **Type**: String
- **Default**: `"logs/transaction_costs.log"`
- **Description**: Path to log file
- **Example**:
  ```json
  "file_path": "/var/log/transaction_costs.log"
  ```

#### `logging.max_file_size`
- **Type**: String
- **Default**: `"10MB"`
- **Options**: Size in bytes, KB, MB, or GB
- **Description**: Maximum log file size before rotation
- **Example**:
  ```json
  "max_file_size": "50MB"
  ```

#### `logging.backup_count`
- **Type**: Integer
- **Default**: `5`
- **Range**: 0-50
- **Description**: Number of backup log files to keep
- **Example**:
  ```json
  "backup_count": 10
  ```

### Broker Configuration

Controls broker-specific settings and defaults.

#### `brokers.default_broker`
- **Type**: String
- **Default**: `"Zerodha"`
- **Options**: `"Zerodha"`, `"ICICI_Securities"`, `"Angel_Broking"`
- **Description**: Default broker for calculations
- **Example**:
  ```json
  "default_broker": "ICICI_Securities"
  ```

#### `brokers.broker_configs`
- **Type**: Object
- **Description**: Broker-specific configuration objects
- **Example**:
  ```json
  "broker_configs": {
    "Zerodha": {
      "api_key": "your_api_key",
      "api_secret": "your_api_secret",
      "exchange": "NSE"
    },
    "ICICI_Securities": {
      "user_id": "your_user_id",
      "password": "your_password",
      "exchange": "NSE"
    }
  }
  ```

### Market Data Configuration

Controls market data and timing settings.

#### `market_data.default_market_hours`
- **Type**: Object
- **Description**: Default market hours for validation
- **Example**:
  ```json
  "default_market_hours": {
    "start": "09:15",
    "end": "15:30",
    "timezone": "Asia/Kolkata"
  }
  ```

#### `market_data.enable_market_timing_validation`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Validate transactions against market hours
- **Example**:
  ```json
  "enable_market_timing_validation": false
  ```

### Regulatory Configuration

Controls regulatory compliance and fee calculations.

#### `regulatory.stt_rates`
- **Type**: Object
- **Description**: Securities Transaction Tax rates by instrument type
- **Example**:
  ```json
  "stt_rates": {
    "equity_delivery": 0.00025,
    "equity_intraday": 0.00025,
    "options": 0.00525,
    "futures": 0.0002
  }
  ```

#### `regulatory.exchange_charges`
- **Type**: Object
- **Description**: Exchange-specific charges
- **Example**:
  ```json
  "exchange_charges": {
    "NSE": 0.0000345,
    "BSE": 0.0000375
  }
  ```

## Environment Variables

Configuration can also be controlled via environment variables:

### Core Settings
```bash
# System settings
export TC_PRECISION_DECIMAL_PLACES=4
export TC_DEFAULT_CURRENCY=INR
export TC_DEFAULT_EXCHANGE=NSE
export TC_ENABLE_VALIDATION=true
export TC_STRICT_MODE=false

# Calculation settings
export TC_TIMEOUT_SECONDS=60
export TC_ENABLE_PARALLEL_PROCESSING=true
export TC_MAX_BATCH_SIZE=500
export TC_DEFAULT_CALCULATION_MODE=real_time
export TC_ENABLE_CACHING=true

# Caching settings
export TC_CACHE_BACKEND=redis
export TC_CACHE_TTL_SECONDS=1800
export TC_CACHE_MAX_SIZE=50000
export TC_CACHE_KEY_PREFIX=tc_

# Redis settings (if using Redis backend)
export TC_REDIS_HOST=localhost
export TC_REDIS_PORT=6379
export TC_REDIS_DB=0
export TC_REDIS_PASSWORD=yourpassword

# Performance settings
export TC_ENABLE_METRICS_COLLECTION=true
export TC_METRICS_RETENTION_DAYS=7
export TC_ENABLE_PROFILING=false
export TC_PROFILE_SAMPLE_RATE=0.1

# Logging settings
export TC_LOG_LEVEL=DEBUG
export TC_LOG_FILE_PATH=/var/log/transaction_costs.log
export TC_LOG_MAX_FILE_SIZE=50MB
export TC_LOG_BACKUP_COUNT=10
export TC_ENABLE_CONSOLE_LOGGING=true

# Broker settings
export TC_DEFAULT_BROKER=Zerodha
export TC_ENABLE_BROKER_DISCOVERY=true

# Market data settings
export TC_MARKET_START_TIME=09:15
export TC_MARKET_END_TIME=15:30
export TC_MARKET_TIMEZONE=Asia/Kolkata
export TC_ENABLE_MARKET_TIMING_VALIDATION=true
```

### Broker API Credentials
```bash
# Zerodha credentials
export ZERODHA_API_KEY=your_api_key
export ZERODHA_API_SECRET=your_api_secret
export ZERODHA_USER_ID=your_user_id
export ZERODHA_PASSWORD=your_password

# ICICI Securities credentials
export ICICI_USER_ID=your_user_id
export ICICI_PASSWORD=your_password
export ICICI_API_KEY=your_api_key
```

## Configuration Management

### Loading Configuration

```python
from src.trading.cost_config.base_config import CostConfiguration

# Load default configuration
config = CostConfiguration()

# Load from specific file
config = CostConfiguration(config_file='custom_config.json')

# Load with environment variable overrides
config = CostConfiguration(use_env_vars=True)
```

### Updating Configuration

```python
# Update individual settings
config.update_setting('calculation.timeout_seconds', 60)
config.update_setting('caching.ttl_seconds', 1800)

# Update multiple settings
config.update_settings({
    'system.precision_decimal_places': 4,
    'performance.enable_profiling': True,
    'logging.level': 'DEBUG'
})

# Save changes
config.save_config()
```

### Validating Configuration

```python
from src.trading.cost_config.config_validator import CostConfigurationValidator

# Create validator
validator = CostConfigurationValidator(strict_mode=True)

# Validate current configuration
result = validator.validate_full_configuration(config.get_all_settings())

if result.is_valid:
    print("✅ Configuration is valid")
else:
    print("❌ Configuration errors:")
    for error in result.errors:
        print(f"  - {error}")
```

### Configuration Profiles

Create different configuration profiles for different environments:

#### Development Profile (`configs/cost_config/development.json`)
```json
{
  "system": {
    "strict_mode": false
  },
  "calculation": {
    "timeout_seconds": 10
  },
  "caching": {
    "backend": "memory",
    "ttl_seconds": 300
  },
  "logging": {
    "level": "DEBUG",
    "enable_console_logging": true
  },
  "performance": {
    "enable_profiling": true,
    "profile_sample_rate": 1.0
  }
}
```

#### Production Profile (`configs/cost_config/production.json`)
```json
{
  "system": {
    "strict_mode": true
  },
  "calculation": {
    "timeout_seconds": 30,
    "enable_parallel_processing": true
  },
  "caching": {
    "backend": "redis",
    "ttl_seconds": 3600
  },
  "logging": {
    "level": "INFO",
    "enable_console_logging": false,
    "file_path": "/var/log/transaction_costs.log"
  },
  "performance": {
    "enable_profiling": false,
    "enable_metrics_collection": true
  }
}
```

#### Loading Profiles
```python
# Load specific profile
config = CostConfiguration(config_file='configs/cost_config/production.json')

# Or use environment variable
import os
profile = os.getenv('TC_PROFILE', 'development')
config = CostConfiguration(config_file=f'configs/cost_config/{profile}.json')
```

## Best Practices

### 1. Environment-Specific Configuration

Use different configurations for different environments:

```bash
# .env.development
TC_LOG_LEVEL=DEBUG
TC_ENABLE_PROFILING=true
TC_CACHE_BACKEND=memory

# .env.production
TC_LOG_LEVEL=INFO
TC_ENABLE_PROFILING=false
TC_CACHE_BACKEND=redis
```

### 2. Secure Credential Management

Never store credentials in configuration files:

```python
# Good: Use environment variables
import os
api_key = os.getenv('ZERODHA_API_KEY')

# Bad: Store in config file
# config['brokers']['zerodha']['api_key'] = 'your_key'  # Don't do this
```

### 3. Configuration Validation

Always validate configuration before use:

```python
def setup_calculator():
    config = CostConfiguration()
    
    # Validate configuration
    validator = CostConfigurationValidator()
    result = validator.validate_full_configuration(config.get_all_settings())
    
    if not result.is_valid:
        raise ValueError(f"Invalid configuration: {result.errors}")
    
    return config
```

### 4. Performance Tuning

Adjust configuration based on usage patterns:

```python
# For high-frequency trading
config.update_settings({
    'calculation.enable_parallel_processing': True,
    'calculation.max_batch_size': 10000,
    'caching.backend': 'redis',
    'caching.ttl_seconds': 600,  # 10 minutes
    'performance.enable_metrics_collection': False  # Reduce overhead
})

# For research/backtesting
config.update_settings({
    'calculation.default_calculation_mode': 'historical',
    'caching.ttl_seconds': 86400,  # 24 hours
    'performance.enable_profiling': True,
    'logging.level': 'DEBUG'
})
```

## Troubleshooting Configuration

### Common Issues

#### Configuration File Not Found
```python
try:
    config = CostConfiguration(config_file='missing_file.json')
except FileNotFoundError:
    print("Using default configuration")
    config = CostConfiguration()
```

#### Invalid Configuration Values
```python
try:
    config.update_setting('calculation.timeout_seconds', -1)
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

#### Redis Connection Issues
```python
try:
    config.update_setting('caching.backend', 'redis')
    # Test Redis connection
    cache = config.get_cache_backend()
    cache.ping()
except Exception as e:
    print(f"Redis unavailable, falling back to memory cache: {e}")
    config.update_setting('caching.backend', 'memory')
```

### Configuration Debugging

Enable debug mode to troubleshoot configuration issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = CostConfiguration(debug=True)
```

## Related Documentation

- [Installation Guide](../user_guide/installation.md) - Initial setup and configuration
- [Broker Configuration](broker_configs.md) - Broker-specific settings
- [Performance Tuning](performance_tuning.md) - Optimization guidelines
- [Environment Setup](environment_setup.md) - Environment configuration
- [Troubleshooting](../troubleshooting/configuration_errors.md) - Common configuration issues