# Installation Guide

This guide will walk you through setting up the Transaction Cost Modeling System.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB free space for installation

### Recommended Requirements
- **Python**: 3.9 or 3.10 (latest stable)
- **Memory**: 16GB RAM for large-scale operations
- **Storage**: 2GB free space for data and logs

## Installation Methods

### Method 1: Project Setup (Recommended)

If you're working with the full AI-ML-Based-Stock-Price-Prediction project:

#### 1. Clone the Repository

```bash
# Clone the main repository
git clone https://github.com/1998prakhargupta/AI-ML-Based-Stock-Price-Prediction.git

# Navigate to the project directory
cd AI-ML-Based-Stock-Price-Prediction
```

#### 2. Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or use the Makefile
make install
```

#### 4. Verify Installation

```bash
# Run the setup script
python scripts/setup.py

# Test the transaction cost system
python -c "from src.trading.transaction_costs.models import TransactionRequest; print('✅ Installation successful!')"
```

### Method 2: Standalone Installation

If you only need the transaction cost system:

#### 1. Install Core Dependencies

```bash
# Create a new environment
python -m venv transaction_costs_env
source transaction_costs_env/bin/activate  # Linux/macOS
# transaction_costs_env\Scripts\activate  # Windows

# Install required packages
pip install decimal pandas numpy python-dateutil typing-extensions
```

#### 2. Download Source Files

```bash
# Create project structure
mkdir -p transaction_costs/src/trading
cd transaction_costs

# Download the transaction costs module
# (You would copy the src/trading/transaction_costs/ directory here)
```

## Configuration Setup

### 1. Environment Configuration

Create a `.env` file in your project root:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the configuration
nano .env  # or your preferred editor
```

**Example .env file:**
```bash
# Transaction Cost Configuration
TRANSACTION_COST_PRECISION=2
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600
DEFAULT_EXCHANGE=NSE
DEFAULT_CURRENCY=INR

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/transaction_costs.log

# Performance Settings
ENABLE_PERFORMANCE_TRACKING=true
BATCH_SIZE_LIMIT=1000
PARALLEL_PROCESSING=true
```

### 2. Broker Configuration

Initialize default broker configurations:

```python
from src.trading.cost_config.base_config import CostConfiguration

# Initialize configuration system
config = CostConfiguration()

# Create default broker configurations
config.create_default_broker_configurations()

# Save configuration
config.save_config()

print("✅ Broker configurations initialized")
```

### 3. Verify Configuration

```python
# Test configuration loading
from src.trading.cost_config.base_config import CostConfiguration

config = CostConfiguration()

# Check if brokers are configured
brokers = config.get_all_broker_configurations()
print(f"Configured brokers: {list(brokers.keys())}")

# Test a calculation
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
calculator = ZerodhaCalculator()
print("✅ Configuration verified")
```

## Database Setup (Optional)

For production use with caching and performance tracking:

### 1. Redis Setup (for caching)

```bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping
# Should return: PONG
```

### 2. Configure Redis Caching

```python
from src.trading.transaction_costs.cache.redis_cache import RedisCache

# Initialize Redis cache
cache = RedisCache(
    host='localhost',
    port=6379,
    db=0,
    password=None  # Set if Redis has authentication
)

# Test cache connection
cache.set('test_key', 'test_value', ttl=60)
value = cache.get('test_key')
print(f"Cache test: {value}")  # Should print: test_value
```

## Development Setup

For contributors and advanced users:

### 1. Install Development Dependencies

```bash
# Install development tools
pip install -r requirements-dev.txt

# Or install individually
pip install pytest pytest-cov black isort flake8 mypy pre-commit
```

### 2. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test pre-commit
pre-commit run --all-files
```

### 3. Run Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Run with coverage
python -m pytest tests/ --cov=src/trading/transaction_costs/ --cov-report=html
```

## Platform-Specific Instructions

### Windows Setup

#### 1. Install Python

```powershell
# Download Python from python.org or use Chocolatey
choco install python

# Verify installation
python --version
pip --version
```

#### 2. Windows-Specific Dependencies

```powershell
# Install Visual C++ Build Tools (if needed for some packages)
# Download from Microsoft Visual Studio website

# Install Windows-specific packages
pip install pywin32  # For Windows services integration
```

#### 3. Set Environment Variables

```powershell
# Add to system PATH (if needed)
setx PATH "%PATH%;C:\Python\Scripts"

# Set Python path for project
setx PYTHONPATH "%CD%\src"
```

### macOS Setup

#### 1. Install Prerequisites

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install Git (if not already installed)
brew install git
```

#### 2. macOS-Specific Configuration

```bash
# Install command line tools
xcode-select --install

# Set up PATH for Homebrew Python
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Linux Setup

#### 1. Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install development tools
sudo apt install build-essential python3-dev

# Install Git (if not already installed)
sudo apt install git
```

#### 2. CentOS/RHEL/Fedora

```bash
# Install Python and pip
sudo dnf install python3 python3-pip

# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

## Docker Setup (Advanced)

For containerized deployment:

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY docs/transaction_costs/examples/ examples/

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.trading.transaction_costs.models import TransactionRequest; print('OK')" || exit 1

CMD ["python", "examples/basic_calculations.py"]
```

### 2. Build and Run Docker Container

```bash
# Build the image
docker build -t transaction-costs .

# Run the container
docker run -it --rm transaction-costs

# Run with volume mounting for development
docker run -it --rm -v $(pwd):/app transaction-costs bash
```

## Common Installation Issues

### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Set PYTHONPATH environment variable
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Or add to your shell profile
echo 'export PYTHONPATH="${PWD}/src:${PYTHONPATH}"' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: Decimal Precision Issues

**Problem**: Incorrect calculation results due to floating-point precision

**Solution**:
```python
# Always use Decimal for financial calculations
from decimal import Decimal, getcontext

# Set precision (default is usually sufficient)
getcontext().prec = 28

# Use Decimal for all monetary values
price = Decimal('2500.00')  # Correct
# price = 2500.0  # Avoid this for financial calculations
```

### Issue 3: Permission Errors

**Problem**: Permission denied when accessing log files or configuration

**Solution**:
```bash
# Create necessary directories with proper permissions
mkdir -p logs configs/cost_config
chmod 755 logs configs/cost_config

# On Linux/macOS, ensure proper ownership
sudo chown -R $USER:$USER logs configs
```

### Issue 4: Performance Issues

**Problem**: Slow calculations for large datasets

**Solution**:
```python
# Enable parallel processing
calculator = ZerodhaCalculator()
results = calculator.calculate_batch(requests, parallel=True)

# Enable caching
calculator.configure_caching(enable=True, ttl_seconds=3600)

# Use async operations for I/O-bound tasks
import asyncio
results = await calculator.calculate_cost_async(request)
```

## Validation and Testing

### 1. Quick Validation

```python
#!/usr/bin/env python3
"""Quick validation script for installation."""

def validate_installation():
    """Validate that all components are working correctly."""
    
    try:
        # Test imports
        from src.trading.transaction_costs.models import TransactionRequest, TransactionType, InstrumentType
        from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
        from decimal import Decimal
        print("✅ Imports successful")
        
        # Test basic calculation
        request = TransactionRequest(
            symbol='TEST',
            quantity=10,
            price=Decimal('100.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        calculator = ZerodhaCalculator()
        result = calculator.calculate_cost(request)
        print(f"✅ Basic calculation successful: ₹{result.total_cost}")
        
        # Test configuration
        from src.trading.cost_config.base_config import CostConfiguration
        config = CostConfiguration()
        print("✅ Configuration system working")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    if validate_installation():
        print("\n🎉 Installation validation successful!")
        print("You're ready to use the Transaction Cost Modeling System.")
    else:
        print("\n💥 Installation validation failed!")
        print("Please check the installation steps and try again.")
```

### 2. Run Example Scripts

```bash
# Run basic examples
python docs/transaction_costs/examples/basic_calculations.py

# Run integration examples
python docs/transaction_costs/examples/ml_integration.py

# Run performance tests
python -m pytest tests/performance/ -v
```

## Next Steps

After successful installation:

1. **Read the [Getting Started Guide](../getting_started.md)** for basic usage
2. **Explore [Examples](../examples/)** for real-world scenarios  
3. **Review [Configuration Options](../configuration/configuration_reference.md)** for customization
4. **Check [API Documentation](../api/)** for detailed method references
5. **Join the Community** - contribute to the project or ask questions

## Support

If you encounter issues during installation:

1. **Check the [Troubleshooting Guide](../troubleshooting/common_issues.md)**
2. **Review the [FAQ](../troubleshooting/common_issues.md#frequently-asked-questions)**
3. **Search existing [GitHub Issues](https://github.com/1998prakhargupta/AI-ML-Based-Stock-Price-Prediction/issues)**
4. **Create a new issue** with the `installation` and `transaction-costs` labels

## Contributing

If you'd like to contribute to the project:

1. **Fork the repository** on GitHub
2. **Set up the development environment** using the instructions above
3. **Run the test suite** to ensure everything works
4. **Make your changes** and add tests
5. **Submit a pull request** with a clear description

Welcome to the Transaction Cost Modeling System! 🚀