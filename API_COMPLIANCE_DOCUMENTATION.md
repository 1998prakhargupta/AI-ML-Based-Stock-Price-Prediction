# 🛡️ API COMPLIANCE AND DATA PROVIDER TERMS DOCUMENTATION

## Overview

This document provides comprehensive documentation for API compliance and data provider terms of service adherence in the price prediction project. All data fetching operations now include strict compliance monitoring, rate limiting, and terms of service validation.

## 📋 Data Providers and Terms of Service

### 1. Breeze Connect API

**Commercial Status**: ✅ Commercial use allowed with proper licensing
**Attribution Required**: ❌ No attribution required
**Data Redistribution**: ❌ Prohibited

#### Rate Limits Implemented:
- **Requests per Second**: 2.0 (conservative estimate)
- **Requests per Minute**: 100
- **Requests per Hour**: 1,000
- **Requests per Day**: 5,000
- **Max Concurrent Requests**: 3
- **Minimum Request Interval**: 0.5 seconds
- **Cooldown After Error**: 2.0 seconds

#### Compliance Features:
- ✅ Automatic request throttling
- ✅ Error-based backoff
- ✅ Request monitoring and analytics
- ✅ Large request splitting
- ✅ Response data validation
- ✅ Usage statistics tracking

#### Terms Compliance:
```python
terms_compliance = {
    'commercial_use_allowed': True,           # With proper licensing
    'data_redistribution_prohibited': True,   # Cannot share data
    'attribution_required': False,            # No attribution needed
    'real_time_data_restrictions': True,      # Subject to usage limits
    'bulk_download_limitations': True,        # Rate limited
    'api_abuse_prevention': True              # Strict monitoring
}
```

### 2. Yahoo Finance

**Commercial Status**: ❌ Commercial use prohibited without licensing
**Attribution Required**: ✅ "Data provided by Yahoo Finance"
**Data Redistribution**: ❌ Strictly prohibited

#### Rate Limits Implemented:
- **Requests per Second**: 1.0 (very conservative)
- **Requests per Minute**: 30
- **Requests per Hour**: 500
- **Requests per Day**: 2,000
- **Max Concurrent Requests**: 2
- **Minimum Request Interval**: 1.0 second
- **Cooldown After Error**: 5.0 seconds

#### Compliance Features:
- ✅ Educational/research use validation
- ✅ Commercial use blocking
- ✅ Automatic data attribution
- ✅ Response caching (24-hour expiry)
- ✅ Batch request splitting
- ✅ Comprehensive usage warnings

#### Terms Compliance:
```python
terms_compliance = {
    'commercial_use_prohibited': True,        # Strictly enforced
    'personal_use_only': True,               # Educational/research only
    'data_redistribution_prohibited': True,   # Cannot share data
    'attribution_required': True,            # Must credit Yahoo Finance
    'api_abuse_prevention': True,            # Strict rate limiting
    'rate_limiting_mandatory': True,         # Required by terms
    'caching_recommended': True,             # Reduces API calls
    'research_use_allowed': True             # Academic research OK
}
```

## 🔧 Implementation Architecture

### Compliance Manager (`api_compliance.py`)

Central compliance management system that provides:

- **Multi-Provider Support**: Handles different API providers with specific configurations
- **Rate Limiting**: Implements sophisticated rate limiting with burst capacity
- **Request Monitoring**: Tracks all API requests with detailed analytics
- **Compliance Levels**: Configurable enforcement (Strict, Moderate, Lenient, Monitoring)
- **Response Caching**: Intelligent caching to reduce API calls
- **Usage Reporting**: Comprehensive compliance and usage reports

#### Key Classes:

```python
class ComplianceManager:
    def check_rate_limit(provider, endpoint, priority) -> Dict
    def request_permission(provider, endpoint, params) -> APIRequest
    def record_response(request, success, response_time, data_size) -> APIResponse
    def get_usage_statistics(provider) -> Dict
    def generate_compliance_report() -> str
```

#### Decorator Usage:

```python
@compliance_decorator(DataProvider.YAHOO_FINANCE, "download")
def fetch_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)
```

### Enhanced Breeze Utilities (`compliance_breeze_utils.py`)

Breeze API wrapper with comprehensive compliance:

- **Automatic Authentication**: Compliant session management
- **Request Validation**: Parameter validation and data estimation
- **Large Request Handling**: Automatic splitting for compliance
- **Response Validation**: Data quality and completeness checks
- **Usage Monitoring**: Real-time compliance status tracking

#### Key Features:

```python
class ComplianceBreezeDataManager:
    def authenticate() -> bool                    # Compliant authentication
    def get_quotes_safe(stock_code, exchange)     # Rate-limited quotes
    def get_historical_data_safe(request)         # Compliant historical data
    def get_session_compliance_report()           # Usage statistics
    def save_compliance_documentation()           # Generate reports
```

### Yahoo Finance Utilities (`compliance_yahoo_utils.py`)

Yahoo Finance wrapper with strict terms compliance:

- **Terms Validation**: Enforces personal/educational use only
- **Commercial Use Blocking**: Prevents unauthorized commercial usage
- **Attribution Management**: Automatic data source attribution
- **Intelligent Caching**: 24-hour cache to reduce API calls
- **Batch Processing**: Compliant multi-symbol downloading

#### Key Features:

```python
class ComplianceYahooFinanceManager:
    def download_symbol_data(symbol, period, interval)    # Compliant download
    def download_multiple_symbols(symbols)                # Batch processing
    def get_terms_compliance_report()                     # Terms validation
    def save_compliance_documentation()                   # Usage reports
```

## 📊 Usage Examples

### 1. Compliant Breeze API Usage

```python
from compliance_breeze_utils import ComplianceBreezeDataManager
from api_compliance import ComplianceLevel

# Initialize with strict compliance
manager = ComplianceBreezeDataManager(
    compliance_level=ComplianceLevel.STRICT
)

# Authenticate
if manager.authenticate():
    # Get live quotes (rate limited)
    response = manager.get_quotes_safe("TCS", "NSE")
    
    # Get historical data (validated and split if needed)
    request = MarketDataRequest(
        stock_code="TCS",
        exchange_code="NSE", 
        product_type="cash",
        interval="5minute",
        from_date="2024-01-01",
        to_date="2024-01-31"
    )
    
    response = manager.get_historical_data_safe(request)
    
    # Generate compliance report
    report_path = manager.save_compliance_documentation()
    print(f"Compliance report: {report_path}")

# Cleanup
manager.cleanup()
```

### 2. Compliant Yahoo Finance Usage

```python
from compliance_yahoo_utils import ComplianceYahooFinanceManager
from api_compliance import ComplianceLevel

# Initialize with strict compliance (educational use only)
manager = ComplianceYahooFinanceManager(
    compliance_level=ComplianceLevel.STRICT
)

# Download single symbol (with caching and attribution)
result = manager.download_symbol_data(
    symbol="AAPL",
    period="1mo", 
    interval="1d"
)

if result['success']:
    data = result['data']  # Includes proper attribution
    print(f"Downloaded {len(data)} data points")

# Download multiple symbols (batch processed with rate limiting)
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
results = manager.download_multiple_symbols(
    symbols=symbols,
    period="3mo",
    interval="1d"
)

# Generate compliance report
report_path = manager.save_compliance_documentation()

# Cleanup
manager.cleanup()
```

### 3. Direct Compliance Manager Usage

```python
from api_compliance import get_compliance_manager, DataProvider, compliance_decorator

# Get global compliance manager
manager = get_compliance_manager()

# Use decorator for custom functions
@compliance_decorator(DataProvider.YAHOO_FINANCE, "custom_endpoint")
def my_data_function(symbol):
    # Your data fetching logic here
    return fetch_some_data(symbol)

# Manual compliance checking
rate_check = manager.check_rate_limit(
    provider=DataProvider.BREEZE_CONNECT,
    endpoint="get_quotes"
)

if rate_check['allowed']:
    # Make API call
    pass
else:
    print(f"Rate limited, wait {rate_check['wait_time']} seconds")

# Generate comprehensive report
report = manager.generate_compliance_report()
print(report)
```

## 📈 Monitoring and Reporting

### Usage Statistics Tracked

For each API provider, the system tracks:

- **Request Counts**: Total, successful, failed, rate-limited
- **Response Times**: Average, min, max response times
- **Data Volume**: Total bytes transferred, data points retrieved
- **Error Rates**: Success rates, error patterns
- **Compliance Scores**: Overall compliance health metrics
- **Cache Performance**: Cache hit rates, storage efficiency

### Automated Reports

The system generates several types of reports:

1. **Session Reports**: Real-time usage during active sessions
2. **Daily Reports**: Daily usage summaries and compliance status
3. **Compliance Reports**: Terms of service adherence validation
4. **Usage Analytics**: Detailed usage patterns and recommendations

### Report Formats

Reports are generated in multiple formats:

- **Markdown**: Human-readable compliance documentation
- **JSON**: Machine-readable usage statistics
- **CSV**: Detailed request/response logs for analysis

## 🚨 Compliance Alerts and Warnings

### Automatic Monitoring

The system automatically monitors for:

- **Rate Limit Violations**: Requests exceeding provider limits
- **Terms Violations**: Commercial use of non-commercial APIs
- **Error Rate Spikes**: High failure rates indicating issues
- **Unusual Patterns**: Suspicious usage that might violate terms

### Alert Types

1. **Info Alerts**: Normal operational information
2. **Warning Alerts**: Approaching limits or minor violations
3. **Error Alerts**: Violations requiring immediate attention
4. **Critical Alerts**: Severe violations that could result in API suspension

### Example Alerts

```
🚨 COMPLIANCE ALERT: High request rate detected for Yahoo Finance
⚠️  WARNING: Approaching hourly rate limit for Breeze Connect (850/1000)
ℹ️  INFO: Using cached data to reduce API calls
❌ ERROR: Commercial use detected for Yahoo Finance (prohibited)
```

## 🔄 Compliance Levels

### Strict Compliance
- **Enforcement**: All limits strictly enforced
- **Violations**: Immediate blocking of violating requests
- **Monitoring**: Comprehensive logging and alerting
- **Use Case**: Production environments, commercial applications

### Moderate Compliance
- **Enforcement**: Most limits enforced with some flexibility
- **Violations**: Warnings logged, minor violations allowed
- **Monitoring**: Standard logging and periodic alerts
- **Use Case**: Development environments, testing phases

### Lenient Compliance
- **Enforcement**: Basic limits with significant flexibility
- **Violations**: Logged but generally allowed
- **Monitoring**: Minimal monitoring, basic statistics
- **Use Case**: Experimental development, research prototypes

### Monitoring Only
- **Enforcement**: No enforcement, monitoring only
- **Violations**: All violations logged but allowed
- **Monitoring**: Comprehensive tracking without blocking
- **Use Case**: Compliance assessment, usage analysis

## 📝 Best Practices

### 1. API Usage Best Practices

- **Always use compliance decorators** for API functions
- **Implement proper error handling** for rate limit exceptions
- **Cache responses** when appropriate to reduce API calls
- **Use batch requests** instead of individual calls when possible
- **Monitor usage statistics** regularly to stay within limits
- **Respect provider terms** even when technically possible to violate

### 2. Data Handling Best Practices

- **Attribute data sources** properly in all outputs
- **Do not redistribute** data without permission
- **Use data only for approved purposes** (commercial vs. personal)
- **Implement proper data retention** policies
- **Validate data quality** after retrieval
- **Document data lineage** for compliance auditing

### 3. Development Best Practices

- **Test with strict compliance** during development
- **Implement graceful degradation** for rate-limited scenarios
- **Use appropriate compliance levels** for different environments
- **Monitor compliance metrics** in production
- **Regular compliance reviews** and updates
- **Keep documentation updated** with provider term changes

## 🔧 Configuration

### Environment Variables

```bash
# Compliance configuration
API_COMPLIANCE_LEVEL=strict          # strict, moderate, lenient, monitoring
API_CACHE_DURATION=24               # Hours to cache responses
API_RATE_LIMIT_BUFFER=0.8           # Buffer factor for rate limits (80% of max)
API_MAX_RETRIES=3                   # Maximum retry attempts
API_RETRY_DELAY=1.0                 # Base delay between retries

# Provider-specific settings
BREEZE_RATE_LIMIT_DELAY=0.5         # Seconds between Breeze requests
YAHOO_RATE_LIMIT_DELAY=1.0          # Seconds between Yahoo requests
ALPHA_VANTAGE_RATE_LIMIT_DELAY=12.0 # Seconds between Alpha Vantage requests
```

### Configuration Files

Create `api_compliance_config.json`:

```json
{
  "breeze_connect": {
    "requests_per_second": 2.0,
    "requests_per_minute": 100,
    "requests_per_hour": 1000,
    "requests_per_day": 5000,
    "commercial_use_allowed": true,
    "attribution_required": false
  },
  "yahoo_finance": {
    "requests_per_second": 1.0,
    "requests_per_minute": 30,
    "requests_per_hour": 500,
    "requests_per_day": 2000,
    "commercial_use_allowed": false,
    "attribution_required": true,
    "cache_duration_hours": 24
  }
}
```

## 🎯 Compliance Checklist

### Before Deployment

- [ ] All API calls use compliance decorators
- [ ] Rate limiting configured for all providers
- [ ] Terms of service compliance validated
- [ ] Proper attribution implemented
- [ ] Error handling for rate limits
- [ ] Usage monitoring enabled
- [ ] Compliance documentation generated
- [ ] Commercial use permissions verified

### During Operation

- [ ] Monitor compliance dashboards
- [ ] Review usage statistics regularly
- [ ] Respond to compliance alerts promptly
- [ ] Update configurations as needed
- [ ] Generate periodic compliance reports
- [ ] Maintain API provider relationships
- [ ] Keep terms of service documentation current

### Regular Maintenance

- [ ] Review provider terms updates
- [ ] Update rate limit configurations
- [ ] Analyze usage patterns
- [ ] Optimize caching strategies
- [ ] Update compliance documentation
- [ ] Train team on compliance requirements
- [ ] Audit data usage practices

## 📞 Support and Resources

### Internal Documentation

- `api_compliance.py` - Core compliance management
- `compliance_breeze_utils.py` - Breeze API compliance wrapper
- `compliance_yahoo_utils.py` - Yahoo Finance compliance wrapper
- This document - Comprehensive compliance guide

### Provider Documentation

- **Breeze Connect**: [API Documentation](https://breezeconnect.com)
- **Yahoo Finance**: [Terms of Service](https://finance.yahoo.com/terms)
- **Alpha Vantage**: [API Terms](https://www.alphavantage.co/terms_of_service/)

### Compliance Resources

- Monitor provider websites for terms updates
- Subscribe to API provider newsletters
- Join developer communities for best practices
- Regular legal review for commercial use cases

---

**Last Updated**: January 2024  
**Next Review**: March 2024  
**Document Version**: 1.0

**Compliance Officer**: Development Team  
**Legal Review**: Required for commercial deployment
