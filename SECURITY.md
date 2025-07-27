# Security Policy

## 🔒 Security Statement

The Price Predictor Project takes security seriously. This document outlines our security practices, how to report vulnerabilities, and guidelines for secure development.

## 📧 Reporting Security Vulnerabilities

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them responsibly by emailing us directly:

📧 **Security Contact**: [1998prakhargupta@gmail.com](mailto:1998prakhargupta@gmail.com)

### What to Include in Your Report

When reporting a security vulnerability, please include:

1. **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
2. **Full paths** of source file(s) related to the manifestation of the issue
3. **Location** of the affected source code (tag/branch/commit or direct URL)
4. **Special configuration** required to reproduce the issue
5. **Step-by-step instructions** to reproduce the issue
6. **Proof-of-concept or exploit code** (if possible)
7. **Impact** of the issue, including how an attacker might exploit the issue

We prefer all communications to be in English.

### Response Timeline

- **Acknowledgment**: Within 48 hours of report submission
- **Initial Assessment**: Within 1 week
- **Regular Updates**: Every 1-2 weeks until resolution
- **Resolution**: Varies based on complexity and severity

## 🛡️ Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| 0.9.x   | ✅ Yes             |
| 0.8.x   | ⚠️ Critical fixes only |
| < 0.8   | ❌ No              |

## 🔐 Security Best Practices

### For Users

#### API Key Security
```bash
# ✅ DO: Store credentials in environment variables
export BREEZE_API_KEY="your_api_key_here"
export BREEZE_SECRET="your_secret_here"

# ❌ DON'T: Hard-code credentials in files
api_key = "your_api_key_here"  # Never do this!
```

#### Environment Configuration
```bash
# Create secure .env file
cp .env.example .env
chmod 600 .env  # Restrict file permissions

# Add your credentials
echo "BREEZE_API_KEY=your_key" >> .env
echo "BREEZE_SECRET=your_secret" >> .env
```

#### Data Protection
- **Never commit** real financial data to version control
- **Use test data** for development and testing
- **Encrypt sensitive data** when storing locally
- **Regular cleanup** of temporary files and logs

### For Developers

#### Secure Coding Practices

1. **Input Validation**
   ```python
   def validate_stock_symbol(symbol: str) -> bool:
       """Validate stock symbol to prevent injection attacks."""
       if not symbol or len(symbol) > 10:
           return False
       return symbol.isalnum() and symbol.isupper()
   ```

2. **API Key Handling**
   ```python
   import os
   from typing import Optional
   
   def get_api_credentials() -> Optional[tuple]:
       """Safely retrieve API credentials from environment."""
       api_key = os.getenv('BREEZE_API_KEY')
       secret = os.getenv('BREEZE_SECRET')
       
       if not api_key or not secret:
           raise ValueError("API credentials not found in environment")
       
       return api_key, secret
   ```

3. **Data Sanitization**
   ```python
   import re
   
   def sanitize_file_path(path: str) -> str:
       """Sanitize file paths to prevent directory traversal."""
       # Remove any path traversal attempts
       path = re.sub(r'\.\./', '', path)
       path = re.sub(r'\.\.\\', '', path)
       
       # Ensure path is within allowed directories
       allowed_dirs = ['data/', 'outputs/', 'reports/']
       if not any(path.startswith(dir_) for dir_ in allowed_dirs):
           raise ValueError(f"Path not allowed: {path}")
       
       return path
   ```

#### Dependency Security

1. **Regular Updates**
   ```bash
   # Check for security vulnerabilities
   pip-audit
   
   # Update dependencies
   pip install --upgrade -r requirements.txt
   ```

2. **Dependency Scanning**
   ```bash
   # Add to CI/CD pipeline
   bandit -r src/
   safety check
   ```

#### Code Review Security Checklist

- [ ] **No hardcoded credentials** in code
- [ ] **Input validation** for all user inputs
- [ ] **Proper error handling** without information leakage
- [ ] **SQL injection prevention** (if applicable)
- [ ] **File path validation** for file operations
- [ ] **Secure API communications** (HTTPS only)
- [ ] **Logging excludes** sensitive information

## 🔒 Infrastructure Security

### Development Environment

1. **Virtual Environment Isolation**
   ```bash
   # Use virtual environments
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies in isolation
   pip install -r requirements.txt
   ```

2. **Git Security**
   ```bash
   # Add security-focused git hooks
   pre-commit install
   
   # Scan for secrets before commit
   git-secrets --scan
   ```

### Production Deployment

1. **Environment Variables**
   ```yaml
   # docker-compose.yml
   services:
     app:
       environment:
         - BREEZE_API_KEY=${BREEZE_API_KEY}
         - BREEZE_SECRET=${BREEZE_SECRET}
       env_file:
         - .env  # Never commit this file
   ```

2. **Container Security**
   ```dockerfile
   # Dockerfile
   FROM python:3.9-slim
   
   # Create non-root user
   RUN useradd --create-home --shell /bin/bash app
   USER app
   
   # Install dependencies
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   # Copy application
   COPY --chown=app:app . /app
   WORKDIR /app
   ```

### API Security

1. **Rate Limiting**
   ```python
   from functools import wraps
   import time
   
   def rate_limit(calls_per_minute: int):
       """Rate limiting decorator for API calls."""
       def decorator(func):
           last_called = [0.0]
           
           @wraps(func)
           def wrapper(*args, **kwargs):
               elapsed = time.time() - last_called[0]
               left_to_wait = 60.0 / calls_per_minute - elapsed
               if left_to_wait > 0:
                   time.sleep(left_to_wait)
               ret = func(*args, **kwargs)
               last_called[0] = time.time()
               return ret
           return wrapper
       return decorator
   ```

2. **Request Validation**
   ```python
   def validate_api_request(request_data: dict) -> bool:
       """Validate API request data."""
       required_fields = ['symbol', 'timeframe']
       
       for field in required_fields:
           if field not in request_data:
               return False
       
       # Additional validation logic
       return True
   ```

## 🔍 Security Testing

### Automated Security Scanning

1. **Static Analysis**
   ```bash
   # Security linting
   bandit -r src/ -f json -o security-report.json
   
   # Dependency vulnerabilities
   safety check --json --output security-deps.json
   
   # SAST scanning
   semgrep --config=auto src/
   ```

2. **Dynamic Testing**
   ```bash
   # API security testing
   pytest tests/security/
   
   # Load testing for DoS prevention
   locust -f tests/load/locustfile.py
   ```

### Manual Security Testing

1. **Input Validation Testing**
   - Test with malformed inputs
   - Test boundary conditions
   - Test injection attacks

2. **Authentication Testing**
   - Test with invalid credentials
   - Test session management
   - Test privilege escalation

3. **Data Security Testing**
   - Test data encryption
   - Test secure transmission
   - Test data retention policies

## 📊 Monitoring and Incident Response

### Security Monitoring

1. **Logging Security Events**
   ```python
   import logging
   
   security_logger = logging.getLogger('security')
   
   def log_security_event(event_type: str, details: dict):
       """Log security-related events."""
       security_logger.warning(
           f"Security Event: {event_type}",
           extra={
               'event_type': event_type,
               'details': details,
               'timestamp': time.time()
           }
       )
   ```

2. **Anomaly Detection**
   ```python
   def detect_unusual_activity(api_calls: list) -> bool:
       """Detect unusual API usage patterns."""
       if len(api_calls) > 1000:  # Too many calls
           return True
       
       # Check for rapid succession calls
       timestamps = [call['timestamp'] for call in api_calls]
       if max(timestamps) - min(timestamps) < 60:  # All within 1 minute
           return True
       
       return False
   ```

### Incident Response Plan

1. **Detection Phase**
   - Monitor logs for security events
   - Automated alerting for anomalies
   - User reports of suspicious activity

2. **Containment Phase**
   - Isolate affected systems
   - Preserve evidence
   - Prevent further damage

3. **Eradication Phase**
   - Remove malicious code/access
   - Patch vulnerabilities
   - Update security measures

4. **Recovery Phase**
   - Restore systems from clean backups
   - Monitor for continued activity
   - Gradual return to normal operations

5. **Lessons Learned**
   - Document incident details
   - Update security procedures
   - Improve detection capabilities

## 🔧 Security Tools and Resources

### Recommended Security Tools

1. **Static Analysis**
   - [Bandit](https://bandit.readthedocs.io/): Python security linter
   - [Safety](https://pyup.io/safety/): Dependency vulnerability scanner
   - [Semgrep](https://semgrep.dev/): Multi-language SAST tool

2. **Dependency Management**
   - [pip-audit](https://pypi.org/project/pip-audit/): Audit Python dependencies
   - [Dependabot](https://github.com/dependabot): Automated dependency updates

3. **Secrets Detection**
   - [git-secrets](https://github.com/awslabs/git-secrets): Prevent committing secrets
   - [truffleHog](https://github.com/trufflesecurity/trufflehog): Find secrets in git history

### Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security](https://python-security.readthedocs.io/)
- [Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [API Security Checklist](https://github.com/shieldfy/API-Security-Checklist)

## 📝 Security Configuration Examples

### .env.example (Security Template)
```bash
# API Credentials (NEVER commit real values)
BREEZE_API_KEY=your_breeze_api_key_here
BREEZE_SECRET=your_breeze_secret_here
ZERODHA_API_KEY=your_zerodha_api_key_here
ZERODHA_SECRET=your_zerodha_secret_here

# Database (if applicable)
DATABASE_URL=sqlite:///app.db
DATABASE_ENCRYPTION_KEY=your_encryption_key_here

# Security Settings
SECRET_KEY=your_secret_key_for_sessions
JWT_SECRET=your_jwt_secret_here

# Logging
LOG_LEVEL=INFO
SECURITY_LOG_LEVEL=WARNING

# Rate Limiting
API_RATE_LIMIT=100
API_BURST_LIMIT=200
```

### Security Headers Configuration
```python
# Flask/FastAPI security headers
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'",
    'Referrer-Policy': 'strict-origin-when-cross-origin'
}
```

## ⚖️ Compliance and Legal

### Data Protection
- **GDPR Compliance**: For EU users
- **Data Minimization**: Collect only necessary data
- **Data Retention**: Implement retention policies
- **User Rights**: Provide data access and deletion

### Financial Regulations
- **Disclaimer**: Not financial advice
- **License Requirements**: Check local regulations
- **Data Usage**: Comply with exchange terms
- **Audit Trail**: Maintain transaction logs

## 🔄 Security Updates

### Staying Updated
- **Subscribe** to security advisories for dependencies
- **Monitor** CVE databases for relevant vulnerabilities
- **Review** security patches before applying
- **Test** updates in staging environment first

### Version Control Security
```bash
# Sign commits
git config --global user.signingkey YOUR_GPG_KEY
git config --global commit.gpgsign true

# Verify signatures
git log --show-signature
```

## 🆘 Emergency Contacts

### Security Team
- **Primary Contact**: 1998prakhargupta@gmail.com
- **Backup Contact**: 1998prakhargupta@gmail.com

### Escalation Process
1. **Immediate**: Email security team
2. **Critical**: Call emergency hotline (if established)
3. **Follow-up**: Create private GitHub issue

---

## Summary

Security is a shared responsibility. By following these guidelines and best practices, we can maintain a secure environment for all users and contributors.

Remember:
- 🔐 **Never commit secrets**
- 🔍 **Validate all inputs**
- 📊 **Monitor for anomalies**
- 🔄 **Keep dependencies updated**
- 📧 **Report issues responsibly**

For questions about this security policy, contact: 1998prakhargupta@gmail.com
