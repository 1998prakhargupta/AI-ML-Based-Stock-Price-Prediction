# Contributing to Price Predictor Project

First off, thank you for considering contributing to the Price Predictor Project! It's people like you that make this project a great tool for stock price prediction and analysis.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.6+ 
- Git
- Virtual environment tool (venv, conda, etc.)
- Basic understanding of machine learning and financial markets

### Quick Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/1998prakhargupta/Major_Project.git
   cd Major_Project
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   make install
   # Or manually: pip install -r requirements.txt -r requirements-dev.txt
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to ensure everything works**
   ```bash
   make test
   ```

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 📖 **Documentation**: Improve or add documentation
- 💻 **Code Contributions**: Bug fixes, new features, optimizations
- 🧪 **Testing**: Add or improve test coverage
- 🎨 **UI/UX**: Improve visualization and user experience
- 📊 **Data Sources**: Add new financial data providers
- 🤖 **ML Models**: Implement new prediction algorithms

### Contribution Workflow

1. **Check existing issues** to see if your contribution is already being worked on
2. **Create an issue** for substantial changes to discuss the approach
3. **Fork and create a branch** for your contribution
4. **Make your changes** following our coding standards
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request** with a clear description

## Development Setup

### Environment Configuration

1. **Copy environment template**
   ```bash
   cp .env.example .env
   ```

2. **Configure API credentials** (for testing)
   ```bash
   # Edit .env with your test API credentials
   # Never commit real credentials!
   nano .env
   ```

3. **Set up database** (if needed)
   ```bash
   # Run setup script
   python scripts/setup.py
   ```

### Development Commands

```bash
# Install dependencies
make install

# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-compliance

# Code quality checks
make lint
make format

# Generate documentation
make docs

# Run compliance checks
make compliance
```

## Pull Request Process

### Before Submitting

1. ✅ **Tests pass**: `make test`
2. ✅ **Code quality**: `make lint`
3. ✅ **Documentation updated**: For new features
4. ✅ **Changelog updated**: For user-facing changes
5. ✅ **No merge conflicts**: Rebase if needed

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested this with real market data

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues
Fixes #(issue_number)
Closes #(issue_number)
Related to #(issue_number)
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** on different environments if applicable
4. **Documentation review** for user-facing changes
5. **Approval** by maintainer before merging

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 88 characters (Black formatter)
- **Import sorting**: Using isort with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all classes and functions

### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code
black .
isort .

# Check formatting
flake8
pylint src/

# Type checking
mypy src/
```

### Example Code Style

```python
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

class StockPredictor:
    """Stock price prediction model.
    
    This class provides functionality for training and predicting
    stock prices using various machine learning algorithms.
    
    Args:
        model_type: The type of ML model to use
        config: Configuration parameters
        
    Attributes:
        model: The trained model instance
        features: List of feature names
    """
    
    def __init__(
        self, 
        model_type: str, 
        config: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> None:
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.features: List[str] = []
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict[str, float]:
        """Train the prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary containing training metrics
            
        Raises:
            ValueError: If training data is invalid
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty")
            
        # Implementation here
        return {"mse": 0.0, "r2": 0.0}
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for system interactions
├── performance/       # Performance and benchmark tests
├── fixtures/          # Test data and fixtures
└── conftest.py       # Pytest configuration
```

### Writing Tests

1. **Test naming**: Use descriptive names that explain what is being tested
2. **Test isolation**: Each test should be independent
3. **Test data**: Use fixtures for consistent test data
4. **Assertions**: Use specific assertions with clear error messages
5. **Coverage**: Aim for >95% code coverage

### Example Test

```python
import pytest
import pandas as pd
from src.models.predictor import StockPredictor

class TestStockPredictor:
    """Test suite for StockPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return pd.DataFrame({
            'price': [100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300]
        })
    
    def test_train_with_valid_data(self, sample_data):
        """Test training with valid data returns metrics."""
        predictor = StockPredictor("random_forest")
        X = sample_data[['volume']]
        y = sample_data['price']
        
        metrics = predictor.train(X, y)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
    
    def test_train_with_empty_data_raises_error(self):
        """Test training with empty data raises ValueError."""
        predictor = StockPredictor("random_forest")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            predictor.train(empty_df, empty_series)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_predictor.py

# Run tests matching pattern
pytest -k "test_train"

# Run with verbose output
pytest -v
```

## Documentation

### Documentation Types

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step tutorials
3. **Developer Guides**: Technical implementation details
4. **README Files**: Quick start and overview

### Documentation Standards

- **Clear and concise**: Easy to understand
- **Examples**: Include code examples
- **Up-to-date**: Keep synchronized with code changes
- **Accessible**: Use simple language when possible

### Building Documentation

```bash
# Generate API documentation
make docs

# Serve documentation locally
cd docs && python -m http.server 8000
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Bug description**: Clear description of the issue
2. **Steps to reproduce**: Detailed steps to reproduce the bug
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Python version, package versions
6. **Logs**: Relevant error messages or logs
7. **Screenshots**: If applicable

### Feature Requests

For feature requests, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: Detailed description of the feature
3. **Alternatives**: Alternative solutions considered
4. **Additional context**: Any other relevant information

### Issue Templates

We provide issue templates to help you provide the necessary information:

- 🐛 **Bug Report Template**
- ✨ **Feature Request Template**
- 📖 **Documentation Template**
- ❓ **Question Template**

## Community

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For security-related issues

### Getting Help

1. **Check documentation**: Look in `docs/` directory
2. **Search existing issues**: Someone might have had the same problem
3. **Ask questions**: Create a GitHub discussion or issue
4. **Join community**: Participate in discussions and reviews

### Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Mention of significant contributions
- **GitHub**: Contributor graphs and statistics

## Development Best Practices

### Git Workflow

1. **Create feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make small, focused commits**
   ```bash
   git add .
   git commit -m "feat: add new prediction model"
   ```

3. **Keep branch up-to-date**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add support for new broker API
fix(models): resolve memory leak in LSTM model
docs: update installation instructions
test: add unit tests for data processors
```

### Security Considerations

- **Never commit credentials**: Use environment variables
- **Validate inputs**: Always validate user inputs
- **Follow security best practices**: Use secure coding practices
- **Report security issues**: Email maintainers for security vulnerabilities

## Getting Recognition

### Ways to Get Involved

1. **Start small**: Fix typos, improve documentation
2. **Help others**: Answer questions in issues and discussions
3. **Review PRs**: Provide constructive feedback on pull requests
4. **Contribute features**: Implement new functionality
5. **Improve tests**: Add test coverage and performance tests

### Contributor Levels

- **Contributor**: Anyone who contributes to the project
- **Regular Contributor**: Active contributors with multiple merged PRs
- **Maintainer**: Trusted contributors with commit access
- **Core Team**: Project leaders and primary maintainers

Thank you for contributing to the Price Predictor Project! Your contributions help make financial analysis more accessible and accurate for everyone.

---

## Quick Reference

### Essential Commands
```bash
# Setup
make setup
make install

# Development
make test
make lint
make format

# Documentation
make docs

# Clean up
make clean
```

### Important Files
- `src/`: Source code
- `tests/`: Test files
- `docs/`: Documentation
- `configs/`: Configuration files
- `.env.example`: Environment template
- `requirements.txt`: Dependencies
- `Makefile`: Build commands

### Need Help?
- 📖 Check the [documentation](docs/)
- 🐛 [Report a bug](../../issues/new?template=bug_report.md)
- ✨ [Request a feature](../../issues/new?template=feature_request.md)
- ❓ [Ask a question](../../discussions)
