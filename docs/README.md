# Stock Price Prediction Project

This project implements a comprehensive stock price prediction system using multiple data sources and machine learning models.

## Security Setup

**IMPORTANT**: Never commit real API credentials to version control.

### Method 1: Environment Variables (Recommended)
Set the following environment variables in your system:

```bash
export BREEZE_API_KEY="your_actual_api_key"
export BREEZE_API_SECRET="your_actual_api_secret" 
export BREEZE_SESSION_TOKEN="your_actual_session_token"
export DATA_SAVE_PATH="/your/preferred/data/path"
```

### Method 2: Local Config File
1. Copy `config.json` to `config.local.json`
2. Edit `config.local.json` with your real credentials
3. Add `config.local.json` to `.gitignore`

```json
{
  "BREEZE_API_KEY": "your_actual_api_key",
  "BREEZE_API_SECRET": "your_actual_api_secret",
  "BREEZE_SESSION_TOKEN": "your_actual_session_token"
}
```

## File Structure

### Core Files
- `config.py` - Secure configuration management system
- `config.json` - Template configuration file (DO NOT commit real credentials)
- `.gitignore` - Protects sensitive files from version control

### Utility Modules
- `breeze_utils.py` - Secure Breeze API data fetching and processing with BreezeDataManager class
- `index_utils.py` - NSE index data management using yfinance with IndexDataManager class
- `model_utils.py` - Machine learning utilities with ModelDataProcessor, ModelEvaluator, and ModelManager classes

### Notebooks
- `breeze_data.ipynb` - Secure Breeze API data collection with modular utilities
- `index_data_fetch.ipynb` - NSE index data fetching, analysis, and correlation studies
- `stock_ML_Model.ipynb` - Multi-asset ensemble prediction models with comprehensive evaluation

## Key Features

### Security & Configuration
- ✅ **Environment variable support** with automatic fallback to config files
- ✅ **Credential validation** and secure storage
- ✅ **Git protection** with comprehensive .gitignore
- ✅ **Template-based configuration** to prevent credential exposure

### Data Management
- ✅ **Modular data managers** for different data sources
- ✅ **Automatic directory management** for organized data storage
- ✅ **Memory-efficient processing** for large datasets
- ✅ **Comprehensive error handling** and logging

### Machine Learning
- ✅ **Professional ML utilities** with feature engineering, scaling, and evaluation
- ✅ **Model management** with saving/loading capabilities
- ✅ **Ensemble methods** with intelligent weight optimization
- ✅ **Visualization tools** for model comparison and analysis

## Usage

1. Set up your credentials using one of the methods above
2. Run the notebooks in order:
   - `breeze_data.ipynb` for stock/options/futures data
   - `index_data_fetch.ipynb` for market index data
   - `stock_ML_Model.ipynb` for model training and prediction

## Dependencies

Install required packages:
```bash
pip install pandas numpy ta plotly breeze_connect yfinance scikit-learn xgboost lightgbm tensorflow
```

## Security Notes

- Never commit real API keys to git
- Use environment variables in production
- Keep local config files in .gitignore
- Regularly rotate API keys
