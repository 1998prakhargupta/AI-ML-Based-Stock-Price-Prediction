# Data Validation Framework - Quick Reference Guide

## 🚀 Quick Start

### 1. Basic Usage Pattern

```python
# Import validation utilities (from notebook cell 1)
from validation_utils import (
    validate_dataframe_structure, 
    ensure_numeric_columns, 
    validate_datetime_column,
    safe_column_operation
)

# Standard validation pipeline
def process_data_safely(df, target_col='close'):
    # Step 1: Validate structure
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime']
    df = validate_dataframe_structure(df, required_cols)
    
    # Step 2: Validate datetime
    df = validate_datetime_column(df, 'datetime')
    
    # Step 3: Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df = ensure_numeric_columns(df, numeric_cols)
    
    # Step 4: Safe operations
    result = safe_column_operation(df, your_analysis_function)
    
    return result
```

## 📋 Validation Functions Reference

### Core Functions

#### `validate_dataframe_structure(df, required_columns, optional_columns=None)`
**Purpose**: Ensures all required columns exist, adds missing ones with NaN
**Use Case**: Before any column-dependent operations

```python
# Example
df = validate_dataframe_structure(
    df, 
    required_columns=['open', 'high', 'low', 'close'],
    optional_columns=['volume', 'open_interest']
)
```

#### `ensure_numeric_columns(df, columns, fill_method='forward')`
**Purpose**: Converts columns to numeric types with error handling
**Use Case**: Before mathematical operations

```python
# Example
df = ensure_numeric_columns(
    df, 
    columns=['open', 'high', 'low', 'close', 'volume'],
    fill_method='forward'  # Options: 'forward', 'backward', 'zero', 'mean'
)
```

#### `validate_datetime_column(df, datetime_col='datetime')`
**Purpose**: Validates and formats datetime column with fallbacks
**Use Case**: Before time-series operations

```python
# Example
df = validate_datetime_column(df, 'datetime')
```

#### `safe_column_operation(df, operation_func, *args, **kwargs)`
**Purpose**: Safely performs operations with graceful error handling
**Use Case**: Wrapping complex operations that might fail

```python
# Example
def calculate_indicators(df):
    df['sma'] = df['close'].rolling(20).mean()
    return df

df = safe_column_operation(df, calculate_indicators)
```

## 🏷️ Data Type Specific Validation

### Equity Data Validation

```python
# Required columns for equity data
equity_required = ['open', 'high', 'low', 'close', 'volume', 'datetime']

def validate_equity_data(df):
    # Structure validation
    df = validate_dataframe_structure(df, equity_required)
    
    # Type validation
    df = ensure_numeric_columns(df, ['open', 'high', 'low', 'close', 'volume'])
    
    # OHLC logic validation
    def fix_ohlc_logic(data):
        for idx in data.index:
            o, h, l, c = data.loc[idx, ['open', 'high', 'low', 'close']]
            if not pd.isna([o, h, l, c]).any():
                data.loc[idx, 'high'] = max(o, h, l, c)
                data.loc[idx, 'low'] = min(o, h, l, c)
        return data
    
    df = safe_column_operation(df, fix_ohlc_logic)
    return df
```

### Futures Data Validation

```python
def validate_futures_data(df):
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime']
    optional_cols = ['open_interest', 'oi']
    
    df = validate_dataframe_structure(df, required_cols, optional_cols)
    df = ensure_numeric_columns(df, required_cols[:-1])  # Exclude datetime
    
    # Futures-specific validations
    if 'volume' in df.columns:
        # Replace zero/negative volumes
        df.loc[df['volume'] <= 0, 'volume'] = 1
    
    if 'open_interest' in df.columns:
        # Ensure non-negative OI
        df.loc[df['open_interest'] < 0, 'open_interest'] = 0
    
    return df
```

### Options Data Validation

```python
def validate_options_data(df):
    required_cols = ['strike', 'option_type', 'premium', 'datetime']
    optional_cols = ['delta', 'gamma', 'theta', 'vega', 'iv']
    
    df = validate_dataframe_structure(df, required_cols, optional_cols)
    
    # Options-specific validations
    if 'strike' in df.columns:
        df = df[df['strike'] > 0]  # Remove invalid strikes
    
    if 'option_type' in df.columns:
        # Standardize option types
        df['option_type'] = df['option_type'].str.upper()
        df['option_type'] = df['option_type'].replace({'CALL': 'CE', 'PUT': 'PE'})
        df = df[df['option_type'].isin(['CE', 'PE'])]
    
    if 'premium' in df.columns:
        df.loc[df['premium'] < 0, 'premium'] = 0  # Fix negative premiums
    
    # Greeks validation
    if 'delta' in df.columns:
        df.loc[(df['delta'] < -1) | (df['delta'] > 1), 'delta'] = np.nan
    
    return df
```

## ⚡ Common Usage Patterns

### Pattern 1: Data Loading with Validation

```python
def load_and_validate_data(file_path, data_type='equity'):
    try:
        # Load data
        df = pd.read_parquet(file_path)
        
        # Apply appropriate validation
        if data_type == 'equity':
            df = validate_equity_data(df)
        elif data_type == 'futures':
            df = validate_futures_data(df)
        elif data_type == 'options':
            df = validate_options_data(df)
        
        # Quality assessment
        quality_score = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f"Data loaded and validated: {df.shape}, Quality: {quality_score:.1f}%")
        
        return df
        
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return None
```

### Pattern 2: ML Data Preparation with Validation

```python
def prepare_ml_data(df, target_col='close'):
    # Validate for ML
    df = validate_ml_dataframe(df, target_column=target_col)
    
    # Feature preparation
    features, target = safe_feature_preparation(df, target_col)
    
    if features is None:
        raise ValueError("Feature preparation failed")
    
    # Validate features and target
    X, y, feature_names = validate_ml_features(features, target)
    
    return X, y, feature_names
```

### Pattern 3: Safe Model Training

```python
def train_model_safely(model_class, X, y, **model_params):
    def train_function(features, targets):
        model = model_class(**model_params)
        model.fit(features, targets)
        return model
    
    return safe_model_training(train_function, X, y)
```

## 🚨 Error Handling Best Practices

### 1. Always Use Try-Except with Validation

```python
def process_data(df):
    try:
        # Validation first
        df = validate_dataframe_structure(df, required_columns)
        
        # Your processing logic here
        result = df.groupby('symbol').agg({'close': 'mean'})
        
        return result
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return None  # Graceful degradation
```

### 2. Implement Fallback Strategies

```python
def get_target_column(df, preferred='close'):
    # Primary choice
    if preferred in df.columns:
        return preferred
    
    # Fallback options
    close_cols = [col for col in df.columns if 'close' in col.lower()]
    if close_cols:
        logger.warning(f"Using {close_cols[0]} instead of {preferred}")
        return close_cols[0]
    
    # Last resort
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.warning(f"Using {numeric_cols[0]} as fallback target")
        return numeric_cols[0]
    
    raise ValueError("No suitable target column found")
```

### 3. Comprehensive Logging

```python
import logging

def setup_validation_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('validation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('validation')

logger = setup_validation_logging()
```

## 📊 Quality Metrics and Monitoring

### Calculate Data Quality Score

```python
def calculate_data_quality(df):
    metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'completeness': (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100,
        'consistency': check_ohlc_consistency(df) if all(col in df.columns for col in ['open', 'high', 'low', 'close']) else 100,
        'validity': check_data_validity(df)
    }
    
    overall_score = (metrics['completeness'] + metrics['consistency'] + metrics['validity']) / 3
    metrics['overall_score'] = overall_score
    
    return metrics
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

1. **"Column not found" errors**
   - Solution: Use `validate_dataframe_structure()` before accessing columns

2. **Type conversion errors**
   - Solution: Use `ensure_numeric_columns()` before mathematical operations

3. **Invalid OHLC relationships**
   - Solution: Implement OHLC logic validation in your pipeline

4. **Memory issues with large datasets**
   - Solution: Use chunked processing with validation applied to each chunk

5. **Performance degradation**
   - Solution: Apply validation only to critical operations, cache validation results

### Debug Mode

```python
def debug_validation(df, verbose=True):
    if verbose:
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {dict(df.dtypes)}")
        print(f"Missing values: {df.isna().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
    
    return df
```

## 📝 Migration Checklist

### Converting Existing Notebooks

- [ ] Add validation framework imports
- [ ] Replace direct column access with validated access
- [ ] Add error handling around data operations
- [ ] Implement quality assessment and logging
- [ ] Test with various data scenarios
- [ ] Update documentation and comments

### New Notebook Template

```python
# Cell 1: Validation Framework
# (Copy validation utilities from breeze_data.ipynb Cell 1)

# Cell 2: Data Loading with Validation
df = load_and_validate_data('data.parquet', 'equity')

# Cell 3: Data Processing with Validation
processed_df = process_data_safely(df)

# Cell 4: Quality Assessment
quality_metrics = calculate_data_quality(processed_df)
print(f"Data quality: {quality_metrics['overall_score']:.1f}%")
```

---

## 📞 Support and Resources

- **Full Implementation**: See `DATA_VALIDATION_IMPLEMENTATION_COMPLETE.md`
- **Demo Script**: Run `validation_demo.py` for examples
- **Test Framework**: Use `test_validation_framework.py` for validation
- **Log Files**: Check validation logs for debugging information

**Remember**: The validation framework is designed to be fail-safe. When in doubt, it chooses safety and graceful degradation over crashes.
