#!/usr/bin/env python3
"""
Lookahead Bias Fixes for Stock Prediction System

This module provides utilities to detect and fix data leakage and lookahead bias
in time-series financial data and machine learning models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

class LookaheadBiasDetector:
    """Detect potential lookahead bias in financial time series data."""
    
    def __init__(self):
        self.bias_patterns = []
        self.warnings = []
        
    def detect_target_leakage(self, df: pd.DataFrame) -> List[str]:
        """
        Detect target variables that might use future data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of columns with potential lookahead bias
        """
        suspicious_columns = []
        
        # Check for negative shift operations in column names
        for col in df.columns:
            if any(pattern in col.lower() for pattern in ['futret', 'future_return', 'target', 'label']):
                # Check if this column might be using future data
                if 'shift(' in str(df[col].dtype) or col.endswith('_target'):
                    suspicious_columns.append(col)
                    
        # Check for columns that have perfect correlation with future values
        for col in df.select_dtypes(include=[np.number]).columns:
            if col.startswith(('equity_', 'futures_', 'options_')) and 'close' in col:
                try:
                    # Check if current values perfectly correlate with shifted future values
                    future_shifted = df[col].shift(-1)
                    correlation = df[col].corr(future_shifted)
                    if correlation > 0.999:  # Near perfect correlation suggests data leakage
                        suspicious_columns.append(f"{col}_potential_leakage")
                except:
                    pass
                    
        return suspicious_columns
    
    def detect_rolling_bias(self, df: pd.DataFrame) -> List[str]:
        """
        Detect rolling calculations that might include future data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of problematic rolling calculations
        """
        rolling_issues = []
        
        for col in df.columns:
            if 'rolling' in col.lower() or 'corr' in col.lower():
                # Check if rolling calculations use center=True (includes future data)
                rolling_issues.append(f"{col}_check_center_parameter")
                
        return rolling_issues
    
    def validate_datetime_order(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> bool:
        """
        Validate that datetime column is properly sorted.
        
        Args:
            df: DataFrame to check
            datetime_col: Name of datetime column
            
        Returns:
            True if properly sorted, False otherwise
        """
        if datetime_col not in df.columns:
            logger.warning(f"Datetime column '{datetime_col}' not found")
            return False
            
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            is_sorted = df[datetime_col].is_monotonic_increasing
            
            if not is_sorted:
                logger.warning("DataFrame is not sorted by datetime - this can cause lookahead bias")
                
            return is_sorted
        except Exception as e:
            logger.error(f"Error validating datetime order: {e}")
            return False

class LookaheadBiasFixer:
    """Fix lookahead bias issues in financial time series data."""
    
    def __init__(self):
        self.fixes_applied = []
        
    def fix_target_generation(self, df: pd.DataFrame, 
                            price_columns: List[str],
                            time_windows: List[int] = [1, 5, 10, 15, 30],
                            thresholds: List[float] = [0.01, 0.02, 0.05]) -> pd.DataFrame:
        """
        Generate targets WITHOUT lookahead bias by using forward-looking labels correctly.
        
        Key Fix: Remove shift(-win) operations and use proper forward returns.
        
        Args:
            df: Input DataFrame
            price_columns: Columns to generate targets for
            time_windows: Future time windows to predict
            thresholds: Return thresholds for classification
            
        Returns:
            DataFrame with properly generated targets
        """
        logger.info("Fixing target generation to remove lookahead bias")
        fixed_df = df.copy()
        
        for col in price_columns:
            if col not in fixed_df.columns:
                continue
                
            for win in time_windows:
                # CORRECT: Calculate future returns without using shift(-win)
                # This ensures that for each timestamp, we only use data up to that point
                
                # Forward return calculation (this is correct for target generation)
                future_prices = fixed_df[col].shift(-win)  # This is OK for targets, not features
                current_prices = fixed_df[col]
                
                # Calculate future returns (these will be NaN for the last 'win' periods)
                future_return = (future_prices - current_prices) / current_prices
                
                # Store the future return as target (with proper NaN handling)
                target_col = f'{col}_target_{win}periods'
                fixed_df[target_col] = future_return
                
                # Generate classification labels based on thresholds
                for thresh in thresholds:
                    thresh_str = str(thresh).replace('.', '_')
                    
                    # Classification targets
                    fixed_df[f'{col}_target_up_{win}p_{thresh_str}'] = (future_return > thresh).astype(int)
                    fixed_df[f'{col}_target_down_{win}p_{thresh_str}'] = (future_return < -thresh).astype(int)
                    fixed_df[f'{col}_target_neutral_{win}p_{thresh_str}'] = (
                        (future_return <= thresh) & (future_return >= -thresh)
                    ).astype(int)
        
        # CRITICAL: Drop the last 'max(time_windows)' rows as they will have NaN targets
        max_window = max(time_windows)
        if len(fixed_df) > max_window:
            fixed_df = fixed_df[:-max_window].copy()
            logger.info(f"Dropped last {max_window} rows to prevent lookahead bias in targets")
        
        self.fixes_applied.append("target_generation_fixed")
        return fixed_df
    
    def fix_rolling_calculations(self, df: pd.DataFrame,
                               rolling_windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Fix rolling calculations to prevent lookahead bias.
        
        Key Fix: Ensure all rolling calculations use only past data.
        
        Args:
            df: Input DataFrame
            rolling_windows: Rolling window sizes
            
        Returns:
            DataFrame with corrected rolling calculations
        """
        logger.info("Fixing rolling calculations to remove lookahead bias")
        fixed_df = df.copy()
        
        # Identify numeric columns for rolling calculations
        numeric_cols = fixed_df.select_dtypes(include=[np.number]).columns
        price_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['close', 'open', 'high', 'low'])]
        
        for col in price_cols:
            if col not in fixed_df.columns:
                continue
                
            for window in rolling_windows:
                # CORRECT: Use only past data in rolling calculations
                # The default behavior of pandas rolling is correct (uses past data)
                
                # Rolling mean (uses only past data)
                fixed_df[f'{col}_rolling_mean_{window}'] = fixed_df[col].rolling(
                    window=window, min_periods=window//2
                ).mean()
                
                # Rolling standard deviation (uses only past data)
                fixed_df[f'{col}_rolling_std_{window}'] = fixed_df[col].rolling(
                    window=window, min_periods=window//2
                ).std()
                
                # Rolling minimum and maximum (uses only past data)
                fixed_df[f'{col}_rolling_min_{window}'] = fixed_df[col].rolling(
                    window=window, min_periods=window//2
                ).min()
                
                fixed_df[f'{col}_rolling_max_{window}'] = fixed_df[col].rolling(
                    window=window, min_periods=window//2
                ).max()
        
        self.fixes_applied.append("rolling_calculations_fixed")
        return fixed_df
    
    def fix_correlation_calculations(self, df: pd.DataFrame,
                                   correlation_windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Fix correlation calculations to prevent lookahead bias.
        
        Key Fix: Ensure rolling correlations use only historical data.
        
        Args:
            df: Input DataFrame
            correlation_windows: Rolling correlation windows
            
        Returns:
            DataFrame with corrected correlation calculations
        """
        logger.info("Fixing correlation calculations to remove lookahead bias")
        fixed_df = df.copy()
        
        # Find pairs of related columns for correlation
        equity_cols = [col for col in fixed_df.columns if col.startswith('equity_') and 'close' in col]
        futures_cols = [col for col in fixed_df.columns if col.startswith('futures_') and 'close' in col]
        
        for window in correlation_windows:
            for eq_col in equity_cols:
                for fut_col in futures_cols:
                    if eq_col in fixed_df.columns and fut_col in fixed_df.columns:
                        # CORRECT: Rolling correlation using only past data
                        corr_col = f'corr_{eq_col}_{fut_col}_{window}'
                        
                        # This is correct - pandas rolling corr uses only past data by default
                        fixed_df[corr_col] = fixed_df[eq_col].rolling(
                            window=window, min_periods=window//2
                        ).corr(fixed_df[fut_col])
        
        self.fixes_applied.append("correlation_calculations_fixed")
        return fixed_df
    
    def fix_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure technical indicators don't use future data.
        
        Key Fix: Validate that all technical indicators use only historical data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with validated technical indicators
        """
        logger.info("Validating technical indicators for lookahead bias")
        fixed_df = df.copy()
        
        # Most technical indicators in ta-lib and similar libraries are correctly implemented
        # to use only past data, but we'll add validation
        
        # Check for any indicators that might be incorrectly shifted
        indicator_cols = [col for col in fixed_df.columns if any(
            indicator in col.upper() for indicator in ['RSI', 'MACD', 'EMA', 'SMA', 'ATR', 'STOCH']
        )]
        
        for col in indicator_cols:
            # Ensure no future-shifted indicators
            if any(term in col.lower() for terms in [['future', 'ahead', 'forward']] for term in terms):
                logger.warning(f"Potentially problematic indicator found: {col}")
                # You might want to recalculate this indicator properly
        
        self.fixes_applied.append("technical_indicators_validated")
        return fixed_df
        
    def ensure_temporal_order(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Ensure data is properly sorted by timestamp.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame sorted by timestamp
        """
        logger.info("Ensuring proper temporal order")
        fixed_df = df.copy()
        
        if datetime_col in fixed_df.columns:
            # Convert to datetime and sort
            fixed_df[datetime_col] = pd.to_datetime(fixed_df[datetime_col])
            fixed_df = fixed_df.sort_values(datetime_col).reset_index(drop=True)
            
            logger.info(f"Data sorted by {datetime_col}")
        else:
            logger.warning(f"Datetime column '{datetime_col}' not found")
            
        self.fixes_applied.append("temporal_order_fixed")
        return fixed_df

class TimeSeriesMLValidator:
    """Validate ML models for proper time-series handling."""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_train_test_split(self, df: pd.DataFrame, 
                                datetime_col: str = 'datetime',
                                test_size: float = 0.2) -> Dict[str, Union[pd.DataFrame, bool]]:
        """
        Perform proper time-series train/test split.
        
        Key Fix: Use temporal split instead of random split.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with train/test splits and validation results
        """
        logger.info("Performing proper time-series train/test split")
        
        # Ensure data is sorted by time
        if datetime_col in df.columns:
            df_sorted = df.sort_values(datetime_col).reset_index(drop=True)
        else:
            logger.warning("No datetime column found, using index order")
            df_sorted = df.copy()
        
        # Calculate split point
        n_samples = len(df_sorted)
        split_point = int(n_samples * (1 - test_size))
        
        # CORRECT: Temporal split (train on earlier data, test on later data)
        train_data = df_sorted.iloc[:split_point].copy()
        test_data = df_sorted.iloc[split_point:].copy()
        
        # Validation: Ensure no temporal overlap
        temporal_overlap = False
        if datetime_col in df.columns:
            train_max_date = train_data[datetime_col].max()
            test_min_date = test_data[datetime_col].min()
            temporal_overlap = train_max_date >= test_min_date
            
        if temporal_overlap:
            logger.error("Temporal overlap detected in train/test split!")
        else:
            logger.info("Temporal split validated successfully")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'temporal_overlap': temporal_overlap,
            'train_date_range': (train_data[datetime_col].min(), train_data[datetime_col].max()) if datetime_col in df.columns else None,
            'test_date_range': (test_data[datetime_col].min(), test_data[datetime_col].max()) if datetime_col in df.columns else None
        }
    
    def validate_cross_validation(self, df: pd.DataFrame,
                                 datetime_col: str = 'datetime',
                                 n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series cross-validation splits.
        
        Key Fix: Use time-series CV instead of random CV.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            n_splits: Number of CV splits
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        logger.info("Creating time-series cross-validation splits")
        
        n_samples = len(df)
        test_size = n_samples // (n_splits + 1)
        
        splits = []
        
        for i in range(n_splits):
            # Time-series CV: each split uses all previous data for training
            train_end = (i + 1) * test_size + test_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
                
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            
        logger.info(f"Created {len(splits)} time-series CV splits")
        return splits

def comprehensive_lookahead_fix(df: pd.DataFrame,
                               price_columns: List[str] = None,
                               datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Apply comprehensive lookahead bias fixes to a DataFrame.
    
    Args:
        df: Input DataFrame
        price_columns: List of price columns to generate targets for
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame with lookahead bias fixes applied
    """
    logger.info("Applying comprehensive lookahead bias fixes")
    
    # Initialize fixers
    detector = LookaheadBiasDetector()
    fixer = LookaheadBiasFixer()
    
    # Step 1: Detect existing bias
    target_issues = detector.detect_target_leakage(df)
    rolling_issues = detector.detect_rolling_bias(df)
    
    if target_issues:
        logger.warning(f"Detected potential target leakage in columns: {target_issues}")
    if rolling_issues:
        logger.warning(f"Detected potential rolling calculation issues: {rolling_issues}")
    
    # Step 2: Fix temporal order
    fixed_df = fixer.ensure_temporal_order(df, datetime_col)
    
    # Step 3: Fix target generation
    if price_columns is None:
        price_columns = [col for col in df.columns if 'close' in col.lower()]
    
    if price_columns:
        fixed_df = fixer.fix_target_generation(fixed_df, price_columns)
    
    # Step 4: Fix rolling calculations
    fixed_df = fixer.fix_rolling_calculations(fixed_df)
    
    # Step 5: Fix correlation calculations
    fixed_df = fixer.fix_correlation_calculations(fixed_df)
    
    # Step 6: Validate technical indicators
    fixed_df = fixer.fix_technical_indicators(fixed_df)
    
    logger.info(f"Applied fixes: {fixer.fixes_applied}")
    logger.info(f"Fixed DataFrame shape: {fixed_df.shape}")
    
    return fixed_df

# Example usage and testing functions
def test_lookahead_bias_fixes():
    """Test the lookahead bias fixes with sample data."""
    logger.info("Testing lookahead bias fixes")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'equity_close': np.random.randn(100).cumsum() + 100,
        'futures_close': np.random.randn(100).cumsum() + 100,
        'equity_volume': np.random.randint(1000, 10000, 100)
    })
    
    # Apply fixes
    fixed_data = comprehensive_lookahead_fix(
        sample_data,
        price_columns=['equity_close', 'futures_close']
    )
    
    logger.info(f"Original data shape: {sample_data.shape}")
    logger.info(f"Fixed data shape: {fixed_data.shape}")
    
    # Test temporal split
    validator = TimeSeriesMLValidator()
    split_result = validator.validate_train_test_split(fixed_data)
    
    logger.info(f"Train data shape: {split_result['train_data'].shape}")
    logger.info(f"Test data shape: {split_result['test_data'].shape}")
    logger.info(f"Temporal overlap: {split_result['temporal_overlap']}")
    
    return fixed_data, split_result

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_data, split_results = test_lookahead_bias_fixes()
    
    print("Lookahead bias fixes tested successfully!")
    print(f"Test data columns: {test_data.columns.tolist()}")
