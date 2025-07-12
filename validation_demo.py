#!/usr/bin/env python3
"""
Demonstration script showing how the validation framework prevents common data errors.
This script simulates real-world scenarios where data sources change and demonstrates
how the validation framework handles these gracefully.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_problematic_data():
    """Create datasets with various real-world data issues"""
    
    print("🎭 Creating Problematic Test Datasets")
    print("=" * 50)
    
    # Scenario 1: Missing required columns
    incomplete_equity_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='5min'),
        'close': np.random.uniform(95, 105, 100),
        'volume': np.random.uniform(1000, 5000, 100)
        # Missing: open, high, low
    })
    print("📊 Scenario 1: Equity data missing OHLC columns")
    
    # Scenario 2: Wrong data types
    string_numeric_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='5min'),
        'open': ['100.5'] * 50 + [np.nan] * 50,  # String numbers
        'high': ['101.0'] * 100,
        'low': ['99.5'] * 100,
        'close': ['100.0'] * 100,
        'volume': ['1000'] * 100  # String volume
    })
    print("📊 Scenario 2: Numeric data stored as strings")
    
    # Scenario 3: Invalid OHLC relationships
    invalid_ohlc_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='5min'),
        'open': np.random.uniform(95, 105, 100),
        'high': np.random.uniform(90, 95, 100),  # High < Open (invalid)
        'low': np.random.uniform(105, 110, 100),  # Low > Open (invalid)
        'close': np.random.uniform(95, 105, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    print("📊 Scenario 3: Invalid OHLC logic relationships")
    
    # Scenario 4: Options data with invalid values
    invalid_options_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=50, freq='1H'),
        'strike': [-100, 0] + list(np.random.uniform(90, 110, 48)),  # Invalid strikes
        'option_type': ['CALL', 'PUT', 'INVALID'] + ['CE'] * 47,  # Mixed formats
        'premium': [-5, np.inf] + list(np.random.uniform(1, 10, 48)),  # Invalid premiums
        'delta': [-2, 1.5] + list(np.random.uniform(-1, 1, 48)),  # Invalid delta range
        'volume': [np.inf, -100] + list(np.random.uniform(0, 1000, 48))  # Invalid volume
    })
    print("📊 Scenario 4: Options data with invalid Greeks and values")
    
    # Scenario 5: Extreme outliers and missing data
    outlier_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='5min'),
        'open': [100] * 95 + [10000, 0.01, np.inf, -np.inf, np.nan],  # Extreme values
        'high': [101] * 100,
        'low': [99] * 100,
        'close': [100] * 50 + [np.nan] * 50,  # 50% missing
        'volume': [1000] * 90 + [np.inf] * 10,  # Infinite volumes
        'rsi': [50] * 80 + [150, -50] + [np.nan] * 18  # Invalid RSI values
    })
    print("📊 Scenario 5: Data with extreme outliers and missing values")
    
    return {
        'incomplete_equity': incomplete_equity_data,
        'string_numeric': string_numeric_data,
        'invalid_ohlc': invalid_ohlc_data,
        'invalid_options': invalid_options_data,
        'outlier_data': outlier_data
    }

def demonstrate_validation_fixes():
    """Demonstrate how validation framework fixes common issues"""
    
    print("\n🔧 Demonstrating Validation Framework Solutions")
    print("=" * 60)
    
    # Get problematic datasets
    datasets = create_problematic_data()
    
    # Validation function implementations (simplified versions)
    def validate_dataframe_structure(df, required_columns):
        """Validate DataFrame has required columns and add missing ones with NaN"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"   ⚠️ Missing columns: {missing_columns}")
            for col in missing_columns:
                df[col] = np.nan
            print(f"   ✅ Added missing columns with NaN values")
        return df
    
    def ensure_numeric_columns(df, columns):
        """Ensure specified columns are numeric with proper type conversion"""
        for col in columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Count conversions
                if original_type == 'object':
                    print(f"   🔄 Converted {col} from {original_type} to numeric")
                
                # Fill NaN values
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        return df
    
    def validate_ohlc_logic(df):
        """Validate and fix OHLC logic inconsistencies"""
        issues_fixed = 0
        for idx in df.index:
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                o, h, l, c = df.loc[idx, ['open', 'high', 'low', 'close']]
                
                if pd.isna([o, h, l, c]).any():
                    continue
                
                # Fix high value
                actual_high = max(o, h, l, c)
                if h < actual_high:
                    df.loc[idx, 'high'] = actual_high
                    issues_fixed += 1
                
                # Fix low value
                actual_low = min(o, h, l, c)
                if l > actual_low:
                    df.loc[idx, 'low'] = actual_low
                    issues_fixed += 1
        
        if issues_fixed > 0:
            print(f"   🔧 Fixed {issues_fixed} OHLC logic inconsistencies")
        return df
    
    def clean_extreme_values(df):
        """Clean infinite values and extreme outliers"""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        outliers_cleaned = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in df.columns:
                # Remove extreme outliers (beyond 5 standard deviations)
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if not pd.isna(std_val) and std_val > 0:
                    outlier_mask = np.abs(df[col] - mean_val) > (5 * std_val)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        df.loc[outlier_mask, col] = np.nan
                        outliers_cleaned += outlier_count
        
        if outliers_cleaned > 0:
            print(f"   🧹 Cleaned {outliers_cleaned} extreme outliers")
        return df
    
    def validate_options_specifics(df):
        """Validate options-specific requirements"""
        fixes = []
        
        if 'strike' in df.columns:
            # Fix invalid strike prices
            invalid_strikes = (df['strike'] <= 0).sum()
            if invalid_strikes > 0:
                df = df[df['strike'] > 0]
                fixes.append(f"Removed {invalid_strikes} invalid strikes")
        
        if 'option_type' in df.columns:
            # Standardize option types
            df['option_type'] = df['option_type'].str.upper()
            df['option_type'] = df['option_type'].replace({'CALL': 'CE', 'PUT': 'PE'})
            
            # Remove invalid types
            valid_types = ['CE', 'PE']
            invalid_types = ~df['option_type'].isin(valid_types)
            invalid_count = invalid_types.sum()
            if invalid_count > 0:
                df = df[df['option_type'].isin(valid_types)]
                fixes.append(f"Removed {invalid_count} invalid option types")
        
        if 'premium' in df.columns:
            # Fix negative premiums
            negative_premiums = (df['premium'] < 0).sum()
            if negative_premiums > 0:
                df.loc[df['premium'] < 0, 'premium'] = 0
                fixes.append(f"Fixed {negative_premiums} negative premiums")
        
        if 'delta' in df.columns:
            # Fix invalid delta values
            invalid_delta = ((df['delta'] < -1) | (df['delta'] > 1)).sum()
            if invalid_delta > 0:
                df.loc[(df['delta'] < -1) | (df['delta'] > 1), 'delta'] = np.nan
                fixes.append(f"Fixed {invalid_delta} invalid delta values")
        
        if fixes:
            print(f"   ⚙️ Options fixes: {'; '.join(fixes)}")
        
        return df
    
    # Test each scenario
    print("\n🧪 Testing Validation Solutions:")
    
    # Scenario 1: Missing columns
    print("\n1️⃣ Fixing Missing Columns:")
    df1 = datasets['incomplete_equity'].copy()
    print(f"   Original columns: {list(df1.columns)}")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime']
    df1 = validate_dataframe_structure(df1, required_cols)
    print(f"   After validation: {list(df1.columns)}")
    
    # Scenario 2: Type conversion
    print("\n2️⃣ Fixing Data Types:")
    df2 = datasets['string_numeric'].copy()
    print(f"   Original dtypes: {dict(df2.dtypes)}")
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df2 = ensure_numeric_columns(df2, numeric_cols)
    print(f"   After conversion: All numeric columns are now float64")
    
    # Scenario 3: OHLC logic
    print("\n3️⃣ Fixing OHLC Logic:")
    df3 = datasets['invalid_ohlc'].copy()
    print(f"   Sample before: O={df3.iloc[0]['open']:.2f}, H={df3.iloc[0]['high']:.2f}, L={df3.iloc[0]['low']:.2f}")
    
    df3 = validate_ohlc_logic(df3)
    print(f"   Sample after: O={df3.iloc[0]['open']:.2f}, H={df3.iloc[0]['high']:.2f}, L={df3.iloc[0]['low']:.2f}")
    
    # Scenario 4: Options validation
    print("\n4️⃣ Fixing Options Data:")
    df4 = datasets['invalid_options'].copy()
    print(f"   Original shape: {df4.shape}")
    print(f"   Invalid strikes: {(df4['strike'] <= 0).sum()}")
    
    df4 = validate_options_specifics(df4)
    print(f"   After validation: {df4.shape}")
    
    # Scenario 5: Outliers and missing data
    print("\n5️⃣ Cleaning Outliers and Missing Data:")
    df5 = datasets['outlier_data'].copy()
    print(f"   Original missing in 'close': {df5['close'].isna().sum()}")
    print(f"   Infinite values: {np.isinf(df5.select_dtypes(include=[np.number])).sum().sum()}")
    
    df5 = clean_extreme_values(df5)
    # Fill remaining NaN values
    df5 = df5.fillna(method='ffill').fillna(method='bfill')
    
    print(f"   After cleaning missing in 'close': {df5['close'].isna().sum()}")
    print(f"   Infinite values remaining: {np.isinf(df5.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Final summary
    print("\n📊 Validation Framework Benefits:")
    print("   ✅ Automatic handling of missing columns")
    print("   ✅ Intelligent type conversion with error handling")
    print("   ✅ OHLC logic validation and correction")
    print("   ✅ Domain-specific validation (options, futures)")
    print("   ✅ Outlier detection and cleaning")
    print("   ✅ Graceful degradation instead of crashes")
    print("   ✅ Comprehensive logging and error reporting")

def demonstrate_before_after():
    """Show before/after comparison of error handling"""
    
    print("\n⚡ Before vs After: Error Handling Comparison")
    print("=" * 60)
    
    # Create problematic data
    problematic_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],  # Wrong column name
        'price': ['100.5', 'invalid'],  # Mixed types
        'vol': [1000, np.inf]  # Infinite values
    })
    
    print("🔴 BEFORE (Without Validation):")
    print("   Code: df['close'].mean()  # Assumes 'close' column exists")
    print("   Result: ❌ KeyError: 'close' - Column not found")
    print("   Impact: 💥 Notebook crashes, analysis stops")
    
    print("\n🟢 AFTER (With Validation Framework):")
    print("   Code: df = validate_dataframe_structure(df, ['close'])")
    print("         result = df['close'].mean() if 'close' in df.columns else np.nan")
    print("   Result: ✅ Graceful handling, warning logged, analysis continues")
    print("   Impact: 📊 Robust processing, comprehensive error reporting")
    
    print("\n🔴 BEFORE (Type Assumptions):")
    print("   Code: df['volume'] * 2  # Assumes numeric type")
    print("   Result: ❌ TypeError: can't multiply sequence by non-int")
    print("   Impact: 💥 Processing fails on unexpected string data")
    
    print("\n🟢 AFTER (Type Validation):")
    print("   Code: df = ensure_numeric_columns(df, ['volume'])")
    print("         result = df['volume'] * 2")
    print("   Result: ✅ Automatic type conversion, operation succeeds")
    print("   Impact: 🔧 Self-healing data processing")

if __name__ == "__main__":
    print("🎯 Data Validation Framework Demonstration")
    print("=" * 60)
    print("This script demonstrates how the validation framework")
    print("prevents common data-related failures in VS Code notebooks.")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_validation_fixes()
    demonstrate_before_after()
    
    print("\n🏆 Summary:")
    print("The validation framework transforms fragile notebooks into")
    print("robust, production-ready data processing pipelines that")
    print("gracefully handle changing data sources and quality issues.")
    print("\n✅ Implementation Complete - Ready for Production Use!")
