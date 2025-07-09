"""
Index data utilities for fetching and processing NSE index data using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from functools import reduce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from app_config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexDataManager:
    """Manager for NSE index data operations using yfinance."""
    
    def __init__(self):
        self.config = Config()
        self.save_path = self.config.get_data_save_path()
        self.index_symbols = self.config.get_index_symbols()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.save_path, exist_ok=True)
        logger.info(f"Index data save path: {self.save_path}")
    
    def fetch_all_indices(self, start_date=None, end_date=None, interval="1d"):
        """
        Fetch data for all configured index symbols.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format  
            interval (str): Data interval (1d, 1h, 5m, etc.)
        
        Returns:
            dict: Dictionary with index names as keys and DataFrames as values
        """
        if not start_date:
            start_date = self.config.get('START_DATE')
        if not end_date:
            end_date = self.config.get('END_DATE')
            
        logger.info(f"Fetching index data from {start_date} to {end_date}")
        logger.info(f"Processing {len(self.index_symbols)} indices")
        
        results = {}
        
        for name, symbol in self.index_symbols.items():
            try:
                logger.info(f"Fetching {name} ({symbol})")
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                
                if not df.empty:
                    # Save to CSV file
                    filename = f"{name}_history.csv"
                    filepath = os.path.join(self.save_path, filename)
                    df.to_csv(filepath)
                    
                    results[name] = df
                    logger.info(f"✅ Fetched {len(df)} rows for {name}")
                else:
                    logger.warning(f"⚠️ No data fetched for {name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to fetch {name}: {e}")
                
        logger.info("Index data fetch completed")
        return results
    
    def fetch_single_index(self, name, symbol=None, start_date=None, end_date=None, interval="1d"):
        """
        Fetch data for a single index.
        
        Args:
            name (str): Index name
            symbol (str): Yahoo Finance symbol (if None, looks up from config)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: Index data
        """
        if not symbol:
            symbol = self.index_symbols.get(name)
            if not symbol:
                raise ValueError(f"Symbol not found for index: {name}")
        
        if not start_date:
            start_date = self.config.get('START_DATE')
        if not end_date:
            end_date = self.config.get('END_DATE')
            
        try:
            logger.info(f"Fetching {name} ({symbol})")
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            
            if not df.empty:
                # Save to CSV file
                filename = f"{name}_history.csv"
                filepath = os.path.join(self.save_path, filename)
                df.to_csv(filepath)
                logger.info(f"✅ Fetched {len(df)} rows for {name}")
                return df
            else:
                logger.warning(f"⚠️ No data fetched for {name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch {name}: {e}")
            return pd.DataFrame()
    
    def load_saved_index_data(self, index_name):
        """
        Load previously saved index data from CSV.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            pd.DataFrame: Loaded index data
        """
        filename = f"{index_name}_history.csv"
        filepath = os.path.join(self.save_path, filename)
        
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                logger.info(f"✅ Loaded {len(df)} rows for {index_name}")
                return df
            else:
                logger.warning(f"⚠️ File not found: {filepath}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Failed to load {index_name}: {e}")
            return pd.DataFrame()
    
    def get_available_indices(self):
        """
        Get list of available index files.
        
        Returns:
            list: List of available index names
        """
        available = []
        for index_name in self.index_symbols.keys():
            filename = f"{index_name}_history.csv"
            filepath = os.path.join(self.save_path, filename)
            if os.path.exists(filepath):
                available.append(index_name)
        
        logger.info(f"Found {len(available)} saved index files")
        return available
    
    def calculate_index_correlations(self, indices=None, column='Close'):
        """
        Calculate correlations between indices.
        
        Args:
            indices (list): List of index names to include (None for all)
            column (str): Column to use for correlation
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if indices is None:
            indices = self.get_available_indices()
        
        data = {}
        for index_name in indices:
            df = self.load_saved_index_data(index_name)
            if not df.empty and column in df.columns:
                data[index_name] = df[column]
        
        if data:
            combined_df = pd.DataFrame(data)
            correlation_matrix = combined_df.corr()
            logger.info(f"Calculated correlations for {len(data)} indices")
            return correlation_matrix
        else:
            logger.warning("No data available for correlation calculation")
            return pd.DataFrame()
    
    def get_index_summary(self):
        """
        Get summary statistics for all available indices.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        available_indices = self.get_available_indices()
        summary_data = []
        
        for index_name in available_indices:
            df = self.load_saved_index_data(index_name)
            if not df.empty:
                try:
                    summary = {
                        'Index': index_name,
                        'Symbol': self.index_symbols.get(index_name, 'Unknown'),
                        'Start_Date': df.index.min().strftime('%Y-%m-%d'),
                        'End_Date': df.index.max().strftime('%Y-%m-%d'),
                        'Total_Days': len(df),
                        'Current_Price': df['Close'].iloc[-1] if 'Close' in df.columns else None,
                        'Min_Price': df['Close'].min() if 'Close' in df.columns else None,
                        'Max_Price': df['Close'].max() if 'Close' in df.columns else None,
                        'Avg_Volume': df['Volume'].mean() if 'Volume' in df.columns else None
                    }
                    summary_data.append(summary)
                except Exception as e:
                    logger.error(f"Error calculating summary for {index_name}: {e}")
        
        return pd.DataFrame(summary_data)

# Utility functions
def calculate_returns(price_series, method='simple'):
    """
    Calculate returns from price series.
    
    Args:
        price_series (pd.Series): Price series
        method (str): 'simple' or 'log'
        
    Returns:
        pd.Series: Returns series
    """
    if method == 'simple':
        return price_series.pct_change()
    elif method == 'log':
        return np.log(price_series / price_series.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")

def calculate_volatility(returns, window=30):
    """
    Calculate rolling volatility.
    
    Args:
        returns (pd.Series): Returns series
        window (int): Rolling window size
        
    Returns:
        pd.Series: Volatility series
    """
    return returns.rolling(window=window).std() * (252 ** 0.5)  # Annualized

def detect_market_regime(index_data, short_ma=50, long_ma=200):
    """
    Detect bull/bear market regime based on moving averages.
    
    Args:
        index_data (pd.DataFrame): Index data with Close column
        short_ma (int): Short moving average period
        long_ma (int): Long moving average period
        
    Returns:
        pd.Series: Market regime ('Bull' or 'Bear')
    """
    if 'Close' not in index_data.columns:
        raise ValueError("Close column not found in data")
    
    short_ma_values = index_data['Close'].rolling(window=short_ma).mean()
    long_ma_values = index_data['Close'].rolling(window=long_ma).mean()
    
    regime = pd.Series(index=index_data.index, dtype=str)
    regime[short_ma_values > long_ma_values] = 'Bull'
    regime[short_ma_values <= long_ma_values] = 'Bear'
    
    return regime


class IndexDataProcessor:
    """Advanced data processing for index data, preserving original logic."""
    
    def __init__(self, data_manager: IndexDataManager):
        self.data_manager = data_manager
        self.logger = logger
    
    def clean_and_merge_data(self, fetched_data: dict) -> pd.DataFrame:
        """
        Clean and merge all index data - preserves original cleaning logic.
        
        Args:
            fetched_data (dict): Dictionary of index DataFrames from fetch_all_indices
            
        Returns:
            pd.DataFrame: Merged and cleaned data
        """
        self.logger.info("Starting data cleaning and merging process")
        
        cleaned_data = {}
        
        # Clean individual index data (preserving original logic)
        for name, df in fetched_data.items():
            if not df.empty:
                # Ensure proper column names (yfinance may return different formats)
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)  # Remove ticker level if present
                
                # Standardize column names to lowercase
                df.columns = [col.lower() for col in df.columns]
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    cleaned_data[name] = df[required_cols].copy()
                    self.logger.info(f"✅ Cleaned {name}: {len(df)} rows")
                else:
                    self.logger.warning(f"⚠️ Skipping {name}: missing required columns")
        
        if not cleaned_data:
            self.logger.error("No valid data to merge")
            return pd.DataFrame()
        
        # Merge all indices (preserving original merge logic)
        dfs = []
        for name, df in cleaned_data.items():
            # Add suffix to prevent column collision
            df = df.add_suffix(f'_{name}')
            dfs.append(df)
        
        # Perform outer merge on date index
        merged_indices = reduce(lambda left, right: pd.merge(left, right, 
                                                           left_index=True, right_index=True, 
                                                           how='outer'), dfs)
        
        # Fill missing values (preserving original forward/backward fill logic)
        merged_indices.ffill(inplace=True)
        merged_indices.bfill(inplace=True)
        
        self.logger.info(f"Merged indices shape: {merged_indices.shape}")
        return merged_indices
    
    def create_normalized_data(self, cleaned_data: dict) -> pd.DataFrame:
        """
        Create normalized data - preserves original normalization logic.
        
        Args:
            cleaned_data (dict): Dictionary of cleaned DataFrames
            
        Returns:
            pd.DataFrame: Normalized and merged data
        """
        self.logger.info("Creating normalized data")
        
        normalized_all = {}
        
        for name, df in cleaned_data.items():
            # Normalize all columns by first row (preserving original logic)
            norm_df = df / df.iloc[0]
            norm_df.columns = [f"{col}_{name}" for col in norm_df.columns]
            normalized_all[name] = norm_df
        
        # Combine all normalized data
        normalized_merged = pd.concat(normalized_all.values(), axis=1)
        
        self.logger.info(f"Normalized data shape: {normalized_merged.shape}")
        return normalized_merged
    
    def apply_scaling_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply multiple scaling transformations - preserves original scaling logic.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with scaling transformations added
        """
        self.logger.info("Applying scaling transformations")
        
        # Clean data first (preserving original cleaning logic)
        df = df.copy()
        df = df.sort_index().dropna(how="all")
        
        # Apply scaling transformations (preserving original logic)
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        
        df_std = pd.DataFrame(standard_scaler.fit_transform(df), 
                             columns=df.columns, index=df.index)
        df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), 
                                columns=df.columns, index=df.index)
        df_robust = pd.DataFrame(robust_scaler.fit_transform(df), 
                                columns=df.columns, index=df.index)
        
        # Add suffixes to avoid collision (preserving original logic)
        df_std.columns = [f"{col}_std" for col in df.columns]
        df_minmax.columns = [f"{col}_minmax" for col in df.columns]
        df_robust.columns = [f"{col}_robust" for col in df.columns]
        
        # Merge all scaling variations
        result_df = df.join([df_std, df_minmax, df_robust])
        
        self.logger.info(f"Scaled data shape: {result_df.shape}")
        return result_df
    
    def calculate_row_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate row-level statistics - preserves original logic.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with row statistics
        """
        self.logger.info("Calculating row-level statistics")
        
        row_stats = pd.DataFrame(index=df.index)
        
        # Row-level stats from raw data (preserving original logic)
        row_stats["row_mean"] = df.mean(axis=1)
        row_stats["row_std"] = df.std(axis=1)
        row_stats["row_variance"] = df.var(axis=1)
        
        # Row-level stats from returns (preserving original logic)
        returns = df.pct_change().replace([np.inf, -np.inf], np.nan)
        row_stats["row_return_mean"] = returns.mean(axis=1)
        row_stats["row_return_std"] = returns.std(axis=1)
        row_stats["row_sharpe_like"] = row_stats["row_return_mean"] / (row_stats["row_return_std"] + 1e-8)
        
        # Non-zero volume ratio (preserving original logic)
        vol_cols = [col for col in df.columns if "volume" in col.lower()]
        if vol_cols:
            row_stats["nonzero_volume_ratio"] = (df[vol_cols] > 0).sum(axis=1) / len(vol_cols)
        
        # Equal weight column (preserving original logic)
        row_stats["equal_weight"] = 1 / df.shape[1]
        
        self.logger.info(f"Row statistics calculated: {len(row_stats.columns)} features")
        return row_stats
    
    def add_rolling_features(self, df: pd.DataFrame, rolling_windows: list = [5, 20]) -> pd.DataFrame:
        """
        Add rolling window features - preserves original logic.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            rolling_windows (list): List of rolling window sizes
            
        Returns:
            pd.DataFrame: DataFrame with rolling features added
        """
        self.logger.info(f"Adding rolling features with windows: {rolling_windows}")
        
        rolling_df = df.copy()
        
        # Compute rolling features for each window (preserving original logic)
        for window in rolling_windows:
            self.logger.info(f"Processing rolling window: {window}")
            
            rolling_mean = rolling_df.rolling(window=window).mean()
            rolling_mean.columns = [f"{col}_rolling_mean_{window}" for col in rolling_df.columns]
            
            rolling_std = rolling_df.rolling(window=window).std()
            rolling_std.columns = [f"{col}_rolling_std_{window}" for col in rolling_df.columns]
            
            rolling_max = rolling_df.rolling(window=window).max()
            rolling_max.columns = [f"{col}_rolling_max_{window}" for col in rolling_df.columns]
            
            rolling_min = rolling_df.rolling(window=window).min()
            rolling_min.columns = [f"{col}_rolling_min_{window}" for col in rolling_df.columns]
            
            # Join all rolling features (preserving original logic)
            rolling_df = rolling_df.join([rolling_mean, rolling_std, rolling_max, rolling_min])
        
        self.logger.info(f"Rolling features added. Final shape: {rolling_df.shape}")
        return rolling_df
    
    def process_complete_pipeline(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Execute the complete data processing pipeline - preserves all original logic.
        
        Args:
            start_date (str): Start date for data fetching
            end_date (str): End date for data fetching
            
        Returns:
            pd.DataFrame: Fully processed and enriched DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPLETE INDEX DATA PROCESSING PIPELINE")
        self.logger.info("=" * 60)
        
        # Step 1: Fetch all index data
        self.logger.info("Step 1: Fetching index data")
        fetched_data = self.data_manager.fetch_all_indices(start_date, end_date)
        
        if not fetched_data:
            self.logger.error("No data fetched. Pipeline cannot continue.")
            return pd.DataFrame()
        
        # Step 2: Clean and merge data
        self.logger.info("Step 2: Cleaning and merging data")
        merged_indices = self.clean_and_merge_data(fetched_data)
        
        # Step 3: Apply scaling transformations
        self.logger.info("Step 3: Applying scaling transformations")
        scaled_data = self.apply_scaling_transformations(merged_indices)
        
        # Step 4: Calculate row statistics
        self.logger.info("Step 4: Calculating row statistics")
        row_stats = self.calculate_row_statistics(merged_indices)
        
        # Step 5: Merge scaled data with row statistics
        self.logger.info("Step 5: Merging scaled data with statistics")
        enriched_data = scaled_data.join(row_stats)
        
        # Step 6: Add rolling features
        self.logger.info("Step 6: Adding rolling features")
        final_data = self.add_rolling_features(enriched_data)
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"Final dataset shape: {final_data.shape}")
        self.logger.info(f"Final dataset columns: {len(final_data.columns)}")
        self.logger.info("=" * 60)
        
        return final_data


def create_index_processor() -> IndexDataProcessor:
    """Convenience function to create a configured IndexDataProcessor."""
    data_manager = IndexDataManager()
    return IndexDataProcessor(data_manager)
