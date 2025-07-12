"""
Advanced data processing utilities for breeze data with comprehensive error handling and logging.
This module provides secure, modular, and robust data processing capabilities.
"""

import logging
import os
import traceback
import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
from dataclasses import dataclass
from enum import Enum

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataQuality(Enum):
    """Enum for data quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ProcessingResult:
    """Data structure for processing results with comprehensive metadata."""
    data: pd.DataFrame
    success: bool
    quality: DataQuality
    errors: List[str]
    warnings: List[str]
    metadata: Dict
    processing_time: float

class TechnicalIndicatorProcessor:
    """Robust technical indicator calculation with comprehensive error handling."""
    
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)
        self.successful_indicators = []
        self.failed_indicators = []
        
    def validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame for technical indicator calculation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            if df.empty:
                raise ValidationError("Input DataFrame is empty")
                
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
                
            # Check for sufficient data points
            if len(df) < 20:
                self.logger.warning(f"Limited data: only {len(df)} rows available")
                
            # Validate OHLC logic
            invalid_ohlc = df[(df['high'] < df['low']) | 
                             (df['high'] < df['open']) | 
                             (df['high'] < df['close']) |
                             (df['low'] > df['open']) | 
                             (df['low'] > df['close'])]
            
            if not invalid_ohlc.empty:
                self.logger.warning(f"Found {len(invalid_ohlc)} rows with invalid OHLC data")
                
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in data validation: {e}")
            raise ValidationError(f"Data validation failed: {e}")
    
    def safe_indicator_calculation(self, func, func_name: str, *args, **kwargs) -> Optional[pd.Series]:
        """
        Safely calculate a technical indicator with comprehensive error handling.
        
        Args:
            func: Function to calculate indicator
            func_name (str): Name of the indicator for logging
            *args, **kwargs: Arguments for the function
            
        Returns:
            Optional[pd.Series]: Calculated indicator or None if failed
        """
        try:
            self.logger.debug(f"Calculating {func_name}")
            result = func(*args, **kwargs)
            
            if result is None or (hasattr(result, 'empty') and result.empty):
                self.logger.warning(f"{func_name}: Returned empty result")
                self.failed_indicators.append(f"{func_name}: Empty result")
                return None
                
            # Check for excessive NaN values
            if hasattr(result, 'isna'):
                nan_percentage = result.isna().sum() / len(result) * 100
                if nan_percentage > 50:
                    self.logger.warning(f"{func_name}: {nan_percentage:.1f}% NaN values")
                    
            self.successful_indicators.append(func_name)
            self.logger.debug(f"Successfully calculated {func_name}")
            return result
            
        except Exception as e:
            error_msg = f"{func_name} calculation failed: {str(e)}"
            self.logger.error(error_msg)
            self.failed_indicators.append(error_msg)
            return None
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators with comprehensive error handling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with trend indicators added
        """
        self.logger.info("Adding trend indicators")
        result_df = df.copy()
        
        try:
            # ADX Family
            adx_result = self.safe_indicator_calculation(
                ta.trend.adx, "ADX_14", result_df['high'], result_df['low'], result_df['close']
            )
            if adx_result is not None:
                result_df['ADX_14'] = adx_result
                
            adx_pos = self.safe_indicator_calculation(
                ta.trend.adx_pos, "ADX_pos", result_df['high'], result_df['low'], result_df['close']
            )
            if adx_pos is not None:
                result_df['ADX_pos'] = adx_pos
                
            adx_neg = self.safe_indicator_calculation(
                ta.trend.adx_neg, "ADX_neg", result_df['high'], result_df['low'], result_df['close']
            )
            if adx_neg is not None:
                result_df['ADX_neg'] = adx_neg
            
            # Aroon indicators
            aroon_up = self.safe_indicator_calculation(
                ta.trend.aroon_up, "Aroon_Up", result_df['high'], result_df['low']
            )
            if aroon_up is not None:
                result_df['Aroon_Up'] = aroon_up
                
            aroon_down = self.safe_indicator_calculation(
                ta.trend.aroon_down, "Aroon_Down", result_df['high'], result_df['low']
            )
            if aroon_down is not None:
                result_df['Aroon_Down'] = aroon_down
                result_df['Aroon_Osc'] = result_df['Aroon_Up'] - result_df['Aroon_Down']
            
            # MACD
            try:
                macd_obj = ta.trend.MACD(result_df['close'])
                macd_result = self.safe_indicator_calculation(macd_obj.macd, "MACD")
                if macd_result is not None:
                    result_df['MACD'] = macd_result
                    
                macd_signal = self.safe_indicator_calculation(macd_obj.macd_signal, "MACD_signal")
                if macd_signal is not None:
                    result_df['MACD_signal'] = macd_signal
                    
                macd_diff = self.safe_indicator_calculation(macd_obj.macd_diff, "MACD_diff")
                if macd_diff is not None:
                    result_df['MACD_diff'] = macd_diff
                    result_df['MACD_hist'] = result_df['MACD'] - result_df['MACD_signal']
                    
            except Exception as e:
                self.logger.error(f"MACD calculation failed: {e}")
                self.failed_indicators.append(f"MACD: {e}")
            
            # Moving Averages
            ma_windows = [5, 10, 20, 50, 100, 200]
            for window in ma_windows:
                sma_result = self.safe_indicator_calculation(
                    ta.trend.sma_indicator, f"SMA_{window}", result_df['close'], window=window
                )
                if sma_result is not None:
                    result_df[f'SMA_{window}'] = sma_result
                    
                ema_result = self.safe_indicator_calculation(
                    ta.trend.ema_indicator, f"EMA_{window}", result_df['close'], window=window
                )
                if ema_result is not None:
                    result_df[f'EMA_{window}'] = ema_result
                    
                wma_result = self.safe_indicator_calculation(
                    ta.trend.wma_indicator, f"WMA_{window}", result_df['close'], window=window
                )
                if wma_result is not None:
                    result_df[f'WMA_{window}'] = wma_result
            
            self.logger.info(f"Trend indicators completed. Success: {len(self.successful_indicators)}, Failed: {len(self.failed_indicators)}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Critical error in trend indicators: {e}")
            raise ProcessingError(f"Trend indicator processing failed: {e}")
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators with comprehensive error handling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with momentum indicators added
        """
        self.logger.info("Adding momentum indicators")
        result_df = df.copy()
        
        try:
            # RSI Family
            rsi_windows = [7, 14, 21, 28]
            for window in rsi_windows:
                rsi_result = self.safe_indicator_calculation(
                    ta.momentum.rsi, f"RSI_{window}", result_df['close'], window=window
                )
                if rsi_result is not None:
                    result_df[f'RSI_{window}'] = rsi_result
            
            # Stochastic Oscillator
            try:
                stoch_obj = ta.momentum.StochasticOscillator(
                    result_df['high'], result_df['low'], result_df['close']
                )
                stoch_k = self.safe_indicator_calculation(stoch_obj.stoch, "Stoch_%K")
                if stoch_k is not None:
                    result_df['Stoch_%K'] = stoch_k
                    
                stoch_d = self.safe_indicator_calculation(stoch_obj.stoch_signal, "Stoch_%D")
                if stoch_d is not None:
                    result_df['Stoch_%D'] = stoch_d
                    
                stoch_rsi = self.safe_indicator_calculation(
                    ta.momentum.stochrsi, "Stoch_RSI", result_df['close']
                )
                if stoch_rsi is not None:
                    result_df['Stoch_RSI'] = stoch_rsi
                    
            except Exception as e:
                self.logger.error(f"Stochastic calculation failed: {e}")
                self.failed_indicators.append(f"Stochastic: {e}")
            
            # Other momentum indicators
            momentum_indicators = [
                ('CCI', lambda: ta.trend.cci(result_df['high'], result_df['low'], result_df['close'])),
                ('DPO', lambda: ta.trend.dpo(result_df['close'])),
                ('KST', lambda: ta.trend.kst(result_df['close'])),
                ('KST_sig', lambda: ta.trend.kst_sig(result_df['close'])),
                ('TSI', lambda: ta.momentum.tsi(result_df['close'])),
                ('ROC', lambda: ta.momentum.roc(result_df['close'])),
                ('PPO', lambda: ta.momentum.ppo(result_df['close'])),
                ('KAMA', lambda: ta.momentum.kama(result_df['close'])),
                ('WILLR', lambda: ta.momentum.williams_r(result_df['high'], result_df['low'], result_df['close']))
            ]
            
            for name, func in momentum_indicators:
                indicator_result = self.safe_indicator_calculation(func, name)
                if indicator_result is not None:
                    result_df[name] = indicator_result
            
            self.logger.info("Momentum indicators completed")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Critical error in momentum indicators: {e}")
            raise ProcessingError(f"Momentum indicator processing failed: {e}")
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators with comprehensive error handling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with volatility indicators added
        """
        self.logger.info("Adding volatility indicators")
        result_df = df.copy()
        
        try:
            # ATR
            atr_result = self.safe_indicator_calculation(
                ta.volatility.average_true_range, "ATR_14", 
                result_df['high'], result_df['low'], result_df['close']
            )
            if atr_result is not None:
                result_df['ATR_14'] = atr_result
            
            # Bollinger Bands
            try:
                bb_obj = ta.volatility.BollingerBands(result_df['close'])
                bb_mavg = self.safe_indicator_calculation(bb_obj.bollinger_mavg, "BB_MAVG")
                if bb_mavg is not None:
                    result_df['BB_MAVG'] = bb_mavg
                    
                bb_high = self.safe_indicator_calculation(bb_obj.bollinger_hband, "BB_HIGH")
                if bb_high is not None:
                    result_df['BB_HIGH'] = bb_high
                    
                bb_low = self.safe_indicator_calculation(bb_obj.bollinger_lband, "BB_LOW")
                if bb_low is not None:
                    result_df['BB_LOW'] = bb_low
                    
                bb_width = self.safe_indicator_calculation(bb_obj.bollinger_wband, "BB_WIDTH")
                if bb_width is not None:
                    result_df['BB_WIDTH'] = bb_width
                    
                bb_percent = self.safe_indicator_calculation(bb_obj.bollinger_pband, "BB_PERCENT")
                if bb_percent is not None:
                    result_df['BB_PERCENT'] = bb_percent
                    
            except Exception as e:
                self.logger.error(f"Bollinger Bands calculation failed: {e}")
                self.failed_indicators.append(f"Bollinger Bands: {e}")
            
            # Donchian Channel
            dc_high = self.safe_indicator_calculation(
                ta.volatility.donchian_channel_hband, "DC_HIGH",
                result_df['high'], result_df['low'], result_df['close']
            )
            if dc_high is not None:
                result_df['DC_HIGH'] = dc_high
                
            dc_low = self.safe_indicator_calculation(
                ta.volatility.donchian_channel_lband, "DC_LOW",
                result_df['high'], result_df['low'], result_df['close']
            )
            if dc_low is not None:
                result_df['DC_LOW'] = dc_low
                result_df['DC_MID'] = (result_df['DC_HIGH'] + result_df['DC_LOW']) / 2
            
            self.logger.info("Volatility indicators completed")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Critical error in volatility indicators: {e}")
            raise ProcessingError(f"Volatility indicator processing failed: {e}")
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators with comprehensive error handling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with volume indicators added
        """
        self.logger.info("Adding volume indicators")
        result_df = df.copy()
        
        try:
            if 'volume' not in result_df.columns:
                self.logger.warning("Volume column not found, skipping volume indicators")
                return result_df
            
            # Volume indicators
            volume_indicators = [
                ('OBV', lambda: ta.volume.on_balance_volume(result_df['close'], result_df['volume'])),
                ('CMF', lambda: ta.volume.chaikin_money_flow(result_df['high'], result_df['low'], result_df['close'], result_df['volume'])),
                ('MFI', lambda: ta.volume.money_flow_index(result_df['high'], result_df['low'], result_df['close'], result_df['volume'])),
                ('ADI', lambda: ta.volume.acc_dist_index(result_df['high'], result_df['low'], result_df['close'], result_df['volume'])),
                ('EOM', lambda: ta.volume.ease_of_movement(result_df['high'], result_df['low'], result_df['volume'])),
                ('VWAP', lambda: ta.volume.volume_weighted_average_price(result_df['high'], result_df['low'], result_df['close'], result_df['volume'])),
                ('FI', lambda: ta.volume.force_index(result_df['close'], result_df['volume']))
            ]
            
            for name, func in volume_indicators:
                indicator_result = self.safe_indicator_calculation(func, name)
                if indicator_result is not None:
                    result_df[name] = indicator_result
            
            # Volume Moving Averages
            for window in [5, 10, 20, 50]:
                vol_ma = self.safe_indicator_calculation(
                    lambda w=window: result_df['volume'].rolling(window=w).mean(),
                    f"Volume_MA_{window}"
                )
                if vol_ma is not None:
                    result_df[f'Volume_MA_{window}'] = vol_ma
            
            self.logger.info("Volume indicators completed")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Critical error in volume indicators: {e}")
            raise ProcessingError(f"Volume indicator processing failed: {e}")
    
    def add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom indicators with comprehensive error handling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with custom indicators added
        """
        self.logger.info("Adding custom indicators")
        result_df = df.copy()
        
        try:
            # Price * Volume
            if 'volume' in result_df.columns:
                result_df['PV'] = result_df['close'] * result_df['volume']
            
            # Returns
            daily_return = self.safe_indicator_calculation(
                ta.others.daily_return, "Daily_Return", result_df['close']
            )
            if daily_return is not None:
                result_df['Daily_Return'] = daily_return
                
            cum_return = self.safe_indicator_calculation(
                ta.others.cumulative_return, "Cum_Return", result_df['close']
            )
            if cum_return is not None:
                result_df['Cum_Return'] = cum_return
                
            # Log returns with safe calculation
            try:
                log_return = np.log(result_df['close'] / result_df['close'].shift(1))
                log_return = log_return.replace([np.inf, -np.inf], np.nan)
                result_df['Log_Return'] = log_return
                self.successful_indicators.append("Log_Return")
            except Exception as e:
                self.logger.error(f"Log return calculation failed: {e}")
                self.failed_indicators.append(f"Log_Return: {e}")
            
            # Price changes
            result_df['Price_Change'] = result_df['close'].diff()
            result_df['Pct_Change'] = result_df['close'].pct_change(fill_method=None)
            
            # Volatility measures
            result_df['HL_Pct'] = (result_df['high'] - result_df['low']) / result_df['low'] * 100
            result_df['OC_Pct'] = (result_df['close'] - result_df['open']) / result_df['open'] * 100
            
            self.logger.info("Custom indicators completed")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Critical error in custom indicators: {e}")
            raise ProcessingError(f"Custom indicator processing failed: {e}")
    
    def process_all_indicators(self, df: pd.DataFrame) -> ProcessingResult:
        """
        Process all technical indicators with comprehensive error handling and reporting.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            ProcessingResult: Comprehensive processing result
        """
        start_time = datetime.now()
        self.logger.info("Starting comprehensive technical indicator processing")
        
        try:
            # Reset counters
            self.successful_indicators = []
            self.failed_indicators = []
            
            # Validate input
            self.validate_input_data(df)
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].ffill()  # Forward fill NaN values
            
            # Process each indicator category
            result_df = df.copy()
            result_df = self.add_trend_indicators(result_df)
            result_df = self.add_momentum_indicators(result_df)
            result_df = self.add_volatility_indicators(result_df)
            result_df = self.add_volume_indicators(result_df)
            result_df = self.add_custom_indicators(result_df)
            
            # Clean up infinite values
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            
            # Assess data quality
            quality = self._assess_data_quality(result_df)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = ProcessingResult(
                data=result_df,
                success=True,
                quality=quality,
                errors=self.failed_indicators,
                warnings=[],
                metadata={
                    'total_indicators': len(self.successful_indicators) + len(self.failed_indicators),
                    'successful_indicators': len(self.successful_indicators),
                    'failed_indicators': len(self.failed_indicators),
                    'success_rate': len(self.successful_indicators) / (len(self.successful_indicators) + len(self.failed_indicators)) * 100,
                    'original_shape': df.shape,
                    'final_shape': result_df.shape,
                    'added_columns': result_df.shape[1] - df.shape[1]
                },
                processing_time=processing_time
            )
            
            self.logger.info(f"Technical indicator processing completed in {processing_time:.2f}s")
            self.logger.info(f"Success rate: {result.metadata['success_rate']:.1f}% ({result.metadata['successful_indicators']}/{result.metadata['total_indicators']})")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Critical failure in technical indicator processing: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ProcessingResult(
                data=pd.DataFrame(),
                success=False,
                quality=DataQuality.INVALID,
                errors=[str(e)],
                warnings=[],
                metadata={'processing_time': processing_time},
                processing_time=processing_time
            )
    
    def _assess_data_quality(self, df: pd.DataFrame) -> DataQuality:
        """
        Assess the quality of processed data.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            DataQuality: Quality assessment
        """
        try:
            if df.empty:
                return DataQuality.INVALID
            
            # Calculate metrics
            total_cells = df.shape[0] * df.shape[1]
            nan_count = df.isna().sum().sum()
            nan_percentage = (nan_count / total_cells) * 100
            
            success_rate = len(self.successful_indicators) / (len(self.successful_indicators) + len(self.failed_indicators)) * 100
            
            # Quality assessment
            if nan_percentage < 5 and success_rate > 90:
                return DataQuality.EXCELLENT
            elif nan_percentage < 15 and success_rate > 75:
                return DataQuality.GOOD
            elif nan_percentage < 30 and success_rate > 50:
                return DataQuality.FAIR
            elif nan_percentage < 50 and success_rate > 25:
                return DataQuality.POOR
            else:
                return DataQuality.INVALID
                
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return DataQuality.INVALID

class OptionsDataProcessor:
    """Robust options data processing with comprehensive error handling."""
    
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)
        
    def validate_options_data(self, df: pd.DataFrame) -> bool:
        """
        Validate options data structure and content.
        
        Args:
            df (pd.DataFrame): Options DataFrame
            
        Returns:
            bool: True if valid
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            required_cols = ['strike', 'right']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValidationError(f"Missing required options columns: {missing_cols}")
                
            # Validate option types
            valid_rights = {'call', 'put', 'c', 'p'}
            invalid_rights = set(df['right'].str.lower().unique()) - valid_rights
            
            if invalid_rights:
                raise ValidationError(f"Invalid option rights found: {invalid_rights}")
                
            # Validate strike prices
            if df['strike'].isna().any():
                raise ValidationError("Strike prices contain NaN values")
                
            if (df['strike'] <= 0).any():
                raise ValidationError("Strike prices must be positive")
                
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Options data validation failed: {e}")
    
    def process_options_with_grouping(self, df: pd.DataFrame) -> ProcessingResult:
        """
        Process options data with proper grouping and error handling.
        
        Args:
            df (pd.DataFrame): Options DataFrame
            
        Returns:
            ProcessingResult: Processing result with comprehensive metadata
        """
        start_time = datetime.now()
        self.logger.info("Starting options data processing")
        
        try:
            # Validate input
            self.validate_options_data(df)
            
            # Initialize technical indicator processor
            indicator_processor = TechnicalIndicatorProcessor()
            
            # Process each strike-right combination
            processed_groups = []
            errors = []
            warnings = []
            
            for (strike, right), group in df.groupby(['strike', 'right']):
                try:
                    self.logger.debug(f"Processing {right.upper()} {strike}")
                    
                    # Process indicators for this group
                    result = indicator_processor.process_all_indicators(group)
                    
                    if result.success:
                        processed_groups.append(result.data)
                        self.logger.debug(f"Successfully processed {right.upper()} {strike}")
                    else:
                        error_msg = f"Failed to process {right.upper()} {strike}: {result.errors}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error processing {right.upper()} {strike}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            if not processed_groups:
                raise ProcessingError("No options groups were successfully processed")
            
            # Combine all processed groups
            combined_df = pd.concat(processed_groups, ignore_index=True)
            
            # Assess quality
            success_rate = len(processed_groups) / len(df.groupby(['strike', 'right'])) * 100
            
            if success_rate > 90:
                quality = DataQuality.EXCELLENT
            elif success_rate > 75:
                quality = DataQuality.GOOD
            elif success_rate > 50:
                quality = DataQuality.FAIR
            else:
                quality = DataQuality.POOR
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                data=combined_df,
                success=True,
                quality=quality,
                errors=errors,
                warnings=warnings,
                metadata={
                    'total_groups': len(df.groupby(['strike', 'right'])),
                    'processed_groups': len(processed_groups),
                    'success_rate': success_rate,
                    'original_shape': df.shape,
                    'final_shape': combined_df.shape
                },
                processing_time=processing_time
            )
            
            self.logger.info(f"Options processing completed in {processing_time:.2f}s")
            self.logger.info(f"Processed {len(processed_groups)}/{len(df.groupby(['strike', 'right']))} groups successfully")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Critical failure in options processing: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ProcessingResult(
                data=pd.DataFrame(),
                success=False,
                quality=DataQuality.INVALID,
                errors=[str(e)],
                warnings=[],
                metadata={'processing_time': processing_time},
                processing_time=processing_time
            )

def filter_trading_hours_safe(df: pd.DataFrame) -> ProcessingResult:
    """
    Safely filter data to trading hours with comprehensive error handling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        ProcessingResult: Filtered data with metadata
    """
    start_time = datetime.now()
    logger.info("Filtering data to trading hours")
    
    try:
        if df.empty:
            raise ValidationError("Input DataFrame is empty")
            
        if 'datetime' not in df.columns:
            raise ValidationError("'datetime' column not found")
        
        # Convert datetime column safely
        original_count = len(df)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Check for conversion errors
        invalid_dates = df['datetime'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid datetime entries")
        
        # Filter to trading hours
        df = df.set_index('datetime')
        filtered_df = df.between_time("09:15", "15:30").reset_index()
        
        filtered_count = len(filtered_df)
        filter_percentage = (filtered_count / original_count) * 100
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Assess quality based on how much data was retained
        if filter_percentage > 80:
            quality = DataQuality.EXCELLENT
        elif filter_percentage > 60:
            quality = DataQuality.GOOD
        elif filter_percentage > 40:
            quality = DataQuality.FAIR
        else:
            quality = DataQuality.POOR
        
        result = ProcessingResult(
            data=filtered_df,
            success=True,
            quality=quality,
            errors=[],
            warnings=[f"Filtered out {original_count - filtered_count} non-trading hour records"] if filtered_count < original_count else [],
            metadata={
                'original_count': original_count,
                'filtered_count': filtered_count,
                'retention_percentage': filter_percentage,
                'invalid_datetime_count': invalid_dates
            },
            processing_time=processing_time
        )
        
        logger.info(f"Trading hours filtering completed: {filtered_count}/{original_count} records retained ({filter_percentage:.1f}%)")
        return result
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Trading hours filtering failed: {e}")
        
        return ProcessingResult(
            data=pd.DataFrame(),
            success=False,
            quality=DataQuality.INVALID,
            errors=[str(e)],
            warnings=[],
            metadata={'processing_time': processing_time},
            processing_time=processing_time
        )

def load_and_process_file_safe(file_path: str) -> ProcessingResult:
    """
    Safely load and process a data file with comprehensive error handling.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        ProcessingResult: Processing result with comprehensive metadata
    """
    start_time = datetime.now()
    logger.info(f"Loading and processing file: {file_path}")
    
    try:
        # Check file existence and size
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValidationError(f"File is empty: {file_path}")
            
        logger.info(f"File size: {file_size / 1024:.1f} KB")
        
        # Load data
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValidationError("Loaded DataFrame is empty")
            
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Filter trading hours
        filter_result = filter_trading_hours_safe(df)
        if not filter_result.success:
            raise ProcessingError(f"Trading hours filtering failed: {filter_result.errors}")
            
        df = filter_result.data
        
        # Determine processing type based on columns
        if 'strike' in df.columns and 'right' in df.columns:
            logger.info("Detected options data, processing with options processor")
            processor = OptionsDataProcessor()
            result = processor.process_options_with_grouping(df)
        else:
            logger.info("Detected regular data, processing with technical indicators")
            processor = TechnicalIndicatorProcessor()
            result = processor.process_all_indicators(df)
        
        if not result.success:
            raise ProcessingError(f"Data processing failed: {result.errors}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update metadata
        result.metadata.update({
            'file_path': file_path,
            'file_size_kb': file_size / 1024,
            'total_processing_time': processing_time
        })
        
        logger.info(f"File processing completed successfully in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"File processing failed for {file_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return ProcessingResult(
            data=pd.DataFrame(),
            success=False,
            quality=DataQuality.INVALID,
            errors=[str(e)],
            warnings=[],
            metadata={
                'file_path': file_path,
                'processing_time': processing_time
            },
            processing_time=processing_time
        )

# Export main classes and functions
__all__ = [
    'TechnicalIndicatorProcessor',
    'OptionsDataProcessor', 
    'ProcessingResult',
    'DataQuality',
    'ProcessingError',
    'ValidationError',
    'filter_trading_hours_safe',
    'load_and_process_file_safe'
]
