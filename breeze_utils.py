"""
Enhanced Breeze data utilities with comprehensive error handling and logging.
This module provides backward compatibility while integrating with the new enhanced utilities.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import ta
from breeze_connect import BreezeConnect
from app_config import Config
from data_processing_utils import ProcessingResult, ValidationError, ProcessingError

# Constants
ISO_DATETIME_SUFFIX = ".000Z"

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class BreezeDataManager:
    """Enhanced secure manager for Breeze API data operations with improved error handling."""
    
    def __init__(self):
        try:
            self.config = Config()
            self.breeze = None
            self.save_path = self.config.get_data_save_path()
            self._ensure_directories()
            logger.info("BreezeDataManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BreezeDataManager: {str(e)}")
            raise ProcessingError(f"BreezeDataManager initialization failed: {str(e)}")
        
    def _ensure_directories(self):
        """Ensure required directories exist with proper error handling."""
        try:
            os.makedirs(self.save_path, exist_ok=True)
            logger.info(f"Data save path confirmed: {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            raise ProcessingError(f"Directory creation failed: {str(e)}")
    
    def authenticate(self):
        """Authenticate with Breeze API using secure credentials with enhanced error handling."""
        try:
            creds = self.config.get_breeze_credentials()
            
            if not all(key in creds for key in ['api_key', 'api_secret', 'session_token']):
                raise ValidationError("Missing required Breeze credentials")
            
            self.breeze = BreezeConnect(api_key=creds['api_key'])
            session_result = self.breeze.generate_session(
                api_secret=creds['api_secret'], 
                session_token=creds['session_token']
            )
            
            # Validate session creation
            if not session_result or session_result.get('Status') != 200:
                raise ProcessingError(f"Session generation failed: {session_result}")
            
            logger.info("✅ BreezeConnect authenticated successfully")
            return True
            
        except ValidationError as e:
            logger.error(f"❌ Validation error during authentication: {str(e)}")
            return False
        except ProcessingError as e:
            logger.error(f"❌ Processing error during authentication: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during authentication: {str(e)}")
            return False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")

    
def get_date_iso(self, days_ago=0):
    """Get ISO formatted date string with error handling."""
    try:
        date = datetime.now() - timedelta(days=days_ago)
        date = date.replace(hour=9, minute=0, second=0, microsecond=0)
        return date.isoformat() + ISO_DATETIME_SUFFIX
    except Exception as e:
        logger.error(f"Date formatting error: {str(e)}")
        raise ProcessingError(f"Failed to format date: {str(e)}")

def get_end_date_iso(self):
    """Get ISO formatted end date string with error handling."""
    try:
        date = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
        return date.isoformat() + ISO_DATETIME_SUFFIX
    except Exception as e:
        logger.error(f"End date formatting error: {str(e)}")
        raise ProcessingError(f"Failed to format end date: {str(e)}")

def get_last_trading_day(self, days_back=0):
    """Get the last trading day with enhanced error handling."""
    try:
        if not self.breeze:
            raise ValidationError("Breeze not authenticated")
            
        date = datetime.now() - timedelta(days=days_back)
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            # Skip weekends
            # 5=Saturday, 6=Sunday
            if date.weekday() >= 5:  
                date -= timedelta(days=1)
                attempts += 1
                continue

            # Verify trading day by checking index data
            try:
                response = self.breeze.get_historical_data_v2(
                    interval="1day",
                    from_date=date.replace(hour=9, minute=0, second=0, microsecond=0).isoformat() + ISO_DATETIME_SUFFIX,
                    to_date=date.replace(hour=15, minute=30, second=0, microsecond=0).isoformat() + ISO_DATETIME_SUFFIX,
                    stock_code="NIFTY",
                    exchange_code="NSE",
                    product_type="cash"
                )
                
                if response.get('Status') == 200 and response.get('Success'):
                    logger.info(f"Found trading day: {date.strftime('%Y-%m-%d')}")
                    return date
                    
            except Exception as api_error:
                logger.debug(f"API check failed for {date}: {str(api_error)}")

            date -= timedelta(days=1)
            attempts += 1

        # Fallback to current date if no trading day found
        logger.warning(f"Could not find trading day in {max_attempts} attempts, using current date")
        return datetime.now()
        
    except ValidationError as e:
        logger.error(f"Validation error in get_last_trading_day: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_last_trading_day: {str(e)}")
        return datetime.now()  # Graceful fallback

def fetch_historical_data(self, stock_code, exchange_code, from_date, to_date, interval="5minute"):
    """Fetch historical data for a given stock."""
    if not self.breeze:
        raise ValueError("Breeze API not authenticated")
        
    try:
        response = self.breeze.get_historical_data_v2(
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="cash"
        )
        
        if response['Status'] == 200:
            df = pd.DataFrame(response['Result'])
            if not df.empty:
                # Clean and process the data
                df = self._clean_data(df)
                return df
        else:
            logger.error(f"API Error: {response}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching data for {stock_code}: {e}")
        return pd.DataFrame()

def _clean_data(self, df):
    """Clean and standardize the data format."""
    try:
        # Convert datetime column
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return df

def detect_strike_step(strikes):
    """Detect strike price step from a list of strikes."""
    if len(strikes) < 2:
        return None
    return int(min(np.diff(sorted(strikes))))

def get_nearest_strike_price(ltp, valid_step):
    """Get nearest strike price based on LTP and step."""
    if ltp and valid_step:
        return int(round(ltp / valid_step) * valid_step)
    return None

def add_all_technical_indicators(df):
    df = df.copy()
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill()
    # Trend indicators
    try:
        df['ADX_14'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['ADX_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
        df['ADX_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
    except Exception as e:
        logger.warning(f"ADX Error: {e}")
    try:
        df['Aroon_Up'] = ta.trend.aroon_up(df['high'], df['low'])
        df['Aroon_Down'] = ta.trend.aroon_down(df['high'], df['low'])
        df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']
    except Exception as e:
        logger.warning(f"Aroon Error: {e}")
    try:
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    except Exception as e:
        logger.warning(f"MACD Error: {e}")
    try:
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['Ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception as e:
        logger.warning(f"Ichimoku Error: {e}")
    ma_windows = [5, 10, 20, 50, 100, 200]
    for window in ma_windows:
        try:
            df[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
        except Exception as e:
            logger.warning(f"SMA_{window} Error: {e}")
        try:
            df[f'EMA_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
        except Exception as e:
            logger.warning(f"EMA_{window} Error: {e}")
        try:
            df[f'WMA_{window}'] = ta.trend.wma_indicator(df['close'], window=window)
        except Exception as e:
            logger.warning(f"WMA_{window} Error: {e}")
    rsi_windows = [7, 14, 21, 28]
    for window in rsi_windows:
        try:
            df[f'RSI_{window}'] = ta.momentum.rsi(df['close'], window=window)
        except Exception as e:
            logger.warning(f"RSI_{window} Error: {e}")
    try:
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['Stoch_%K'] = stoch.stoch()
        df['Stoch_%D'] = stoch.stoch_signal()
        df['Stoch_RSI'] = ta.momentum.stochrsi(df['close'])
    except Exception as e:
        logger.warning(f"Stochastic Error: {e}")
    # ... (other indicators as in original)
    return df

def filter_trading_hours(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.between_time("09:15", "15:30").reset_index()
    return df

def process_options_data(df):
    if not all(col in df.columns for col in ['strike', 'right']):
        raise ValueError("DataFrame must contain 'strike' and 'right' columns")
    options_df = df.copy()
    options_df = options_df.groupby(['strike', 'right'], group_keys=False)\
        .apply(lambda x: add_all_technical_indicators(x), include_groups=False).reset_index(drop=True)
    logger.info(f"Processed options data with {len(options_df.columns)} indicators")
    return options_df

def load_and_process(file_path):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        logger.warning(f"Skipping {file_path} — File is missing or empty.")
        return None
    df = pd.read_csv(file_path)
    df = filter_trading_hours(df)
    if 'strike' in df.columns and 'right' in df.columns:
        return process_options_data(df)
    else:
        df = add_all_technical_indicators(df)
        logger.info(f"{file_path} finally has columns: {df.columns}")
    return df
