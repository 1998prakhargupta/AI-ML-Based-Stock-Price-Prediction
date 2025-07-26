"""
Enhanced Breeze API utilities with comprehensive error handling, logging, modular design,
and API compliance monitoring.

This module provides robust data fetching, processing, and management capabilities
with strict adherence to API terms of service and rate limiting.
"""

import logging
import traceback
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
import json

from breeze_connect import BreezeConnect
from src.utils.app_config import Config
from ..data.processors import (
    ProcessingResult, DataQuality, ProcessingError, ValidationError,
    TechnicalIndicatorProcessor
)
from src.utils.file_management_utils import SafeFileManager, SaveStrategy, FileFormat

# Import compliance management
try:
    from ..compliance.api_compliance import compliance_decorator, DataProvider, ComplianceLevel
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False

# Constants
ISO_DATETIME_SUFFIX = ".000Z"

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('breeze_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log compliance status
if not COMPLIANCE_AVAILABLE:
    logger.warning("⚠️ API compliance module not available - running without compliance monitoring")

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class AuthenticationError(Exception):
    """Custom exception for authentication errors."""
    pass

class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass

class MarketDataType(Enum):
    """Enum for different types of market data."""
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    INDEX = "index"

@dataclass
class APIResponse:
    """Structured response from API calls."""
    success: bool
    data: Optional[pd.DataFrame]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    response_time: float

@dataclass
class MarketDataRequest:
    """Structured request for market data."""
    stock_code: str
    exchange_code: str
    product_type: str
    interval: str
    from_date: str
    to_date: str
    expiry_date: Optional[str] = None
    strike_price: Optional[float] = None
    right: Optional[str] = None

class EnhancedBreezeDataManager:
    """
    Enhanced Breeze API data manager with comprehensive error handling,
    logging, retry mechanisms, and data validation.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the enhanced Breeze data manager.
        
        Args:
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retry attempts in seconds
        """
        self.logger = logger.getChild(self.__class__.__name__)
        self.config = Config()
        self.breeze = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.save_path = self._initialize_paths()
        self.api_call_count = 0
        self.rate_limit_delay = 0.1  # Minimum delay between API calls
        self.last_api_call = 0
        
        # Initialize safe file manager for enhanced file operations
        self.file_manager = SafeFileManager(
            base_path=self.save_path,
            default_strategy=SaveStrategy.VERSION
        )
        
    def _initialize_paths(self) -> str:
        """
        Initialize and create necessary directories.
        
        Returns:
            str: Data save path
        """
        try:
            save_path = self.config.get_data_save_path()
            os.makedirs(save_path, exist_ok=True)
            self.logger.info(f"Data save path initialized: {save_path}")
            return save_path
        except Exception as e:
            self.logger.error(f"Failed to initialize paths: {e}")
            # Fallback to current directory
            fallback_path = os.path.join(os.getcwd(), "data")
            os.makedirs(fallback_path, exist_ok=True)
            self.logger.warning(f"Using fallback path: {fallback_path}")
            return fallback_path
    
    def authenticate(self) -> bool:
        """
        Authenticate with Breeze API with comprehensive error handling and compliance monitoring.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        self.logger.info("Attempting Breeze API authentication with compliance monitoring")
        
        # Apply compliance decorator if available
        if COMPLIANCE_AVAILABLE:
            @compliance_decorator(DataProvider.BREEZE_CONNECT, "authenticate")
            def _authenticate_with_compliance():
                return self._do_authenticate()
            return _authenticate_with_compliance()
        else:
            return self._do_authenticate()
    
    def _do_authenticate(self) -> bool:
        """Internal authentication implementation"""
        try:
            # Get credentials securely
            creds = self.config.get_breeze_credentials()
            
            # Validate credentials
            required_fields = ['api_key', 'api_secret', 'session_token']
            missing_fields = [field for field in required_fields if not creds.get(field)]
            
            if missing_fields:
                raise AuthenticationError(f"Missing credentials: {missing_fields}")
            
            # Initialize BreezeConnect
            self.breeze = BreezeConnect(api_key=creds['api_key'])
            
            # Generate session with retry logic
            for attempt in range(self.max_retries):
                try:
                    self.logger.debug(f"Authentication attempt {attempt + 1}/{self.max_retries}")
                    
                    response = self.breeze.generate_session(
                        api_secret=creds['api_secret'],
                        session_token=creds['session_token']
                    )
                    
                    # Validate response
                    if hasattr(response, 'get') and response.get('Status') == 200:
                        self.logger.info("✅ Breeze API authentication successful")
                        return True
                    else:
                        error_msg = response.get('Error', 'Unknown authentication error') if hasattr(response, 'get') else str(response)
                        self.logger.warning(f"Authentication attempt {attempt + 1} failed: {error_msg}")
                        
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                        
                except Exception as e:
                    self.logger.warning(f"Authentication attempt {attempt + 1} exception: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise
            
            raise AuthenticationError("All authentication attempts failed")
            
        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during authentication: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    def _rate_limit_check(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
        self.api_call_count += 1
    
    def _make_api_call(self, api_func, *args, **kwargs) -> APIResponse:
        """
        Make an API call with retry logic and error handling.
        
        Args:
            api_func: API function to call
            *args, **kwargs: Arguments for the API function
            
        Returns:
            APIResponse: Structured API response
        """
        start_time = time.time()
        
        if not self.breeze:
            return APIResponse(
                success=False,
                data=None,
                errors=["Breeze API not authenticated"],
                warnings=[],
                metadata={},
                response_time=0
            )
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                
                # Rate limiting
                self._rate_limit_check()
                
                # Make the API call
                response = api_func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Validate response
                if hasattr(response, 'get'):
                    if response.get('Status') == 200:
                        data = response.get('Success', [])
                        return APIResponse(
                            success=True,
                            data=pd.DataFrame(data) if data else pd.DataFrame(),
                            errors=[],
                            warnings=[],
                            metadata={
                                'api_call_count': self.api_call_count,
                                'attempt': attempt + 1,
                                'raw_response_size': len(str(response))
                            },
                            response_time=response_time
                        )
                    else:
                        error_msg = response.get('Error', 'Unknown API error')
                        self.logger.warning(f"API call attempt {attempt + 1} failed: {error_msg}")
                        
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                        else:
                            return APIResponse(
                                success=False,
                                data=None,
                                errors=[error_msg],
                                warnings=[],
                                metadata={'attempts': self.max_retries},
                                response_time=response_time
                            )
                else:
                    error_msg = f"Invalid response format: {type(response)}"
                    self.logger.error(error_msg)
                    return APIResponse(
                        success=False,
                        data=None,
                        errors=[error_msg],
                        warnings=[],
                        metadata={},
                        response_time=response_time
                    )
                    
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} exception: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    response_time = time.time() - start_time
                    return APIResponse(
                        success=False,
                        data=None,
                        errors=[str(e)],
                        warnings=[],
                        metadata={'attempts': self.max_retries},
                        response_time=response_time
                    )
        
        # This should never be reached, but included for completeness
        return APIResponse(
            success=False,
            data=None,
            errors=["Unexpected error in API call"],
            warnings=[],
            metadata={},
            response_time=time.time() - start_time
        )
    
    def get_quotes_safe(self, stock_code: str, exchange_code: str) -> APIResponse:
        """
        Safely get stock quotes with comprehensive error handling and compliance monitoring.
        
        Args:
            stock_code (str): Stock symbol
            exchange_code (str): Exchange code
            
        Returns:
            APIResponse: Quote data with metadata
        """
        self.logger.info(f"Fetching quotes for {stock_code} on {exchange_code}")
        
        # Apply compliance decorator if available
        if COMPLIANCE_AVAILABLE:
            @compliance_decorator(DataProvider.BREEZE_CONNECT, "get_quotes")
            def _get_quotes_with_compliance():
                return self._do_get_quotes(stock_code, exchange_code)
            return _get_quotes_with_compliance()
        else:
            return self._do_get_quotes(stock_code, exchange_code)
    
    def _do_get_quotes(self, stock_code: str, exchange_code: str) -> APIResponse:
        """Internal get quotes implementation"""
        try:
            # Validate inputs
            if not stock_code or not exchange_code:
                raise ValidationError("Stock code and exchange code are required")
            
            response = self._make_api_call(
                self.breeze.get_quotes,
                stock_code=stock_code,
                exchange_code=exchange_code
            )
            
            if response.success and not response.data.empty:
                # Extract LTP if available
                ltp = float(response.data.iloc[0]['ltp']) if 'ltp' in response.data.columns else None
                response.metadata['ltp'] = ltp
                self.logger.info(f"📦 LTP for {stock_code}: {ltp}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error fetching quotes for {stock_code}: {e}")
            return APIResponse(
                success=False,
                data=None,
                errors=[str(e)],
                warnings=[],
                metadata={'stock_code': stock_code, 'exchange_code': exchange_code},
                response_time=0
            )
    
    def get_historical_data_safe(self, request: MarketDataRequest) -> APIResponse:
        """
        Safely fetch historical data with comprehensive validation, error handling, and compliance monitoring.
        
        Args:
            request (MarketDataRequest): Data request parameters
            
        Returns:
            APIResponse: Historical data with metadata
        """
        self.logger.info(f"Fetching {request.product_type} data for {request.stock_code}")
        
        # Apply compliance decorator if available
        if COMPLIANCE_AVAILABLE:
            @compliance_decorator(DataProvider.BREEZE_CONNECT, "get_historical_data")
            def _get_historical_data_with_compliance():
                return self._do_get_historical_data(request)
            return _get_historical_data_with_compliance()
        else:
            return self._do_get_historical_data(request)
    
    def _do_get_historical_data(self, request: MarketDataRequest) -> APIResponse:
        """Internal historical data implementation"""
        try:
            # Validate request
            self._validate_data_request(request)
            
            # Prepare API parameters
            params = {
                'interval': request.interval,
                'from_date': request.from_date,
                'to_date': request.to_date,
                'stock_code': request.stock_code,
                'exchange_code': request.exchange_code,
                'product_type': request.product_type
            }
            
            # Add optional parameters
            if request.product_type == "futures" and request.expiry_date:
                params['expiry_date'] = request.expiry_date
                self.logger.debug(f"Futures expiry: {request.expiry_date}")
                
            if request.product_type == "options":
                if not all([request.expiry_date, request.strike_price, request.right]):
                    raise ValidationError("Options require expiry_date, strike_price, and right")
                params.update({
                    'expiry_date': request.expiry_date,
                    'strike_price': request.strike_price,
                    'right': request.right.lower()
                })
                self.logger.debug(f"Options: {request.right.upper()} {request.strike_price} exp {request.expiry_date}")
            
            # Make API call
            response = self._make_api_call(
                self.breeze.get_historical_data_v2,
                **params
            )
            
            if response.success:
                # Add request metadata
                response.metadata.update({
                    'request_params': params,
                    'data_type': request.product_type,
                    'records_fetched': len(response.data) if response.data is not None else 0
                })
                
                self.logger.info(f"✅ Fetched {len(response.data)} records for {request.stock_code}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return APIResponse(
                success=False,
                data=None,
                errors=[str(e)],
                warnings=[],
                metadata={'request': request.__dict__},
                response_time=0
            )
    
    def _validate_data_request(self, request: MarketDataRequest):
        """
        Validate data request parameters.
        
        Args:
            request (MarketDataRequest): Request to validate
            
        Raises:
            ValidationError: If request is invalid
        """
        if not request.stock_code:
            raise ValidationError("Stock code is required")
            
        if not request.exchange_code:
            raise ValidationError("Exchange code is required")
            
        valid_products = ['cash', 'futures', 'options']
        if request.product_type not in valid_products:
            raise ValidationError(f"Product type must be one of: {valid_products}")
            
        valid_intervals = ['1minute', '5minute', '30minute', '1day']
        if request.interval not in valid_intervals:
            raise ValidationError(f"Interval must be one of: {valid_intervals}")
            
        # Validate date format
        try:
            datetime.fromisoformat(request.from_date.replace('Z', ''))
            datetime.fromisoformat(request.to_date.replace('Z', ''))
        except ValueError as e:
            raise ValidationError(f"Invalid date format: {e}")
    
    def get_date_iso(self, days_ago: int = 0) -> str:
        """
        Get ISO formatted date string with error handling.
        
        Args:
            days_ago (int): Number of days ago
            
        Returns:
            str: ISO formatted date string
        """
        try:
            date = datetime.now() - timedelta(days=days_ago)
            date = date.replace(hour=9, minute=0, second=0, microsecond=0)
            return date.isoformat() + ISO_DATETIME_SUFFIX
        except Exception as e:
            self.logger.error(f"Error generating date ISO: {e}")
            # Fallback to current time
            return datetime.now().isoformat() + ISO_DATETIME_SUFFIX
    
    def get_end_date_iso(self) -> str:
        """
        Get end of trading day ISO formatted date string.
        
        Returns:
            str: ISO formatted date string
        """
        try:
            date = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
            return date.isoformat() + ISO_DATETIME_SUFFIX
        except Exception as e:
            self.logger.error(f"Error generating end date ISO: {e}")
            return datetime.now().isoformat() + ISO_DATETIME_SUFFIX
    
    def get_last_trading_day(self, days_back: int = 0) -> datetime:
        """
        Get the last trading day with comprehensive error handling.
        
        Args:
            days_back (int): Number of days to go back
            
        Returns:
            datetime: Last trading day
        """
        try:
            if not self.breeze:
                self.logger.warning("Breeze not authenticated, using current date")
                return datetime.now() - timedelta(days=days_back)
            
            date = datetime.now() - timedelta(days=days_back)
            attempts = 0
            max_attempts = 10
            
            while attempts < max_attempts:
                # Skip weekends
                if date.weekday() >= 5:
                    date -= timedelta(days=1)
                    attempts += 1
                    continue
                
                try:
                    # Verify trading day by checking market data
                    test_request = MarketDataRequest(
                        stock_code="NIFTY",
                        exchange_code="NSE",
                        product_type="cash",
                        interval="1day",
                        from_date=self.get_date_iso(0),
                        to_date=self.get_end_date_iso()
                    )
                    
                    response = self.get_historical_data_safe(test_request)
                    if response.success and not response.data.empty:
                        self.logger.debug(f"Confirmed trading day: {date.date()}")
                        return date
                        
                except Exception as e:
                    self.logger.debug(f"Error checking trading day {date.date()}: {e}")
                
                date -= timedelta(days=1)
                attempts += 1
            
            # Fallback
            self.logger.warning(f"Could not determine trading day after {max_attempts} attempts, using current date")
            return datetime.now() - timedelta(days=days_back)
            
        except Exception as e:
            self.logger.error(f"Error determining last trading day: {e}")
            return datetime.now() - timedelta(days=days_back)
    
    def save_data_safe(self, data: pd.DataFrame, filename: str, 
                      additional_metadata: Optional[Dict] = None,
                      strategy: SaveStrategy = None,
                      file_format: FileFormat = FileFormat.CSV) -> ProcessingResult:
        """
        🛡️ FIXED: Enhanced safe data saving with versioning and backup support.
        
        IMPROVEMENTS:
        - Automatic file versioning to prevent overwrites
        - Backup creation for existing files
        - Comprehensive metadata tracking
        - Multiple file format support
        - File existence checks
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
            additional_metadata (Optional[Dict]): Additional metadata to save
            strategy (SaveStrategy): Strategy for handling existing files
            file_format (FileFormat): File format to save in
            
        Returns:
            ProcessingResult: Comprehensive result with save operation details
        """
        try:
            if data.empty:
                self.logger.warning(f"Data is empty, not saving {filename}")
                return ProcessingResult(
                    data=pd.DataFrame(),
                    success=False,
                    quality=DataQuality.INVALID,
                    processing_time=0.0,
                    error_message="Data is empty"
                )
            
            # Use the safe file manager for enhanced saving
            save_result = self.file_manager.save_dataframe(
                df=data,
                filename=filename,
                strategy=strategy,
                file_format=file_format,
                metadata=additional_metadata
            )
            
            if save_result.success:
                self.logger.info(f"✅ Data saved safely: {save_result.final_filename} ({data.shape[0]} rows, {data.shape[1]} cols)")
                if save_result.backup_created:
                    self.logger.info(f"🔄 Backup created: {save_result.backup_path}")
                if save_result.strategy_used != SaveStrategy.OVERWRITE:
                    self.logger.info(f"🛡️ Strategy used: {save_result.strategy_used.value}")
                
                return ProcessingResult(
                    data=data,
                    success=True,
                    quality=DataQuality.GOOD,
                    processing_time=0.0,
                    metadata={
                        'save_result': save_result.__dict__,
                        'additional_metadata': additional_metadata
                    }
                )
            else:
                return ProcessingResult(
                    data=pd.DataFrame(),
                    success=False,
                    quality=DataQuality.INVALID,
                    processing_time=0.0,
                    error_message=save_result.error_message
                )
            
        except Exception as e:
            error_msg = f"Error in safe data saving: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                data=pd.DataFrame(),
                success=False,
                quality=DataQuality.INVALID,
                processing_time=0.0,
                error_message=error_msg
            )
    
    def save_data_with_backup(self, data: pd.DataFrame, filename: str, 
                             additional_metadata: Optional[Dict] = None) -> ProcessingResult:
        """
        Save data with automatic backup of existing files.
        
        This is a convenience method that uses BACKUP_OVERWRITE strategy.
        """
        return self.save_data_safe(
            data=data,
            filename=filename,
            additional_metadata=additional_metadata,
            strategy=SaveStrategy.BACKUP_OVERWRITE
        )
    
    def save_data_versioned(self, data: pd.DataFrame, filename: str,
                           additional_metadata: Optional[Dict] = None) -> ProcessingResult:
        """
        Save data with automatic versioning (file_v1.csv, file_v2.csv, etc.).
        
        This is a convenience method that uses VERSION strategy.
        """
        return self.save_data_safe(
            data=data,
            filename=filename,
            additional_metadata=additional_metadata,
            strategy=SaveStrategy.VERSION
        )
    
    def save_data_timestamped(self, data: pd.DataFrame, filename: str,
                             additional_metadata: Optional[Dict] = None) -> ProcessingResult:
        """
        Save data with timestamp in filename.
        
        This is a convenience method that uses TIMESTAMP strategy.
        """
        return self.save_data_safe(
            data=data,
            filename=filename,
            additional_metadata=additional_metadata,
            strategy=SaveStrategy.TIMESTAMP
        )

class OptionChainAnalyzer:
    """Advanced option chain analysis with comprehensive error handling."""
    
    def __init__(self, breeze_manager: EnhancedBreezeDataManager):
        self.logger = logger.getChild(self.__class__.__name__)
        self.breeze_manager = breeze_manager
        
    def get_valid_expiry_dates(self, stock_code: str, max_weeks: int = 4) -> List[str]:
        """
        Get valid expiry dates with comprehensive validation.
        
        Args:
            stock_code (str): Stock symbol
            max_weeks (int): Maximum weeks to check
            
        Returns:
            List[str]: Valid expiry dates
        """
        self.logger.info(f"Finding valid expiry dates for {stock_code}")
        
        try:
            today = datetime.today()
            valid_expiries = []
            
            for i in range(max_weeks):
                # Calculate next Thursday
                days_ahead = ((3 - today.weekday()) % 7) + (i * 7)
                potential_expiry = today + timedelta(days=days_ahead)
                expiry_str = potential_expiry.strftime('%Y-%m-%d')
                
                self.logger.debug(f"🔍 Checking expiry: {expiry_str}")
                
                try:
                    # Test with futures data
                    request = MarketDataRequest(
                        stock_code=stock_code,
                        exchange_code="NFO",
                        product_type="futures",
                        interval="30minute",
                        from_date=self.breeze_manager.get_last_trading_day(5).strftime('%Y-%m-%d') + "T09:00:00" + ISO_DATETIME_SUFFIX,
                        to_date=self.breeze_manager.get_last_trading_day(0).strftime('%Y-%m-%d') + "T15:45:00" + ISO_DATETIME_SUFFIX,
                        expiry_date=expiry_str
                    )
                    
                    response = self.breeze_manager.get_historical_data_safe(request)
                    
                    if response.success and not response.data.empty:
                        valid_expiries.append(expiry_str)
                        self.logger.info(f"✅ Valid expiry found: {expiry_str}")
                        
                except Exception as e:
                    self.logger.debug(f"⚠️ Error checking expiry {expiry_str}: {e}")
            
            self.logger.info(f"Found {len(valid_expiries)} valid expiry dates")
            return valid_expiries
            
        except Exception as e:
            self.logger.error(f"Error finding valid expiry dates: {e}")
            return []
    
    def fetch_option_chain_safe(self, stock_code: str, expiry_date: str, 
                               ltp: float, strike_range: int = 800) -> ProcessingResult:
        """
        Safely fetch complete option chain with comprehensive error handling.
        
        Args:
            stock_code (str): Stock symbol
            expiry_date (str): Expiry date
            ltp (float): Last traded price
            strike_range (int): Strike range around LTP
            
        Returns:
            ProcessingResult: Complete option chain data
        """
        start_time = datetime.now()
        self.logger.info(f"Fetching option chain for {stock_code} expiry {expiry_date}")
        
        try:
            if not expiry_date:
                raise ValidationError("No expiry date provided")
                
            if ltp <= 0:
                raise ValidationError("Invalid LTP provided")
            
            # Determine strike step
            strike_step = 20 if stock_code.upper() == "TCS" else 50
            atm_strike = round(ltp / strike_step) * strike_step
            
            # Generate strike range
            min_strike = atm_strike - strike_range
            max_strike = atm_strike + strike_range
            strikes = range(
                int(min_strike // strike_step * strike_step),
                int(max_strike // strike_step * strike_step) + strike_step,
                strike_step
            )
            
            self.logger.info(f"🔢 Fetching {len(strikes)} strikes from {min(strikes)} to {max(strikes)}")
            
            # Fetch all options data
            all_options_data = []
            successful_fetches = 0
            failed_fetches = 0
            errors = []
            
            for strike in strikes:
                for right in ['call', 'put']:
                    try:
                        self.logger.debug(f"🔄 Fetching {right.upper()} {strike}")
                        
                        request = MarketDataRequest(
                            stock_code=stock_code,
                            exchange_code="NFO",
                            product_type="options",
                            interval="5minute",
                            from_date=self.breeze_manager.get_last_trading_day(30).strftime('%Y-%m-%d') + "T09:00:00" + ISO_DATETIME_SUFFIX,
                            to_date=self.breeze_manager.get_last_trading_day(0).strftime('%Y-%m-%d') + "T15:30:00" + ISO_DATETIME_SUFFIX,
                            expiry_date=expiry_date,
                            strike_price=strike,
                            right=right
                        )
                        
                        response = self.breeze_manager.get_historical_data_safe(request)
                        
                        if response.success and not response.data.empty:
                            # Add option metadata
                            response.data['strike'] = strike
                            response.data['right'] = right
                            response.data['expiry_date'] = expiry_date
                            all_options_data.append(response.data)
                            successful_fetches += 1
                            self.logger.debug(f"✅ {right.upper()} {strike} fetched ({len(response.data)} records)")
                        else:
                            failed_fetches += 1
                            error_msg = f"Failed to fetch {right.upper()} {strike}: {response.errors}"
                            errors.append(error_msg)
                            self.logger.warning(error_msg)
                            
                    except Exception as e:
                        failed_fetches += 1
                        error_msg = f"Error fetching {right.upper()} {strike}: {e}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
            
            if not all_options_data:
                raise ProcessingError("No options data was successfully fetched")
            
            # Combine all data
            combined_df = pd.concat(all_options_data, ignore_index=True)
            
            # Add derived columns
            combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
            combined_df['date'] = combined_df['datetime'].dt.date
            combined_df['time'] = combined_df['datetime'].dt.time
            
            # Create option symbols
            expiry_date_str = pd.to_datetime(expiry_date).strftime('%d%b%y').upper()
            combined_df['symbol'] = (
                stock_code +
                expiry_date_str +
                combined_df['strike'].astype(int).astype(str) +
                combined_df['right'].str[0].str.upper()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Assess quality
            total_expected = len(strikes) * 2  # calls and puts
            success_rate = (successful_fetches / total_expected) * 100
            
            if success_rate > 90:
                quality = DataQuality.EXCELLENT
            elif success_rate > 75:
                quality = DataQuality.GOOD
            elif success_rate > 50:
                quality = DataQuality.FAIR
            else:
                quality = DataQuality.POOR
            
            result = ProcessingResult(
                data=combined_df,
                success=True,
                quality=quality,
                errors=errors,
                warnings=[],
                metadata={
                    'stock_code': stock_code,
                    'expiry_date': expiry_date,
                    'ltp': ltp,
                    'strike_range': strike_range,
                    'total_strikes': len(strikes),
                    'successful_fetches': successful_fetches,
                    'failed_fetches': failed_fetches,
                    'success_rate': success_rate,
                    'total_records': len(combined_df),
                    'unique_options': len(combined_df['symbol'].unique())
                },
                processing_time=processing_time
            )
            
            self.logger.info(f"Option chain fetch completed in {processing_time:.2f}s")
            self.logger.info(f"Success rate: {success_rate:.1f}% ({successful_fetches}/{total_expected})")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Critical error in option chain fetch: {e}")
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

# Utility functions for backward compatibility and convenience
def detect_strike_step(strikes: List[float]) -> Optional[int]:
    """
    Detect strike price step from a list of strikes.
    
    Args:
        strikes (List[float]): List of strike prices
        
    Returns:
        Optional[int]: Strike step or None if cannot determine
    """
    try:
        if len(strikes) < 2:
            return None
        return int(min(np.diff(sorted(strikes))))
    except Exception as e:
        logger.error(f"Error detecting strike step: {e}")
        return None

def get_nearest_strike_price(ltp: float, valid_step: int) -> Optional[int]:
    """
    Get nearest strike price based on LTP and step.
    
    Args:
        ltp (float): Last traded price
        valid_step (int): Strike step
        
    Returns:
        Optional[int]: Nearest strike price
    """
    try:
        if ltp and valid_step:
            return int(round(ltp / valid_step) * valid_step)
        return None
    except Exception as e:
        logger.error(f"Error calculating nearest strike: {e}")
        return None

# Export main classes and functions
__all__ = [
    'EnhancedBreezeDataManager',
    'OptionChainAnalyzer',
    'MarketDataRequest',
    'APIResponse',
    'MarketDataType',
    'APIError',
    'AuthenticationError',
    'DataFetchError',
    'detect_strike_step',
    'get_nearest_strike_price'
]
