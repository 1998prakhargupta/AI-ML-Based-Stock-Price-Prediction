"""
Configuration management for the stock prediction project.
This file handles all configuration loading from environment variables and config files.
"""

import os
import json
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration class that loads settings from environment variables and config files."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'config.json'
        self._config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment variables and config file."""
        # Load from environment variables first (highest priority)
        self._config.update({
            'BREEZE_API_KEY': os.getenv('BREEZE_API_KEY'),
            'BREEZE_API_SECRET': os.getenv('BREEZE_API_SECRET'),
            'BREEZE_SESSION_TOKEN': os.getenv('BREEZE_SESSION_TOKEN'),
            'GOOGLE_DRIVE_PATH': os.getenv('GOOGLE_DRIVE_PATH', '/content/drive/MyDrive/'),
            'DATA_SAVE_PATH': os.getenv('DATA_SAVE_PATH'),
            'STOCK_NAME': os.getenv('STOCK_NAME', 'TCS'),
            'INTERVAL': os.getenv('INTERVAL', '5minute'),
            'START_DATE': os.getenv('START_DATE', '2025-02-01'),
            'END_DATE': os.getenv('END_DATE', '2025-04-28'),
            'MODEL_SAVE_PATH': os.getenv('MODEL_SAVE_PATH')
        })
        
        # Load from config file if it exists (lower priority)
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Only use file config if env var is not set
                    for key, value in file_config.items():
                        if self._config.get(key) is None:
                            self._config[key] = value
                logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file {self.config_file}: {e}")
        
        # Validate required credentials
        self._validate_credentials()
    
    def _validate_credentials(self):
        """Validate that required credentials are present."""
        required_for_breeze = ['BREEZE_API_KEY', 'BREEZE_API_SECRET', 'BREEZE_SESSION_TOKEN']
        missing_breeze = [key for key in required_for_breeze if not self._config.get(key)]
        
        if missing_breeze:
            logger.error(f"Missing Breeze credentials: {missing_breeze}")
            logger.error("Please set environment variables or add to config.json file")
            raise ValueError(f"Missing required Breeze credentials: {missing_breeze}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def get_breeze_credentials(self) -> Dict[str, str]:
        """Get Breeze API credentials."""
        return {
            'api_key': self._config['BREEZE_API_KEY'],
            'api_secret': self._config['BREEZE_API_SECRET'],
            'session_token': self._config['BREEZE_SESSION_TOKEN']
        }
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get data storage paths."""
        google_drive = self._config.get('GOOGLE_DRIVE_PATH', '/content/drive/MyDrive/')
        data_path = self._config.get('DATA_SAVE_PATH') or os.path.join(google_drive, 'BreezeData')
        model_path = self._config.get('MODEL_SAVE_PATH') or os.path.join(google_drive, 'Models')
        
        return {
            'google_drive': google_drive,
            'data_save_path': data_path,
            'model_save_path': model_path
        }
    
    def get_trading_params(self) -> Dict[str, str]:
        """Get trading parameters."""
        return {
            'stock_name': self._config.get('STOCK_NAME', 'TCS'),
            'interval': self._config.get('INTERVAL', '5minute'),
            'start_date': self._config.get('START_DATE', '2025-02-01'),
            'end_date': self._config.get('END_DATE', '2025-04-28')
        }
    
    def get_index_symbols(self) -> Dict[str, str]:
        """Get NSE index symbols for yfinance."""
        default_symbols = {
            "NIFTY50": "^NSEI",
            "NIFTYNEXT50": "^NSMIDCP",
            "NIFTY100": "^CNX100",
            "NIFTY500": "^CRSLDX",
            "NIFTYMIDCAP50": "^NSEMDCP50",
            "NIFTYMIDCAP100": "NIFTY_MIDCAP_100.NS",
            "BANKNIFTY": "^NSEBANK",
            "NIFTYIT": "^CNXIT",
            "NIFTYPSE": "^CNXPSE",
            "NIFTYFMCG": "^CNXFMCG",
            "NIFTYENERGY": "^CNXENERGY",
            "NIFTYINFRA": "^CNXINFRA",
            "NIFTYPHARMA": "^CNXPHARMA",
            "NIFTYAUTO": "^CNXAUTO",
            "NIFTYMETAL": "^CNXMETAL",
            "NIFTYREALTY": "^CNXREALTY",
            "NIFTYCONSUMPTION": "^CNXCONSUM",
            "NIFTYCOMMODITIES": "^CNXCMDT",
            "NIFTYMEDIA": "^CNXMEDIA",
            "NIFTYPSUBANK": "^CNXPSUBANK",
            "NIFTYPVTBANK": "NIFTY_PVT_BANK.NS",
            "NIFTYSERVSECTOR": "^CNXSERVICE",
            "NIFTYDIVOPPS50": "^CNXDIVOP",
            "SENSEX": "^BSESN",
            "NIFTYFINSERVICE": "NIFTY_FIN_SERVICE.NS"
        }
        return self._config.get('INDEX_SYMBOLS', default_symbols)
    
    def get_data_save_path(self) -> str:
        """Get the data save path."""
        paths = self.get_data_paths()
        return paths['data_save_path']
    
    def get_model_save_path(self) -> str:
        """Get the model save path."""
        paths = self.get_data_paths()
        return paths['model_save_path']

# Global config instance
config = Config()
