"""
Application Configuration Module
===============================

Centralized configuration management for the Stock Price Predictor project.
Handles all application settings, paths, and environment configurations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration management for the application.
    Handles all settings, paths, and environment configurations.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to custom config file
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_file = config_file or self.project_root / "configs" / "config.json"
        
        # Default configuration
        self._default_config = {
            "data": {
                "save_path": "data",
                "raw_path": "data/raw",
                "processed_path": "data/processed",
                "cache_path": "data/cache",
                "plots_path": "data/plots",
                "reports_path": "data/reports"
            },
            "models": {
                "save_path": "models",
                "checkpoints_path": "models/checkpoints",
                "production_path": "models/production",
                "experiments_path": "models/experiments"
            },
            "api": {
                "rate_limit_delay": 1.0,
                "request_timeout": 30,
                "max_retries": 3,
                "cache_duration": 3600
            },
            "ml": {
                "random_seed": 42,
                "test_size": 0.2,
                "validation_size": 0.1,
                "cross_validation_folds": 5
            },
            "reporting": {
                "enable_html": True,
                "enable_json": True,
                "enable_plots": True,
                "plot_format": "png",
                "plot_dpi": 300
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_file": "logs/application.log"
            }
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info("Configuration loaded successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults
                config = self._default_config.copy()
                self._deep_update(config, file_config)
                return config
            else:
                logger.info(f"Config file not found: {self.config_file}. Using defaults.")
                return self._default_config.copy()
                
        except Exception as e:
            logger.warning(f"Error loading config file: {e}. Using defaults.")
            return self._default_config.copy()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update base dictionary with update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        directories = [
            self.get_data_save_path(),
            self.get_data_raw_path(),
            self.get_data_processed_path(),
            self.get_data_cache_path(),
            self.get_plots_path(),
            self.get_reports_path(),
            self.get_models_save_path(),
            self.get_models_checkpoints_path(),
            self.get_models_production_path(),
            self.get_models_experiments_path(),
            self.project_root / "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Data paths
    def get_data_save_path(self) -> str:
        """Get data save path."""
        return str(self.project_root / self.config["data"]["save_path"])
    
    def get_data_raw_path(self) -> str:
        """Get raw data path."""
        return str(self.project_root / self.config["data"]["raw_path"])
    
    def get_data_processed_path(self) -> str:
        """Get processed data path."""
        return str(self.project_root / self.config["data"]["processed_path"])
    
    def get_data_cache_path(self) -> str:
        """Get cache data path."""
        return str(self.project_root / self.config["data"]["cache_path"])
    
    def get_plots_path(self) -> str:
        """Get plots save path."""
        return str(self.project_root / self.config["data"]["plots_path"])
    
    def get_reports_path(self) -> str:
        """Get reports save path."""
        return str(self.project_root / self.config["data"]["reports_path"])
    
    # Model paths
    def get_models_save_path(self) -> str:
        """Get models save path."""
        return str(self.project_root / self.config["models"]["save_path"])
    
    def get_models_checkpoints_path(self) -> str:
        """Get model checkpoints path."""
        return str(self.project_root / self.config["models"]["checkpoints_path"])
    
    def get_models_production_path(self) -> str:
        """Get production models path."""
        return str(self.project_root / self.config["models"]["production_path"])
    
    def get_models_experiments_path(self) -> str:
        """Get experimental models path."""
        return str(self.project_root / self.config["models"]["experiments_path"])
    
    # API settings
    def get_api_rate_limit_delay(self) -> float:
        """Get API rate limit delay."""
        return self.config["api"]["rate_limit_delay"]
    
    def get_api_request_timeout(self) -> int:
        """Get API request timeout."""
        return self.config["api"]["request_timeout"]
    
    def get_api_max_retries(self) -> int:
        """Get API max retries."""
        return self.config["api"]["max_retries"]
    
    def get_api_cache_duration(self) -> int:
        """Get API cache duration."""
        return self.config["api"]["cache_duration"]
    
    # ML settings
    def get_ml_random_seed(self) -> int:
        """Get ML random seed."""
        return self.config["ml"]["random_seed"]
    
    def get_ml_test_size(self) -> float:
        """Get ML test size."""
        return self.config["ml"]["test_size"]
    
    def get_ml_validation_size(self) -> float:
        """Get ML validation size."""
        return self.config["ml"]["validation_size"]
    
    def get_ml_cv_folds(self) -> int:
        """Get cross-validation folds."""
        return self.config["ml"]["cross_validation_folds"]
    
    # Reporting settings
    def get_reporting_enable_html(self) -> bool:
        """Get HTML reporting enabled."""
        return self.config["reporting"]["enable_html"]
    
    def get_reporting_enable_json(self) -> bool:
        """Get JSON reporting enabled."""
        return self.config["reporting"]["enable_json"]
    
    def get_reporting_enable_plots(self) -> bool:
        """Get plots reporting enabled."""
        return self.config["reporting"]["enable_plots"]
    
    def get_plot_format(self) -> str:
        """Get plot format."""
        return self.config["reporting"]["plot_format"]
    
    def get_plot_dpi(self) -> int:
        """Get plot DPI."""
        return self.config["reporting"]["plot_dpi"]
    
    # Logging settings
    def get_logging_level(self) -> str:
        """Get logging level."""
        return self.config["logging"]["level"]
    
    def get_logging_format(self) -> str:
        """Get logging format."""
        return self.config["logging"]["format"]
    
    def get_log_file_path(self) -> str:
        """Get log file path."""
        return str(self.project_root / self.config["logging"]["log_file"])
    
    # Generic getters
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = filepath or self.config_file
        
        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all configured paths."""
        return {
            'data_save': self.get_data_save_path(),
            'data_raw': self.get_data_raw_path(),
            'data_processed': self.get_data_processed_path(),
            'data_cache': self.get_data_cache_path(),
            'plots': self.get_plots_path(),
            'reports': self.get_reports_path(),
            'models_save': self.get_models_save_path(),
            'models_checkpoints': self.get_models_checkpoints_path(),
            'models_production': self.get_models_production_path(),
            'models_experiments': self.get_models_experiments_path(),
            'logs': str(self.project_root / "logs")
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(project_root={self.project_root}, config_file={self.config_file})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return self.__str__()


# Global configuration instance
config = Config()

logger.info("Application configuration module loaded successfully")
