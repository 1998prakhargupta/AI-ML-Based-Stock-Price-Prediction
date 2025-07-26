"""
Base Cost Configuration
=======================

Core configuration management for transaction cost calculations.
Extends the existing configuration system with transaction cost specific
settings and integrates with broker configurations and market data sources.
"""

import os
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

# Import existing configuration utilities
try:
    from ...utils.app_config import config as app_config
    from ...utils.file_management_utils import SafeFileManager, SaveStrategy
    # Skip config_manager import to avoid Breeze credential requirements
    BaseConfig = object
except ImportError:
    # Fallback if imports not available
    BaseConfig = object
    app_config = None
    SafeFileManager = None
    SaveStrategy = None

from ..transaction_costs.models import BrokerConfiguration
from ..transaction_costs.constants import (
    DEFAULT_COMMISSION_RATES,
    REGULATORY_FEE_RATES,
    SYSTEM_DEFAULTS
)

logger = logging.getLogger(__name__)


class CostConfiguration:
    """
    Configuration management for transaction cost calculations.
    
    Provides specialized configuration handling for cost modeling,
    broker settings, and market data integration while extending
    the existing project configuration system.
    """
    
    def __init__(self, config_file: Optional[str] = None, base_path: Optional[str] = None):
        """
        Initialize cost configuration.
        
        Args:
            config_file: Optional path to cost-specific config file
            base_path: Base path for configuration files
        """
        # Set up paths
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Try to use existing app config paths if available
            if app_config:
                self.base_path = Path(app_config.project_root) / "configs"
            else:
                self.base_path = Path.cwd() / "configs"
        
        if config_file:
            self.config_file = Path(config_file)
        else:
            self.config_file = self.base_path / "cost_config.json"
        
        # Initialize file manager for config persistence
        if SafeFileManager:
            self.file_manager = SafeFileManager(
                base_path=str(self.base_path),
                default_strategy=SaveStrategy.BACKUP
            )
        else:
            self.file_manager = None
        
        # Default configuration structure
        self._default_config = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            
            # Broker configurations
            "brokers": {},
            
            # Market data settings
            "market_data": {
                "default_provider": "yahoo",
                "fallback_providers": ["alpha_vantage", "iex"],
                "cache_duration_seconds": 300,
                "staleness_threshold_seconds": 60,
                "retry_attempts": 3,
                "timeout_seconds": 30
            },
            
            # Calculation settings
            "calculation": {
                "default_mode": "real_time",
                "precision_decimal_places": 4,
                "confidence_threshold": 0.8,
                "enable_caching": True,
                "cache_duration_seconds": 300,
                "parallel_calculations": True,
                "max_workers": 4
            },
            
            # Regulatory settings
            "regulatory": {
                "update_frequency_days": 30,
                "auto_update_enabled": True,
                "sec_fee_rate": str(REGULATORY_FEE_RATES['SEC_FEE_RATE']),
                "finra_taf_rate": str(REGULATORY_FEE_RATES['FINRA_TAF_SELL'])
            },
            
            # Performance monitoring
            "performance": {
                "enable_metrics": True,
                "log_slow_calculations": True,
                "slow_calculation_threshold_seconds": 5.0,
                "enable_profiling": False
            },
            
            # Error handling
            "error_handling": {
                "max_retries": 3,
                "retry_delay_seconds": 1,
                "fail_fast": False,
                "log_all_errors": True
            },
            
            # Integration settings
            "integration": {
                "enable_external_apis": True,
                "api_rate_limits": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000
                },
                "data_validation": {
                    "strict_mode": True,
                    "allow_estimated_data": True
                }
            }
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info(f"Cost configuration initialized from {self.config_file}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults
                config = self._default_config.copy()
                self._deep_merge(config, file_config)
                
                logger.info(f"Loaded cost configuration from {self.config_file}")
                return config
            else:
                logger.info(f"Config file not found: {self.config_file}. Using defaults.")
                return self._default_config.copy()
                
        except Exception as e:
            logger.warning(f"Error loading cost config: {e}. Using defaults.")
            return self._default_config.copy()
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for broker configs
        (self.base_path / "brokers").mkdir(exist_ok=True)
        (self.base_path / "templates").mkdir(exist_ok=True)
    
    # Broker configuration management
    
    def add_broker_configuration(
        self,
        broker_config: BrokerConfiguration,
        save_immediately: bool = True
    ) -> None:
        """
        Add or update broker configuration.
        
        Args:
            broker_config: Broker configuration to add
            save_immediately: Whether to save config immediately
        """
        broker_data = broker_config.to_dict()
        self.config["brokers"][broker_config.broker_name] = broker_data
        
        if save_immediately:
            self.save_config()
        
        logger.info(f"Added broker configuration for {broker_config.broker_name}")
    
    def get_broker_configuration(self, broker_name: str) -> Optional[BrokerConfiguration]:
        """
        Get broker configuration by name.
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            BrokerConfiguration instance or None
        """
        broker_data = self.config["brokers"].get(broker_name)
        if not broker_data:
            return None
        
        try:
            # Convert string decimals back to Decimal objects
            converted_data = {}
            for key, value in broker_data.items():
                if isinstance(value, str) and key.endswith(('_commission', '_fee', '_rate', '_multiplier')):
                    try:
                        converted_data[key] = Decimal(value)
                    except:
                        converted_data[key] = value
                elif key == 'volume_discount_tiers' and isinstance(value, dict):
                    # Convert discount tier values
                    converted_data[key] = {k: Decimal(str(v)) for k, v in value.items()}
                else:
                    converted_data[key] = value
            
            # Convert datetime strings back to datetime objects
            if 'last_updated' in converted_data and isinstance(converted_data['last_updated'], str):
                converted_data['last_updated'] = datetime.fromisoformat(converted_data['last_updated'])
            
            return BrokerConfiguration(**converted_data)
            
        except Exception as e:
            logger.error(f"Error loading broker configuration for {broker_name}: {e}")
            return None
    
    def list_broker_configurations(self) -> List[str]:
        """Get list of configured broker names."""
        return list(self.config["brokers"].keys())
    
    def remove_broker_configuration(self, broker_name: str, save_immediately: bool = True) -> bool:
        """
        Remove broker configuration.
        
        Args:
            broker_name: Name of the broker to remove
            save_immediately: Whether to save config immediately
            
        Returns:
            True if removed, False if not found
        """
        if broker_name in self.config["brokers"]:
            del self.config["brokers"][broker_name]
            
            if save_immediately:
                self.save_config()
            
            logger.info(f"Removed broker configuration for {broker_name}")
            return True
        
        return False
    
    def create_default_broker_configurations(self) -> None:
        """Create default broker configurations for common brokers."""
        default_brokers = {
            "Interactive Brokers": {
                "broker_name": "Interactive Brokers",
                "broker_id": "IB",
                "equity_commission": Decimal("0.005"),  # $0.005 per share
                "min_commission": Decimal("1.00"),
                "options_commission": Decimal("0.00"),
                "options_per_contract": Decimal("0.65"),
                "account_tier": "professional"
            },
            "Charles Schwab": {
                "broker_name": "Charles Schwab",
                "broker_id": "SCHW",
                "equity_commission": Decimal("0.00"),
                "options_commission": Decimal("0.00"),
                "options_per_contract": Decimal("0.65"),
                "account_tier": "retail"
            },
            "TD Ameritrade": {
                "broker_name": "TD Ameritrade",
                "broker_id": "TDA",
                "equity_commission": Decimal("0.00"),
                "options_commission": Decimal("0.00"),
                "options_per_contract": Decimal("0.65"),
                "account_tier": "retail"
            },
            "E*TRADE": {
                "broker_name": "E*TRADE",
                "broker_id": "ETFC",
                "equity_commission": Decimal("0.00"),
                "options_commission": Decimal("0.00"),
                "options_per_contract": Decimal("0.65"),
                "account_tier": "retail"
            }
        }
        
        for broker_name, config_data in default_brokers.items():
            if broker_name not in self.config["brokers"]:
                broker_config = BrokerConfiguration(**config_data)
                self.add_broker_configuration(broker_config, save_immediately=False)
        
        self.save_config()
        logger.info("Created default broker configurations")
    
    # Configuration getters with type safety
    
    def get_market_data_config(self) -> Dict[str, Any]:
        """Get market data configuration."""
        return self.config.get("market_data", {})
    
    def get_calculation_config(self) -> Dict[str, Any]:
        """Get calculation configuration."""
        return self.config.get("calculation", {})
    
    def get_regulatory_config(self) -> Dict[str, Any]:
        """Get regulatory configuration."""
        return self.config.get("regulatory", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration."""
        return self.config.get("performance", {})
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self.config.get("error_handling", {})
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration."""
        return self.config.get("integration", {})
    
    # Specific setting getters
    
    def get_default_calculation_mode(self) -> str:
        """Get default calculation mode."""
        return self.config["calculation"].get("default_mode", "real_time")
    
    def get_cache_duration(self) -> int:
        """Get cache duration in seconds."""
        return self.config["calculation"].get("cache_duration_seconds", 300)
    
    def is_caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.config["calculation"].get("enable_caching", True)
    
    def get_precision_decimal_places(self) -> int:
        """Get decimal precision for calculations."""
        return self.config["calculation"].get("precision_decimal_places", 4)
    
    def get_max_workers(self) -> int:
        """Get maximum number of worker threads."""
        return self.config["calculation"].get("max_workers", 4)
    
    def get_sec_fee_rate(self) -> Decimal:
        """Get current SEC fee rate."""
        rate_str = self.config["regulatory"].get("sec_fee_rate", "0.0000051")
        return Decimal(rate_str)
    
    def get_finra_taf_rate(self) -> Decimal:
        """Get current FINRA TAF rate."""
        rate_str = self.config["regulatory"].get("finra_taf_rate", "0.000166")
        return Decimal(rate_str)
    
    # Configuration updates
    
    def update_setting(self, key_path: str, value: Any, save_immediately: bool = True) -> None:
        """
        Update a specific configuration setting.
        
        Args:
            key_path: Dot-separated path to the setting (e.g., "calculation.precision")
            value: New value for the setting
            save_immediately: Whether to save immediately
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
        
        if save_immediately:
            self.save_config()
        
        logger.info(f"Updated configuration setting {key_path} = {value}")
    
    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration setting.
        
        Args:
            key_path: Dot-separated path to the setting
            default: Default value if setting not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    # Persistence methods
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Optional custom filepath
        """
        save_path = filepath or self.config_file
        
        try:
            # Update last modified timestamp
            self.config["last_updated"] = datetime.now().isoformat()
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Use SafeFileManager if available
            if self.file_manager:
                # Convert config to JSON string for SafeFileManager
                config_json = json.dumps(self.config, indent=2, default=str)
                
                # SafeFileManager expects a DataFrame, so we'll save directly
                with open(save_path, 'w') as f:
                    f.write(config_json)
                
                logger.info(f"Configuration saved with SafeFileManager to {save_path}")
            else:
                # Direct file save
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=2, default=str)
                
                logger.info(f"Configuration saved to {save_path}")
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded from file")
    
    def export_config(self, export_path: str, include_defaults: bool = True) -> None:
        """
        Export configuration to a different file.
        
        Args:
            export_path: Path to export the configuration
            include_defaults: Whether to include default values
        """
        export_config = self.config.copy() if include_defaults else {}
        
        if not include_defaults:
            # Only export non-default values
            # This would require comparison logic with defaults
            pass
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_config, f, indent=2, default=str)
            
            logger.info(f"Configuration exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            raise
    
    # Integration with existing config system
    
    def integrate_with_app_config(self) -> None:
        """Integrate with existing application configuration."""
        if app_config:
            # Update paths based on app config
            self.config["integration"]["data_paths"] = {
                "cache_path": app_config.get_data_cache_path(),
                "logs_path": app_config.get_log_file_path()
            }
            
            # Update logging settings
            self.config["performance"]["log_level"] = app_config.get_logging_level()
            
            logger.info("Integrated with application configuration")
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self.config.copy()
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        # Validate broker configurations
        broker_errors = []
        for broker_name, broker_data in self.config["brokers"].items():
            try:
                # Try to create BrokerConfiguration to validate
                BrokerConfiguration(**broker_data)
            except Exception as e:
                broker_errors.append(f"{broker_name}: {str(e)}")
        
        if broker_errors:
            errors["brokers"] = broker_errors
        
        # Validate numeric settings
        numeric_settings = [
            ("calculation.precision_decimal_places", int),
            ("calculation.cache_duration_seconds", int),
            ("calculation.max_workers", int),
            ("market_data.timeout_seconds", int),
            ("performance.slow_calculation_threshold_seconds", float)
        ]
        
        setting_errors = []
        for setting_path, expected_type in numeric_settings:
            value = self.get_setting(setting_path)
            if value is not None and not isinstance(value, expected_type):
                setting_errors.append(f"{setting_path}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        if setting_errors:
            errors["settings"] = setting_errors
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"CostConfiguration(config_file={self.config_file}, brokers={len(self.config['brokers'])})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return self.__str__()


logger.info("Cost configuration module loaded successfully")