"""
Configuration Manager for Transaction Cost Modeling
===================================================

Comprehensive configuration management system that provides hierarchical 
configuration loading, environment-specific overrides, runtime updates,
and configuration change notifications.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import threading
from dataclasses import dataclass, field
from enum import Enum

from .base_config import CostConfiguration
from .config_validator import CostConfigurationValidator, ValidationResult


logger = logging.getLogger(__name__)


class ConfigurationPriority(Enum):
    """Configuration source priority levels."""
    SYSTEM_DEFAULT = 1
    CONFIG_FILE = 2
    ENVIRONMENT_FILE = 3
    ENVIRONMENT_VARIABLE = 4
    RUNTIME_OVERRIDE = 5


@dataclass
class ConfigurationSource:
    """Represents a configuration source with metadata."""
    source_type: ConfigurationPriority
    source_path: Optional[str] = None
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """
    Advanced configuration manager for transaction cost modeling.
    
    Provides:
    - Hierarchical configuration loading with priority system
    - Environment-specific configuration files
    - Runtime configuration updates
    - Configuration change notifications
    - Automatic configuration reloading
    - Configuration validation and defaults
    """
    
    def __init__(
        self,
        base_config_path: Optional[str] = None,
        environment: Optional[str] = None,
        auto_reload: bool = True,
        validation_enabled: bool = True
    ):
        """
        Initialize the configuration manager.
        
        Args:
            base_config_path: Base path for configuration files
            environment: Current environment (development, staging, production)
            auto_reload: Whether to automatically reload configs when files change
            validation_enabled: Whether to validate configurations
        """
        self.base_config_path = Path(base_config_path) if base_config_path else Path.cwd() / "configs"
        self.cost_config_path = self.base_config_path / "cost_config"
        self.environment = environment or self._detect_environment()
        self.auto_reload = auto_reload
        self.validation_enabled = validation_enabled
        
        # Configuration sources by priority
        self._sources: Dict[ConfigurationPriority, ConfigurationSource] = {}
        
        # Configuration cache and state
        self._merged_config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()
        self._change_listeners: List[Callable[[Dict[str, Any]], None]] = []
        
        # File watchers for auto-reload
        self._watched_files: Dict[str, float] = {}
        
        # Validation
        self._validator = CostConfigurationValidator(strict_mode=True) if validation_enabled else None
        
        # Initialize
        self._ensure_directories()
        self._load_all_configurations()
        
        logger.info(f"ConfigurationManager initialized for environment: {self.environment}")
    
    def _detect_environment(self) -> str:
        """Detect current environment from various sources."""
        # Check environment variable first
        env = os.getenv('COST_CONFIG_ENV', os.getenv('APP_ENV', os.getenv('ENV')))
        if env:
            return env.lower()
        
        # Check for common CI/development indicators
        if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
            return 'ci'
        
        if os.getenv('JUPYTER_SERVER_ROOT') or os.getenv('COLAB_GPU'):
            return 'development'
        
        # Default to development
        return 'development'
    
    def _ensure_directories(self) -> None:
        """Ensure required configuration directories exist."""
        directories = [
            self.cost_config_path,
            self.cost_config_path / "broker_templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_all_configurations(self) -> None:
        """Load configurations from all sources in priority order."""
        with self._config_lock:
            self._sources.clear()
            
            # 1. Load system defaults
            self._load_system_defaults()
            
            # 2. Load base configuration file
            self._load_base_config_file()
            
            # 3. Load environment-specific configuration
            self._load_environment_config()
            
            # 4. Load environment variables
            self._load_environment_variables()
            
            # 5. Merge all configurations
            self._merge_configurations()
            
            # 6. Validate if enabled
            if self.validation_enabled:
                self._validate_configuration()
            
            # 7. Setup file watching if auto-reload enabled
            if self.auto_reload:
                self._setup_file_watching()
    
    def _load_system_defaults(self) -> None:
        """Load system default configuration."""
        try:
            from .defaults import get_default_configuration
            default_config = get_default_configuration()
            
            self._sources[ConfigurationPriority.SYSTEM_DEFAULT] = ConfigurationSource(
                source_type=ConfigurationPriority.SYSTEM_DEFAULT,
                source_path="system_defaults",
                last_modified=datetime.now(),
                data=default_config
            )
            
            logger.debug("Loaded system default configuration")
            
        except ImportError:
            logger.warning("Default configuration module not found, using minimal defaults")
            self._sources[ConfigurationPriority.SYSTEM_DEFAULT] = ConfigurationSource(
                source_type=ConfigurationPriority.SYSTEM_DEFAULT,
                source_path="minimal_defaults",
                data={"version": "1.0.0", "brokers": {}, "calculation": {}, "market_data": {}}
            )
    
    def _load_base_config_file(self) -> None:
        """Load base configuration file."""
        config_file = self.cost_config_path / "default_cost_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                self._sources[ConfigurationPriority.CONFIG_FILE] = ConfigurationSource(
                    source_type=ConfigurationPriority.CONFIG_FILE,
                    source_path=str(config_file),
                    last_modified=datetime.fromtimestamp(config_file.stat().st_mtime),
                    data=config_data
                )
                
                logger.debug(f"Loaded base configuration from {config_file}")
                
            except Exception as e:
                logger.error(f"Error loading base configuration from {config_file}: {e}")
        else:
            logger.info(f"Base configuration file not found: {config_file}")
    
    def _load_environment_config(self) -> None:
        """Load environment-specific configuration."""
        env_config_file = self.cost_config_path / f"{self.environment}.json"
        
        if env_config_file.exists():
            try:
                with open(env_config_file, 'r') as f:
                    env_config = json.load(f)
                
                self._sources[ConfigurationPriority.ENVIRONMENT_FILE] = ConfigurationSource(
                    source_type=ConfigurationPriority.ENVIRONMENT_FILE,
                    source_path=str(env_config_file),
                    last_modified=datetime.fromtimestamp(env_config_file.stat().st_mtime),
                    data=env_config
                )
                
                logger.debug(f"Loaded environment configuration from {env_config_file}")
                
            except Exception as e:
                logger.error(f"Error loading environment configuration from {env_config_file}: {e}")
        else:
            logger.info(f"Environment configuration file not found: {env_config_file}")
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'COST_CONFIG_CACHE_DURATION': ('calculation', 'cache_duration_seconds'),
            'COST_CONFIG_MAX_WORKERS': ('calculation', 'max_workers'),
            'COST_CONFIG_PRECISION': ('calculation', 'precision_decimal_places'),
            'COST_CONFIG_DEFAULT_MODE': ('calculation', 'default_mode'),
            'COST_CONFIG_ENABLE_CACHING': ('calculation', 'enable_caching'),
            'COST_CONFIG_MARKET_PROVIDER': ('market_data', 'default_provider'),
            'COST_CONFIG_MARKET_TIMEOUT': ('market_data', 'timeout_seconds'),
            'COST_CONFIG_ENABLE_METRICS': ('performance', 'enable_metrics'),
            'COST_CONFIG_LOG_SLOW_CALCS': ('performance', 'log_slow_calculations'),
            'COST_CONFIG_MAX_RETRIES': ('error_handling', 'max_retries'),
            'COST_CONFIG_RETRY_DELAY': ('error_handling', 'retry_delay_seconds'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                if section not in env_config:
                    env_config[section] = {}
                env_config[section][key] = converted_value
                
                logger.debug(f"Loaded environment variable {env_var}={converted_value}")
        
        if env_config:
            self._sources[ConfigurationPriority.ENVIRONMENT_VARIABLE] = ConfigurationSource(
                source_type=ConfigurationPriority.ENVIRONMENT_VARIABLE,
                source_path="environment_variables",
                last_modified=datetime.now(),
                data=env_config
            )
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _merge_configurations(self) -> None:
        """Merge configurations from all sources by priority."""
        merged = {}
        
        # Sort sources by priority (lowest to highest)
        sorted_sources = sorted(
            self._sources.items(),
            key=lambda x: x[0].value
        )
        
        for priority, source in sorted_sources:
            self._deep_merge(merged, source.data)
            logger.debug(f"Merged configuration from {source.source_path} (priority: {priority.name})")
        
        # Store previous config for change detection
        previous_config = self._merged_config.copy()
        self._merged_config = merged
        
        # Notify listeners if configuration changed
        if previous_config != merged:
            self._notify_configuration_change(merged)
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_configuration(self) -> None:
        """Validate the merged configuration."""
        if not self._validator:
            return
        
        try:
            result = self._validator.validate_full_configuration(self._merged_config)
            
            if not result.is_valid:
                logger.error(f"Configuration validation failed with {len(result.errors)} errors:")
                for error in result.errors:
                    logger.error(f"  - {error}")
                # Don't raise exception, just log errors
            
            if result.warnings:
                logger.warning(f"Configuration validation warnings:")
                for warning in result.warnings:
                    logger.warning(f"  - {warning}")
            
            if result.info:
                for info in result.info:
                    logger.info(f"  - {info}")
            
        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
    
    def _setup_file_watching(self) -> None:
        """Setup file watching for auto-reload."""
        watched_files = []
        
        for source in self._sources.values():
            if source.source_path and Path(source.source_path).exists():
                watched_files.append(source.source_path)
        
        # Store file modification times
        for file_path in watched_files:
            try:
                self._watched_files[file_path] = Path(file_path).stat().st_mtime
            except Exception as e:
                logger.warning(f"Cannot watch file {file_path}: {e}")
    
    def check_for_configuration_changes(self) -> bool:
        """Check if any watched configuration files have changed."""
        if not self.auto_reload:
            return False
        
        changed_files = []
        
        for file_path, last_mtime in self._watched_files.items():
            try:
                current_mtime = Path(file_path).stat().st_mtime
                if current_mtime > last_mtime:
                    changed_files.append(file_path)
                    self._watched_files[file_path] = current_mtime
            except Exception as e:
                logger.warning(f"Cannot check file {file_path}: {e}")
        
        if changed_files:
            logger.info(f"Configuration files changed: {changed_files}")
            self._load_all_configurations()
            return True
        
        return False
    
    def reload_configuration(self) -> None:
        """Manually reload configuration from all sources."""
        logger.info("Manually reloading configuration")
        with self._config_lock:
            self._load_all_configurations()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the complete merged configuration."""
        with self._config_lock:
            # Check for file changes if auto-reload is enabled
            if self.auto_reload:
                self.check_for_configuration_changes()
            
            return self._merged_config.copy()
    
    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration setting using dot notation.
        
        Args:
            key_path: Dot-separated path to the setting (e.g., "calculation.precision")
            default: Default value if setting not found
            
        Returns:
            Configuration value or default
        """
        with self._config_lock:
            config = self.get_configuration()
            
            keys = key_path.split('.')
            value = config
            
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default
    
    def update_setting(
        self,
        key_path: str,
        value: Any,
        source: ConfigurationPriority = ConfigurationPriority.RUNTIME_OVERRIDE,
        persist: bool = False
    ) -> None:
        """
        Update a configuration setting at runtime.
        
        Args:
            key_path: Dot-separated path to the setting
            value: New value for the setting
            source: Priority level for this update
            persist: Whether to persist the change to file
        """
        with self._config_lock:
            # Update runtime overrides
            if source not in self._sources:
                self._sources[source] = ConfigurationSource(
                    source_type=source,
                    source_path="runtime_overrides",
                    last_modified=datetime.now(),
                    data={}
                )
            
            # Navigate to the parent of the target key
            keys = key_path.split('.')
            config_ref = self._sources[source].data
            
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the value
            config_ref[keys[-1]] = value
            self._sources[source].last_modified = datetime.now()
            
            # Re-merge configurations
            self._merge_configurations()
            
            # Validate if enabled
            if self.validation_enabled:
                self._validate_configuration()
            
            logger.info(f"Updated configuration setting {key_path} = {value} (source: {source.name})")
            
            # Persist to file if requested
            if persist:
                self._persist_runtime_overrides()
    
    def _persist_runtime_overrides(self) -> None:
        """Persist runtime overrides to a file."""
        try:
            runtime_source = self._sources.get(ConfigurationPriority.RUNTIME_OVERRIDE)
            if runtime_source and runtime_source.data:
                override_file = self.cost_config_path / "runtime_overrides.json"
                
                with open(override_file, 'w') as f:
                    json.dump(runtime_source.data, f, indent=2, default=str)
                
                logger.info(f"Persisted runtime overrides to {override_file}")
        except Exception as e:
            logger.error(f"Error persisting runtime overrides: {e}")
    
    def add_change_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Add a configuration change listener."""
        self._change_listeners.append(listener)
        logger.debug(f"Added configuration change listener: {listener.__name__}")
    
    def remove_change_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a configuration change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            logger.debug(f"Removed configuration change listener: {listener.__name__}")
    
    def _notify_configuration_change(self, new_config: Dict[str, Any]) -> None:
        """Notify all listeners of configuration changes."""
        for listener in self._change_listeners:
            try:
                listener(new_config)
            except Exception as e:
                logger.error(f"Error in configuration change listener {listener.__name__}: {e}")
    
    def get_broker_configuration(self, broker_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific broker."""
        config = self.get_configuration()
        return config.get("brokers", {}).get(broker_name)
    
    def list_available_brokers(self) -> List[str]:
        """Get list of configured brokers."""
        config = self.get_configuration()
        return list(config.get("brokers", {}).keys())
    
    def switch_environment(self, new_environment: str) -> None:
        """Switch to a different environment configuration."""
        logger.info(f"Switching environment from {self.environment} to {new_environment}")
        self.environment = new_environment
        self._load_all_configurations()
    
    def export_configuration(self, output_path: str, include_sources: bool = False) -> None:
        """
        Export current configuration to a file.
        
        Args:
            output_path: Path to save the configuration
            include_sources: Whether to include source metadata
        """
        try:
            config_data = self.get_configuration()
            
            if include_sources:
                export_data = {
                    "merged_configuration": config_data,
                    "sources": {
                        priority.name: {
                            "source_path": source.source_path,
                            "last_modified": source.last_modified.isoformat() if source.last_modified else None,
                            "data": source.data
                        }
                        for priority, source in self._sources.items()
                    },
                    "environment": self.environment,
                    "export_timestamp": datetime.now().isoformat()
                }
            else:
                export_data = config_data
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Configuration exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            raise
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get information about the current configuration state."""
        return {
            "environment": self.environment,
            "auto_reload": self.auto_reload,
            "validation_enabled": self.validation_enabled,
            "sources": {
                priority.name: {
                    "source_path": source.source_path,
                    "last_modified": source.last_modified.isoformat() if source.last_modified else None
                }
                for priority, source in self._sources.items()
            },
            "watched_files": len(self._watched_files),
            "change_listeners": len(self._change_listeners),
            "config_sections": list(self._merged_config.keys()),
            "broker_count": len(self._merged_config.get("brokers", {}))
        }
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"ConfigurationManager(env={self.environment}, sources={len(self._sources)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration manager."""
        return self.__str__()


logger.info("Configuration manager module loaded successfully")