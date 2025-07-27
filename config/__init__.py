"""
Enterprise Configuration Manager
===============================

Centralized configuration management for the Price Predictor project.
Supports environment-specific configurations, validation, and runtime updates.

Author: 1998prakhargupta
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import configparser

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigSource:
    """Configuration source metadata."""
    name: str
    path: str
    format: str
    priority: int
    environment: Optional[str] = None
    loaded: bool = False
    data: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """
    Enterprise-grade configuration manager with hierarchical loading,
    environment overrides, and validation.
    """
    
    def __init__(self, 
                 config_dir: str = "config",
                 environment: Optional[str] = None,
                 validate_on_load: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Base configuration directory
            environment: Target environment (development, testing, production)
            validate_on_load: Whether to validate configuration on load
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.validate_on_load = validate_on_load
        
        # Configuration sources (ordered by priority)
        self.sources: List[ConfigSource] = []
        self._config_cache: Dict[str, Any] = {}
        self._loaded = False
        
        # Initialize configuration sources
        self._initialize_sources()
        
        # Load configurations
        if self.validate_on_load:
            self.load_all()
    
    def _initialize_sources(self) -> None:
        """Initialize configuration sources in priority order."""
        # 1. Base application configuration (lowest priority)
        self.sources.append(ConfigSource(
            name="app_base",
            path=str(self.config_dir / "app.yaml"),
            format="yaml",
            priority=1
        ))
        
        # 2. Environment-specific configuration
        env_config = self.config_dir / "environments" / f"{self.environment}.yaml"
        if env_config.exists():
            self.sources.append(ConfigSource(
                name=f"env_{self.environment}",
                path=str(env_config),
                format="yaml",
                priority=2,
                environment=self.environment
            ))
        
        # 3. Local overrides (if exists)
        local_config = self.config_dir / "local.yaml"
        if local_config.exists():
            self.sources.append(ConfigSource(
                name="local_overrides",
                path=str(local_config),
                format="yaml",
                priority=3
            ))
        
        # 4. Environment variables (highest priority)
        self.sources.append(ConfigSource(
            name="environment_variables",
            path="",
            format="env",
            priority=4
        ))
    
    def load_all(self) -> None:
        """Load all configuration sources."""
        try:
            for source in self.sources:
                self._load_source(source)
            
            # Merge all configurations
            self._merge_configurations()
            self._loaded = True
            
            logger.info(f"Successfully loaded configuration for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_source(self, source: ConfigSource) -> None:
        """Load a single configuration source."""
        try:
            if source.format == "yaml":
                source.data = self._load_yaml(source.path)
            elif source.format == "json":
                source.data = self._load_json(source.path)
            elif source.format == "env":
                source.data = self._load_environment_variables()
            else:
                raise ValueError(f"Unsupported configuration format: {source.format}")
            
            source.loaded = True
            logger.debug(f"Loaded configuration source: {source.name}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration source {source.name}: {e}")
            source.data = {}
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Replace environment variables
            content = self._substitute_environment_variables(content)
            return yaml.safe_load(content) or {}
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Replace environment variables
            content = self._substitute_environment_variables(content)
            return json.loads(content)
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to configuration keys
        env_mappings = {
            'APP_HOST': 'app.server.host',
            'APP_PORT': 'app.server.port',
            'DB_HOST': 'app.database.primary.host',
            'DB_PORT': 'app.database.primary.port',
            'DB_NAME': 'app.database.primary.name',
            'DB_USER': 'app.database.primary.user',
            'DB_PASSWORD': 'app.database.primary.password',
            'REDIS_HOST': 'app.database.cache.host',
            'REDIS_PORT': 'app.database.cache.port',
            'REDIS_PASSWORD': 'app.database.cache.password',
            'LOG_LEVEL': 'app.logging.level',
            'SECRET_KEY': 'app.security.secret_key',
            'BREEZE_API_KEY': 'credentials.breeze.api_key',
            'BREEZE_API_SECRET': 'credentials.breeze.api_secret',
            'BREEZE_SESSION_TOKEN': 'credentials.breeze.session_token',
            'ALPHA_VANTAGE_API_KEY': 'credentials.alpha_vantage.api_key',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, config_path, value)
        
        return env_config
    
    def _substitute_environment_variables(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                return os.getenv(var_expr.strip(), '')
        
        # Replace ${VAR:default} and ${VAR} patterns
        return re.sub(r'\\$\\{([^}]+)\\}', replace_env_var, content)
    
    def _merge_configurations(self) -> None:
        """Merge all configuration sources by priority."""
        merged = {}
        
        # Sort sources by priority
        sorted_sources = sorted(self.sources, key=lambda x: x.priority)
        
        for source in sorted_sources:
            if source.loaded and source.data:
                self._deep_merge(merged, source.data)
        
        self._config_cache = merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        if isinstance(value, str):
            # Try to convert to bool
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            # Try to convert to int
            elif value.isdigit():
                value = int(value)
            # Try to convert to float
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        if not self._loaded:
            self.load_all()
        
        keys = key.split('.')
        current = self._config_cache
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value at runtime."""
        if not self._loaded:
            self.load_all()
        
        self._set_nested_value(self._config_cache, key, value)
    
    def reload(self) -> None:
        """Reload all configuration sources."""
        self._config_cache.clear()
        self._loaded = False
        for source in self.sources:
            source.loaded = False
            source.data.clear()
        self.load_all()
    
    def export_config(self, format: str = "yaml") -> str:
        """Export current configuration."""
        if not self._loaded:
            self.load_all()
        
        if format.lower() == "yaml":
            return yaml.dump(self._config_cache, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(self._config_cache, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        if not self._loaded:
            self.load_all()
        
        issues = []
        
        # Required configurations
        required_keys = [
            'app.name',
            'app.version',
            'app.server.host',
            'app.server.port',
            'app.logging.level'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                issues.append(f"Missing required configuration: {key}")
        
        # Validate data types
        type_validations = {
            'app.server.port': int,
            'app.debug': bool,
            'app.database.primary.pool_size': int,
        }
        
        for key, expected_type in type_validations.items():
            value = self.get(key)
            if value is not None and not isinstance(value, expected_type):
                issues.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        return issues


# Global configuration instance
config_manager = ConfigurationManager()


def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    return config_manager


def get_setting(key: str, default: Any = None) -> Any:
    """Convenience function to get configuration setting."""
    return config_manager.get(key, default)


def get_database_url() -> str:
    """Get database URL from configuration."""
    db_config = config_manager.get_section('app.database.primary')
    return (f"{db_config.get('type', 'postgresql')}://"
            f"{db_config.get('user', 'postgres')}:"
            f"{db_config.get('password', '')}@"
            f"{db_config.get('host', 'localhost')}:"
            f"{db_config.get('port', 5432)}/"
            f"{db_config.get('name', 'price_predictor')}")


def get_redis_url() -> str:
    """Get Redis URL from configuration."""
    redis_config = config_manager.get_section('app.database.cache')
    password_part = f":{redis_config.get('password')}@" if redis_config.get('password') else ""
    return (f"redis://{password_part}"
            f"{redis_config.get('host', 'localhost')}:"
            f"{redis_config.get('port', 6379)}/"
            f"{redis_config.get('db', 0)}")


if __name__ == "__main__":
    # Example usage and testing
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Environment: {config.environment}")
    print(f"App name: {config.get('app.name')}")
    print(f"Server port: {config.get('app.server.port')}")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validation passed!")
