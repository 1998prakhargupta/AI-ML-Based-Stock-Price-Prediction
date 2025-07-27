"""
Configuration Validators
========================

Extended validation utilities for transaction cost configuration.
Provides validation functions for specific configuration sections and
integration with the main configuration system.

This module extends the existing config_validator.py with additional
validators for environment-specific configurations and runtime validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from decimal import Decimal

from .config_validator import CostConfigurationValidator, ValidationResult
from .defaults import get_default_configuration

logger = logging.getLogger(__name__)


class EnvironmentConfigurationValidator:
    """
    Validator for environment-specific configurations.
    
    Validates that environment configurations are properly structured
    and contain valid overrides for the base configuration.
    """
    
    def __init__(self, base_validator: Optional[CostConfigurationValidator] = None):
        """
        Initialize environment configuration validator.
        
        Args:
            base_validator: Base configuration validator to use
        """
        self.base_validator = base_validator or CostConfigurationValidator(strict_mode=True)
        self.supported_environments = ['development', 'staging', 'production', 'testing', 'local']
    
    def validate_environment_config(
        self,
        env_config: Dict[str, Any],
        environment: str,
        base_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate an environment-specific configuration.
        
        Args:
            env_config: Environment configuration to validate
            environment: Environment name
            base_config: Base configuration to validate against
            
        Returns:
            ValidationResult with validation issues
        """
        result = ValidationResult()
        
        # Validate environment name
        if environment not in self.supported_environments:
            result.add_warning(f"Environment '{environment}' is not in supported list: {self.supported_environments}")
        
        # Validate structure
        if not isinstance(env_config, dict):
            result.add_error("Environment configuration must be a dictionary")
            return result
        
        # Check for required metadata
        if 'environment' in env_config and env_config['environment'] != environment:
            result.add_warning(f"Environment mismatch: config says '{env_config['environment']}', expected '{environment}'")
        
        # Validate against base configuration if provided
        if base_config:
            merged_config = self._merge_with_base(base_config, env_config)
            base_result = self.base_validator.validate_full_configuration(merged_config)
            result.merge(base_result)
        
        # Environment-specific validations
        env_result = self._validate_environment_specific_settings(env_config, environment)
        result.merge(env_result)
        
        return result
    
    def _merge_with_base(self, base_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment config with base config."""
        merged = base_config.copy()
        self._deep_merge(merged, env_config)
        return merged
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_environment_specific_settings(
        self,
        env_config: Dict[str, Any],
        environment: str
    ) -> ValidationResult:
        """Validate environment-specific settings."""
        result = ValidationResult()
        
        # Development environment validations
        if environment == 'development':
            result.merge(self._validate_development_config(env_config))
        
        # Production environment validations
        elif environment == 'production':
            result.merge(self._validate_production_config(env_config))
        
        # Staging environment validations
        elif environment == 'staging':
            result.merge(self._validate_staging_config(env_config))
        
        return result
    
    def _validate_development_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate development-specific configuration."""
        result = ValidationResult()
        
        # Development should have debug logging
        logging_config = config.get('logging', {})
        if logging_config.get('level') not in ['DEBUG', 'INFO']:
            result.add_warning("Development environment should use DEBUG or INFO logging level")
        
        # Development should disable caching for testing
        cache_config = config.get('cache', {})
        if cache_config.get('enable_caching', True):
            result.add_info("Development environment has caching enabled (consider disabling for testing)")
        
        # Development should fail fast
        error_config = config.get('error_handling', {})
        if not error_config.get('fail_fast', False):
            result.add_info("Development environment should consider enabling fail_fast for quicker debugging")
        
        return result
    
    def _validate_production_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate production-specific configuration."""
        result = ValidationResult()
        
        # Production should have appropriate logging
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        if log_level == 'DEBUG':
            result.add_warning("Production environment should not use DEBUG logging level")
        
        if not logging_config.get('enable_file_logging', True):
            result.add_error("Production environment must enable file logging")
        
        # Production should have performance monitoring
        perf_config = config.get('performance', {})
        if not perf_config.get('enable_metrics', True):
            result.add_warning("Production environment should enable performance metrics")
        
        # Production should have circuit breaker
        error_config = config.get('error_handling', {})
        circuit_breaker = error_config.get('circuit_breaker', {})
        if not circuit_breaker.get('enable_circuit_breaker', True):
            result.add_warning("Production environment should enable circuit breaker")
        
        # Production should validate SSL certificates
        integration_config = config.get('integration', {})
        security_config = integration_config.get('security', {})
        if not security_config.get('validate_ssl_certificates', True):
            result.add_error("Production environment must validate SSL certificates")
        
        return result
    
    def _validate_staging_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate staging-specific configuration."""
        result = ValidationResult()
        
        # Staging should be similar to production but with some relaxed settings
        logging_config = config.get('logging', {})
        if not logging_config.get('enable_file_logging', True):
            result.add_warning("Staging environment should enable file logging")
        
        # Staging should have metrics enabled
        perf_config = config.get('performance', {})
        if not perf_config.get('enable_metrics', True):
            result.add_info("Staging environment should enable performance metrics for testing")
        
        return result


class RuntimeConfigurationValidator:
    """
    Validator for runtime configuration changes.
    
    Validates configuration changes made at runtime to ensure they
    don't break the system or introduce invalid states.
    """
    
    def __init__(self, base_validator: Optional[CostConfigurationValidator] = None):
        """
        Initialize runtime configuration validator.
        
        Args:
            base_validator: Base configuration validator to use
        """
        self.base_validator = base_validator or CostConfigurationValidator(strict_mode=False)
        
        # Define which settings can be changed at runtime
        self.runtime_changeable_settings = {
            'calculation.cache_duration_seconds',
            'calculation.max_workers',
            'calculation.enable_caching',
            'performance.enable_metrics',
            'performance.slow_calculation_threshold_seconds',
            'error_handling.max_retries',
            'error_handling.retry_delay_seconds',
            'market_data.timeout_seconds',
            'market_data.retry_attempts',
            'logging.level'
        }
        
        # Define settings that require restart
        self.restart_required_settings = {
            'calculation.parallel_calculations',
            'integration.enable_external_apis',
            'cache.cache_type',
            'logging.enable_file_logging'
        }
    
    def validate_runtime_change(
        self,
        setting_path: str,
        new_value: Any,
        current_config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate a runtime configuration change.
        
        Args:
            setting_path: Dot-separated path to the setting
            new_value: New value for the setting
            current_config: Current complete configuration
            
        Returns:
            ValidationResult with validation issues
        """
        result = ValidationResult()
        
        # Check if setting can be changed at runtime
        if setting_path not in self.runtime_changeable_settings:
            if setting_path in self.restart_required_settings:
                result.add_warning(f"Setting '{setting_path}' requires system restart to take effect")
            else:
                result.add_error(f"Setting '{setting_path}' cannot be changed at runtime")
        
        # Validate the new value
        value_result = self._validate_setting_value(setting_path, new_value)
        result.merge(value_result)
        
        # Create test configuration with the new value
        test_config = current_config.copy()
        self._set_nested_value(test_config, setting_path, new_value)
        
        # Validate the updated configuration
        config_result = self.base_validator.validate_full_configuration(test_config)
        if not config_result.is_valid:
            result.add_error(f"Configuration would be invalid after changing '{setting_path}': {config_result.errors}")
        
        return result
    
    def _validate_setting_value(self, setting_path: str, value: Any) -> ValidationResult:
        """Validate a specific setting value."""
        result = ValidationResult()
        
        # Type and range validations based on setting path
        if 'timeout_seconds' in setting_path:
            if not isinstance(value, (int, float)) or value <= 0 or value > 300:
                result.add_error(f"Timeout must be a positive number <= 300 seconds, got {value}")
        
        elif 'max_retries' in setting_path:
            if not isinstance(value, int) or value < 0 or value > 10:
                result.add_error(f"Max retries must be an integer between 0 and 10, got {value}")
        
        elif 'max_workers' in setting_path:
            if not isinstance(value, int) or value < 1 or value > 32:
                result.add_error(f"Max workers must be an integer between 1 and 32, got {value}")
        
        elif 'enable_' in setting_path:
            if not isinstance(value, bool):
                result.add_error(f"Enable settings must be boolean, got {type(value).__name__}")
        
        elif 'level' in setting_path and 'log' in setting_path:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if value not in valid_levels:
                result.add_error(f"Log level must be one of {valid_levels}, got {value}")
        
        return result
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set a nested value in configuration dictionary."""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class ConfigurationFileValidator:
    """
    Validator for configuration files on disk.
    
    Validates configuration files for syntax, structure, and content.
    """
    
    def __init__(self):
        """Initialize configuration file validator."""
        self.base_validator = CostConfigurationValidator(strict_mode=True)
        self.env_validator = EnvironmentConfigurationValidator(self.base_validator)
    
    def validate_configuration_file(self, file_path: str) -> ValidationResult:
        """
        Validate a configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            ValidationResult with validation issues
        """
        result = ValidationResult()
        
        # Check if file exists
        if not Path(file_path).exists():
            result.add_error(f"Configuration file not found: {file_path}")
            return result
        
        # Try to parse JSON
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in configuration file: {e}")
            return result
        except Exception as e:
            result.add_error(f"Error reading configuration file: {e}")
            return result
        
        # Determine file type
        file_name = Path(file_path).name
        
        if 'template' in file_name:
            # Validate broker template
            template_result = self._validate_broker_template(config)
            result.merge(template_result)
        
        elif any(env in file_name for env in ['development', 'staging', 'production', 'local']):
            # Validate environment configuration
            environment = next(env for env in ['development', 'staging', 'production', 'local'] if env in file_name)
            env_result = self.env_validator.validate_environment_config(config, environment)
            result.merge(env_result)
        
        else:
            # Validate as main configuration
            config_result = self.base_validator.validate_full_configuration(config)
            result.merge(config_result)
        
        return result
    
    def _validate_broker_template(self, template_config: Dict[str, Any]) -> ValidationResult:
        """Validate a broker template configuration."""
        result = ValidationResult()
        
        # Check required template fields
        required_fields = ['template_name', 'template_version', 'description']
        for field in required_fields:
            if field not in template_config:
                result.add_error(f"Missing required template field: {field}")
        
        # Validate fee structure if present
        if 'fee_structure' in template_config:
            # Convert string decimals for validation
            fee_structure = template_config['fee_structure']
            for key, value in fee_structure.items():
                if isinstance(value, str) and key.endswith(('_commission', '_fee', '_rate', '_multiplier')):
                    try:
                        Decimal(value)
                    except:
                        result.add_error(f"Invalid decimal value in fee_structure.{key}: {value}")
        
        return result
    
    def validate_configuration_directory(self, directory_path: str) -> Dict[str, ValidationResult]:
        """
        Validate all configuration files in a directory.
        
        Args:
            directory_path: Path to the configuration directory
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        directory = Path(directory_path)
        
        if not directory.exists():
            results[directory_path] = ValidationResult()
            results[directory_path].add_error(f"Configuration directory not found: {directory_path}")
            return results
        
        # Find all JSON configuration files
        config_files = list(directory.glob("*.json"))
        config_files.extend(directory.glob("**/*.json"))
        
        for config_file in config_files:
            try:
                result = self.validate_configuration_file(str(config_file))
                results[str(config_file)] = result
            except Exception as e:
                error_result = ValidationResult()
                error_result.add_error(f"Exception during validation: {e}")
                results[str(config_file)] = error_result
        
        return results


def validate_configuration_setup(config_base_path: str = "configs/cost_config") -> ValidationResult:
    """
    Validate the complete configuration setup.
    
    Args:
        config_base_path: Base path for configuration files
        
    Returns:
        ValidationResult with overall validation status
    """
    result = ValidationResult()
    validator = ConfigurationFileValidator()
    
    # Validate configuration directory structure
    base_path = Path(config_base_path)
    
    if not base_path.exists():
        result.add_error(f"Configuration base directory does not exist: {config_base_path}")
        return result
    
    # Check for required files
    required_files = [
        'default_cost_config.json',
        'development.json',
        'staging.json',
        'production.json'
    ]
    
    for required_file in required_files:
        file_path = base_path / required_file
        if not file_path.exists():
            result.add_warning(f"Recommended configuration file missing: {required_file}")
    
    # Validate all configuration files
    file_results = validator.validate_configuration_directory(str(base_path))
    
    total_errors = 0
    total_warnings = 0
    
    for file_path, file_result in file_results.items():
        if not file_result.is_valid:
            total_errors += len(file_result.errors)
            result.add_error(f"Errors in {file_path}: {len(file_result.errors)} errors")
        
        if file_result.warnings:
            total_warnings += len(file_result.warnings)
            result.add_warning(f"Warnings in {file_path}: {len(file_result.warnings)} warnings")
    
    # Summary
    if total_errors == 0 and total_warnings == 0:
        result.add_info("Configuration setup validation passed successfully")
    else:
        result.add_info(f"Configuration validation completed with {total_errors} errors and {total_warnings} warnings")
    
    return result


logger.info("Extended configuration validators loaded successfully")