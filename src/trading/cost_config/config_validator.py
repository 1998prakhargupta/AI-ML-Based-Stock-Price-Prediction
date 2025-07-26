"""
Cost Configuration Validator
============================

Validation utilities for transaction cost configuration.
Provides comprehensive validation for broker configurations,
calculation settings, and integration parameters.
"""

from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import re

from ..transaction_costs.models import BrokerConfiguration, InstrumentType
from ..transaction_costs.constants import (
    SUPPORTED_CURRENCIES,
    SYSTEM_DEFAULTS,
    CONFIDENCE_LEVELS
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


class CostConfigurationValidator:
    """
    Comprehensive validator for transaction cost configurations.
    
    Validates broker configurations, calculation settings, market data
    parameters, and integration settings to ensure they are consistent
    and within acceptable ranges.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
        
        # Validation rules and limits
        self.decimal_precision_limit = 10
        self.max_commission_rate = Decimal('1000.00')  # $1000 max commission
        self.max_fee_rate = Decimal('1.0')  # 100% max fee rate
        self.max_timeout_seconds = 300
        self.max_cache_duration_seconds = 3600
        self.max_workers = 32
        
        logger.info(f"CostConfigurationValidator initialized (strict_mode={strict_mode})")
    
    def validate_full_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete configuration dictionary.
        
        Args:
            config: Complete configuration dictionary
            
        Returns:
            ValidationResult with all validation issues
        """
        result = ValidationResult()
        
        # Validate structure
        structure_result = self._validate_config_structure(config)
        result.merge(structure_result)
        
        # Validate brokers section
        if "brokers" in config:
            brokers_result = self.validate_brokers_configuration(config["brokers"])
            result.merge(brokers_result)
        
        # Validate calculation settings
        if "calculation" in config:
            calc_result = self._validate_calculation_config(config["calculation"])
            result.merge(calc_result)
        
        # Validate market data settings
        if "market_data" in config:
            market_result = self._validate_market_data_config(config["market_data"])
            result.merge(market_result)
        
        # Validate regulatory settings
        if "regulatory" in config:
            reg_result = self._validate_regulatory_config(config["regulatory"])
            result.merge(reg_result)
        
        # Validate performance settings
        if "performance" in config:
            perf_result = self._validate_performance_config(config["performance"])
            result.merge(perf_result)
        
        # Validate error handling settings
        if "error_handling" in config:
            error_result = self._validate_error_handling_config(config["error_handling"])
            result.merge(error_result)
        
        # Validate integration settings
        if "integration" in config:
            int_result = self._validate_integration_config(config["integration"])
            result.merge(int_result)
        
        # Cross-validation checks
        cross_result = self._validate_cross_dependencies(config)
        result.merge(cross_result)
        
        if result.is_valid:
            result.add_info("Configuration validation passed")
        else:
            logger.warning(f"Configuration validation failed with {len(result.errors)} errors")
        
        return result
    
    def validate_broker_configuration(self, broker_config: Union[BrokerConfiguration, Dict]) -> ValidationResult:
        """
        Validate a single broker configuration.
        
        Args:
            broker_config: BrokerConfiguration instance or dictionary
            
        Returns:
            ValidationResult for the broker configuration
        """
        result = ValidationResult()
        
        try:
            # Convert to dictionary if needed
            if isinstance(broker_config, BrokerConfiguration):
                config_dict = broker_config.to_dict()
                broker_name = broker_config.broker_name
            else:
                config_dict = broker_config
                broker_name = config_dict.get('broker_name', 'Unknown')
            
            # Validate required fields
            required_fields = ['broker_name']
            for field in required_fields:
                if field not in config_dict or not config_dict[field]:
                    result.add_error(f"Broker {broker_name}: Missing required field '{field}'")
            
            # Validate broker name
            if 'broker_name' in config_dict:
                name_result = self._validate_broker_name(config_dict['broker_name'])
                result.merge(name_result)
            
            # Validate commission rates
            commission_fields = [
                'equity_commission', 'options_commission', 'futures_commission',
                'min_commission', 'max_commission'
            ]
            for field in commission_fields:
                if field in config_dict:
                    field_result = self._validate_decimal_field(
                        config_dict[field], field, broker_name,
                        min_value=Decimal('0.0'),
                        max_value=self.max_commission_rate
                    )
                    result.merge(field_result)
            
            # Validate per-contract fees
            contract_fields = ['options_per_contract', 'futures_per_contract']
            for field in contract_fields:
                if field in config_dict:
                    field_result = self._validate_decimal_field(
                        config_dict[field], field, broker_name,
                        min_value=Decimal('0.0'),
                        max_value=Decimal('100.0')  # $100 per contract seems reasonable max
                    )
                    result.merge(field_result)
            
            # Validate regulatory fee rates
            reg_fields = ['sec_fee_rate', 'finra_taf_rate']
            for field in reg_fields:
                if field in config_dict:
                    field_result = self._validate_decimal_field(
                        config_dict[field], field, broker_name,
                        min_value=Decimal('0.0'),
                        max_value=self.max_fee_rate
                    )
                    result.merge(field_result)
            
            # Validate platform and data fees
            fee_fields = ['platform_fee', 'data_fee']
            for field in fee_fields:
                if field in config_dict:
                    field_result = self._validate_decimal_field(
                        config_dict[field], field, broker_name,
                        min_value=Decimal('0.0'),
                        max_value=Decimal('1000.0')  # $1000 max for platform fees
                    )
                    result.merge(field_result)
            
            # Validate multipliers
            multiplier_fields = ['pre_market_multiplier', 'after_hours_multiplier']
            for field in multiplier_fields:
                if field in config_dict:
                    field_result = self._validate_decimal_field(
                        config_dict[field], field, broker_name,
                        min_value=Decimal('0.1'),
                        max_value=Decimal('10.0')  # 10x multiplier max
                    )
                    result.merge(field_result)
            
            # Validate currency
            if 'base_currency' in config_dict:
                currency_result = self._validate_currency(config_dict['base_currency'], broker_name)
                result.merge(currency_result)
            
            # Validate account tier
            if 'account_tier' in config_dict:
                tier_result = self._validate_account_tier(config_dict['account_tier'], broker_name)
                result.merge(tier_result)
            
            # Validate commission logic consistency
            logic_result = self._validate_commission_logic(config_dict, broker_name)
            result.merge(logic_result)
            
            # Try to create BrokerConfiguration object to test serialization
            if result.is_valid:
                try:
                    # Convert string values to proper types before creating object
                    test_config = {}
                    for key, value in config_dict.items():
                        if isinstance(value, str) and key.endswith(('_commission', '_fee', '_rate', '_multiplier')):
                            try:
                                test_config[key] = Decimal(value)
                            except:
                                test_config[key] = value
                        elif key == 'volume_discount_tiers' and isinstance(value, dict):
                            test_config[key] = {k: Decimal(str(v)) for k, v in value.items()}
                        else:
                            test_config[key] = value
                    
                    BrokerConfiguration(**test_config)
                    result.add_info(f"Broker {broker_name}: Configuration object creation successful")
                except Exception as e:
                    result.add_error(f"Broker {broker_name}: Cannot create BrokerConfiguration object: {str(e)}")
            
        except Exception as e:
            result.add_error(f"Broker configuration validation failed: {str(e)}")
        
        return result
    
    def validate_brokers_configuration(self, brokers_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate multiple broker configurations.
        
        Args:
            brokers_config: Dictionary of broker configurations
            
        Returns:
            ValidationResult for all broker configurations
        """
        result = ValidationResult()
        
        if not isinstance(brokers_config, dict):
            result.add_error("Brokers configuration must be a dictionary")
            return result
        
        if not brokers_config:
            result.add_warning("No broker configurations found")
            return result
        
        broker_names = set()
        broker_ids = set()
        
        for broker_name, broker_config in brokers_config.items():
            # Validate individual broker
            broker_result = self.validate_broker_configuration(broker_config)
            result.merge(broker_result)
            
            # Check for duplicate names
            config_broker_name = broker_config.get('broker_name', broker_name)
            if config_broker_name in broker_names:
                result.add_error(f"Duplicate broker name: {config_broker_name}")
            broker_names.add(config_broker_name)
            
            # Check for duplicate IDs
            broker_id = broker_config.get('broker_id')
            if broker_id:
                if broker_id in broker_ids:
                    result.add_error(f"Duplicate broker ID: {broker_id}")
                broker_ids.add(broker_id)
        
        result.add_info(f"Validated {len(brokers_config)} broker configurations")
        return result
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate basic configuration structure."""
        result = ValidationResult()
        
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return result
        
        # Check for required top-level sections
        required_sections = ['version']
        for section in required_sections:
            if section not in config:
                result.add_error(f"Missing required configuration section: {section}")
        
        # Check for recommended sections
        recommended_sections = ['brokers', 'calculation', 'market_data']
        for section in recommended_sections:
            if section not in config:
                result.add_warning(f"Recommended configuration section missing: {section}")
        
        # Validate version format
        if 'version' in config:
            version_result = self._validate_version(config['version'])
            result.merge(version_result)
        
        return result
    
    def _validate_calculation_config(self, calc_config: Dict[str, Any]) -> ValidationResult:
        """Validate calculation configuration."""
        result = ValidationResult()
        
        # Validate default mode
        if 'default_mode' in calc_config:
            valid_modes = ['real_time', 'batch', 'historical', 'simulation']
            if calc_config['default_mode'] not in valid_modes:
                result.add_error(f"Invalid default_mode: {calc_config['default_mode']}. Must be one of {valid_modes}")
        
        # Validate precision
        if 'precision_decimal_places' in calc_config:
            precision = calc_config['precision_decimal_places']
            if not isinstance(precision, int) or precision < 0 or precision > self.decimal_precision_limit:
                result.add_error(f"precision_decimal_places must be integer between 0 and {self.decimal_precision_limit}")
        
        # Validate confidence threshold
        if 'confidence_threshold' in calc_config:
            threshold = calc_config['confidence_threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
                result.add_error("confidence_threshold must be a number between 0.0 and 1.0")
        
        # Validate cache duration
        if 'cache_duration_seconds' in calc_config:
            duration = calc_config['cache_duration_seconds']
            if not isinstance(duration, int) or duration < 0 or duration > self.max_cache_duration_seconds:
                result.add_error(f"cache_duration_seconds must be integer between 0 and {self.max_cache_duration_seconds}")
        
        # Validate max workers
        if 'max_workers' in calc_config:
            workers = calc_config['max_workers']
            if not isinstance(workers, int) or workers < 1 or workers > self.max_workers:
                result.add_error(f"max_workers must be integer between 1 and {self.max_workers}")
        
        # Validate boolean settings
        bool_settings = ['enable_caching', 'parallel_calculations']
        for setting in bool_settings:
            if setting in calc_config and not isinstance(calc_config[setting], bool):
                result.add_error(f"{setting} must be a boolean value")
        
        return result
    
    def _validate_market_data_config(self, market_config: Dict[str, Any]) -> ValidationResult:
        """Validate market data configuration."""
        result = ValidationResult()
        
        # Validate provider names
        if 'default_provider' in market_config:
            provider = market_config['default_provider']
            if not isinstance(provider, str) or not provider.strip():
                result.add_error("default_provider must be a non-empty string")
        
        if 'fallback_providers' in market_config:
            providers = market_config['fallback_providers']
            if not isinstance(providers, list):
                result.add_error("fallback_providers must be a list")
            elif providers and not all(isinstance(p, str) and p.strip() for p in providers):
                result.add_error("All fallback_providers must be non-empty strings")
        
        # Validate timeout settings
        timeout_settings = ['timeout_seconds', 'cache_duration_seconds', 'staleness_threshold_seconds']
        for setting in timeout_settings:
            if setting in market_config:
                value = market_config[setting]
                if not isinstance(value, int) or value < 0 or value > self.max_timeout_seconds:
                    result.add_error(f"{setting} must be integer between 0 and {self.max_timeout_seconds}")
        
        # Validate retry attempts
        if 'retry_attempts' in market_config:
            attempts = market_config['retry_attempts']
            if not isinstance(attempts, int) or attempts < 0 or attempts > 10:
                result.add_error("retry_attempts must be integer between 0 and 10")
        
        return result
    
    def _validate_regulatory_config(self, reg_config: Dict[str, Any]) -> ValidationResult:
        """Validate regulatory configuration."""
        result = ValidationResult()
        
        # Validate update frequency
        if 'update_frequency_days' in reg_config:
            freq = reg_config['update_frequency_days']
            if not isinstance(freq, int) or freq < 1 or freq > 365:
                result.add_error("update_frequency_days must be integer between 1 and 365")
        
        # Validate auto update setting
        if 'auto_update_enabled' in reg_config and not isinstance(reg_config['auto_update_enabled'], bool):
            result.add_error("auto_update_enabled must be a boolean value")
        
        # Validate fee rates
        fee_rate_fields = ['sec_fee_rate', 'finra_taf_rate']
        for field in fee_rate_fields:
            if field in reg_config:
                field_result = self._validate_decimal_field(
                    reg_config[field], field, "regulatory",
                    min_value=Decimal('0.0'),
                    max_value=self.max_fee_rate
                )
                result.merge(field_result)
        
        return result
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> ValidationResult:
        """Validate performance configuration."""
        result = ValidationResult()
        
        # Validate boolean settings
        bool_settings = ['enable_metrics', 'log_slow_calculations', 'enable_profiling']
        for setting in bool_settings:
            if setting in perf_config and not isinstance(perf_config[setting], bool):
                result.add_error(f"{setting} must be a boolean value")
        
        # Validate threshold
        if 'slow_calculation_threshold_seconds' in perf_config:
            threshold = perf_config['slow_calculation_threshold_seconds']
            if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 300.0:
                result.add_error("slow_calculation_threshold_seconds must be number between 0.0 and 300.0")
        
        return result
    
    def _validate_error_handling_config(self, error_config: Dict[str, Any]) -> ValidationResult:
        """Validate error handling configuration."""
        result = ValidationResult()
        
        # Validate retry settings
        if 'max_retries' in error_config:
            retries = error_config['max_retries']
            if not isinstance(retries, int) or retries < 0 or retries > 10:
                result.add_error("max_retries must be integer between 0 and 10")
        
        if 'retry_delay_seconds' in error_config:
            delay = error_config['retry_delay_seconds']
            if not isinstance(delay, (int, float)) or delay < 0.0 or delay > 60.0:
                result.add_error("retry_delay_seconds must be number between 0.0 and 60.0")
        
        # Validate boolean settings
        bool_settings = ['fail_fast', 'log_all_errors']
        for setting in bool_settings:
            if setting in error_config and not isinstance(error_config[setting], bool):
                result.add_error(f"{setting} must be a boolean value")
        
        return result
    
    def _validate_integration_config(self, int_config: Dict[str, Any]) -> ValidationResult:
        """Validate integration configuration."""
        result = ValidationResult()
        
        # Validate API settings
        if 'enable_external_apis' in int_config and not isinstance(int_config['enable_external_apis'], bool):
            result.add_error("enable_external_apis must be a boolean value")
        
        # Validate rate limits
        if 'api_rate_limits' in int_config:
            rate_limits = int_config['api_rate_limits']
            if isinstance(rate_limits, dict):
                for limit_name, limit_value in rate_limits.items():
                    if not isinstance(limit_value, int) or limit_value < 0:
                        result.add_error(f"api_rate_limits.{limit_name} must be a non-negative integer")
            else:
                result.add_error("api_rate_limits must be a dictionary")
        
        # Validate data validation settings
        if 'data_validation' in int_config:
            validation_config = int_config['data_validation']
            if isinstance(validation_config, dict):
                bool_settings = ['strict_mode', 'allow_estimated_data']
                for setting in bool_settings:
                    if setting in validation_config and not isinstance(validation_config[setting], bool):
                        result.add_error(f"data_validation.{setting} must be a boolean value")
            else:
                result.add_error("data_validation must be a dictionary")
        
        return result
    
    def _validate_cross_dependencies(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate cross-dependencies between configuration sections."""
        result = ValidationResult()
        
        # Check if caching is enabled but cache duration is 0
        calc_config = config.get('calculation', {})
        if (calc_config.get('enable_caching', True) and 
            calc_config.get('cache_duration_seconds', 0) == 0):
            result.add_warning("Caching is enabled but cache duration is 0")
        
        # Check if parallel calculations are enabled but max_workers is 1
        if (calc_config.get('parallel_calculations', True) and 
            calc_config.get('max_workers', 4) == 1):
            result.add_warning("Parallel calculations enabled but max_workers is 1")
        
        # Check if strict data validation is enabled but estimated data is allowed
        int_config = config.get('integration', {})
        validation_config = int_config.get('data_validation', {})
        if (validation_config.get('strict_mode', True) and 
            validation_config.get('allow_estimated_data', True)):
            result.add_warning("Strict validation mode enabled but estimated data is allowed")
        
        return result
    
    # Helper validation methods
    
    def _validate_decimal_field(
        self,
        value: Any,
        field_name: str,
        context: str,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None
    ) -> ValidationResult:
        """Validate a decimal field value."""
        result = ValidationResult()
        
        # Handle None values
        if value is None:
            return result  # None is acceptable for optional fields
        
        try:
            # Convert to Decimal
            if isinstance(value, str):
                decimal_value = Decimal(value)
            elif isinstance(value, (int, float)):
                decimal_value = Decimal(str(value))
            elif isinstance(value, Decimal):
                decimal_value = value
            else:
                result.add_error(f"{context}: {field_name} must be a number, string, or Decimal")
                return result
            
            # Check range
            if min_value is not None and decimal_value < min_value:
                result.add_error(f"{context}: {field_name} must be >= {min_value}, got {decimal_value}")
            
            if max_value is not None and decimal_value > max_value:
                result.add_error(f"{context}: {field_name} must be <= {max_value}, got {decimal_value}")
            
            # Check precision
            if abs(decimal_value.as_tuple().exponent) > self.decimal_precision_limit:
                result.add_warning(f"{context}: {field_name} has very high precision, may cause rounding errors")
            
        except (InvalidOperation, ValueError) as e:
            result.add_error(f"{context}: {field_name} is not a valid decimal number: {str(e)}")
        
        return result
    
    def _validate_broker_name(self, broker_name: str) -> ValidationResult:
        """Validate broker name format."""
        result = ValidationResult()
        
        if not isinstance(broker_name, str):
            result.add_error("Broker name must be a string")
            return result
        
        if not broker_name.strip():
            result.add_error("Broker name cannot be empty")
            return result
        
        if len(broker_name) > 100:
            result.add_error("Broker name cannot exceed 100 characters")
        
        # Check for invalid characters (allow some common trading symbols)
        if re.search(r'[<>:"/\\|?]', broker_name):
            result.add_error("Broker name contains invalid characters")
        
        return result
    
    def _validate_currency(self, currency: str, context: str) -> ValidationResult:
        """Validate currency code."""
        result = ValidationResult()
        
        if not isinstance(currency, str):
            result.add_error(f"{context}: Currency must be a string")
            return result
        
        currency_upper = currency.upper()
        
        if currency_upper not in SUPPORTED_CURRENCIES:
            if self.strict_mode:
                result.add_error(f"{context}: Unsupported currency '{currency}'. Supported: {list(SUPPORTED_CURRENCIES.keys())}")
            else:
                result.add_warning(f"{context}: Currency '{currency}' may not be fully supported")
        
        if len(currency) != 3:
            result.add_error(f"{context}: Currency code must be 3 characters, got '{currency}'")
        
        return result
    
    def _validate_account_tier(self, tier: str, context: str) -> ValidationResult:
        """Validate account tier."""
        result = ValidationResult()
        
        if not isinstance(tier, str):
            result.add_error(f"{context}: Account tier must be a string")
            return result
        
        valid_tiers = ['retail', 'professional', 'institutional']
        if tier.lower() not in valid_tiers:
            if self.strict_mode:
                result.add_error(f"{context}: Invalid account tier '{tier}'. Valid tiers: {valid_tiers}")
            else:
                result.add_warning(f"{context}: Account tier '{tier}' may not be recognized")
        
        return result
    
    def _validate_commission_logic(self, config_dict: Dict, broker_name: str) -> ValidationResult:
        """Validate commission logic consistency."""
        result = ValidationResult()
        
        try:
            min_commission = config_dict.get('min_commission')
            max_commission = config_dict.get('max_commission')
            
            if min_commission is not None and max_commission is not None:
                min_val = Decimal(str(min_commission))
                max_val = Decimal(str(max_commission))
                
                if min_val > max_val:
                    result.add_error(f"{broker_name}: min_commission ({min_val}) cannot be greater than max_commission ({max_val})")
            
            # Check for unusually high commission rates
            commission_fields = ['equity_commission', 'options_commission', 'futures_commission']
            for field in commission_fields:
                if field in config_dict:
                    commission = Decimal(str(config_dict[field]))
                    if commission > Decimal('100.0'):  # $100 commission seems high
                        result.add_warning(f"{broker_name}: {field} of ${commission} seems unusually high")
        
        except Exception as e:
            result.add_error(f"{broker_name}: Error validating commission logic: {str(e)}")
        
        return result
    
    def _validate_version(self, version: str) -> ValidationResult:
        """Validate version format."""
        result = ValidationResult()
        
        if not isinstance(version, str):
            result.add_error("Version must be a string")
            return result
        
        # Basic semantic version pattern
        version_pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$'
        if not re.match(version_pattern, version):
            result.add_warning(f"Version '{version}' does not follow semantic versioning (x.y.z)")
        
        return result


logger.info("Cost configuration validator loaded successfully")