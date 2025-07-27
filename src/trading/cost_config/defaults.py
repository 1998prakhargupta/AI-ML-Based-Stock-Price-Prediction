"""
Default Configuration Values
============================

Default configuration values for transaction cost modeling system.
Provides sensible defaults for all configuration aspects and serves
as the base configuration layer.
"""

from decimal import Decimal
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_default_configuration() -> Dict[str, Any]:
    """
    Get complete default configuration for transaction cost modeling.
    
    Returns:
        Dictionary containing all default configuration sections
    """
    return {
        "version": "1.0.0",
        "description": "Default configuration for transaction cost modeling system",
        
        # Broker configurations (empty by default, populated as needed)
        "brokers": {},
        
        # Market data configuration
        "market_data": get_default_market_data_config(),
        
        # Calculation settings
        "calculation": get_default_calculation_config(),
        
        # Regulatory settings
        "regulatory": get_default_regulatory_config(),
        
        # Performance monitoring
        "performance": get_default_performance_config(),
        
        # Error handling
        "error_handling": get_default_error_handling_config(),
        
        # Integration settings
        "integration": get_default_integration_config(),
        
        # Logging configuration
        "logging": get_default_logging_config(),
        
        # Cache configuration
        "cache": get_default_cache_config(),
        
        # Validation settings
        "validation": get_default_validation_config()
    }


def get_default_market_data_config() -> Dict[str, Any]:
    """Get default market data configuration."""
    return {
        "default_provider": "yahoo",
        "fallback_providers": ["alpha_vantage", "iex"],
        "cache_duration_seconds": 300,
        "staleness_threshold_seconds": 60,
        "retry_attempts": 3,
        "timeout_seconds": 30,
        "rate_limiting": {
            "requests_per_minute": 100,
            "requests_per_hour": 1000,
            "burst_allowance": 10
        },
        "data_quality": {
            "min_data_points": 10,
            "max_staleness_minutes": 30,
            "enable_data_validation": True,
            "outlier_detection_threshold": 3.0
        },
        "supported_data_types": [
            "quotes",
            "trades", 
            "level1",
            "level2",
            "historical",
            "fundamentals"
        ]
    }


def get_default_calculation_config() -> Dict[str, Any]:
    """Get default calculation configuration."""
    return {
        "default_mode": "real_time",
        "precision_decimal_places": 4,
        "confidence_threshold": 0.8,
        "enable_caching": True,
        "cache_duration_seconds": 300,
        "parallel_calculations": True,
        "max_workers": 4,
        "batch_size": 100,
        "optimization": {
            "enable_vectorization": True,
            "use_numpy_acceleration": True,
            "memory_efficient_mode": False,
            "lazy_loading": True
        },
        "rounding": {
            "commission_rounding": "round_half_up",
            "fee_rounding": "round_half_up", 
            "final_cost_rounding": "round_half_up",
            "intermediate_precision": 6
        },
        "supported_calculation_modes": [
            "real_time",
            "batch",
            "historical",
            "simulation",
            "monte_carlo"
        ]
    }


def get_default_regulatory_config() -> Dict[str, Any]:
    """Get default regulatory configuration."""
    return {
        "update_frequency_days": 30,
        "auto_update_enabled": True,
        
        # US regulatory fees (as of 2024)
        "sec_fee_rate": str(Decimal("0.0000051")),  # $5.10 per million
        "finra_taf_sell_rate": str(Decimal("0.000166")),  # Trading Activity Fee
        "finra_taf_max": str(Decimal("8.50")),  # Maximum TAF per trade
        
        # Fee calculation settings
        "apply_sec_fees": True,
        "apply_finra_fees": True,
        "round_regulatory_fees": True,
        "min_regulatory_fee": str(Decimal("0.01")),
        
        # Compliance settings
        "enable_compliance_checks": True,
        "max_position_size_check": True,
        "pattern_day_trader_rules": True,
        "settlement_period_days": 2,
        
        # Reporting requirements
        "enable_regulatory_reporting": True,
        "report_threshold_amount": str(Decimal("10000.0")),
        "large_trader_threshold": str(Decimal("2000000.0"))
    }


def get_default_performance_config() -> Dict[str, Any]:
    """Get default performance monitoring configuration."""
    return {
        "enable_metrics": True,
        "log_slow_calculations": True,
        "slow_calculation_threshold_seconds": 5.0,
        "enable_profiling": False,
        "memory_monitoring": {
            "enable_memory_tracking": True,
            "memory_limit_mb": 512,
            "garbage_collection_threshold": 0.8
        },
        "metrics_collection": {
            "collect_timing_metrics": True,
            "collect_accuracy_metrics": True,
            "collect_cache_metrics": True,
            "metrics_retention_days": 30
        },
        "alerting": {
            "enable_performance_alerts": True,
            "cpu_threshold_percent": 80,
            "memory_threshold_percent": 85,
            "error_rate_threshold_percent": 5
        }
    }


def get_default_error_handling_config() -> Dict[str, Any]:
    """Get default error handling configuration."""
    return {
        "max_retries": 3,
        "retry_delay_seconds": 1,
        "exponential_backoff": True,
        "backoff_multiplier": 2.0,
        "max_retry_delay_seconds": 30,
        "fail_fast": False,
        "log_all_errors": True,
        "error_categories": {
            "data_errors": {
                "retry": True,
                "max_retries": 5,
                "fallback_strategy": "use_cached_data"
            },
            "calculation_errors": {
                "retry": True,
                "max_retries": 3,
                "fallback_strategy": "use_simplified_model"
            },
            "network_errors": {
                "retry": True,
                "max_retries": 5,
                "fallback_strategy": "use_alternative_provider"
            },
            "validation_errors": {
                "retry": False,
                "fallback_strategy": "use_default_values"
            }
        },
        "circuit_breaker": {
            "enable_circuit_breaker": True,
            "failure_threshold": 5,
            "recovery_timeout_seconds": 60,
            "half_open_max_calls": 3
        }
    }


def get_default_integration_config() -> Dict[str, Any]:
    """Get default integration configuration."""
    return {
        "enable_external_apis": True,
        "api_rate_limits": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_allowance": 10
        },
        "data_validation": {
            "strict_mode": True,
            "allow_estimated_data": True,
            "validate_input_data": True,
            "validate_output_data": True
        },
        "external_services": {
            "enable_market_data_apis": True,
            "enable_broker_apis": True,
            "enable_reference_data_apis": True,
            "enable_risk_apis": False
        },
        "fallback_behavior": {
            "use_cached_data_on_failure": True,
            "degrade_gracefully": True,
            "provide_error_details": True
        },
        "security": {
            "encrypt_api_keys": True,
            "validate_ssl_certificates": True,
            "api_key_rotation_days": 90,
            "log_security_events": True
        }
    }


def get_default_logging_config() -> Dict[str, Any]:
    """Get default logging configuration."""
    return {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "enable_file_logging": True,
        "log_file_path": "logs/transaction_costs.log",
        "log_rotation": {
            "max_file_size_mb": 10,
            "backup_count": 5,
            "rotation_time": "midnight"
        },
        "logger_levels": {
            "src.trading.cost_config": "INFO",
            "src.trading.transaction_costs": "INFO",
            "src.models": "WARNING",
            "urllib3": "WARNING",
            "requests": "WARNING"
        },
        "structured_logging": {
            "enable_json_logging": False,
            "include_transaction_id": True,
            "include_timing_info": True,
            "include_user_context": False
        },
        "sensitive_data": {
            "mask_api_keys": True,
            "mask_personal_info": True,
            "sanitize_symbols": False
        }
    }


def get_default_cache_config() -> Dict[str, Any]:
    """Get default cache configuration."""
    return {
        "enable_caching": True,
        "cache_type": "memory",  # memory, redis, file
        "default_ttl_seconds": 300,
        "max_cache_size_mb": 100,
        "cache_policies": {
            "broker_configs": {
                "ttl_seconds": 3600,
                "max_entries": 100
            },
            "market_data": {
                "ttl_seconds": 60,
                "max_entries": 1000
            },
            "calculation_results": {
                "ttl_seconds": 300,
                "max_entries": 500
            },
            "regulatory_fees": {
                "ttl_seconds": 86400,  # 24 hours
                "max_entries": 50
            }
        },
        "cache_warming": {
            "enable_cache_warming": True,
            "warm_on_startup": True,
            "background_refresh": True,
            "refresh_threshold_percent": 80
        },
        "cache_persistence": {
            "enable_persistence": False,
            "persistence_file": "cache/cost_calculation_cache.json",
            "save_interval_seconds": 300
        }
    }


def get_default_validation_config() -> Dict[str, Any]:
    """Get default validation configuration."""
    return {
        "enable_input_validation": True,
        "enable_output_validation": True,
        "strict_validation": True,
        "validation_rules": {
            "trade_amount": {
                "min_value": str(Decimal("0.01")),
                "max_value": str(Decimal("1000000000.0")),  # $1B max
                "required": True
            },
            "share_quantity": {
                "min_value": 1,
                "max_value": 1000000000,  # 1B shares max
                "required": True
            },
            "price_per_share": {
                "min_value": str(Decimal("0.0001")),
                "max_value": str(Decimal("100000.0")),  # $100K max
                "required": True
            },
            "commission_rate": {
                "min_value": str(Decimal("0.0")),
                "max_value": str(Decimal("1000.0")),  # $1000 max commission
                "required": False
            }
        },
        "data_quality_checks": {
            "check_for_outliers": True,
            "outlier_threshold_std_dev": 3.0,
            "check_data_freshness": True,
            "max_staleness_minutes": 60,
            "validate_data_consistency": True
        },
        "error_handling": {
            "fail_on_validation_error": False,
            "log_validation_warnings": True,
            "provide_validation_details": True,
            "sanitize_invalid_data": True
        }
    }


def get_development_overrides() -> Dict[str, Any]:
    """Get configuration overrides for development environment."""
    return {
        "logging": {
            "level": "DEBUG",
            "enable_file_logging": False
        },
        "performance": {
            "enable_profiling": True,
            "slow_calculation_threshold_seconds": 1.0
        },
        "error_handling": {
            "fail_fast": True,
            "log_all_errors": True
        },
        "validation": {
            "strict_validation": False
        },
        "cache": {
            "enable_caching": False  # Disable caching in development
        }
    }


def get_production_overrides() -> Dict[str, Any]:
    """Get configuration overrides for production environment."""
    return {
        "logging": {
            "level": "WARNING",
            "enable_file_logging": True,
            "structured_logging": {
                "enable_json_logging": True
            }
        },
        "performance": {
            "enable_profiling": False,
            "enable_metrics": True
        },
        "error_handling": {
            "fail_fast": False,
            "circuit_breaker": {
                "enable_circuit_breaker": True
            }
        },
        "validation": {
            "strict_validation": True
        },
        "cache": {
            "enable_caching": True,
            "cache_persistence": {
                "enable_persistence": True
            }
        },
        "security": {
            "encrypt_api_keys": True,
            "validate_ssl_certificates": True
        }
    }


def get_testing_overrides() -> Dict[str, Any]:
    """Get configuration overrides for testing environment."""
    return {
        "logging": {
            "level": "WARNING",
            "enable_file_logging": False
        },
        "performance": {
            "enable_metrics": False,
            "enable_profiling": False
        },
        "error_handling": {
            "fail_fast": True,
            "max_retries": 1
        },
        "cache": {
            "enable_caching": False
        },
        "integration": {
            "enable_external_apis": False,
            "data_validation": {
                "strict_mode": False
            }
        },
        "market_data": {
            "timeout_seconds": 5,
            "retry_attempts": 1
        }
    }


def get_minimal_configuration() -> Dict[str, Any]:
    """Get minimal configuration for basic functionality."""
    return {
        "version": "1.0.0",
        "brokers": {},
        "calculation": {
            "default_mode": "real_time",
            "precision_decimal_places": 4,
            "enable_caching": False,
            "parallel_calculations": False,
            "max_workers": 1
        },
        "market_data": {
            "default_provider": "yahoo",
            "timeout_seconds": 30,
            "retry_attempts": 1
        },
        "regulatory": {
            "sec_fee_rate": str(Decimal("0.0000051")),
            "finra_taf_sell_rate": str(Decimal("0.000166")),
            "finra_taf_max": str(Decimal("8.50"))
        },
        "error_handling": {
            "max_retries": 1,
            "fail_fast": True
        },
        "logging": {
            "level": "INFO",
            "enable_file_logging": False
        }
    }


def get_broker_specific_defaults() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for specific brokers."""
    return {
        "interactive_brokers": {
            "broker_name": "Interactive Brokers",
            "broker_type": "traditional",
            "account_tier": "professional",
            "fee_structure": {
                "equity_commission_per_share": str(Decimal("0.005")),
                "min_equity_commission": str(Decimal("1.00")),
                "max_equity_commission": str(Decimal("1.0")),
                "options_commission_per_contract": str(Decimal("0.65")),
                "platform_fee_monthly": str(Decimal("10.00"))
            }
        },
        "charles_schwab": {
            "broker_name": "Charles Schwab",
            "broker_type": "discount",
            "account_tier": "retail",
            "fee_structure": {
                "equity_commission_per_share": str(Decimal("0.0")),
                "options_commission_per_contract": str(Decimal("0.65")),
                "platform_fee_monthly": str(Decimal("0.0"))
            }
        },
        "breeze": {
            "broker_name": "Breeze",
            "broker_type": "discount",
            "account_tier": "retail",
            "fee_structure": {
                "equity_commission_percentage": str(Decimal("0.0005")),
                "min_equity_commission": str(Decimal("20.0")),
                "options_commission_per_contract": str(Decimal("20.0")),
                "futures_commission_per_contract": str(Decimal("20.0"))
            }
        }
    }


logger.info("Default configuration module loaded successfully")