#!/usr/bin/env python3
"""
Configuration Management System Demo
====================================

Demonstrates the comprehensive configuration management system
for transaction cost modeling, including all major features:

1. Hierarchical configuration loading
2. Environment-specific overrides  
3. Runtime configuration updates
4. Configuration validation
5. Broker configuration management
6. Market parameter configuration
7. Configuration change notifications
"""

import os
import sys
import json
from decimal import Decimal
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading.cost_config import *

def demonstrate_configuration_manager():
    """Demonstrate the main configuration manager functionality."""
    print("=" * 60)
    print("CONFIGURATION MANAGER DEMONSTRATION")
    print("=" * 60)
    
    # 1. Basic Configuration Loading
    print("\n1. Basic Configuration Loading")
    print("-" * 30)
    
    config_manager = ConfigurationManager(environment='development')
    print(f"Environment: {config_manager.environment}")
    print(f"Configuration sources: {len(config_manager._sources)}")
    
    config = config_manager.get_configuration()
    print(f"Configuration sections: {list(config.keys())}")
    print(f"Total brokers configured: {len(config.get('brokers', {}))}")
    
    # 2. Environment-specific Configuration
    print("\n2. Environment-specific Configuration")
    print("-" * 35)
    
    environments = ['development', 'staging', 'production']
    for env in environments:
        env_manager = ConfigurationManager(environment=env)
        log_level = env_manager.get_setting('logging.level')
        caching = env_manager.get_setting('calculation.enable_caching')
        workers = env_manager.get_setting('calculation.max_workers')
        print(f"  {env:12}: log={log_level:7}, cache={caching!s:5}, workers={workers}")
    
    # 3. Runtime Configuration Updates
    print("\n3. Runtime Configuration Updates")
    print("-" * 32)
    
    original_timeout = config_manager.get_setting('market_data.timeout_seconds')
    print(f"Original timeout: {original_timeout} seconds")
    
    # Add change listener
    changes_received = []
    def config_change_listener(new_config):
        changes_received.append(len(new_config))
        print(f"  Configuration changed! New timeout: {new_config['market_data']['timeout_seconds']} seconds")
    
    config_manager.add_change_listener(config_change_listener)
    
    # Update configuration
    config_manager.update_setting('market_data.timeout_seconds', 45)
    new_timeout = config_manager.get_setting('market_data.timeout_seconds')
    print(f"Updated timeout: {new_timeout} seconds")
    print(f"Change notifications received: {len(changes_received)}")
    
    # 4. Environment Variable Override
    print("\n4. Environment Variable Override")
    print("-" * 32)
    
    # Simulate environment variable (normally set externally)
    os.environ['COST_CONFIG_MAX_WORKERS'] = '12'
    os.environ['COST_CONFIG_ENABLE_CACHING'] = 'false'
    
    env_config_manager = ConfigurationManager()
    max_workers = env_config_manager.get_setting('calculation.max_workers')
    enable_caching = env_config_manager.get_setting('calculation.enable_caching')
    
    print(f"Max workers from env var: {max_workers}")
    print(f"Enable caching from env var: {enable_caching}")
    
    # Clean up environment variables
    del os.environ['COST_CONFIG_MAX_WORKERS']
    del os.environ['COST_CONFIG_ENABLE_CACHING']
    
    # 5. Configuration Information
    print("\n5. Configuration Information")
    print("-" * 28)
    
    info = config_manager.get_configuration_info()
    print(f"Environment: {info['environment']}")
    print(f"Auto-reload enabled: {info['auto_reload']}")
    print(f"Validation enabled: {info['validation_enabled']}")
    print(f"Configuration sources: {len(info['sources'])}")
    print(f"Configuration sections: {info['config_sections']}")


def demonstrate_broker_configuration():
    """Demonstrate broker configuration management."""
    print("\n\n" + "=" * 60)
    print("BROKER CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    broker_manager = BrokerConfigurationManager()
    
    # 1. List Available Templates
    print("\n1. Available Broker Templates")
    print("-" * 29)
    
    templates = broker_manager.list_available_templates()
    print(f"Available templates: {templates}")
    
    # 2. Create Broker from Template
    print("\n2. Create Broker from Template")
    print("-" * 30)
    
    # Create a broker from Breeze template with custom overrides
    overrides = {
        "additional_settings": {
            "custom_setting": "demo_value",
            "trading_limit": "1000000.0"
        }
    }
    
    broker_config = broker_manager.create_broker_from_template(
        broker_name="Demo Breeze Broker",
        template_name="Breeze",
        overrides=overrides
    )
    
    print(f"Created broker: {broker_config['broker_name']}")
    print(f"Broker type: {broker_config['broker_type']}")
    print(f"Account tier: {broker_config['account_tier']}")
    
    # 3. Custom Fee Structure
    print("\n3. Custom Fee Structure")
    print("-" * 22)
    
    custom_fee_structure = FeeStructure(
        equity_commission_per_share=Decimal('0.01'),
        min_equity_commission=Decimal('5.00'),
        max_equity_commission=Decimal('25.00'),
        options_commission_per_contract=Decimal('1.50'),
        platform_fee_monthly=Decimal('15.00')
    )
    
    api_config = APIConfiguration(
        base_url="https://api.custom-broker.com",
        rate_limit_requests_per_minute=150,
        supports_websocket=True,
        websocket_url="wss://ws.custom-broker.com"
    )
    
    custom_broker_config = broker_manager.create_broker_configuration(
        broker_name="Custom Demo Broker",
        broker_type=BrokerType.TRADITIONAL,
        account_tier=AccountTier.PROFESSIONAL,
        fee_structure=custom_fee_structure,
        api_config=api_config
    )
    
    print(f"Custom broker: {custom_broker_config['broker_name']}")
    print(f"Equity commission: ${custom_fee_structure.equity_commission_per_share}/share")
    print(f"Min commission: ${custom_fee_structure.min_equity_commission}")
    print(f"API rate limit: {api_config.rate_limit_requests_per_minute}/min")
    
    # 4. Broker Fee Comparison
    print("\n4. Broker Fee Comparison")
    print("-" * 24)
    
    # First save the custom broker
    broker_manager.save_broker_configuration("Custom Demo Broker")
    
    comparison = broker_manager.compare_broker_fees(
        broker_names=["Demo Breeze Broker", "Custom Demo Broker"],
        trade_amount=Decimal('10000.0')
    )
    
    print(f"Fee comparison for $10,000 trade:")
    for broker, fees in comparison.items():
        if 'error' not in fees:
            print(f"  {broker}:")
            print(f"    Total cost: ${fees['total_cost']:.2f}")
            print(f"    Cost (bps): {fees['cost_basis_points']:.2f}")
    
    # 5. List Configured Brokers
    print("\n5. Configured Brokers")
    print("-" * 18)
    
    configured_brokers = broker_manager.list_configured_brokers()
    print(f"Configured brokers: {configured_brokers}")


def demonstrate_market_parameters():
    """Demonstrate market parameter configuration."""
    print("\n\n" + "=" * 60)
    print("MARKET PARAMETER CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    market_manager = MarketParameterManager()
    
    # 1. Volatility Configuration
    print("\n1. Volatility Configuration")
    print("-" * 27)
    
    vol_config = market_manager.get_volatility_configuration(InstrumentClass.EQUITY)
    print(f"Short-term window: {vol_config.short_term_window} days")
    print(f"Medium-term window: {vol_config.medium_term_window} days")
    print(f"Long-term window: {vol_config.long_term_window} days")
    print(f"EWMA lambda: {vol_config.ewma_lambda}")
    print(f"Minimum volatility: {vol_config.min_volatility * 100}%")
    
    # 2. Market Sessions
    print("\n2. Market Sessions")
    print("-" * 16)
    
    current_session = market_manager.get_current_market_session()
    print(f"Current market session: {current_session.value}")
    
    for session in MarketSession:
        multiplier = market_manager.get_session_multiplier(session, InstrumentClass.EQUITY)
        print(f"  {session.value:12}: {multiplier}x cost multiplier")
    
    # 3. Liquidity Configuration
    print("\n3. Liquidity Configuration")
    print("-" * 25)
    
    liq_config = market_manager.get_liquidity_configuration(InstrumentClass.EQUITY)
    print(f"High liquidity percentile: {liq_config.high_liquidity_percentile * 100}%")
    print(f"High liquidity discount: {(1 - liq_config.high_liquidity_discount) * 100:.0f}%")
    print(f"Low liquidity premium: {(liq_config.low_liquidity_premium - 1) * 100:.0f}%")
    print(f"Average volume days: {liq_config.average_volume_days}")
    
    # 4. Market Impact Configuration
    print("\n4. Market Impact Configuration")
    print("-" * 30)
    
    impact_config = market_manager.get_market_impact_configuration(InstrumentClass.EQUITY)
    print(f"Primary model: {impact_config.primary_model}")
    print(f"Small trade threshold: {impact_config.small_trade_threshold * 100}% of ADV")
    print(f"Large trade threshold: {impact_config.large_trade_threshold * 100}% of ADV")
    print(f"Market open multiplier: {impact_config.market_open_multiplier}x")
    
    # 5. Symbol-Specific Configuration
    print("\n5. Symbol-Specific Configuration")
    print("-" * 32)
    
    # Create symbol-specific overrides for a high-volatility stock
    symbol_overrides = {
        "volatility": {
            "intraday_scaling_factor": "2.5",
            "min_volatility": "0.15"
        },
        "spread": {
            "min_spread_bps": "2.0",
            "market_open_spread_multiplier": "2.0"
        }
    }
    
    market_manager.create_symbol_specific_configuration(
        InstrumentClass.EQUITY,
        "VOLATILE_STOCK",
        symbol_overrides
    )
    
    # Retrieve the symbol-specific configuration
    symbol_vol_config = market_manager.get_volatility_configuration(
        InstrumentClass.EQUITY, 
        "VOLATILE_STOCK"
    )
    
    print(f"Symbol-specific config for VOLATILE_STOCK:")
    print(f"  Intraday scaling factor: {symbol_vol_config.intraday_scaling_factor}")
    print(f"  Minimum volatility: {symbol_vol_config.min_volatility * 100}%")


def demonstrate_configuration_validation():
    """Demonstrate configuration validation."""
    print("\n\n" + "=" * 60)
    print("CONFIGURATION VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    # 1. Basic Configuration Validation
    print("\n1. Basic Configuration Validation")
    print("-" * 32)
    
    validator = CostConfigurationValidator(strict_mode=True)
    config_manager = ConfigurationManager()
    config = config_manager.get_configuration()
    
    result = validator.validate_full_configuration(config)
    print(f"Configuration valid: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:3]:  # Show first 3 warnings
            print(f"  - {warning}")
    
    # 2. Environment Configuration Validation
    print("\n2. Environment Configuration Validation")
    print("-" * 37)
    
    env_validator = EnvironmentConfigurationValidator()
    
    # Validate production environment
    prod_manager = ConfigurationManager(environment='production')
    prod_config = prod_manager.get_configuration()
    
    env_result = env_validator.validate_environment_config(
        prod_config, 'production', get_default_configuration()
    )
    
    print(f"Production config valid: {env_result.is_valid}")
    print(f"Production warnings: {len(env_result.warnings)}")
    
    # 3. Runtime Configuration Validation
    print("\n3. Runtime Configuration Validation")
    print("-" * 34)
    
    runtime_validator = RuntimeConfigurationValidator()
    
    # Test valid runtime change
    valid_result = runtime_validator.validate_runtime_change(
        'calculation.max_workers', 8, config
    )
    print(f"Runtime change (max_workers=8) valid: {valid_result.is_valid}")
    
    # Test invalid runtime change
    invalid_result = runtime_validator.validate_runtime_change(
        'calculation.max_workers', -1, config
    )
    print(f"Runtime change (max_workers=-1) valid: {invalid_result.is_valid}")
    if invalid_result.errors:
        print(f"  Error: {invalid_result.errors[0]}")
    
    # 4. Configuration Setup Validation
    print("\n4. Configuration Setup Validation")
    print("-" * 33)
    
    setup_result = validate_configuration_setup()
    print(f"Configuration setup valid: {setup_result.is_valid}")
    print(f"Setup validation info:")
    for info in setup_result.info:
        print(f"  - {info}")


def demonstrate_integration_features():
    """Demonstrate integration features with existing system."""
    print("\n\n" + "=" * 60)
    print("INTEGRATION FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # 1. Configuration Export/Import
    print("\n1. Configuration Export/Import")
    print("-" * 30)
    
    config_manager = ConfigurationManager()
    
    # Export configuration
    export_path = '/tmp/demo_config_export.json'
    config_manager.export_configuration(export_path, include_sources=True)
    print(f"Configuration exported to: {export_path}")
    
    # Show export content structure
    with open(export_path, 'r') as f:
        exported = json.load(f)
    
    print(f"Export contains: {list(exported.keys())}")
    print(f"Sources in export: {list(exported['sources'].keys())}")
    
    # 2. Configuration Monitoring
    print("\n2. Configuration Monitoring")
    print("-" * 26)
    
    info = config_manager.get_configuration_info()
    print(f"Configuration monitoring info:")
    print(f"  Environment: {info['environment']}")
    print(f"  Sources: {len(info['sources'])}")
    print(f"  Watched files: {info['watched_files']}")
    print(f"  Change listeners: {info['change_listeners']}")
    print(f"  Broker count: {info['broker_count']}")
    
    # 3. Default Configuration Integration
    print("\n3. Default Configuration Integration")
    print("-" * 37)
    
    defaults = get_default_configuration()
    dev_overrides = get_development_overrides()
    prod_overrides = get_production_overrides()
    
    print(f"Default config sections: {len(defaults)}")
    print(f"Development overrides: {len(dev_overrides)}")
    print(f"Production overrides: {len(prod_overrides)}")
    
    # Show a sample of default broker configurations
    broker_defaults = get_broker_specific_defaults()
    print(f"Default broker templates: {list(broker_defaults.keys())}")
    
    # 4. Environment Detection
    print("\n4. Environment Detection")
    print("-" * 22)
    
    # Show current environment detection
    auto_config_manager = ConfigurationManager()
    print(f"Auto-detected environment: {auto_config_manager.environment}")
    
    # Show what different environment variables would produce
    test_envs = {
        'COST_CONFIG_ENV': 'custom',
        'APP_ENV': 'testing',
        'ENV': 'local'
    }
    
    for env_var, env_value in test_envs.items():
        os.environ[env_var] = env_value
        test_manager = ConfigurationManager()
        print(f"  {env_var}={env_value} -> {test_manager.environment}")
        del os.environ[env_var]


def main():
    """Run the complete configuration system demonstration."""
    print("TRANSACTION COST CONFIGURATION MANAGEMENT SYSTEM")
    print("=" * 54)
    print("Complete demonstration of all configuration features")
    
    try:
        demonstrate_configuration_manager()
        demonstrate_broker_configuration()
        demonstrate_market_parameters()
        demonstrate_configuration_validation()
        demonstrate_integration_features()
        
        print("\n\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nAll configuration management features are working correctly.")
        print("The system provides comprehensive configuration control for")
        print("transaction cost modeling with environment-specific settings,")
        print("runtime updates, validation, and broker management capabilities.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())