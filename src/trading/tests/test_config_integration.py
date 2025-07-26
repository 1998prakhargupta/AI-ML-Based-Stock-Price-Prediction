"""
Test Configuration Integration
=============================

Integration tests for the cost configuration system and its
integration with the existing project configuration framework.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from decimal import Decimal

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.cost_config.base_config import CostConfiguration
from trading.cost_config.config_validator import CostConfigurationValidator, ValidationResult
from trading.transaction_costs.models import BrokerConfiguration


class TestConfigIntegration(unittest.TestCase):
    """Test suite for configuration integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test configs
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_cost_config.json')
        
        # Sample broker configuration
        self.sample_broker_config = BrokerConfiguration(
            broker_name='Test Broker',
            equity_commission=Decimal('0.00'),
            options_commission=Decimal('0.65'),
            min_commission=Decimal('0.00'),
            max_commission=Decimal('25.00')
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_cost_configuration_creation(self):
        """Test CostConfiguration creation."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        self.assertIsInstance(config, CostConfiguration)
        self.assertEqual(str(config.config_file), self.config_file)
    
    def test_broker_configuration_management(self):
        """Test broker configuration add/get/remove operations."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Add broker configuration
        config.add_broker_configuration(self.sample_broker_config, save_immediately=False)
        
        # Verify it was added
        broker_names = config.list_broker_configurations()
        self.assertIn('Test Broker', broker_names)
        
        # Retrieve broker configuration
        retrieved_broker = config.get_broker_configuration('Test Broker')
        self.assertIsNotNone(retrieved_broker)
        self.assertEqual(retrieved_broker.broker_name, 'Test Broker')
        self.assertEqual(retrieved_broker.equity_commission, Decimal('0.00'))
        
        # Remove broker configuration
        removed = config.remove_broker_configuration('Test Broker', save_immediately=False)
        self.assertTrue(removed)
        
        # Verify removal
        broker_names_after = config.list_broker_configurations()
        self.assertNotIn('Test Broker', broker_names_after)
    
    def test_configuration_settings_getters(self):
        """Test configuration setting getter methods."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Test default values
        self.assertEqual(config.get_default_calculation_mode(), 'real_time')
        self.assertEqual(config.get_cache_duration(), 300)
        self.assertTrue(config.is_caching_enabled())
        self.assertEqual(config.get_precision_decimal_places(), 4)
        self.assertEqual(config.get_max_workers(), 4)
        
        # Test regulatory rates
        sec_rate = config.get_sec_fee_rate()
        finra_rate = config.get_finra_taf_rate()
        self.assertIsInstance(sec_rate, Decimal)
        self.assertIsInstance(finra_rate, Decimal)
        self.assertGreater(sec_rate, Decimal('0.0'))
        self.assertGreater(finra_rate, Decimal('0.0'))
    
    def test_configuration_updates(self):
        """Test configuration update functionality."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Update a setting
        config.update_setting('calculation.precision_decimal_places', 6, save_immediately=False)
        
        # Verify update
        precision = config.get_precision_decimal_places()
        self.assertEqual(precision, 6)
        
        # Test nested setting access
        retrieved_value = config.get_setting('calculation.precision_decimal_places')
        self.assertEqual(retrieved_value, 6)
        
        # Test default value for non-existent setting
        default_value = config.get_setting('non.existent.setting', 'default')
        self.assertEqual(default_value, 'default')
    
    def test_configuration_persistence(self):
        """Test configuration saving and loading."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Add a broker and update a setting
        config.add_broker_configuration(self.sample_broker_config, save_immediately=False)
        config.update_setting('calculation.max_workers', 8, save_immediately=False)
        
        # Save configuration
        config.save_config()
        
        # Verify file exists
        self.assertTrue(os.path.exists(self.config_file))
        
        # Create new config instance and verify data persisted
        config2 = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Verify broker was persisted
        broker_names = config2.list_broker_configurations()
        self.assertIn('Test Broker', broker_names)
        
        # Verify setting was persisted
        max_workers = config2.get_max_workers()
        self.assertEqual(max_workers, 8)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Add valid broker
        config.add_broker_configuration(self.sample_broker_config, save_immediately=False)
        
        # Validate configuration
        errors = config.validate_configuration()
        
        # Should have no errors for valid configuration
        self.assertIsInstance(errors, dict)
        
        # If there are broker errors, they should be empty for valid config
        if 'brokers' in errors:
            self.assertEqual(len(errors['brokers']), 0)
    
    def test_default_broker_creation(self):
        """Test creation of default broker configurations."""
        config = CostConfiguration(
            config_file=self.config_file,
            base_path=self.temp_dir
        )
        
        # Create default brokers
        config.create_default_broker_configurations()
        
        # Verify default brokers were created
        broker_names = config.list_broker_configurations()
        expected_brokers = ['Interactive Brokers', 'Charles Schwab', 'TD Ameritrade', 'E*TRADE']
        
        for expected_broker in expected_brokers:
            self.assertIn(expected_broker, broker_names)
        
        # Verify a specific broker configuration
        schwab_config = config.get_broker_configuration('Charles Schwab')
        self.assertIsNotNone(schwab_config)
        self.assertEqual(schwab_config.equity_commission, Decimal('0.00'))


class TestConfigValidator(unittest.TestCase):
    """Test suite for configuration validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = CostConfigurationValidator(strict_mode=True)
        
        self.valid_broker_config = {
            'broker_name': 'Test Broker',
            'equity_commission': '0.00',
            'options_commission': '0.65',
            'min_commission': '0.00',
            'max_commission': '25.00',
            'base_currency': 'USD'
        }
        
        self.valid_full_config = {
            'version': '1.0.0',
            'brokers': {
                'test_broker': self.valid_broker_config
            },
            'calculation': {
                'default_mode': 'real_time',
                'precision_decimal_places': 4,
                'enable_caching': True,
                'cache_duration_seconds': 300,
                'max_workers': 4
            },
            'market_data': {
                'default_provider': 'yahoo',
                'timeout_seconds': 30,
                'retry_attempts': 3
            }
        }
    
    def test_valid_broker_configuration(self):
        """Test validation of valid broker configuration."""
        result = self.validator.validate_broker_configuration(self.valid_broker_config)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_invalid_broker_configuration(self):
        """Test validation of invalid broker configuration."""
        invalid_config = self.valid_broker_config.copy()
        invalid_config['broker_name'] = ''  # Invalid empty name
        invalid_config['equity_commission'] = '-1.00'  # Invalid negative commission
        
        result = self.validator.validate_broker_configuration(invalid_config)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_commission_logic_validation(self):
        """Test commission logic validation."""
        invalid_config = self.valid_broker_config.copy()
        invalid_config['min_commission'] = '50.00'
        invalid_config['max_commission'] = '25.00'  # Max < Min
        
        result = self.validator.validate_broker_configuration(invalid_config)
        
        self.assertFalse(result.is_valid)
        # Should have error about min > max
        error_messages = ' '.join(result.errors)
        self.assertIn('min_commission', error_messages)
        self.assertIn('max_commission', error_messages)
    
    def test_currency_validation(self):
        """Test currency validation."""
        # Test valid currency
        valid_config = self.valid_broker_config.copy()
        valid_config['base_currency'] = 'USD'
        
        result = self.validator.validate_broker_configuration(valid_config)
        self.assertTrue(result.is_valid)
        
        # Test invalid currency in strict mode
        invalid_config = self.valid_broker_config.copy()
        invalid_config['base_currency'] = 'XYZ'  # Invalid currency
        
        result = self.validator.validate_broker_configuration(invalid_config)
        self.assertFalse(result.is_valid)
    
    def test_full_configuration_validation(self):
        """Test validation of complete configuration."""
        result = self.validator.validate_full_configuration(self.valid_full_config)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_configuration_structure_validation(self):
        """Test configuration structure validation."""
        # Test missing version
        invalid_config = self.valid_full_config.copy()
        del invalid_config['version']
        
        result = self.validator.validate_full_configuration(invalid_config)
        
        self.assertFalse(result.is_valid)
        error_messages = ' '.join(result.errors)
        self.assertIn('version', error_messages)
    
    def test_calculation_config_validation(self):
        """Test calculation configuration validation."""
        # Test invalid calculation config
        invalid_config = self.valid_full_config.copy()
        invalid_config['calculation']['default_mode'] = 'invalid_mode'
        invalid_config['calculation']['precision_decimal_places'] = -1
        invalid_config['calculation']['max_workers'] = 0
        
        result = self.validator.validate_full_configuration(invalid_config)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validation_result_operations(self):
        """Test ValidationResult operations."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")
        
        result2 = ValidationResult()
        result2.add_error("Error 2")
        result2.add_info("Info 1")
        
        # Test merge
        result1.merge(result2)
        
        self.assertEqual(len(result1.errors), 2)
        self.assertEqual(len(result1.warnings), 1)
        self.assertEqual(len(result1.info), 1)
        self.assertFalse(result1.is_valid)
        
        # Test to_dict
        result_dict = result1.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['error_count'], 2)
        self.assertEqual(result_dict['warning_count'], 1)
        self.assertFalse(result_dict['is_valid'])


class TestConfigurationIntegration(unittest.TestCase):
    """Test integration between configuration and validation."""
    
    def test_configuration_with_validation(self):
        """Test configuration creation with validation."""
        temp_dir = tempfile.mkdtemp()
        config_file = os.path.join(temp_dir, 'test_config.json')
        
        try:
            # Create configuration
            config = CostConfiguration(
                config_file=config_file,
                base_path=temp_dir
            )
            
            # Add some broker configurations
            config.create_default_broker_configurations()
            
            # Validate the configuration
            validator = CostConfigurationValidator()
            all_settings = config.get_all_settings()
            validation_result = validator.validate_full_configuration(all_settings)
            
            # Should be valid
            self.assertTrue(validation_result.is_valid)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            # Clean up on error
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e


if __name__ == '__main__':
    unittest.main()