"""
Test Base Cost Calculator
========================

Unit tests for the abstract base cost calculator class and its
core functionality including validation, caching, and error handling.
"""

import unittest
from unittest.mock import Mock, patch
import asyncio
from datetime import datetime
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.transaction_costs.base_cost_calculator import CostCalculatorBase, CalculationMode
from trading.transaction_costs.models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    TransactionType,
    InstrumentType
)
from trading.transaction_costs.exceptions import (
    InvalidTransactionError,
    BrokerConfigurationError,
    CalculationError,
    UnsupportedInstrumentError
)


class MockCostCalculator(CostCalculatorBase):
    """Mock implementation of CostCalculatorBase for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            calculator_name="MockCalculator",
            version="1.0.0",
            supported_instruments=[InstrumentType.EQUITY, InstrumentType.OPTION],
            **kwargs
        )
    
    def _calculate_commission(self, request, broker_config):
        """Mock commission calculation."""
        if request.instrument_type == InstrumentType.EQUITY:
            return broker_config.equity_commission * Decimal(str(request.quantity))
        elif request.instrument_type == InstrumentType.OPTION:
            return broker_config.options_per_contract * Decimal(str(request.quantity))
        return Decimal('0.00')
    
    def _calculate_regulatory_fees(self, request, broker_config):
        """Mock regulatory fees calculation."""
        notional = request.notional_value
        return notional * Decimal('0.0001')  # 1 basis point
    
    def _calculate_market_impact(self, request, market_conditions):
        """Mock market impact calculation."""
        if market_conditions and market_conditions.volume:
            volume_ratio = request.quantity / market_conditions.volume
            return request.notional_value * Decimal(str(volume_ratio)) * Decimal('0.001')
        return Decimal('0.00')
    
    def _get_supported_instruments(self):
        """Return supported instruments."""
        return [InstrumentType.EQUITY, InstrumentType.OPTION]


class TestBaseCostCalculator(unittest.TestCase):
    """Test suite for the base cost calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MockCostCalculator()
        
        self.valid_request = TransactionRequest(
            symbol='AAPL',
            quantity=100,
            price=Decimal('150.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        self.valid_broker_config = BrokerConfiguration(
            broker_name='Test Broker',
            equity_commission=Decimal('0.005'),  # $0.005 per share
            options_per_contract=Decimal('0.65'),
            min_commission=Decimal('1.00')
        )
        
        self.valid_market_conditions = MarketConditions(
            bid_price=Decimal('149.50'),
            ask_price=Decimal('150.50'),
            volume=1000000,
            timestamp=datetime.now()
        )
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        self.assertEqual(self.calculator.calculator_name, "MockCalculator")
        self.assertEqual(self.calculator.version, "1.0.0")
        self.assertIn(InstrumentType.EQUITY, self.calculator.supported_instruments)
        self.assertIn(InstrumentType.OPTION, self.calculator.supported_instruments)
    
    def test_basic_cost_calculation(self):
        """Test basic cost calculation functionality."""
        result = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        self.assertIsInstance(result, TransactionCostBreakdown)
        self.assertGreater(result.total_cost, Decimal('0.00'))
        self.assertEqual(result.calculator_version, "MockCalculator v1.0.0")
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        result = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        # For equity with 100 shares at $0.005 per share
        expected_commission = Decimal('0.005') * Decimal('100')
        self.assertEqual(result.commission, expected_commission)
    
    def test_regulatory_fees_calculation(self):
        """Test regulatory fees calculation."""
        result = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        # Mock calculation: notional * 0.0001
        expected_fees = self.valid_request.notional_value * Decimal('0.0001')
        self.assertEqual(result.regulatory_fees, expected_fees)
    
    def test_request_validation(self):
        """Test transaction request validation."""
        # Test that invalid requests are caught during creation
        with self.assertRaises(ValueError):
            TransactionRequest(
                symbol='AAPL',
                quantity=-100,  # Invalid negative quantity
                price=Decimal('150.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY
            )
        
        with self.assertRaises(ValueError):
            TransactionRequest(
                symbol='AAPL',
                quantity=100,
                price=Decimal('-150.00'),  # Invalid negative price
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY
            )
        
        with self.assertRaises(ValueError):
            TransactionRequest(
                symbol='',  # Invalid empty symbol
                quantity=100,
                price=Decimal('150.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY
            )
    
    def test_broker_config_validation(self):
        """Test broker configuration validation."""
        # Test inactive broker config
        inactive_broker = BrokerConfiguration(
            broker_name='Inactive Broker',
            active=False
        )
        
        with self.assertRaises(BrokerConfigurationError):
            self.calculator.calculate_cost(
                self.valid_request,
                inactive_broker,
                self.valid_market_conditions
            )
    
    def test_unsupported_instrument(self):
        """Test unsupported instrument handling."""
        unsupported_request = TransactionRequest(
            symbol='TEST',
            quantity=100,
            price=Decimal('50.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.BOND  # Not supported by mock calculator
        )
        
        with self.assertRaises(UnsupportedInstrumentError):
            self.calculator.calculate_cost(
                unsupported_request,
                self.valid_broker_config,
                self.valid_market_conditions
            )
    
    def test_calculation_mode_validation(self):
        """Test calculation mode validation."""
        with self.assertRaises(CalculationError):
            self.calculator.calculate_cost(
                self.valid_request,
                self.valid_broker_config,
                self.valid_market_conditions,
                mode="invalid_mode"
            )
    
    def test_confidence_level_determination(self):
        """Test confidence level determination."""
        result = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        # Should have high confidence with fresh market data
        self.assertIsNotNone(result.confidence_level)
        self.assertGreater(result.confidence_level, 0.8)
    
    def test_caching_functionality(self):
        """Test result caching."""
        # Enable caching
        self.calculator.enable_caching = True
        
        # First calculation
        result1 = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        # Second calculation (should be cached)
        result2 = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        # Results should be the same (cached)
        self.assertEqual(result1.total_cost, result2.total_cost)
        
        # Cache should contain the result
        self.assertGreater(len(self.calculator._result_cache), 0)
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Enable caching and calculate
        self.calculator.enable_caching = True
        self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        # Verify cache has content
        self.assertGreater(len(self.calculator._result_cache), 0)
        
        # Clear cache
        self.calculator.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.calculator._result_cache), 0)
    
    def test_batch_calculation(self):
        """Test batch cost calculation."""
        requests = []
        for i in range(5):
            request = TransactionRequest(
                symbol=f'TEST{i}',
                quantity=100 * (i + 1),
                price=Decimal('50.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY
            )
            requests.append(request)
        
        results = self.calculator.calculate_batch_costs(
            requests,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        self.assertEqual(len(results), len(requests))
        for result in results:
            self.assertIsInstance(result, TransactionCostBreakdown)
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        # Perform a calculation
        self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        stats = self.calculator.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['calculator_name'], 'MockCalculator')
        self.assertEqual(stats['total_calculations'], 1)
        self.assertIsNotNone(stats['average_calculation_time'])
        self.assertIsNotNone(stats['last_calculation_time'])
    
    def test_supported_features(self):
        """Test supported features reporting."""
        features = self.calculator.get_supported_features()
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features['calculator_name'], 'MockCalculator')
        self.assertIn('EQUITY', features['supported_instruments'])
        self.assertIn('OPTION', features['supported_instruments'])
        self.assertTrue(features['async_support'])
        self.assertTrue(features['batch_support'])
    
    def test_calculation_details(self):
        """Test calculation details in result."""
        result = self.calculator.calculate_cost(
            self.valid_request,
            self.valid_broker_config,
            self.valid_market_conditions
        )
        
        details = result.cost_details
        self.assertIsInstance(details, dict)
        self.assertEqual(details['calculator'], 'MockCalculator')
        self.assertEqual(details['broker'], 'Test Broker')
        self.assertTrue(details['has_market_data'])


class TestAsyncCalculation(unittest.TestCase):
    """Test asynchronous calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MockCostCalculator()
        
        self.valid_request = TransactionRequest(
            symbol='AAPL',
            quantity=100,
            price=Decimal('150.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        self.valid_broker_config = BrokerConfiguration(
            broker_name='Test Broker',
            equity_commission=Decimal('0.005')
        )
    
    def test_async_calculation(self):
        """Test asynchronous cost calculation."""
        async def run_async_test():
            result = await self.calculator.calculate_cost_async(
                self.valid_request,
                self.valid_broker_config
            )
            self.assertIsInstance(result, TransactionCostBreakdown)
            return result
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_async_test())
            self.assertIsNotNone(result)
        finally:
            loop.close()


if __name__ == '__main__':
    unittest.main()