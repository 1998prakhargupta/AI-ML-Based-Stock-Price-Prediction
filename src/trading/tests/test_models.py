"""
Test Transaction Cost Data Models
=================================

Unit tests for transaction cost data models including validation,
serialization, and business logic.
"""

import unittest
from datetime import datetime
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.transaction_costs.models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    TransactionType,
    InstrumentType,
    OrderType,
    MarketTiming
)


class TestTransactionModels(unittest.TestCase):
    """Test suite for transaction cost data models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_request_data = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': Decimal('150.00'),
            'transaction_type': TransactionType.BUY,
            'instrument_type': InstrumentType.EQUITY
        }
        
        self.valid_broker_data = {
            'broker_name': 'Test Broker',
            'equity_commission': Decimal('0.00'),
            'options_commission': Decimal('0.65'),
            'min_commission': Decimal('0.00')
        }
    
    def test_transaction_request_creation(self):
        """Test TransactionRequest creation and validation."""
        # Test valid creation
        request = TransactionRequest(**self.valid_request_data)
        self.assertEqual(request.symbol, 'AAPL')
        self.assertEqual(request.quantity, 100)
        self.assertEqual(request.price, Decimal('150.00'))
        self.assertEqual(request.notional_value, Decimal('15000.00'))
    
    def test_transaction_request_validation(self):
        """Test TransactionRequest validation logic."""
        # Test invalid quantity
        with self.assertRaises(ValueError):
            invalid_data = self.valid_request_data.copy()
            invalid_data['quantity'] = -100
            TransactionRequest(**invalid_data)
        
        # Test invalid price
        with self.assertRaises(ValueError):
            invalid_data = self.valid_request_data.copy()
            invalid_data['price'] = Decimal('-10.00')
            TransactionRequest(**invalid_data)
        
        # Test empty symbol
        with self.assertRaises(ValueError):
            invalid_data = self.valid_request_data.copy()
            invalid_data['symbol'] = ''
            TransactionRequest(**invalid_data)
    
    def test_transaction_request_serialization(self):
        """Test TransactionRequest serialization."""
        request = TransactionRequest(**self.valid_request_data)
        data_dict = request.to_dict()
        
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict['symbol'], 'AAPL')
        self.assertEqual(data_dict['quantity'], 100)
        self.assertEqual(data_dict['price'], 150.0)  # Converted to float
        self.assertEqual(data_dict['transaction_type'], 'BUY')
        self.assertEqual(data_dict['instrument_type'], 'EQUITY')
    
    def test_transaction_cost_breakdown_calculations(self):
        """Test TransactionCostBreakdown cost calculations."""
        breakdown = TransactionCostBreakdown()
        breakdown.commission = Decimal('5.00')
        breakdown.regulatory_fees = Decimal('1.00')
        breakdown.exchange_fees = Decimal('0.50')
        breakdown.bid_ask_spread_cost = Decimal('2.00')
        breakdown.market_impact_cost = Decimal('3.00')
        
        # Test total calculations
        self.assertEqual(breakdown.total_explicit_costs, Decimal('6.50'))
        self.assertEqual(breakdown.total_implicit_costs, Decimal('5.00'))
        self.assertEqual(breakdown.total_cost, Decimal('11.50'))
        
        # Test basis points calculation
        notional_value = Decimal('10000.00')
        basis_points = breakdown.cost_as_basis_points(notional_value)
        expected_bp = (Decimal('11.50') / Decimal('10000.00')) * Decimal('10000')
        self.assertEqual(basis_points, expected_bp)
    
    def test_market_conditions_calculations(self):
        """Test MarketConditions calculated properties."""
        conditions = MarketConditions()
        conditions.bid_price = Decimal('99.50')
        conditions.ask_price = Decimal('100.50')
        
        # Test calculated properties
        self.assertEqual(conditions.bid_ask_spread, Decimal('1.00'))
        self.assertEqual(conditions.mid_price, Decimal('100.00'))
    
    def test_broker_configuration_validation(self):
        """Test BrokerConfiguration validation."""
        # Test valid creation
        broker = BrokerConfiguration(**self.valid_broker_data)
        self.assertEqual(broker.broker_name, 'Test Broker')
        
        # Test empty broker name
        with self.assertRaises(ValueError):
            invalid_data = self.valid_broker_data.copy()
            invalid_data['broker_name'] = ''
            BrokerConfiguration(**invalid_data)
        
        # Test negative commission
        with self.assertRaises(ValueError):
            invalid_data = self.valid_broker_data.copy()
            invalid_data['equity_commission'] = Decimal('-1.00')
            BrokerConfiguration(**invalid_data)
    
    def test_broker_configuration_commission_rates(self):
        """Test BrokerConfiguration commission rate methods."""
        broker = BrokerConfiguration(**self.valid_broker_data)
        
        # Test commission rate retrieval
        equity_rate = broker.get_commission_rate(InstrumentType.EQUITY)
        options_rate = broker.get_commission_rate(InstrumentType.OPTION)
        
        self.assertEqual(equity_rate, Decimal('0.00'))
        self.assertEqual(options_rate, Decimal('0.65'))
        
        # Test default rate for unsupported instrument
        default_rate = broker.get_commission_rate(InstrumentType.BOND)
        self.assertEqual(default_rate, Decimal('0.00'))  # Should default to equity rate
    
    def test_broker_configuration_serialization(self):
        """Test BrokerConfiguration serialization."""
        broker = BrokerConfiguration(**self.valid_broker_data)
        data_dict = broker.to_dict()
        
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict['broker_name'], 'Test Broker')
        self.assertEqual(data_dict['equity_commission'], 0.0)  # Converted to float
        self.assertEqual(data_dict['options_commission'], 0.65)
    
    def test_data_model_default_values(self):
        """Test data models with default values."""
        # Test TransactionRequest defaults
        request = TransactionRequest(
            symbol='TEST',
            quantity=100,
            price=Decimal('50.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        self.assertEqual(request.order_type, OrderType.MARKET)
        self.assertEqual(request.market_timing, MarketTiming.MARKET_HOURS)
        self.assertEqual(request.currency, 'USD')
        self.assertIsInstance(request.timestamp, datetime)
        
        # Test BrokerConfiguration defaults
        broker = BrokerConfiguration(broker_name='Test')
        self.assertEqual(broker.equity_commission, Decimal('0.00'))
        self.assertEqual(broker.base_currency, 'USD')
        self.assertEqual(broker.pre_market_multiplier, Decimal('1.0'))
        self.assertTrue(broker.active)


class TestMockCostCalculator(unittest.TestCase):
    """Test implementation of a mock cost calculator for testing base functionality."""
    
    def test_mock_calculator_can_be_imported(self):
        """Test that we can create a simple mock calculator."""
        # This is a placeholder test to verify the test structure works
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()