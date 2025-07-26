"""
Test Broker-Specific Calculators
================================

Comprehensive unit tests for Indian broker-specific cost calculators
including Zerodha and ICICI Securities (Breeze Connect).
"""

import unittest
from decimal import Decimal
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.transaction_costs.models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    TransactionType,
    InstrumentType,
    OrderType
)
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
from src.trading.transaction_costs.brokers.breeze_calculator import BreezeCalculator
from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory
from src.trading.transaction_costs.brokers.regulatory.stt_calculator import STTCalculator
from src.trading.transaction_costs.brokers.regulatory.gst_calculator import GSTCalculator
from src.trading.transaction_costs.brokers.regulatory.stamp_duty_calculator import StampDutyCalculator
from src.trading.transaction_costs.exceptions import BrokerConfigurationError


class TestZerodhaCalculator(unittest.TestCase):
    """Test suite for Zerodha cost calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ZerodhaCalculator()
        
        self.broker_config = BrokerConfiguration(
            broker_name='Zerodha',
            base_currency='INR'
        )
    
    def test_equity_delivery_commission(self):
        """Test commission calculation for equity delivery."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'delivery'}
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # Zerodha charges ₹0 for delivery
        self.assertEqual(commission, Decimal('0.00'))
    
    def test_equity_intraday_commission(self):
        """Test commission calculation for equity intraday."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'intraday'}
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # Should be 0.03% or ₹20, whichever is lower
        notional = Decimal('250000.00')  # 100 * 2500
        expected = min(notional * Decimal('0.0003'), Decimal('20.00'))
        self.assertEqual(commission, expected)
    
    def test_options_commission(self):
        """Test commission calculation for options."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=1,
            price=Decimal('50.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.OPTION
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # Zerodha charges ₹20 per options order
        self.assertEqual(commission, Decimal('20.00'))
    
    def test_futures_commission(self):
        """Test commission calculation for futures."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=1,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.FUTURE
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # Should be 0.03% or ₹20, whichever is lower
        notional = Decimal('2500.00')
        expected = min(notional * Decimal('0.0003'), Decimal('20.00'))
        self.assertEqual(commission, expected)


class TestBreezeCalculator(unittest.TestCase):
    """Test suite for ICICI Securities (Breeze) cost calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = BreezeCalculator()
        
        self.broker_config = BrokerConfiguration(
            broker_name='ICICI Securities',
            base_currency='INR'
        )
    
    def test_equity_delivery_commission(self):
        """Test commission calculation for equity delivery."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'delivery'}
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # ICICI charges ₹20 per order for delivery
        self.assertEqual(commission, Decimal('20.00'))
    
    def test_equity_intraday_commission(self):
        """Test commission calculation for equity intraday."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'intraday'}
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # Should be 0.05% or ₹20, whichever is lower
        notional = Decimal('250000.00')  # 100 * 2500
        expected = min(notional * Decimal('0.0005'), Decimal('20.00'))
        self.assertEqual(commission, expected)
    
    def test_options_commission(self):
        """Test commission calculation for options."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=1,
            price=Decimal('50.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.OPTION
        )
        
        commission = self.calculator._calculate_commission(request, self.broker_config)
        
        # ICICI charges ₹20 per options order
        self.assertEqual(commission, Decimal('20.00'))


class TestSTTCalculator(unittest.TestCase):
    """Test suite for STT calculator."""
    
    def test_equity_stt_calculation(self):
        """Test STT calculation for equity."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        stt = STTCalculator.calculate_stt(request)
        
        # STT for equity is 0.1% on both buy and sell
        expected = request.notional_value * Decimal('0.001')
        self.assertEqual(stt, expected)
    
    def test_options_stt_calculation(self):
        """Test STT calculation for options."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=1,
            price=Decimal('50.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.OPTION
        )
        
        stt = STTCalculator.calculate_stt(request)
        
        # STT for options is 0.05%
        expected = request.notional_value * Decimal('0.0005')
        self.assertEqual(stt, expected)


class TestGSTCalculator(unittest.TestCase):
    """Test suite for GST calculator."""
    
    def test_gst_calculation(self):
        """Test GST calculation on brokerage."""
        brokerage = Decimal('20.00')
        
        gst = GSTCalculator.calculate_gst(brokerage)
        
        # GST is 18%
        expected = brokerage * Decimal('0.18')
        self.assertEqual(gst, expected)
    
    def test_total_with_gst(self):
        """Test total amount calculation including GST."""
        base_amount = Decimal('100.00')
        
        total = GSTCalculator.calculate_total_with_gst(base_amount)
        
        # Total should be base + 18% GST
        expected = base_amount * Decimal('1.18')
        self.assertEqual(total, expected)


class TestStampDutyCalculator(unittest.TestCase):
    """Test suite for stamp duty calculator."""
    
    def test_stamp_duty_buy_transaction(self):
        """Test stamp duty calculation for buy transaction."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        stamp_duty = StampDutyCalculator.calculate_stamp_duty(request)
        
        # Stamp duty is 0.003% on buy side
        expected = request.notional_value * Decimal('0.00003')
        self.assertEqual(stamp_duty, expected)
    
    def test_stamp_duty_sell_transaction(self):
        """Test stamp duty calculation for sell transaction (should be zero)."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.SELL,
            instrument_type=InstrumentType.EQUITY
        )
        
        stamp_duty = StampDutyCalculator.calculate_stamp_duty(request)
        
        # No stamp duty on sell transactions
        self.assertEqual(stamp_duty, Decimal('0.00'))
    
    def test_stamp_duty_maximum_cap(self):
        """Test stamp duty maximum cap of ₹1500."""
        # Create a large transaction that would exceed ₹1500 stamp duty
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=10000,
            price=Decimal('10000.00'),  # ₹10 crore transaction
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        stamp_duty = StampDutyCalculator.calculate_stamp_duty(request)
        
        # Should be capped at ₹1500
        self.assertEqual(stamp_duty, Decimal('1500.00'))


class TestBrokerFactory(unittest.TestCase):
    """Test suite for broker factory."""
    
    def test_create_zerodha_calculator(self):
        """Test creating Zerodha calculator through factory."""
        calculator = BrokerFactory.create_calculator('zerodha')
        
        self.assertIsInstance(calculator, ZerodhaCalculator)
        self.assertEqual(calculator.calculator_name, 'Zerodha')
    
    def test_create_icici_calculator(self):
        """Test creating ICICI calculator through factory."""
        calculator = BrokerFactory.create_calculator('icici')
        
        self.assertIsInstance(calculator, BreezeCalculator)
        self.assertEqual(calculator.calculator_name, 'ICICI_Securities')
    
    def test_unsupported_broker(self):
        """Test error handling for unsupported broker."""
        with self.assertRaises(BrokerConfigurationError):
            BrokerFactory.create_calculator('unsupported_broker')
    
    def test_get_supported_brokers(self):
        """Test getting list of supported brokers."""
        brokers = BrokerFactory.get_supported_brokers()
        
        self.assertIn('zerodha', brokers)
        self.assertIn('icici', brokers)
        self.assertIn('kite', brokers)  # Alias for Zerodha
        self.assertIn('breeze', brokers)  # Alias for ICICI
    
    def test_broker_info(self):
        """Test getting broker information."""
        info = BrokerFactory.get_broker_info('zerodha')
        
        self.assertIsNotNone(info)
        self.assertEqual(info['broker_name'], 'zerodha')
        self.assertIn('supported_instruments', info)


class TestIntegratedCalculations(unittest.TestCase):
    """Test integrated calculations with all components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.zerodha_calc = ZerodhaCalculator()
        self.icici_calc = BreezeCalculator()
        
        self.broker_config = BrokerConfiguration(
            broker_name='Test Broker',
            base_currency='INR'
        )
        
        self.market_conditions = MarketConditions(
            bid_price=Decimal('2499.50'),
            ask_price=Decimal('2500.50'),
            volume=1000000,
            timestamp=datetime.now()
        )
    
    def test_zerodha_full_calculation(self):
        """Test full cost calculation for Zerodha."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'delivery'}
        )
        
        result = self.zerodha_calc.calculate_cost(
            request,
            self.broker_config,
            self.market_conditions
        )
        
        self.assertIsInstance(result, TransactionCostBreakdown)
        self.assertEqual(result.commission, Decimal('0.00'))  # Free delivery
        self.assertGreater(result.total_cost, Decimal('0.00'))  # Should have regulatory fees
    
    def test_icici_full_calculation(self):
        """Test full cost calculation for ICICI Securities."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'delivery'}
        )
        
        result = self.icici_calc.calculate_cost(
            request,
            self.broker_config,
            self.market_conditions
        )
        
        self.assertIsInstance(result, TransactionCostBreakdown)
        self.assertEqual(result.commission, Decimal('20.00'))  # ₹20 for delivery
        self.assertGreater(result.total_cost, Decimal('20.00'))  # Should include regulatory fees
    
    def test_cost_comparison(self):
        """Test cost comparison between brokers."""
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'intraday'}
        )
        
        zerodha_result = self.zerodha_calc.calculate_cost(
            request,
            self.broker_config,
            self.market_conditions
        )
        
        icici_result = self.icici_calc.calculate_cost(
            request,
            self.broker_config,
            self.market_conditions
        )
        
        # For this large intraday transaction, both should charge similar amounts
        # (both would hit the ₹20 cap)
        self.assertAlmostEqual(
            float(zerodha_result.commission),
            float(icici_result.commission),
            places=2
        )


if __name__ == '__main__':
    unittest.main()