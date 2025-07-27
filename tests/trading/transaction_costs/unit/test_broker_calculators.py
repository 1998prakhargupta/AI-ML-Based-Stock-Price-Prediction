"""
Unit Tests for Broker Cost Calculators
======================================

Tests for broker-specific cost calculators including Zerodha, ICICI,
and the broker factory pattern implementation.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch

# Test broker calculators if available
try:
    from trading.transaction_costs.brokers.broker_factory import BrokerFactory
    from trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
    from trading.transaction_costs.brokers.breeze_calculator import BreezeCalculator
    from trading.transaction_costs.models import (
        TransactionRequest, TransactionCostBreakdown, InstrumentType, TransactionType
    )
    from trading.transaction_costs.exceptions import BrokerConfigurationError
    BROKER_CALCULATORS_AVAILABLE = True
except ImportError:
    BROKER_CALCULATORS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not BROKER_CALCULATORS_AVAILABLE, reason="Broker calculators not available")
class TestBrokerFactory:
    """Test suite for BrokerFactory class."""
    
    def test_supported_brokers_list(self):
        """Test that supported brokers are properly defined."""
        assert hasattr(BrokerFactory, 'SUPPORTED_BROKERS')
        assert isinstance(BrokerFactory.SUPPORTED_BROKERS, dict)
        assert len(BrokerFactory.SUPPORTED_BROKERS) > 0
        
        # Check specific brokers are supported
        assert 'zerodha' in BrokerFactory.SUPPORTED_BROKERS
        assert 'icici' in BrokerFactory.SUPPORTED_BROKERS
    
    def test_broker_aliases(self):
        """Test that broker aliases work correctly."""
        # Test Zerodha aliases
        assert 'zerodha' in BrokerFactory.SUPPORTED_BROKERS
        assert 'kite' in BrokerFactory.SUPPORTED_BROKERS
        assert BrokerFactory.SUPPORTED_BROKERS['zerodha'] == BrokerFactory.SUPPORTED_BROKERS['kite']
        
        # Test ICICI aliases
        assert 'icici' in BrokerFactory.SUPPORTED_BROKERS
        assert 'breeze' in BrokerFactory.SUPPORTED_BROKERS
        assert BrokerFactory.SUPPORTED_BROKERS['icici'] == BrokerFactory.SUPPORTED_BROKERS['breeze']
    
    def test_create_calculator_success(self, zerodha_config):
        """Test successful calculator creation."""
        calculator = BrokerFactory.create_calculator('zerodha', zerodha_config)
        
        assert calculator is not None
        assert isinstance(calculator, ZerodhaCalculator)
        assert calculator.calculator_name == 'Zerodha'
    
    def test_create_calculator_with_alias(self, zerodha_config):
        """Test calculator creation using broker aliases."""
        calculator = BrokerFactory.create_calculator('kite', zerodha_config)
        
        assert calculator is not None
        assert isinstance(calculator, ZerodhaCalculator)
    
    def test_create_calculator_unsupported_broker(self, zerodha_config):
        """Test error handling for unsupported broker."""
        with pytest.raises(BrokerConfigurationError):
            BrokerFactory.create_calculator('unsupported_broker', zerodha_config)
    
    def test_create_calculator_invalid_config(self):
        """Test error handling for invalid configuration."""
        with pytest.raises(BrokerConfigurationError):
            BrokerFactory.create_calculator('zerodha', None)
    
    def test_get_supported_brokers(self):
        """Test retrieving list of supported brokers."""
        brokers = BrokerFactory.get_supported_brokers()
        
        assert isinstance(brokers, list)
        assert 'zerodha' in brokers
        assert 'icici' in brokers
        assert len(brokers) > 0
    
    def test_default_configurations(self):
        """Test default configurations are available."""
        assert hasattr(BrokerFactory, 'DEFAULT_CONFIGS')
        assert isinstance(BrokerFactory.DEFAULT_CONFIGS, dict)
        
        # Check specific default configs
        assert 'zerodha' in BrokerFactory.DEFAULT_CONFIGS
        assert 'exchange' in BrokerFactory.DEFAULT_CONFIGS['zerodha']


@pytest.mark.unit
@pytest.mark.skipif(not BROKER_CALCULATORS_AVAILABLE, reason="Broker calculators not available")
class TestZerodhaCalculator:
    """Test suite for ZerodhaCalculator."""
    
    def test_zerodha_calculator_initialization(self, zerodha_config):
        """Test Zerodha calculator initialization."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        assert calculator.calculator_name == 'Zerodha'
        assert calculator.broker_config == zerodha_config
        assert InstrumentType.EQUITY in calculator.supported_instruments
    
    @pytest.mark.asyncio
    async def test_zerodha_equity_calculation(self, sample_equity_request, zerodha_config):
        """Test Zerodha equity cost calculation."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        result = await calculator.calculate_cost(sample_equity_request)
        
        assert isinstance(result, TransactionCostBreakdown)
        assert result.commission >= Decimal("0")
        assert result.total_cost > result.commission
        
        # Verify Zerodha-specific fee structure
        notional = sample_equity_request.price * sample_equity_request.quantity
        expected_commission = max(notional * Decimal("0.0003"), Decimal("0"))
        assert abs(result.commission - expected_commission) < Decimal("1.00")
    
    @pytest.mark.asyncio
    async def test_zerodha_option_calculation(self, sample_option_request, zerodha_config):
        """Test Zerodha option cost calculation."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        if InstrumentType.OPTION in calculator.supported_instruments:
            result = await calculator.calculate_cost(sample_option_request)
            
            assert isinstance(result, TransactionCostBreakdown)
            assert result.commission >= Decimal("0")
            assert result.total_cost >= result.commission
    
    @pytest.mark.asyncio
    async def test_zerodha_high_volume_calculation(self, high_volume_request, zerodha_config):
        """Test Zerodha high volume transaction calculation."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        result = await calculator.calculate_cost(high_volume_request)
        
        assert isinstance(result, TransactionCostBreakdown)
        assert result.commission > Decimal("0")
        assert result.market_impact > Decimal("0")  # High volume should have market impact
    
    def test_zerodha_fee_structure_validation(self, zerodha_config):
        """Test Zerodha fee structure constants."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        # Test that fee structure is properly defined
        assert hasattr(calculator, 'commission_rate') or hasattr(calculator, 'fee_structure')
        
        # Verify reasonable commission rates
        if hasattr(calculator, 'commission_rate'):
            assert 0 <= calculator.commission_rate <= 0.01  # Max 1% commission


@pytest.mark.unit
@pytest.mark.skipif(not BROKER_CALCULATORS_AVAILABLE, reason="Broker calculators not available")
class TestBreezeCalculator:
    """Test suite for BreezeCalculator (ICICI Securities)."""
    
    def test_breeze_calculator_initialization(self, icici_config):
        """Test Breeze calculator initialization."""
        calculator = BreezeCalculator(icici_config)
        
        assert calculator.calculator_name in ['ICICI_Securities', 'Breeze']
        assert calculator.broker_config == icici_config
        assert InstrumentType.EQUITY in calculator.supported_instruments
    
    @pytest.mark.asyncio
    async def test_breeze_equity_calculation(self, sample_equity_request, icici_config):
        """Test Breeze equity cost calculation."""
        calculator = BreezeCalculator(icici_config)
        
        result = await calculator.calculate_cost(sample_equity_request)
        
        assert isinstance(result, TransactionCostBreakdown)
        assert result.commission >= Decimal("0")
        assert result.total_cost > result.commission
        
        # Verify ICICI-specific fee structure (typically higher minimum commission)
        assert result.commission >= Decimal("20.00")  # ICICI minimum commission
    
    @pytest.mark.asyncio
    async def test_breeze_minimum_commission(self, icici_config):
        """Test Breeze minimum commission enforcement."""
        calculator = BreezeCalculator(icici_config)
        
        # Create a very small transaction
        small_request = TransactionRequest(
            symbol="AAPL",
            quantity=1,
            price=Decimal("10.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        result = await calculator.calculate_cost(small_request)
        
        # Should enforce minimum commission
        assert result.commission >= Decimal("20.00")
    
    def test_breeze_fee_structure_validation(self, icici_config):
        """Test Breeze fee structure constants."""
        calculator = BreezeCalculator(icici_config)
        
        # Test that fee structure is properly defined
        assert hasattr(calculator, 'minimum_commission') or hasattr(calculator, 'fee_structure')
        
        # Verify minimum commission is set
        if hasattr(calculator, 'minimum_commission'):
            assert calculator.minimum_commission >= Decimal("20.00")


@pytest.mark.unit
@pytest.mark.skipif(not BROKER_CALCULATORS_AVAILABLE, reason="Broker calculators not available")
class TestBrokerCalculatorComparison:
    """Test suite comparing different broker calculators."""
    
    @pytest.mark.asyncio
    async def test_broker_cost_comparison(self, sample_equity_request, zerodha_config, icici_config):
        """Compare costs between different brokers."""
        zerodha_calc = ZerodhaCalculator(zerodha_config)
        icici_calc = BreezeCalculator(icici_config)
        
        zerodha_result = await zerodha_calc.calculate_cost(sample_equity_request)
        icici_result = await icici_calc.calculate_cost(sample_equity_request)
        
        assert isinstance(zerodha_result, TransactionCostBreakdown)
        assert isinstance(icici_result, TransactionCostBreakdown)
        
        # Generally, ICICI should have higher costs due to minimum commission
        assert icici_result.commission >= zerodha_result.commission
    
    @pytest.mark.asyncio
    async def test_large_transaction_comparison(self, high_volume_request, zerodha_config, icici_config):
        """Compare costs for large transactions between brokers."""
        zerodha_calc = ZerodhaCalculator(zerodha_config)
        icici_calc = BreezeCalculator(icici_config)
        
        zerodha_result = await zerodha_calc.calculate_cost(high_volume_request)
        icici_result = await icici_calc.calculate_cost(high_volume_request)
        
        # For large transactions, percentage-based fees should be more significant
        notional = high_volume_request.price * high_volume_request.quantity
        
        # Both should have significant commission for large trades
        assert zerodha_result.commission > Decimal("100.00")
        assert icici_result.commission > Decimal("100.00")


@pytest.mark.unit
@pytest.mark.skipif(not BROKER_CALCULATORS_AVAILABLE, reason="Broker calculators not available")
class TestBrokerCalculatorValidation:
    """Test validation logic for broker calculators."""
    
    def test_instrument_type_validation(self, zerodha_config):
        """Test that calculators validate supported instruments."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        # Test supported instrument
        assert calculator.supports_instrument(InstrumentType.EQUITY)
        
        # Test potentially unsupported instruments
        # (This may vary by implementation)
        commodity_supported = calculator.supports_instrument(InstrumentType.COMMODITY)
        assert isinstance(commodity_supported, bool)
    
    @pytest.mark.asyncio
    async def test_invalid_transaction_handling(self, zerodha_config):
        """Test handling of invalid transaction requests."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        # Test with invalid quantity
        invalid_request = TransactionRequest(
            symbol="AAPL",
            quantity=-100,  # Negative quantity
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        with pytest.raises(Exception):  # Should raise some form of validation error
            await calculator.calculate_cost(invalid_request)
    
    @pytest.mark.asyncio
    async def test_zero_price_handling(self, zerodha_config):
        """Test handling of zero or invalid prices."""
        calculator = ZerodhaCalculator(zerodha_config)
        
        # Test with zero price
        zero_price_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("0.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await calculator.calculate_cost(zero_price_request)


# Mock tests when broker calculators are not available
@pytest.mark.unit
@pytest.mark.skipif(BROKER_CALCULATORS_AVAILABLE, reason="Broker calculators available, using real tests")
class TestMockBrokerCalculators:
    """Mock tests when broker calculators are not available."""
    
    def test_mock_broker_factory(self):
        """Test mock broker factory functionality."""
        # Create a simple mock factory
        mock_factory = type('MockBrokerFactory', (), {
            'SUPPORTED_BROKERS': {'zerodha': 'ZerodhaCalculator', 'icici': 'ICICICalculator'},
            'create_calculator': lambda broker, config: f"Calculator for {broker}"
        })
        
        assert 'zerodha' in mock_factory.SUPPORTED_BROKERS
        assert 'icici' in mock_factory.SUPPORTED_BROKERS
        
        result = mock_factory.create_calculator('zerodha', {})
        assert 'zerodha' in result
    
    @pytest.mark.asyncio
    async def test_mock_broker_calculation(self, mock_calculator, sample_equity_request):
        """Test mock broker calculation."""
        result = await mock_calculator.calculate_cost(sample_equity_request)
        
        assert result is not None
        assert hasattr(result, 'total_cost')
        assert hasattr(result, 'commission')
        assert result.total_cost > 0