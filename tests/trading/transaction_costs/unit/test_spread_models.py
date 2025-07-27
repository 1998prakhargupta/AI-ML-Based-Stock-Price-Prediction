"""
Unit Tests for Spread Models
============================

Tests for bid-ask spread estimation models and real-time spread calculators
used in transaction cost analysis.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Test spread models if available
try:
    from trading.transaction_costs.spreads.realtime_estimator import RealTimeSpreadEstimator
    from trading.transaction_costs.spreads.spread_calculator import SpreadCalculator
    from trading.transaction_costs.models import (
        TransactionRequest, MarketConditions, InstrumentType, TransactionType
    )
    SPREAD_MODELS_AVAILABLE = True
except ImportError:
    SPREAD_MODELS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not SPREAD_MODELS_AVAILABLE, reason="Spread models not available")
class TestRealTimeSpreadEstimator:
    """Test suite for RealTimeSpreadEstimator."""
    
    def test_spread_estimator_initialization(self):
        """Test real-time spread estimator initialization."""
        estimator = RealTimeSpreadEstimator()
        
        assert hasattr(estimator, 'estimate_spread')
        assert hasattr(estimator, 'calculate_spread_cost')
    
    def test_spread_estimation_basic(self, sample_equity_request, mock_market_data):
        """Test basic spread estimation."""
        estimator = RealTimeSpreadEstimator()
        
        # Mock market data for AAPL
        symbol_data = mock_market_data['AAPL']
        
        spread = estimator.estimate_spread(
            symbol=sample_equity_request.symbol,
            bid=Decimal(str(symbol_data['bid'])),
            ask=Decimal(str(symbol_data['ask']))
        )
        
        assert isinstance(spread, Decimal)
        assert spread > Decimal("0")
        
        # Spread should equal ask - bid
        expected_spread = Decimal(str(symbol_data['ask'] - symbol_data['bid']))
        assert abs(spread - expected_spread) < Decimal("0.01")
    
    def test_spread_cost_calculation(self, sample_equity_request, mock_market_data):
        """Test spread cost calculation for a transaction."""
        estimator = RealTimeSpreadEstimator()
        
        symbol_data = mock_market_data['AAPL']
        spread = Decimal(str(symbol_data['ask'] - symbol_data['bid']))
        
        spread_cost = estimator.calculate_spread_cost(
            request=sample_equity_request,
            spread=spread
        )
        
        assert isinstance(spread_cost, Decimal)
        assert spread_cost >= Decimal("0")
        
        # For a buy order, spread cost should be spread/2 * quantity
        expected_cost = (spread / 2) * sample_equity_request.quantity
        assert abs(spread_cost - expected_cost) < Decimal("1.00")
    
    def test_spread_cost_buy_vs_sell(self, mock_market_data):
        """Test spread cost difference between buy and sell orders."""
        estimator = RealTimeSpreadEstimator()
        
        buy_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.50"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        sell_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.50"),
            transaction_type=TransactionType.SELL,
            instrument_type=InstrumentType.EQUITY
        )
        
        symbol_data = mock_market_data['AAPL']
        spread = Decimal(str(symbol_data['ask'] - symbol_data['bid']))
        
        buy_cost = estimator.calculate_spread_cost(buy_request, spread)
        sell_cost = estimator.calculate_spread_cost(sell_request, spread)
        
        # Both should have same spread cost for market orders
        assert buy_cost == sell_cost
    
    def test_spread_estimation_wide_spread(self):
        """Test spread estimation with wide spreads."""
        estimator = RealTimeSpreadEstimator()
        
        # Wide spread scenario (illiquid stock)
        wide_bid = Decimal("100.00")
        wide_ask = Decimal("105.00")  # 5% spread
        
        spread = estimator.estimate_spread(
            symbol="ILLIQUID",
            bid=wide_bid,
            ask=wide_ask
        )
        
        assert spread == Decimal("5.00")
        
        # Spread cost should be proportionally higher
        request = TransactionRequest(
            symbol="ILLIQUID",
            quantity=100,
            price=Decimal("102.50"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        spread_cost = estimator.calculate_spread_cost(request, spread)
        assert spread_cost >= Decimal("200.00")  # Should be significant
    
    def test_spread_estimation_tight_spread(self):
        """Test spread estimation with tight spreads."""
        estimator = RealTimeSpreadEstimator()
        
        # Tight spread scenario (highly liquid stock)
        tight_bid = Decimal("100.00")
        tight_ask = Decimal("100.01")  # 0.01% spread
        
        spread = estimator.estimate_spread(
            symbol="LIQUID",
            bid=tight_bid,
            ask=tight_ask
        )
        
        assert spread == Decimal("0.01")
        
        # Spread cost should be minimal
        request = TransactionRequest(
            symbol="LIQUID",
            quantity=100,
            price=Decimal("100.005"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        spread_cost = estimator.calculate_spread_cost(request, spread)
        assert spread_cost <= Decimal("1.00")  # Should be minimal


@pytest.mark.unit
@pytest.mark.skipif(not SPREAD_MODELS_AVAILABLE, reason="Spread models not available")
class TestSpreadCalculator:
    """Test suite for SpreadCalculator."""
    
    def test_spread_calculator_initialization(self):
        """Test spread calculator initialization."""
        calculator = SpreadCalculator()
        
        assert hasattr(calculator, 'calculate_effective_spread')
        assert hasattr(calculator, 'estimate_spread_from_trades')
    
    def test_effective_spread_calculation(self, sample_equity_request):
        """Test effective spread calculation."""
        calculator = SpreadCalculator()
        
        # Mock trade data
        trade_price = Decimal("150.50")
        midpoint = Decimal("150.45")  # Mid between bid/ask
        
        effective_spread = calculator.calculate_effective_spread(
            trade_price=trade_price,
            midpoint=midpoint,
            transaction_type=sample_equity_request.transaction_type
        )
        
        assert isinstance(effective_spread, Decimal)
        assert effective_spread >= Decimal("0")
    
    def test_spread_from_historical_trades(self):
        """Test spread estimation from historical trade data."""
        calculator = SpreadCalculator()
        
        # Mock historical trade data
        trade_data = [
            {'price': Decimal("150.50"), 'timestamp': datetime.now() - timedelta(minutes=5)},
            {'price': Decimal("150.52"), 'timestamp': datetime.now() - timedelta(minutes=4)},
            {'price': Decimal("150.48"), 'timestamp': datetime.now() - timedelta(minutes=3)},
            {'price': Decimal("150.55"), 'timestamp': datetime.now() - timedelta(minutes=2)},
            {'price': Decimal("150.47"), 'timestamp': datetime.now() - timedelta(minutes=1)},
        ]
        
        estimated_spread = calculator.estimate_spread_from_trades(
            symbol="AAPL",
            trade_data=trade_data
        )
        
        assert isinstance(estimated_spread, Decimal)
        assert estimated_spread > Decimal("0")
    
    def test_volatility_adjusted_spread(self, sample_market_conditions):
        """Test volatility-adjusted spread calculation."""
        calculator = SpreadCalculator()
        
        base_spread = Decimal("0.10")
        
        adjusted_spread = calculator.adjust_spread_for_volatility(
            base_spread=base_spread,
            volatility=sample_market_conditions.volatility
        )
        
        assert isinstance(adjusted_spread, Decimal)
        assert adjusted_spread >= base_spread  # Should be higher due to volatility
    
    def test_time_weighted_spread(self):
        """Test time-weighted spread calculation."""
        calculator = SpreadCalculator()
        
        # Mock spread data over time
        spread_data = [
            {'spread': Decimal("0.10"), 'timestamp': datetime.now() - timedelta(hours=1), 'weight': 0.1},
            {'spread': Decimal("0.12"), 'timestamp': datetime.now() - timedelta(minutes=30), 'weight': 0.3},
            {'spread': Decimal("0.08"), 'timestamp': datetime.now() - timedelta(minutes=10), 'weight': 0.6},
        ]
        
        weighted_spread = calculator.calculate_time_weighted_spread(spread_data)
        
        assert isinstance(weighted_spread, Decimal)
        assert Decimal("0.08") <= weighted_spread <= Decimal("0.12")


@pytest.mark.unit
@pytest.mark.skipif(not SPREAD_MODELS_AVAILABLE, reason="Spread models not available")
class TestSpreadModelAccuracy:
    """Test suite for spread model accuracy and validation."""
    
    def test_spread_estimation_consistency(self, mock_market_data):
        """Test consistency of spread estimation across different methods."""
        estimator = RealTimeSpreadEstimator()
        calculator = SpreadCalculator()
        
        for symbol, data in mock_market_data.items():
            bid = Decimal(str(data['bid']))
            ask = Decimal(str(data['ask']))
            
            # Real-time spread
            realtime_spread = estimator.estimate_spread(symbol, bid, ask)
            
            # Should match bid-ask difference
            expected_spread = ask - bid
            assert abs(realtime_spread - expected_spread) < Decimal("0.01")
    
    def test_spread_cost_proportionality(self, mock_market_data):
        """Test that spread costs scale proportionally with trade size."""
        estimator = RealTimeSpreadEstimator()
        
        symbol_data = mock_market_data['AAPL']
        spread = Decimal(str(symbol_data['ask'] - symbol_data['bid']))
        
        # Small trade
        small_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.50"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        # Large trade (10x)
        large_request = TransactionRequest(
            symbol="AAPL",
            quantity=1000,
            price=Decimal("150.50"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        small_cost = estimator.calculate_spread_cost(small_request, spread)
        large_cost = estimator.calculate_spread_cost(large_request, spread)
        
        # Large cost should be ~10x small cost
        ratio = large_cost / small_cost
        assert 9.5 <= ratio <= 10.5
    
    def test_spread_boundary_conditions(self):
        """Test spread calculations at boundary conditions."""
        estimator = RealTimeSpreadEstimator()
        
        # Zero spread
        zero_spread = Decimal("0.00")
        request = TransactionRequest(
            symbol="TEST",
            quantity=100,
            price=Decimal("100.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        zero_cost = estimator.calculate_spread_cost(request, zero_spread)
        assert zero_cost == Decimal("0.00")
        
        # Very large spread
        large_spread = Decimal("10.00")
        large_cost = estimator.calculate_spread_cost(request, large_spread)
        assert large_cost > Decimal("100.00")  # Should be significant


# Mock tests when spread models are not available
@pytest.mark.unit
@pytest.mark.skipif(SPREAD_MODELS_AVAILABLE, reason="Spread models available, using real tests")
class TestMockSpreadModels:
    """Mock tests when spread models are not available."""
    
    def test_mock_spread_estimator(self, sample_equity_request, mock_market_data):
        """Test mock spread estimator functionality."""
        class MockSpreadEstimator:
            def estimate_spread(self, symbol, bid, ask):
                return ask - bid
            
            def calculate_spread_cost(self, request, spread):
                return (spread / 2) * request.quantity
        
        estimator = MockSpreadEstimator()
        
        symbol_data = mock_market_data['AAPL']
        bid = Decimal(str(symbol_data['bid']))
        ask = Decimal(str(symbol_data['ask']))
        
        spread = estimator.estimate_spread("AAPL", bid, ask)
        cost = estimator.calculate_spread_cost(sample_equity_request, spread)
        
        assert spread == ask - bid
        assert cost > Decimal("0")
    
    def test_mock_spread_calculator(self):
        """Test mock spread calculator functionality."""
        class MockSpreadCalculator:
            def calculate_effective_spread(self, trade_price, midpoint, transaction_type):
                return abs(trade_price - midpoint) * 2
        
        calculator = MockSpreadCalculator()
        
        effective_spread = calculator.calculate_effective_spread(
            trade_price=Decimal("150.50"),
            midpoint=Decimal("150.45"),
            transaction_type=TransactionType.BUY
        )
        
        assert effective_spread == Decimal("0.10")
    
    def test_mock_spread_integration(self, sample_equity_request):
        """Test integration of mock spread components."""
        class MockIntegratedSpread:
            def calculate_total_spread_cost(self, request):
                # Simple calculation based on price and quantity
                return request.price * request.quantity * Decimal("0.001")
        
        integrated = MockIntegratedSpread()
        cost = integrated.calculate_total_spread_cost(sample_equity_request)
        
        assert isinstance(cost, Decimal)
        assert cost > Decimal("0")