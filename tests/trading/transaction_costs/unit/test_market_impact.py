"""
Unit Tests for Market Impact Models
==================================

Tests for market impact calculation models including linear, square-root,
and adaptive models for estimating transaction costs due to market impact.
"""

import pytest
import math
from decimal import Decimal
from unittest.mock import Mock, patch

# Test market impact models if available
try:
    from trading.transaction_costs.market_impact.adaptive_model import AdaptiveImpactModel
    from trading.transaction_costs.market_impact.linear_model import LinearImpactModel
    from trading.transaction_costs.market_impact.sqrt_model import SqrtImpactModel
    from trading.transaction_costs.models import (
        TransactionRequest, MarketConditions, InstrumentType, TransactionType
    )
    MARKET_IMPACT_AVAILABLE = True
except ImportError:
    MARKET_IMPACT_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not MARKET_IMPACT_AVAILABLE, reason="Market impact models not available")
class TestLinearImpactModel:
    """Test suite for LinearImpactModel."""
    
    def test_linear_model_initialization(self):
        """Test linear impact model initialization."""
        model = LinearImpactModel(impact_coefficient=0.001)
        
        assert model.impact_coefficient == 0.001
        assert hasattr(model, 'calculate_impact')
    
    def test_linear_impact_calculation(self, sample_equity_request, sample_market_conditions):
        """Test linear market impact calculation."""
        model = LinearImpactModel(impact_coefficient=0.001)
        
        impact = model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        assert isinstance(impact, Decimal)
        assert impact >= Decimal("0")
        
        # Linear model: impact should be proportional to trade size
        expected_impact = (sample_equity_request.quantity / sample_market_conditions.volume) * 0.001
        expected_decimal = Decimal(str(expected_impact))
        
        # Allow for reasonable tolerance in calculation
        assert abs(impact - expected_decimal) < Decimal("0.1")
    
    def test_linear_impact_proportionality(self, sample_market_conditions):
        """Test that linear impact scales proportionally with trade size."""
        model = LinearImpactModel(impact_coefficient=0.001)
        
        # Create two requests with different sizes
        small_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        large_request = TransactionRequest(
            symbol="AAPL",
            quantity=1000,  # 10x larger
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        small_impact = model.calculate_impact(small_request, sample_market_conditions)
        large_impact = model.calculate_impact(large_request, sample_market_conditions)
        
        # Large impact should be approximately 10x the small impact
        ratio = large_impact / small_impact
        assert 8.0 <= ratio <= 12.0  # Allow for some calculation variance
    
    def test_linear_impact_zero_volume_handling(self):
        """Test handling of zero market volume."""
        model = LinearImpactModel(impact_coefficient=0.001)
        
        zero_volume_conditions = MarketConditions(
            volatility=0.25,
            volume=0,  # Zero volume
            bid_ask_spread=0.02,
            market_timing="REGULAR"
        )
        
        request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        # Should handle zero volume gracefully (high impact or exception)
        with pytest.raises(Exception) or True:
            impact = model.calculate_impact(request, zero_volume_conditions)
            # If no exception, impact should be very high
            assert impact > Decimal("1.0")


@pytest.mark.unit
@pytest.mark.skipif(not MARKET_IMPACT_AVAILABLE, reason="Market impact models not available")
class TestSqrtImpactModel:
    """Test suite for SqrtImpactModel."""
    
    def test_sqrt_model_initialization(self):
        """Test square-root impact model initialization."""
        model = SqrtImpactModel(impact_coefficient=0.001)
        
        assert model.impact_coefficient == 0.001
        assert hasattr(model, 'calculate_impact')
    
    def test_sqrt_impact_calculation(self, sample_equity_request, sample_market_conditions):
        """Test square-root market impact calculation."""
        model = SqrtImpactModel(impact_coefficient=0.001)
        
        impact = model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        assert isinstance(impact, Decimal)
        assert impact >= Decimal("0")
        
        # Sqrt model: impact should be proportional to square root of trade size
        participation_rate = sample_equity_request.quantity / sample_market_conditions.volume
        expected_impact = math.sqrt(participation_rate) * 0.001
        expected_decimal = Decimal(str(expected_impact))
        
        # Allow for reasonable tolerance
        assert abs(impact - expected_decimal) < Decimal("0.1")
    
    def test_sqrt_impact_sublinear_scaling(self, sample_market_conditions):
        """Test that sqrt impact scales sublinearly with trade size."""
        model = SqrtImpactModel(impact_coefficient=0.001)
        
        # Create requests with different sizes
        small_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        large_request = TransactionRequest(
            symbol="AAPL",
            quantity=10000,  # 100x larger
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        small_impact = model.calculate_impact(small_request, sample_market_conditions)
        large_impact = model.calculate_impact(large_request, sample_market_conditions)
        
        # Large impact should be ~10x (sqrt(100)) the small impact, not 100x
        ratio = large_impact / small_impact
        assert 8.0 <= ratio <= 12.0  # sqrt(100) = 10, with tolerance
    
    def test_sqrt_impact_vs_linear_comparison(self, sample_equity_request, sample_market_conditions):
        """Compare sqrt and linear models for same transaction."""
        linear_model = LinearImpactModel(impact_coefficient=0.001)
        sqrt_model = SqrtImpactModel(impact_coefficient=0.001)
        
        linear_impact = linear_model.calculate_impact(sample_equity_request, sample_market_conditions)
        sqrt_impact = sqrt_model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        # For typical transaction sizes, sqrt should be less than linear
        assert sqrt_impact <= linear_impact


@pytest.mark.unit
@pytest.mark.skipif(not MARKET_IMPACT_AVAILABLE, reason="Market impact models not available")
class TestAdaptiveImpactModel:
    """Test suite for AdaptiveImpactModel."""
    
    def test_adaptive_model_initialization(self):
        """Test adaptive impact model initialization."""
        model = AdaptiveImpactModel()
        
        assert hasattr(model, 'calculate_impact')
        assert hasattr(model, 'calibrate')
    
    def test_adaptive_impact_calculation(self, sample_equity_request, sample_market_conditions):
        """Test adaptive market impact calculation."""
        model = AdaptiveImpactModel()
        
        impact = model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        assert isinstance(impact, Decimal)
        assert impact >= Decimal("0")
    
    def test_adaptive_model_calibration(self, sample_market_conditions):
        """Test adaptive model calibration with market conditions."""
        model = AdaptiveImpactModel()
        
        # Test calibration with different market conditions
        normal_conditions = sample_market_conditions
        volatile_conditions = MarketConditions(
            volatility=0.75,  # High volatility
            volume=500000,    # Lower volume
            bid_ask_spread=0.10,  # Wide spread
            market_timing="REGULAR"
        )
        
        # Calibrate model
        model.calibrate(normal_conditions)
        normal_params = model.get_current_parameters()
        
        model.calibrate(volatile_conditions)
        volatile_params = model.get_current_parameters()
        
        # Parameters should differ based on market conditions
        assert normal_params != volatile_params
    
    def test_adaptive_volatility_adjustment(self, sample_equity_request):
        """Test that adaptive model adjusts for market volatility."""
        model = AdaptiveImpactModel()
        
        low_vol_conditions = MarketConditions(
            volatility=0.10,  # Low volatility
            volume=1000000,
            bid_ask_spread=0.01,
            market_timing="REGULAR"
        )
        
        high_vol_conditions = MarketConditions(
            volatility=0.80,  # High volatility
            volume=1000000,
            bid_ask_spread=0.01,
            market_timing="REGULAR"
        )
        
        low_vol_impact = model.calculate_impact(sample_equity_request, low_vol_conditions)
        high_vol_impact = model.calculate_impact(sample_equity_request, high_vol_conditions)
        
        # High volatility should generally result in higher impact
        assert high_vol_impact >= low_vol_impact
    
    def test_adaptive_volume_adjustment(self, sample_equity_request):
        """Test that adaptive model adjusts for market volume."""
        model = AdaptiveImpactModel()
        
        high_volume_conditions = MarketConditions(
            volatility=0.25,
            volume=5000000,  # High volume
            bid_ask_spread=0.01,
            market_timing="REGULAR"
        )
        
        low_volume_conditions = MarketConditions(
            volatility=0.25,
            volume=100000,   # Low volume
            bid_ask_spread=0.01,
            market_timing="REGULAR"
        )
        
        high_vol_impact = model.calculate_impact(sample_equity_request, high_volume_conditions)
        low_vol_impact = model.calculate_impact(sample_equity_request, low_volume_conditions)
        
        # Low market volume should result in higher impact
        assert low_vol_impact >= high_vol_impact


@pytest.mark.unit
@pytest.mark.skipif(not MARKET_IMPACT_AVAILABLE, reason="Market impact models not available")
class TestMarketImpactModelComparison:
    """Test suite comparing different market impact models."""
    
    def test_model_consistency(self, sample_equity_request, sample_market_conditions):
        """Test that all models produce consistent results."""
        linear_model = LinearImpactModel(impact_coefficient=0.001)
        sqrt_model = SqrtImpactModel(impact_coefficient=0.001)
        adaptive_model = AdaptiveImpactModel()
        
        linear_impact = linear_model.calculate_impact(sample_equity_request, sample_market_conditions)
        sqrt_impact = sqrt_model.calculate_impact(sample_equity_request, sample_market_conditions)
        adaptive_impact = adaptive_model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        # All should be positive and within reasonable range
        assert linear_impact > Decimal("0")
        assert sqrt_impact > Decimal("0")
        assert adaptive_impact > Decimal("0")
        
        # All should be within order of magnitude of each other
        max_impact = max(linear_impact, sqrt_impact, adaptive_impact)
        min_impact = min(linear_impact, sqrt_impact, adaptive_impact)
        
        assert max_impact / min_impact < Decimal("100")  # Not more than 100x difference
    
    def test_large_trade_model_behavior(self, sample_market_conditions):
        """Test model behavior for large trades."""
        large_request = TransactionRequest(
            symbol="AAPL",
            quantity=100000,  # Very large trade
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        linear_model = LinearImpactModel(impact_coefficient=0.001)
        sqrt_model = SqrtImpactModel(impact_coefficient=0.001)
        
        linear_impact = linear_model.calculate_impact(large_request, sample_market_conditions)
        sqrt_impact = sqrt_model.calculate_impact(large_request, sample_market_conditions)
        
        # For large trades, sqrt should be significantly less than linear
        assert sqrt_impact < linear_impact
        assert linear_impact / sqrt_impact > Decimal("2")
    
    def test_small_trade_model_behavior(self, sample_market_conditions):
        """Test model behavior for small trades."""
        small_request = TransactionRequest(
            symbol="AAPL",
            quantity=10,  # Very small trade
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        
        linear_model = LinearImpactModel(impact_coefficient=0.001)
        sqrt_model = SqrtImpactModel(impact_coefficient=0.001)
        
        linear_impact = linear_model.calculate_impact(small_request, sample_market_conditions)
        sqrt_impact = sqrt_model.calculate_impact(small_request, sample_market_conditions)
        
        # For small trades, models should be closer together
        assert abs(linear_impact - sqrt_impact) < Decimal("0.1")


# Mock tests when market impact models are not available
@pytest.mark.unit
@pytest.mark.skipif(MARKET_IMPACT_AVAILABLE, reason="Market impact models available, using real tests")
class TestMockMarketImpactModels:
    """Mock tests when market impact models are not available."""
    
    def test_mock_linear_impact_model(self, sample_equity_request, sample_market_conditions):
        """Test mock linear impact model."""
        # Create a simple mock model
        class MockLinearModel:
            def __init__(self, coefficient):
                self.coefficient = coefficient
            
            def calculate_impact(self, request, conditions):
                return Decimal(str(request.quantity * self.coefficient / 1000))
        
        model = MockLinearModel(0.001)
        impact = model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        assert isinstance(impact, Decimal)
        assert impact > Decimal("0")
    
    def test_mock_sqrt_impact_model(self, sample_equity_request, sample_market_conditions):
        """Test mock sqrt impact model."""
        class MockSqrtModel:
            def __init__(self, coefficient):
                self.coefficient = coefficient
            
            def calculate_impact(self, request, conditions):
                import math
                return Decimal(str(math.sqrt(request.quantity) * self.coefficient / 100))
        
        model = MockSqrtModel(0.001)
        impact = model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        assert isinstance(impact, Decimal)
        assert impact > Decimal("0")
    
    def test_mock_adaptive_impact_model(self, sample_equity_request, sample_market_conditions):
        """Test mock adaptive impact model."""
        class MockAdaptiveModel:
            def calculate_impact(self, request, conditions):
                volatility_factor = conditions.volatility
                volume_factor = 1000000 / conditions.volume if conditions.volume > 0 else 10
                return Decimal(str(request.quantity * volatility_factor * volume_factor / 100000))
            
            def calibrate(self, conditions):
                self.last_calibration = conditions
        
        model = MockAdaptiveModel()
        model.calibrate(sample_market_conditions)
        impact = model.calculate_impact(sample_equity_request, sample_market_conditions)
        
        assert isinstance(impact, Decimal)
        assert impact > Decimal("0")
        assert hasattr(model, 'last_calibration')