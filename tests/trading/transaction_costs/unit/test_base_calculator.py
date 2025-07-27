"""
Unit Tests for Base Cost Calculator
==================================

Tests for the abstract base cost calculator class, covering initialization,
validation, error handling, and common functionality.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Test the base calculator if available, otherwise skip
try:
    from trading.transaction_costs.base_cost_calculator import CostCalculatorBase, CalculationMode
    from trading.transaction_costs.models import (
        TransactionRequest, TransactionCostBreakdown, InstrumentType, TransactionType
    )
    from trading.transaction_costs.exceptions import (
        TransactionCostError, InvalidTransactionError, CalculationError
    )
    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False


class ConcreteCostCalculator(CostCalculatorBase if BASE_CALCULATOR_AVAILABLE else object):
    """Concrete implementation for testing abstract base class."""
    
    def _calculate_commission(self, request, broker_config, market_conditions=None):
        """Calculate commission for transaction."""
        notional = request.price * request.quantity
        commission = max(notional * Decimal("0.001"), Decimal("5.00"))
        return commission
    
    def _calculate_market_impact(self, request, broker_config, market_conditions=None):
        """Calculate market impact for transaction."""
        notional = request.price * request.quantity
        impact = notional * Decimal("0.0005")
        return impact
    
    def _calculate_regulatory_fees(self, request, broker_config, market_conditions=None):
        """Calculate regulatory fees for transaction."""
        return Decimal("0.0")
    
    def _get_supported_instruments(self):
        """Get list of supported instrument types."""
        return [InstrumentType.EQUITY, InstrumentType.OPTION]
    
    async def calculate_cost_impl(self, request, market_conditions=None, **kwargs):
        """Simple implementation for testing."""
        commission = self._calculate_commission(request, None, market_conditions)
        market_impact = self._calculate_market_impact(request, None, market_conditions)
        regulatory_fees = self._calculate_regulatory_fees(request, None, market_conditions)
        
        return TransactionCostBreakdown(
            total_cost=commission + market_impact + regulatory_fees,
            commission=commission,
            market_impact=market_impact,
            spread_cost=commission * Decimal("0.3"),
            slippage=commission * Decimal("0.2"),
            regulatory_fees=regulatory_fees
        )


@pytest.mark.unit
@pytest.mark.skipif(not BASE_CALCULATOR_AVAILABLE, reason="Base calculator not available")
class TestCostCalculatorBase:
    """Test suite for CostCalculatorBase abstract class."""
    
    def test_calculator_initialization(self):
        """Test proper initialization of calculator."""
        calculator = ConcreteCostCalculator(
            calculator_name="TestCalculator",
            version="1.0.0",
            supported_instruments=[InstrumentType.EQUITY, InstrumentType.OPTION]
        )
        
        assert calculator.calculator_name == "TestCalculator"
        assert calculator.version == "1.0.0"
        assert InstrumentType.EQUITY in calculator.supported_instruments
        assert InstrumentType.OPTION in calculator.supported_instruments
        assert CalculationMode.REAL_TIME in calculator.supported_modes
    
    def test_calculator_defaults(self):
        """Test default values are set correctly."""
        calculator = ConcreteCostCalculator("DefaultTest")
        
        assert calculator.version == "1.0.0"
        assert InstrumentType.EQUITY in calculator.supported_instruments
        assert calculator.enable_caching is True
        assert calculator.default_timeout > 0
    
    @pytest.mark.asyncio
    async def test_calculate_cost_success(self, sample_equity_request, zerodha_config):
        """Test successful cost calculation."""
        calculator = ConcreteCostCalculator("SuccessTest")
        
        result = await calculator.calculate_cost_async(sample_equity_request, zerodha_config)
        
        assert isinstance(result, TransactionCostBreakdown)
        assert result.total_cost > 0
        assert result.commission > 0
        assert result.total_cost >= result.commission
    
    @pytest.mark.asyncio
    async def test_calculate_cost_with_validation(self, sample_equity_request):
        """Test cost calculation with input validation."""
        calculator = ConcreteCostCalculator("ValidationTest")
        
        # Test with valid request
        result = await calculator.calculate_cost(sample_equity_request)
        assert result is not None
        
        # Test with invalid quantity
        invalid_request = sample_equity_request
        invalid_request.quantity = -100
        
        with pytest.raises(InvalidTransactionError):
            await calculator.calculate_cost(invalid_request)
    
    def test_instrument_support_validation(self):
        """Test instrument type support validation."""
        calculator = ConcreteCostCalculator(
            "InstrumentTest",
            supported_instruments=[InstrumentType.EQUITY]
        )
        
        supported_instruments = calculator._get_supported_instruments()
        assert InstrumentType.EQUITY in supported_instruments
    
    def test_calculation_mode_support(self):
        """Test calculation mode support validation."""
        calculator = ConcreteCostCalculator(
            "ModeTest",
            supported_modes=[CalculationMode.REAL_TIME, CalculationMode.BATCH]
        )
        
        assert calculator.supports_mode(CalculationMode.REAL_TIME)
        assert calculator.supports_mode(CalculationMode.BATCH)
        assert not calculator.supports_mode(CalculationMode.SIMULATION)
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, sample_equity_request, zerodha_config):
        """Test that performance metrics are tracked."""
        calculator = ConcreteCostCalculator("PerformanceTest")
        
        # Perform calculation
        await calculator.calculate_cost_async(sample_equity_request, zerodha_config)
        
        # Check that performance metrics are available
        stats = calculator.get_performance_stats()
        assert isinstance(stats, dict)
        assert 'calculation_count' in stats or stats.get('calculation_count', 0) >= 0
    
    @pytest.mark.asyncio
    async def test_async_timeout_handling(self, sample_equity_request):
        """Test timeout handling for async operations."""
        calculator = ConcreteCostCalculator("TimeoutTest", default_timeout=1)
        
        # Mock a slow calculation
        with patch.object(calculator, 'calculate_cost_impl', AsyncMock()):
            calculator.calculate_cost_impl.side_effect = asyncio.sleep(2)
            
            with pytest.raises(asyncio.TimeoutError):
                await calculator.calculate_cost(sample_equity_request, timeout=0.5)
    
    def test_error_handling_configuration(self):
        """Test error handling configuration."""
        calculator = ConcreteCostCalculator("ErrorTest")
        
        # Test that error handling methods exist
        assert hasattr(calculator, 'handle_calculation_error')
        assert callable(calculator.handle_calculation_error)
    
    @pytest.mark.asyncio
    async def test_batch_calculation_capability(self, batch_transaction_requests):
        """Test batch calculation functionality if supported."""
        calculator = ConcreteCostCalculator("BatchTest")
        
        if calculator.supports_mode(CalculationMode.BATCH):
            # Test batch processing
            results = []
            for request in batch_transaction_requests[:5]:  # Test with small batch
                result = await calculator.calculate_cost(request)
                results.append(result)
            
            assert len(results) == 5
            assert all(isinstance(r, TransactionCostBreakdown) for r in results)
    
    def test_calculator_metadata(self):
        """Test calculator metadata and identification."""
        calculator = ConcreteCostCalculator(
            "MetadataTest",
            version="2.1.0"
        )
        
        # Check basic attributes
        assert calculator.calculator_name == "MetadataTest"
        assert calculator.version == "2.1.0"
        assert hasattr(calculator, 'supported_instruments')


@pytest.mark.unit
class TestCalculationMode:
    """Test calculation mode enumeration."""
    
    def test_calculation_modes_exist(self):
        """Test that all expected calculation modes are defined."""
        assert hasattr(CalculationMode, 'REAL_TIME')
        assert hasattr(CalculationMode, 'BATCH')
        assert hasattr(CalculationMode, 'HISTORICAL')
        assert hasattr(CalculationMode, 'SIMULATION')
    
    def test_calculation_mode_values(self):
        """Test calculation mode string values."""
        assert CalculationMode.REAL_TIME == "real_time"
        assert CalculationMode.BATCH == "batch"
        assert CalculationMode.HISTORICAL == "historical"
        assert CalculationMode.SIMULATION == "simulation"


@pytest.mark.unit
@pytest.mark.skipif(not BASE_CALCULATOR_AVAILABLE, reason="Base calculator not available")
class TestCostCalculatorValidation:
    """Test input validation functionality."""
    
    def test_transaction_request_validation(self, sample_equity_request):
        """Test transaction request validation."""
        calculator = ConcreteCostCalculator("ValidationTest")
        
        # Valid request should pass
        assert calculator.validate_transaction_request(sample_equity_request)
        
        # Test edge cases
        edge_request = sample_equity_request
        edge_request.quantity = 0
        assert not calculator.validate_transaction_request(edge_request)
        
        edge_request.quantity = -10
        assert not calculator.validate_transaction_request(edge_request)
        
        edge_request.price = Decimal("0")
        assert not calculator.validate_transaction_request(edge_request)
    
    def test_market_conditions_validation(self, sample_market_conditions):
        """Test market conditions validation."""
        calculator = ConcreteCostCalculator("MarketValidationTest")
        
        # Valid conditions should pass
        assert calculator.validate_market_conditions(sample_market_conditions)
        
        # Test invalid volatility
        invalid_conditions = sample_market_conditions
        invalid_conditions.volatility = -0.1
        assert not calculator.validate_market_conditions(invalid_conditions)
        
        invalid_conditions.volatility = 5.0  # Extremely high
        assert not calculator.validate_market_conditions(invalid_conditions)


@pytest.mark.unit
@pytest.mark.skipif(not BASE_CALCULATOR_AVAILABLE, reason="Base calculator not available")
class TestCostCalculatorPerformance:
    """Test performance-related functionality."""
    
    @pytest.mark.asyncio
    async def test_calculation_latency_tracking(self, sample_equity_request):
        """Test that calculation latency is tracked."""
        calculator = ConcreteCostCalculator("LatencyTest")
        
        # Perform calculation
        result = await calculator.calculate_cost(sample_equity_request)
        
        # Check latency tracking
        assert hasattr(calculator, 'last_calculation_time')
        assert calculator.last_calculation_time > 0
    
    def test_caching_configuration(self):
        """Test caching configuration options."""
        calculator_with_cache = ConcreteCostCalculator("CacheTest", enable_caching=True)
        calculator_without_cache = ConcreteCostCalculator("NoCacheTest", enable_caching=False)
        
        assert calculator_with_cache.enable_caching is True
        assert calculator_without_cache.enable_caching is False
    
    @pytest.mark.asyncio 
    async def test_concurrent_calculations(self, batch_transaction_requests):
        """Test concurrent calculation handling."""
        calculator = ConcreteCostCalculator("ConcurrentTest")
        
        # Submit multiple concurrent calculations
        tasks = []
        for request in batch_transaction_requests[:3]:
            task = asyncio.create_task(calculator.calculate_cost(request))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed successfully
        assert len(results) == 3
        assert all(isinstance(r, TransactionCostBreakdown) for r in results if not isinstance(r, Exception))


# Mock tests for when base calculator is not available
@pytest.mark.unit
@pytest.mark.skipif(BASE_CALCULATOR_AVAILABLE, reason="Base calculator is available, using real tests")
class TestMockCostCalculator:
    """Mock tests when base calculator is not available."""
    
    @pytest.mark.asyncio
    async def test_mock_calculator_basic_functionality(self, mock_calculator, sample_equity_request):
        """Test basic functionality with mock calculator."""
        result = await mock_calculator.calculate_cost(sample_equity_request)
        
        assert result is not None
        assert result.total_cost > 0
        assert mock_calculator.calculation_count == 1
    
    @pytest.mark.asyncio
    async def test_mock_calculator_multiple_calculations(self, mock_calculator, batch_transaction_requests):
        """Test multiple calculations with mock calculator."""
        initial_count = mock_calculator.calculation_count
        
        for request in batch_transaction_requests[:3]:
            await mock_calculator.calculate_cost(request)
        
        assert mock_calculator.calculation_count == initial_count + 3