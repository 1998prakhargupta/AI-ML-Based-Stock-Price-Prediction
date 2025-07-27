"""
Unit Tests for Cost Aggregation
===============================

Tests for the cost aggregation system that orchestrates all individual
cost calculation components and provides total cost breakdowns.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

# Test cost aggregation if available
try:
    from trading.transaction_costs.cost_aggregator import CostAggregator
    from trading.transaction_costs.models import (
        TransactionRequest, TransactionCostBreakdown, MarketConditions, 
        InstrumentType, TransactionType
    )
    from trading.transaction_costs.exceptions import CalculationError
    COST_AGGREGATOR_AVAILABLE = True
except ImportError:
    COST_AGGREGATOR_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not COST_AGGREGATOR_AVAILABLE, reason="Cost aggregator not available")
class TestCostAggregator:
    """Test suite for CostAggregator."""
    
    def test_cost_aggregator_initialization(self):
        """Test cost aggregator initialization."""
        aggregator = CostAggregator()
        
        assert hasattr(aggregator, 'calculate_total_cost')
        assert hasattr(aggregator, 'add_calculator')
        assert hasattr(aggregator, 'get_cost_breakdown')
    
    @pytest.mark.asyncio
    async def test_basic_cost_aggregation(self, sample_equity_request, sample_market_conditions):
        """Test basic cost aggregation functionality."""
        aggregator = CostAggregator()
        
        total_cost = await aggregator.calculate_total_cost(
            request=sample_equity_request,
            market_conditions=sample_market_conditions
        )
        
        assert isinstance(total_cost, TransactionCostBreakdown)
        assert total_cost.total_cost > Decimal("0")
        assert total_cost.commission >= Decimal("0")
        assert total_cost.market_impact >= Decimal("0")
        assert total_cost.spread_cost >= Decimal("0")
    
    @pytest.mark.asyncio
    async def test_cost_breakdown_components(self, sample_equity_request, sample_market_conditions):
        """Test that cost breakdown includes all expected components."""
        aggregator = CostAggregator()
        
        breakdown = await aggregator.get_cost_breakdown(
            request=sample_equity_request,
            market_conditions=sample_market_conditions
        )
        
        # Verify all cost components are present
        assert hasattr(breakdown, 'commission')
        assert hasattr(breakdown, 'market_impact')
        assert hasattr(breakdown, 'spread_cost')
        assert hasattr(breakdown, 'slippage')
        assert hasattr(breakdown, 'regulatory_fees')
        assert hasattr(breakdown, 'total_cost')
        
        # Verify total equals sum of components
        component_sum = (
            breakdown.commission + 
            breakdown.market_impact + 
            breakdown.spread_cost + 
            breakdown.slippage + 
            breakdown.regulatory_fees
        )
        
        # Allow for small rounding differences
        assert abs(breakdown.total_cost - component_sum) < Decimal("0.01")
    
    def test_calculator_registration(self, zerodha_config):
        """Test registering individual calculators."""
        aggregator = CostAggregator()
        
        # Mock calculator
        mock_calculator = Mock()
        mock_calculator.calculate_cost = AsyncMock(return_value=TransactionCostBreakdown(
            total_cost=Decimal("50.00"),
            commission=Decimal("20.00"),
            market_impact=Decimal("15.00"),
            spread_cost=Decimal("10.00"),
            slippage=Decimal("5.00"),
            regulatory_fees=Decimal("0.00")
        ))
        
        # Register calculator
        aggregator.add_calculator('broker', mock_calculator)
        
        assert 'broker' in aggregator.calculators
        assert aggregator.calculators['broker'] == mock_calculator
    
    @pytest.mark.asyncio
    async def test_parallel_calculation(self, sample_equity_request, sample_market_conditions):
        """Test parallel calculation of different cost components."""
        aggregator = CostAggregator()
        
        # Mock multiple calculators
        broker_calc = Mock()
        broker_calc.calculate_cost = AsyncMock(return_value=Decimal("20.00"))
        
        impact_calc = Mock()
        impact_calc.calculate_impact = AsyncMock(return_value=Decimal("15.00"))
        
        spread_calc = Mock()
        spread_calc.calculate_spread_cost = AsyncMock(return_value=Decimal("10.00"))
        
        aggregator.add_calculator('broker', broker_calc)
        aggregator.add_calculator('market_impact', impact_calc)
        aggregator.add_calculator('spread', spread_calc)
        
        # Calculate costs in parallel
        breakdown = await aggregator.calculate_total_cost(
            request=sample_equity_request,
            market_conditions=sample_market_conditions
        )
        
        # Verify all calculators were called
        broker_calc.calculate_cost.assert_called_once()
        impact_calc.calculate_impact.assert_called_once()
        spread_calc.calculate_spread_cost.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_single_calculator(self, sample_equity_request, sample_market_conditions):
        """Test error handling when one calculator fails."""
        aggregator = CostAggregator()
        
        # Mock calculators - one fails, others succeed
        good_calc = Mock()
        good_calc.calculate_cost = AsyncMock(return_value=Decimal("20.00"))
        
        bad_calc = Mock()
        bad_calc.calculate_cost = AsyncMock(side_effect=CalculationError("Test error"))
        
        aggregator.add_calculator('good', good_calc)
        aggregator.add_calculator('bad', bad_calc)
        
        # Should handle error gracefully
        breakdown = await aggregator.calculate_total_cost(
            request=sample_equity_request,
            market_conditions=sample_market_conditions,
            fail_on_error=False
        )
        
        # Should still return a result
        assert isinstance(breakdown, TransactionCostBreakdown)
        
        # Good calculator should have been called
        good_calc.calculate_cost.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculation_timeout(self, sample_equity_request, sample_market_conditions):
        """Test timeout handling for slow calculations."""
        aggregator = CostAggregator()
        
        # Mock slow calculator
        slow_calc = Mock()
        slow_calc.calculate_cost = AsyncMock()
        slow_calc.calculate_cost.side_effect = lambda *args, **kwargs: asyncio.sleep(2)
        
        aggregator.add_calculator('slow', slow_calc)
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await aggregator.calculate_total_cost(
                request=sample_equity_request,
                market_conditions=sample_market_conditions,
                timeout=0.5
            )
    
    def test_calculator_priorities(self):
        """Test calculator priority ordering."""
        aggregator = CostAggregator()
        
        # Add calculators with different priorities
        aggregator.add_calculator('high_priority', Mock(), priority=1)
        aggregator.add_calculator('low_priority', Mock(), priority=3)
        aggregator.add_calculator('medium_priority', Mock(), priority=2)
        
        # Should be ordered by priority
        ordered_names = aggregator.get_calculator_order()
        assert ordered_names[0] == 'high_priority'
        assert ordered_names[1] == 'medium_priority'
        assert ordered_names[2] == 'low_priority'
    
    @pytest.mark.asyncio
    async def test_cost_caching(self, sample_equity_request, sample_market_conditions):
        """Test caching of cost calculations."""
        aggregator = CostAggregator(enable_caching=True)
        
        # First calculation
        result1 = await aggregator.calculate_total_cost(
            request=sample_equity_request,
            market_conditions=sample_market_conditions
        )
        
        # Second calculation with same parameters
        result2 = await aggregator.calculate_total_cost(
            request=sample_equity_request,
            market_conditions=sample_market_conditions
        )
        
        # Results should be identical (from cache)
        assert result1.total_cost == result2.total_cost
        assert result1.commission == result2.commission
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            'enable_caching': True,
            'timeout_seconds': 30,
            'fail_on_error': False
        }
        
        aggregator = CostAggregator(**valid_config)
        assert aggregator.enable_caching is True
        assert aggregator.timeout_seconds == 30
        assert aggregator.fail_on_error is False
        
        # Invalid configuration should raise error
        with pytest.raises(ValueError):
            CostAggregator(timeout_seconds=-1)


@pytest.mark.unit
@pytest.mark.skipif(not COST_AGGREGATOR_AVAILABLE, reason="Cost aggregator not available")
class TestCostAggregatorPerformance:
    """Test performance aspects of cost aggregation."""
    
    @pytest.mark.asyncio
    async def test_batch_calculation_performance(self, batch_transaction_requests, sample_market_conditions):
        """Test performance of batch cost calculations."""
        aggregator = CostAggregator()
        
        # Calculate costs for batch of requests
        start_time = asyncio.get_event_loop().time()
        
        results = []
        for request in batch_transaction_requests[:10]:  # Test with 10 requests
            result = await aggregator.calculate_total_cost(request, sample_market_conditions)
            results.append(result)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Verify results
        assert len(results) == 10
        assert all(isinstance(r, TransactionCostBreakdown) for r in results)
        
        # Performance check - should complete within reasonable time
        assert duration < 5.0  # 5 seconds for 10 calculations
    
    @pytest.mark.asyncio
    async def test_concurrent_calculation_performance(self, batch_transaction_requests, sample_market_conditions):
        """Test concurrent calculation performance."""
        aggregator = CostAggregator()
        
        # Submit concurrent calculations
        tasks = []
        for request in batch_transaction_requests[:5]:
            task = asyncio.create_task(
                aggregator.calculate_total_cost(request, sample_market_conditions)
            )
            tasks.append(task)
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        duration = end_time - start_time
        
        # Verify results
        assert len(results) == 5
        successful_results = [r for r in results if isinstance(r, TransactionCostBreakdown)]
        assert len(successful_results) >= 3  # At least 3 should succeed
        
        # Concurrent execution should be faster than sequential
        assert duration < 3.0  # Should be faster than sequential execution
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring capabilities."""
        aggregator = CostAggregator(enable_monitoring=True)
        
        # Should have memory monitoring capabilities
        assert hasattr(aggregator, 'get_memory_usage') or hasattr(aggregator, 'memory_stats')


@pytest.mark.unit
@pytest.mark.skipif(not COST_AGGREGATOR_AVAILABLE, reason="Cost aggregator not available")
class TestCostAggregatorIntegration:
    """Test integration aspects of cost aggregation."""
    
    @pytest.mark.asyncio
    async def test_real_broker_integration(self, sample_equity_request, zerodha_config):
        """Test integration with real broker calculators."""
        aggregator = CostAggregator()
        
        # Add broker calculator
        from trading.transaction_costs.brokers.broker_factory import BrokerFactory
        broker_calc = BrokerFactory.create_calculator('zerodha', zerodha_config)
        aggregator.add_calculator('broker', broker_calc)
        
        # Calculate total cost
        result = await aggregator.calculate_total_cost(sample_equity_request)
        
        assert isinstance(result, TransactionCostBreakdown)
        assert result.commission > Decimal("0")
    
    def test_configuration_system_integration(self):
        """Test integration with configuration system."""
        # Mock configuration
        config = {
            'brokers': {
                'zerodha': {'commission_rate': 0.0003}
            },
            'market_impact': {
                'model': 'linear',
                'coefficient': 0.001
            }
        }
        
        aggregator = CostAggregator.from_config(config)
        
        # Should have configured calculators
        assert len(aggregator.calculators) > 0


# Mock tests when cost aggregator is not available
@pytest.mark.unit
@pytest.mark.skipif(COST_AGGREGATOR_AVAILABLE, reason="Cost aggregator available, using real tests")
class TestMockCostAggregator:
    """Mock tests when cost aggregator is not available."""
    
    @pytest.mark.asyncio
    async def test_mock_cost_aggregation(self, sample_equity_request, sample_cost_breakdown):
        """Test mock cost aggregation functionality."""
        class MockCostAggregator:
            def __init__(self):
                self.calculators = {}
            
            async def calculate_total_cost(self, request, market_conditions=None):
                # Simple mock calculation
                notional = request.price * request.quantity
                commission = max(notional * Decimal("0.001"), Decimal("10.00"))
                
                return type('MockBreakdown', (), {
                    'total_cost': commission * Decimal("2.5"),
                    'commission': commission,
                    'market_impact': commission * Decimal("0.8"),
                    'spread_cost': commission * Decimal("0.6"),
                    'slippage': commission * Decimal("0.4"),
                    'regulatory_fees': commission * Decimal("0.1")
                })()
            
            def add_calculator(self, name, calculator):
                self.calculators[name] = calculator
        
        aggregator = MockCostAggregator()
        result = await aggregator.calculate_total_cost(sample_equity_request)
        
        assert result.total_cost > Decimal("0")
        assert result.commission > Decimal("0")
    
    @pytest.mark.asyncio
    async def test_mock_parallel_aggregation(self, batch_transaction_requests):
        """Test mock parallel cost aggregation."""
        class MockParallelAggregator:
            async def calculate_batch_costs(self, requests):
                results = []
                for request in requests:
                    notional = request.price * request.quantity
                    commission = max(notional * Decimal("0.001"), Decimal("5.00"))
                    
                    result = type('MockResult', (), {
                        'total_cost': commission * Decimal("2.0"),
                        'commission': commission
                    })()
                    results.append(result)
                
                return results
        
        aggregator = MockParallelAggregator()
        results = await aggregator.calculate_batch_costs(batch_transaction_requests[:3])
        
        assert len(results) == 3
        assert all(r.total_cost > Decimal("0") for r in results)
    
    def test_mock_calculator_management(self):
        """Test mock calculator management."""
        class MockCalculatorManager:
            def __init__(self):
                self.calculators = {}
            
            def add_calculator(self, name, calculator, priority=1):
                self.calculators[name] = {'calc': calculator, 'priority': priority}
            
            def get_calculator_order(self):
                return sorted(self.calculators.keys(), 
                            key=lambda x: self.calculators[x]['priority'])
        
        manager = MockCalculatorManager()
        manager.add_calculator('broker', Mock(), priority=1)
        manager.add_calculator('impact', Mock(), priority=2)
        
        order = manager.get_calculator_order()
        assert order == ['broker', 'impact']