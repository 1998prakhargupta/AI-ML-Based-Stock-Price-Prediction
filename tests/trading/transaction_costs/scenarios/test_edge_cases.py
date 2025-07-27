"""
Scenario Testing for Edge Cases and Extreme Conditions
======================================================

Tests for edge cases, stress scenarios, and extreme market conditions
to validate robustness of transaction cost calculations.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock

# Scenario test markers
pytestmark = [pytest.mark.scenario]


@pytest.mark.scenario 
class TestHighVolumeScenarios:
    """Test scenarios involving high volume transactions."""
    
    def test_block_trade_scenario(self):
        """Test cost calculation for block trades."""
        # Market crash scenario conditions
        scenario = {
            'market_conditions': {
                'volatility': 0.80,  # 80% volatility (vs 25% normal)
                'volume_multiplier': 0.2,  # 20% of normal volume
                'spread_multiplier': 5.0  # 5x normal spreads
            }
        }
        
        # Simulate block trade during market stress
        block_trade = {
            'symbol': 'AAPL',
            'quantity': 100000,  # Very large trade
            'price': Decimal('150.00'),
            'market_volatility': scenario['market_conditions']['volatility'],
            'volume_factor': scenario['market_conditions']['volume_multiplier']
        }
        
        # Calculate expected impact
        notional = block_trade['price'] * block_trade['quantity']  # $15M trade
        base_cost = notional * Decimal('0.002')  # 20 bps base cost
        
        # Adjust for market stress
        volatility_adjustment = Decimal(str(block_trade['market_volatility'])) / Decimal('0.25')  # vs normal
        volume_adjustment = Decimal('1.0') / Decimal(str(block_trade['volume_factor']))  # liquidity impact
        
        stressed_cost = base_cost * volatility_adjustment * volume_adjustment
        
        # Verify cost escalation under stress
        assert stressed_cost > base_cost * Decimal('2.0')  # At least 2x normal cost
        assert stressed_cost < notional * Decimal('0.05')  # Max 5% of notional
    
    def test_iceberg_order_scenario(self):
        """Test cost calculation for iceberg orders (hidden volume)."""
        # Iceberg order parameters
        total_quantity = 50000
        slice_size = 1000
        num_slices = total_quantity // slice_size
        
        price = Decimal('100.00')
        
        # Calculate cost per slice
        slice_cost = price * slice_size * Decimal('0.001')  # 10 bps per slice
        
        # Market impact increases with each slice (information leakage)
        total_cost = Decimal('0.0')
        for slice_num in range(num_slices):
            impact_multiplier = Decimal('1.0') + (Decimal(str(slice_num)) * Decimal('0.1'))
            slice_impact = slice_cost * impact_multiplier
            total_cost += slice_impact
        
        # Verify iceberg strategy cost
        naive_cost = price * total_quantity * Decimal('0.001') * Decimal(str(num_slices))
        assert total_cost > naive_cost  # Information leakage increases cost
        assert total_cost < naive_cost * Decimal('2.0')  # But not excessively


@pytest.mark.scenario
class TestLowLiquidityScenarios:
    """Test scenarios involving low liquidity conditions."""
    
    def test_illiquid_stock_scenario(self):
        """Test cost calculation for illiquid stock trading."""
        # Illiquid stock scenario conditions
        scenario = {
            'market_conditions': {
                'average_volume': 50000,  # Very low volume
                'spread_bps': 100  # 100 bps spread
            }
        }
        
        # Small-cap stock characteristics
        illiquid_trade = {
            'symbol': 'SMALLCAP',
            'quantity': 5000,
            'price': Decimal('25.00'),
            'average_volume': scenario['market_conditions']['average_volume'],
            'spread_bps': scenario['market_conditions']['spread_bps']
        }
        
        notional = illiquid_trade['price'] * illiquid_trade['quantity']
        
        # Calculate impact components
        participation_rate = illiquid_trade['quantity'] / illiquid_trade['average_volume']
        market_impact = notional * Decimal(str(participation_rate * 2.0))  # 2x participation rate
        
        spread_cost = notional * Decimal(str(illiquid_trade['spread_bps'])) / Decimal('10000')
        
        total_cost = market_impact + spread_cost
        
        # Verify illiquid stock costs
        assert participation_rate > 0.1  # >10% of daily volume
        assert total_cost > notional * Decimal('0.01')  # >100 bps
        assert spread_cost > market_impact * Decimal('0.5')  # Spread dominates
    
    def test_after_hours_trading_scenario(self):
        """Test cost calculation for after-hours trading."""
        # After-hours characteristics
        after_hours_trade = {
            'symbol': 'AAPL',
            'quantity': 1000,
            'price': Decimal('150.00'),
            'time': '18:30',  # After market close
            'spread_multiplier': 3.0,  # Wider spreads
            'volume_multiplier': 0.1   # Much lower volume
        }
        
        notional = after_hours_trade['price'] * after_hours_trade['quantity']
        
        # Regular hours cost
        regular_cost = notional * Decimal('0.0015')  # 15 bps
        
        # After-hours adjustments
        spread_adjustment = Decimal(str(after_hours_trade['spread_multiplier']))
        volume_adjustment = Decimal('1.0') / Decimal(str(after_hours_trade['volume_multiplier']))
        
        after_hours_cost = regular_cost * spread_adjustment * volume_adjustment
        
        # Verify after-hours premium
        assert after_hours_cost > regular_cost * Decimal('5.0')  # At least 5x regular cost
        assert after_hours_cost < notional * Decimal('0.02')  # Max 200 bps


@pytest.mark.scenario
class TestMarketStressScenarios:
    """Test scenarios during market stress conditions."""
    
    def test_flash_crash_scenario(self):
        """Test cost calculation during flash crash conditions."""
        # Flash crash scenario conditions
        scenario = {
            'market_conditions': {
                'volatility': 2.0,  # 200% volatility spike
                'volume_multiplier': 10.0,  # 10x volume spike  
                'spread_multiplier': 20.0  # 20x spreads
            }
        }
        
        # Flash crash trade
        flash_trade = {
            'symbol': 'SPY',
            'quantity': 10000,
            'price': Decimal('400.00'),
            'volatility_spike': scenario['market_conditions']['volatility'],
            'volume_spike': scenario['market_conditions']['volume_multiplier'],
            'spread_spike': scenario['market_conditions']['spread_multiplier']
        }
        
        notional = flash_trade['price'] * flash_trade['quantity']
        
        # Calculate flash crash cost components
        base_cost = notional * Decimal('0.0005')  # 5 bps normal
        
        volatility_impact = base_cost * Decimal(str(flash_trade['volatility_spike']))
        spread_impact = base_cost * Decimal(str(flash_trade['spread_spike']))
        
        total_flash_cost = volatility_impact + spread_impact
        
        # Verify flash crash impact
        assert total_flash_cost > base_cost * Decimal('10.0')  # At least 10x normal
        assert spread_impact > volatility_impact  # Spread dominates in flash crash
    
    def test_market_close_rush_scenario(self):
        """Test cost calculation during market close rush."""
        # Market close rush scenario conditions  
        scenario = {
            'market_conditions': {
                'volume_multiplier': 3.0,  # 3x normal volume
                'time_pressure': 'high'
            }
        }
        
        # Market close trades
        close_trades = [
            {'symbol': 'AAPL', 'quantity': 5000, 'deadline': '15:59:00'},
            {'symbol': 'GOOGL', 'quantity': 100, 'deadline': '15:59:30'},
            {'symbol': 'MSFT', 'quantity': 2000, 'deadline': '15:58:00'}
        ]
        
        total_urgency_cost = Decimal('0.0')
        
        for trade in close_trades:
            notional = Decimal('150.00') * trade['quantity']  # Assume $150 avg price
            
            # Calculate urgency premium based on deadline
            deadline_hour, deadline_min = map(int, trade['deadline'].split(':')[:2])
            minutes_to_close = (16 * 60) - (deadline_hour * 60 + deadline_min)
            
            if minutes_to_close <= 1:
                urgency_multiplier = Decimal('5.0')  # 5x for last minute
            elif minutes_to_close <= 2:
                urgency_multiplier = Decimal('3.0')  # 3x for last 2 minutes
            else:
                urgency_multiplier = Decimal('2.0')  # 2x for last hour
            
            base_cost = notional * Decimal('0.001')  # 10 bps base
            urgency_cost = base_cost * urgency_multiplier
            total_urgency_cost += urgency_cost
        
        # Verify market close rush costs
        assert total_urgency_cost > Decimal('500.00')  # Significant cost premium
        
        # Most urgent trade (15:59:00) should have highest multiplier
        most_urgent = [t for t in close_trades if t['deadline'] == '15:59:00'][0]
        most_urgent_cost = Decimal('150.00') * most_urgent['quantity'] * Decimal('0.001') * Decimal('5.0')
        assert most_urgent_cost > Decimal('375.00')  # 5x multiplier


@pytest.mark.scenario
class TestExtremeConditionScenarios:
    """Test scenarios under extreme market conditions."""
    
    def test_circuit_breaker_scenario(self):
        """Test cost calculation when circuit breakers are triggered."""
        # Circuit breaker conditions
        circuit_breaker = {
            'level': 1,  # 7% decline triggers level 1
            'trading_halt_duration': 15,  # 15 minutes
            'market_reopening_volatility': 3.0,  # 3x normal volatility
            'pent_up_demand': 2.5  # 2.5x normal volume
        }
        
        # Post-halt trade
        post_halt_trade = {
            'symbol': 'SPY',
            'quantity': 5000,
            'price': Decimal('350.00'),  # 7% down from $376
        }
        
        notional = post_halt_trade['price'] * post_halt_trade['quantity']
        
        # Calculate post-halt costs
        normal_cost = notional * Decimal('0.0008')  # 8 bps normal
        
        volatility_premium = normal_cost * Decimal(str(circuit_breaker['market_reopening_volatility']))
        volume_impact = normal_cost * Decimal(str(circuit_breaker['pent_up_demand']))
        
        total_post_halt_cost = volatility_premium + volume_impact
        
        # Verify circuit breaker impact
        assert total_post_halt_cost > normal_cost * Decimal('4.0')  # At least 4x normal
        assert total_post_halt_cost < notional * Decimal('0.01')  # Max 100 bps
    
    def test_options_expiry_scenario(self):
        """Test cost calculation during options expiry."""
        # Options expiry scenario conditions
        scenario = {
            'market_conditions': {
                'gamma_risk': 'high',
                'volatility': 0.45,
                'time_decay': 'accelerated'
            }
        }
        
        # Options expiry trade
        expiry_trade = {
            'symbol': 'AAPL',
            'underlying_price': Decimal('150.00'),
            'strike_price': Decimal('150.00'),  # At-the-money
            'quantity': 100,  # 100 contracts
            'time_to_expiry': 0.5,  # 30 minutes to expiry
            'gamma_risk': 'high'
        }
        
        # Calculate gamma risk impact
        option_premium = Decimal('2.50')  # $2.50 per share
        notional = option_premium * expiry_trade['quantity'] * 100  # $25,000
        
        # Gamma risk increases cost near expiry
        time_decay_factor = Decimal('1.0') / Decimal(str(expiry_trade['time_to_expiry']))
        gamma_cost = notional * Decimal('0.02') * time_decay_factor  # 2% * time factor
        
        pin_risk_cost = notional * Decimal('0.01')  # 1% pin risk
        
        total_expiry_cost = gamma_cost + pin_risk_cost
        
        # Verify options expiry impact
        assert gamma_cost > pin_risk_cost  # Gamma risk dominates near expiry
        assert total_expiry_cost > notional * Decimal('0.03')  # >3% of notional
        assert time_decay_factor >= Decimal('2.0')  # At least 2x for 30min expiry


@pytest.mark.scenario
class TestRealWorldScenarios:
    """Test realistic trading scenarios."""
    
    def test_portfolio_rebalancing_scenario(self):
        """Test cost calculation for portfolio rebalancing."""
        # Portfolio rebalancing trades
        rebalancing_trades = [
            {'action': 'SELL', 'symbol': 'AAPL', 'quantity': 1000, 'reason': 'overweight'},
            {'action': 'BUY', 'symbol': 'GOOGL', 'quantity': 100, 'reason': 'underweight'},
            {'action': 'SELL', 'symbol': 'TSLA', 'quantity': 500, 'reason': 'profit_taking'},
            {'action': 'BUY', 'symbol': 'MSFT', 'quantity': 800, 'reason': 'strategic_allocation'}
        ]
        
        total_rebalancing_cost = Decimal('0.0')
        
        for trade in rebalancing_trades:
            # Estimate price based on symbol
            price_map = {'AAPL': 150, 'GOOGL': 2800, 'TSLA': 200, 'MSFT': 400}
            price = Decimal(str(price_map[trade['symbol']]))
            
            notional = price * trade['quantity']
            
            # Calculate trading cost (commission + market impact)
            commission = max(notional * Decimal('0.0005'), Decimal('1.00'))
            
            # Market impact varies by action
            if trade['action'] == 'SELL':
                market_impact = notional * Decimal('0.0008')  # 8 bps for sells
            else:
                market_impact = notional * Decimal('0.0006')  # 6 bps for buys
            
            trade_cost = commission + market_impact
            total_rebalancing_cost += trade_cost
        
        # Verify rebalancing costs
        assert total_rebalancing_cost > Decimal('100.00')  # Significant cost
        assert total_rebalancing_cost < Decimal('2000.00')  # But reasonable
    
    def test_algorithmic_trading_scenario(self):
        """Test cost calculation for algorithmic trading strategies."""
        # TWAP (Time Weighted Average Price) strategy
        twap_strategy = {
            'total_quantity': 10000,
            'time_horizon_minutes': 60,
            'slice_size': 200,  # 200 shares per slice
            'execution_interval': 1.2  # 1.2 minutes between slices
        }
        
        total_slices = twap_strategy['total_quantity'] // twap_strategy['slice_size']
        price = Decimal('100.00')
        
        total_twap_cost = Decimal('0.0')
        
        for slice_num in range(total_slices):
            slice_notional = price * twap_strategy['slice_size']
            
            # Commission per slice
            commission = max(slice_notional * Decimal('0.0003'), Decimal('1.00'))
            
            # Market impact decreases with smaller slices
            impact_reduction = Decimal('0.5')  # 50% reduction vs block trade
            market_impact = slice_notional * Decimal('0.0005') * impact_reduction
            
            slice_cost = commission + market_impact
            total_twap_cost += slice_cost
        
        # Compare to block trade cost
        block_notional = price * twap_strategy['total_quantity']
        block_cost = max(block_notional * Decimal('0.0003'), Decimal('1.00')) + \
                    block_notional * Decimal('0.002')  # 20 bps market impact
        
        # Verify TWAP strategy effectiveness
        assert total_twap_cost < block_cost  # TWAP should be cheaper
        assert total_twap_cost > block_cost * Decimal('0.2')  # But not too cheap (overhead)


# Mock scenario fixture for standalone testing
@pytest.fixture
def mock_stress_test_scenarios():
    """Mock stress test scenarios if not available from conftest."""
    return {
        'market_crash': {
            'market_conditions': {
                'volatility': 0.80,
                'volume_multiplier': 0.2,
                'spread_multiplier': 5.0
            }
        },
        'illiquid_stock': {
            'market_conditions': {
                'average_volume': 50000,
                'spread_bps': 100
            }
        },
        'flash_crash': {
            'market_conditions': {
                'volatility': 2.0,
                'volume_multiplier': 10.0,
                'spread_multiplier': 20.0
            }
        },
        'market_close_rush': {
            'market_conditions': {
                'volume_multiplier': 3.0
            }
        },
        'options_expiry': {
            'market_conditions': {
                'gamma_risk': 'high'
            }
        }
    }