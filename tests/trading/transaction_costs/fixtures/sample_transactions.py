"""
Sample Transaction Data Fixtures
===============================

Fixtures providing sample transaction data, market conditions,
and broker responses for testing transaction cost calculations.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random


def generate_sample_transactions() -> List[Dict[str, Any]]:
    """Generate sample transaction data for testing."""
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'ORCL']
    
    transactions = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(100):
        symbol = random.choice(symbols)
        
        # Generate realistic price based on symbol
        price_map = {
            'AAPL': 150, 'GOOGL': 2800, 'MSFT': 400, 'TSLA': 200, 'AMZN': 140,
            'META': 300, 'NVDA': 450, 'NFLX': 400, 'CRM': 200, 'ORCL': 100
        }
        
        base_price = price_map.get(symbol, 100)
        price = Decimal(str(base_price + random.uniform(-10, 10)))
        
        # Generate realistic quantities
        quantity = random.choice([50, 100, 200, 500, 1000, 2000])
        
        # Generate transaction type
        transaction_type = random.choice(['BUY', 'SELL'])
        
        # Generate timestamp
        timestamp = base_time + timedelta(minutes=i * 15)
        
        transaction = {
            'id': f'TXN_{i:04d}',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'transaction_type': transaction_type,
            'instrument_type': 'EQUITY',
            'order_type': 'MARKET',
            'timestamp': timestamp,
            'notional_value': price * quantity
        }
        
        transactions.append(transaction)
    
    return transactions


def generate_market_data_samples() -> Dict[str, Dict[str, Any]]:
    """Generate sample market data for testing."""
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'ORCL']
    
    market_data = {}
    
    for symbol in symbols:
        # Base prices
        price_map = {
            'AAPL': 150.50, 'GOOGL': 2875.25, 'MSFT': 415.80, 'TSLA': 205.30, 'AMZN': 142.15,
            'META': 315.60, 'NVDA': 465.90, 'NFLX': 425.40, 'CRM': 195.75, 'ORCL': 105.25
        }
        
        mid_price = price_map.get(symbol, 100.0)
        
        # Generate bid-ask spread (0.01% to 0.1% of price)
        spread_pct = random.uniform(0.0001, 0.001)
        spread = mid_price * spread_pct
        
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2
        
        # Generate volume (realistic ranges)
        volume_map = {
            'AAPL': 75000000, 'GOOGL': 1200000, 'MSFT': 30000000, 'TSLA': 85000000, 'AMZN': 45000000,
            'META': 20000000, 'NVDA': 50000000, 'NFLX': 8000000, 'CRM': 3000000, 'ORCL': 15000000
        }
        
        avg_volume = volume_map.get(symbol, 1000000)
        volume = int(avg_volume * random.uniform(0.5, 2.0))  # ±50% variation
        
        # Generate volatility (annualized)
        volatility = random.uniform(0.15, 0.45)  # 15% to 45%
        
        market_data[symbol] = {
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'last': mid_price,
            'volume': volume,
            'volatility': volatility,
            'spread': spread,
            'spread_bps': (spread / mid_price) * 10000,
            'timestamp': datetime.now()
        }
    
    return market_data


def generate_broker_responses() -> Dict[str, Dict[str, Any]]:
    """Generate sample broker API responses for testing."""
    
    brokers = ['zerodha', 'icici', 'upstox', 'angel', 'fyers']
    
    broker_responses = {}
    
    for broker in brokers:
        # Commission structures vary by broker
        commission_structures = {
            'zerodha': {
                'equity_delivery': {'rate': 0.0, 'minimum': 0.0, 'maximum': 20.0},
                'equity_intraday': {'rate': 0.0003, 'minimum': 0.0, 'maximum': 20.0},
                'equity_f&o': {'flat': 20.0},
                'currency': {'rate': 0.0003, 'minimum': 0.0, 'maximum': 20.0}
            },
            'icici': {
                'equity_delivery': {'rate': 0.0005, 'minimum': 20.0},
                'equity_intraday': {'rate': 0.0005, 'minimum': 20.0},
                'equity_f&o': {'rate': 0.0005, 'minimum': 20.0},
                'currency': {'rate': 0.0005, 'minimum': 10.0}
            },
            'upstox': {
                'equity_delivery': {'rate': 0.0, 'minimum': 0.0},
                'equity_intraday': {'rate': 0.0003, 'minimum': 0.0, 'maximum': 20.0},
                'equity_f&o': {'flat': 20.0}
            },
            'angel': {
                'equity_delivery': {'rate': 0.0, 'minimum': 0.0},
                'equity_intraday': {'rate': 0.00025, 'minimum': 0.0, 'maximum': 20.0},
                'equity_f&o': {'flat': 20.0}
            },
            'fyers': {
                'equity_delivery': {'rate': 0.0, 'minimum': 0.0},
                'equity_intraday': {'rate': 0.0003, 'minimum': 0.0, 'maximum': 20.0},
                'equity_f&o': {'flat': 20.0}
            }
        }
        
        # Additional fees
        additional_fees = {
            'stt': 0.001,  # Securities Transaction Tax
            'exchange_charges': 0.0000345,
            'gst': 0.18,  # 18% GST on brokerage
            'stamp_duty': 0.00003,
            'sebi_charges': 0.0000001
        }
        
        broker_responses[broker] = {
            'broker_name': broker,
            'commission_structure': commission_structures.get(broker, commission_structures['zerodha']),
            'additional_fees': additional_fees,
            'api_endpoints': {
                'calculate_charges': f'https://api.{broker}.com/charges',
                'get_margins': f'https://api.{broker}.com/margins',
                'place_order': f'https://api.{broker}.com/orders'
            },
            'supported_exchanges': ['NSE', 'BSE', 'MCX', 'NCDEX'],
            'supported_instruments': ['EQUITY', 'FUTURES', 'OPTIONS', 'CURRENCY'],
            'rate_limits': {
                'orders_per_second': 10,
                'api_calls_per_minute': 3000
            }
        }
    
    return broker_responses


def generate_historical_cost_data() -> List[Dict[str, Any]]:
    """Generate historical cost data for backtesting and validation."""
    
    cost_data = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(1000):  # Generate 1000 historical records
        date = base_date + timedelta(days=i * 0.365)  # ~1 record per day
        
        # Random transaction
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        symbol = random.choice(symbols)
        
        quantity = random.choice([100, 200, 500, 1000])
        price = Decimal(str(random.uniform(50, 500)))
        notional = price * quantity
        
        # Calculate historical costs
        commission = max(notional * Decimal('0.0005'), Decimal('20.0'))
        market_impact = notional * Decimal(str(random.uniform(0.0002, 0.002)))
        spread_cost = notional * Decimal(str(random.uniform(0.0001, 0.001)))
        slippage = notional * Decimal(str(random.uniform(0.0, 0.0005)))
        regulatory_fees = notional * Decimal('0.0001')
        
        total_cost = commission + market_impact + spread_cost + slippage + regulatory_fees
        
        cost_record = {
            'date': date,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'notional_value': notional,
            'commission': commission,
            'market_impact': market_impact,
            'spread_cost': spread_cost,
            'slippage': slippage,
            'regulatory_fees': regulatory_fees,
            'total_cost': total_cost,
            'cost_bps': float((total_cost / notional) * 10000),  # Basis points
            'broker': random.choice(['zerodha', 'icici', 'upstox']),
            'market_conditions': {
                'volatility': random.uniform(0.15, 0.50),
                'volume': random.randint(100000, 10000000),
                'spread_bps': random.uniform(1, 20)
            }
        }
        
        cost_data.append(cost_record)
    
    return cost_data


def generate_stress_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """Generate stress test scenarios for edge case testing."""
    
    scenarios = {
        'market_crash': {
            'description': 'Market crash scenario with high volatility and low liquidity',
            'market_conditions': {
                'volatility': 0.80,  # 80% volatility
                'volume_multiplier': 0.2,  # 20% of normal volume
                'spread_multiplier': 5.0,  # 5x normal spreads
                'correlation_breakdown': True
            },
            'transactions': [
                {'symbol': 'AAPL', 'quantity': 10000, 'urgency': 'high'},
                {'symbol': 'TSLA', 'quantity': 5000, 'urgency': 'medium'},
                {'symbol': 'NFLX', 'quantity': 2000, 'urgency': 'low'}
            ]
        },
        
        'flash_crash': {
            'description': 'Flash crash with extreme price movements',
            'market_conditions': {
                'volatility': 2.0,  # 200% volatility spike
                'volume_multiplier': 10.0,  # 10x volume spike
                'spread_multiplier': 20.0,  # 20x spreads
                'price_gap': 0.1  # 10% price gap
            },
            'transactions': [
                {'symbol': 'SPY', 'quantity': 50000, 'urgency': 'immediate'},
                {'symbol': 'QQQ', 'quantity': 25000, 'urgency': 'immediate'}
            ]
        },
        
        'illiquid_stock': {
            'description': 'Trading in illiquid small-cap stock',
            'market_conditions': {
                'volatility': 0.60,
                'average_volume': 50000,  # Very low volume
                'spread_bps': 100,  # 100 bps spread
                'market_depth': 'shallow'
            },
            'transactions': [
                {'symbol': 'SMALLCAP', 'quantity': 10000, 'urgency': 'medium'}
            ]
        },
        
        'high_frequency_trading': {
            'description': 'High frequency trading scenario',
            'market_conditions': {
                'volatility': 0.25,
                'volume_multiplier': 1.0,
                'trade_frequency': 'milliseconds',
                'latency_sensitive': True
            },
            'transactions': [
                {'symbol': 'AAPL', 'quantity': 100, 'frequency': 1000},  # 1000 trades
                {'symbol': 'MSFT', 'quantity': 200, 'frequency': 500}
            ]
        },
        
        'market_close_rush': {
            'description': 'End-of-day trading rush',
            'market_conditions': {
                'volatility': 0.35,
                'volume_multiplier': 3.0,  # 3x normal volume
                'time_pressure': 'high',
                'market_timing': 'closing_auction'
            },
            'transactions': [
                {'symbol': 'AAPL', 'quantity': 5000, 'deadline': '15:59:00'},
                {'symbol': 'GOOGL', 'quantity': 100, 'deadline': '15:59:30'},
                {'symbol': 'MSFT', 'quantity': 2000, 'deadline': '15:58:00'}
            ]
        },
        
        'currency_crisis': {
            'description': 'Currency trading during crisis',
            'market_conditions': {
                'volatility': 1.5,  # 150% volatility
                'correlation_spike': True,
                'flight_to_quality': True,
                'central_bank_intervention': 'possible'
            },
            'transactions': [
                {'symbol': 'USDINR', 'quantity': 1000000, 'urgency': 'high'},
                {'symbol': 'EURUSD', 'quantity': 500000, 'urgency': 'medium'}
            ]
        },
        
        'options_expiry': {
            'description': 'Options expiry with high gamma risk',
            'market_conditions': {
                'volatility': 0.45,
                'gamma_risk': 'high',
                'pin_risk': 'moderate',
                'time_decay': 'accelerated'
            },
            'transactions': [
                {'symbol': 'AAPL240315C00150000', 'quantity': 100, 'expiry': 'today'},
                {'symbol': 'SPY240315P00420000', 'quantity': 50, 'expiry': 'today'}
            ]
        }
    }
    
    return scenarios


def generate_performance_test_data() -> Dict[str, Any]:
    """Generate data for performance testing."""
    
    return {
        'latency_test_requests': [
            {
                'id': f'PERF_{i:05d}',
                'symbol': f'STOCK{i % 100:03d}',
                'quantity': 100,
                'price': Decimal(str(100 + i % 50)),
                'transaction_type': 'BUY' if i % 2 == 0 else 'SELL'
            }
            for i in range(10000)
        ],
        
        'throughput_test_batches': [
            [
                {
                    'symbol': f'BATCH{batch}_{i:03d}',
                    'quantity': random.randint(50, 500),
                    'price': Decimal(str(random.uniform(50, 200)))
                }
                for i in range(100)
            ]
            for batch in range(50)
        ],
        
        'memory_test_large_transactions': [
            {
                'symbol': 'LARGE_TXN',
                'quantity': 1000000,  # 1M shares
                'price': Decimal('100.00'),
                'metadata': {
                    'large_order': True,
                    'split_required': True,
                    'market_impact_high': True
                }
            }
            for _ in range(100)
        ],
        
        'concurrent_access_patterns': {
            'user_sessions': [
                {
                    'user_id': f'USER_{i:04d}',
                    'session_duration': random.randint(300, 3600),  # 5min to 1hr
                    'transaction_frequency': random.uniform(0.1, 2.0),  # per minute
                    'preferred_symbols': random.sample(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], 3)
                }
                for i in range(1000)
            ]
        }
    }


# Export main functions
__all__ = [
    'generate_sample_transactions',
    'generate_market_data_samples', 
    'generate_broker_responses',
    'generate_historical_cost_data',
    'generate_stress_test_scenarios',
    'generate_performance_test_data'
]