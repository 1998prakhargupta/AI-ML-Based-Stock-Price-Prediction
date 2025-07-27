"""
Market Data Sample Fixtures
===========================

Fixtures providing realistic market data samples for testing
transaction cost calculations under various market conditions.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json


class MarketDataGenerator:
    """Generate realistic market data for testing."""
    
    def __init__(self):
        self.symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 
            'CRM', 'ORCL', 'IBM', 'UBER', 'LYFT', 'ZOOM', 'SHOP'
        ]
        
        # Base prices and typical characteristics
        self.symbol_characteristics = {
            'AAPL': {'base_price': 150.0, 'volatility': 0.25, 'avg_volume': 75000000, 'typical_spread_bps': 2},
            'GOOGL': {'base_price': 2800.0, 'volatility': 0.30, 'avg_volume': 1200000, 'typical_spread_bps': 3},
            'MSFT': {'base_price': 415.0, 'volatility': 0.22, 'avg_volume': 30000000, 'typical_spread_bps': 2},
            'TSLA': {'base_price': 205.0, 'volatility': 0.55, 'avg_volume': 85000000, 'typical_spread_bps': 4},
            'AMZN': {'base_price': 142.0, 'volatility': 0.35, 'avg_volume': 45000000, 'typical_spread_bps': 3},
            'META': {'base_price': 315.0, 'volatility': 0.40, 'avg_volume': 20000000, 'typical_spread_bps': 3},
            'NVDA': {'base_price': 465.0, 'volatility': 0.50, 'avg_volume': 50000000, 'typical_spread_bps': 4},
            'NFLX': {'base_price': 425.0, 'volatility': 0.45, 'avg_volume': 8000000, 'typical_spread_bps': 5},
            'CRM': {'base_price': 195.0, 'volatility': 0.35, 'avg_volume': 3000000, 'typical_spread_bps': 6},
            'ORCL': {'base_price': 105.0, 'volatility': 0.20, 'avg_volume': 15000000, 'typical_spread_bps': 4},
        }
    
    def generate_real_time_quotes(self, timestamp: datetime = None) -> Dict[str, Dict[str, Any]]:
        """Generate real-time quotes for all symbols."""
        if timestamp is None:
            timestamp = datetime.now()
        
        quotes = {}
        
        for symbol in self.symbols:
            characteristics = self.symbol_characteristics.get(symbol, {
                'base_price': 100.0, 'volatility': 0.30, 'avg_volume': 1000000, 'typical_spread_bps': 5
            })
            
            # Generate current price with some random movement
            import random
            price_movement = random.gauss(0, characteristics['volatility'] / 100)
            current_price = characteristics['base_price'] * (1 + price_movement)
            
            # Generate bid-ask spread
            spread_bps = characteristics['typical_spread_bps'] * random.uniform(0.5, 2.0)
            spread = current_price * spread_bps / 10000
            
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            # Generate volume
            volume_factor = random.uniform(0.3, 3.0)  # 30% to 300% of average
            current_volume = int(characteristics['avg_volume'] * volume_factor)
            
            quotes[symbol] = {
                'symbol': symbol,
                'timestamp': timestamp,
                'bid': round(bid, 2),
                'ask': round(ask, 2),
                'last': round(current_price, 2),
                'volume': current_volume,
                'high': round(current_price * 1.02, 2),
                'low': round(current_price * 0.98, 2),
                'open': round(current_price * random.uniform(0.99, 1.01), 2),
                'spread': round(spread, 4),
                'spread_bps': round(spread_bps, 2),
                'volatility': characteristics['volatility'],
                'market_cap': current_price * random.randint(1000000, 10000000),  # Shares outstanding
                'sector': self._get_sector(symbol),
                'exchange': 'NASDAQ' if symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'] else 'NYSE'
            }
        
        return quotes
    
    def generate_historical_data(self, symbol: str, days: int = 252) -> List[Dict[str, Any]]:
        """Generate historical market data for a symbol."""
        if symbol not in self.symbol_characteristics:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        characteristics = self.symbol_characteristics[symbol]
        historical_data = []
        
        base_date = datetime.now() - timedelta(days=days)
        current_price = characteristics['base_price']
        
        import random
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Generate daily price movement
            daily_volatility = characteristics['volatility'] / (252 ** 0.5)  # Annualized to daily
            price_change = random.gauss(0.001, daily_volatility)  # Small positive drift
            current_price *= (1 + price_change)
            
            # Generate OHLC
            high = current_price * random.uniform(1.001, 1.02)
            low = current_price * random.uniform(0.98, 0.999)
            open_price = current_price * random.uniform(0.995, 1.005)
            
            # Generate volume
            volume_factor = random.uniform(0.5, 2.0)
            volume = int(characteristics['avg_volume'] * volume_factor)
            
            # Generate spread
            spread_bps = characteristics['typical_spread_bps'] * random.uniform(0.8, 1.5)
            spread = current_price * spread_bps / 10000
            
            historical_data.append({
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'spread': round(spread, 4),
                'spread_bps': round(spread_bps, 2),
                'volatility': characteristics['volatility']
            })
        
        return historical_data
    
    def generate_intraday_data(self, symbol: str, hours: int = 6) -> List[Dict[str, Any]]:
        """Generate intraday market data (minute-by-minute)."""
        if symbol not in self.symbol_characteristics:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        characteristics = self.symbol_characteristics[symbol]
        intraday_data = []
        
        # Start at market open (9:30 AM)
        base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        current_price = characteristics['base_price']
        
        import random
        
        for i in range(hours * 60):  # Minute-by-minute data
            timestamp = base_time + timedelta(minutes=i)
            
            # Generate minute-level price movement
            minute_volatility = characteristics['volatility'] / (252 * 24 * 60) ** 0.5
            price_change = random.gauss(0, minute_volatility)
            current_price *= (1 + price_change)
            
            # Generate bid-ask spread (tighter during active hours)
            time_factor = 1.0
            if 9.5 <= timestamp.hour + timestamp.minute/60 <= 10.0:  # Opening hour
                time_factor = 2.0
            elif 15.5 <= timestamp.hour + timestamp.minute/60 <= 16.0:  # Closing hour
                time_factor = 1.5
            
            spread_bps = characteristics['typical_spread_bps'] * time_factor * random.uniform(0.7, 1.3)
            spread = current_price * spread_bps / 10000
            
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            # Generate volume (higher during opening/closing)
            volume_base = characteristics['avg_volume'] / (6 * 60)  # Average per minute
            if 9.5 <= timestamp.hour + timestamp.minute/60 <= 10.0:
                volume_factor = random.uniform(2.0, 5.0)
            elif 15.5 <= timestamp.hour + timestamp.minute/60 <= 16.0:
                volume_factor = random.uniform(1.5, 3.0)
            else:
                volume_factor = random.uniform(0.5, 1.5)
            
            volume = int(volume_base * volume_factor)
            
            intraday_data.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'price': round(current_price, 2),
                'bid': round(bid, 2),
                'ask': round(ask, 2),
                'volume': volume,
                'spread': round(spread, 4),
                'spread_bps': round(spread_bps, 2)
            })
        
        return intraday_data
    
    def generate_options_chain(self, underlying_symbol: str, expiry_date: str = None) -> Dict[str, Any]:
        """Generate options chain data for testing."""
        if underlying_symbol not in self.symbol_characteristics:
            raise ValueError(f"Unknown underlying symbol: {underlying_symbol}")
        
        if expiry_date is None:
            # Default to next monthly expiry (3rd Friday)
            expiry_date = self._get_next_expiry()
        
        characteristics = self.symbol_characteristics[underlying_symbol]
        underlying_price = characteristics['base_price']
        
        # Generate strike prices around current price
        strikes = []
        for i in range(-10, 11):  # 21 strikes
            if underlying_price < 50:
                strike_interval = 2.5
            elif underlying_price < 200:
                strike_interval = 5.0
            else:
                strike_interval = 10.0
            
            strike = underlying_price + (i * strike_interval)
            if strike > 0:
                strikes.append(strike)
        
        options_chain = {
            'underlying_symbol': underlying_symbol,
            'underlying_price': underlying_price,
            'expiry_date': expiry_date,
            'calls': [],
            'puts': []
        }
        
        import random
        import math
        
        # Calculate time to expiry in years
        expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        time_to_expiry = (expiry_dt - datetime.now()).days / 365.0
        
        # Risk-free rate and volatility
        risk_free_rate = 0.05  # 5%
        volatility = characteristics['volatility']
        
        for strike in strikes:
            # Simplified Black-Scholes for option pricing
            d1 = (math.log(underlying_price / strike) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * math.sqrt(time_to_expiry))
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Cumulative normal distribution (approximation)
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            # Call price
            call_price = (underlying_price * norm_cdf(d1) - 
                         strike * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2))
            
            # Put price (put-call parity)
            put_price = (call_price - underlying_price + 
                        strike * math.exp(-risk_free_rate * time_to_expiry))
            
            # Add bid-ask spreads
            call_spread = max(call_price * 0.02, 0.05)  # 2% or 5 cents minimum
            put_spread = max(put_price * 0.02, 0.05)
            
            call_data = {
                'strike': strike,
                'bid': max(0, round(call_price - call_spread/2, 2)),
                'ask': round(call_price + call_spread/2, 2),
                'last': round(call_price, 2),
                'volume': random.randint(0, 1000),
                'open_interest': random.randint(100, 10000),
                'implied_volatility': round(volatility + random.uniform(-0.05, 0.05), 3),
                'delta': round(norm_cdf(d1), 3),
                'gamma': round(norm_cdf(d1) / (underlying_price * volatility * math.sqrt(time_to_expiry)), 4),
                'theta': round(-call_price * 0.01, 2),  # Simplified
                'vega': round(underlying_price * norm_cdf(d1) * math.sqrt(time_to_expiry) * 0.01, 2)
            }
            
            put_data = {
                'strike': strike,
                'bid': max(0, round(put_price - put_spread/2, 2)),
                'ask': round(put_price + put_spread/2, 2),
                'last': round(put_price, 2),
                'volume': random.randint(0, 1000),
                'open_interest': random.randint(100, 10000),
                'implied_volatility': round(volatility + random.uniform(-0.05, 0.05), 3),
                'delta': round(norm_cdf(d1) - 1, 3),
                'gamma': round(norm_cdf(d1) / (underlying_price * volatility * math.sqrt(time_to_expiry)), 4),
                'theta': round(-put_price * 0.01, 2),  # Simplified
                'vega': round(underlying_price * norm_cdf(d1) * math.sqrt(time_to_expiry) * 0.01, 2)
            }
            
            options_chain['calls'].append(call_data)
            options_chain['puts'].append(put_data)
        
        return options_chain
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        sector_map = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology', 
            'MSFT': 'Technology',
            'TSLA': 'Consumer Discretionary',
            'AMZN': 'Consumer Discretionary',
            'META': 'Technology',
            'NVDA': 'Technology',
            'NFLX': 'Communication Services',
            'CRM': 'Technology',
            'ORCL': 'Technology'
        }
        return sector_map.get(symbol, 'Technology')
    
    def _get_next_expiry(self) -> str:
        """Get next monthly options expiry (3rd Friday)."""
        today = datetime.now()
        
        # Find next month's 3rd Friday
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        # Find 3rd Friday
        first_day = next_month.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday.strftime('%Y-%m-%d')


def generate_market_scenarios() -> Dict[str, Dict[str, Any]]:
    """Generate different market scenario data."""
    
    scenarios = {
        'normal_market': {
            'description': 'Normal market conditions',
            'volatility_multiplier': 1.0,
            'volume_multiplier': 1.0,
            'spread_multiplier': 1.0,
            'correlation': 0.3
        },
        
        'high_volatility': {
            'description': 'High volatility market',
            'volatility_multiplier': 2.5,
            'volume_multiplier': 1.5,
            'spread_multiplier': 2.0,
            'correlation': 0.7
        },
        
        'low_liquidity': {
            'description': 'Low liquidity conditions',
            'volatility_multiplier': 1.2,
            'volume_multiplier': 0.3,
            'spread_multiplier': 4.0,
            'correlation': 0.2
        },
        
        'trending_market': {
            'description': 'Strong trending market',
            'volatility_multiplier': 0.8,
            'volume_multiplier': 1.8,
            'spread_multiplier': 0.7,
            'correlation': 0.8,
            'trend_direction': 'up',
            'trend_strength': 0.15
        },
        
        'market_stress': {
            'description': 'Market stress conditions',
            'volatility_multiplier': 3.0,
            'volume_multiplier': 2.5,
            'spread_multiplier': 5.0,
            'correlation': 0.9,
            'flight_to_quality': True
        }
    }
    
    return scenarios


# Export functions
__all__ = [
    'MarketDataGenerator',
    'generate_market_scenarios'
]