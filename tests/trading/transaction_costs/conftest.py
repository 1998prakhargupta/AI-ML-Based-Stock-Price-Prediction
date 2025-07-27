"""
Transaction Cost Testing Configuration
======================================

Central configuration and fixtures for transaction cost testing suite.
Provides shared test data, mock objects, and common testing utilities.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
import asyncio
import logging
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from trading.transaction_costs.models import (
        TransactionRequest,
        TransactionCostBreakdown,
        MarketConditions,
        BrokerConfiguration,
        InstrumentType,
        TransactionType,
        OrderType,
        MarketTiming
    )
except ImportError:
    # Mock the models if they're not available
    from enum import Enum, auto
    from dataclasses import dataclass
    from typing import Optional
    
    class TransactionType(Enum):
        BUY = auto()
        SELL = auto()
    
    class InstrumentType(Enum):
        EQUITY = auto()
        OPTION = auto()
    
    class OrderType(Enum):
        MARKET = auto()
        LIMIT = auto()
    
    class MarketTiming(Enum):
        REGULAR = auto()
        PRE_MARKET = auto()
        POST_MARKET = auto()
    
    @dataclass
    class TransactionRequest:
        symbol: str
        quantity: int
        price: Decimal
        transaction_type: TransactionType
        instrument_type: InstrumentType
        order_type: OrderType = OrderType.MARKET
        timestamp: datetime = None
    
    @dataclass
    class TransactionCostBreakdown:
        total_cost: Decimal
        commission: Decimal
        market_impact: Decimal
        spread_cost: Decimal
        slippage: Decimal
        regulatory_fees: Decimal
    
    @dataclass
    class MarketConditions:
        volatility: float
        volume: int
        bid_ask_spread: float
        market_timing: MarketTiming
    
    @dataclass
    class BrokerConfiguration:
        broker_name: str
        commission_rate: float
        minimum_commission: float


# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_equity_request():
    """Standard equity transaction request for testing."""
    return TransactionRequest(
        symbol="AAPL",
        quantity=100,
        price=Decimal("150.50"),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY,
        order_type=OrderType.MARKET,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_option_request():
    """Sample option transaction request for testing."""
    return TransactionRequest(
        symbol="AAPL240315C00150000",
        quantity=1,
        price=Decimal("5.25"),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.OPTION,
        order_type=OrderType.LIMIT,
        timestamp=datetime.now()
    )


@pytest.fixture
def high_volume_request():
    """High volume transaction for performance testing."""
    return TransactionRequest(
        symbol="AAPL",
        quantity=10000,
        price=Decimal("150.50"),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY,
        order_type=OrderType.MARKET,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_market_conditions():
    """Standard market conditions for testing."""
    return MarketConditions(
        volatility=0.25,
        volume=1000000,
        bid_ask_spread=0.02,
        market_timing="REGULAR"
    )


@pytest.fixture
def volatile_market_conditions():
    """High volatility market conditions for stress testing."""
    return MarketConditions(
        volatility=0.75,
        volume=500000,
        bid_ask_spread=0.10,
        market_timing="REGULAR"
    )


@pytest.fixture
def zerodha_config():
    """Zerodha broker configuration."""
    return BrokerConfiguration(
        broker_name="zerodha",
        commission_rate=0.0003,
        minimum_commission=0.0
    )


@pytest.fixture
def icici_config():
    """ICICI Securities broker configuration."""
    return BrokerConfiguration(
        broker_name="icici",
        commission_rate=0.0005,
        minimum_commission=20.0
    )


@pytest.fixture
def sample_cost_breakdown():
    """Standard cost breakdown for validation testing."""
    return TransactionCostBreakdown(
        total_cost=Decimal("45.23"),
        commission=Decimal("15.08"),
        market_impact=Decimal("12.50"),
        spread_cost=Decimal("10.15"),
        slippage=Decimal("5.25"),
        regulatory_fees=Decimal("2.25")
    )


@pytest.fixture
def batch_transaction_requests():
    """Batch of transaction requests for throughput testing."""
    requests = []
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    for i in range(100):
        requests.append(TransactionRequest(
            symbol=symbols[i % len(symbols)],
            quantity=100 + (i * 10),
            price=Decimal(f"{100 + i}.{50 + (i % 50):02d}"),
            transaction_type=TransactionType.BUY if i % 2 == 0 else TransactionType.SELL,
            instrument_type=InstrumentType.EQUITY,
            order_type=OrderType.MARKET,
            timestamp=datetime.now() + timedelta(milliseconds=i)
        ))
    return requests


@pytest.fixture
def performance_test_config():
    """Configuration for performance testing."""
    return {
        'max_latency_ms': 10,  # 10ms per calculation
        'min_throughput': 100,  # 100 transactions/second
        'max_memory_mb': 1024,  # 1GB memory limit
        'cache_hit_rate_target': 0.9,  # 90% cache hit rate
        'availability_target': 0.999  # 99.9% availability
    }


@pytest.fixture
def accuracy_benchmarks():
    """Known cost calculation benchmarks for accuracy validation."""
    return {
        'zerodha_equity_buy_100_shares_150': {
            'expected_commission': Decimal("20.00"),
            'expected_total_range': (Decimal("40.00"), Decimal("60.00"))
        },
        'icici_equity_sell_500_shares_100': {
            'expected_commission': Decimal("50.00"),
            'expected_total_range': (Decimal("80.00"), Decimal("120.00"))
        }
    }


@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return {
        'AAPL': {
            'bid': 150.45,
            'ask': 150.55,
            'last': 150.50,
            'volume': 25000000,
            'volatility': 0.25
        },
        'GOOGL': {
            'bid': 2875.25,
            'ask': 2875.75,
            'last': 2875.50,
            'volume': 1500000,
            'volatility': 0.30
        },
        'MSFT': {
            'bid': 415.80,
            'ask': 415.90,
            'last': 415.85,
            'volume': 18000000,
            'volatility': 0.22
        }
    }


@pytest.fixture
def stress_test_scenarios():
    """Scenarios for stress and edge case testing."""
    return {
        'extreme_high_volume': {
            'quantity': 1000000,
            'description': 'Extremely high volume trade'
        },
        'micro_trade': {
            'quantity': 1,
            'description': 'Single share trade'
        },
        'market_close_trade': {
            'market_timing': "POST_MARKET",
            'description': 'After-hours trading'
        },
        'high_volatility': {
            'volatility': 2.0,
            'description': 'Extreme market volatility'
        }
    }


class MockCostCalculator:
    """Mock cost calculator for testing."""
    
    def __init__(self, name="MockCalculator"):
        self.name = name
        self.calculation_count = 0
        
    async def calculate_cost(self, request: TransactionRequest) -> TransactionCostBreakdown:
        """Mock cost calculation."""
        self.calculation_count += 1
        
        # Simple mock calculation
        notional = request.price * request.quantity
        commission = max(notional * Decimal("0.0005"), Decimal("1.00"))
        
        return TransactionCostBreakdown(
            total_cost=commission * Decimal("2.5"),
            commission=commission,
            market_impact=commission * Decimal("0.8"),
            spread_cost=commission * Decimal("0.6"),
            slippage=commission * Decimal("0.4"),
            regulatory_fees=commission * Decimal("0.1")
        )


@pytest.fixture
def mock_calculator():
    """Mock cost calculator instance."""
    return MockCostCalculator()


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.validation = pytest.mark.validation
pytest.mark.scenario = pytest.mark.scenario