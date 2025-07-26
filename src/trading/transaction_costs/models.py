"""
Transaction Cost Data Models
============================

Core data structures for transaction cost modeling and analysis.
These models define the standard format for transaction requests, cost breakdowns,
market conditions, and broker configurations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Transaction type enumeration."""
    BUY = auto()
    SELL = auto()
    SHORT = auto()
    COVER = auto()


class InstrumentType(Enum):
    """Financial instrument type enumeration."""
    EQUITY = auto()
    OPTION = auto()
    FUTURE = auto()
    ETF = auto()
    MUTUAL_FUND = auto()
    BOND = auto()
    COMMODITY = auto()
    CURRENCY = auto()
    DERIVATIVE = auto()


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()


class MarketTiming(Enum):
    """Market timing enumeration."""
    MARKET_HOURS = auto()
    PRE_MARKET = auto()
    AFTER_HOURS = auto()
    EXTENDED_HOURS = auto()


@dataclass
class TransactionRequest:
    """
    Represents a transaction request for cost calculation.
    
    This is the primary input structure for all cost calculators,
    containing all information needed to determine transaction costs.
    """
    
    # Core transaction details
    symbol: str
    quantity: int
    price: Decimal
    transaction_type: TransactionType
    instrument_type: InstrumentType
    
    # Order specifications
    order_type: OrderType = OrderType.MARKET
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Market context
    market_timing: MarketTiming = MarketTiming.MARKET_HOURS
    exchange: Optional[str] = None
    currency: str = "USD"
    
    # Optional context
    account_type: Optional[str] = None  # margin, cash, etc.
    portfolio_value: Optional[Decimal] = None
    
    # Metadata
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize transaction request data."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.price <= 0:
            raise ValueError("Price must be positive")
        
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Symbol cannot be empty")
        
        # Normalize symbol
        self.symbol = self.symbol.strip().upper()
        
        # Ensure price is Decimal
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the transaction."""
        return Decimal(str(self.quantity)) * self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': float(self.price),
            'transaction_type': self.transaction_type.name,
            'instrument_type': self.instrument_type.name,
            'order_type': self.order_type.name,
            'timestamp': self.timestamp.isoformat(),
            'market_timing': self.market_timing.name,
            'exchange': self.exchange,
            'currency': self.currency,
            'account_type': self.account_type,
            'portfolio_value': float(self.portfolio_value) if self.portfolio_value else None,
            'request_id': self.request_id,
            'metadata': self.metadata
        }


@dataclass
class TransactionCostBreakdown:
    """
    Detailed breakdown of all costs associated with a transaction.
    
    Provides comprehensive cost analysis including regulatory fees,
    broker commissions, market impact, and total cost calculations.
    """
    
    # Primary costs
    commission: Decimal = Decimal('0.00')
    regulatory_fees: Decimal = Decimal('0.00')
    exchange_fees: Decimal = Decimal('0.00')
    
    # Market impact costs
    bid_ask_spread_cost: Decimal = Decimal('0.00')
    market_impact_cost: Decimal = Decimal('0.00')
    timing_cost: Decimal = Decimal('0.00')
    
    # Additional costs
    borrowing_cost: Decimal = Decimal('0.00')  # for short positions
    overnight_financing: Decimal = Decimal('0.00')
    currency_conversion: Decimal = Decimal('0.00')
    
    # Other fees
    platform_fees: Decimal = Decimal('0.00')
    data_fees: Decimal = Decimal('0.00')
    miscellaneous_fees: Decimal = Decimal('0.00')
    
    # Calculation metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    calculator_version: Optional[str] = None
    confidence_level: Optional[float] = None  # 0.0 to 1.0
    
    # Cost breakdown details
    cost_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_explicit_costs(self) -> Decimal:
        """Calculate total explicit costs (commissions + fees)."""
        return (
            self.commission +
            self.regulatory_fees +
            self.exchange_fees +
            self.platform_fees +
            self.data_fees +
            self.miscellaneous_fees +
            self.currency_conversion +
            self.borrowing_cost +
            self.overnight_financing
        )
    
    @property
    def total_implicit_costs(self) -> Decimal:
        """Calculate total implicit costs (market impact + timing)."""
        return (
            self.bid_ask_spread_cost +
            self.market_impact_cost +
            self.timing_cost
        )
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total transaction cost."""
        return self.total_explicit_costs + self.total_implicit_costs
    
    def cost_as_basis_points(self, notional_value: Decimal) -> Decimal:
        """Calculate total cost in basis points of notional value."""
        if notional_value <= 0:
            return Decimal('0.00')
        return (self.total_cost / notional_value) * Decimal('10000')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'commission': float(self.commission),
            'regulatory_fees': float(self.regulatory_fees),
            'exchange_fees': float(self.exchange_fees),
            'bid_ask_spread_cost': float(self.bid_ask_spread_cost),
            'market_impact_cost': float(self.market_impact_cost),
            'timing_cost': float(self.timing_cost),
            'borrowing_cost': float(self.borrowing_cost),
            'overnight_financing': float(self.overnight_financing),
            'currency_conversion': float(self.currency_conversion),
            'platform_fees': float(self.platform_fees),
            'data_fees': float(self.data_fees),
            'miscellaneous_fees': float(self.miscellaneous_fees),
            'total_explicit_costs': float(self.total_explicit_costs),
            'total_implicit_costs': float(self.total_implicit_costs),
            'total_cost': float(self.total_cost),
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'calculator_version': self.calculator_version,
            'confidence_level': self.confidence_level,
            'cost_details': self.cost_details
        }


@dataclass
class MarketConditions:
    """
    Current market conditions affecting transaction costs.
    
    Captures market state information that impacts cost calculations
    such as volatility, liquidity, and bid-ask spreads.
    """
    
    # Core market data
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    last_price: Optional[Decimal] = None
    volume: Optional[int] = None
    
    # Market microstructure
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    spread_bps: Optional[Decimal] = None
    
    # Volatility measures
    implied_volatility: Optional[Decimal] = None
    realized_volatility: Optional[Decimal] = None
    
    # Liquidity indicators
    average_daily_volume: Optional[int] = None
    market_cap: Optional[Decimal] = None
    days_to_expiry: Optional[int] = None  # for options/futures
    
    # Market state
    market_open: bool = True
    circuit_breaker: bool = False
    halt_status: bool = False
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    data_source: Optional[str] = None
    
    @property
    def bid_ask_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread if both prices available."""
        if self.bid_price is not None and self.ask_price is not None:
            return self.ask_price - self.bid_price
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price if both bid and ask available."""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / Decimal('2')
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bid_price': float(self.bid_price) if self.bid_price else None,
            'ask_price': float(self.ask_price) if self.ask_price else None,
            'last_price': float(self.last_price) if self.last_price else None,
            'volume': self.volume,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'spread_bps': float(self.spread_bps) if self.spread_bps else None,
            'implied_volatility': float(self.implied_volatility) if self.implied_volatility else None,
            'realized_volatility': float(self.realized_volatility) if self.realized_volatility else None,
            'average_daily_volume': self.average_daily_volume,
            'market_cap': float(self.market_cap) if self.market_cap else None,
            'days_to_expiry': self.days_to_expiry,
            'market_open': self.market_open,
            'circuit_breaker': self.circuit_breaker,
            'halt_status': self.halt_status,
            'timestamp': self.timestamp.isoformat(),
            'data_source': self.data_source
        }


@dataclass
class BrokerConfiguration:
    """
    Broker-specific configuration for cost calculations.
    
    Contains all broker-specific parameters needed for accurate
    cost calculation including fee schedules and account settings.
    """
    
    # Broker identification
    broker_name: str
    broker_id: Optional[str] = None
    
    # Commission structure
    equity_commission: Decimal = Decimal('0.00')
    options_commission: Decimal = Decimal('0.00')
    futures_commission: Decimal = Decimal('0.00')
    
    # Commission minimums and maximums
    min_commission: Decimal = Decimal('0.00')
    max_commission: Optional[Decimal] = None
    
    # Per-contract fees (options/futures)
    options_per_contract: Decimal = Decimal('0.00')
    futures_per_contract: Decimal = Decimal('0.00')
    
    # Regulatory fees (typically passed through)
    sec_fee_rate: Decimal = Decimal('0.0000051')  # Current SEC fee
    finra_taf_rate: Decimal = Decimal('0.000166')  # Current FINRA TAF
    
    # Platform and data fees
    platform_fee: Decimal = Decimal('0.00')
    data_fee: Decimal = Decimal('0.00')
    
    # Account-specific settings
    account_tier: Optional[str] = None  # professional, retail, institutional
    volume_discount_tiers: Dict[str, Decimal] = field(default_factory=dict)
    
    # Special rates
    pre_market_multiplier: Decimal = Decimal('1.0')
    after_hours_multiplier: Decimal = Decimal('1.0')
    
    # Currency and location
    base_currency: str = "USD"
    timezone: str = "US/Eastern"
    
    # Metadata
    config_version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)
    active: bool = True
    
    def __post_init__(self):
        """Validate broker configuration."""
        if not self.broker_name or not self.broker_name.strip():
            raise ValueError("Broker name cannot be empty")
        
        # Ensure all rates are positive
        numeric_fields = [
            'equity_commission', 'options_commission', 'futures_commission',
            'min_commission', 'options_per_contract', 'futures_per_contract',
            'sec_fee_rate', 'finra_taf_rate', 'platform_fee', 'data_fee'
        ]
        
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
    
    def get_commission_rate(self, instrument_type: InstrumentType) -> Decimal:
        """Get commission rate for specific instrument type."""
        if instrument_type == InstrumentType.EQUITY:
            return self.equity_commission
        elif instrument_type == InstrumentType.OPTION:
            return self.options_commission
        elif instrument_type == InstrumentType.FUTURE:
            return self.futures_commission
        else:
            return self.equity_commission  # Default to equity rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'broker_name': self.broker_name,
            'broker_id': self.broker_id,
            'equity_commission': float(self.equity_commission),
            'options_commission': float(self.options_commission),
            'futures_commission': float(self.futures_commission),
            'min_commission': float(self.min_commission),
            'max_commission': float(self.max_commission) if self.max_commission else None,
            'options_per_contract': float(self.options_per_contract),
            'futures_per_contract': float(self.futures_per_contract),
            'sec_fee_rate': float(self.sec_fee_rate),
            'finra_taf_rate': float(self.finra_taf_rate),
            'platform_fee': float(self.platform_fee),
            'data_fee': float(self.data_fee),
            'account_tier': self.account_tier,
            'volume_discount_tiers': {k: float(v) for k, v in self.volume_discount_tiers.items()},
            'pre_market_multiplier': float(self.pre_market_multiplier),
            'after_hours_multiplier': float(self.after_hours_multiplier),
            'base_currency': self.base_currency,
            'timezone': self.timezone,
            'config_version': self.config_version,
            'last_updated': self.last_updated.isoformat(),
            'active': self.active
        }


logger.info("Transaction cost data models loaded successfully")