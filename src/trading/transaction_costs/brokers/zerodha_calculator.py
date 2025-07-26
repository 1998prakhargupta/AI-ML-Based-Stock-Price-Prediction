"""
Zerodha (Kite Connect) Cost Calculator
======================================

Implementation of transaction cost calculator for Zerodha broker
with accurate fee structures as per their official brokerage calculator.

Zerodha Fee Structure:
- Equity Delivery: ₹0 (Free)
- Equity Intraday: 0.03% or ₹20 (whichever is lower)
- Options: ₹20 per order
- Futures: 0.03% or ₹20 (whichever is lower)
- Currency: 0.03% or ₹20 (whichever is lower)
- Commodity: 0.03% or ₹20 (whichever is lower)
"""

from decimal import Decimal
from typing import List, Dict, Any
import logging

from src.trading.transaction_costs.base_cost_calculator import CostCalculatorBase, CalculationMode
from src.trading.transaction_costs.models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    InstrumentType,
    TransactionType,
    OrderType
)
from src.trading.transaction_costs.exceptions import CalculationError
from .regulatory.charges_calculator import RegulatoryChargesCalculator

logger = logging.getLogger(__name__)


class ZerodhaCalculator(CostCalculatorBase):
    """
    Zerodha (Kite Connect) specific cost calculator.
    
    Implements the official Zerodha brokerage structure with
    accurate calculations for all supported instrument types.
    """
    
    # Zerodha brokerage rates
    BROKERAGE_RATES = {
        InstrumentType.EQUITY: {
            'delivery': Decimal('0.00'),      # Free for delivery
            'intraday': Decimal('0.0003'),    # 0.03% or ₹20, whichever lower
            'max_intraday': Decimal('20.00')
        },
        InstrumentType.OPTION: {
            'flat_fee': Decimal('20.00')      # ₹20 per order
        },
        InstrumentType.FUTURE: {
            'rate': Decimal('0.0003'),        # 0.03% or ₹20, whichever lower
            'max_fee': Decimal('20.00')
        },
        InstrumentType.CURRENCY: {
            'rate': Decimal('0.0003'),        # 0.03% or ₹20, whichever lower
            'max_fee': Decimal('20.00')
        },
        InstrumentType.COMMODITY: {
            'rate': Decimal('0.0003'),        # 0.03% or ₹20, whichever lower
            'max_fee': Decimal('20.00')
        }
    }
    
    def __init__(self, exchange: str = 'NSE'):
        """
        Initialize Zerodha calculator.
        
        Args:
            exchange: Exchange name (NSE/BSE)
        """
        super().__init__(
            calculator_name="Zerodha",
            version="1.0.0",
            supported_instruments=[
                InstrumentType.EQUITY,
                InstrumentType.OPTION,
                InstrumentType.FUTURE,
                InstrumentType.CURRENCY,
                InstrumentType.COMMODITY
            ],
            supported_modes=[
                CalculationMode.REAL_TIME,
                CalculationMode.BATCH,
                CalculationMode.HISTORICAL,
                CalculationMode.SIMULATION
            ]
        )
        
        # Initialize regulatory charges calculator
        self.regulatory_calculator = RegulatoryChargesCalculator(exchange=exchange)
        
        logger.info(f"Zerodha calculator initialized for {exchange} exchange")
    
    def _calculate_commission(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """
        Calculate Zerodha commission based on instrument type and order type.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration (not used for Zerodha as rates are fixed)
            
        Returns:
            Commission amount in INR
        """
        instrument_type = request.instrument_type
        
        if instrument_type not in self.BROKERAGE_RATES:
            raise CalculationError(
                f"Unsupported instrument type for Zerodha: {instrument_type.name}",
                calculation_step="commission_calculation",
                context={'instrument_type': instrument_type.name}
            )
        
        rates = self.BROKERAGE_RATES[instrument_type]
        
        if instrument_type == InstrumentType.EQUITY:
            return self._calculate_equity_commission(request, rates)
        elif instrument_type == InstrumentType.OPTION:
            return rates['flat_fee']
        else:
            # Futures, Currency, Commodity
            return self._calculate_percentage_commission(request, rates)
    
    def _calculate_equity_commission(
        self,
        request: TransactionRequest,
        rates: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate commission for equity transactions."""
        # Determine if it's delivery or intraday based on order type or metadata
        is_intraday = (
            request.order_type in [OrderType.MARKET, OrderType.LIMIT] and
            request.metadata.get('position_type') == 'intraday'
        )
        
        if is_intraday:
            # Intraday: 0.03% or ₹20, whichever is lower
            commission = request.notional_value * rates['intraday']
            return min(commission, rates['max_intraday'])
        else:
            # Delivery: Free
            return rates['delivery']
    
    def _calculate_percentage_commission(
        self,
        request: TransactionRequest,
        rates: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate percentage-based commission with maximum cap."""
        commission = request.notional_value * rates['rate']
        return min(commission, rates['max_fee'])
    
    def _calculate_regulatory_fees(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration
    ) -> Decimal:
        """
        Calculate total regulatory fees using the regulatory calculator.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration
            
        Returns:
            Total regulatory fees amount
        """
        # First calculate commission to determine GST base
        commission = self._calculate_commission(request, broker_config)
        
        # Calculate all regulatory charges
        charges = self.regulatory_calculator.calculate_all_charges(
            request, commission, broker_config
        )
        
        # Return total regulatory charges
        return charges['total_regulatory']
    
    def _calculate_market_impact(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions
    ) -> Decimal:
        """
        Calculate market impact cost.
        
        Market impact depends on order size relative to market liquidity.
        For retail orders, this is typically minimal.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            
        Returns:
            Estimated market impact cost
        """
        if not market_conditions or not market_conditions.volume:
            return Decimal('0.00')
        
        # Simple market impact model
        volume_ratio = Decimal(str(request.quantity)) / Decimal(str(market_conditions.volume))
        
        # For small orders (< 1% of daily volume), impact is minimal
        if volume_ratio < Decimal('0.01'):
            return Decimal('0.00')
        
        # For larger orders, estimate impact based on spread and volume
        if market_conditions.bid_ask_spread:
            impact_factor = min(volume_ratio * Decimal('0.1'), Decimal('0.005'))  # Max 0.5%
            return request.notional_value * impact_factor
        
        return Decimal('0.00')
    
    def _get_supported_instruments(self) -> List[InstrumentType]:
        """Return list of supported instrument types."""
        return list(self.BROKERAGE_RATES.keys())
    
    def get_detailed_breakdown(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: MarketConditions = None
    ) -> Dict[str, Any]:
        """
        Get detailed cost breakdown specific to Zerodha.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration
            market_conditions: Current market conditions
            
        Returns:
            Detailed breakdown dictionary
        """
        commission = self._calculate_commission(request, broker_config)
        
        # Get regulatory charges breakdown
        reg_charges = self.regulatory_calculator.calculate_all_charges(
            request, commission, broker_config
        )
        
        breakdown = {
            'broker': 'Zerodha',
            'brokerage': {
                'commission': float(commission),
                'rate_applied': self._get_rate_description(request),
                'calculation_method': self._get_calculation_method(request)
            },
            'regulatory_charges': self.regulatory_calculator.get_charge_breakdown_summary(reg_charges),
            'total_explicit_cost': float(commission + reg_charges['total_regulatory']),
            'cost_per_share': float((commission + reg_charges['total_regulatory']) / Decimal(str(request.quantity))),
            'cost_percentage': float((commission + reg_charges['total_regulatory']) / request.notional_value * 100)
        }
        
        if market_conditions:
            market_impact = self._calculate_market_impact(request, market_conditions)
            breakdown['market_impact'] = float(market_impact)
            breakdown['total_cost'] = breakdown['total_explicit_cost'] + float(market_impact)
        
        return breakdown
    
    def _get_rate_description(self, request: TransactionRequest) -> str:
        """Get human-readable rate description."""
        instrument_type = request.instrument_type
        
        if instrument_type == InstrumentType.EQUITY:
            is_intraday = request.metadata.get('position_type') == 'intraday'
            if is_intraday:
                return "0.03% or ₹20 (whichever lower) - Intraday"
            else:
                return "₹0 - Delivery"
        elif instrument_type == InstrumentType.OPTION:
            return "₹20 per order"
        else:
            return "0.03% or ₹20 (whichever lower)"
    
    def _get_calculation_method(self, request: TransactionRequest) -> str:
        """Get calculation method description."""
        instrument_type = request.instrument_type
        
        if instrument_type == InstrumentType.EQUITY:
            is_intraday = request.metadata.get('position_type') == 'intraday'
            if is_intraday:
                return f"min({request.notional_value} × 0.03%, ₹20)"
            else:
                return "No charge for delivery"
        elif instrument_type == InstrumentType.OPTION:
            return "Flat fee per order"
        else:
            return f"min({request.notional_value} × 0.03%, ₹20)"


logger.info("Zerodha calculator loaded successfully")