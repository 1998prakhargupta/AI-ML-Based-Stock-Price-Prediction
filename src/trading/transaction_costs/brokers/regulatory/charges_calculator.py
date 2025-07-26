"""
Regulatory Charges Calculator
============================

Comprehensive calculator for all regulatory charges applicable
to securities transactions in India including STT, CTT, GST,
SEBI charges, exchange charges, and stamp duty.

This module coordinates between different charge calculators
to provide a complete regulatory fee breakdown.
"""

from decimal import Decimal
from typing import Dict, Any
import logging

from src.trading.transaction_costs.models import (
    TransactionRequest, 
    InstrumentType, 
    TransactionType,
    BrokerConfiguration
)
from .stt_calculator import STTCalculator
from .gst_calculator import GSTCalculator
from .stamp_duty_calculator import StampDutyCalculator

logger = logging.getLogger(__name__)


class RegulatoryChargesCalculator:
    """
    Comprehensive calculator for all regulatory charges.
    
    Coordinates calculation of STT, CTT, GST, SEBI charges,
    exchange charges, and stamp duty for securities transactions.
    """
    
    # SEBI charges: ₹10 per crore of transaction value
    SEBI_CHARGE_RATE = Decimal('0.0000001')  # ₹10 per ₹1 crore
    
    # Exchange transaction charges (approximate rates)
    EXCHANGE_CHARGES = {
        'NSE': {
            InstrumentType.EQUITY: Decimal('0.00003'),      # 0.003%
            InstrumentType.OPTION: Decimal('0.0005'),       # 0.05%
            InstrumentType.FUTURE: Decimal('0.0002'),       # 0.02%
            InstrumentType.CURRENCY: Decimal('0.00004'),    # 0.004%
            InstrumentType.COMMODITY: Decimal('0.00026'),   # 0.026%
        },
        'BSE': {
            InstrumentType.EQUITY: Decimal('0.00003'),      # 0.003%
            InstrumentType.OPTION: Decimal('0.0005'),       # 0.05%
            InstrumentType.FUTURE: Decimal('0.0002'),       # 0.02%
        }
    }
    
    # Clearing charges (approximate rates)
    CLEARING_CHARGES = {
        'NSE': Decimal('0.00001'),   # 0.001%
        'BSE': Decimal('0.00001'),   # 0.001%
    }
    
    # CTT for commodities
    CTT_RATE = Decimal('0.0001')  # 0.01% on sell side
    
    def __init__(self, exchange: str = 'NSE'):
        """
        Initialize regulatory charges calculator.
        
        Args:
            exchange: Exchange name (NSE/BSE)
        """
        self.exchange = exchange.upper()
        if self.exchange not in ['NSE', 'BSE']:
            logger.warning(f"Unknown exchange {exchange}, defaulting to NSE")
            self.exchange = 'NSE'
    
    def calculate_all_charges(
        self, 
        request: TransactionRequest,
        brokerage_amount: Decimal,
        broker_config: BrokerConfiguration
    ) -> Dict[str, Decimal]:
        """
        Calculate all regulatory charges for a transaction.
        
        Args:
            request: Transaction request details
            brokerage_amount: Brokerage charge amount (for GST calculation)
            broker_config: Broker configuration
            
        Returns:
            Dictionary containing all regulatory charge components
        """
        charges = {}
        
        # STT calculation
        charges['stt'] = STTCalculator.calculate_stt(request)
        
        # CTT for commodities (sell side only)
        charges['ctt'] = self._calculate_ctt(request)
        
        # Exchange transaction charges
        charges['exchange_transaction_charge'] = self._calculate_exchange_charges(request)
        
        # Exchange clearing charges
        charges['clearing_charge'] = self._calculate_clearing_charges(request)
        
        # SEBI charges
        charges['sebi_charge'] = self._calculate_sebi_charges(request)
        
        # Stamp duty (buy side only)
        charges['stamp_duty'] = StampDutyCalculator.calculate_stamp_duty(request)
        
        # Calculate GST on brokerage and applicable charges
        gst_applicable_amount = brokerage_amount + charges['sebi_charge']
        charges['gst'] = GSTCalculator.calculate_gst(gst_applicable_amount)
        
        # Total regulatory charges
        charges['total_regulatory'] = sum(charges.values())
        
        logger.debug(
            f"Regulatory charges for {request.symbol}: "
            f"STT: ₹{charges['stt']:.2f}, "
            f"Exchange: ₹{charges['exchange_transaction_charge']:.2f}, "
            f"Stamp Duty: ₹{charges['stamp_duty']:.2f}, "
            f"GST: ₹{charges['gst']:.2f}, "
            f"Total: ₹{charges['total_regulatory']:.2f}"
        )
        
        return charges
    
    def _calculate_ctt(self, request: TransactionRequest) -> Decimal:
        """Calculate Commodities Transaction Tax (CTT)."""
        if (request.instrument_type == InstrumentType.COMMODITY and 
            request.transaction_type == TransactionType.SELL):
            return request.notional_value * self.CTT_RATE
        return Decimal('0.00')
    
    def _calculate_exchange_charges(self, request: TransactionRequest) -> Decimal:
        """Calculate exchange transaction charges."""
        if self.exchange not in self.EXCHANGE_CHARGES:
            return Decimal('0.00')
        
        exchange_rates = self.EXCHANGE_CHARGES[self.exchange]
        rate = exchange_rates.get(request.instrument_type, Decimal('0.00'))
        
        return request.notional_value * rate
    
    def _calculate_clearing_charges(self, request: TransactionRequest) -> Decimal:
        """Calculate exchange clearing charges."""
        rate = self.CLEARING_CHARGES.get(self.exchange, Decimal('0.00'))
        return request.notional_value * rate
    
    def _calculate_sebi_charges(self, request: TransactionRequest) -> Decimal:
        """Calculate SEBI charges (₹10 per crore)."""
        return request.notional_value * self.SEBI_CHARGE_RATE
    
    def get_charge_breakdown_summary(
        self, 
        charges: Dict[str, Decimal]
    ) -> Dict[str, Any]:
        """
        Get a summary of charge breakdown for reporting.
        
        Args:
            charges: Dictionary of calculated charges
            
        Returns:
            Formatted summary of charges
        """
        return {
            'statutory_charges': {
                'stt': float(charges.get('stt', 0)),
                'ctt': float(charges.get('ctt', 0)),
                'stamp_duty': float(charges.get('stamp_duty', 0))
            },
            'exchange_charges': {
                'transaction_charge': float(charges.get('exchange_transaction_charge', 0)),
                'clearing_charge': float(charges.get('clearing_charge', 0))
            },
            'regulatory_charges': {
                'sebi_charge': float(charges.get('sebi_charge', 0))
            },
            'taxes': {
                'gst': float(charges.get('gst', 0))
            },
            'total_regulatory': float(charges.get('total_regulatory', 0)),
            'exchange': self.exchange
        }


logger.info("Regulatory Charges Calculator loaded successfully")