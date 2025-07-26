"""
Securities Transaction Tax (STT) Calculator
===========================================

Calculates STT charges for different types of securities transactions
in the Indian market according to current tax rates.

STT rates vary by instrument type and transaction type (buy/sell).
"""

from decimal import Decimal
from typing import Dict, Any
import logging

from src.trading.transaction_costs.models import TransactionRequest, InstrumentType, TransactionType

logger = logging.getLogger(__name__)


class STTCalculator:
    """
    Calculator for Securities Transaction Tax (STT) charges.
    
    STT is a tax levied on the purchase and sale of securities listed
    on Indian stock exchanges. Rates vary by instrument and transaction type.
    """
    
    # STT rates as of 2024 (subject to change by government notification)
    STT_RATES = {
        InstrumentType.EQUITY: {
            TransactionType.BUY: Decimal('0.001'),    # 0.1% on purchase
            TransactionType.SELL: Decimal('0.001'),   # 0.1% on sale
        },
        InstrumentType.OPTION: {
            TransactionType.BUY: Decimal('0.0005'),   # 0.05% on purchase of options
            TransactionType.SELL: Decimal('0.0005'),  # 0.05% on sale of options
        },
        InstrumentType.FUTURE: {
            TransactionType.BUY: Decimal('0.0001'),   # 0.01% on purchase of futures
            TransactionType.SELL: Decimal('0.0001'),  # 0.01% on sale of futures
        },
        InstrumentType.ETF: {
            TransactionType.BUY: Decimal('0.001'),    # 0.1% on purchase
            TransactionType.SELL: Decimal('0.001'),   # 0.1% on sale
        }
    }
    
    @classmethod
    def calculate_stt(cls, request: TransactionRequest) -> Decimal:
        """
        Calculate STT for a transaction.
        
        Args:
            request: Transaction request details
            
        Returns:
            STT amount in INR
            
        Raises:
            ValueError: If instrument type not supported for STT
        """
        if request.instrument_type not in cls.STT_RATES:
            logger.warning(f"No STT rate defined for {request.instrument_type.name}")
            return Decimal('0.00')
        
        instrument_rates = cls.STT_RATES[request.instrument_type]
        
        if request.transaction_type not in instrument_rates:
            logger.warning(
                f"No STT rate defined for {request.transaction_type.name} "
                f"of {request.instrument_type.name}"
            )
            return Decimal('0.00')
        
        rate = instrument_rates[request.transaction_type]
        notional_value = request.notional_value
        stt_amount = notional_value * rate
        
        logger.debug(
            f"STT calculation: {request.symbol} {request.instrument_type.name} "
            f"{request.transaction_type.name} - Rate: {rate:.4%}, "
            f"Notional: ₹{notional_value}, STT: ₹{stt_amount:.2f}"
        )
        
        return stt_amount
    
    @classmethod
    def get_rate(
        cls, 
        instrument_type: InstrumentType, 
        transaction_type: TransactionType
    ) -> Decimal:
        """
        Get STT rate for specific instrument and transaction type.
        
        Args:
            instrument_type: Type of financial instrument
            transaction_type: Type of transaction
            
        Returns:
            STT rate as decimal (e.g., 0.001 for 0.1%)
        """
        if instrument_type not in cls.STT_RATES:
            return Decimal('0.00')
        
        instrument_rates = cls.STT_RATES[instrument_type]
        return instrument_rates.get(transaction_type, Decimal('0.00'))
    
    @classmethod
    def get_all_rates(cls) -> Dict[str, Dict[str, float]]:
        """
        Get all STT rates for documentation purposes.
        
        Returns:
            Dictionary of all STT rates by instrument and transaction type
        """
        rates = {}
        for instrument, transactions in cls.STT_RATES.items():
            rates[instrument.name] = {}
            for transaction, rate in transactions.items():
                rates[instrument.name][transaction.name] = float(rate)
        
        return rates


logger.info("STT Calculator loaded successfully")