"""
Stamp Duty Calculator
====================

Calculates stamp duty charges for securities transactions in India.
Stamp duty is typically charged on the buy side of transactions.

Current rate is 0.003% of transaction value with a maximum cap of ₹1500.
"""

from decimal import Decimal
import logging

from src.trading.transaction_costs.models import TransactionRequest, TransactionType

logger = logging.getLogger(__name__)


class StampDutyCalculator:
    """
    Calculator for stamp duty charges on securities transactions.
    
    Stamp duty is charged on purchase transactions (buy side) in India
    at 0.003% of transaction value with a maximum cap of ₹1500.
    """
    
    # Current stamp duty rate and maximum as of 2024
    STAMP_DUTY_RATE = Decimal('0.00003')  # 0.003%
    STAMP_DUTY_MAX = Decimal('1500.00')   # Maximum ₹1500
    
    @classmethod
    def calculate_stamp_duty(cls, request: TransactionRequest) -> Decimal:
        """
        Calculate stamp duty for a transaction.
        
        Stamp duty is only charged on buy transactions.
        
        Args:
            request: Transaction request details
            
        Returns:
            Stamp duty amount in INR
        """
        # Stamp duty only applies to buy transactions
        if request.transaction_type not in [TransactionType.BUY]:
            logger.debug(
                f"No stamp duty for {request.transaction_type.name} transaction"
            )
            return Decimal('0.00')
        
        notional_value = request.notional_value
        stamp_duty_amount = notional_value * cls.STAMP_DUTY_RATE
        
        # Apply maximum cap
        if stamp_duty_amount > cls.STAMP_DUTY_MAX:
            stamp_duty_amount = cls.STAMP_DUTY_MAX
            logger.debug(
                f"Stamp duty capped at maximum: ₹{cls.STAMP_DUTY_MAX} "
                f"for transaction value ₹{notional_value}"
            )
        
        logger.debug(
            f"Stamp duty calculation: {request.symbol} "
            f"Notional: ₹{notional_value}, Rate: {cls.STAMP_DUTY_RATE:.5%}, "
            f"Stamp Duty: ₹{stamp_duty_amount:.2f}"
        )
        
        return stamp_duty_amount
    
    @classmethod
    def get_rate(cls) -> Decimal:
        """
        Get current stamp duty rate.
        
        Returns:
            Stamp duty rate as decimal
        """
        return cls.STAMP_DUTY_RATE
    
    @classmethod
    def get_maximum_cap(cls) -> Decimal:
        """
        Get maximum stamp duty cap.
        
        Returns:
            Maximum stamp duty amount
        """
        return cls.STAMP_DUTY_MAX
    
    @classmethod
    def calculate_notional_for_max_stamp_duty(cls) -> Decimal:
        """
        Calculate the transaction value at which stamp duty reaches maximum.
        
        Returns:
            Transaction value where stamp duty = maximum cap
        """
        return cls.STAMP_DUTY_MAX / cls.STAMP_DUTY_RATE


logger.info("Stamp Duty Calculator loaded successfully")