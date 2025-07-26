"""
Goods and Services Tax (GST) Calculator
=======================================

Calculates GST charges on brokerage and statutory charges
for securities transactions in India.

GST is currently 18% on brokerage and most statutory charges.
"""

from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class GSTCalculator:
    """
    Calculator for Goods and Services Tax (GST) charges.
    
    GST is applied to brokerage charges and certain statutory charges
    in securities transactions in India.
    """
    
    # Current GST rate (18% as of 2024)
    GST_RATE = Decimal('0.18')
    
    @classmethod
    def calculate_gst(cls, taxable_amount: Decimal) -> Decimal:
        """
        Calculate GST on taxable amount.
        
        Args:
            taxable_amount: Amount subject to GST (brokerage + applicable charges)
            
        Returns:
            GST amount in INR
        """
        if taxable_amount <= 0:
            return Decimal('0.00')
        
        gst_amount = taxable_amount * cls.GST_RATE
        
        logger.debug(
            f"GST calculation: Taxable Amount: ₹{taxable_amount:.2f}, "
            f"GST Rate: {cls.GST_RATE:.1%}, GST: ₹{gst_amount:.2f}"
        )
        
        return gst_amount
    
    @classmethod
    def calculate_gst_on_brokerage(cls, brokerage_amount: Decimal) -> Decimal:
        """
        Calculate GST specifically on brokerage charges.
        
        Args:
            brokerage_amount: Brokerage charge amount
            
        Returns:
            GST amount on brokerage
        """
        return cls.calculate_gst(brokerage_amount)
    
    @classmethod
    def calculate_gst_on_charges(cls, charges_amount: Decimal) -> Decimal:
        """
        Calculate GST on statutory/regulatory charges.
        
        Note: Not all charges are subject to GST. STT, for example, is not.
        
        Args:
            charges_amount: Amount of charges subject to GST
            
        Returns:
            GST amount on charges
        """
        return cls.calculate_gst(charges_amount)
    
    @classmethod
    def get_gst_rate(cls) -> Decimal:
        """
        Get current GST rate.
        
        Returns:
            GST rate as decimal (e.g., 0.18 for 18%)
        """
        return cls.GST_RATE
    
    @classmethod
    def calculate_total_with_gst(cls, base_amount: Decimal) -> Decimal:
        """
        Calculate total amount including GST.
        
        Args:
            base_amount: Base amount before GST
            
        Returns:
            Total amount including GST
        """
        gst_amount = cls.calculate_gst(base_amount)
        return base_amount + gst_amount


logger.info("GST Calculator loaded successfully")