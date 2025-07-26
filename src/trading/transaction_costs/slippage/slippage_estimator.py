"""
Slippage Estimator
=================

Provides unified interface for slippage estimation using multiple models.
Combines different slippage components for comprehensive cost estimation.
"""

from decimal import Decimal
from datetime import timedelta
from typing import Dict, Any, Optional, List
import logging

from .delay_slippage import DelaySlippageModel
from .size_slippage import SizeSlippageModel
from .condition_slippage import ConditionSlippageModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class SlippageEstimator:
    """
    Unified slippage estimator that combines multiple slippage models.
    
    Provides comprehensive slippage estimation by combining:
    - Execution delay effects
    - Order size effects
    - Market condition effects
    """
    
    def __init__(
        self,
        include_delay_slippage: bool = True,
        include_size_slippage: bool = True,
        include_condition_slippage: bool = True,
        correlation_adjustment: Decimal = Decimal('0.8')
    ):
        """
        Initialize the slippage estimator.
        
        Args:
            include_delay_slippage: Include delay-based slippage
            include_size_slippage: Include size-based slippage
            include_condition_slippage: Include condition-based slippage
            correlation_adjustment: Adjustment factor for component correlation (0.5-1.0)
        """
        self.include_delay_slippage = include_delay_slippage
        self.include_size_slippage = include_size_slippage
        self.include_condition_slippage = include_condition_slippage
        self.correlation_adjustment = correlation_adjustment
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize component models
        self.delay_model = DelaySlippageModel() if include_delay_slippage else None
        self.size_model = SizeSlippageModel() if include_size_slippage else None
        self.condition_model = ConditionSlippageModel() if include_condition_slippage else None
        
        self.logger.info(
            f"Initialized Slippage Estimator: "
            f"delay={include_delay_slippage}, "
            f"size={include_size_slippage}, "
            f"condition={include_condition_slippage}, "
            f"correlation_adj={correlation_adjustment}"
        )
    
    def estimate_total_slippage(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions] = None,
        execution_delay: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Estimate total slippage combining all enabled models.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Expected execution delay
            
        Returns:
            Comprehensive slippage estimation results
        """
        components = {}
        total_slippage = Decimal('0.0')
        errors = []
        
        # Calculate delay slippage
        if self.delay_model:
            try:
                delay_slippage = self.delay_model.calculate_slippage(
                    request, market_conditions, execution_delay
                )
                components['delay_slippage'] = float(delay_slippage)
                total_slippage += delay_slippage
            except Exception as e:
                error_msg = f"Delay slippage calculation failed: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                components['delay_slippage'] = 0.0
        
        # Calculate size slippage
        if self.size_model:
            try:
                size_slippage = self.size_model.calculate_slippage(
                    request, market_conditions, execution_delay
                )
                components['size_slippage'] = float(size_slippage)
                total_slippage += size_slippage
            except Exception as e:
                error_msg = f"Size slippage calculation failed: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                components['size_slippage'] = 0.0
        
        # Calculate condition slippage
        if self.condition_model:
            try:
                condition_slippage = self.condition_model.calculate_slippage(
                    request, market_conditions, execution_delay
                )
                components['condition_slippage'] = float(condition_slippage)
                total_slippage += condition_slippage
            except Exception as e:
                error_msg = f"Condition slippage calculation failed: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                components['condition_slippage'] = 0.0
        
        # Apply correlation adjustment to avoid double-counting
        adjusted_slippage = total_slippage * self.correlation_adjustment
        
        # Calculate slippage as basis points
        notional_value = Decimal(str(request.quantity)) * request.price
        slippage_bps = (adjusted_slippage / notional_value * Decimal('10000')) if notional_value > 0 else Decimal('0')
        
        result = {
            'total_slippage': float(adjusted_slippage),
            'slippage_bps': float(slippage_bps),
            'components': components,
            'correlation_adjustment': float(self.correlation_adjustment),
            'raw_total': float(total_slippage),
            'errors': errors,
            'calculation_details': self._get_calculation_details(request, market_conditions, execution_delay)
        }
        
        self.logger.debug(
            f"Total slippage estimated: {adjusted_slippage:.2f} "
            f"({slippage_bps:.1f} bps) for {request.symbol}"
        )
        
        return result
    
    def estimate_component_slippage(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions] = None,
        execution_delay: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Estimate slippage components separately without combination.
        
        Args:
            request: Transaction request details
            market_conditions: Current market conditions
            execution_delay: Expected execution delay
            
        Returns:
            Individual component slippage estimates
        """
        components = {}
        
        # Delay slippage
        if self.delay_model:
            try:
                delay_result = self.delay_model.calculate_slippage(
                    request, market_conditions, execution_delay
                )
                components['delay'] = {
                    'slippage': float(delay_result),
                    'model': self.delay_model.get_model_description(),
                    'success': True
                }
            except Exception as e:
                components['delay'] = {
                    'slippage': 0.0,
                    'error': str(e),
                    'success': False
                }
        
        # Size slippage
        if self.size_model:
            try:
                size_result = self.size_model.calculate_slippage(
                    request, market_conditions, execution_delay
                )
                components['size'] = {
                    'slippage': float(size_result),
                    'model': self.size_model.get_model_description(),
                    'success': True
                }
            except Exception as e:
                components['size'] = {
                    'slippage': 0.0,
                    'error': str(e),
                    'success': False
                }
        
        # Condition slippage
        if self.condition_model:
            try:
                condition_result = self.condition_model.calculate_slippage(
                    request, market_conditions, execution_delay
                )
                components['condition'] = {
                    'slippage': float(condition_result),
                    'model': self.condition_model.get_model_description(),
                    'success': True
                }
            except Exception as e:
                components['condition'] = {
                    'slippage': 0.0,
                    'error': str(e),
                    'success': False
                }
        
        return {
            'components': components,
            'calculation_details': self._get_calculation_details(request, market_conditions, execution_delay)
        }
    
    def estimate_batch_slippage(
        self,
        requests: List[TransactionRequest],
        market_conditions: Optional[MarketConditions] = None,
        execution_delay: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate slippage for multiple transactions in batch.
        
        Args:
            requests: List of transaction requests
            market_conditions: Current market conditions (shared)
            execution_delay: Expected execution delay (shared)
            
        Returns:
            List of slippage estimates
        """
        results = []
        
        self.logger.info(f"Starting batch slippage estimation for {len(requests)} transactions")
        
        for i, request in enumerate(requests):
            try:
                result = self.estimate_total_slippage(request, market_conditions, execution_delay)
                result['batch_index'] = i
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(requests)} slippage estimates")
                    
            except Exception as e:
                error_result = {
                    'batch_index': i,
                    'total_slippage': 0.0,
                    'slippage_bps': 0.0,
                    'error': str(e),
                    'success': False
                }
                results.append(error_result)
                self.logger.warning(f"Failed to estimate slippage for request {i}: {e}")
        
        self.logger.info(f"Batch slippage estimation completed: {len(results)} results")
        return results
    
    def _get_calculation_details(
        self,
        request: TransactionRequest,
        market_conditions: Optional[MarketConditions],
        execution_delay: Optional[timedelta]
    ) -> Dict[str, Any]:
        """
        Get calculation details for reporting.
        
        Args:
            request: Transaction request
            market_conditions: Market conditions
            execution_delay: Execution delay
            
        Returns:
            Calculation details
        """
        details = {
            'symbol': request.symbol,
            'quantity': request.quantity,
            'price': float(request.price),
            'order_type': request.order_type.name,
            'market_timing': request.market_timing.name,
            'has_market_data': market_conditions is not None,
            'execution_delay_seconds': execution_delay.total_seconds() if execution_delay else None,
            'enabled_models': {
                'delay': self.include_delay_slippage,
                'size': self.include_size_slippage,
                'condition': self.include_condition_slippage
            }
        }
        
        if market_conditions:
            details['market_conditions'] = {
                'has_bid_ask': bool(market_conditions.bid_price and market_conditions.ask_price),
                'has_volume': bool(market_conditions.volume),
                'market_open': market_conditions.market_open,
                'data_age_seconds': (
                    (request.timestamp - market_conditions.timestamp).total_seconds()
                    if market_conditions.timestamp else None
                )
            }
        
        return details
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current estimator configuration.
        
        Returns:
            Configuration details
        """
        return {
            'include_delay_slippage': self.include_delay_slippage,
            'include_size_slippage': self.include_size_slippage,
            'include_condition_slippage': self.include_condition_slippage,
            'correlation_adjustment': float(self.correlation_adjustment),
            'component_models': {
                'delay': self.delay_model.get_model_parameters() if self.delay_model else None,
                'size': self.size_model.get_model_parameters() if self.size_model else None,
                'condition': self.condition_model.get_model_parameters() if self.condition_model else None
            }
        }