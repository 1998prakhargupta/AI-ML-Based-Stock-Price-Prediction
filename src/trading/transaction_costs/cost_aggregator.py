"""
Cost Aggregator
===============

Central cost aggregation system that orchestrates all individual cost calculation 
components, provides total cost breakdowns, and handles calculation errors gracefully.

This module coordinates between all cost calculation components including:
- Broker commissions and fees
- Market impact costs  
- Bid-ask spread costs
- Slippage estimates
- Regulatory fees
- Platform and data fees
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Import existing components
from .models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    InstrumentType,
    TransactionType
)
from .exceptions import (
    TransactionCostError,
    CalculationError,
    BrokerConfigurationError,
    raise_calculation_error
)
from .constants import SYSTEM_DEFAULTS, CONFIDENCE_LEVELS

# Import existing cost calculators
from .brokers.broker_factory import BrokerFactory
from .market_impact.adaptive_model import AdaptiveImpactModel
from .slippage.slippage_estimator import SlippageEstimator
from .spreads.realtime_estimator import RealTimeSpreadEstimator

logger = logging.getLogger(__name__)


@dataclass
class CostComponent:
    """Represents a single cost component calculation."""
    name: str
    calculator: Any
    enabled: bool = True
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 5
    
    
@dataclass
class AggregationResult:
    """Result of cost aggregation with detailed breakdown."""
    cost_breakdown: TransactionCostBreakdown
    component_results: Dict[str, Any] = field(default_factory=dict)
    calculation_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_score: float = 1.0


class CostAggregator:
    """
    Central cost aggregation engine that coordinates all cost calculation components.
    
    Features:
    - Orchestrates multiple cost calculators
    - Applies cost correlations and adjustments
    - Provides comprehensive cost breakdowns
    - Handles calculation errors gracefully
    - Supports both sync and async operations
    """
    
    def __init__(
        self,
        broker_factory: Optional[BrokerFactory] = None,
        enable_market_impact: bool = True,
        enable_slippage: bool = True,
        enable_spreads: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize the cost aggregator.
        
        Args:
            broker_factory: Factory for broker calculators
            enable_market_impact: Whether to include market impact calculations
            enable_slippage: Whether to include slippage calculations  
            enable_spreads: Whether to include spread calculations
            max_workers: Maximum number of parallel workers
        """
        self.broker_factory = broker_factory or BrokerFactory()
        self.max_workers = max_workers
        
        # Initialize cost components
        self.components: Dict[str, CostComponent] = {}
        self._setup_components(enable_market_impact, enable_slippage, enable_spreads)
        
        # Thread pool for parallel calculations
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cost_agg_")
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.error_count = 0
        
        # Cost correlation adjustments (can be configured)
        self.cost_correlations: Dict[str, Dict[str, float]] = {
            'market_impact': {'slippage': -0.3},  # Market impact and slippage are negatively correlated
            'slippage': {'spreads': 0.2}  # Slippage and spreads are positively correlated
        }
        
        logger.info(f"Cost aggregator initialized with {len(self.components)} components")
    
    def _setup_components(self, enable_market_impact: bool, enable_slippage: bool, enable_spreads: bool):
        """Setup available cost calculation components."""
        
        # Broker costs (always enabled)
        self.components['broker'] = CostComponent(
            name='broker',
            calculator=None,  # Will be set per calculation based on broker config
            enabled=True,
            weight=1.0,
            timeout_seconds=2
        )
        
        # Market impact costs
        if enable_market_impact:
            self.components['market_impact'] = CostComponent(
                name='market_impact',
                calculator=AdaptiveImpactModel(),
                enabled=True,
                weight=1.0,
                dependencies=['spreads'],
                timeout_seconds=5
            )
        
        # Slippage costs
        if enable_slippage:
            self.components['slippage'] = CostComponent(
                name='slippage',
                calculator=SlippageEstimator(),
                enabled=True,
                weight=1.0,
                timeout_seconds=3
            )
        
        # Spread costs
        if enable_spreads:
            self.components['spreads'] = CostComponent(
                name='spreads',
                calculator=RealTimeSpreadEstimator(),
                enabled=True,
                weight=1.0,
                timeout_seconds=3
            )
    
    def calculate_total_cost(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        component_overrides: Optional[Dict[str, bool]] = None
    ) -> AggregationResult:
        """
        Calculate total transaction cost by aggregating all components.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration
            market_conditions: Current market conditions
            component_overrides: Override enabled status for specific components
            
        Returns:
            Aggregated cost result with detailed breakdown
        """
        start_time = datetime.now()
        
        try:
            # Apply component overrides
            active_components = self._get_active_components(component_overrides)
            
            # Calculate individual cost components
            component_results = self._calculate_components(
                request, broker_config, market_conditions, active_components
            )
            
            # Aggregate results into final cost breakdown
            cost_breakdown = self._aggregate_results(
                request, component_results, market_conditions
            )
            
            # Apply cost correlations and adjustments
            self._apply_cost_adjustments(cost_breakdown, component_results)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(component_results, market_conditions)
            cost_breakdown.confidence_level = confidence_score
            
            # Compile final result
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            result = AggregationResult(
                cost_breakdown=cost_breakdown,
                component_results=component_results,
                calculation_time=calculation_time,
                confidence_score=confidence_score
            )
            
            # Update performance metrics
            self._update_metrics(calculation_time, len(component_results.get('errors', [])))
            
            logger.info(
                f"Cost aggregation completed for {request.symbol} in {calculation_time:.3f}s "
                f"(Total cost: {cost_breakdown.total_cost:.4f})"
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cost aggregation failed: {e}")
            raise CalculationError(
                f"Cost aggregation failed: {str(e)}",
                calculation_step="aggregate_costs",
                original_exception=e
            )
    
    async def calculate_total_cost_async(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        component_overrides: Optional[Dict[str, bool]] = None
    ) -> AggregationResult:
        """
        Calculate total transaction cost asynchronously.
        
        Args:
            request: Transaction request details
            broker_config: Broker configuration
            market_conditions: Current market conditions
            component_overrides: Override enabled status for specific components
            
        Returns:
            Aggregated cost result with detailed breakdown
        """
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self.calculate_total_cost,
            request,
            broker_config,
            market_conditions,
            component_overrides
        )
    
    def calculate_batch_costs(
        self,
        requests: List[TransactionRequest],
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        component_overrides: Optional[Dict[str, bool]] = None
    ) -> List[AggregationResult]:
        """
        Calculate costs for multiple transactions in parallel.
        
        Args:
            requests: List of transaction requests
            broker_config: Broker configuration
            market_conditions: Current market conditions
            component_overrides: Override enabled status for specific components
            
        Returns:
            List of aggregated cost results
        """
        if not requests:
            return []
        
        results = []
        start_time = datetime.now()
        
        logger.info(f"Starting batch cost aggregation for {len(requests)} transactions")
        
        # Submit all calculations to thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_request = {
                executor.submit(
                    self.calculate_total_cost,
                    request,
                    broker_config,
                    market_conditions,
                    component_overrides
                ): request for request in requests
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch calculation failed for {request.symbol}: {e}")
                    # Create error result
                    error_breakdown = TransactionCostBreakdown()
                    error_breakdown.cost_details = {'error': str(e)}
                    error_result = AggregationResult(
                        cost_breakdown=error_breakdown,
                        errors=[str(e)]
                    )
                    results.append(error_result)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Batch cost aggregation completed in {total_time:.2f}s "
            f"({len(requests)/total_time:.1f} calc/s)"
        )
        
        return results
    
    def _get_active_components(self, overrides: Optional[Dict[str, bool]]) -> Dict[str, CostComponent]:
        """Get active components considering overrides."""
        active = {}
        
        for name, component in self.components.items():
            enabled = component.enabled
            if overrides and name in overrides:
                enabled = overrides[name]
            
            if enabled:
                active[name] = component
        
        return active
    
    def _calculate_components(
        self,
        request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        active_components: Dict[str, CostComponent]
    ) -> Dict[str, Any]:
        """Calculate individual cost components."""
        results = {
            'costs': {},
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Calculate broker costs first (no dependencies)
        if 'broker' in active_components:
            try:
                broker_calculator = self.broker_factory.get_calculator(broker_config.broker_name)
                broker_result = broker_calculator.calculate_cost(
                    request, broker_config, market_conditions
                )
                results['costs']['broker'] = broker_result
                results['metadata']['broker'] = {
                    'calculator_name': broker_calculator.calculator_name,
                    'version': broker_calculator.version
                }
            except Exception as e:
                results['errors'].append(f"Broker calculation failed: {e}")
                logger.warning(f"Broker cost calculation failed: {e}")
        
        # Calculate other components
        for name, component in active_components.items():
            if name == 'broker':  # Already calculated
                continue
                
            try:
                if name == 'market_impact' and hasattr(component.calculator, 'calculate_impact'):
                    impact_cost = component.calculator.calculate_impact(request, market_conditions)
                    results['costs'][name] = {'market_impact_cost': impact_cost}
                    
                elif name == 'slippage' and hasattr(component.calculator, 'estimate_slippage'):
                    slippage_cost = component.calculator.estimate_slippage(request, market_conditions)
                    results['costs'][name] = {'timing_cost': slippage_cost}
                    
                elif name == 'spreads' and hasattr(component.calculator, 'estimate_spread_cost'):
                    spread_cost = component.calculator.estimate_spread_cost(request, market_conditions)
                    results['costs'][name] = {'bid_ask_spread_cost': spread_cost}
                
                results['metadata'][name] = {
                    'calculator_type': type(component.calculator).__name__,
                    'weight': component.weight
                }
                    
            except Exception as e:
                error_msg = f"{name} calculation failed: {e}"
                results['errors'].append(error_msg)
                logger.warning(error_msg)
        
        return results
    
    def _aggregate_results(
        self,
        request: TransactionRequest,
        component_results: Dict[str, Any],
        market_conditions: Optional[MarketConditions]
    ) -> TransactionCostBreakdown:
        """Aggregate individual component results into final cost breakdown."""
        breakdown = TransactionCostBreakdown()
        breakdown.calculator_version = f"CostAggregator v1.0"
        
        costs = component_results.get('costs', {})
        
        # Aggregate broker costs
        if 'broker' in costs and isinstance(costs['broker'], TransactionCostBreakdown):
            broker_breakdown = costs['broker']
            breakdown.commission = broker_breakdown.commission
            breakdown.regulatory_fees = broker_breakdown.regulatory_fees
            breakdown.exchange_fees = broker_breakdown.exchange_fees
            breakdown.platform_fees = broker_breakdown.platform_fees
            breakdown.data_fees = broker_breakdown.data_fees
            breakdown.borrowing_cost = broker_breakdown.borrowing_cost
            breakdown.overnight_financing = broker_breakdown.overnight_financing
            breakdown.currency_conversion = broker_breakdown.currency_conversion
            breakdown.miscellaneous_fees = broker_breakdown.miscellaneous_fees
        
        # Aggregate market impact costs
        if 'market_impact' in costs:
            impact_data = costs['market_impact']
            if 'market_impact_cost' in impact_data:
                breakdown.market_impact_cost = Decimal(str(impact_data['market_impact_cost']))
        
        # Aggregate slippage costs
        if 'slippage' in costs:
            slippage_data = costs['slippage']
            if 'timing_cost' in slippage_data:
                breakdown.timing_cost = Decimal(str(slippage_data['timing_cost']))
        
        # Aggregate spread costs
        if 'spreads' in costs:
            spread_data = costs['spreads']
            if 'bid_ask_spread_cost' in spread_data:
                breakdown.bid_ask_spread_cost = Decimal(str(spread_data['bid_ask_spread_cost']))
        
        # Add calculation metadata
        breakdown.cost_details = {
            'aggregator_version': '1.0',
            'components_used': list(costs.keys()),
            'errors': component_results.get('errors', []),
            'warnings': component_results.get('warnings', []),
            'component_metadata': component_results.get('metadata', {})
        }
        
        return breakdown
    
    def _apply_cost_adjustments(
        self,
        cost_breakdown: TransactionCostBreakdown,
        component_results: Dict[str, Any]
    ):
        """Apply cost correlations and adjustments."""
        costs = component_results.get('costs', {})
        
        # Apply correlation adjustments
        for component1, correlations in self.cost_correlations.items():
            if component1 not in costs:
                continue
                
            for component2, correlation in correlations.items():
                if component2 not in costs:
                    continue
                
                # Apply simple correlation adjustment
                # This is a simplified model - in practice, this would be more sophisticated
                adjustment_factor = 1.0 + (correlation * 0.1)  # Small adjustment based on correlation
                
                if component1 == 'market_impact' and hasattr(cost_breakdown, 'market_impact_cost'):
                    cost_breakdown.market_impact_cost *= Decimal(str(adjustment_factor))
                elif component1 == 'slippage' and hasattr(cost_breakdown, 'timing_cost'):
                    cost_breakdown.timing_cost *= Decimal(str(adjustment_factor))
    
    def _calculate_confidence_score(
        self,
        component_results: Dict[str, Any],
        market_conditions: Optional[MarketConditions]
    ) -> float:
        """Calculate overall confidence score for the aggregated result."""
        base_confidence = 1.0
        
        # Reduce confidence based on errors
        error_count = len(component_results.get('errors', []))
        if error_count > 0:
            base_confidence *= (1.0 - min(error_count * 0.1, 0.5))
        
        # Reduce confidence based on missing market data
        if not market_conditions:
            base_confidence *= 0.7
        elif market_conditions:
            # Check data freshness
            data_age = (datetime.now() - market_conditions.timestamp).total_seconds()
            if data_age > 300:  # 5 minutes
                base_confidence *= 0.8
            elif data_age > 60:  # 1 minute
                base_confidence *= 0.9
        
        # Adjust based on component availability
        total_components = len(self.components)
        successful_components = len(component_results.get('costs', {}))
        component_ratio = successful_components / total_components if total_components > 0 else 0
        base_confidence *= component_ratio
        
        return max(base_confidence, 0.1)  # Minimum 10% confidence
    
    def _update_metrics(self, calculation_time: float, error_count: int):
        """Update performance metrics."""
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        if error_count > 0:
            self.error_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = (
            self.total_calculation_time / self.calculation_count 
            if self.calculation_count > 0 else 0.0
        )
        
        error_rate = (
            self.error_count / self.calculation_count 
            if self.calculation_count > 0 else 0.0
        )
        
        return {
            'total_calculations': self.calculation_count,
            'average_calculation_time': avg_time,
            'total_calculation_time': self.total_calculation_time,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'components_enabled': {name: comp.enabled for name, comp in self.components.items()}
        }
    
    def get_supported_components(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported components."""
        return {
            name: {
                'enabled': comp.enabled,
                'weight': comp.weight,
                'dependencies': comp.dependencies,
                'timeout_seconds': comp.timeout_seconds,
                'calculator_type': type(comp.calculator).__name__ if comp.calculator else None
            }
            for name, comp in self.components.items()
        }
    
    def configure_component(self, component_name: str, **kwargs):
        """Configure a specific component."""
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found")
        
        component = self.components[component_name]
        
        if 'enabled' in kwargs:
            component.enabled = kwargs['enabled']
        if 'weight' in kwargs:
            component.weight = kwargs['weight']
        if 'timeout_seconds' in kwargs:
            component.timeout_seconds = kwargs['timeout_seconds']
        
        logger.info(f"Component '{component_name}' configuration updated")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


logger.info("Cost aggregator module loaded successfully")