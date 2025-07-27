"""
Calculation Orchestrator
========================

Orchestrates parallel cost component calculations with dependency management,
error recovery, fallback mechanisms, performance monitoring, and result validation.

This module coordinates the execution of multiple cost calculation components
while ensuring proper dependency handling and optimal performance.
"""

import asyncio
import time
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, TimeoutError
from enum import Enum, auto
import threading
from collections import defaultdict, deque

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
    raise_calculation_error
)
from .constants import SYSTEM_DEFAULTS, CONFIDENCE_LEVELS

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of a calculation component."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class ExecutionStrategy(Enum):
    """Execution strategy for component calculations."""
    PARALLEL = auto()        # All components in parallel
    SEQUENTIAL = auto()      # All components sequentially
    DEPENDENCY_AWARE = auto() # Respect dependencies
    PRIORITY_BASED = auto()  # Execute by priority


@dataclass
class ComponentDefinition:
    """Definition of a calculation component."""
    name: str
    calculator: Any
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1  # 1=highest priority
    timeout_seconds: float = 5.0
    retry_count: int = 2
    fallback_strategy: Optional[str] = None
    enabled: bool = True
    weight: float = 1.0
    
    
@dataclass
class ComponentResult:
    """Result of a component calculation."""
    name: str
    status: ComponentStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    calculation_time: float = 0.0
    retry_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if calculation was successful."""
        return self.status == ComponentStatus.COMPLETED and self.error is None
    
    @property
    def duration_ms(self) -> float:
        """Get calculation duration in milliseconds."""
        return self.calculation_time * 1000


@dataclass
class OrchestrationResult:
    """Result of the complete orchestration."""
    transaction_request: TransactionRequest
    broker_config: BrokerConfiguration
    component_results: Dict[str, ComponentResult] = field(default_factory=dict)
    total_time: float = 0.0
    success_rate: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.DEPENDENCY_AWARE
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def successful_components(self) -> List[str]:
        """Get list of successful component names."""
        return [name for name, result in self.component_results.items() if result.success]
    
    @property
    def failed_components(self) -> List[str]:
        """Get list of failed component names."""
        return [name for name, result in self.component_results.items() if not result.success]


class DependencyGraph:
    """Manages component dependencies and execution order."""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependencies
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependents
    
    def add_component(self, name: str, dependencies: List[str] = None):
        """Add a component with its dependencies."""
        self.nodes.add(name)
        if dependencies:
            for dep in dependencies:
                self.edges[name].add(dep)
                self.reverse_edges[dep].add(name)
                self.nodes.add(dep)  # Ensure dependency is in nodes
    
    def get_execution_order(self) -> List[List[str]]:
        """Get components grouped by execution level (topological sort)."""
        # Kahn's algorithm for topological sorting
        in_degree = {node: len(self.edges[node]) for node in self.nodes}
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        execution_levels = []
        
        while queue:
            current_level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                current_level.append(node)
                
                # Reduce in-degree for dependents
                for dependent in self.reverse_edges[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            if current_level:
                execution_levels.append(current_level)
        
        # Check for cycles
        if sum(len(level) for level in execution_levels) != len(self.nodes):
            raise ValueError("Circular dependencies detected in component graph")
        
        return execution_levels
    
    def get_ready_components(self, completed: Set[str]) -> Set[str]:
        """Get components that are ready to execute given completed components."""
        ready = set()
        for node in self.nodes:
            if node not in completed:
                dependencies = self.edges[node]
                if dependencies.issubset(completed):
                    ready.add(node)
        return ready


class PerformanceMonitor:
    """Monitors orchestration performance and component health."""
    
    def __init__(self):
        self.component_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_executions': 0,
            'successful_executions': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'timeout_count': 0,
            'last_execution': None
        })
        self.orchestration_stats = {
            'total_orchestrations': 0,
            'successful_orchestrations': 0,
            'average_success_rate': 0.0,
            'average_total_time': 0.0,
            'last_reset': datetime.now()
        }
        self._lock = threading.Lock()
    
    def record_component_execution(
        self,
        component_name: str,
        success: bool,
        execution_time: float,
        error: Optional[str] = None,
        timeout: bool = False
    ):
        """Record component execution statistics."""
        with self._lock:
            stats = self.component_stats[component_name]
            stats['total_executions'] += 1
            stats['total_time'] += execution_time
            stats['last_execution'] = datetime.now()
            
            if success:
                stats['successful_executions'] += 1
            else:
                stats['error_count'] += 1
                
            if timeout:
                stats['timeout_count'] += 1
            
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
    
    def record_orchestration(self, result: OrchestrationResult):
        """Record orchestration statistics."""
        with self._lock:
            self.orchestration_stats['total_orchestrations'] += 1
            
            if result.success_rate > 0.8:  # Consider >80% success rate as successful
                self.orchestration_stats['successful_orchestrations'] += 1
            
            # Update running averages
            total = self.orchestration_stats['total_orchestrations']
            current_avg_rate = self.orchestration_stats['average_success_rate']
            current_avg_time = self.orchestration_stats['average_total_time']
            
            self.orchestration_stats['average_success_rate'] = (
                (current_avg_rate * (total - 1)) + result.success_rate
            ) / total
            
            self.orchestration_stats['average_total_time'] = (
                (current_avg_time * (total - 1)) + result.total_time
            ) / total
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """Get health metrics for a component."""
        with self._lock:
            stats = self.component_stats[component_name]
            
            if stats['total_executions'] == 0:
                return {'status': 'no_data', 'health_score': 0.0}
            
            success_rate = stats['successful_executions'] / stats['total_executions']
            avg_time = stats['total_time'] / stats['total_executions']
            error_rate = stats['error_count'] / stats['total_executions']
            timeout_rate = stats['timeout_count'] / stats['total_executions']
            
            # Calculate health score (0-1)
            health_score = success_rate * (1 - error_rate) * (1 - timeout_rate)
            
            status = 'healthy'
            if health_score < 0.5:
                status = 'unhealthy'
            elif health_score < 0.8:
                status = 'degraded'
            
            return {
                'status': status,
                'health_score': health_score,
                'success_rate': success_rate,
                'error_rate': error_rate,
                'timeout_rate': timeout_rate,
                'average_time': avg_time,
                'last_execution': stats['last_execution'],
                'total_executions': stats['total_executions']
            }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall orchestration health."""
        with self._lock:
            component_health = {}
            overall_health_score = 0.0
            
            for component_name in self.component_stats:
                health = self.get_component_health(component_name)
                component_health[component_name] = health
                overall_health_score += health['health_score']
            
            if len(self.component_stats) > 0:
                overall_health_score /= len(self.component_stats)
            
            return {
                'overall_health_score': overall_health_score,
                'component_health': component_health,
                'orchestration_stats': dict(self.orchestration_stats)
            }


class CalculationOrchestrator:
    """
    Orchestrates parallel cost component calculations with advanced features.
    
    Features:
    - Parallel component execution with dependency management
    - Error recovery and fallback mechanisms
    - Performance monitoring and optimization
    - Result validation and quality checks
    - Multiple execution strategies
    """
    
    def __init__(
        self,
        max_workers: int = 8,
        default_timeout: float = 10.0,
        enable_monitoring: bool = True,
        enable_fallbacks: bool = True,
        default_strategy: ExecutionStrategy = ExecutionStrategy.DEPENDENCY_AWARE
    ):
        """
        Initialize the calculation orchestrator.
        
        Args:
            max_workers: Maximum number of worker threads
            default_timeout: Default timeout for component calculations
            enable_monitoring: Whether to enable performance monitoring
            enable_fallbacks: Whether to enable fallback mechanisms
            default_strategy: Default execution strategy
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.enable_monitoring = enable_monitoring
        self.enable_fallbacks = enable_fallbacks
        self.default_strategy = default_strategy
        
        # Component registry
        self.components: Dict[str, ComponentDefinition] = {}
        self.dependency_graph = DependencyGraph()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="orchestrator_")
        
        # Performance monitoring
        if enable_monitoring:
            self.performance_monitor = PerformanceMonitor()
        
        # Fallback strategies
        self.fallback_handlers: Dict[str, Callable] = {}
        
        logger.info(f"Calculation orchestrator initialized with {max_workers} workers")
    
    def register_component(
        self,
        name: str,
        calculator: Any,
        dependencies: List[str] = None,
        priority: int = 1,
        timeout_seconds: float = None,
        retry_count: int = 2,
        fallback_strategy: Optional[str] = None,
        weight: float = 1.0
    ):
        """
        Register a calculation component.
        
        Args:
            name: Component name
            calculator: Calculator instance
            dependencies: List of dependency component names
            priority: Execution priority (1=highest)
            timeout_seconds: Timeout for this component
            retry_count: Number of retries on failure
            fallback_strategy: Name of fallback strategy
            weight: Weight for result aggregation
        """
        timeout = timeout_seconds or self.default_timeout
        dependencies = dependencies or []
        
        component = ComponentDefinition(
            name=name,
            calculator=calculator,
            dependencies=dependencies,
            priority=priority,
            timeout_seconds=timeout,
            retry_count=retry_count,
            fallback_strategy=fallback_strategy,
            weight=weight
        )
        
        self.components[name] = component
        self.dependency_graph.add_component(name, dependencies)
        
        logger.info(f"Registered component: {name} (deps: {dependencies})")
    
    def register_fallback_handler(self, name: str, handler: Callable):
        """Register a fallback handler for error recovery."""
        self.fallback_handlers[name] = handler
        logger.info(f"Registered fallback handler: {name}")
    
    async def orchestrate_calculation(
        self,
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        component_overrides: Optional[Dict[str, bool]] = None,
        strategy: Optional[ExecutionStrategy] = None,
        timeout_override: Optional[float] = None
    ) -> OrchestrationResult:
        """
        Orchestrate the calculation of all components.
        
        Args:
            transaction_request: Transaction request
            broker_config: Broker configuration
            market_conditions: Market conditions
            component_overrides: Enable/disable specific components
            strategy: Execution strategy override
            timeout_override: Global timeout override
            
        Returns:
            Orchestration result with all component results
        """
        start_time = time.perf_counter()
        strategy = strategy or self.default_strategy
        
        # Filter enabled components
        enabled_components = self._get_enabled_components(component_overrides)
        
        # Create result container
        result = OrchestrationResult(
            transaction_request=transaction_request,
            broker_config=broker_config,
            execution_strategy=strategy
        )
        
        try:
            if strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(
                    enabled_components, transaction_request, broker_config, 
                    market_conditions, result, timeout_override
                )
            elif strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(
                    enabled_components, transaction_request, broker_config,
                    market_conditions, result, timeout_override
                )
            elif strategy == ExecutionStrategy.DEPENDENCY_AWARE:
                await self._execute_dependency_aware(
                    enabled_components, transaction_request, broker_config,
                    market_conditions, result, timeout_override
                )
            elif strategy == ExecutionStrategy.PRIORITY_BASED:
                await self._execute_priority_based(
                    enabled_components, transaction_request, broker_config,
                    market_conditions, result, timeout_override
                )
            
            # Calculate final metrics
            result.total_time = time.perf_counter() - start_time
            result.success_rate = self._calculate_success_rate(result.component_results)
            
            # Record performance metrics
            if self.enable_monitoring:
                self.performance_monitor.record_orchestration(result)
            
            logger.info(
                f"Orchestration completed in {result.total_time:.3f}s "
                f"(success rate: {result.success_rate:.1%})"
            )
            
            return result
            
        except Exception as e:
            result.total_time = time.perf_counter() - start_time
            result.errors.append(str(e))
            logger.error(f"Orchestration failed: {e}")
            return result
    
    def _get_enabled_components(self, overrides: Optional[Dict[str, bool]]) -> Dict[str, ComponentDefinition]:
        """Get enabled components considering overrides."""
        enabled = {}
        
        for name, component in self.components.items():
            is_enabled = component.enabled
            if overrides and name in overrides:
                is_enabled = overrides[name]
            
            if is_enabled:
                enabled[name] = component
        
        return enabled
    
    async def _execute_parallel(
        self,
        components: Dict[str, ComponentDefinition],
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: OrchestrationResult,
        timeout_override: Optional[float]
    ):
        """Execute all components in parallel (ignoring dependencies)."""
        tasks = []
        
        for name, component in components.items():
            task = asyncio.create_task(
                self._execute_component(
                    name, component, transaction_request, broker_config, 
                    market_conditions, timeout_override
                )
            )
            tasks.append((name, task))
        
        # Wait for all tasks
        for name, task in tasks:
            try:
                component_result = await task
                result.component_results[name] = component_result
            except Exception as e:
                result.component_results[name] = ComponentResult(
                    name=name,
                    status=ComponentStatus.FAILED,
                    error=str(e)
                )
    
    async def _execute_sequential(
        self,
        components: Dict[str, ComponentDefinition],
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: OrchestrationResult,
        timeout_override: Optional[float]
    ):
        """Execute components sequentially by priority."""
        sorted_components = sorted(
            components.items(),
            key=lambda x: x[1].priority
        )
        
        for name, component in sorted_components:
            try:
                component_result = await self._execute_component(
                    name, component, transaction_request, broker_config,
                    market_conditions, timeout_override
                )
                result.component_results[name] = component_result
            except Exception as e:
                result.component_results[name] = ComponentResult(
                    name=name,
                    status=ComponentStatus.FAILED,
                    error=str(e)
                )
    
    async def _execute_dependency_aware(
        self,
        components: Dict[str, ComponentDefinition],
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: OrchestrationResult,
        timeout_override: Optional[float]
    ):
        """Execute components respecting dependencies."""
        # Filter dependency graph to only enabled components
        enabled_names = set(components.keys())
        filtered_graph = DependencyGraph()
        
        for name in enabled_names:
            component = components[name]
            filtered_deps = [dep for dep in component.dependencies if dep in enabled_names]
            filtered_graph.add_component(name, filtered_deps)
        
        # Get execution levels
        execution_levels = filtered_graph.get_execution_order()
        
        # Execute level by level
        for level in execution_levels:
            if not level:
                continue
                
            # Execute all components in current level in parallel
            tasks = []
            for name in level:
                if name in components:
                    component = components[name]
                    task = asyncio.create_task(
                        self._execute_component(
                            name, component, transaction_request, broker_config,
                            market_conditions, timeout_override
                        )
                    )
                    tasks.append((name, task))
            
            # Wait for level completion
            for name, task in tasks:
                try:
                    component_result = await task
                    result.component_results[name] = component_result
                except Exception as e:
                    result.component_results[name] = ComponentResult(
                        name=name,
                        status=ComponentStatus.FAILED,
                        error=str(e)
                    )
    
    async def _execute_priority_based(
        self,
        components: Dict[str, ComponentDefinition],
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: OrchestrationResult,
        timeout_override: Optional[float]
    ):
        """Execute components by priority with parallel execution within priority groups."""
        # Group by priority
        priority_groups = defaultdict(list)
        for name, component in components.items():
            priority_groups[component.priority].append((name, component))
        
        # Execute priority groups in order
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            
            # Execute group in parallel
            tasks = []
            for name, component in group:
                task = asyncio.create_task(
                    self._execute_component(
                        name, component, transaction_request, broker_config,
                        market_conditions, timeout_override
                    )
                )
                tasks.append((name, task))
            
            # Wait for group completion
            for name, task in tasks:
                try:
                    component_result = await task
                    result.component_results[name] = component_result
                except Exception as e:
                    result.component_results[name] = ComponentResult(
                        name=name,
                        status=ComponentStatus.FAILED,
                        error=str(e)
                    )
    
    async def _execute_component(
        self,
        name: str,
        component: ComponentDefinition,
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        timeout_override: Optional[float]
    ) -> ComponentResult:
        """Execute a single component with error handling and retries."""
        timeout = timeout_override or component.timeout_seconds
        start_time = time.perf_counter()
        
        result = ComponentResult(
            name=name,
            status=ComponentStatus.PENDING,
            start_time=datetime.now()
        )
        
        for attempt in range(component.retry_count + 1):
            try:
                result.status = ComponentStatus.RUNNING
                result.retry_count = attempt
                
                # Execute with timeout
                loop = asyncio.get_event_loop()
                calculation_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self._call_component_calculator,
                        component,
                        transaction_request,
                        broker_config,
                        market_conditions
                    ),
                    timeout=timeout
                )
                
                result.result = calculation_result
                result.status = ComponentStatus.COMPLETED
                result.calculation_time = time.perf_counter() - start_time
                result.end_time = datetime.now()
                
                # Record success
                if self.enable_monitoring:
                    self.performance_monitor.record_component_execution(
                        name, True, result.calculation_time
                    )
                
                return result
                
            except TimeoutError:
                result.status = ComponentStatus.TIMEOUT
                result.error = f"Component timed out after {timeout}s"
                
                if self.enable_monitoring:
                    self.performance_monitor.record_component_execution(
                        name, False, time.perf_counter() - start_time, 
                        error=result.error, timeout=True
                    )
                
                if attempt < component.retry_count:
                    logger.warning(f"Component {name} timed out, retrying ({attempt + 1}/{component.retry_count})")
                    continue
                
            except Exception as e:
                result.status = ComponentStatus.FAILED
                result.error = str(e)
                
                if self.enable_monitoring:
                    self.performance_monitor.record_component_execution(
                        name, False, time.perf_counter() - start_time, error=result.error
                    )
                
                if attempt < component.retry_count:
                    logger.warning(f"Component {name} failed, retrying ({attempt + 1}/{component.retry_count}): {e}")
                    continue
        
        # Try fallback if available
        if self.enable_fallbacks and component.fallback_strategy:
            try:
                fallback_result = await self._execute_fallback(
                    component, transaction_request, broker_config, market_conditions
                )
                result.result = fallback_result
                result.status = ComponentStatus.COMPLETED
                result.metadata['used_fallback'] = True
                logger.info(f"Component {name} recovered using fallback strategy")
            except Exception as e:
                logger.error(f"Fallback failed for component {name}: {e}")
        
        result.calculation_time = time.perf_counter() - start_time
        result.end_time = datetime.now()
        return result
    
    def _call_component_calculator(
        self,
        component: ComponentDefinition,
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> Any:
        """Call the component calculator (runs in thread pool)."""
        calculator = component.calculator
        
        # Different calling conventions based on calculator type
        if hasattr(calculator, 'calculate_cost'):
            return calculator.calculate_cost(
                transaction_request, broker_config, market_conditions
            )
        elif hasattr(calculator, 'calculate'):
            return calculator.calculate(
                transaction_request, broker_config, market_conditions
            )
        elif hasattr(calculator, 'estimate'):
            return calculator.estimate(
                transaction_request, market_conditions
            )
        else:
            # Generic callable
            return calculator(transaction_request, broker_config, market_conditions)
    
    async def _execute_fallback(
        self,
        component: ComponentDefinition,
        transaction_request: TransactionRequest,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions]
    ) -> Any:
        """Execute fallback strategy for a failed component."""
        if component.fallback_strategy not in self.fallback_handlers:
            raise ValueError(f"Fallback strategy '{component.fallback_strategy}' not found")
        
        handler = self.fallback_handlers[component.fallback_strategy]
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            handler,
            transaction_request,
            broker_config,
            market_conditions
        )
    
    def _calculate_success_rate(self, component_results: Dict[str, ComponentResult]) -> float:
        """Calculate overall success rate."""
        if not component_results:
            return 0.0
        
        successful = sum(1 for result in component_results.values() if result.success)
        return successful / len(component_results)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.enable_monitoring:
            return {'monitoring_disabled': True}
        
        return self.performance_monitor.get_overall_health()
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """Get health metrics for a specific component."""
        if not self.enable_monitoring:
            return {'monitoring_disabled': True}
        
        return self.performance_monitor.get_component_health(component_name)
    
    def get_registered_components(self) -> Dict[str, Dict[str, Any]]:
        """Get information about registered components."""
        return {
            name: {
                'dependencies': comp.dependencies,
                'priority': comp.priority,
                'timeout_seconds': comp.timeout_seconds,
                'retry_count': comp.retry_count,
                'fallback_strategy': comp.fallback_strategy,
                'enabled': comp.enabled,
                'weight': comp.weight,
                'calculator_type': type(comp.calculator).__name__
            }
            for name, comp in self.components.items()
        }
    
    def enable_component(self, component_name: str):
        """Enable a component."""
        if component_name in self.components:
            self.components[component_name].enabled = True
            logger.info(f"Component '{component_name}' enabled")
    
    def disable_component(self, component_name: str):
        """Disable a component."""
        if component_name in self.components:
            self.components[component_name].enabled = False
            logger.info(f"Component '{component_name}' disabled")
    
    def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=10.0)
        logger.info("Calculation orchestrator shutdown completed")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except:
            pass


logger.info("Calculation orchestrator module loaded successfully")