"""
Result Validator
===============

Validates transaction cost calculation results for accuracy, consistency,
and reasonableness.
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

# Import existing components
from ..models import (
    TransactionRequest,
    TransactionCostBreakdown,
    MarketConditions,
    BrokerConfiguration,
    InstrumentType,
    TransactionType
)
from ..exceptions import DataValidationError

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = auto()      # Basic sanity checks
    STANDARD = auto()   # Standard validation rules
    STRICT = auto()     # Strict validation with market constraints
    PARANOID = auto()   # Maximum validation


class ValidationSeverity(Enum):
    """Validation issue severity."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ValidationIssue:
    """Validation issue found during result checking."""
    field: str
    severity: ValidationSeverity
    message: str
    current_value: Any
    expected_range: Optional[Tuple[Any, Any]] = None
    rule_name: str = ""
    suggestion: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'severity': self.severity.name,
            'message': self.message,
            'current_value': str(self.current_value),
            'expected_range': self.expected_range,
            'rule_name': self.rule_name,
            'suggestion': self.suggestion
        }


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def has_errors(self) -> bool:
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'has_errors': self.has_errors,
            'has_warnings': self.has_warnings,
            'validation_level': self.validation_level.name,
            'timestamp': self.timestamp.isoformat(),
            'issues': [issue.to_dict() for issue in self.issues],
            'issue_count_by_severity': {
                severity.name: len(self.get_issues_by_severity(severity))
                for severity in ValidationSeverity
            }
        }


class ResultValidator:
    """
    Validates transaction cost calculation results.
    
    Features:
    - Multi-level validation (basic to paranoid)
    - Market-aware validation rules
    - Cross-component consistency checks
    - Historical comparison validation
    - Custom validation rules
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_market_validation: bool = True,
        enable_historical_validation: bool = False
    ):
        """
        Initialize result validator.
        
        Args:
            validation_level: Default validation strictness
            enable_market_validation: Enable market-aware validation
            enable_historical_validation: Enable historical comparison
        """
        self.validation_level = validation_level
        self.enable_market_validation = enable_market_validation
        self.enable_historical_validation = enable_historical_validation
        
        # Historical data for validation
        self.historical_results: Dict[str, List[TransactionCostBreakdown]] = {}
        
        # Market-based validation ranges
        self.market_ranges = self._setup_market_ranges()
        
        # Custom validation rules
        self.custom_rules: List[Callable] = []
        
        logger.info(f"Result validator initialized (level: {validation_level.name})")
    
    def _setup_market_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Setup market-based validation ranges."""
        return {
            'EQUITY': {
                'commission_bps': (0.0, 50.0),      # 0-50 basis points
                'regulatory_fees_bps': (0.0, 5.0),   # 0-5 basis points
                'bid_ask_spread_bps': (0.1, 500.0),  # 1-500 basis points
                'market_impact_bps': (0.0, 100.0),   # 0-100 basis points
                'total_cost_bps': (0.1, 1000.0)      # 1-1000 basis points
            },
            'OPTION': {
                'commission_bps': (0.0, 100.0),
                'regulatory_fees_bps': (0.0, 10.0),
                'bid_ask_spread_bps': (1.0, 2000.0),
                'market_impact_bps': (0.0, 500.0),
                'total_cost_bps': (1.0, 5000.0)
            },
            'FUTURE': {
                'commission_bps': (0.0, 20.0),
                'regulatory_fees_bps': (0.0, 2.0),
                'bid_ask_spread_bps': (0.1, 100.0),
                'market_impact_bps': (0.0, 50.0),
                'total_cost_bps': (0.1, 500.0)
            }
        }
    
    def validate_result(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None,
        validation_level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate a transaction cost calculation result.
        
        Args:
            request: Original transaction request
            cost_breakdown: Calculated cost breakdown
            broker_config: Broker configuration used
            market_conditions: Market conditions at calculation time
            validation_level: Override default validation level
            
        Returns:
            Validation result with any issues found
        """
        level = validation_level or self.validation_level
        result = ValidationResult(is_valid=True, validation_level=level)
        
        try:
            # Basic validation (always performed)
            self._validate_basic_structure(cost_breakdown, result)
            self._validate_basic_math(cost_breakdown, result)
            
            if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_standard_rules(request, cost_breakdown, broker_config, result)
                
                if self.enable_market_validation and market_conditions:
                    self._validate_market_consistency(request, cost_breakdown, market_conditions, result)
            
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_strict_rules(request, cost_breakdown, broker_config, result)
                
                if self.enable_historical_validation:
                    self._validate_historical_consistency(request, cost_breakdown, result)
            
            if level == ValidationLevel.PARANOID:
                self._validate_paranoid_rules(request, cost_breakdown, broker_config, market_conditions, result)
            
            # Apply custom validation rules
            for rule in self.custom_rules:
                try:
                    rule(request, cost_breakdown, broker_config, market_conditions, result)
                except Exception as e:
                    logger.warning(f"Custom validation rule failed: {e}")
            
            # Determine overall validity
            result.is_valid = not result.has_errors
            
            # Store result for historical validation
            if self.enable_historical_validation:
                self._store_historical_result(request, cost_breakdown)
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            result.issues.append(ValidationIssue(
                field="validation_process",
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation process failed: {str(e)}",
                current_value=str(e),
                rule_name="validation_error"
            ))
            result.is_valid = False
        
        return result
    
    def _validate_basic_structure(self, cost_breakdown: TransactionCostBreakdown, result: ValidationResult):
        """Validate basic structure and data types."""
        if not isinstance(cost_breakdown, TransactionCostBreakdown):
            result.issues.append(ValidationIssue(
                field="cost_breakdown",
                severity=ValidationSeverity.CRITICAL,
                message="Cost breakdown must be TransactionCostBreakdown instance",
                current_value=type(cost_breakdown).__name__,
                rule_name="structure_validation"
            ))
            return
        
        # Check for null/negative values where inappropriate
        numeric_fields = [
            'commission', 'regulatory_fees', 'exchange_fees',
            'bid_ask_spread_cost', 'market_impact_cost', 'timing_cost',
            'borrowing_cost', 'overnight_financing', 'currency_conversion',
            'platform_fees', 'data_fees', 'miscellaneous_fees'
        ]
        
        for field in numeric_fields:
            value = getattr(cost_breakdown, field, None)
            if value is None:
                result.issues.append(ValidationIssue(
                    field=field,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field {field} cannot be None",
                    current_value=None,
                    rule_name="null_check"
                ))
            elif value < 0:
                result.issues.append(ValidationIssue(
                    field=field,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field {field} cannot be negative",
                    current_value=float(value),
                    rule_name="negative_check",
                    suggestion="Verify calculation logic"
                ))
    
    def _validate_basic_math(self, cost_breakdown: TransactionCostBreakdown, result: ValidationResult):
        """Validate basic mathematical consistency."""
        # Check total calculations
        calculated_explicit = cost_breakdown.total_explicit_costs
        calculated_implicit = cost_breakdown.total_implicit_costs
        calculated_total = cost_breakdown.total_cost
        
        expected_total = calculated_explicit + calculated_implicit
        
        # Allow small rounding differences
        tolerance = Decimal('0.0001')
        
        if abs(calculated_total - expected_total) > tolerance:
            result.issues.append(ValidationIssue(
                field="total_cost",
                severity=ValidationSeverity.ERROR,
                message="Total cost doesn't match sum of explicit and implicit costs",
                current_value=float(calculated_total),
                expected_range=(float(expected_total - tolerance), float(expected_total + tolerance)),
                rule_name="math_consistency",
                suggestion="Check total cost calculation logic"
            ))
    
    def _validate_standard_rules(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        broker_config: BrokerConfiguration,
        result: ValidationResult
    ):
        """Validate standard business rules."""
        notional = request.notional_value
        
        # Commission reasonableness
        if cost_breakdown.commission > notional * Decimal('0.1'):  # 10% of notional
            result.issues.append(ValidationIssue(
                field="commission",
                severity=ValidationSeverity.WARNING,
                message="Commission seems unusually high",
                current_value=float(cost_breakdown.commission),
                expected_range=(0.0, float(notional * Decimal('0.01'))),
                rule_name="commission_reasonableness",
                suggestion="Verify commission calculation and broker configuration"
            ))
        
        # Regulatory fees reasonableness
        if cost_breakdown.regulatory_fees > notional * Decimal('0.01'):  # 1% of notional
            result.issues.append(ValidationIssue(
                field="regulatory_fees",
                severity=ValidationSeverity.WARNING,
                message="Regulatory fees seem unusually high",
                current_value=float(cost_breakdown.regulatory_fees),
                rule_name="regulatory_fees_reasonableness"
            ))
        
        # Total cost reasonableness
        if cost_breakdown.total_cost > notional * Decimal('0.2'):  # 20% of notional
            result.issues.append(ValidationIssue(
                field="total_cost",
                severity=ValidationSeverity.WARNING,
                message="Total cost seems unusually high relative to notional value",
                current_value=float(cost_breakdown.total_cost),
                rule_name="total_cost_reasonableness",
                suggestion="Review all cost components for accuracy"
            ))
        
        # Confidence level check
        if hasattr(cost_breakdown, 'confidence_level') and cost_breakdown.confidence_level:
            if cost_breakdown.confidence_level < 0.5:
                result.issues.append(ValidationIssue(
                    field="confidence_level",
                    severity=ValidationSeverity.WARNING,
                    message="Low confidence level in calculation",
                    current_value=cost_breakdown.confidence_level,
                    rule_name="confidence_check",
                    suggestion="Check input data quality and market conditions"
                ))
    
    def _validate_market_consistency(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        market_conditions: MarketConditions,
        result: ValidationResult
    ):
        """Validate consistency with market conditions."""
        instrument_type = request.instrument_type.name
        
        if instrument_type not in self.market_ranges:
            return
        
        ranges = self.market_ranges[instrument_type]
        notional = request.notional_value
        
        # Convert costs to basis points for comparison
        cost_bps = {
            'commission': float(cost_breakdown.commission / notional * 10000),
            'regulatory_fees': float(cost_breakdown.regulatory_fees / notional * 10000),
            'bid_ask_spread': float(cost_breakdown.bid_ask_spread_cost / notional * 10000),
            'market_impact': float(cost_breakdown.market_impact_cost / notional * 10000),
            'total_cost': float(cost_breakdown.total_cost / notional * 10000)
        }
        
        # Check each cost component against market ranges
        for cost_type, bps_value in cost_bps.items():
            range_key = f"{cost_type}_bps"
            if range_key in ranges:
                min_val, max_val = ranges[range_key]
                
                if bps_value < min_val:
                    result.issues.append(ValidationIssue(
                        field=cost_type,
                        severity=ValidationSeverity.INFO,
                        message=f"{cost_type} is below typical market range",
                        current_value=bps_value,
                        expected_range=(min_val, max_val),
                        rule_name="market_range_check"
                    ))
                elif bps_value > max_val:
                    result.issues.append(ValidationIssue(
                        field=cost_type,
                        severity=ValidationSeverity.WARNING,
                        message=f"{cost_type} is above typical market range",
                        current_value=bps_value,
                        expected_range=(min_val, max_val),
                        rule_name="market_range_check",
                        suggestion="Verify calculation parameters and market conditions"
                    ))
        
        # Volume-based validation
        if market_conditions.volume and request.quantity > market_conditions.volume * 0.1:
            result.issues.append(ValidationIssue(
                field="market_impact_cost",
                severity=ValidationSeverity.WARNING,
                message="Large order relative to market volume - market impact may be underestimated",
                current_value=request.quantity,
                rule_name="volume_impact_check",
                suggestion="Consider using advanced market impact models for large orders"
            ))
    
    def _validate_strict_rules(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        broker_config: BrokerConfiguration,
        result: ValidationResult
    ):
        """Validate strict business rules."""
        # Check minimum commission rules
        if broker_config.min_commission > 0:
            if cost_breakdown.commission < broker_config.min_commission:
                result.issues.append(ValidationIssue(
                    field="commission",
                    severity=ValidationSeverity.ERROR,
                    message="Commission below broker minimum",
                    current_value=float(cost_breakdown.commission),
                    expected_range=(float(broker_config.min_commission), None),
                    rule_name="min_commission_check",
                    suggestion="Apply minimum commission from broker configuration"
                ))
        
        # Check maximum commission rules
        if broker_config.max_commission:
            if cost_breakdown.commission > broker_config.max_commission:
                result.issues.append(ValidationIssue(
                    field="commission",
                    severity=ValidationSeverity.ERROR,
                    message="Commission above broker maximum",
                    current_value=float(cost_breakdown.commission),
                    expected_range=(None, float(broker_config.max_commission)),
                    rule_name="max_commission_check",
                    suggestion="Apply maximum commission cap from broker configuration"
                ))
    
    def _validate_historical_consistency(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        result: ValidationResult
    ):
        """Validate consistency with historical patterns."""
        symbol = request.symbol
        
        if symbol not in self.historical_results:
            return
        
        historical = self.historical_results[symbol]
        if len(historical) < 5:  # Need minimum history
            return
        
        # Calculate historical statistics
        historical_costs = [float(h.total_cost) for h in historical]
        avg_cost = sum(historical_costs) / len(historical_costs)
        
        current_cost = float(cost_breakdown.total_cost)
        
        # Check for significant deviation (more than 3x average)
        if current_cost > avg_cost * 3:
            result.issues.append(ValidationIssue(
                field="total_cost",
                severity=ValidationSeverity.WARNING,
                message="Cost significantly higher than historical average",
                current_value=current_cost,
                rule_name="historical_deviation_check",
                suggestion="Verify market conditions and calculation parameters"
            ))
    
    def _validate_paranoid_rules(
        self,
        request: TransactionRequest,
        cost_breakdown: TransactionCostBreakdown,
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions],
        result: ValidationResult
    ):
        """Validate with maximum paranoia."""
        # Check for suspiciously round numbers (may indicate default values)
        fields_to_check = ['commission', 'regulatory_fees', 'market_impact_cost']
        
        for field in fields_to_check:
            value = getattr(cost_breakdown, field)
            if value > 0 and float(value) == round(float(value), 2):
                # Check if it's exactly a round number
                if float(value) in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
                    result.issues.append(ValidationIssue(
                        field=field,
                        severity=ValidationSeverity.INFO,
                        message=f"Suspiciously round value for {field}",
                        current_value=float(value),
                        rule_name="round_number_check",
                        suggestion="Verify this is not a default or placeholder value"
                    ))
        
        # Check calculation timestamp freshness
        if hasattr(cost_breakdown, 'calculation_timestamp'):
            age = (datetime.now() - cost_breakdown.calculation_timestamp).total_seconds()
            if age > 3600:  # 1 hour old
                result.issues.append(ValidationIssue(
                    field="calculation_timestamp",
                    severity=ValidationSeverity.WARNING,
                    message="Calculation result is stale",
                    current_value=cost_breakdown.calculation_timestamp.isoformat(),
                    rule_name="freshness_check",
                    suggestion="Consider recalculating with fresh data"
                ))
    
    def _store_historical_result(self, request: TransactionRequest, cost_breakdown: TransactionCostBreakdown):
        """Store result for historical validation."""
        symbol = request.symbol
        
        if symbol not in self.historical_results:
            self.historical_results[symbol] = []
        
        # Keep only recent history (last 100 results)
        self.historical_results[symbol].append(cost_breakdown)
        if len(self.historical_results[symbol]) > 100:
            self.historical_results[symbol].pop(0)
    
    def add_custom_rule(self, rule: Callable):
        """Add custom validation rule."""
        self.custom_rules.append(rule)
        logger.info(f"Added custom validation rule: {rule.__name__}")
    
    def validate_batch_results(
        self,
        requests: List[TransactionRequest],
        cost_breakdowns: List[TransactionCostBreakdown],
        broker_config: BrokerConfiguration,
        market_conditions: Optional[MarketConditions] = None
    ) -> List[ValidationResult]:
        """Validate multiple results in batch."""
        if len(requests) != len(cost_breakdowns):
            raise ValueError("Requests and cost breakdowns must have same length")
        
        results = []
        for request, breakdown in zip(requests, cost_breakdowns):
            validation_result = self.validate_result(
                request, breakdown, broker_config, market_conditions
            )
            results.append(validation_result)
        
        return results
    
    def get_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results."""
        total_results = len(validation_results)
        valid_results = sum(1 for r in validation_results if r.is_valid)
        
        severity_counts = {severity.name: 0 for severity in ValidationSeverity}
        
        for result in validation_results:
            for issue in result.issues:
                severity_counts[issue.severity.name] += 1
        
        return {
            'total_results': total_results,
            'valid_results': valid_results,
            'invalid_results': total_results - valid_results,
            'validity_rate': valid_results / total_results if total_results > 0 else 0.0,
            'issue_counts_by_severity': severity_counts,
            'total_issues': sum(severity_counts.values())
        }


logger.info("Result validator module loaded successfully")