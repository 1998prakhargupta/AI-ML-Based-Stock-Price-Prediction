"""
Spread Validator
===============

Comprehensive validation system for spread data quality, 
consistency checks, and anomaly detection.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

from .base_spread_model import SpreadEstimate, SpreadData, MarketCondition

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    check_function: callable
    threshold: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of spread data validation."""
    is_valid: bool
    severity: str
    rule_name: str
    message: str
    actual_value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    confidence: float = 1.0


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_checks: int
    passed_checks: int
    warnings: int
    errors: int
    overall_score: float  # 0-1 quality score
    results: List[ValidationResult]
    timestamp: datetime


class SpreadValidator:
    """
    Comprehensive spread data validator with anomaly detection.
    
    Features:
    - Data quality validation
    - Consistency checks
    - Anomaly detection
    - Historical comparison
    - Market condition validation
    """
    
    def __init__(
        self,
        enable_statistical_checks: bool = True,
        enable_temporal_checks: bool = True,
        enable_market_checks: bool = True,
        historical_window_days: int = 30,
        outlier_threshold_sigma: float = 3.0
    ):
        """
        Initialize spread validator.
        
        Args:
            enable_statistical_checks: Enable statistical validation
            enable_temporal_checks: Enable time-based validation
            enable_market_checks: Enable market condition validation
            historical_window_days: Days of historical data to keep
            outlier_threshold_sigma: Standard deviations for outlier detection
        """
        self.enable_statistical_checks = enable_statistical_checks
        self.enable_temporal_checks = enable_temporal_checks
        self.enable_market_checks = enable_market_checks
        self.historical_window_days = historical_window_days
        self.outlier_threshold_sigma = outlier_threshold_sigma
        
        # Historical data for validation
        self._historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._market_statistics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Validation rules
        self._validation_rules: List[ValidationRule] = []
        self._initialize_validation_rules()
        
        # Performance tracking
        self._validation_count = 0
        self._validation_time_total = 0.0
        
        logger.info("Spread validator initialized")
    
    def validate_spread_data(self, spread_data: SpreadData) -> ValidationSummary:
        """
        Validate spread data comprehensively.
        
        Args:
            spread_data: Spread data to validate
            
        Returns:
            Validation summary with detailed results
        """
        start_time = datetime.now()
        results = []
        
        try:
            # Run all validation rules
            for rule in self._validation_rules:
                try:
                    result = rule.check_function(spread_data)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Validation rule {rule.name} failed: {e}")
                    results.append(ValidationResult(
                        is_valid=False,
                        severity='error',
                        rule_name=rule.name,
                        message=f"Validation rule execution failed: {e}",
                        confidence=0.0
                    ))
            
            # Statistical validation if enabled
            if self.enable_statistical_checks:
                stat_results = self._validate_statistical_properties(spread_data)
                results.extend(stat_results)
            
            # Temporal validation if enabled
            if self.enable_temporal_checks:
                temporal_results = self._validate_temporal_consistency(spread_data)
                results.extend(temporal_results)
            
            # Market condition validation if enabled
            if self.enable_market_checks:
                market_results = self._validate_market_conditions(spread_data)
                results.extend(market_results)
            
            # Calculate summary statistics
            total_checks = len(results)
            passed_checks = sum(1 for r in results if r.is_valid)
            warnings = sum(1 for r in results if r.severity == 'warning')
            errors = sum(1 for r in results if r.severity == 'error')
            
            # Calculate overall quality score
            if total_checks > 0:
                error_penalty = errors / total_checks
                warning_penalty = warnings / total_checks * 0.5
                overall_score = max(0.0, 1.0 - error_penalty - warning_penalty)
            else:
                overall_score = 1.0
            
            # Store data for future validation
            self._store_historical_data(spread_data)
            
            # Update performance metrics
            validation_time = (datetime.now() - start_time).total_seconds()
            self._validation_count += 1
            self._validation_time_total += validation_time
            
            return ValidationSummary(
                total_checks=total_checks,
                passed_checks=passed_checks,
                warnings=warnings,
                errors=errors,
                overall_score=overall_score,
                results=results,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {spread_data.symbol}: {e}")
            return ValidationSummary(
                total_checks=1,
                passed_checks=0,
                warnings=0,
                errors=1,
                overall_score=0.0,
                results=[ValidationResult(
                    is_valid=False,
                    severity='error',
                    rule_name='validation_system',
                    message=f"Validation system error: {e}",
                    confidence=0.0
                )],
                timestamp=datetime.now()
            )
    
    def validate_spread_estimate(self, estimate: SpreadEstimate) -> ValidationSummary:
        """
        Validate spread estimate.
        
        Args:
            estimate: Spread estimate to validate
            
        Returns:
            Validation summary
        """
        # Convert estimate to spread data for validation
        spread_data = SpreadData(
            symbol=estimate.symbol,
            spread=estimate.estimated_spread,
            spread_bps=estimate.spread_bps,
            timestamp=estimate.timestamp
        )
        
        # Validate as spread data
        summary = self.validate_spread_data(spread_data)
        
        # Additional estimate-specific validations
        estimate_results = self._validate_estimate_specific(estimate)
        summary.results.extend(estimate_results)
        
        # Recalculate summary statistics
        summary.total_checks = len(summary.results)
        summary.passed_checks = sum(1 for r in summary.results if r.is_valid)
        summary.warnings = sum(1 for r in summary.results if r.severity == 'warning')
        summary.errors = sum(1 for r in summary.results if r.severity == 'error')
        
        if summary.total_checks > 0:
            error_penalty = summary.errors / summary.total_checks
            warning_penalty = summary.warnings / summary.total_checks * 0.5
            summary.overall_score = max(0.0, 1.0 - error_penalty - warning_penalty)
        
        return summary
    
    def get_data_quality_score(self, symbol: str) -> float:
        """
        Get overall data quality score for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Quality score (0-1)
        """
        if symbol not in self._historical_data:
            return 0.5  # Neutral score for no data
        
        historical_data = list(self._historical_data[symbol])
        if not historical_data:
            return 0.5
        
        # Validate recent data points
        recent_data = historical_data[-min(100, len(historical_data)):]
        
        quality_scores = []
        for data in recent_data:
            summary = self.validate_spread_data(data)
            quality_scores.append(summary.overall_score)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    
    def detect_anomalies(
        self,
        symbol: str,
        lookback_periods: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in spread data.
        
        Args:
            symbol: Trading symbol
            lookback_periods: Number of recent periods to analyze
            
        Returns:
            List of detected anomalies
        """
        if symbol not in self._historical_data:
            return []
        
        historical_data = list(self._historical_data[symbol])
        if len(historical_data) < 20:  # Need minimum data
            return []
        
        # Get recent data
        recent_data = historical_data[-lookback_periods:] if len(historical_data) > lookback_periods else historical_data
        
        anomalies = []
        
        # Analyze spread values
        spreads = [float(d.spread) for d in recent_data if d.spread]
        if len(spreads) >= 10:
            mean_spread = statistics.mean(spreads)
            std_spread = statistics.stdev(spreads)
            
            for i, data in enumerate(recent_data):
                if data.spread:
                    z_score = abs(float(data.spread) - mean_spread) / max(std_spread, 0.0001)
                    
                    if z_score > self.outlier_threshold_sigma:
                        anomalies.append({
                            'type': 'spread_outlier',
                            'timestamp': data.timestamp,
                            'value': float(data.spread),
                            'z_score': z_score,
                            'expected_range': (mean_spread - 2*std_spread, mean_spread + 2*std_spread),
                            'severity': 'high' if z_score > 4 else 'medium'
                        })
        
        # Analyze spread changes (jumps)
        if len(recent_data) >= 2:
            for i in range(1, len(recent_data)):
                prev_data = recent_data[i-1]
                curr_data = recent_data[i]
                
                if prev_data.spread and curr_data.spread:
                    change_pct = abs(float(curr_data.spread - prev_data.spread)) / max(float(prev_data.spread), 0.0001)
                    
                    if change_pct > 0.5:  # 50% change threshold
                        anomalies.append({
                            'type': 'spread_jump',
                            'timestamp': curr_data.timestamp,
                            'change_percentage': change_pct * 100,
                            'previous_value': float(prev_data.spread),
                            'current_value': float(curr_data.spread),
                            'severity': 'high' if change_pct > 1.0 else 'medium'
                        })
        
        return anomalies
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        avg_validation_time = (
            self._validation_time_total / self._validation_count
            if self._validation_count > 0 else 0.0
        )
        
        # Calculate quality statistics across all symbols
        symbol_scores = []
        for symbol in self._historical_data.keys():
            score = self.get_data_quality_score(symbol)
            symbol_scores.append(score)
        
        avg_quality_score = sum(symbol_scores) / len(symbol_scores) if symbol_scores else 0.0
        
        return {
            'total_validations': self._validation_count,
            'average_validation_time_seconds': avg_validation_time,
            'tracked_symbols': len(self._historical_data),
            'average_quality_score': avg_quality_score,
            'validation_rules_count': len(self._validation_rules)
        }
    
    # Private methods
    
    def _initialize_validation_rules(self) -> None:
        """Initialize standard validation rules."""
        
        # Basic data integrity rules
        self._validation_rules.extend([
            ValidationRule(
                name="non_null_symbol",
                description="Symbol must not be null or empty",
                severity="error",
                check_function=self._check_non_null_symbol
            ),
            ValidationRule(
                name="valid_spread_positive",
                description="Spread must be positive",
                severity="error",
                check_function=self._check_positive_spread
            ),
            ValidationRule(
                name="valid_timestamp",
                description="Timestamp must be valid and recent",
                severity="warning",
                check_function=self._check_valid_timestamp
            ),
            ValidationRule(
                name="reasonable_spread_magnitude",
                description="Spread should be within reasonable bounds",
                severity="warning",
                check_function=self._check_reasonable_spread,
                threshold=0.1  # 10% max spread
            ),
            ValidationRule(
                name="bid_ask_consistency",
                description="Ask price must be greater than bid price",
                severity="error",
                check_function=self._check_bid_ask_consistency
            )
        ])
    
    def _check_non_null_symbol(self, data: SpreadData) -> Optional[ValidationResult]:
        """Check that symbol is not null or empty."""
        if not data.symbol or not data.symbol.strip():
            return ValidationResult(
                is_valid=False,
                severity='error',
                rule_name='non_null_symbol',
                message="Symbol is null or empty",
                confidence=1.0
            )
        return ValidationResult(
            is_valid=True,
            severity='info',
            rule_name='non_null_symbol',
            message="Symbol is valid",
            confidence=1.0
        )
    
    def _check_positive_spread(self, data: SpreadData) -> Optional[ValidationResult]:
        """Check that spread is positive."""
        if data.spread is not None:
            if data.spread <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity='error',
                    rule_name='valid_spread_positive',
                    message=f"Spread is not positive: {data.spread}",
                    actual_value=float(data.spread),
                    confidence=1.0
                )
        return ValidationResult(
            is_valid=True,
            severity='info',
            rule_name='valid_spread_positive',
            message="Spread is positive",
            confidence=1.0
        )
    
    def _check_valid_timestamp(self, data: SpreadData) -> Optional[ValidationResult]:
        """Check that timestamp is valid and recent."""
        if data.timestamp is None:
            return ValidationResult(
                is_valid=False,
                severity='warning',
                rule_name='valid_timestamp',
                message="Timestamp is missing",
                confidence=1.0
            )
        
        # Check if timestamp is too old (more than 1 day)
        age = (datetime.now() - data.timestamp).total_seconds()
        if age > 86400:  # 24 hours
            return ValidationResult(
                is_valid=False,
                severity='warning',
                rule_name='valid_timestamp',
                message=f"Timestamp is too old: {age/3600:.1f} hours",
                actual_value=age,
                confidence=0.8
            )
        
        # Check if timestamp is in the future
        if data.timestamp > datetime.now() + timedelta(minutes=5):
            return ValidationResult(
                is_valid=False,
                severity='warning',
                rule_name='valid_timestamp',
                message="Timestamp is in the future",
                confidence=1.0
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            rule_name='valid_timestamp',
            message="Timestamp is valid",
            confidence=1.0
        )
    
    def _check_reasonable_spread(self, data: SpreadData) -> Optional[ValidationResult]:
        """Check that spread is within reasonable bounds."""
        if data.spread is None:
            return None
        
        # Check spread as percentage of price
        if data.bid_price and data.ask_price:
            mid_price = (data.bid_price + data.ask_price) / 2
            spread_pct = float(data.spread / mid_price)
            
            if spread_pct > 0.1:  # 10% spread seems excessive
                return ValidationResult(
                    is_valid=False,
                    severity='warning',
                    rule_name='reasonable_spread_magnitude',
                    message=f"Spread is unusually large: {spread_pct*100:.2f}%",
                    actual_value=spread_pct,
                    expected_range=(0.0, 0.1),
                    confidence=0.8
                )
        
        # Check absolute spread value
        if float(data.spread) > 10.0:  # Arbitrary large spread threshold
            return ValidationResult(
                is_valid=False,
                severity='warning',
                rule_name='reasonable_spread_magnitude',
                message=f"Spread value is very large: {data.spread}",
                actual_value=float(data.spread),
                confidence=0.9
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            rule_name='reasonable_spread_magnitude',
            message="Spread magnitude is reasonable",
            confidence=1.0
        )
    
    def _check_bid_ask_consistency(self, data: SpreadData) -> Optional[ValidationResult]:
        """Check bid-ask price consistency."""
        if data.bid_price is not None and data.ask_price is not None:
            if data.ask_price <= data.bid_price:
                return ValidationResult(
                    is_valid=False,
                    severity='error',
                    rule_name='bid_ask_consistency',
                    message=f"Ask price ({data.ask_price}) <= bid price ({data.bid_price})",
                    confidence=1.0
                )
            
            # Check if calculated spread matches provided spread
            if data.spread is not None:
                calculated_spread = data.ask_price - data.bid_price
                spread_diff = abs(calculated_spread - data.spread)
                
                if spread_diff > Decimal('0.0001'):  # Small tolerance for rounding
                    return ValidationResult(
                        is_valid=False,
                        severity='warning',
                        rule_name='bid_ask_consistency',
                        message=f"Calculated spread ({calculated_spread}) != provided spread ({data.spread})",
                        confidence=0.9
                    )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            rule_name='bid_ask_consistency',
            message="Bid-ask prices are consistent",
            confidence=1.0
        )
    
    def _validate_statistical_properties(self, data: SpreadData) -> List[ValidationResult]:
        """Validate statistical properties against historical data."""
        results = []
        
        if data.symbol not in self._historical_data or data.spread is None:
            return results
        
        historical_data = list(self._historical_data[data.symbol])
        if len(historical_data) < 10:
            return results
        
        # Get historical spreads
        historical_spreads = [float(d.spread) for d in historical_data if d.spread]
        
        if len(historical_spreads) < 10:
            return results
        
        # Statistical tests
        mean_spread = statistics.mean(historical_spreads)
        std_spread = statistics.stdev(historical_spreads)
        current_spread = float(data.spread)
        
        # Z-score test
        z_score = abs(current_spread - mean_spread) / max(std_spread, 0.0001)
        
        if z_score > self.outlier_threshold_sigma:
            results.append(ValidationResult(
                is_valid=False,
                severity='warning' if z_score < 4 else 'error',
                rule_name='statistical_outlier',
                message=f"Spread is statistical outlier: z-score = {z_score:.2f}",
                actual_value=current_spread,
                expected_range=(mean_spread - 2*std_spread, mean_spread + 2*std_spread),
                confidence=min(1.0, z_score / 10.0)
            ))
        
        return results
    
    def _validate_temporal_consistency(self, data: SpreadData) -> List[ValidationResult]:
        """Validate temporal consistency."""
        results = []
        
        if data.symbol not in self._historical_data or data.spread is None:
            return results
        
        historical_data = list(self._historical_data[data.symbol])
        if len(historical_data) < 2:
            return results
        
        # Check for sudden jumps
        latest_data = historical_data[-1]
        if latest_data.spread:
            spread_change = abs(float(data.spread) - float(latest_data.spread))
            spread_change_pct = spread_change / max(float(latest_data.spread), 0.0001)
            
            if spread_change_pct > 0.5:  # 50% change threshold
                results.append(ValidationResult(
                    is_valid=False,
                    severity='warning',
                    rule_name='temporal_jump',
                    message=f"Large spread change: {spread_change_pct*100:.1f}%",
                    actual_value=spread_change_pct,
                    confidence=0.8
                ))
        
        return results
    
    def _validate_market_conditions(self, data: SpreadData) -> List[ValidationResult]:
        """Validate data against market conditions."""
        results = []
        
        # Check market hours (basic check)
        if data.timestamp:
            hour = data.timestamp.hour
            
            # Check if during typical trading hours (9 AM - 4 PM)
            if hour < 9 or hour > 16:
                results.append(ValidationResult(
                    is_valid=True,  # Not invalid, just informational
                    severity='info',
                    rule_name='market_hours',
                    message=f"Data timestamp outside normal trading hours: {hour}:00",
                    confidence=0.7
                ))
        
        return results
    
    def _validate_estimate_specific(self, estimate: SpreadEstimate) -> List[ValidationResult]:
        """Validate estimate-specific properties."""
        results = []
        
        # Check confidence level
        if estimate.confidence_level < 0 or estimate.confidence_level > 1:
            results.append(ValidationResult(
                is_valid=False,
                severity='error',
                rule_name='confidence_range',
                message=f"Confidence level outside valid range: {estimate.confidence_level}",
                actual_value=estimate.confidence_level,
                expected_range=(0.0, 1.0),
                confidence=1.0
            ))
        
        # Check estimation method
        if not estimate.estimation_method:
            results.append(ValidationResult(
                is_valid=False,
                severity='warning',
                rule_name='estimation_method',
                message="Estimation method not specified",
                confidence=1.0
            ))
        
        # Check spread vs spread_bps consistency
        if estimate.estimated_spread and estimate.spread_bps:
            # This would require mid price to validate properly
            # For now, just check that both are positive
            if float(estimate.estimated_spread) <= 0 or float(estimate.spread_bps) <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity='warning',
                    rule_name='spread_bps_consistency',
                    message="Spread or spread_bps is not positive",
                    confidence=0.8
                ))
        
        return results
    
    def _store_historical_data(self, data: SpreadData) -> None:
        """Store data for historical validation."""
        if data.symbol and data.spread:
            self._historical_data[data.symbol].append(data)


logger.info("Spread validator class loaded successfully")