"""
Historical Spread Analyzer
==========================

Statistical spread analysis with time-series modeling, seasonal pattern detection,
and correlation analysis between spreads, volatility, and volume.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
import asyncio

from .base_spread_model import (
    BaseSpreadModel, SpreadEstimate, SpreadData, 
    SpreadAnalysisResult, MarketCondition
)

logger = logging.getLogger(__name__)


@dataclass
class SpreadStatistics:
    """Statistical measures of spread data."""
    mean: Decimal
    median: Decimal
    std_dev: Decimal
    min_value: Decimal
    max_value: Decimal
    percentile_25: Decimal
    percentile_75: Decimal
    percentile_95: Decimal
    count: int


@dataclass
class SeasonalPattern:
    """Seasonal pattern in spread data."""
    period_type: str  # 'hourly', 'daily', 'weekly'
    pattern_data: Dict[str, Decimal]  # period -> average spread
    strength: float  # pattern strength (0-1)
    confidence: float  # confidence in pattern


@dataclass
class CorrelationAnalysis:
    """Correlation analysis results."""
    spread_volume_correlation: float
    spread_volatility_correlation: float
    spread_price_correlation: float
    analysis_period: Tuple[datetime, datetime]
    sample_size: int


class HistoricalSpreadAnalyzer(BaseSpreadModel):
    """
    Historical spread analyzer with comprehensive statistical modeling.
    
    Features:
    - Statistical spread analysis
    - Time-series spread modeling
    - Seasonal pattern detection
    - Volatility-spread relationships
    - Volume-spread correlations
    """
    
    def __init__(
        self,
        min_data_points: int = 100,
        analysis_window_days: int = 30,
        pattern_detection_threshold: float = 0.1,
        **kwargs
    ):
        """
        Initialize historical spread analyzer.
        
        Args:
            min_data_points: Minimum data points required for analysis
            analysis_window_days: Default analysis window in days
            pattern_detection_threshold: Threshold for pattern significance
        """
        super().__init__(
            model_name="HistoricalSpreadAnalyzer",
            version="1.0.0",
            **kwargs
        )
        
        self.min_data_points = min_data_points
        self.analysis_window_days = analysis_window_days
        self.pattern_detection_threshold = pattern_detection_threshold
        
        # Data storage
        self._spread_history: Dict[str, List[SpreadData]] = defaultdict(list)
        self._cached_statistics: Dict[str, Tuple[SpreadStatistics, datetime]] = {}
        self._cached_patterns: Dict[str, Tuple[List[SeasonalPattern], datetime]] = {}
        
        logger.info("Historical spread analyzer initialized")
    
    def add_historical_data(self, symbol: str, spread_data: List[SpreadData]) -> None:
        """
        Add historical spread data for analysis.
        
        Args:
            symbol: Trading symbol
            spread_data: List of historical spread data
        """
        if not spread_data:
            return
        
        # Validate and filter data
        valid_data = [data for data in spread_data if self.validate_spread_data(data)]
        
        if not valid_data:
            logger.warning(f"No valid historical data provided for {symbol}")
            return
        
        # Sort by timestamp
        valid_data.sort(key=lambda x: x.timestamp or datetime.min)
        
        # Store data
        self._spread_history[symbol].extend(valid_data)
        
        # Remove duplicates and sort
        self._spread_history[symbol] = self._deduplicate_data(self._spread_history[symbol])
        
        # Clear cached results for this symbol
        self._invalidate_cache(symbol)
        
        logger.info(f"Added {len(valid_data)} historical data points for {symbol}")
    
    def analyze_spread_statistics(self, symbol: str, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> SpreadStatistics:
        """
        Analyze spread statistics for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Statistical analysis results
        """
        # Get filtered data
        data = self._get_filtered_data(symbol, start_date, end_date)
        
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient data points for {symbol}: {len(data)} < {self.min_data_points}")
        
        # Extract spread values
        spreads = [d.spread for d in data if d.spread is not None]
        
        if not spreads:
            raise ValueError(f"No spread values found for {symbol}")
        
        # Calculate statistics
        spreads_float = [float(s) for s in spreads]
        sorted_spreads = sorted(spreads)
        
        stats = SpreadStatistics(
            mean=Decimal(str(statistics.mean(spreads_float))),
            median=Decimal(str(statistics.median(spreads_float))),
            std_dev=Decimal(str(statistics.stdev(spreads_float) if len(spreads_float) > 1 else 0)),
            min_value=min(spreads),
            max_value=max(spreads),
            percentile_25=self._calculate_percentile(sorted_spreads, 0.25),
            percentile_75=self._calculate_percentile(sorted_spreads, 0.75),
            percentile_95=self._calculate_percentile(sorted_spreads, 0.95),
            count=len(spreads)
        )
        
        return stats
    
    def detect_seasonal_patterns(self, symbol: str,
                                 period_types: Optional[List[str]] = None) -> List[SeasonalPattern]:
        """
        Detect seasonal patterns in spread data.
        
        Args:
            symbol: Trading symbol
            period_types: Types of periods to analyze
            
        Returns:
            List of detected patterns
        """
        if period_types is None:
            period_types = ['hourly', 'daily', 'weekly']
        
        # Check cache first
        cache_key = f"{symbol}_patterns"
        if cache_key in self._cached_patterns:
            patterns, cache_time = self._cached_patterns[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 3600:  # 1 hour cache
                return patterns
        
        data = self._get_filtered_data(symbol)
        
        if len(data) < self.min_data_points:
            return []
        
        patterns = []
        
        for period_type in period_types:
            pattern = self._analyze_period_pattern(data, period_type)
            if pattern and pattern.strength > self.pattern_detection_threshold:
                patterns.append(pattern)
        
        # Cache results
        self._cached_patterns[cache_key] = (patterns, datetime.now())
        
        return patterns
    
    def analyze_correlations(self, symbol: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> CorrelationAnalysis:
        """
        Analyze correlations between spread and other market variables.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Correlation analysis results
        """
        data = self._get_filtered_data(symbol, start_date, end_date)
        
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient data for correlation analysis: {len(data)}")
        
        # Extract data for correlation analysis
        spreads = []
        volumes = []
        prices = []
        
        for d in data:
            if d.spread is not None:
                spreads.append(float(d.spread))
                volumes.append(d.volume or 0)
                
                # Use mid price if available, otherwise average of bid/ask
                if d.bid_price and d.ask_price:
                    prices.append(float((d.bid_price + d.ask_price) / 2))
                elif d.bid_price:
                    prices.append(float(d.bid_price))
                elif d.ask_price:
                    prices.append(float(d.ask_price))
                else:
                    prices.append(0)
        
        # Calculate correlations
        spread_volume_corr = self._calculate_correlation(spreads, volumes)
        spread_price_corr = self._calculate_correlation(spreads, prices)
        
        # For volatility correlation, we need to calculate price volatility
        price_volatility = self._calculate_rolling_volatility(prices)
        spread_volatility_corr = self._calculate_correlation(spreads[-len(price_volatility):], price_volatility)
        
        analysis_period = (
            min(d.timestamp for d in data if d.timestamp),
            max(d.timestamp for d in data if d.timestamp)
        )
        
        return CorrelationAnalysis(
            spread_volume_correlation=spread_volume_corr,
            spread_volatility_correlation=spread_volatility_corr,
            spread_price_correlation=spread_price_corr,
            analysis_period=analysis_period,
            sample_size=len(spreads)
        )
    
    def estimate_spread(self,
                        symbol: str,
                        market_data: Optional[Dict[str, Any]] = None,
                        historical_data: Optional[List[SpreadData]] = None) -> SpreadEstimate:
        """
        Estimate spread based on historical analysis.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Additional historical data
            
        Returns:
            Historical model-based spread estimate
        """
        try:
            # Add any provided historical data
            if historical_data:
                self.add_historical_data(symbol, historical_data)
            
            # Get recent statistics
            stats = self.analyze_spread_statistics(symbol)
            
            # Get seasonal adjustment
            seasonal_adjustment = self._get_seasonal_adjustment(symbol)
            
            # Base estimate on recent average with seasonal adjustment
            base_estimate = stats.mean
            if seasonal_adjustment:
                base_estimate += seasonal_adjustment
            
            # Adjust based on current market conditions if available
            if market_data:
                market_adjustment = self._calculate_market_adjustment(symbol, market_data, stats)
                base_estimate += market_adjustment
            
            # Calculate confidence based on data quality
            confidence = self._calculate_historical_confidence(symbol, stats)
            
            # Convert to basis points
            mid_price = Decimal('100')  # Default, should be updated with actual mid price
            if market_data and 'mid_price' in market_data:
                mid_price = Decimal(str(market_data['mid_price']))
            
            spread_bps = (base_estimate / mid_price) * Decimal('10000')
            
            estimate = SpreadEstimate(
                symbol=symbol,
                estimated_spread=base_estimate,
                spread_bps=spread_bps,
                confidence_level=confidence,
                estimation_method="historical_statistical",
                timestamp=datetime.now(),
                market_condition=MarketCondition.NORMAL,  # Would need real-time data to determine
                supporting_data={
                    'historical_mean': float(stats.mean),
                    'historical_std': float(stats.std_dev),
                    'seasonal_adjustment': float(seasonal_adjustment or 0),
                    'data_points': stats.count,
                    'analysis_window_days': self.analysis_window_days
                }
            )
            
            return estimate
            
        except Exception as e:
            logger.error(f"Failed to estimate spread using historical analysis for {symbol}: {e}")
            raise
    
    def validate_spread_data(self, spread_data: SpreadData) -> bool:
        """
        Validate historical spread data.
        
        Args:
            spread_data: Spread data to validate
            
        Returns:
            True if data is valid for historical analysis
        """
        if not spread_data or not spread_data.symbol:
            return False
        
        # For historical analysis, we're more lenient on timestamp freshness
        if spread_data.timestamp is None:
            return False
        
        # Validate spread values
        if spread_data.spread is not None:
            if spread_data.spread < 0:
                return False
        
        # Validate prices if available
        if spread_data.bid_price and spread_data.ask_price:
            if spread_data.bid_price <= 0 or spread_data.ask_price <= 0:
                return False
            if spread_data.ask_price <= spread_data.bid_price:
                return False
        
        return True
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols with historical data."""
        return [symbol for symbol, data in self._spread_history.items() if len(data) >= self.min_data_points]
    
    def get_analysis_summary(self, symbol: str) -> SpreadAnalysisResult:
        """
        Get comprehensive analysis summary for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Complete analysis results
        """
        data = self._get_filtered_data(symbol)
        
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient data for analysis: {len(data)}")
        
        # Get statistics
        stats = self.analyze_spread_statistics(symbol)
        
        # Get patterns
        patterns = self.detect_seasonal_patterns(symbol)
        
        # Classify market conditions
        market_conditions = self._classify_historical_conditions(data)
        
        # Analysis period
        analysis_period = (
            min(d.timestamp for d in data if d.timestamp),
            max(d.timestamp for d in data if d.timestamp)
        )
        
        return SpreadAnalysisResult(
            symbol=symbol,
            analysis_period=analysis_period,
            average_spread=stats.mean,
            median_spread=stats.median,
            spread_volatility=stats.std_dev,
            min_spread=stats.min_value,
            max_spread=stats.max_value,
            total_observations=stats.count,
            market_conditions=market_conditions,
            patterns={
                'seasonal_patterns': [
                    {
                        'type': p.period_type,
                        'strength': p.strength,
                        'confidence': p.confidence
                    } for p in patterns
                ],
                'statistics': {
                    'percentile_25': float(stats.percentile_25),
                    'percentile_75': float(stats.percentile_75),
                    'percentile_95': float(stats.percentile_95)
                }
            }
        )
    
    # Private methods
    
    def _get_filtered_data(self, symbol: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[SpreadData]:
        """Get filtered spread data for a symbol."""
        if symbol not in self._spread_history:
            return []
        
        data = self._spread_history[symbol]
        
        # Apply date filters
        if start_date or end_date:
            filtered_data = []
            for d in data:
                if d.timestamp is None:
                    continue
                
                if start_date and d.timestamp < start_date:
                    continue
                
                if end_date and d.timestamp > end_date:
                    continue
                
                filtered_data.append(d)
            
            return filtered_data
        
        return data
    
    def _deduplicate_data(self, data: List[SpreadData]) -> List[SpreadData]:
        """Remove duplicate data points."""
        seen = set()
        unique_data = []
        
        for d in data:
            # Create a key based on symbol and timestamp
            key = (d.symbol, d.timestamp)
            if key not in seen:
                seen.add(key)
                unique_data.append(d)
        
        return sorted(unique_data, key=lambda x: x.timestamp or datetime.min)
    
    def _invalidate_cache(self, symbol: str) -> None:
        """Invalidate cached results for a symbol."""
        keys_to_remove = [key for key in self._cached_statistics.keys() if symbol in key]
        for key in keys_to_remove:
            del self._cached_statistics[key]
        
        keys_to_remove = [key for key in self._cached_patterns.keys() if symbol in key]
        for key in keys_to_remove:
            del self._cached_patterns[key]
    
    def _calculate_percentile(self, sorted_values: List[Decimal], percentile: float) -> Decimal:
        """Calculate percentile value."""
        if not sorted_values:
            return Decimal('0')
        
        index = int(len(sorted_values) * percentile)
        index = max(0, min(index, len(sorted_values) - 1))
        
        return sorted_values[index]
    
    def _analyze_period_pattern(self, data: List[SpreadData], period_type: str) -> Optional[SeasonalPattern]:
        """Analyze pattern for a specific period type."""
        if not data:
            return None
        
        # Group data by period
        period_groups = defaultdict(list)
        
        for d in data:
            if d.timestamp is None or d.spread is None:
                continue
            
            period_key = self._get_period_key(d.timestamp, period_type)
            if period_key:
                period_groups[period_key].append(float(d.spread))
        
        if len(period_groups) < 3:  # Need at least 3 periods
            return None
        
        # Calculate average spread for each period
        pattern_data = {}
        for period, spreads in period_groups.items():
            if spreads:
                pattern_data[period] = Decimal(str(statistics.mean(spreads)))
        
        # Calculate pattern strength (coefficient of variation)
        if len(pattern_data) < 2:
            return None
        
        values = [float(v) for v in pattern_data.values()]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        strength = std_val / mean_val if mean_val > 0 else 0
        confidence = min(1.0, len(period_groups) / 10.0)  # More periods = higher confidence
        
        return SeasonalPattern(
            period_type=period_type,
            pattern_data=pattern_data,
            strength=strength,
            confidence=confidence
        )
    
    def _get_period_key(self, timestamp: datetime, period_type: str) -> Optional[str]:
        """Get period key for grouping."""
        if period_type == 'hourly':
            return str(timestamp.hour)
        elif period_type == 'daily':
            return timestamp.strftime('%A')  # Day of week
        elif period_type == 'weekly':
            return str(timestamp.isocalendar()[1])  # Week of year
        
        return None
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # Filter out zero values and ensure same length
        valid_pairs = [(xi, yi) for xi, yi in zip(x, y) if xi != 0 and yi != 0]
        
        if len(valid_pairs) < 2:
            return 0.0
        
        x_vals, y_vals = zip(*valid_pairs)
        
        try:
            n = len(x_vals)
            sum_x = sum(x_vals)
            sum_y = sum(y_vals)
            sum_xy = sum(xi * yi for xi, yi in zip(x_vals, y_vals))
            sum_x_sq = sum(xi * xi for xi in x_vals)
            sum_y_sq = sum(yi * yi for yi in y_vals)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator_x = n * sum_x_sq - sum_x * sum_x
            denominator_y = n * sum_y_sq - sum_y * sum_y
            
            denominator = (denominator_x * denominator_y) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
            
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def _calculate_rolling_volatility(self, prices: List[float], window: int = 20) -> List[float]:
        """Calculate rolling volatility of prices."""
        if len(prices) < window:
            return []
        
        volatilities = []
        
        for i in range(window, len(prices) + 1):
            window_prices = prices[i-window:i]
            if len(window_prices) > 1:
                volatility = statistics.stdev(window_prices)
                volatilities.append(volatility)
        
        return volatilities
    
    def _get_seasonal_adjustment(self, symbol: str) -> Optional[Decimal]:
        """Get seasonal adjustment for current time."""
        patterns = self.detect_seasonal_patterns(symbol, ['hourly'])
        
        if not patterns:
            return None
        
        current_hour = datetime.now().hour
        hourly_pattern = next((p for p in patterns if p.period_type == 'hourly'), None)
        
        if hourly_pattern and str(current_hour) in hourly_pattern.pattern_data:
            # Calculate adjustment as difference from overall average
            pattern_values = list(hourly_pattern.pattern_data.values())
            overall_avg = sum(pattern_values) / len(pattern_values)
            current_hour_avg = hourly_pattern.pattern_data[str(current_hour)]
            
            return current_hour_avg - overall_avg
        
        return None
    
    def _calculate_market_adjustment(self, symbol: str, market_data: Dict[str, Any], stats: SpreadStatistics) -> Decimal:
        """Calculate adjustment based on current market conditions."""
        adjustment = Decimal('0')
        
        # Adjust based on volume if available
        if 'volume' in market_data:
            current_volume = market_data['volume']
            
            # Get historical volume data
            data = self._get_filtered_data(symbol)
            volumes = [d.volume for d in data if d.volume is not None]
            
            if volumes:
                avg_volume = statistics.mean(volumes)
                if current_volume < avg_volume * 0.5:  # Low volume
                    adjustment += stats.mean * Decimal('0.1')  # Increase spread estimate
        
        return adjustment
    
    def _calculate_historical_confidence(self, symbol: str, stats: SpreadStatistics) -> float:
        """Calculate confidence based on historical data quality."""
        confidence = 0.8  # Base confidence
        
        # Increase confidence with more data points
        if stats.count > 1000:
            confidence += 0.1
        elif stats.count > 500:
            confidence += 0.05
        
        # Decrease confidence with high volatility
        if stats.std_dev > stats.mean * Decimal('0.5'):  # High relative volatility
            confidence -= 0.2
        
        # Increase confidence if patterns are detected
        patterns = self.detect_seasonal_patterns(symbol)
        if patterns:
            confidence += 0.1 * len(patterns)
        
        return max(0.1, min(1.0, confidence))
    
    def _classify_historical_conditions(self, data: List[SpreadData]) -> Dict[MarketCondition, int]:
        """Classify historical market conditions."""
        conditions = {condition: 0 for condition in MarketCondition}
        
        spreads = [d.spread for d in data if d.spread is not None]
        if not spreads:
            return conditions
        
        # Calculate thresholds
        sorted_spreads = sorted(spreads)
        p75_spread = self._calculate_percentile(sorted_spreads, 0.75)
        p90_spread = self._calculate_percentile(sorted_spreads, 0.90)
        
        # Classify each data point
        for d in data:
            if d.spread is None:
                continue
            
            if d.spread >= p90_spread:
                conditions[MarketCondition.STRESSED] += 1
            elif d.spread >= p75_spread:
                conditions[MarketCondition.VOLATILE] += 1
            elif d.volume is not None and d.volume < 1000:
                conditions[MarketCondition.ILLIQUID] += 1
            else:
                conditions[MarketCondition.NORMAL] += 1
        
        return conditions


logger.info("Historical spread analyzer class loaded successfully")