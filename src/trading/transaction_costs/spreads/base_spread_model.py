"""
Base Spread Model
================

Abstract base class for all bid-ask spread models.
Provides the standard interface and common functionality for spread estimation,
validation, and performance tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SpreadType(Enum):
    """Types of spread calculations."""
    QUOTED = "quoted"  # Actual quoted bid-ask spread
    EFFECTIVE = "effective"  # Effective spread including market impact
    REALIZED = "realized"  # Realized spread post-execution
    PREDICTED = "predicted"  # Predicted spread


class MarketCondition(Enum):
    """Market condition classifications."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    STRESSED = "stressed"


@dataclass
class SpreadData:
    """Container for spread-related data."""
    symbol: str
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    spread_bps: Optional[Decimal] = None  # Spread in basis points
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None
    market_condition: Optional[MarketCondition] = None
    confidence: Optional[float] = None


@dataclass
class SpreadEstimate:
    """Container for spread estimation results."""
    symbol: str
    estimated_spread: Decimal
    spread_bps: Decimal
    confidence_level: float
    estimation_method: str
    timestamp: datetime
    market_condition: MarketCondition
    supporting_data: Optional[Dict[str, Any]] = None


@dataclass
class SpreadAnalysisResult:
    """Container for spread analysis results."""
    symbol: str
    analysis_period: Tuple[datetime, datetime]
    average_spread: Decimal
    median_spread: Decimal
    spread_volatility: Decimal
    min_spread: Decimal
    max_spread: Decimal
    total_observations: int
    market_conditions: Dict[MarketCondition, int]
    patterns: Optional[Dict[str, Any]] = None


class BaseSpreadModel(ABC):
    """
    Abstract base class for all spread models.
    
    This class provides the standard interface and common functionality
    for spread estimation, validation, and performance tracking.
    """
    
    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        supported_instruments: Optional[List[str]] = None,
        min_confidence_threshold: float = 0.5,
        cache_duration: int = 60  # seconds
    ):
        """
        Initialize the spread model.
        
        Args:
            model_name: Unique name for this model
            version: Version of the model
            supported_instruments: List of supported instrument types
            min_confidence_threshold: Minimum confidence level for estimates
            cache_duration: Cache duration in seconds
        """
        self.model_name = model_name
        self.version = version
        self.supported_instruments = supported_instruments or ["EQUITY"]
        self.min_confidence_threshold = min_confidence_threshold
        self.cache_duration = cache_duration
        
        # Performance tracking
        self._estimation_count = 0
        self._total_estimation_time = 0.0
        self._last_estimation_time = None
        
        # Cache for recent estimates
        self._estimate_cache: Dict[str, Tuple[SpreadEstimate, datetime]] = {}
        
        logger.info(f"Initialized {model_name} spread model v{version}")
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def estimate_spread(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[SpreadData]] = None
    ) -> SpreadEstimate:
        """
        Estimate the bid-ask spread for a given symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Historical spread data
            
        Returns:
            Spread estimate with confidence level
        """
        pass
    
    @abstractmethod
    def validate_spread_data(self, spread_data: SpreadData) -> bool:
        """
        Validate spread data quality.
        
        Args:
            spread_data: Spread data to validate
            
        Returns:
            True if data is valid
        """
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported trading symbols.
        
        Returns:
            List of supported symbols
        """
        pass
    
    # Common utility methods
    
    def calculate_spread_bps(self, bid: Decimal, ask: Decimal, mid_price: Optional[Decimal] = None) -> Decimal:
        """
        Calculate spread in basis points.
        
        Args:
            bid: Bid price
            ask: Ask price
            mid_price: Mid price (optional, will calculate if not provided)
            
        Returns:
            Spread in basis points
        """
        if bid <= 0 or ask <= 0 or ask <= bid:
            raise ValueError("Invalid bid/ask prices")
        
        spread = ask - bid
        if mid_price is None:
            mid_price = (bid + ask) / 2
        
        if mid_price <= 0:
            raise ValueError("Invalid mid price")
        
        return (spread / mid_price) * Decimal('10000')
    
    def classify_market_condition(
        self,
        current_spread: Decimal,
        historical_spreads: List[Decimal],
        volume: Optional[int] = None
    ) -> MarketCondition:
        """
        Classify current market condition based on spread and volume.
        
        Args:
            current_spread: Current spread value
            historical_spreads: Historical spread values
            volume: Current trading volume
            
        Returns:
            Market condition classification
        """
        if not historical_spreads:
            return MarketCondition.NORMAL
        
        # Calculate percentiles
        sorted_spreads = sorted(historical_spreads)
        n = len(sorted_spreads)
        p75_index = int(0.75 * n)
        p90_index = int(0.90 * n)
        
        p75_spread = sorted_spreads[min(p75_index, n-1)]
        p90_spread = sorted_spreads[min(p90_index, n-1)]
        
        # Classify based on spread percentiles
        if current_spread >= p90_spread:
            return MarketCondition.STRESSED
        elif current_spread >= p75_spread:
            return MarketCondition.VOLATILE
        elif volume is not None and volume < 1000:  # Low volume threshold
            return MarketCondition.ILLIQUID
        else:
            return MarketCondition.NORMAL
    
    def get_cached_estimate(self, symbol: str) -> Optional[SpreadEstimate]:
        """
        Get cached spread estimate if available and fresh.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cached estimate or None
        """
        if symbol in self._estimate_cache:
            estimate, timestamp = self._estimate_cache[symbol]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age <= self.cache_duration:
                return estimate
            else:
                # Remove stale cache entry
                del self._estimate_cache[symbol]
        
        return None
    
    def cache_estimate(self, estimate: SpreadEstimate) -> None:
        """
        Cache a spread estimate.
        
        Args:
            estimate: Spread estimate to cache
        """
        self._estimate_cache[estimate.symbol] = (estimate, datetime.now())
        
        # Simple cache cleanup
        if len(self._estimate_cache) > 1000:
            # Remove oldest 10% of entries
            sorted_items = sorted(
                self._estimate_cache.items(),
                key=lambda x: x[1][1]
            )
            for symbol, _ in sorted_items[:100]:
                del self._estimate_cache[symbol]
    
    def update_performance_metrics(self, estimation_time: float) -> None:
        """Update performance tracking metrics."""
        self._estimation_count += 1
        self._total_estimation_time += estimation_time
        self._last_estimation_time = estimation_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = (
            self._total_estimation_time / self._estimation_count
            if self._estimation_count > 0
            else 0.0
        )
        
        return {
            'model_name': self.model_name,
            'version': self.version,
            'total_estimations': self._estimation_count,
            'average_estimation_time': avg_time,
            'last_estimation_time': self._last_estimation_time,
            'cache_size': len(self._estimate_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the estimate cache."""
        self._estimate_cache.clear()
        logger.info(f"Cache cleared for {self.model_name}")
    
    def is_symbol_supported(self, symbol: str) -> bool:
        """
        Check if a symbol is supported by this model.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if symbol is supported
        """
        return symbol in self.get_supported_symbols()
    
    def validate_confidence_level(self, confidence: float) -> bool:
        """
        Validate confidence level meets minimum threshold.
        
        Args:
            confidence: Confidence level to validate
            
        Returns:
            True if confidence meets threshold
        """
        return confidence >= self.min_confidence_threshold


logger.info("Base spread model class loaded successfully")