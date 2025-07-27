"""
Bid-Ask Spread Modeling Package
==============================

Comprehensive bid-ask spread modeling with real-time estimation capabilities,
historical analysis, and dynamic adjustments based on market conditions.

This package provides:
- Real-time spread estimation with sub-second accuracy
- Historical spread analysis with statistical modeling
- Predictive spread models using machine learning
- Liquidity-based spread adjustments
- Dynamic real-time spread adaptation
- High-performance caching system
- Comprehensive data validation

Example usage:
    >>> from spreads import RealTimeSpreadEstimator, HistoricalSpreadAnalyzer
    >>> 
    >>> # Real-time estimation
    >>> estimator = RealTimeSpreadEstimator()
    >>> estimator.add_symbol("TCS")
    >>> estimator.start_real_time_updates()
    >>> 
    >>> estimate = estimator.estimate_spread("TCS")
    >>> print(f"Current spread: {estimate.estimated_spread}")
    >>> 
    >>> # Historical analysis
    >>> analyzer = HistoricalSpreadAnalyzer()
    >>> analyzer.add_historical_data("TCS", historical_data)
    >>> analysis = analyzer.get_analysis_summary("TCS")
    >>> print(f"Average spread: {analysis.average_spread}")
"""

from .base_spread_model import (
    BaseSpreadModel,
    SpreadData,
    SpreadEstimate,
    SpreadAnalysisResult,
    MarketCondition,
    SpreadType
)

from .realtime_estimator import (
    RealTimeSpreadEstimator,
    OrderBookSnapshot,
    OrderBookLevel
)

from .historical_analyzer import (
    HistoricalSpreadAnalyzer,
    SpreadStatistics,
    SeasonalPattern,
    CorrelationAnalysis
)

from .predictive_model import (
    PredictiveSpreadModel,
    MarketFeatures,
    PredictionResult
)

from .liquidity_analyzer import (
    LiquidityAnalyzer,
    LiquidityMetrics,
    LiquidityProfile
)

from .dynamic_adjuster import (
    DynamicSpreadAdjuster,
    AdjustmentRule,
    AdjustmentEvent,
    DynamicParameters
)

from .spread_cache import (
    SpreadCache,
    CacheEntry,
    CacheStats,
    get_global_cache,
    configure_global_cache
)

from .spread_validator import (
    SpreadValidator,
    ValidationRule,
    ValidationResult,
    ValidationSummary
)

# Version information
__version__ = "1.0.0"
__author__ = "1998prakhargupta"
__email__ = "1998prakhargupta@gmail.com"

# Package metadata
__all__ = [
    # Base classes and data structures
    "BaseSpreadModel",
    "SpreadData", 
    "SpreadEstimate",
    "SpreadAnalysisResult",
    "MarketCondition",
    "SpreadType",
    
    # Real-time estimation
    "RealTimeSpreadEstimator",
    "OrderBookSnapshot",
    "OrderBookLevel",
    
    # Historical analysis
    "HistoricalSpreadAnalyzer",
    "SpreadStatistics",
    "SeasonalPattern", 
    "CorrelationAnalysis",
    
    # Predictive modeling
    "PredictiveSpreadModel",
    "MarketFeatures",
    "PredictionResult",
    
    # Liquidity analysis
    "LiquidityAnalyzer",
    "LiquidityMetrics",
    "LiquidityProfile",
    
    # Dynamic adjustment
    "DynamicSpreadAdjuster",
    "AdjustmentRule",
    "AdjustmentEvent", 
    "DynamicParameters",
    
    # Caching
    "SpreadCache",
    "CacheEntry",
    "CacheStats",
    "get_global_cache",
    "configure_global_cache",
    
    # Validation
    "SpreadValidator",
    "ValidationRule",
    "ValidationResult",
    "ValidationSummary"
]

# Default configuration
DEFAULT_CONFIG = {
    "cache": {
        "max_entries": 10000,
        "default_ttl_seconds": 300,
        "max_memory_mb": 100
    },
    "real_time": {
        "update_interval": 0.1,
        "history_window": 1000,
        "depth_levels": 5
    },
    "historical": {
        "min_data_points": 100,
        "analysis_window_days": 30,
        "pattern_detection_threshold": 0.1
    },
    "predictive": {
        "prediction_horizons": ["5min", "15min", "1h"],
        "min_training_samples": 500,
        "feature_window_minutes": 60
    },
    "liquidity": {
        "max_depth_levels": 10,
        "liquidity_window_minutes": 30,
        "min_volume_threshold": 100
    },
    "dynamic": {
        "adjustment_interval": 1.0,
        "max_adjustment_history": 1000,
        "enable_auto_tuning": True
    },
    "validation": {
        "enable_statistical_checks": True,
        "enable_temporal_checks": True,
        "enable_market_checks": True,
        "outlier_threshold_sigma": 3.0
    }
}


def create_spread_estimator_suite(config: dict = None) -> dict:
    """
    Create a complete suite of spread estimation tools.
    
    Args:
        config: Configuration dictionary (uses DEFAULT_CONFIG if None)
        
    Returns:
        Dictionary containing all spread estimation components
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Initialize cache
    cache = SpreadCache(**config.get("cache", {}))
    
    # Initialize validator
    validator = SpreadValidator(**config.get("validation", {}))
    
    # Initialize estimators
    real_time_estimator = RealTimeSpreadEstimator(**config.get("real_time", {}))
    historical_analyzer = HistoricalSpreadAnalyzer(**config.get("historical", {}))
    predictive_model = PredictiveSpreadModel(**config.get("predictive", {}))
    liquidity_analyzer = LiquidityAnalyzer(**config.get("liquidity", {}))
    dynamic_adjuster = DynamicSpreadAdjuster(**config.get("dynamic", {}))
    
    return {
        "cache": cache,
        "validator": validator,
        "real_time_estimator": real_time_estimator,
        "historical_analyzer": historical_analyzer,
        "predictive_model": predictive_model,
        "liquidity_analyzer": liquidity_analyzer,
        "dynamic_adjuster": dynamic_adjuster,
        "config": config
    }


def get_package_info() -> dict:
    """Get package information and statistics."""
    return {
        "version": __version__,
        "author": __author__,
        "components": len(__all__),
        "default_config": DEFAULT_CONFIG,
        "supported_features": [
            "Real-time spread estimation",
            "Historical spread analysis", 
            "Predictive modeling",
            "Liquidity analysis",
            "Dynamic adjustments",
            "Performance caching",
            "Data validation"
        ]
    }


# Module-level logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Bid-ask spread modeling package v{__version__} loaded successfully")