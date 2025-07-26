"""
Market Condition Analyzer
=========================

Analyzes current market conditions to support impact and slippage calculations.
Provides market regime detection and condition assessment.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

from ..models import MarketConditions

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime enumeration."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"


class LiquidityLevel(Enum):
    """Liquidity level enumeration."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    VERY_LOW = "very_low"


class MarketConditionAnalyzer:
    """
    Analyzes market conditions for impact and slippage modeling.
    
    Provides assessment of:
    - Market regime (trending vs mean-reverting)
    - Volatility level
    - Liquidity conditions
    - Market stress indicators
    """
    
    def __init__(self):
        """Initialize the market condition analyzer."""
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Volatility thresholds (daily basis)
        self.volatility_thresholds = {
            'low': Decimal('0.01'),      # 1% daily
            'normal': Decimal('0.02'),   # 2% daily
            'high': Decimal('0.03')      # 3% daily
        }
        
        # Spread thresholds (as percentage of price)
        self.spread_thresholds = {
            'tight': Decimal('0.0005'),   # 0.05%
            'normal': Decimal('0.002'),   # 0.2%
            'wide': Decimal('0.005')      # 0.5%
        }
    
    def analyze_conditions(self, market_conditions: MarketConditions) -> Dict[str, Any]:
        """
        Perform comprehensive market condition analysis.
        
        Args:
            market_conditions: Current market conditions
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            'timestamp': datetime.now(),
            'data_timestamp': market_conditions.timestamp,
            'symbol_analysis': self._analyze_symbol_conditions(market_conditions),
            'volatility_analysis': self._analyze_volatility(market_conditions),
            'liquidity_analysis': self._analyze_liquidity(market_conditions),
            'market_stress': self._assess_market_stress(market_conditions),
            'regime_detection': self._detect_market_regime(market_conditions)
        }
        
        self.logger.debug(f"Market condition analysis completed: {analysis['regime_detection']}")
        return analysis
    
    def _analyze_symbol_conditions(self, market_conditions: MarketConditions) -> Dict[str, Any]:
        """
        Analyze symbol-specific conditions.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Symbol condition analysis
        """
        analysis = {
            'has_bid_ask': bool(market_conditions.bid_price and market_conditions.ask_price),
            'has_volume': bool(market_conditions.volume),
            'has_sizes': bool(market_conditions.bid_size and market_conditions.ask_size),
            'data_age_seconds': (datetime.now() - market_conditions.timestamp).total_seconds(),
            'market_open': market_conditions.market_open,
            'circuit_breaker': market_conditions.circuit_breaker,
            'halt_status': market_conditions.halt_status
        }
        
        # Data quality assessment
        if analysis['data_age_seconds'] < 60:
            analysis['data_quality'] = 'fresh'
        elif analysis['data_age_seconds'] < 300:
            analysis['data_quality'] = 'stale'
        else:
            analysis['data_quality'] = 'old'
        
        return analysis
    
    def _analyze_volatility(self, market_conditions: MarketConditions) -> Dict[str, Any]:
        """
        Analyze volatility conditions.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Volatility analysis
        """
        # Get volatility measure
        volatility = None
        volatility_source = None
        
        if market_conditions.realized_volatility:
            volatility = market_conditions.realized_volatility
            volatility_source = 'realized'
        elif market_conditions.implied_volatility:
            volatility = market_conditions.implied_volatility
            volatility_source = 'implied'
        
        analysis = {
            'volatility': float(volatility) if volatility else None,
            'volatility_source': volatility_source,
            'level': self._classify_volatility_level(volatility),
            'regime': self._classify_volatility_regime(volatility)
        }
        
        return analysis
    
    def _analyze_liquidity(self, market_conditions: MarketConditions) -> Dict[str, Any]:
        """
        Analyze liquidity conditions.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Liquidity analysis
        """
        analysis = {
            'level': LiquidityLevel.NORMAL,
            'spread_analysis': {},
            'size_analysis': {},
            'volume_analysis': {}
        }
        
        # Spread analysis
        if market_conditions.bid_price and market_conditions.ask_price:
            spread = market_conditions.ask_price - market_conditions.bid_price
            mid_price = (market_conditions.bid_price + market_conditions.ask_price) / 2
            spread_pct = spread / mid_price
            
            analysis['spread_analysis'] = {
                'absolute_spread': float(spread),
                'percentage_spread': float(spread_pct * 100),
                'classification': self._classify_spread_width(spread_pct)
            }
        
        # Size analysis
        if market_conditions.bid_size and market_conditions.ask_size:
            total_size = market_conditions.bid_size + market_conditions.ask_size
            size_imbalance = abs(market_conditions.bid_size - market_conditions.ask_size) / total_size
            
            analysis['size_analysis'] = {
                'bid_size': market_conditions.bid_size,
                'ask_size': market_conditions.ask_size,
                'total_size': total_size,
                'imbalance': float(size_imbalance)
            }
        
        # Volume analysis
        if market_conditions.volume and market_conditions.average_daily_volume:
            volume_ratio = market_conditions.volume / market_conditions.average_daily_volume
            analysis['volume_analysis'] = {
                'current_volume': market_conditions.volume,
                'average_daily_volume': market_conditions.average_daily_volume,
                'volume_ratio': float(volume_ratio)
            }
        
        # Overall liquidity level
        analysis['level'] = self._assess_liquidity_level(analysis)
        
        return analysis
    
    def _assess_market_stress(self, market_conditions: MarketConditions) -> Dict[str, Any]:
        """
        Assess market stress indicators.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Market stress analysis
        """
        stress_indicators = []
        stress_score = 0
        
        # Circuit breaker or halt
        if market_conditions.circuit_breaker:
            stress_indicators.append('circuit_breaker_active')
            stress_score += 3
        
        if market_conditions.halt_status:
            stress_indicators.append('trading_halted')
            stress_score += 2
        
        # High volatility
        volatility = self._get_volatility(market_conditions)
        if volatility and volatility > self.volatility_thresholds['high']:
            stress_indicators.append('high_volatility')
            stress_score += 2
        
        # Wide spreads
        if market_conditions.bid_price and market_conditions.ask_price:
            spread = market_conditions.ask_price - market_conditions.bid_price
            mid_price = (market_conditions.bid_price + market_conditions.ask_price) / 2
            spread_pct = spread / mid_price
            
            if spread_pct > self.spread_thresholds['wide']:
                stress_indicators.append('wide_spreads')
                stress_score += 1
        
        # Classify stress level
        if stress_score >= 5:
            stress_level = 'high'
        elif stress_score >= 3:
            stress_level = 'medium'
        elif stress_score >= 1:
            stress_level = 'low'
        else:
            stress_level = 'normal'
        
        return {
            'stress_level': stress_level,
            'stress_score': stress_score,
            'indicators': stress_indicators
        }
    
    def _detect_market_regime(self, market_conditions: MarketConditions) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Detected market regime
        """
        volatility = self._get_volatility(market_conditions)
        
        if not volatility:
            return MarketRegime.NORMAL
        
        # Volatility-based regime detection
        if volatility > self.volatility_thresholds['high']:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.volatility_thresholds['low']:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL
    
    def _classify_volatility_level(self, volatility: Optional[Decimal]) -> str:
        """Classify volatility level."""
        if not volatility:
            return 'unknown'
        
        if volatility > self.volatility_thresholds['high']:
            return 'high'
        elif volatility > self.volatility_thresholds['normal']:
            return 'normal'
        elif volatility > self.volatility_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def _classify_volatility_regime(self, volatility: Optional[Decimal]) -> str:
        """Classify volatility regime."""
        if not volatility:
            return 'unknown'
        
        if volatility > self.volatility_thresholds['high']:
            return 'stressed'
        elif volatility < self.volatility_thresholds['low']:
            return 'calm'
        else:
            return 'normal'
    
    def _classify_spread_width(self, spread_pct: Decimal) -> str:
        """Classify spread width."""
        if spread_pct > self.spread_thresholds['wide']:
            return 'wide'
        elif spread_pct > self.spread_thresholds['normal']:
            return 'normal'
        elif spread_pct > self.spread_thresholds['tight']:
            return 'tight'
        else:
            return 'very_tight'
    
    def _assess_liquidity_level(self, analysis: Dict[str, Any]) -> LiquidityLevel:
        """Assess overall liquidity level from analysis components."""
        spread_class = analysis.get('spread_analysis', {}).get('classification')
        
        if spread_class == 'very_tight':
            return LiquidityLevel.HIGH
        elif spread_class in ['tight', 'normal']:
            return LiquidityLevel.NORMAL
        elif spread_class == 'wide':
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW
    
    def _get_volatility(self, market_conditions: MarketConditions) -> Optional[Decimal]:
        """Get volatility from market conditions."""
        if market_conditions.realized_volatility:
            return market_conditions.realized_volatility
        elif market_conditions.implied_volatility:
            return market_conditions.implied_volatility
        return None