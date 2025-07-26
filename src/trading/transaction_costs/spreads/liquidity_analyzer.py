"""
Liquidity Analyzer
=================

Comprehensive liquidity analysis for bid-ask spread adjustments.
Analyzes order book depth, market impact, and liquidity patterns.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import statistics

from .base_spread_model import (
    BaseSpreadModel, SpreadEstimate, SpreadData, 
    MarketCondition
)

logger = logging.getLogger(__name__)


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""
    bid_depth: Decimal  # Total bid volume
    ask_depth: Decimal  # Total ask volume
    total_depth: Decimal  # Combined depth
    depth_imbalance: float  # (bid_depth - ask_depth) / total_depth
    effective_spread: Decimal  # Spread adjusted for depth
    liquidity_score: float  # Overall liquidity score (0-1)
    market_impact_estimate: Decimal  # Estimated market impact
    time_to_fill: Optional[float]  # Estimated time to fill order


@dataclass
class OrderBookLevel:
    """Order book level data."""
    price: Decimal
    volume: int
    cumulative_volume: int
    distance_from_mid: Decimal  # Distance from mid price in bps


@dataclass
class LiquidityProfile:
    """Complete liquidity profile for a symbol."""
    symbol: str
    timestamp: datetime
    bid_levels: List[OrderBookLevel]
    ask_levels: List[OrderBookLevel]
    metrics: LiquidityMetrics
    market_condition: MarketCondition


class LiquidityAnalyzer(BaseSpreadModel):
    """
    Comprehensive liquidity analyzer for spread adjustments.
    
    Features:
    - Order book depth analysis
    - Market impact estimation
    - Liquidity scoring
    - Time-to-fill estimation
    - Liquidity-adjusted spread calculations
    """
    
    def __init__(
        self,
        max_depth_levels: int = 10,
        liquidity_window_minutes: int = 30,
        min_volume_threshold: int = 100,
        **kwargs
    ):
        """
        Initialize liquidity analyzer.
        
        Args:
            max_depth_levels: Maximum order book levels to analyze
            liquidity_window_minutes: Window for liquidity pattern analysis
            min_volume_threshold: Minimum volume for liquidity calculation
        """
        super().__init__(
            model_name="LiquidityAnalyzer",
            version="1.0.0",
            **kwargs
        )
        
        self.max_depth_levels = max_depth_levels
        self.liquidity_window_minutes = liquidity_window_minutes
        self.min_volume_threshold = min_volume_threshold
        
        # Liquidity data storage
        self._liquidity_history: Dict[str, deque] = {}
        self._order_book_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        
        # Liquidity model parameters
        self._liquidity_weights = {
            'depth_score': 0.4,
            'balance_score': 0.2,
            'consistency_score': 0.25,
            'efficiency_score': 0.15
        }
        
        logger.info("Liquidity analyzer initialized")
    
    def analyze_liquidity(
        self,
        symbol: str,
        order_book_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> LiquidityProfile:
        """
        Analyze liquidity for a symbol.
        
        Args:
            symbol: Trading symbol
            order_book_data: Order book data
            market_data: Current market data
            
        Returns:
            Complete liquidity profile
        """
        try:
            # Get or create order book data
            if not order_book_data:
                order_book_data = self._get_cached_order_book(symbol) or self._create_mock_order_book(symbol, market_data)
            
            # Parse order book levels
            bid_levels, ask_levels = self._parse_order_book(order_book_data)
            
            # Calculate liquidity metrics
            metrics = self._calculate_liquidity_metrics(bid_levels, ask_levels)
            
            # Determine market condition
            market_condition = self._classify_liquidity_condition(metrics)
            
            # Create liquidity profile
            profile = LiquidityProfile(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                metrics=metrics,
                market_condition=market_condition
            )
            
            # Store in history
            self._store_liquidity_data(symbol, profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to analyze liquidity for {symbol}: {e}")
            raise
    
    def estimate_market_impact(
        self,
        symbol: str,
        order_size: int,
        side: str,  # 'buy' or 'sell'
        order_book_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[Decimal, float]:
        """
        Estimate market impact for a given order size.
        
        Args:
            symbol: Trading symbol
            order_size: Order size in shares
            side: Order side ('buy' or 'sell')
            order_book_data: Order book data
            
        Returns:
            Tuple of (impact_cost, confidence)
        """
        try:
            profile = self.analyze_liquidity(symbol, order_book_data)
            
            # Select appropriate side
            levels = profile.ask_levels if side == 'buy' else profile.bid_levels
            
            if not levels:
                return Decimal('0'), 0.0
            
            # Calculate impact by walking through order book
            remaining_size = order_size
            total_cost = Decimal('0')
            total_shares = 0
            
            for level in levels:
                if remaining_size <= 0:
                    break
                
                shares_at_level = min(remaining_size, level.volume)
                cost_at_level = level.price * shares_at_level
                
                total_cost += cost_at_level
                total_shares += shares_at_level
                remaining_size -= shares_at_level
            
            if total_shares == 0:
                return Decimal('0'), 0.0
            
            # Calculate average execution price
            avg_execution_price = total_cost / total_shares
            
            # Calculate impact relative to mid price
            mid_price = self._calculate_mid_price(profile.bid_levels, profile.ask_levels)
            if mid_price == 0:
                return Decimal('0'), 0.0
            
            impact = abs(avg_execution_price - mid_price)
            
            # Calculate confidence based on order book depth
            confidence = min(1.0, total_shares / order_size)
            
            return impact, confidence
            
        except Exception as e:
            logger.error(f"Failed to estimate market impact for {symbol}: {e}")
            return Decimal('0'), 0.0
    
    def estimate_time_to_fill(
        self,
        symbol: str,
        order_size: int,
        historical_volume_data: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[float]:
        """
        Estimate time to fill an order based on historical volume patterns.
        
        Args:
            symbol: Trading symbol
            order_size: Order size in shares
            historical_volume_data: Historical volume data
            
        Returns:
            Estimated fill time in minutes
        """
        try:
            # Get recent liquidity data
            if symbol not in self._liquidity_history:
                return None
            
            recent_profiles = list(self._liquidity_history[symbol])[-20:]  # Last 20 data points
            
            if len(recent_profiles) < 5:
                return None
            
            # Calculate average volume rate
            total_volume = 0
            time_span = 0
            
            for i, profile in enumerate(recent_profiles):
                volume = int(profile.metrics.total_depth)
                total_volume += volume
                
                if i > 0:
                    time_diff = (profile.timestamp - recent_profiles[i-1].timestamp).total_seconds() / 60
                    time_span += time_diff
            
            if time_span == 0:
                return None
            
            volume_rate = total_volume / time_span  # Shares per minute
            
            if volume_rate == 0:
                return None
            
            # Estimate fill time (conservative estimate - assume we get 20% of market volume)
            participation_rate = 0.2
            effective_rate = volume_rate * participation_rate
            
            estimated_time = order_size / effective_rate
            
            # Apply liquidity adjustment
            current_profile = recent_profiles[-1]
            liquidity_factor = current_profile.metrics.liquidity_score
            
            # Lower liquidity -> longer fill time
            adjusted_time = estimated_time / max(0.1, liquidity_factor)
            
            return adjusted_time
            
        except Exception as e:
            logger.error(f"Failed to estimate fill time for {symbol}: {e}")
            return None
    
    def calculate_liquidity_adjusted_spread(
        self,
        symbol: str,
        base_spread: Decimal,
        order_size: Optional[int] = None,
        order_book_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[Decimal, float]:
        """
        Calculate liquidity-adjusted spread.
        
        Args:
            symbol: Trading symbol
            base_spread: Base bid-ask spread
            order_size: Expected order size
            order_book_data: Order book data
            
        Returns:
            Tuple of (adjusted_spread, confidence)
        """
        try:
            profile = self.analyze_liquidity(symbol, order_book_data)
            
            # Base adjustment factor
            liquidity_factor = 1.0 + (1.0 - profile.metrics.liquidity_score)
            
            # Order size adjustment
            if order_size:
                market_impact, impact_confidence = self.estimate_market_impact(symbol, order_size, 'buy', order_book_data)
                # Convert market impact to spread adjustment
                impact_adjustment = market_impact * 2  # Both sides of spread
                adjusted_spread = base_spread + impact_adjustment
            else:
                adjusted_spread = base_spread * Decimal(str(liquidity_factor))
            
            # Depth adjustment
            if profile.metrics.total_depth < self.min_volume_threshold:
                depth_penalty = Decimal('1.5')  # 50% penalty for low depth
                adjusted_spread *= depth_penalty
            
            # Imbalance adjustment
            if abs(profile.metrics.depth_imbalance) > 0.3:  # Significant imbalance
                imbalance_penalty = Decimal('1.2')  # 20% penalty
                adjusted_spread *= imbalance_penalty
            
            # Calculate confidence
            confidence = profile.metrics.liquidity_score
            
            return adjusted_spread, confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate liquidity-adjusted spread for {symbol}: {e}")
            return base_spread, 0.5
    
    def estimate_spread(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[SpreadData]] = None
    ) -> SpreadEstimate:
        """
        Estimate spread using liquidity analysis.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Historical spread data
            
        Returns:
            Liquidity-based spread estimate
        """
        try:
            # Analyze current liquidity
            profile = self.analyze_liquidity(symbol, market_data.get('order_book') if market_data else None, market_data)
            
            # Base spread from liquidity metrics
            base_spread = profile.metrics.effective_spread
            
            # Adjust for liquidity conditions
            adjusted_spread, confidence = self.calculate_liquidity_adjusted_spread(symbol, base_spread)
            
            # Convert to basis points
            mid_price = self._calculate_mid_price(profile.bid_levels, profile.ask_levels)
            spread_bps = (adjusted_spread / mid_price) * Decimal('10000') if mid_price > 0 else Decimal('0')
            
            estimate = SpreadEstimate(
                symbol=symbol,
                estimated_spread=adjusted_spread,
                spread_bps=spread_bps,
                confidence_level=confidence,
                estimation_method="liquidity_analysis",
                timestamp=datetime.now(),
                market_condition=profile.market_condition,
                supporting_data={
                    'liquidity_score': profile.metrics.liquidity_score,
                    'bid_depth': float(profile.metrics.bid_depth),
                    'ask_depth': float(profile.metrics.ask_depth),
                    'depth_imbalance': profile.metrics.depth_imbalance,
                    'market_impact_estimate': float(profile.metrics.market_impact_estimate),
                    'order_book_levels': len(profile.bid_levels) + len(profile.ask_levels)
                }
            )
            
            return estimate
            
        except Exception as e:
            logger.error(f"Failed to estimate spread using liquidity analysis for {symbol}: {e}")
            raise
    
    def validate_spread_data(self, spread_data: SpreadData) -> bool:
        """Validate spread data for liquidity analysis."""
        return spread_data is not None and spread_data.symbol is not None
    
    def get_supported_symbols(self) -> List[str]:
        """Get symbols with liquidity data."""
        return list(self._liquidity_history.keys())
    
    def get_liquidity_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get liquidity summary for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Liquidity summary statistics
        """
        if symbol not in self._liquidity_history:
            return None
        
        profiles = list(self._liquidity_history[symbol])
        if not profiles:
            return None
        
        # Calculate summary statistics
        liquidity_scores = [p.metrics.liquidity_score for p in profiles]
        depth_values = [float(p.metrics.total_depth) for p in profiles]
        imbalances = [abs(p.metrics.depth_imbalance) for p in profiles]
        
        return {
            'symbol': symbol,
            'data_points': len(profiles),
            'average_liquidity_score': statistics.mean(liquidity_scores),
            'liquidity_score_std': statistics.stdev(liquidity_scores) if len(liquidity_scores) > 1 else 0,
            'average_depth': statistics.mean(depth_values),
            'average_imbalance': statistics.mean(imbalances),
            'last_update': profiles[-1].timestamp.isoformat(),
            'market_condition_distribution': self._get_condition_distribution(profiles)
        }
    
    # Private methods
    
    def _parse_order_book(self, order_book_data: Dict[str, Any]) -> Tuple[List[OrderBookLevel], List[OrderBookLevel]]:
        """Parse order book data into structured levels."""
        bid_levels = []
        ask_levels = []
        
        # Get mid price for distance calculation
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])
        
        if bids and asks:
            best_bid = Decimal(str(bids[0]['price']))
            best_ask = Decimal(str(asks[0]['price']))
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = Decimal('100')  # Default
        
        # Parse bid levels
        cumulative_volume = 0
        for i, level in enumerate(bids[:self.max_depth_levels]):
            price = Decimal(str(level['price']))
            volume = int(level['volume'])
            cumulative_volume += volume
            
            distance_bps = ((mid_price - price) / mid_price) * Decimal('10000') if mid_price > 0 else Decimal('0')
            
            bid_levels.append(OrderBookLevel(
                price=price,
                volume=volume,
                cumulative_volume=cumulative_volume,
                distance_from_mid=distance_bps
            ))
        
        # Parse ask levels
        cumulative_volume = 0
        for i, level in enumerate(asks[:self.max_depth_levels]):
            price = Decimal(str(level['price']))
            volume = int(level['volume'])
            cumulative_volume += volume
            
            distance_bps = ((price - mid_price) / mid_price) * Decimal('10000') if mid_price > 0 else Decimal('0')
            
            ask_levels.append(OrderBookLevel(
                price=price,
                volume=volume,
                cumulative_volume=cumulative_volume,
                distance_from_mid=distance_bps
            ))
        
        return bid_levels, ask_levels
    
    def _calculate_liquidity_metrics(self, bid_levels: List[OrderBookLevel], ask_levels: List[OrderBookLevel]) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics."""
        # Basic depth metrics
        bid_depth = sum(Decimal(str(level.volume)) for level in bid_levels)
        ask_depth = sum(Decimal(str(level.volume)) for level in ask_levels)
        total_depth = bid_depth + ask_depth
        
        # Depth imbalance
        depth_imbalance = float((bid_depth - ask_depth) / total_depth) if total_depth > 0 else 0.0
        
        # Effective spread (weighted by volume)
        effective_spread = self._calculate_effective_spread(bid_levels, ask_levels)
        
        # Market impact estimate
        market_impact = self._estimate_basic_market_impact(bid_levels, ask_levels)
        
        # Liquidity score (composite measure)
        liquidity_score = self._calculate_liquidity_score(bid_levels, ask_levels, total_depth)
        
        return LiquidityMetrics(
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            depth_imbalance=depth_imbalance,
            effective_spread=effective_spread,
            liquidity_score=liquidity_score,
            market_impact_estimate=market_impact,
            time_to_fill=None  # Calculated separately
        )
    
    def _calculate_effective_spread(self, bid_levels: List[OrderBookLevel], ask_levels: List[OrderBookLevel]) -> Decimal:
        """Calculate volume-weighted effective spread."""
        if not bid_levels or not ask_levels:
            return Decimal('0')
        
        # Weight by volume at each level
        total_volume = sum(level.volume for level in bid_levels + ask_levels)
        
        if total_volume == 0:
            # Fallback to simple spread
            return ask_levels[0].price - bid_levels[0].price
        
        # Volume-weighted prices
        weighted_bid = sum(level.price * level.volume for level in bid_levels) / sum(level.volume for level in bid_levels)
        weighted_ask = sum(level.price * level.volume for level in ask_levels) / sum(level.volume for level in ask_levels)
        
        return weighted_ask - weighted_bid
    
    def _estimate_basic_market_impact(self, bid_levels: List[OrderBookLevel], ask_levels: List[OrderBookLevel]) -> Decimal:
        """Estimate market impact for a standard order size."""
        standard_order_size = 1000  # Standard order size for impact estimation
        
        # Estimate impact on buy side
        buy_impact = self._calculate_side_impact(ask_levels, standard_order_size)
        
        # Estimate impact on sell side
        sell_impact = self._calculate_side_impact(bid_levels, standard_order_size)
        
        # Return average impact
        return (buy_impact + sell_impact) / 2
    
    def _calculate_side_impact(self, levels: List[OrderBookLevel], order_size: int) -> Decimal:
        """Calculate impact for one side of the order book."""
        if not levels:
            return Decimal('0')
        
        remaining_size = order_size
        total_cost = Decimal('0')
        total_shares = 0
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            shares_at_level = min(remaining_size, level.volume)
            cost_at_level = level.price * shares_at_level
            
            total_cost += cost_at_level
            total_shares += shares_at_level
            remaining_size -= shares_at_level
        
        if total_shares == 0:
            return Decimal('0')
        
        avg_price = total_cost / total_shares
        reference_price = levels[0].price
        
        return abs(avg_price - reference_price)
    
    def _calculate_liquidity_score(self, bid_levels: List[OrderBookLevel], ask_levels: List[OrderBookLevel], total_depth: Decimal) -> float:
        """Calculate composite liquidity score (0-1)."""
        scores = []
        
        # Depth score (normalized by typical volumes)
        depth_score = min(1.0, float(total_depth) / 10000)  # 10k shares = perfect depth
        scores.append(depth_score * self._liquidity_weights['depth_score'])
        
        # Balance score (how balanced bid/ask are)
        if total_depth > 0:
            bid_depth = sum(Decimal(str(level.volume)) for level in bid_levels)
            ask_depth = sum(Decimal(str(level.volume)) for level in ask_levels)
            imbalance = abs(float((bid_depth - ask_depth) / total_depth))
            balance_score = 1.0 - imbalance  # Lower imbalance = higher score
        else:
            balance_score = 0.0
        scores.append(balance_score * self._liquidity_weights['balance_score'])
        
        # Consistency score (how many levels have reasonable volume)
        all_levels = bid_levels + ask_levels
        if all_levels:
            avg_volume = sum(level.volume for level in all_levels) / len(all_levels)
            consistent_levels = sum(1 for level in all_levels if level.volume >= avg_volume * 0.5)
            consistency_score = consistent_levels / len(all_levels)
        else:
            consistency_score = 0.0
        scores.append(consistency_score * self._liquidity_weights['consistency_score'])
        
        # Efficiency score (tight spreads across levels)
        if len(bid_levels) > 1 and len(ask_levels) > 1:
            # Check spread consistency across levels
            spreads = []
            for i in range(min(3, len(bid_levels), len(ask_levels))):  # Top 3 levels
                spread = ask_levels[i].price - bid_levels[i].price
                spreads.append(float(spread))
            
            if spreads:
                avg_spread = sum(spreads) / len(spreads)
                spread_consistency = 1.0 - (max(spreads) - min(spreads)) / max(avg_spread, 0.001)
                efficiency_score = max(0.0, spread_consistency)
            else:
                efficiency_score = 0.0
        else:
            efficiency_score = 0.5  # Neutral score for insufficient data
        scores.append(efficiency_score * self._liquidity_weights['efficiency_score'])
        
        return max(0.0, min(1.0, sum(scores)))
    
    def _calculate_mid_price(self, bid_levels: List[OrderBookLevel], ask_levels: List[OrderBookLevel]) -> Decimal:
        """Calculate mid price from order book levels."""
        if bid_levels and ask_levels:
            return (bid_levels[0].price + ask_levels[0].price) / 2
        elif bid_levels:
            return bid_levels[0].price
        elif ask_levels:
            return ask_levels[0].price
        else:
            return Decimal('100')  # Default fallback
    
    def _classify_liquidity_condition(self, metrics: LiquidityMetrics) -> MarketCondition:
        """Classify market condition based on liquidity metrics."""
        if metrics.liquidity_score < 0.3:
            return MarketCondition.STRESSED
        elif metrics.total_depth < self.min_volume_threshold:
            return MarketCondition.ILLIQUID
        elif abs(metrics.depth_imbalance) > 0.5:
            return MarketCondition.VOLATILE
        else:
            return MarketCondition.NORMAL
    
    def _store_liquidity_data(self, symbol: str, profile: LiquidityProfile) -> None:
        """Store liquidity profile in history."""
        if symbol not in self._liquidity_history:
            self._liquidity_history[symbol] = deque(maxlen=1000)  # Keep last 1000 profiles
        
        self._liquidity_history[symbol].append(profile)
    
    def _get_cached_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached order book data."""
        if symbol in self._order_book_cache:
            data, timestamp = self._order_book_cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < 5:  # 5 second cache
                return data
        return None
    
    def _create_mock_order_book(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create mock order book for testing."""
        # Use market data if available
        if market_data and 'bid' in market_data and 'ask' in market_data:
            bid_price = float(market_data['bid'])
            ask_price = float(market_data['ask'])
        else:
            bid_price = 100.0
            ask_price = 100.05
        
        # Create simple order book
        return {
            'bids': [
                {'price': bid_price, 'volume': 1000},
                {'price': bid_price - 0.01, 'volume': 2000},
                {'price': bid_price - 0.02, 'volume': 1500}
            ],
            'asks': [
                {'price': ask_price, 'volume': 1000},
                {'price': ask_price + 0.01, 'volume': 2000},
                {'price': ask_price + 0.02, 'volume': 1500}
            ]
        }
    
    def _get_condition_distribution(self, profiles: List[LiquidityProfile]) -> Dict[str, int]:
        """Get distribution of market conditions."""
        distribution = {condition.value: 0 for condition in MarketCondition}
        
        for profile in profiles:
            distribution[profile.market_condition.value] += 1
        
        return distribution


logger.info("Liquidity analyzer class loaded successfully")