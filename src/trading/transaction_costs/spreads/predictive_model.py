"""
Predictive Spread Model
======================

Machine learning-based spread prediction model using market volatility indicators,
trading volume patterns, time-of-day effects, and market conditions.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
import math
from dataclasses import dataclass
from collections import deque

from .base_spread_model import (
    BaseSpreadModel, SpreadEstimate, SpreadData, 
    MarketCondition
)

logger = logging.getLogger(__name__)


@dataclass
class MarketFeatures:
    """Market features for prediction model."""
    volatility_1h: float
    volatility_1d: float
    volume_ratio: float  # Current vs average volume
    time_of_day: float  # Hour of day normalized to [0,1]
    day_of_week: float  # Day of week normalized to [0,1]
    price_momentum: float  # Recent price change
    spread_momentum: float  # Recent spread change
    market_stress_indicator: float  # Composite stress measure


@dataclass
class PredictionResult:
    """Prediction model result."""
    predicted_spread: Decimal
    confidence: float
    features_used: MarketFeatures
    model_version: str
    prediction_horizon: str  # e.g., "5min", "1h"


class PredictiveSpreadModel(BaseSpreadModel):
    """
    Machine learning-based spread prediction model.
    
    Features:
    - Market volatility indicators
    - Trading volume patterns
    - Time of day effects
    - Market maker presence indicators
    - News and event impact estimation
    """
    
    def __init__(
        self,
        prediction_horizons: Optional[List[str]] = None,
        feature_window_minutes: int = 60,
        min_training_samples: int = 500,
        **kwargs
    ):
        """
        Initialize predictive spread model.
        
        Args:
            prediction_horizons: Time horizons for predictions
            feature_window_minutes: Window for feature calculation
            min_training_samples: Minimum samples for model training
        """
        super().__init__(
            model_name="PredictiveSpreadModel",
            version="1.0.0",
            **kwargs
        )
        
        self.prediction_horizons = prediction_horizons or ["5min", "15min", "1h"]
        self.feature_window_minutes = feature_window_minutes
        self.min_training_samples = min_training_samples
        
        # Model state
        self._models: Dict[str, Dict[str, Any]] = {}  # horizon -> model parameters
        self._training_data: Dict[str, deque] = {horizon: deque(maxlen=10000) 
                                                 for horizon in self.prediction_horizons}
        self._feature_cache: Dict[str, Tuple[MarketFeatures, datetime]] = {}
        
        # Model parameters (simple linear model for now)
        self._default_weights = {
            'volatility_1h': 0.3,
            'volatility_1d': 0.2,
            'volume_ratio': -0.1,  # Higher volume -> lower spread
            'time_of_day': 0.1,
            'day_of_week': 0.05,
            'price_momentum': 0.15,
            'spread_momentum': 0.4,
            'market_stress_indicator': 0.25,
            'intercept': 0.001  # Base spread
        }
        
        # Initialize models
        for horizon in self.prediction_horizons:
            self._models[horizon] = {
                'weights': self._default_weights.copy(),
                'bias': 0.0,
                'last_trained': None,
                'training_samples': 0,
                'performance_metrics': {}
            }
        
        logger.info("Predictive spread model initialized")
    
    def estimate_spread(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[SpreadData]] = None
    ) -> SpreadEstimate:
        """
        Predict spread using machine learning model.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Historical spread data
            
        Returns:
            ML-based spread prediction
        """
        try:
            # Extract features
            features = self._extract_features(symbol, market_data, historical_data)
            
            # Get best available model (shortest horizon with sufficient training)
            best_horizon = self._select_best_model(symbol)
            
            # Make prediction
            prediction = self._predict_spread(features, best_horizon)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(symbol, features, best_horizon)
            
            # Determine market condition
            market_condition = self._classify_market_condition_from_features(features)
            
            # Convert to basis points
            mid_price = self._get_mid_price(market_data)
            spread_bps = (prediction.predicted_spread / mid_price) * Decimal('10000')
            
            estimate = SpreadEstimate(
                symbol=symbol,
                estimated_spread=prediction.predicted_spread,
                spread_bps=spread_bps,
                confidence_level=confidence,
                estimation_method=f"ml_prediction_{best_horizon}",
                timestamp=datetime.now(),
                market_condition=market_condition,
                supporting_data={
                    'model_horizon': best_horizon,
                    'features': {
                        'volatility_1h': features.volatility_1h,
                        'volatility_1d': features.volatility_1d,
                        'volume_ratio': features.volume_ratio,
                        'time_of_day': features.time_of_day,
                        'market_stress': features.market_stress_indicator
                    },
                    'model_training_samples': self._models[best_horizon]['training_samples'],
                    'prediction_confidence': prediction.confidence
                }
            )
            
            return estimate
            
        except Exception as e:
            logger.error(f"Failed to predict spread for {symbol}: {e}")
            raise
    
    def train_model(
        self,
        symbol: str,
        training_data: List[SpreadData],
        horizon: str = "5min"
    ) -> Dict[str, Any]:
        """
        Train prediction model for a specific horizon.
        
        Args:
            symbol: Trading symbol
            training_data: Historical spread data for training
            horizon: Prediction horizon
            
        Returns:
            Training results and metrics
        """
        if horizon not in self.prediction_horizons:
            raise ValueError(f"Unsupported horizon: {horizon}")
        
        if len(training_data) < self.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(training_data)} < {self.min_training_samples}")
        
        # Prepare training dataset
        X, y = self._prepare_training_data(training_data, horizon)
        
        if len(X) < self.min_training_samples:
            raise ValueError(f"Insufficient valid training samples: {len(X)}")
        
        # Train simple linear model
        weights, bias, metrics = self._train_linear_model(X, y)
        
        # Update model
        self._models[horizon].update({
            'weights': weights,
            'bias': bias,
            'last_trained': datetime.now(),
            'training_samples': len(X),
            'performance_metrics': metrics
        })
        
        logger.info(f"Trained model for {symbol} horizon {horizon}: {len(X)} samples, R²={metrics.get('r_squared', 0):.3f}")
        
        return {
            'horizon': horizon,
            'training_samples': len(X),
            'performance_metrics': metrics,
            'model_weights': weights
        }
    
    def add_training_sample(
        self,
        symbol: str,
        features: MarketFeatures,
        actual_spread: Decimal,
        horizon: str
    ) -> None:
        """
        Add a training sample for online learning.
        
        Args:
            symbol: Trading symbol
            features: Market features
            actual_spread: Actual observed spread
            horizon: Prediction horizon
        """
        if horizon in self._training_data:
            sample = (features, float(actual_spread))
            self._training_data[horizon].append(sample)
            
            # Trigger retraining if enough new samples
            if len(self._training_data[horizon]) % 100 == 0:
                self._incremental_update(horizon)
    
    def predict_spread_multiple_horizons(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[SpreadData]] = None
    ) -> Dict[str, PredictionResult]:
        """
        Predict spread for multiple time horizons.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            historical_data: Historical data
            
        Returns:
            Predictions for each horizon
        """
        features = self._extract_features(symbol, market_data, historical_data)
        predictions = {}
        
        for horizon in self.prediction_horizons:
            if self._models[horizon]['training_samples'] > 0:
                prediction = self._predict_spread(features, horizon)
                predictions[horizon] = prediction
        
        return predictions
    
    def validate_spread_data(self, spread_data: SpreadData) -> bool:
        """Validate data for prediction model."""
        if not spread_data or not spread_data.symbol:
            return False
        
        # Need timestamp for feature extraction
        if not spread_data.timestamp:
            return False
        
        # Need valid spread or bid/ask prices
        if spread_data.spread is None:
            if not (spread_data.bid_price and spread_data.ask_price):
                return False
        
        return True
    
    def get_supported_symbols(self) -> List[str]:
        """Get symbols with trained models."""
        # For now, return symbols from feature cache
        return list(set(key.split('_')[0] for key in self._feature_cache.keys()))
    
    def get_model_performance(self, horizon: str) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Performance metrics
        """
        if horizon not in self._models:
            return {}
        
        model = self._models[horizon]
        return {
            'horizon': horizon,
            'training_samples': model['training_samples'],
            'last_trained': model['last_trained'],
            'performance_metrics': model['performance_metrics'],
            'model_weights': model['weights']
        }
    
    # Private methods
    
    def _extract_features(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]],
        historical_data: Optional[List[SpreadData]]
    ) -> MarketFeatures:
        """Extract features for prediction."""
        # Check cache first
        cache_key = f"{symbol}_{int(datetime.now().timestamp() // 60)}"  # 1-minute cache
        if cache_key in self._feature_cache:
            features, cache_time = self._feature_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 60:
                return features
        
        # Extract features
        now = datetime.now()
        
        # Time-based features
        time_of_day = now.hour / 24.0
        day_of_week = now.weekday() / 6.0
        
        # Default values
        volatility_1h = 0.01
        volatility_1d = 0.02
        volume_ratio = 1.0
        price_momentum = 0.0
        spread_momentum = 0.0
        market_stress = 0.0
        
        # Extract from market data if available
        if market_data:
            volatility_1h = market_data.get('volatility_1h', volatility_1h)
            volatility_1d = market_data.get('volatility_1d', volatility_1d)
            volume_ratio = market_data.get('volume_ratio', volume_ratio)
            price_momentum = market_data.get('price_momentum', price_momentum)
        
        # Extract from historical data if available
        if historical_data and len(historical_data) > 1:
            recent_data = sorted(historical_data, key=lambda x: x.timestamp or datetime.min)[-10:]
            
            # Calculate spread momentum
            if len(recent_data) >= 2:
                recent_spreads = [float(d.spread) for d in recent_data if d.spread]
                if len(recent_spreads) >= 2:
                    spread_momentum = (recent_spreads[-1] - recent_spreads[0]) / max(recent_spreads[0], 0.001)
            
            # Calculate market stress indicator
            market_stress = self._calculate_market_stress(recent_data)
        
        features = MarketFeatures(
            volatility_1h=volatility_1h,
            volatility_1d=volatility_1d,
            volume_ratio=volume_ratio,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            price_momentum=price_momentum,
            spread_momentum=spread_momentum,
            market_stress_indicator=market_stress
        )
        
        # Cache features
        self._feature_cache[cache_key] = (features, datetime.now())
        
        return features
    
    def _select_best_model(self, symbol: str) -> str:
        """Select best available model for prediction."""
        # Return shortest horizon with sufficient training
        for horizon in sorted(self.prediction_horizons):
            model = self._models[horizon]
            if model['training_samples'] >= self.min_training_samples // 10:  # Relaxed threshold
                return horizon
        
        # Fallback to first horizon
        return self.prediction_horizons[0]
    
    def _predict_spread(self, features: MarketFeatures, horizon: str) -> PredictionResult:
        """Make spread prediction using trained model."""
        model = self._models[horizon]
        weights = model['weights']
        bias = model['bias']
        
        # Feature vector
        feature_vector = [
            features.volatility_1h,
            features.volatility_1d,
            features.volume_ratio,
            features.time_of_day,
            features.day_of_week,
            features.price_momentum,
            features.spread_momentum,
            features.market_stress_indicator,
            1.0  # intercept
        ]
        
        # Linear prediction
        prediction = bias
        for i, (key, weight) in enumerate(weights.items()):
            if i < len(feature_vector):
                prediction += weight * feature_vector[i]
        
        # Ensure positive prediction
        prediction = max(0.0001, prediction)
        
        # Calculate model confidence
        confidence = min(1.0, model['training_samples'] / self.min_training_samples)
        
        return PredictionResult(
            predicted_spread=Decimal(str(prediction)),
            confidence=confidence,
            features_used=features,
            model_version=f"linear_v1_{horizon}",
            prediction_horizon=horizon
        )
    
    def _calculate_prediction_confidence(
        self,
        symbol: str,
        features: MarketFeatures,
        horizon: str
    ) -> float:
        """Calculate confidence in the prediction."""
        base_confidence = 0.7
        
        # Increase confidence with more training data
        model = self._models[horizon]
        training_samples = model['training_samples']
        
        if training_samples >= self.min_training_samples:
            base_confidence += 0.2
        elif training_samples >= self.min_training_samples // 2:
            base_confidence += 0.1
        
        # Decrease confidence in stressed market conditions
        if features.market_stress_indicator > 0.5:
            base_confidence -= 0.2
        
        # Decrease confidence with high volatility
        if features.volatility_1h > 0.05:  # 5% hourly volatility
            base_confidence -= 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _classify_market_condition_from_features(self, features: MarketFeatures) -> MarketCondition:
        """Classify market condition from features."""
        if features.market_stress_indicator > 0.7:
            return MarketCondition.STRESSED
        elif features.volatility_1h > 0.03 or features.volatility_1d > 0.05:
            return MarketCondition.VOLATILE
        elif features.volume_ratio < 0.5:
            return MarketCondition.ILLIQUID
        else:
            return MarketCondition.NORMAL
    
    def _get_mid_price(self, market_data: Optional[Dict[str, Any]]) -> Decimal:
        """Get mid price from market data."""
        if market_data:
            if 'mid_price' in market_data:
                return Decimal(str(market_data['mid_price']))
            elif 'bid' in market_data and 'ask' in market_data:
                bid = Decimal(str(market_data['bid']))
                ask = Decimal(str(market_data['ask']))
                return (bid + ask) / 2
        
        return Decimal('100')  # Default fallback
    
    def _prepare_training_data(
        self,
        data: List[SpreadData],
        horizon: str
    ) -> Tuple[List[List[float]], List[float]]:
        """Prepare training dataset."""
        X = []
        y = []
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp or datetime.min)
        
        # Create feature-target pairs
        for i in range(len(sorted_data) - 1):
            current = sorted_data[i]
            
            # Find target based on horizon
            horizon_minutes = self._parse_horizon_minutes(horizon)
            target_time = current.timestamp + timedelta(minutes=horizon_minutes)
            
            # Find closest future data point
            target = None
            for j in range(i + 1, len(sorted_data)):
                if sorted_data[j].timestamp >= target_time:
                    target = sorted_data[j]
                    break
            
            if target and target.spread:
                # Extract features from current data
                features = self._extract_features_from_data(current, sorted_data[:i+1])
                feature_vector = [
                    features.volatility_1h,
                    features.volatility_1d,
                    features.volume_ratio,
                    features.time_of_day,
                    features.day_of_week,
                    features.price_momentum,
                    features.spread_momentum,
                    features.market_stress_indicator
                ]
                
                X.append(feature_vector)
                y.append(float(target.spread))
        
        return X, y
    
    def _extract_features_from_data(
        self,
        current: SpreadData,
        historical: List[SpreadData]
    ) -> MarketFeatures:
        """Extract features from historical data point."""
        # Basic time features
        timestamp = current.timestamp or datetime.now()
        time_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 6.0
        
        # Calculate features from historical data
        volatility_1h = self._calculate_volatility(historical, hours=1)
        volatility_1d = self._calculate_volatility(historical, hours=24)
        volume_ratio = self._calculate_volume_ratio(current, historical)
        price_momentum = self._calculate_price_momentum(historical)
        spread_momentum = self._calculate_spread_momentum(historical)
        market_stress = self._calculate_market_stress(historical)
        
        return MarketFeatures(
            volatility_1h=volatility_1h,
            volatility_1d=volatility_1d,
            volume_ratio=volume_ratio,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            price_momentum=price_momentum,
            spread_momentum=spread_momentum,
            market_stress_indicator=market_stress
        )
    
    def _calculate_volatility(self, data: List[SpreadData], hours: int) -> float:
        """Calculate price volatility over specified hours."""
        if len(data) < 2:
            return 0.01
        
        # Filter data within time window
        cutoff_time = data[-1].timestamp - timedelta(hours=hours) if data[-1].timestamp else datetime.min
        recent_data = [d for d in data if d.timestamp and d.timestamp >= cutoff_time]
        
        if len(recent_data) < 2:
            return 0.01
        
        # Calculate returns from mid prices
        returns = []
        for i in range(1, len(recent_data)):
            prev_data = recent_data[i-1]
            curr_data = recent_data[i]
            
            if prev_data.bid_price and prev_data.ask_price and curr_data.bid_price and curr_data.ask_price:
                prev_mid = (prev_data.bid_price + prev_data.ask_price) / 2
                curr_mid = (curr_data.bid_price + curr_data.ask_price) / 2
                
                if prev_mid > 0:
                    return_val = float((curr_mid - prev_mid) / prev_mid)
                    returns.append(return_val)
        
        if not returns:
            return 0.01
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return math.sqrt(variance)
    
    def _calculate_volume_ratio(self, current: SpreadData, historical: List[SpreadData]) -> float:
        """Calculate current volume ratio to historical average."""
        if not current.volume:
            return 1.0
        
        historical_volumes = [d.volume for d in historical if d.volume]
        if not historical_volumes:
            return 1.0
        
        avg_volume = sum(historical_volumes) / len(historical_volumes)
        if avg_volume == 0:
            return 1.0
        
        return current.volume / avg_volume
    
    def _calculate_price_momentum(self, data: List[SpreadData]) -> float:
        """Calculate recent price momentum."""
        if len(data) < 2:
            return 0.0
        
        # Use last 5 data points
        recent_data = data[-5:]
        if len(recent_data) < 2:
            return 0.0
        
        # Calculate momentum from mid prices
        first = recent_data[0]
        last = recent_data[-1]
        
        if (first.bid_price and first.ask_price and 
            last.bid_price and last.ask_price):
            
            first_mid = (first.bid_price + first.ask_price) / 2
            last_mid = (last.bid_price + last.ask_price) / 2
            
            if first_mid > 0:
                return float((last_mid - first_mid) / first_mid)
        
        return 0.0
    
    def _calculate_spread_momentum(self, data: List[SpreadData]) -> float:
        """Calculate recent spread momentum."""
        if len(data) < 2:
            return 0.0
        
        # Use last 5 data points with spreads
        recent_spreads = [d.spread for d in data[-5:] if d.spread]
        if len(recent_spreads) < 2:
            return 0.0
        
        first_spread = float(recent_spreads[0])
        last_spread = float(recent_spreads[-1])
        
        if first_spread > 0:
            return (last_spread - first_spread) / first_spread
        
        return 0.0
    
    def _calculate_market_stress(self, data: List[SpreadData]) -> float:
        """Calculate market stress indicator."""
        if len(data) < 5:
            return 0.0
        
        # Combine spread volatility and volume patterns
        spreads = [float(d.spread) for d in data if d.spread]
        volumes = [d.volume for d in data if d.volume]
        
        stress = 0.0
        
        # Spread volatility component
        if len(spreads) > 1:
            mean_spread = sum(spreads) / len(spreads)
            spread_std = math.sqrt(sum((s - mean_spread) ** 2 for s in spreads) / len(spreads))
            if mean_spread > 0:
                spread_cv = spread_std / mean_spread
                stress += min(1.0, spread_cv * 2)  # Normalize
        
        # Volume volatility component
        if len(volumes) > 1:
            mean_volume = sum(volumes) / len(volumes)
            volume_std = math.sqrt(sum((v - mean_volume) ** 2 for v in volumes) / len(volumes))
            if mean_volume > 0:
                volume_cv = volume_std / mean_volume
                stress += min(1.0, volume_cv)
        
        return min(1.0, stress / 2)  # Average and cap at 1.0
    
    def _parse_horizon_minutes(self, horizon: str) -> int:
        """Parse horizon string to minutes."""
        if horizon.endswith('min'):
            return int(horizon[:-3])
        elif horizon.endswith('h'):
            return int(horizon[:-1]) * 60
        else:
            return 5  # Default 5 minutes
    
    def _train_linear_model(self, X: List[List[float]], y: List[float]) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        """Train simple linear regression model."""
        if len(X) != len(y) or len(X) == 0:
            return self._default_weights.copy(), 0.0, {}
        
        # Simple gradient descent implementation
        n_features = len(X[0])
        weights = [0.1] * n_features
        bias = 0.0
        learning_rate = 0.01
        epochs = 100
        
        for epoch in range(epochs):
            # Forward pass
            predictions = []
            for i in range(len(X)):
                pred = bias + sum(w * x for w, x in zip(weights, X[i]))
                predictions.append(pred)
            
            # Calculate gradients
            bias_grad = 0
            weight_grads = [0] * n_features
            
            for i in range(len(X)):
                error = predictions[i] - y[i]
                bias_grad += error
                for j in range(n_features):
                    weight_grads[j] += error * X[i][j]
            
            # Update parameters
            bias -= learning_rate * bias_grad / len(X)
            for j in range(n_features):
                weights[j] -= learning_rate * weight_grads[j] / len(X)
        
        # Calculate R-squared
        y_mean = sum(y) / len(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((predictions[i] - y[i]) ** 2 for i in range(len(y)))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Create weights dictionary
        feature_names = list(self._default_weights.keys())[:-1]  # Exclude intercept
        weights_dict = dict(zip(feature_names, weights))
        weights_dict['intercept'] = bias
        
        metrics = {
            'r_squared': r_squared,
            'mse': ss_res / len(y),
            'training_samples': len(X)
        }
        
        return weights_dict, bias, metrics
    
    def _incremental_update(self, horizon: str) -> None:
        """Perform incremental model update."""
        if len(self._training_data[horizon]) < 50:
            return
        
        # Simple online learning: adjust weights based on recent errors
        recent_samples = list(self._training_data[horizon])[-50:]
        
        model = self._models[horizon]
        weights = model['weights']
        
        # Calculate average error
        total_error = 0
        for features, actual_spread in recent_samples:
            prediction = self._predict_spread(features, horizon)
            error = float(prediction.predicted_spread) - actual_spread
            total_error += error
        
        avg_error = total_error / len(recent_samples)
        
        # Simple bias correction
        model['bias'] -= avg_error * 0.1
        
        logger.debug(f"Incremental update for horizon {horizon}: avg_error={avg_error:.4f}")


logger.info("Predictive spread model class loaded successfully")