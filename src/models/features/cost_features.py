"""
Cost Feature Generator Module
============================

Generates transaction cost-related features for enhanced ML model training.
Provides features like historical average costs, cost volatility, cost-to-return ratios,
broker efficiency metrics, market impact sensitivity, and liquidity-adjusted features.
"""

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    PANDAS_AVAILABLE = False
    # Create mock classes for when dependencies aren't available
    class np:
        @staticmethod
        def array(x): return x
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): return 0
        @staticmethod
        def random(): 
            class _random:
                @staticmethod
                def normal(mean, std, size): return [mean] * size
            return _random()
        inf = float('inf')
        nan = float('nan')
        
    class pd:
        class DataFrame:
            def __init__(self, data=None): 
                self.data = data or {}
                self.columns = list(self.data.keys()) if isinstance(self.data, dict) else []
            def copy(self): return self
            def __len__(self): return 1
        class Series:
            def __init__(self, data=None): pass
            def rolling(self, **kwargs): return self
            def mean(self): return 0
            def pct_change(self): return self

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class CostFeatureType(Enum):
    """Enum for different types of cost features."""
    HISTORICAL_AVERAGE = "historical_average"
    VOLATILITY = "volatility"
    COST_TO_RETURN = "cost_to_return"
    BROKER_EFFICIENCY = "broker_efficiency"
    MARKET_IMPACT = "market_impact"
    LIQUIDITY_ADJUSTED = "liquidity_adjusted"

@dataclass
class CostFeatureConfig:
    """Configuration for cost feature generation."""
    lookback_windows: List[int] = None
    enable_volatility: bool = True
    enable_efficiency_metrics: bool = True
    enable_market_impact: bool = True
    enable_liquidity_features: bool = True
    min_periods: int = 5
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 50]

class CostFeatureGenerator:
    """
    Generates comprehensive cost-related features for ML model enhancement.
    
    This class creates features that help models understand transaction cost patterns
    and their impact on trading performance.
    """
    
    def __init__(self, config: Optional[CostFeatureConfig] = None):
        """
        Initialize cost feature generator.
        
        Args:
            config: Configuration for feature generation
        """
        self.config = config or CostFeatureConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        self.generated_features = []
        
        self.logger.info("CostFeatureGenerator initialized")
    
    def validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame for cost feature generation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If data is invalid
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Check for required base columns
        base_cols = ['close', 'volume'] if 'volume' in df.columns else ['close']
        missing_cols = [col for col in base_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing base columns: {missing_cols}, will generate limited features")
        
        # Check for transaction cost data
        cost_cols = ['transaction_cost', 'bid_ask_spread', 'market_impact', 'commission']
        available_cost_cols = [col for col in cost_cols if col in df.columns]
        
        if not available_cost_cols:
            self.logger.warning("No explicit cost columns found, will generate synthetic cost features")
        
        return True
    
    def generate_historical_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate historical average transaction cost features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with historical average cost features
        """
        result_df = df.copy()
        
        try:
            # Generate synthetic transaction costs if not available
            if 'transaction_cost' not in df.columns:
                result_df = self._generate_synthetic_costs(result_df)
            
            # Calculate rolling averages for different windows
            for window in self.config.lookback_windows:
                feature_name = f'cost_avg_{window}d'
                result_df[feature_name] = result_df['transaction_cost'].rolling(
                    window=window, min_periods=self.config.min_periods
                ).mean()
                self.generated_features.append(feature_name)
                
                # Normalized by price
                price_norm_name = f'cost_avg_{window}d_pct'
                result_df[price_norm_name] = (result_df[feature_name] / result_df['close']) * 100
                self.generated_features.append(price_norm_name)
            
            # Calculate exponentially weighted moving averages
            for alpha in [0.1, 0.3, 0.5]:
                feature_name = f'cost_ewm_alpha_{alpha:.1f}'
                result_df[feature_name] = result_df['transaction_cost'].ewm(alpha=alpha).mean()
                self.generated_features.append(feature_name)
            
            self.logger.debug(f"Generated {len([f for f in self.generated_features if 'cost_avg' in f or 'cost_ewm' in f])} historical average features")
            
        except Exception as e:
            self.logger.error(f"Error generating historical average features: {e}")
        
        return result_df
    
    def generate_cost_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cost volatility indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cost volatility features
        """
        if not self.config.enable_volatility:
            return df
            
        result_df = df.copy()
        
        try:
            if 'transaction_cost' not in result_df.columns:
                result_df = self._generate_synthetic_costs(result_df)
            
            # Calculate cost volatility for different windows
            for window in self.config.lookback_windows:
                vol_name = f'cost_vol_{window}d'
                result_df[vol_name] = result_df['transaction_cost'].rolling(
                    window=window, min_periods=self.config.min_periods
                ).std()
                self.generated_features.append(vol_name)
                
                # Cost volatility relative to price volatility
                if window <= len(result_df):
                    price_vol = result_df['close'].rolling(window=window).std()
                    rel_vol_name = f'cost_vol_rel_{window}d'
                    result_df[rel_vol_name] = result_df[vol_name] / (price_vol + 1e-8)
                    self.generated_features.append(rel_vol_name)
            
            # Cost volatility regime indicators
            short_vol = result_df['transaction_cost'].rolling(window=5).std()
            long_vol = result_df['transaction_cost'].rolling(window=20).std()
            result_df['cost_vol_regime'] = short_vol / (long_vol + 1e-8)
            self.generated_features.append('cost_vol_regime')
            
            self.logger.debug(f"Generated {len([f for f in self.generated_features if 'vol' in f])} volatility features")
            
        except Exception as e:
            self.logger.error(f"Error generating cost volatility features: {e}")
        
        return result_df
    
    def generate_cost_to_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cost-to-return ratio features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cost-to-return features
        """
        result_df = df.copy()
        
        try:
            if 'transaction_cost' not in result_df.columns:
                result_df = self._generate_synthetic_costs(result_df)
            
            # Calculate returns
            result_df['returns'] = result_df['close'].pct_change()
            result_df['abs_returns'] = np.abs(result_df['returns'])
            
            # Cost-to-return ratios for different windows
            for window in self.config.lookback_windows:
                avg_cost = result_df['transaction_cost'].rolling(window=window).mean()
                avg_return = result_df['abs_returns'].rolling(window=window).mean()
                
                ratio_name = f'cost_return_ratio_{window}d'
                result_df[ratio_name] = avg_cost / (avg_return + 1e-8)
                self.generated_features.append(ratio_name)
                
                # Cost drag indicator
                drag_name = f'cost_drag_{window}d'
                result_df[drag_name] = avg_cost / (result_df['close'].rolling(window=window).mean() + 1e-8) * 100
                self.generated_features.append(drag_name)
            
            # Immediate cost-to-return ratio
            result_df['immediate_cost_return_ratio'] = result_df['transaction_cost'] / (result_df['abs_returns'] + 1e-8)
            self.generated_features.append('immediate_cost_return_ratio')
            
            self.logger.debug(f"Generated {len([f for f in self.generated_features if 'ratio' in f or 'drag' in f])} cost-to-return features")
            
        except Exception as e:
            self.logger.error(f"Error generating cost-to-return features: {e}")
        
        return result_df
    
    def generate_broker_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate broker efficiency metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with broker efficiency features
        """
        if not self.config.enable_efficiency_metrics:
            return df
            
        result_df = df.copy()
        
        try:
            if 'transaction_cost' not in result_df.columns:
                result_df = self._generate_synthetic_costs(result_df)
            
            # Calculate efficiency metrics
            for window in self.config.lookback_windows:
                # Cost efficiency (lower is better)
                cost_efficiency_name = f'broker_efficiency_{window}d'
                avg_cost = result_df['transaction_cost'].rolling(window=window).mean()
                min_cost = result_df['transaction_cost'].rolling(window=window).min()
                result_df[cost_efficiency_name] = min_cost / (avg_cost + 1e-8)
                self.generated_features.append(cost_efficiency_name)
                
                # Cost consistency (lower std relative to mean is better)
                consistency_name = f'broker_consistency_{window}d'
                cost_std = result_df['transaction_cost'].rolling(window=window).std()
                result_df[consistency_name] = cost_std / (avg_cost + 1e-8)
                self.generated_features.append(consistency_name)
            
            # Relative cost performance vs theoretical minimum
            if 'bid_ask_spread' in result_df.columns:
                result_df['cost_vs_spread'] = result_df['transaction_cost'] / (result_df['bid_ask_spread'] + 1e-8)
                self.generated_features.append('cost_vs_spread')
            else:
                # Synthetic spread estimate
                spread_estimate = result_df['close'] * 0.001  # 10 bps estimate
                result_df['cost_vs_est_spread'] = result_df['transaction_cost'] / (spread_estimate + 1e-8)
                self.generated_features.append('cost_vs_est_spread')
            
            self.logger.debug(f"Generated {len([f for f in self.generated_features if 'efficiency' in f or 'consistency' in f])} efficiency features")
            
        except Exception as e:
            self.logger.error(f"Error generating broker efficiency features: {e}")
        
        return result_df
    
    def generate_market_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market impact sensitivity features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with market impact features
        """
        if not self.config.enable_market_impact:
            return df
            
        result_df = df.copy()
        
        try:
            if 'volume' not in result_df.columns:
                self.logger.warning("Volume data not available, skipping market impact features")
                return result_df
            
            if 'transaction_cost' not in result_df.columns:
                result_df = self._generate_synthetic_costs(result_df)
            
            # Volume-adjusted cost metrics
            result_df['cost_per_volume'] = result_df['transaction_cost'] / (result_df['volume'] + 1e-8)
            self.generated_features.append('cost_per_volume')
            
            # Market impact indicators for different windows
            for window in self.config.lookback_windows:
                # Average volume impact
                vol_impact_name = f'volume_impact_{window}d'
                avg_vol = result_df['volume'].rolling(window=window).mean()
                result_df[vol_impact_name] = result_df['volume'] / (avg_vol + 1e-8)
                self.generated_features.append(vol_impact_name)
                
                # Cost sensitivity to volume
                cost_vol_corr_name = f'cost_volume_corr_{window}d'
                result_df[cost_vol_corr_name] = result_df['transaction_cost'].rolling(window=window).corr(
                    result_df['volume']
                )
                self.generated_features.append(cost_vol_corr_name)
            
            # Market microstructure indicators
            if len(result_df) > 1:
                result_df['volume_momentum'] = result_df['volume'].pct_change()
                result_df['cost_momentum'] = result_df['transaction_cost'].pct_change()
                result_df['cost_volume_momentum_ratio'] = result_df['cost_momentum'] / (result_df['volume_momentum'] + 1e-8)
                self.generated_features.extend(['volume_momentum', 'cost_momentum', 'cost_volume_momentum_ratio'])
            
            self.logger.debug(f"Generated {len([f for f in self.generated_features if 'impact' in f or 'momentum' in f])} market impact features")
            
        except Exception as e:
            self.logger.error(f"Error generating market impact features: {e}")
        
        return result_df
    
    def generate_liquidity_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate liquidity-adjusted cost features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with liquidity-adjusted features
        """
        if not self.config.enable_liquidity_features:
            return df
            
        result_df = df.copy()
        
        try:
            if 'volume' not in result_df.columns:
                self.logger.warning("Volume data not available, skipping liquidity features")
                return result_df
            
            if 'transaction_cost' not in result_df.columns:
                result_df = self._generate_synthetic_costs(result_df)
            
            # Liquidity proxies
            for window in self.config.lookback_windows:
                # Volume-based liquidity
                liquidity_name = f'liquidity_proxy_{window}d'
                result_df[liquidity_name] = result_df['volume'].rolling(window=window).mean()
                self.generated_features.append(liquidity_name)
                
                # Cost adjusted for liquidity
                liq_adj_cost_name = f'liquidity_adj_cost_{window}d'
                avg_liquidity = result_df[liquidity_name]
                result_df[liq_adj_cost_name] = result_df['transaction_cost'] * (1.0 / (avg_liquidity + 1e-8))
                self.generated_features.append(liq_adj_cost_name)
            
            # Relative liquidity indicators
            if 'high' in result_df.columns and 'low' in result_df.columns:
                # Price range as liquidity indicator
                result_df['price_range'] = (result_df['high'] - result_df['low']) / result_df['close']
                result_df['cost_range_ratio'] = result_df['transaction_cost'] / (result_df['price_range'] + 1e-8)
                self.generated_features.extend(['price_range', 'cost_range_ratio'])
            
            # Time-of-day liquidity effects (if datetime available)
            if 'datetime' in result_df.columns:
                try:
                    result_df['hour'] = pd.to_datetime(result_df['datetime']).dt.hour
                    result_df['is_market_open'] = ((result_df['hour'] >= 9) & (result_df['hour'] <= 15)).astype(int)
                    result_df['cost_market_hours'] = result_df['transaction_cost'] * result_df['is_market_open']
                    self.generated_features.extend(['hour', 'is_market_open', 'cost_market_hours'])
                except Exception as e:
                    self.logger.warning(f"Could not process datetime for liquidity features: {e}")
            
            self.logger.debug(f"Generated {len([f for f in self.generated_features if 'liquidity' in f or 'range' in f])} liquidity features")
            
        except Exception as e:
            self.logger.error(f"Error generating liquidity-adjusted features: {e}")
        
        return result_df
    
    def _generate_synthetic_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic transaction costs based on market data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with synthetic transaction costs
        """
        result_df = df.copy()
        
        try:
            # Base cost as percentage of price (typical values: 0.01% - 0.1%)
            base_cost_pct = 0.0005  # 5 basis points
            result_df['transaction_cost'] = result_df['close'] * base_cost_pct
            
            # Add volatility-based component
            if len(result_df) > 20:
                price_vol = result_df['close'].rolling(window=20).std()
                vol_cost_component = price_vol * 0.001  # Volatility increases costs
                result_df['transaction_cost'] += vol_cost_component.fillna(0)
            
            # Add volume-based component (higher volume -> lower costs)
            if 'volume' in result_df.columns and result_df['volume'].sum() > 0:
                avg_volume = result_df['volume'].rolling(window=20).mean()
                volume_discount = np.where(
                    result_df['volume'] > avg_volume,
                    0.8,  # 20% discount for high volume
                    1.2   # 20% premium for low volume
                )
                result_df['transaction_cost'] *= volume_discount
            
            # Add random noise to make it realistic
            noise = np.random.normal(1.0, 0.1, len(result_df))
            result_df['transaction_cost'] *= np.clip(noise, 0.5, 1.5)
            
            # Ensure positive costs
            result_df['transaction_cost'] = np.maximum(result_df['transaction_cost'], result_df['close'] * 0.0001)
            
            self.logger.debug("Generated synthetic transaction costs")
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic costs: {e}")
            # Fallback to simple percentage
            result_df['transaction_cost'] = result_df['close'] * 0.0005
        
        return result_df
    
    def generate_all_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all cost-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with all cost features
        """
        start_time = datetime.now()
        self.logger.info("Starting comprehensive cost feature generation")
        
        try:
            # Validate input
            self.validate_input_data(df)
            
            # Reset feature tracking
            self.generated_features = []
            
            # Generate each category of features
            result_df = df.copy()
            result_df = self.generate_historical_average_features(result_df)
            result_df = self.generate_cost_volatility_features(result_df)
            result_df = self.generate_cost_to_return_features(result_df)
            result_df = self.generate_broker_efficiency_features(result_df)
            result_df = self.generate_market_impact_features(result_df)
            result_df = self.generate_liquidity_adjusted_features(result_df)
            
            # Clean up infinite and NaN values
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with forward fill, then backward fill
            cost_feature_cols = [col for col in result_df.columns if col in self.generated_features]
            result_df[cost_feature_cols] = result_df[cost_feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Cost feature generation completed in {processing_time:.2f}s")
            self.logger.info(f"Generated {len(self.generated_features)} cost features")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive cost feature generation: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of generated cost features.
        
        Returns:
            List[str]: List of feature names
        """
        return self.generated_features.copy()
    
    def get_feature_importance_estimates(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get estimated importance of cost features based on basic statistics.
        
        Args:
            df: DataFrame with cost features
            
        Returns:
            Dict[str, float]: Feature importance estimates
        """
        importance_estimates = {}
        
        try:
            for feature in self.generated_features:
                if feature in df.columns:
                    # Simple importance based on variance and non-null percentage
                    variance = df[feature].var()
                    non_null_pct = df[feature].notna().mean()
                    
                    # Normalize variance (higher is more important)
                    normalized_variance = min(variance / (df[feature].std() + 1e-8), 10.0)
                    
                    # Combined score
                    importance_estimates[feature] = normalized_variance * non_null_pct
                    
        except Exception as e:
            self.logger.error(f"Error calculating feature importance estimates: {e}")
        
        return importance_estimates