"""
Market Parameter Configuration
==============================

Configuration management for market-related parameters used in transaction
cost calculations, including volatility windows, liquidity adjustments,
market impact models, and time-based factors.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from decimal import Decimal
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market trading sessions."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class InstrumentClass(Enum):
    """Financial instrument classes."""
    EQUITY = "equity"
    OPTIONS = "options"
    FUTURES = "futures"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    BOND = "bond"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class VolatilityConfiguration:
    """Configuration for volatility calculations."""
    # Calculation windows (in trading days)
    short_term_window: int = 5
    medium_term_window: int = 21
    long_term_window: int = 63
    
    # Volatility models
    primary_model: str = "ewma"  # ewma, garch, historical
    fallback_model: str = "historical"
    
    # EWMA parameters
    ewma_lambda: Decimal = Decimal('0.94')
    
    # Volatility floors and caps (annualized)
    min_volatility: Decimal = Decimal('0.05')  # 5% minimum
    max_volatility: Decimal = Decimal('2.0')   # 200% maximum
    
    # Intraday adjustments
    intraday_scaling_factor: Decimal = Decimal('1.5')
    overnight_adjustment: Decimal = Decimal('0.3')
    
    # Data requirements
    min_observations: int = 10
    data_staleness_threshold_minutes: int = 30


@dataclass
class LiquidityConfiguration:
    """Configuration for liquidity adjustments."""
    # Volume percentile thresholds
    high_liquidity_percentile: Decimal = Decimal('0.75')
    low_liquidity_percentile: Decimal = Decimal('0.25')
    
    # Adjustment factors
    high_liquidity_discount: Decimal = Decimal('0.85')  # 15% reduction
    low_liquidity_premium: Decimal = Decimal('1.25')    # 25% increase
    
    # Volume analysis windows
    average_volume_days: int = 20
    min_volume_observations: int = 5
    
    # Tick size considerations
    enable_tick_size_adjustment: bool = True
    large_tick_threshold: Decimal = Decimal('0.01')
    small_tick_threshold: Decimal = Decimal('0.0001')
    
    # Spread-based liquidity metrics
    bid_ask_spread_threshold: Decimal = Decimal('0.005')  # 50 bps
    depth_analysis_levels: int = 5


@dataclass
class MarketImpactConfiguration:
    """Configuration for market impact models."""
    # Model selection
    primary_model: str = "square_root"  # linear, square_root, log
    temporary_impact_decay: Decimal = Decimal('0.5')  # Half-life in minutes
    
    # Linear model parameters
    linear_coefficient: Decimal = Decimal('0.1')
    
    # Square-root model parameters
    sqrt_coefficient: Decimal = Decimal('0.5')
    sqrt_exponent: Decimal = Decimal('0.5')
    
    # Logarithmic model parameters
    log_coefficient: Decimal = Decimal('0.3')
    
    # Size thresholds (as fraction of average daily volume)
    small_trade_threshold: Decimal = Decimal('0.01')   # 1% of ADV
    large_trade_threshold: Decimal = Decimal('0.1')    # 10% of ADV
    
    # Time-based adjustments
    enable_time_of_day_adjustment: bool = True
    market_open_multiplier: Decimal = Decimal('1.3')
    market_close_multiplier: Decimal = Decimal('1.2')
    lunch_time_multiplier: Decimal = Decimal('0.9')
    
    # Cross-impact considerations
    enable_cross_impact: bool = False
    cross_impact_decay: Decimal = Decimal('0.1')


@dataclass
class SpreadConfiguration:
    """Configuration for bid-ask spread estimation."""
    # Estimation methods
    primary_method: str = "quote_based"  # quote_based, trade_based, model_based
    fallback_method: str = "model_based"
    
    # Quote-based parameters
    quote_staleness_threshold_seconds: int = 60
    quote_size_threshold: int = 100  # Minimum size for quote consideration
    
    # Trade-based parameters
    trade_window_minutes: int = 5
    min_trades_for_estimation: int = 3
    
    # Model-based parameters (Roll model)
    enable_roll_model: bool = True
    roll_model_min_observations: int = 20
    
    # Spread bounds
    min_spread_bps: Decimal = Decimal('0.5')   # 0.5 basis points
    max_spread_bps: Decimal = Decimal('500.0') # 500 basis points
    
    # Time-of-day adjustments
    market_open_spread_multiplier: Decimal = Decimal('1.5')
    market_close_spread_multiplier: Decimal = Decimal('1.3')
    overnight_spread_multiplier: Decimal = Decimal('2.0')


@dataclass
class TimeBasedConfiguration:
    """Configuration for time-based adjustments."""
    # Trading hours (in market timezone)
    market_open_time: str = "09:30:00"
    market_close_time: str = "16:00:00"
    pre_market_start: str = "04:00:00"
    after_hours_end: str = "20:00:00"
    
    # Session multipliers
    pre_market_cost_multiplier: Decimal = Decimal('1.5')
    after_hours_cost_multiplier: Decimal = Decimal('1.3')
    regular_hours_multiplier: Decimal = Decimal('1.0')
    
    # Intraday patterns
    enable_intraday_patterns: bool = True
    opening_auction_duration_minutes: int = 15
    closing_auction_duration_minutes: int = 10
    
    # Day-of-week effects
    enable_day_of_week_adjustment: bool = True
    monday_multiplier: Decimal = Decimal('1.1')
    friday_multiplier: Decimal = Decimal('1.05')
    
    # Holiday adjustments
    enable_holiday_adjustment: bool = True
    pre_holiday_multiplier: Decimal = Decimal('1.2')
    post_holiday_multiplier: Decimal = Decimal('1.1')


class MarketParameterManager:
    """
    Comprehensive market parameter configuration management.
    
    Manages configuration for volatility calculations, liquidity adjustments,
    market impact models, spread estimation, and time-based factors.
    """
    
    def __init__(self, config_base_path: Optional[str] = None):
        """
        Initialize market parameter manager.
        
        Args:
            config_base_path: Base path for configuration files
        """
        self.config_base_path = Path(config_base_path) if config_base_path else Path.cwd() / "configs" / "cost_config"
        self.market_params_path = self.config_base_path / "market_params"
        
        # Ensure directories exist
        self.config_base_path.mkdir(parents=True, exist_ok=True)
        self.market_params_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration cache
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Default configurations for different instrument classes
        self._default_configs = self._create_default_configurations()
        
        logger.info(f"MarketParameterManager initialized with base path: {self.config_base_path}")
    
    def _create_default_configurations(self) -> Dict[InstrumentClass, Dict[str, Any]]:
        """Create default configurations for different instrument classes."""
        defaults = {}
        
        # Equity configuration
        equity_config = {
            "volatility": asdict(VolatilityConfiguration()),
            "liquidity": asdict(LiquidityConfiguration()),
            "market_impact": asdict(MarketImpactConfiguration()),
            "spread": asdict(SpreadConfiguration()),
            "time_based": asdict(TimeBasedConfiguration())
        }
        defaults[InstrumentClass.EQUITY] = equity_config
        
        # Options configuration (higher volatility, wider spreads)
        options_volatility = VolatilityConfiguration(
            intraday_scaling_factor=Decimal('2.0'),
            overnight_adjustment=Decimal('0.5')
        )
        options_spread = SpreadConfiguration(
            min_spread_bps=Decimal('2.0'),
            max_spread_bps=Decimal('1000.0'),
            market_open_spread_multiplier=Decimal('2.0')
        )
        options_config = equity_config.copy()
        options_config.update({
            "volatility": asdict(options_volatility),
            "spread": asdict(options_spread)
        })
        defaults[InstrumentClass.OPTIONS] = options_config
        
        # ETF configuration (generally more liquid)
        etf_liquidity = LiquidityConfiguration(
            high_liquidity_discount=Decimal('0.9'),  # Smaller discount
            low_liquidity_premium=Decimal('1.1')     # Smaller premium
        )
        etf_config = equity_config.copy()
        etf_config.update({
            "liquidity": asdict(etf_liquidity)
        })
        defaults[InstrumentClass.ETF] = etf_config
        
        # Copy equity config for other instrument types
        for instrument_type in [InstrumentClass.FUTURES, InstrumentClass.MUTUAL_FUND, 
                               InstrumentClass.BOND, InstrumentClass.FOREX, InstrumentClass.CRYPTO]:
            defaults[instrument_type] = equity_config.copy()
        
        return defaults
    
    def get_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get market parameter configuration for an instrument class.
        
        Args:
            instrument_class: Type of financial instrument
            symbol: Optional specific symbol for custom configuration
            
        Returns:
            Complete market parameter configuration
        """
        config_key = f"{instrument_class.value}_{symbol}" if symbol else instrument_class.value
        
        # Check cache first
        if config_key in self._configs:
            return self._configs[config_key]
        
        # Try to load from file
        config_file = self.market_params_path / f"{config_key}.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self._configs[config_key] = config
                logger.debug(f"Loaded market parameter configuration for {config_key}")
                return config
                
            except Exception as e:
                logger.error(f"Error loading market parameter configuration for {config_key}: {e}")
        
        # Fall back to default configuration
        default_config = self._default_configs.get(instrument_class, self._default_configs[InstrumentClass.EQUITY])
        config = self._convert_decimal_strings(default_config.copy())
        
        # Cache the configuration
        self._configs[config_key] = config
        
        logger.debug(f"Using default market parameter configuration for {config_key}")
        return config
    
    def _convert_decimal_strings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Decimal objects to strings for JSON serialization."""
        if isinstance(config, dict):
            return {k: self._convert_decimal_strings(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._convert_decimal_strings(item) for item in config]
        elif isinstance(config, Decimal):
            return str(config)
        else:
            return config
    
    def save_configuration(
        self,
        instrument_class: InstrumentClass,
        config: Dict[str, Any],
        symbol: Optional[str] = None
    ) -> None:
        """
        Save market parameter configuration to file.
        
        Args:
            instrument_class: Type of financial instrument
            config: Configuration to save
            symbol: Optional specific symbol
        """
        config_key = f"{instrument_class.value}_{symbol}" if symbol else instrument_class.value
        config_file = self.market_params_path / f"{config_key}.json"
        
        try:
            # Add metadata
            config_with_metadata = config.copy()
            config_with_metadata.update({
                "instrument_class": instrument_class.value,
                "symbol": symbol,
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0"
            })
            
            with open(config_file, 'w') as f:
                json.dump(config_with_metadata, f, indent=2, default=str)
            
            # Update cache
            self._configs[config_key] = config_with_metadata
            
            logger.info(f"Saved market parameter configuration for {config_key}")
            
        except Exception as e:
            logger.error(f"Error saving market parameter configuration for {config_key}: {e}")
            raise
    
    def get_volatility_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: Optional[str] = None
    ) -> VolatilityConfiguration:
        """Get volatility configuration for an instrument."""
        config = self.get_configuration(instrument_class, symbol)
        vol_config = config.get("volatility", {})
        
        # Convert string decimals back to Decimal objects
        converted_config = {}
        for key, value in vol_config.items():
            if isinstance(value, str) and key.endswith(('_lambda', '_factor', '_adjustment', '_volatility')):
                try:
                    converted_config[key] = Decimal(value)
                except:
                    converted_config[key] = value
            else:
                converted_config[key] = value
        
        return VolatilityConfiguration(**converted_config)
    
    def get_liquidity_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: Optional[str] = None
    ) -> LiquidityConfiguration:
        """Get liquidity configuration for an instrument."""
        config = self.get_configuration(instrument_class, symbol)
        liq_config = config.get("liquidity", {})
        
        # Convert string decimals back to Decimal objects
        converted_config = {}
        for key, value in liq_config.items():
            if isinstance(value, str) and key.endswith(('_percentile', '_discount', '_premium', '_threshold')):
                try:
                    converted_config[key] = Decimal(value)
                except:
                    converted_config[key] = value
            else:
                converted_config[key] = value
        
        return LiquidityConfiguration(**converted_config)
    
    def get_market_impact_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: Optional[str] = None
    ) -> MarketImpactConfiguration:
        """Get market impact configuration for an instrument."""
        config = self.get_configuration(instrument_class, symbol)
        impact_config = config.get("market_impact", {})
        
        # Convert string decimals back to Decimal objects
        converted_config = {}
        for key, value in impact_config.items():
            if isinstance(value, str) and key.endswith(('_decay', '_coefficient', '_exponent', '_threshold', '_multiplier')):
                try:
                    converted_config[key] = Decimal(value)
                except:
                    converted_config[key] = value
            else:
                converted_config[key] = value
        
        return MarketImpactConfiguration(**converted_config)
    
    def get_spread_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: Optional[str] = None
    ) -> SpreadConfiguration:
        """Get spread configuration for an instrument."""
        config = self.get_configuration(instrument_class, symbol)
        spread_config = config.get("spread", {})
        
        # Convert string decimals back to Decimal objects
        converted_config = {}
        for key, value in spread_config.items():
            if isinstance(value, str) and key.endswith(('_bps', '_multiplier')):
                try:
                    converted_config[key] = Decimal(value)
                except:
                    converted_config[key] = value
            else:
                converted_config[key] = value
        
        return SpreadConfiguration(**converted_config)
    
    def get_time_based_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: Optional[str] = None
    ) -> TimeBasedConfiguration:
        """Get time-based configuration for an instrument."""
        config = self.get_configuration(instrument_class, symbol)
        time_config = config.get("time_based", {})
        
        # Convert string decimals back to Decimal objects
        converted_config = {}
        for key, value in time_config.items():
            if isinstance(value, str) and key.endswith('_multiplier'):
                try:
                    converted_config[key] = Decimal(value)
                except:
                    converted_config[key] = value
            else:
                converted_config[key] = value
        
        return TimeBasedConfiguration(**converted_config)
    
    def update_volatility_configuration(
        self,
        instrument_class: InstrumentClass,
        volatility_config: VolatilityConfiguration,
        symbol: Optional[str] = None
    ) -> None:
        """Update volatility configuration for an instrument."""
        current_config = self.get_configuration(instrument_class, symbol)
        current_config["volatility"] = asdict(volatility_config)
        self.save_configuration(instrument_class, current_config, symbol)
        
        logger.info(f"Updated volatility configuration for {instrument_class.value}{f'_{symbol}' if symbol else ''}")
    
    def create_symbol_specific_configuration(
        self,
        instrument_class: InstrumentClass,
        symbol: str,
        overrides: Dict[str, Any]
    ) -> None:
        """
        Create symbol-specific configuration with overrides.
        
        Args:
            instrument_class: Type of financial instrument
            symbol: Specific symbol
            overrides: Configuration overrides
        """
        # Start with default configuration
        base_config = self.get_configuration(instrument_class)
        
        # Apply overrides
        symbol_config = base_config.copy()
        self._deep_merge(symbol_config, overrides)
        
        # Save symbol-specific configuration
        self.save_configuration(instrument_class, symbol_config, symbol)
        
        logger.info(f"Created symbol-specific configuration for {symbol}")
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_current_market_session(
        self,
        current_time: Optional[datetime] = None,
        instrument_class: InstrumentClass = InstrumentClass.EQUITY
    ) -> MarketSession:
        """
        Determine current market session based on time.
        
        Args:
            current_time: Current time (uses now() if None)
            instrument_class: Instrument class for time configuration
            
        Returns:
            Current market session
        """
        if current_time is None:
            current_time = datetime.now()
        
        time_config = self.get_time_based_configuration(instrument_class)
        
        current_time_only = current_time.time()
        market_open = time.fromisoformat(time_config.market_open_time)
        market_close = time.fromisoformat(time_config.market_close_time)
        pre_market_start = time.fromisoformat(time_config.pre_market_start)
        after_hours_end = time.fromisoformat(time_config.after_hours_end)
        
        # Check if market is closed (weekends, late night)
        if current_time.weekday() >= 5:  # Saturday or Sunday
            return MarketSession.CLOSED
        
        if current_time_only < pre_market_start or current_time_only > after_hours_end:
            return MarketSession.CLOSED
        
        # Check specific sessions
        if pre_market_start <= current_time_only < market_open:
            return MarketSession.PRE_MARKET
        elif market_open <= current_time_only < market_close:
            return MarketSession.REGULAR
        elif market_close <= current_time_only <= after_hours_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED
    
    def get_session_multiplier(
        self,
        session: MarketSession,
        instrument_class: InstrumentClass = InstrumentClass.EQUITY
    ) -> Decimal:
        """Get cost multiplier for a market session."""
        time_config = self.get_time_based_configuration(instrument_class)
        
        multiplier_map = {
            MarketSession.PRE_MARKET: time_config.pre_market_cost_multiplier,
            MarketSession.REGULAR: time_config.regular_hours_multiplier,
            MarketSession.AFTER_HOURS: time_config.after_hours_cost_multiplier,
            MarketSession.CLOSED: Decimal('1.5')  # Higher cost for closed market
        }
        
        return multiplier_map.get(session, Decimal('1.0'))
    
    def list_configured_instruments(self) -> List[Tuple[InstrumentClass, Optional[str]]]:
        """Get list of all configured instruments."""
        instruments = []
        
        # Check configuration files
        for config_file in self.market_params_path.glob("*.json"):
            name_parts = config_file.stem.split('_', 1)
            instrument_class = InstrumentClass(name_parts[0])
            symbol = name_parts[1] if len(name_parts) > 1 else None
            instruments.append((instrument_class, symbol))
        
        # Add cached configurations
        for config_key in self._configs.keys():
            name_parts = config_key.split('_', 1)
            instrument_class = InstrumentClass(name_parts[0])
            symbol = name_parts[1] if len(name_parts) > 1 else None
            
            if (instrument_class, symbol) not in instruments:
                instruments.append((instrument_class, symbol))
        
        return sorted(instruments)
    
    def export_configuration(
        self,
        instrument_class: InstrumentClass,
        export_path: str,
        symbol: Optional[str] = None
    ) -> None:
        """Export market parameter configuration to a file."""
        config = self.get_configuration(instrument_class, symbol)
        
        try:
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            config_key = f"{instrument_class.value}_{symbol}" if symbol else instrument_class.value
            logger.info(f"Exported market parameter configuration for {config_key} to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting market parameter configuration: {e}")
            raise
    
    def import_configuration(
        self,
        import_path: str,
        instrument_class: Optional[InstrumentClass] = None,
        symbol: Optional[str] = None
    ) -> None:
        """Import market parameter configuration from a file."""
        try:
            with open(import_path, 'r') as f:
                config = json.load(f)
            
            # Extract instrument class and symbol from config or use provided values
            final_instrument_class = instrument_class or InstrumentClass(config.get("instrument_class", "equity"))
            final_symbol = symbol or config.get("symbol")
            
            # Remove metadata fields before saving
            metadata_fields = ["instrument_class", "symbol", "last_updated", "version"]
            for field in metadata_fields:
                config.pop(field, None)
            
            self.save_configuration(final_instrument_class, config, final_symbol)
            
            config_key = f"{final_instrument_class.value}_{final_symbol}" if final_symbol else final_instrument_class.value
            logger.info(f"Imported market parameter configuration for {config_key}")
            
        except Exception as e:
            logger.error(f"Error importing market parameter configuration: {e}")
            raise


logger.info("Market parameter configuration system loaded successfully")