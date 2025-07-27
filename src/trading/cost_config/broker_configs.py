"""
Broker Configuration System
============================

Comprehensive broker configuration management with fee structures,
API connection parameters, rate limiting, and integration settings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Types of brokers supported."""
    TRADITIONAL = "traditional"
    DISCOUNT = "discount"
    ROBO_ADVISOR = "robo_advisor"
    CRYPTOCURRENCY = "crypto"
    FOREX = "forex"


class AccountTier(Enum):
    """Account tier levels."""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"


@dataclass
class APIConfiguration:
    """API connection configuration for a broker."""
    base_url: str
    api_version: str = "v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    authentication_type: str = "api_key"  # api_key, oauth2, basic_auth
    supports_websocket: bool = False
    websocket_url: Optional[str] = None
    api_key_header: str = "X-API-Key"
    user_agent: str = "TransactionCostCalculator/1.0"


@dataclass
class FeeStructure:
    """Complete fee structure for a broker."""
    # Commission rates
    equity_commission_per_share: Decimal = Decimal('0.0')
    equity_commission_percentage: Decimal = Decimal('0.0')
    min_equity_commission: Decimal = Decimal('0.0')
    max_equity_commission: Optional[Decimal] = None
    
    # Options fees
    options_commission_per_contract: Decimal = Decimal('0.65')
    options_base_commission: Decimal = Decimal('0.0')
    options_percentage: Decimal = Decimal('0.0')
    
    # Futures fees
    futures_commission_per_contract: Decimal = Decimal('2.25')
    futures_base_commission: Decimal = Decimal('0.0')
    
    # Regulatory fees
    sec_fee_rate: Decimal = Decimal('0.0000051')  # SEC fee per dollar
    finra_taf_sell_rate: Decimal = Decimal('0.000166')  # FINRA TAF on sells
    finra_taf_max: Decimal = Decimal('8.50')  # FINRA TAF maximum
    
    # Platform and data fees
    platform_fee_monthly: Decimal = Decimal('0.0')
    data_fee_monthly: Decimal = Decimal('0.0')
    inactivity_fee_monthly: Decimal = Decimal('0.0')
    
    # Currency and international
    currency_conversion_spread: Decimal = Decimal('0.0025')  # 25 bps
    international_trading_fee: Decimal = Decimal('0.0')
    
    # Time-based multipliers
    pre_market_multiplier: Decimal = Decimal('1.0')
    after_hours_multiplier: Decimal = Decimal('1.0')
    
    # Volume discount tiers (volume -> discount multiplier)
    volume_discount_tiers: Dict[str, Decimal] = None
    
    def __post_init__(self):
        if self.volume_discount_tiers is None:
            self.volume_discount_tiers = {}


@dataclass
class DataProviderConfig:
    """Configuration for market data providers."""
    provider_name: str
    primary: bool = False
    api_config: Optional[APIConfiguration] = None
    data_types: List[str] = None  # ['quotes', 'trades', 'options', 'fundamentals']
    update_frequency_seconds: int = 60
    cache_duration_seconds: int = 300
    quality_score: float = 1.0  # 0.0 to 1.0
    cost_per_request: Decimal = Decimal('0.0')
    monthly_subscription: Decimal = Decimal('0.0')
    
    def __post_init__(self):
        if self.data_types is None:
            self.data_types = ['quotes']


class BrokerConfigurationManager:
    """
    Advanced broker configuration management system.
    
    Manages broker-specific settings, fee structures, API configurations,
    and provides templates for easy broker setup.
    """
    
    def __init__(self, config_base_path: Optional[str] = None):
        """
        Initialize broker configuration manager.
        
        Args:
            config_base_path: Base path for broker configuration files
        """
        self.config_base_path = Path(config_base_path) if config_base_path else Path.cwd() / "configs" / "cost_config"
        self.broker_templates_path = self.config_base_path / "broker_templates"
        
        # Ensure directories exist
        self.config_base_path.mkdir(parents=True, exist_ok=True)
        self.broker_templates_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded configurations
        self._broker_configs: Dict[str, Dict[str, Any]] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"BrokerConfigurationManager initialized with base path: {self.config_base_path}")
    
    def create_broker_configuration(
        self,
        broker_name: str,
        broker_type: BrokerType,
        account_tier: AccountTier,
        fee_structure: FeeStructure,
        api_config: Optional[APIConfiguration] = None,
        data_providers: Optional[List[DataProviderConfig]] = None,
        additional_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a complete broker configuration.
        
        Args:
            broker_name: Name of the broker
            broker_type: Type of broker
            account_tier: Account tier level
            fee_structure: Fee structure configuration
            api_config: API configuration
            data_providers: List of data provider configurations
            additional_settings: Additional broker-specific settings
            
        Returns:
            Complete broker configuration dictionary
        """
        # Base configuration
        config = {
            "broker_name": broker_name,
            "broker_type": broker_type.value,
            "account_tier": account_tier.value,
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0.0",
            
            # Fee structure
            "fee_structure": self._serialize_fee_structure(fee_structure),
            
            # API configuration
            "api_configuration": asdict(api_config) if api_config else None,
            
            # Data providers
            "data_providers": [asdict(provider) for provider in (data_providers or [])],
            
            # Additional settings
            "additional_settings": additional_settings or {}
        }
        
        # Cache the configuration
        self._broker_configs[broker_name] = config
        
        logger.info(f"Created broker configuration for {broker_name}")
        return config
    
    def _serialize_fee_structure(self, fee_structure: FeeStructure) -> Dict[str, Any]:
        """Serialize fee structure to JSON-compatible format."""
        fee_dict = asdict(fee_structure)
        
        # Convert Decimal values to strings for JSON serialization
        for key, value in fee_dict.items():
            if isinstance(value, Decimal):
                fee_dict[key] = str(value)
            elif isinstance(value, dict):
                # Handle volume_discount_tiers
                fee_dict[key] = {k: str(v) for k, v in value.items()}
        
        return fee_dict
    
    def _deserialize_fee_structure(self, fee_dict: Dict[str, Any]) -> FeeStructure:
        """Deserialize fee structure from JSON format."""
        # Convert string values back to Decimal
        converted_dict = {}
        
        for key, value in fee_dict.items():
            if key == "volume_discount_tiers" and isinstance(value, dict):
                converted_dict[key] = {k: Decimal(str(v)) for k, v in value.items()}
            elif isinstance(value, str) and (
                key.endswith(('_rate', '_commission', '_fee', '_multiplier', '_spread', '_percentage')) or
                key.startswith(('equity_', 'options_', 'futures_', 'min_', 'max_', 'sec_', 'finra_', 'platform_', 'data_', 'inactivity_', 'currency_', 'international_', 'pre_', 'after_'))
            ):
                try:
                    converted_dict[key] = Decimal(value)
                except:
                    converted_dict[key] = value
            elif value is None:
                converted_dict[key] = None
            else:
                converted_dict[key] = value
        
        return FeeStructure(**converted_dict)
    
    def save_broker_configuration(self, broker_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save broker configuration to file.
        
        Args:
            broker_name: Name of the broker
            config: Configuration to save (uses cached if None)
        """
        if config is None:
            config = self._broker_configs.get(broker_name)
            if not config:
                raise ValueError(f"No configuration found for broker: {broker_name}")
        
        config_file = self.config_base_path / f"{broker_name.lower().replace(' ', '_')}_config.json"
        
        try:
            # Update last_updated timestamp
            config["last_updated"] = datetime.now().isoformat()
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Saved broker configuration for {broker_name} to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving broker configuration for {broker_name}: {e}")
            raise
    
    def load_broker_configuration(self, broker_name: str) -> Optional[Dict[str, Any]]:
        """
        Load broker configuration from file.
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            Broker configuration dictionary or None if not found
        """
        # Check cache first
        if broker_name in self._broker_configs:
            return self._broker_configs[broker_name]
        
        config_file = self.config_base_path / f"{broker_name.lower().replace(' ', '_')}_config.json"
        
        if not config_file.exists():
            logger.warning(f"Broker configuration file not found: {config_file}")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Cache the configuration
            self._broker_configs[broker_name] = config
            
            logger.info(f"Loaded broker configuration for {broker_name}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading broker configuration for {broker_name}: {e}")
            return None
    
    def get_fee_structure(self, broker_name: str) -> Optional[FeeStructure]:
        """
        Get fee structure for a broker.
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            FeeStructure instance or None if not found
        """
        config = self.load_broker_configuration(broker_name)
        if not config or "fee_structure" not in config:
            return None
        
        try:
            return self._deserialize_fee_structure(config["fee_structure"])
        except Exception as e:
            logger.error(f"Error deserializing fee structure for {broker_name}: {e}")
            return None
    
    def update_fee_structure(self, broker_name: str, fee_structure: FeeStructure) -> None:
        """
        Update fee structure for a broker.
        
        Args:
            broker_name: Name of the broker
            fee_structure: New fee structure
        """
        config = self.load_broker_configuration(broker_name)
        if not config:
            raise ValueError(f"Broker configuration not found: {broker_name}")
        
        config["fee_structure"] = self._serialize_fee_structure(fee_structure)
        config["last_updated"] = datetime.now().isoformat()
        
        self._broker_configs[broker_name] = config
        self.save_broker_configuration(broker_name, config)
        
        logger.info(f"Updated fee structure for {broker_name}")
    
    def get_api_configuration(self, broker_name: str) -> Optional[APIConfiguration]:
        """
        Get API configuration for a broker.
        
        Args:
            broker_name: Name of the broker
            
        Returns:
            APIConfiguration instance or None if not found
        """
        config = self.load_broker_configuration(broker_name)
        if not config or not config.get("api_configuration"):
            return None
        
        try:
            return APIConfiguration(**config["api_configuration"])
        except Exception as e:
            logger.error(f"Error deserializing API configuration for {broker_name}: {e}")
            return None
    
    def list_configured_brokers(self) -> List[str]:
        """Get list of all configured brokers."""
        brokers = []
        
        # Check files in config directory
        for config_file in self.config_base_path.glob("*_config.json"):
            broker_name = config_file.stem.replace("_config", "").replace("_", " ").title()
            brokers.append(broker_name)
        
        # Add cached brokers
        for broker_name in self._broker_configs.keys():
            if broker_name not in brokers:
                brokers.append(broker_name)
        
        return sorted(brokers)
    
    def create_broker_template(self, template_name: str, config: Dict[str, Any]) -> None:
        """
        Create a broker configuration template.
        
        Args:
            template_name: Name of the template
            config: Template configuration
        """
        template_file = self.broker_templates_path / f"{template_name.lower()}_template.json"
        
        # Add template metadata
        template_config = config.copy()
        template_config.update({
            "template_name": template_name,
            "template_version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "description": f"Template configuration for {template_name} brokers"
        })
        
        try:
            with open(template_file, 'w') as f:
                json.dump(template_config, f, indent=2, default=str)
            
            self._templates[template_name] = template_config
            logger.info(f"Created broker template: {template_name}")
            
        except Exception as e:
            logger.error(f"Error creating broker template {template_name}: {e}")
            raise
    
    def load_broker_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a broker configuration template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template configuration or None if not found
        """
        # Check cache first
        if template_name in self._templates:
            return self._templates[template_name]
        
        template_file = self.broker_templates_path / f"{template_name.lower()}_template.json"
        
        if not template_file.exists():
            logger.warning(f"Broker template not found: {template_file}")
            return None
        
        try:
            with open(template_file, 'r') as f:
                template = json.load(f)
            
            self._templates[template_name] = template
            logger.info(f"Loaded broker template: {template_name}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading broker template {template_name}: {e}")
            return None
    
    def create_broker_from_template(
        self,
        broker_name: str,
        template_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a broker configuration from a template.
        
        Args:
            broker_name: Name for the new broker
            template_name: Name of the template to use
            overrides: Configuration overrides
            
        Returns:
            New broker configuration
        """
        template = self.load_broker_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Create new configuration from template
        config = template.copy()
        
        # Remove template-specific fields
        template_fields = ["template_name", "template_version", "description"]
        for field in template_fields:
            config.pop(field, None)
        
        # Update with broker-specific information
        config.update({
            "broker_name": broker_name,
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "template_source": template_name
        })
        
        # Apply overrides
        if overrides:
            self._deep_merge(config, overrides)
        
        # Cache and save
        self._broker_configs[broker_name] = config
        self.save_broker_configuration(broker_name, config)
        
        logger.info(f"Created broker {broker_name} from template {template_name}")
        return config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def list_available_templates(self) -> List[str]:
        """Get list of available broker templates."""
        templates = []
        
        # Check template files
        for template_file in self.broker_templates_path.glob("*_template.json"):
            template_name = template_file.stem.replace("_template", "").replace("_", " ").title()
            templates.append(template_name)
        
        # Add cached templates
        for template_name in self._templates.keys():
            if template_name not in templates:
                templates.append(template_name)
        
        return sorted(templates)
    
    def compare_broker_fees(self, broker_names: List[str], trade_amount: Decimal = Decimal('10000')) -> Dict[str, Dict[str, Any]]:
        """
        Compare fees across multiple brokers for a sample trade.
        
        Args:
            broker_names: List of broker names to compare
            trade_amount: Sample trade amount for comparison
            
        Returns:
            Dictionary with fee comparison results
        """
        comparison = {}
        
        for broker_name in broker_names:
            fee_structure = self.get_fee_structure(broker_name)
            if not fee_structure:
                comparison[broker_name] = {"error": "Fee structure not found"}
                continue
            
            # Calculate sample fees
            shares = 100  # Sample 100 shares
            price_per_share = trade_amount / shares
            
            # Equity commission calculation
            commission_per_share = fee_structure.equity_commission_per_share * shares
            commission_percentage = trade_amount * fee_structure.equity_commission_percentage
            commission = max(
                commission_per_share + commission_percentage,
                fee_structure.min_equity_commission
            )
            
            if fee_structure.max_equity_commission:
                commission = min(commission, fee_structure.max_equity_commission)
            
            # Regulatory fees
            sec_fee = trade_amount * fee_structure.sec_fee_rate
            finra_taf = min(
                shares * fee_structure.finra_taf_sell_rate,
                fee_structure.finra_taf_max
            )
            
            total_cost = commission + sec_fee + finra_taf
            cost_bps = (total_cost / trade_amount) * 10000  # Convert to basis points
            
            comparison[broker_name] = {
                "commission": float(commission),
                "sec_fee": float(sec_fee),
                "finra_taf": float(finra_taf),
                "total_cost": float(total_cost),
                "cost_basis_points": float(cost_bps),
                "cost_percentage": float(cost_bps / 100)
            }
        
        return comparison
    
    def export_broker_configuration(self, broker_name: str, export_path: str) -> None:
        """
        Export broker configuration to a file.
        
        Args:
            broker_name: Name of the broker
            export_path: Path to save the exported configuration
        """
        config = self.load_broker_configuration(broker_name)
        if not config:
            raise ValueError(f"Broker configuration not found: {broker_name}")
        
        try:
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Exported broker configuration for {broker_name} to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting broker configuration: {e}")
            raise
    
    def import_broker_configuration(self, import_path: str, broker_name: Optional[str] = None) -> str:
        """
        Import broker configuration from a file.
        
        Args:
            import_path: Path to the configuration file
            broker_name: Optional name override for the broker
            
        Returns:
            Name of the imported broker configuration
        """
        try:
            with open(import_path, 'r') as f:
                config = json.load(f)
            
            # Use provided name or extract from config
            final_broker_name = broker_name or config.get("broker_name", "Imported Broker")
            
            config["broker_name"] = final_broker_name
            config["last_updated"] = datetime.now().isoformat()
            config["import_source"] = import_path
            
            self._broker_configs[final_broker_name] = config
            self.save_broker_configuration(final_broker_name, config)
            
            logger.info(f"Imported broker configuration: {final_broker_name}")
            return final_broker_name
            
        except Exception as e:
            logger.error(f"Error importing broker configuration: {e}")
            raise


def create_default_broker_templates():
    """Create default broker templates for popular brokers."""
    manager = BrokerConfigurationManager()
    
    # Interactive Brokers Template
    ib_fee_structure = FeeStructure(
        equity_commission_per_share=Decimal('0.005'),
        min_equity_commission=Decimal('1.00'),
        max_equity_commission=Decimal('1.0'),  # IBKR Pro has 1% max
        options_commission_per_contract=Decimal('0.65'),
        futures_commission_per_contract=Decimal('0.85'),
        platform_fee_monthly=Decimal('10.00'),  # Market data fees
        volume_discount_tiers={
            "300000": Decimal('0.8'),  # 20% discount for 300K+ shares
            "3000000": Decimal('0.6')  # 40% discount for 3M+ shares
        }
    )
    
    ib_api_config = APIConfiguration(
        base_url="https://api.interactivebrokers.com",
        rate_limit_requests_per_minute=100,
        supports_websocket=True,
        websocket_url="wss://api.interactivebrokers.com/ws"
    )
    
    ib_config = {
        "broker_type": BrokerType.TRADITIONAL.value,
        "account_tier": AccountTier.PROFESSIONAL.value,
        "fee_structure": manager._serialize_fee_structure(ib_fee_structure),
        "api_configuration": asdict(ib_api_config),
        "data_providers": []
    }
    
    manager.create_broker_template("Interactive Brokers", ib_config)
    
    # Charles Schwab Template (Commission-free)
    schwab_fee_structure = FeeStructure(
        equity_commission_per_share=Decimal('0.0'),
        options_commission_per_contract=Decimal('0.65'),
        platform_fee_monthly=Decimal('0.0')
    )
    
    schwab_config = {
        "broker_type": BrokerType.DISCOUNT.value,
        "account_tier": AccountTier.RETAIL.value,
        "fee_structure": manager._serialize_fee_structure(schwab_fee_structure),
        "api_configuration": None,
        "data_providers": []
    }
    
    manager.create_broker_template("Charles Schwab", schwab_config)
    
    # Breeze (India) Template
    breeze_fee_structure = FeeStructure(
        equity_commission_percentage=Decimal('0.0005'),  # 0.05% or Rs 20 whichever is lower
        min_equity_commission=Decimal('20.0'),  # Rs 20 minimum
        options_commission_per_contract=Decimal('20.0'),  # Rs 20 per lot
        futures_commission_per_contract=Decimal('20.0'),  # Rs 20 per lot
    )
    
    breeze_api_config = APIConfiguration(
        base_url="https://api.icicidirect.com",
        rate_limit_requests_per_minute=100,
        authentication_type="session_token"
    )
    
    breeze_config = {
        "broker_type": BrokerType.DISCOUNT.value,
        "account_tier": AccountTier.RETAIL.value,
        "fee_structure": manager._serialize_fee_structure(breeze_fee_structure),
        "api_configuration": asdict(breeze_api_config),
        "data_providers": []
    }
    
    manager.create_broker_template("Breeze", breeze_config)
    
    logger.info("Created default broker templates")


if __name__ == "__main__":
    # Create default templates when module is run directly
    create_default_broker_templates()

logger.info("Broker configuration system loaded successfully")