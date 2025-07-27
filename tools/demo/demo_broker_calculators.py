#!/usr/bin/env python3
"""
Broker Cost Calculator Demo
===========================

Demonstration script showing how to use the Indian broker-specific
cost calculators for Zerodha and ICICI Securities.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from decimal import Decimal
from datetime import datetime

from src.trading.transaction_costs.models import (
    TransactionRequest,
    BrokerConfiguration, 
    MarketConditions,
    TransactionType,
    InstrumentType,
    OrderType
)
from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory


def create_sample_request():
    """Create a sample transaction request."""
    return TransactionRequest(
        symbol='RELIANCE',
        quantity=100,
        price=Decimal('2500.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY,
        order_type=OrderType.MARKET,
        metadata={'position_type': 'delivery'}  # or 'intraday'
    )


def create_broker_config():
    """Create a sample broker configuration."""
    return BrokerConfiguration(
        broker_name='Demo Broker',
        base_currency='INR'
    )


def create_market_conditions():
    """Create sample market conditions."""
    return MarketConditions(
        bid_price=Decimal('2499.50'),
        ask_price=Decimal('2500.50'),
        volume=1000000,
        timestamp=datetime.now()
    )


def demonstrate_cost_calculation():
    """Demonstrate cost calculation for different brokers."""
    print("=" * 60)
    print("INDIAN BROKER COST CALCULATOR DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    request = create_sample_request()
    broker_config = create_broker_config()
    market_conditions = create_market_conditions()
    
    print(f"\nTransaction Details:")
    print(f"Symbol: {request.symbol}")
    print(f"Quantity: {request.quantity}")
    print(f"Price: ₹{request.price}")
    print(f"Transaction Type: {request.transaction_type.name}")
    print(f"Instrument Type: {request.instrument_type.name}")
    print(f"Notional Value: ₹{request.notional_value:,.2f}")
    print(f"Position Type: {request.metadata.get('position_type', 'delivery')}")
    
    # Test different brokers
    brokers_to_test = ['zerodha', 'icici']
    
    for broker_name in brokers_to_test:
        print(f"\n{'-' * 40}")
        print(f"BROKER: {broker_name.upper()}")
        print(f"{'-' * 40}")
        
        try:
            # Create calculator
            calculator = BrokerFactory.create_calculator(broker_name)
            
            # Calculate costs
            result = calculator.calculate_cost(
                request, broker_config, market_conditions
            )
            
            # Display results
            print(f"Commission: ₹{result.commission:.2f}")
            print(f"Regulatory Fees: ₹{result.regulatory_fees:.2f}")
            print(f"Exchange Fees: ₹{result.exchange_fees:.2f}")
            print(f"Total Explicit Costs: ₹{result.total_explicit_costs:.2f}")
            
            if market_conditions:
                print(f"Bid-Ask Spread Cost: ₹{result.bid_ask_spread_cost:.2f}")
                print(f"Market Impact: ₹{result.market_impact_cost:.2f}")
                print(f"Total Implicit Costs: ₹{result.total_implicit_costs:.2f}")
            
            print(f"TOTAL COST: ₹{result.total_cost:.2f}")
            print(f"Cost per share: ₹{result.total_cost / Decimal(str(request.quantity)):.4f}")
            print(f"Cost as % of notional: {(result.total_cost / request.notional_value * 100):.4f}%")
            
            # Get detailed breakdown
            if hasattr(calculator, 'get_detailed_breakdown'):
                breakdown = calculator.get_detailed_breakdown(
                    request, broker_config, market_conditions
                )
                print(f"Rate Applied: {breakdown['brokerage']['rate_applied']}")
                print(f"Calculation Method: {breakdown['brokerage']['calculation_method']}")
            
        except Exception as e:
            print(f"Error calculating for {broker_name}: {e}")
    
    # Demonstrate intraday vs delivery comparison
    print(f"\n{'-' * 60}")
    print("DELIVERY vs INTRADAY COMPARISON (Zerodha)")
    print(f"{'-' * 60}")
    
    zerodha_calc = BrokerFactory.create_calculator('zerodha')
    
    # Delivery transaction
    delivery_request = create_sample_request()
    delivery_request.metadata = {'position_type': 'delivery'}
    
    delivery_result = zerodha_calc.calculate_cost(
        delivery_request, broker_config, market_conditions
    )
    
    # Intraday transaction
    intraday_request = create_sample_request()
    intraday_request.metadata = {'position_type': 'intraday'}
    
    intraday_result = zerodha_calc.calculate_cost(
        intraday_request, broker_config, market_conditions
    )
    
    print(f"Delivery Commission: ₹{delivery_result.commission:.2f}")
    print(f"Intraday Commission: ₹{intraday_result.commission:.2f}")
    print(f"Delivery Total Cost: ₹{delivery_result.total_cost:.2f}")
    print(f"Intraday Total Cost: ₹{intraday_result.total_cost:.2f}")
    
    savings = delivery_result.total_cost - intraday_result.total_cost
    print(f"Savings with Intraday: ₹{savings:.2f}")


def demonstrate_different_instruments():
    """Demonstrate cost calculation for different instruments."""
    print(f"\n{'-' * 60}")
    print("DIFFERENT INSTRUMENTS COMPARISON")
    print(f"{'-' * 60}")
    
    broker_config = create_broker_config()
    zerodha_calc = BrokerFactory.create_calculator('zerodha')
    
    instruments = [
        (InstrumentType.EQUITY, 100, Decimal('2500.00')),
        (InstrumentType.OPTION, 1, Decimal('50.00')),
        (InstrumentType.FUTURE, 1, Decimal('2500.00')),
    ]
    
    for instrument_type, quantity, price in instruments:
        request = TransactionRequest(
            symbol='RELIANCE',
            quantity=quantity,
            price=price,
            transaction_type=TransactionType.BUY,
            instrument_type=instrument_type,
            metadata={'position_type': 'intraday'}
        )
        
        result = zerodha_calc.calculate_cost(request, broker_config)
        
        print(f"{instrument_type.name:8}: Commission ₹{result.commission:.2f}, "
              f"Total ₹{result.total_cost:.2f}")


def demonstrate_factory_features():
    """Demonstrate broker factory features."""
    print(f"\n{'-' * 60}")
    print("BROKER FACTORY FEATURES")
    print(f"{'-' * 60}")
    
    # List supported brokers
    supported_brokers = BrokerFactory.get_supported_brokers()
    print(f"Supported Brokers: {supported_brokers}")
    
    # Get broker information
    for broker in ['zerodha', 'icici']:
        info = BrokerFactory.get_broker_info(broker)
        if info:
            print(f"\n{broker.upper()} Info:")
            print(f"  Calculator Class: {info['calculator_class']}")
            print(f"  Supported Instruments: {info['supported_instruments']}")
            print(f"  Supported Modes: {info['supported_modes']}")


if __name__ == '__main__':
    try:
        demonstrate_cost_calculation()
        demonstrate_different_instruments() 
        demonstrate_factory_features()
        
        print(f"\n{'=' * 60}")
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()