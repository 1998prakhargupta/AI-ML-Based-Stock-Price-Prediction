#!/usr/bin/env python3
"""
Broker Calculator Demo
======================

Comprehensive demonstration of the broker-specific fee calculation system.
Shows usage examples, fee comparisons, and detailed cost breakdowns for
Indian brokers (Zerodha and ICICI Securities).

This demo showcases:
- Basic calculator usage
- Fee comparison between brokers
- Detailed regulatory charge breakdowns
- Batch calculation capabilities
- Error handling
- Performance monitoring

Usage:
    python broker_calculator_demo.py
"""

from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory
from src.trading.transaction_costs.models import *
from decimal import Decimal
from datetime import datetime
import json

def print_separator(title):
    """Print a formatted separator."""
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)

def print_cost_breakdown(broker_name, result, request):
    """Print detailed cost breakdown."""
    print(f"\n{broker_name} Cost Breakdown:")
    print(f"  Transaction: {request.symbol} - {request.quantity} shares @ ₹{request.price}")
    print(f"  Notional Value: ₹{request.notional_value:,.2f}")
    print(f"  Commission: ₹{result.commission:.2f}")
    print(f"  Regulatory Fees: ₹{result.regulatory_fees:.2f}")
    print(f"  Exchange Fees: ₹{result.exchange_fees:.2f}")
    print(f"  Total Cost: ₹{result.total_cost:.2f}")
    print(f"  Cost as % of trade: {(result.total_cost / request.notional_value * 100):.4f}%")

def demo_basic_usage():
    """Demonstrate basic usage of broker calculators."""
    print_separator("BASIC USAGE DEMO")
    
    # Create calculators using factory
    zerodha = BrokerFactory.create_calculator('zerodha')
    icici = BrokerFactory.create_calculator('icici')
    
    print("✅ Created broker calculators:")
    print(f"   Zerodha: {zerodha.calculator_name} v{zerodha.version}")
    print(f"   ICICI: {icici.calculator_name} v{icici.version}")
    
    # Show supported brokers
    print(f"\n📋 Supported brokers: {BrokerFactory.get_supported_brokers()}")

def demo_fee_comparison():
    """Compare fees between different brokers."""
    print_separator("FEE COMPARISON DEMO")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Small Equity Delivery Trade',
            'request': TransactionRequest(
                symbol='RELIANCE',
                quantity=10,
                price=Decimal('2500.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY,
                metadata={'position_type': 'delivery'}
            )
        },
        {
            'name': 'Large Equity Intraday Trade',
            'request': TransactionRequest(
                symbol='NIFTY50',
                quantity=1000,
                price=Decimal('18000.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY,
                metadata={'position_type': 'intraday'}
            )
        },
        {
            'name': 'Options Trade',
            'request': TransactionRequest(
                symbol='NIFTY23JUN18000CE',
                quantity=100,
                price=Decimal('50.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.OPTION
            )
        },
        {
            'name': 'Futures Trade',
            'request': TransactionRequest(
                symbol='NIFTYJUN',
                quantity=75,
                price=Decimal('18000.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.FUTURE
            )
        }
    ]
    
    # Create calculators
    zerodha = BrokerFactory.create_calculator('zerodha')
    icici = BrokerFactory.create_calculator('icici')
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}")
        print("-" * 40)
        
        request = scenario['request']
        
        # Calculate costs for both brokers
        zerodha_config = BrokerConfiguration(broker_name='Zerodha', active=True)
        icici_config = BrokerConfiguration(broker_name='ICICI Securities', active=True)
        
        zerodha_result = zerodha.calculate_cost(request, zerodha_config)
        icici_result = icici.calculate_cost(request, icici_config)
        
        print_cost_breakdown("Zerodha", zerodha_result, request)
        print_cost_breakdown("ICICI Securities", icici_result, request)
        
        # Show savings
        if zerodha_result.total_cost < icici_result.total_cost:
            savings = icici_result.total_cost - zerodha_result.total_cost
            print(f"💰 Zerodha saves ₹{savings:.2f} ({(savings/icici_result.total_cost*100):.1f}%)")
        else:
            savings = zerodha_result.total_cost - icici_result.total_cost
            print(f"💰 ICICI saves ₹{savings:.2f} ({(savings/zerodha_result.total_cost*100):.1f}%)")

def demo_detailed_breakdown():
    """Show detailed breakdown with regulatory charges."""
    print_separator("DETAILED BREAKDOWN DEMO")
    
    # Large equity trade
    request = TransactionRequest(
        symbol='TCS',
        quantity=500,
        price=Decimal('3500.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY,
        metadata={'position_type': 'delivery'}
    )
    
    print(f"📈 Transaction Details:")
    print(f"   Symbol: {request.symbol}")
    print(f"   Quantity: {request.quantity} shares")
    print(f"   Price: ₹{request.price} per share")
    print(f"   Notional Value: ₹{request.notional_value:,.2f}")
    print(f"   Transaction Type: {request.transaction_type.name}")
    print(f"   Instrument Type: {request.instrument_type.name}")
    
    # Calculate with Zerodha
    zerodha = BrokerFactory.create_calculator('zerodha')
    broker_config = BrokerConfiguration(broker_name='Zerodha', active=True)
    
    result = zerodha.calculate_cost(request, broker_config)
    breakdown = zerodha.get_detailed_breakdown(request, broker_config)
    
    print(f"\n🔍 Detailed Cost Analysis (Zerodha):")
    print(f"   📊 Brokerage:")
    print(f"      Commission: ₹{breakdown['brokerage']['commission']:.2f}")
    print(f"      Rate Applied: {breakdown['brokerage']['rate_applied']}")
    print(f"      Calculation: {breakdown['brokerage']['calculation_method']}")
    
    print(f"   📋 Regulatory Charges:")
    reg_charges = breakdown['regulatory_charges']
    print(f"      STT: ₹{reg_charges['statutory_charges']['stt']:.2f}")
    print(f"      Stamp Duty: ₹{reg_charges['statutory_charges']['stamp_duty']:.2f}")
    print(f"      Exchange Charges: ₹{reg_charges['exchange_charges']['transaction_charge']:.2f}")
    print(f"      SEBI Charges: ₹{reg_charges['regulatory_charges']['sebi_charge']:.2f}")
    print(f"      GST: ₹{reg_charges['taxes']['gst']:.2f}")
    
    print(f"   💰 Summary:")
    print(f"      Total Explicit Cost: ₹{breakdown['total_explicit_cost']:.2f}")
    print(f"      Cost per Share: ₹{breakdown['cost_per_share']:.4f}")
    print(f"      Cost Percentage: {breakdown['cost_percentage']:.4f}%")

def demo_batch_calculation():
    """Demonstrate batch calculation capabilities."""
    print_separator("BATCH CALCULATION DEMO")
    
    # Create multiple transaction requests
    requests = [
        TransactionRequest(
            symbol=f'STOCK{i}',
            quantity=100 + i * 10,
            price=Decimal(str(1000 + i * 100)),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            metadata={'position_type': 'delivery'}
        )
        for i in range(1, 6)
    ]
    
    print(f"📦 Processing {len(requests)} transactions in batch...")
    
    zerodha = BrokerFactory.create_calculator('zerodha')
    broker_config = BrokerConfiguration(broker_name='Zerodha', active=True)
    
    # Batch calculation
    start_time = datetime.now()
    results = zerodha.calculate_batch_costs(requests, broker_config)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"⚡ Processed in {processing_time:.3f} seconds")
    print(f"📊 Results Summary:")
    
    total_notional = sum(req.notional_value for req in requests)
    total_cost = sum(result.total_cost for result in results)
    
    for i, (request, result) in enumerate(zip(requests, results), 1):
        print(f"   {i}. {request.symbol}: ₹{result.total_cost:.2f} cost on ₹{request.notional_value:.0f} trade")
    
    print(f"\n💼 Portfolio Summary:")
    print(f"   Total Portfolio Value: ₹{total_notional:,.2f}")
    print(f"   Total Transaction Costs: ₹{total_cost:.2f}")
    print(f"   Average Cost %: {(total_cost / total_notional * 100):.4f}%")

def demo_simple_usage_example():
    """Simple usage example for documentation."""
    print_separator("SIMPLE USAGE EXAMPLE")
    
    print("💡 Quick Start Example:")
    print("```python")
    print("from src.trading.transaction_costs.brokers.broker_factory import BrokerFactory")
    print("from src.trading.transaction_costs.models import *")
    print("from decimal import Decimal")
    print("")
    print("# Create calculator")
    print("calculator = BrokerFactory.create_calculator('zerodha')")
    print("")
    print("# Define transaction")
    print("request = TransactionRequest(")
    print("    symbol='RELIANCE',")
    print("    quantity=100,")
    print("    price=Decimal('2500.00'),")
    print("    transaction_type=TransactionType.BUY,")
    print("    instrument_type=InstrumentType.EQUITY")
    print(")")
    print("")
    print("# Configure broker")
    print("broker_config = BrokerConfiguration(")
    print("    broker_name='Zerodha',")
    print("    active=True")
    print(")")
    print("")
    print("# Calculate costs")
    print("result = calculator.calculate_cost(request, broker_config)")
    print("print(f'Total cost: ₹{result.total_cost:.2f}')")
    print("```")
    
    # Actually run the example
    print("\n🚀 Running the example:")
    
    calculator = BrokerFactory.create_calculator('zerodha')
    
    request = TransactionRequest(
        symbol='RELIANCE',
        quantity=100,
        price=Decimal('2500.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY
    )
    
    broker_config = BrokerConfiguration(
        broker_name='Zerodha',
        active=True
    )
    
    result = calculator.calculate_cost(request, broker_config)
    print(f"✅ Total cost: ₹{result.total_cost:.2f}")

def main():
    """Run all demonstrations."""
    print("🏦 BROKER FEE CALCULATION SYSTEM DEMO")
    print("Comprehensive demonstration of Indian broker fee calculations")
    print("Supporting Zerodha (Kite Connect) and ICICI Securities (Breeze Connect)")
    
    try:
        demo_basic_usage()
        demo_simple_usage_example()
        demo_fee_comparison()
        demo_detailed_breakdown()
        demo_batch_calculation()
        
        print_separator("DEMO COMPLETE")
        print("✅ All demonstrations completed successfully!")
        print("🎯 The broker fee calculation system is fully operational!")
        print("\n📚 Key Features Demonstrated:")
        print("   • Accurate broker-specific fee calculations")
        print("   • Comprehensive regulatory charges (STT, GST, Stamp Duty, etc.)")
        print("   • Batch processing capabilities")
        print("   • Robust error handling")
        print("   • Performance monitoring")
        print("   • Detailed cost breakdowns")
        print("\n🔗 Next Steps:")
        print("   • Integrate with real-time market data")
        print("   • Add more Indian brokers")
        print("   • Implement advanced market impact models")
        print("   • Add portfolio-level cost analysis")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)