#!/usr/bin/env python3
"""
Basic Transaction Cost Calculations
==================================

This example demonstrates basic usage of the transaction cost system
for common trading scenarios.

Run this example:
    python docs/transaction_costs/examples/basic_calculations.py
"""

import sys
import os
from decimal import Decimal
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.trading.transaction_costs.models import (
    TransactionRequest, 
    TransactionType, 
    InstrumentType,
    OrderType,
    MarketTiming
)
from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
from src.trading.transaction_costs.brokers.breeze_calculator import BreezeCalculator


def basic_equity_calculation():
    """Demonstrate basic equity transaction cost calculation."""
    print("=== Basic Equity Transaction ===")
    
    # Create a simple buy order for 100 shares of Reliance
    request = TransactionRequest(
        symbol='RELIANCE',
        quantity=100,
        price=Decimal('2500.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY,
        order_type=OrderType.MARKET,
        timestamp=datetime.now()
    )
    
    # Calculate using Zerodha
    zerodha_calc = ZerodhaCalculator()
    zerodha_result = zerodha_calc.calculate_cost(request)
    
    print(f"Stock: {request.symbol}")
    print(f"Quantity: {request.quantity} shares")
    print(f"Price: ₹{request.price} per share")
    print(f"Total Value: ₹{request.quantity * request.price:,.2f}")
    print(f"\nZerodha Costs:")
    print(f"  Brokerage: ₹{zerodha_result.brokerage:.2f}")
    print(f"  STT: ₹{zerodha_result.stt:.2f}")
    print(f"  Exchange Charges: ₹{zerodha_result.exchange_charges:.2f}")
    print(f"  GST: ₹{zerodha_result.gst:.2f}")
    print(f"  SEBI Charges: ₹{zerodha_result.sebi_charges:.2f}")
    print(f"  Stamp Duty: ₹{zerodha_result.stamp_duty:.2f}")
    print(f"  Total Cost: ₹{zerodha_result.total_cost:.2f}")
    print(f"  Net Amount: ₹{zerodha_result.net_amount:.2f}")
    print(f"  Effective Rate: {(zerodha_result.total_cost / (request.quantity * request.price) * 100):.3f}%")
    print()


def compare_delivery_vs_intraday():
    """Compare costs for delivery vs intraday trading."""
    print("=== Delivery vs Intraday Comparison ===")
    
    base_request = TransactionRequest(
        symbol='TCS',
        quantity=50,
        price=Decimal('3200.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY
    )
    
    # Delivery order (CNC - Cash and Carry)
    delivery_request = base_request
    delivery_request.metadata = {'product_type': 'CNC'}
    
    # Intraday order (MIS - Margin Intraday Square-off)
    intraday_request = base_request
    intraday_request.metadata = {'product_type': 'MIS'}
    
    calculator = ZerodhaCalculator()
    
    delivery_result = calculator.calculate_cost(delivery_request)
    intraday_result = calculator.calculate_cost(intraday_request)
    
    print(f"Trade: {base_request.quantity} shares of {base_request.symbol} at ₹{base_request.price}")
    print(f"Total Value: ₹{base_request.quantity * base_request.price:,.2f}")
    print()
    
    print("Delivery Trading (CNC):")
    print(f"  Brokerage: ₹{delivery_result.brokerage:.2f}")
    print(f"  Total Cost: ₹{delivery_result.total_cost:.2f}")
    print(f"  Effective Rate: {(delivery_result.total_cost / (base_request.quantity * base_request.price) * 100):.3f}%")
    print()
    
    print("Intraday Trading (MIS):")
    print(f"  Brokerage: ₹{intraday_result.brokerage:.2f}")
    print(f"  Total Cost: ₹{intraday_result.total_cost:.2f}")
    print(f"  Effective Rate: {(intraday_result.total_cost / (base_request.quantity * base_request.price) * 100):.3f}%")
    print()
    
    savings = intraday_result.total_cost - delivery_result.total_cost
    if savings > 0:
        print(f"Delivery saves: ₹{savings:.2f}")
    else:
        print(f"Intraday saves: ₹{abs(savings):.2f}")
    print()


def broker_comparison():
    """Compare costs across different brokers."""
    print("=== Broker Comparison ===")
    
    request = TransactionRequest(
        symbol='HDFC',
        quantity=200,
        price=Decimal('1600.00'),
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.EQUITY,
        metadata={'product_type': 'CNC'}  # Delivery
    )
    
    brokers = {
        'Zerodha': ZerodhaCalculator(),
        'ICICI Securities (Breeze)': BreezeCalculator()
    }
    
    print(f"Trade: {request.quantity} shares of {request.symbol} at ₹{request.price}")
    print(f"Total Value: ₹{request.quantity * request.price:,.2f}")
    print(f"Order Type: Delivery")
    print()
    
    results = {}
    for broker_name, calculator in brokers.items():
        try:
            result = calculator.calculate_cost(request)
            results[broker_name] = result
            
            print(f"{broker_name}:")
            print(f"  Brokerage: ₹{result.brokerage:.2f}")
            print(f"  Total Cost: ₹{result.total_cost:.2f}")
            print(f"  Effective Rate: {(result.total_cost / (request.quantity * request.price) * 100):.3f}%")
            print()
        except Exception as e:
            print(f"{broker_name}: Error - {e}")
            print()
    
    # Find the cheapest broker
    if results:
        cheapest_broker = min(results.items(), key=lambda x: x[1].total_cost)
        print(f"Cheapest Option: {cheapest_broker[0]} (₹{cheapest_broker[1].total_cost:.2f})")
        print()


def options_trading_example():
    """Demonstrate options trading cost calculation."""
    print("=== Options Trading Example ===")
    
    # Buy NIFTY call option
    option_request = TransactionRequest(
        symbol='NIFTY24JAN17500CE',
        quantity=1,  # 1 lot
        price=Decimal('150.00'),  # Premium per share
        transaction_type=TransactionType.BUY,
        instrument_type=InstrumentType.OPTION,
        metadata={
            'lot_size': 50,  # NIFTY lot size
            'strike_price': 17500,
            'expiry': '2024-01-25',
            'option_type': 'CE'  # Call European
        }
    )
    
    calculator = ZerodhaCalculator()
    result = calculator.calculate_cost(option_request)
    
    lot_size = option_request.metadata.get('lot_size', 1)
    total_premium = option_request.quantity * lot_size * option_request.price
    
    print(f"Option: {option_request.symbol}")
    print(f"Quantity: {option_request.quantity} lot(s)")
    print(f"Lot Size: {lot_size} shares per lot")
    print(f"Premium: ₹{option_request.price} per share")
    print(f"Total Premium: ₹{total_premium:,.2f}")
    print()
    
    print("Cost Breakdown:")
    print(f"  Brokerage: ₹{result.brokerage:.2f}")
    print(f"  STT: ₹{result.stt:.2f}")
    print(f"  Exchange Charges: ₹{result.exchange_charges:.2f}")
    print(f"  GST: ₹{result.gst:.2f}")
    print(f"  SEBI Charges: ₹{result.sebi_charges:.2f}")
    print(f"  Total Cost: ₹{result.total_cost:.2f}")
    print(f"  Effective Rate: {(result.total_cost / total_premium * 100):.3f}%")
    print()


def batch_calculation_example():
    """Demonstrate batch processing for multiple transactions."""
    print("=== Batch Calculation Example ===")
    
    # Create multiple transaction requests
    transactions = [
        TransactionRequest(
            symbol='RELIANCE',
            quantity=100,
            price=Decimal('2500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        ),
        TransactionRequest(
            symbol='TCS',
            quantity=50,
            price=Decimal('3200.00'),
            transaction_type=TransactionType.SELL,
            instrument_type=InstrumentType.EQUITY
        ),
        TransactionRequest(
            symbol='INFY',
            quantity=75,
            price=Decimal('1500.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        ),
        TransactionRequest(
            symbol='HDFCBANK',
            quantity=25,
            price=Decimal('1600.00'),
            transaction_type=TransactionType.SELL,
            instrument_type=InstrumentType.EQUITY
        )
    ]
    
    calculator = ZerodhaCalculator()
    
    # Calculate batch - this is more efficient than individual calculations
    results = calculator.calculate_batch(transactions)
    
    print("Portfolio Transactions:")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Type':<5} {'Qty':<5} {'Price':<10} {'Value':<12} {'Cost':<8} {'Rate':<6}")
    print("-" * 80)
    
    total_value = Decimal('0')
    total_cost = Decimal('0')
    
    for transaction, result in zip(transactions, results):
        value = transaction.quantity * transaction.price
        rate = (result.total_cost / value * 100)
        
        print(f"{transaction.symbol:<12} "
              f"{transaction.transaction_type.name:<5} "
              f"{transaction.quantity:<5} "
              f"₹{transaction.price:<9} "
              f"₹{value:>11,.2f} "
              f"₹{result.total_cost:>7.2f} "
              f"{rate:>5.3f}%")
        
        total_value += value
        total_cost += result.total_cost
    
    print("-" * 80)
    print(f"{'TOTAL':<37} ₹{total_value:>11,.2f} ₹{total_cost:>7.2f} {(total_cost/total_value*100):>5.3f}%")
    print()
    
    print(f"Portfolio Summary:")
    print(f"  Total Trade Value: ₹{total_value:,.2f}")
    print(f"  Total Transaction Cost: ₹{total_cost:.2f}")
    print(f"  Overall Cost Rate: {(total_cost/total_value*100):.3f}%")
    print()


def high_frequency_trading_simulation():
    """Simulate high-frequency trading scenario."""
    print("=== High-Frequency Trading Simulation ===")
    
    # Simulate 20 small intraday trades
    trades = []
    for i in range(20):
        trade = TransactionRequest(
            symbol=f'STOCK{i%5 + 1}',  # Rotate between 5 stocks
            quantity=10 + (i % 3) * 5,  # 10, 15, or 20 shares
            price=Decimal(f'{1000 + i * 50}.00'),  # Varying prices
            transaction_type=TransactionType.BUY if i % 2 == 0 else TransactionType.SELL,
            instrument_type=InstrumentType.EQUITY,
            metadata={'product_type': 'MIS'}  # Intraday
        )
        trades.append(trade)
    
    calculator = ZerodhaCalculator()
    
    # Time the batch calculation
    import time
    start_time = time.time()
    results = calculator.calculate_batch(trades, parallel=True)
    end_time = time.time()
    
    # Analyze results
    total_value = sum(trade.quantity * trade.price for trade in trades)
    total_cost = sum(result.total_cost for result in results)
    avg_cost_per_trade = total_cost / len(trades)
    
    print(f"High-Frequency Trading Analysis:")
    print(f"  Number of Trades: {len(trades)}")
    print(f"  Total Trade Value: ₹{total_value:,.2f}")
    print(f"  Total Transaction Cost: ₹{total_cost:.2f}")
    print(f"  Average Cost per Trade: ₹{avg_cost_per_trade:.2f}")
    print(f"  Overall Cost Rate: {(total_cost/total_value*100):.3f}%")
    print(f"  Calculation Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"  Throughput: {len(trades)/(end_time - start_time):.1f} calculations/second")
    print()
    
    # Show impact of transaction costs on profitability
    print("Impact on Trading Strategy:")
    gross_pnl = Decimal('5000.00')  # Assume ₹5000 gross profit
    net_pnl = gross_pnl - total_cost
    cost_impact = (total_cost / gross_pnl * 100)
    
    print(f"  Gross P&L: ₹{gross_pnl:,.2f}")
    print(f"  Transaction Costs: ₹{total_cost:.2f}")
    print(f"  Net P&L: ₹{net_pnl:,.2f}")
    print(f"  Cost Impact: {cost_impact:.1f}% of gross profit")
    print()


def calculate_breakeven_points():
    """Calculate breakeven points for different trading scenarios."""
    print("=== Breakeven Analysis ===")
    
    scenarios = [
        {
            'name': 'Small Intraday Trade',
            'request': TransactionRequest(
                symbol='NIFTY_ETF',
                quantity=10,
                price=Decimal('200.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY,
                metadata={'product_type': 'MIS'}
            )
        },
        {
            'name': 'Medium Delivery Trade',
            'request': TransactionRequest(
                symbol='LARGECAP',
                quantity=100,
                price=Decimal('1000.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY,
                metadata={'product_type': 'CNC'}
            )
        },
        {
            'name': 'Large Delivery Trade',
            'request': TransactionRequest(
                symbol='BLUECHIP',
                quantity=1000,
                price=Decimal('500.00'),
                transaction_type=TransactionType.BUY,
                instrument_type=InstrumentType.EQUITY,
                metadata={'product_type': 'CNC'}
            )
        }
    ]
    
    calculator = ZerodhaCalculator()
    
    for scenario in scenarios:
        request = scenario['request']
        
        # Calculate buy cost
        buy_result = calculator.calculate_cost(request)
        
        # Calculate sell cost (assume same price for simplicity)
        sell_request = request
        sell_request.transaction_type = TransactionType.SELL
        sell_result = calculator.calculate_cost(sell_request)
        
        total_cost = buy_result.total_cost + sell_result.total_cost
        trade_value = request.quantity * request.price
        breakeven_percentage = (total_cost / trade_value * 100)
        breakeven_price_increase = request.price * (total_cost / trade_value)
        
        print(f"{scenario['name']}:")
        print(f"  Trade Size: {request.quantity} shares at ₹{request.price}")
        print(f"  Trade Value: ₹{trade_value:,.2f}")
        print(f"  Buy Cost: ₹{buy_result.total_cost:.2f}")
        print(f"  Sell Cost: ₹{sell_result.total_cost:.2f}")
        print(f"  Total Round-trip Cost: ₹{total_cost:.2f}")
        print(f"  Breakeven %: {breakeven_percentage:.3f}%")
        print(f"  Required Price Increase: ₹{breakeven_price_increase:.2f}")
        print(f"  Breakeven Sell Price: ₹{request.price + breakeven_price_increase:.2f}")
        print()


def main():
    """Run all examples."""
    print("Transaction Cost Calculation Examples")
    print("=" * 50)
    print()
    
    try:
        basic_equity_calculation()
        compare_delivery_vs_intraday()
        broker_comparison()
        options_trading_example()
        batch_calculation_example()
        high_frequency_trading_simulation()
        calculate_breakeven_points()
        
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("- Try modifying the examples with your own stock symbols and quantities")
        print("- Explore the advanced examples in ml_integration.py")
        print("- Check out the configuration options in the user guide")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Please check your setup and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())