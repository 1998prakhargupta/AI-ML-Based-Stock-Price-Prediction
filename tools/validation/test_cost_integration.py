"""
Test Cost Reporting Integration
===============================

Basic test to validate cost reporting and visualization integration.
"""

import sys
import os
import logging
from datetime import datetime
from decimal import Decimal

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_cost_reporting_imports():
    """Test that cost reporting modules can be imported."""
    try:
        from src.visualization.cost_reporting.cost_reporter import CostReporter
        from src.visualization.cost_reporting.cost_analyzer import CostAnalyzer
        from src.visualization.cost_reporting.cost_summary_generator import CostSummaryGenerator
        
        print("✅ Cost reporting modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Cost reporting import failed: {e}")
        return False

def test_cost_charts_imports():
    """Test that cost chart modules can be imported."""
    try:
        from src.visualization.cost_charts.breakdown_charts import CostBreakdownCharts
        from src.visualization.cost_charts.impact_charts import CostImpactCharts
        from src.visualization.cost_charts.comparison_charts import CostComparisonCharts
        
        print("✅ Cost chart modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Cost charts import failed: {e}")
        return False

def test_automated_reporting_integration():
    """Test that automated reporting includes cost capabilities."""
    try:
        from src.visualization.automated_reporting import AutomatedReportGenerator
        from src.utils.config_manager import Config
        
        config = Config()
        
        # Try to initialize with cost reporting
        report_gen = AutomatedReportGenerator(config)
        
        # Check if cost reporting components are available
        has_cost_reporter = hasattr(report_gen, 'cost_reporter')
        has_cost_analyzer = hasattr(report_gen, 'cost_analyzer')
        has_cost_charts = hasattr(report_gen, 'cost_breakdown_charts')
        
        if has_cost_reporter and has_cost_analyzer and has_cost_charts:
            print("✅ Automated reporting integration successful")
            return True
        else:
            print(f"❌ Missing cost components: reporter={has_cost_reporter}, analyzer={has_cost_analyzer}, charts={has_cost_charts}")
            return False
    except Exception as e:
        print(f"❌ Automated reporting integration failed: {e}")
        return False

def test_config_manager_cost_settings():
    """Test cost-related configuration settings."""
    try:
        from src.utils.config_manager import Config
        
        config = Config()
        
        # Test cost reporting config
        cost_config = config.get_cost_reporting_config()
        required_keys = ['enabled', 'include_charts', 'include_broker_comparison']
        
        has_all_keys = all(key in cost_config for key in required_keys)
        
        if has_all_keys:
            print("✅ Cost configuration settings available")
            print(f"   Cost reporting enabled: {cost_config['enabled']}")
            print(f"   Include charts: {cost_config['include_charts']}")
            print(f"   Include broker comparison: {cost_config['include_broker_comparison']}")
            return True
        else:
            print(f"❌ Missing cost config keys: {[k for k in required_keys if k not in cost_config]}")
            return False
    except Exception as e:
        print(f"❌ Cost configuration test failed: {e}")
        return False

def test_basic_cost_analysis():
    """Test basic cost analysis functionality."""
    try:
        from src.visualization.cost_reporting.cost_reporter import CostReporter
        from src.trading.transaction_costs.models import (
            TransactionRequest, TransactionCostBreakdown, 
            TransactionType, InstrumentType, OrderType
        )
        
        # Create sample data
        sample_request = TransactionRequest(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.00"),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY,
            order_type=OrderType.MARKET,
            timestamp=datetime.now()
        )
        
        sample_breakdown = TransactionCostBreakdown(
            commission=Decimal("5.00"),
            regulatory_fees=Decimal("0.50"),
            exchange_fees=Decimal("1.00"),
            bid_ask_spread_cost=Decimal("2.00"),
            market_impact_cost=Decimal("3.00")
        )
        
        # Test cost reporter initialization
        cost_reporter = CostReporter()
        
        print("✅ Basic cost analysis components working")
        print(f"   Sample transaction value: ${sample_request.notional_value}")
        print(f"   Sample total cost: ${sample_breakdown.total_cost}")
        print(f"   Cost in bps: {sample_breakdown.cost_as_basis_points(sample_request.notional_value):.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Basic cost analysis test failed: {e}")
        return False

def run_all_tests():
    """Run all cost reporting integration tests."""
    print("🧪 Running Cost Reporting Integration Tests")
    print("=" * 50)
    
    tests = [
        test_cost_reporting_imports,
        test_cost_charts_imports,
        test_automated_reporting_integration,
        test_config_manager_cost_settings,
        test_basic_cost_analysis
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All cost reporting integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    success = run_all_tests()
    sys.exit(0 if success else 1)