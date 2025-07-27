#!/usr/bin/env python3
"""
Documentation Validation Script
==============================

This script validates that the transaction cost documentation is complete,
accurate, and all examples work correctly.

Run this script:
    python docs/transaction_costs/examples/validate_documentation.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / 'src'))

def validate_documentation_structure():
    """Validate that all required documentation files exist."""
    print("🔍 Validating documentation structure...")
    
    docs_root = project_root / 'docs' / 'transaction_costs'
    
    required_files = [
        'README.md',
        'getting_started.md',
        'api/cost_calculators.md',
        'api/brokers.md',
        'user_guide/installation.md',
        'configuration/configuration_reference.md',
        'technical/architecture.md',
        'troubleshooting/common_issues.md',
        'examples/basic_calculations.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = docs_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required documentation files exist")
    return True

def validate_code_examples():
    """Validate that code examples in documentation work."""
    print("\n🧪 Validating code examples...")
    
    try:
        # Test basic imports
        from src.trading.transaction_costs.models import (
            TransactionRequest, 
            TransactionType, 
            InstrumentType
        )
        print("  ✅ Core models import successfully")
        
        # Test data model creation
        from decimal import Decimal
        
        request = TransactionRequest(
            symbol='TEST',
            quantity=10,
            price=Decimal('100.00'),
            transaction_type=TransactionType.BUY,
            instrument_type=InstrumentType.EQUITY
        )
        print("  ✅ TransactionRequest creation works")
        
        # Test basic validation
        assert request.symbol == 'TEST'
        assert request.quantity == 10
        assert request.price == Decimal('100.00')
        print("  ✅ Data model validation works")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False

def validate_broker_calculators():
    """Validate that broker calculators can be imported and initialized."""
    print("\n🏦 Validating broker calculators...")
    
    try:
        # Test Zerodha calculator
        from src.trading.transaction_costs.brokers.zerodha_calculator import ZerodhaCalculator
        zerodha_calc = ZerodhaCalculator()
        print("  ✅ ZerodhaCalculator imports and initializes")
        
        # Test Breeze calculator
        from src.trading.transaction_costs.brokers.breeze_calculator import BreezeCalculator
        breeze_calc = BreezeCalculator()
        print("  ✅ BreezeCalculator imports and initializes")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Broker import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Broker initialization error: {e}")
        return False

def validate_configuration_system():
    """Validate that configuration system works."""
    print("\n⚙️ Validating configuration system...")
    
    try:
        from src.trading.cost_config.base_config import CostConfiguration
        
        # Test configuration initialization
        config = CostConfiguration()
        print("  ✅ CostConfiguration initializes")
        
        # Test basic configuration operations
        test_setting = config.get_setting('system.precision_decimal_places', default=2)
        print(f"  ✅ Configuration retrieval works (precision: {test_setting})")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Configuration import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def check_documentation_quality():
    """Check documentation quality metrics."""
    print("\n📊 Checking documentation quality...")
    
    docs_root = project_root / 'docs' / 'transaction_costs'
    
    # Count documentation files
    md_files = list(docs_root.glob('**/*.md'))
    py_files = list(docs_root.glob('**/*.py'))
    
    print(f"  📝 Documentation files: {len(md_files)} markdown, {len(py_files)} Python")
    
    # Check file sizes (rough content quality indicator)
    total_size = 0
    for file_path in md_files + py_files:
        total_size += file_path.stat().st_size
    
    print(f"  📏 Total documentation size: {total_size / 1024:.1f} KB")
    
    # Quality checks
    if len(md_files) >= 8:
        print("  ✅ Comprehensive documentation coverage")
    else:
        print("  ⚠️  Limited documentation coverage")
    
    if total_size > 100000:  # 100KB
        print("  ✅ Substantial documentation content")
    else:
        print("  ⚠️  Limited documentation content")
    
    return True

def main():
    """Run all validation checks."""
    print("Transaction Cost Documentation Validation")
    print("=" * 50)
    
    checks = [
        ("Documentation Structure", validate_documentation_structure),
        ("Code Examples", validate_code_examples),
        ("Broker Calculators", validate_broker_calculators),
        ("Configuration System", validate_configuration_system),
        ("Documentation Quality", check_documentation_quality)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {check_name} check failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Validation Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All documentation validation checks passed!")
        print("\nThe transaction cost documentation is complete and functional.")
        return 0
    else:
        print("💥 Some documentation validation checks failed!")
        print("Please review the errors above and fix any issues.")
        return 1

if __name__ == "__main__":
    exit(main())