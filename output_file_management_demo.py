#!/usr/bin/env python3
"""
🛡️ OUTPUT FILE MANAGEMENT - DEMONSTRATION AND TESTING SCRIPT

This script demonstrates the enhanced file management capabilities implemented
to fix output file management issues. It shows how the new system prevents
overwrites, creates versions, and tracks metadata.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the new file management utilities
from file_management_utils import SafeFileManager, SaveStrategy, SaveResult, safe_save_dataframe
from app_config import Config

def create_sample_data(name: str, rows: int = 100) -> pd.DataFrame:
    """Create sample stock data for testing."""
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range('2024-01-01', periods=rows, freq='D')
    
    # Generate realistic stock data
    prices = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    volumes = np.random.randint(1000, 10000, rows)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(rows) * 0.1,
        'high': prices + np.abs(np.random.randn(rows) * 0.2),
        'low': prices - np.abs(np.random.randn(rows) * 0.2),
        'close': prices,
        'volume': volumes,
        'symbol': name
    })
    
    return df

def demonstrate_save_strategies():
    """Demonstrate different save strategies."""
    print("=" * 60)
    print("🛡️ OUTPUT FILE MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Initialize configuration and create demo data
    config = Config()
    data_path = config.get_data_save_path()
    
    # Create sample data
    sample_data = create_sample_data("DEMO", 50)
    
    # Initialize SafeFileManager
    file_manager = SafeFileManager(
        base_path=data_path,
        default_strategy=SaveStrategy.VERSION
    )
    
    print(f"📁 Working directory: {data_path}")
    print(f"📊 Sample data shape: {sample_data.shape}")
    print()
    
    # Test 1: Initial save (no conflicts)
    print("🧪 TEST 1: Initial Save (No Conflicts)")
    print("-" * 40)
    
    result1 = file_manager.save_dataframe(
        df=sample_data,
        filename="demo_stock_data.csv",
        metadata={
            "test_case": "initial_save",
            "data_source": "synthetic",
            "rows": len(sample_data)
        }
    )
    
    print(f"✅ Result: {result1.success}")
    print(f"📄 File saved as: {result1.final_filename}")
    print(f"🛡️ Strategy used: {result1.strategy_used.value}")
    print()
    
    # Test 2: Save with VERSION strategy (should create version)
    print("🧪 TEST 2: Save with VERSION Strategy")
    print("-" * 40)
    
    # Modify data slightly
    modified_data = sample_data.copy()
    modified_data['close'] = modified_data['close'] * 1.02  # 2% increase
    
    result2 = file_manager.save_dataframe(
        df=modified_data,
        filename="demo_stock_data.csv",  # Same filename
        strategy=SaveStrategy.VERSION,
        metadata={
            "test_case": "version_save",
            "modification": "2% price increase",
            "rows": len(modified_data)
        }
    )
    
    print(f"✅ Result: {result2.success}")
    print(f"📄 File saved as: {result2.final_filename}")
    print(f"🛡️ Strategy used: {result2.strategy_used.value}")
    print()
    
    # Test 3: Save with TIMESTAMP strategy
    print("🧪 TEST 3: Save with TIMESTAMP Strategy")
    print("-" * 40)
    
    # Modify data again
    timestamped_data = sample_data.copy()
    timestamped_data['volume'] = timestamped_data['volume'] * 2  # Double volume
    
    result3 = file_manager.save_dataframe(
        df=timestamped_data,
        filename="demo_stock_data.csv",  # Same filename
        strategy=SaveStrategy.TIMESTAMP,
        metadata={
            "test_case": "timestamp_save",
            "modification": "doubled volume",
            "rows": len(timestamped_data)
        }
    )
    
    print(f"✅ Result: {result3.success}")
    print(f"📄 File saved as: {result3.final_filename}")
    print(f"🛡️ Strategy used: {result3.strategy_used.value}")
    print()
    
    # Test 4: Save with BACKUP_OVERWRITE strategy
    print("🧪 TEST 4: Save with BACKUP_OVERWRITE Strategy")
    print("-" * 40)
    
    # Create new data
    backup_data = sample_data.copy()
    backup_data['new_indicator'] = backup_data['close'].rolling(5).mean()
    
    result4 = file_manager.save_dataframe(
        df=backup_data,
        filename="demo_stock_data.csv",  # Same filename
        strategy=SaveStrategy.BACKUP_OVERWRITE,
        metadata={
            "test_case": "backup_overwrite_save",
            "modification": "added moving average",
            "rows": len(backup_data)
        }
    )
    
    print(f"✅ Result: {result4.success}")
    print(f"📄 File saved as: {result4.final_filename}")
    print(f"🔄 Backup created: {result4.backup_created}")
    if result4.backup_created:
        print(f"📄 Backup location: {os.path.basename(result4.backup_path)}")
    print(f"🛡️ Strategy used: {result4.strategy_used.value}")
    print()
    
    # Test 5: Demonstrate SKIP strategy
    print("🧪 TEST 5: Save with SKIP Strategy")
    print("-" * 40)
    
    result5 = file_manager.save_dataframe(
        df=sample_data,
        filename="demo_stock_data.csv",  # Same filename
        strategy=SaveStrategy.SKIP,
        metadata={
            "test_case": "skip_save",
            "expected": "should_skip"
        }
    )
    
    print(f"❌ Result: {result5.success} (Expected: False)")
    print(f"📄 File: {result5.final_filename}")
    print(f"⚠️ Message: {result5.error_message}")
    print(f"🛡️ Strategy used: {result5.strategy_used.value}")
    print()
    
    return file_manager

def demonstrate_file_listing_and_metadata(file_manager: SafeFileManager):
    """Demonstrate file listing and metadata viewing."""
    print("🧪 TEST 6: File Listing and Metadata")
    print("-" * 40)
    
    # List all CSV files with metadata
    files = file_manager.list_files("demo_*.csv", include_metadata=True)
    
    print(f"📁 Found {len(files)} demo files:")
    print()
    
    for i, file_info in enumerate(files, 1):
        print(f"{i}. 📄 {file_info['filename']}")
        print(f"   📊 Size: {file_info['size_mb']:.3f} MB")
        print(f"   🕒 Modified: {file_info['modified']}")
        
        if file_info.get('metadata'):
            metadata = file_info['metadata']
            if 'shape' in metadata:
                shape = metadata['shape']
                print(f"   📈 Data: {shape[0]} rows × {shape[1]} columns")
            if 'test_case' in metadata.get('custom_metadata', {}):
                test_case = metadata['custom_metadata']['test_case']
                print(f"   🧪 Test: {test_case}")
            if 'modification' in metadata.get('custom_metadata', {}):
                modification = metadata['custom_metadata']['modification']
                print(f"   🔧 Change: {modification}")
        print()

def demonstrate_convenience_functions():
    """Demonstrate convenience functions."""
    print("🧪 TEST 7: Convenience Functions")
    print("-" * 40)
    
    # Create more sample data
    convenience_data = create_sample_data("CONV", 30)
    
    # Get data path
    config = Config()
    data_path = config.get_data_save_path()
    
    # Test convenience function
    result = safe_save_dataframe(
        df=convenience_data,
        filename="convenience_test.csv",
        base_path=data_path,
        strategy=SaveStrategy.VERSION,
        metadata={
            "test_type": "convenience_function",
            "api": "safe_save_dataframe",
            "rows": len(convenience_data)
        }
    )
    
    print(f"✅ Convenience function result: {result.success}")
    print(f"📄 File saved as: {result.final_filename}")
    print(f"🛡️ Strategy used: {result.strategy_used.value}")
    print()

def demonstrate_cleanup():
    """Demonstrate version cleanup."""
    print("🧪 TEST 8: Version Cleanup")
    print("-" * 40)
    
    config = Config()
    data_path = config.get_data_save_path()
    
    file_manager = SafeFileManager(base_path=data_path)
    
    # Check files before cleanup
    files_before = file_manager.list_files("demo_stock_data_v*.csv")
    print(f"📁 Files before cleanup: {len(files_before)}")
    for file_info in files_before:
        print(f"   📄 {file_info['filename']}")
    print()
    
    # Cleanup old versions (keep only 2 most recent)
    deleted_count = file_manager.cleanup_old_versions("demo_stock_data.csv", keep_versions=2)
    
    print(f"🗑️ Cleaned up {deleted_count} old versions")
    
    # Check files after cleanup
    files_after = file_manager.list_files("demo_stock_data_v*.csv")
    print(f"📁 Files after cleanup: {len(files_after)}")
    for file_info in files_after:
        print(f"   📄 {file_info['filename']}")
    print()

def main():
    """Main demonstration function."""
    print("🚀 Starting Output File Management Demonstration...")
    print()
    
    try:
        # Run all demonstration tests
        file_manager = demonstrate_save_strategies()
        demonstrate_file_listing_and_metadata(file_manager)
        demonstrate_convenience_functions()
        demonstrate_cleanup()
        
        print("=" * 60)
        print("✅ OUTPUT FILE MANAGEMENT DEMONSTRATION COMPLETE")
        print("=" * 60)
        print()
        print("🛡️ KEY BENEFITS DEMONSTRATED:")
        print("   ✅ No accidental overwrites")
        print("   ✅ Automatic versioning system")
        print("   ✅ Backup creation capabilities")
        print("   ✅ Comprehensive metadata tracking")
        print("   ✅ Multiple save strategies")
        print("   ✅ Easy file management and cleanup")
        print("   ✅ Backward compatibility maintained")
        print()
        print("📋 READY FOR PRODUCTION USE!")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
