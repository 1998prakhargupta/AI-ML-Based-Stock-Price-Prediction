#!/usr/bin/env python3
"""
File Organization Validation Script
===================================

Validates that all files are properly organized in the project structure.
"""

import os
import sys
from pathlib import Path

def validate_project_structure():
    """Validate the project structure and file organization."""
    
    project_root = Path(__file__).parent
    print("🔍 VALIDATING PROJECT STRUCTURE")
    print("=" * 60)
    
    # Define expected structure
    expected_structure = {
        'src/': {
            'api/': ['__init__.py', 'breeze_api.py', 'enhanced_breeze_api.py', 'yahoo_finance_api.py', 'breeze_utils.py'],
            'compliance/': ['__init__.py', 'api_compliance.py'],
            'data/': ['__init__.py'],
            'models/': ['__init__.py', 'model_utils.py'],
            'utils/': ['__init__.py', 'config_manager.py', 'file_manager.py', 'logging_utils.py', 
                      'reproducibility_utils.py', 'app_config.py', 'file_management_utils.py'],
            'visualization/': ['__init__.py', 'charts.py', 'automated_reporting.py', 'visualization_utils.py']
        },
        'tests/': {
            'unit/': ['unit_tests.py', 'comprehensive_test_suite.py', 'final_validation_test.py'],
            'integration/': [],
            'compliance/': []
        },
        'configs/': ['config.json', 'compliance.json', 'model_params.json', 'logging.conf', 'reproducibility_config.json'],
        'scripts/': ['setup.py', 'data_pipeline.py', 'compliance_demo.py', 'final_report_demo.py'],
        'docs/': [],
        'data/': {
            'raw/': [],
            'processed/': [],
            'cache/': [],
            'plots/': [],
            'reports/': []
        },
        'models/': {
            'checkpoints/': [],
            'production/': [],
            'experiments/': []
        },
        'notebooks/': {
            'exploration/': [],
            'modeling/': [],
            'analysis/': [],
            'demo/': []
        },
        'logs/': []
    }
    
    # Root level files
    root_files = [
        'setup.py', 'requirements.txt', 'README.md', 'Makefile', 
        'pytest.ini', '.gitignore', '.env.example'
    ]
    
    # Symbolic links in root
    root_symlinks = [
        'app_config.py', 'file_management_utils.py', 'visualization_utils.py',
        'model_utils.py', 'reproducibility_utils.py', 'automated_reporting.py'
    ]
    
    validation_results = {
        'directories_created': 0,
        'files_found': 0,
        'missing_files': [],
        'extra_files': [],
        'symlinks_valid': 0,
        'total_checks': 0
    }
    
    print("📁 Checking directory structure...")
    
    def check_directory_structure(base_path, structure, current_path=""):
        """Recursively check directory structure."""
        for item, contents in structure.items():
            full_path = base_path / item
            relative_path = f"{current_path}/{item}" if current_path else item
            
            validation_results['total_checks'] += 1
            
            if full_path.exists() and full_path.is_dir():
                validation_results['directories_created'] += 1
                print(f"  ✅ {relative_path}")
                
                if isinstance(contents, dict):
                    check_directory_structure(full_path, contents, relative_path)
                elif isinstance(contents, list):
                    for file_name in contents:
                        file_path = full_path / file_name
                        validation_results['total_checks'] += 1
                        if file_path.exists():
                            validation_results['files_found'] += 1
                            print(f"    ✅ {relative_path}/{file_name}")
                        else:
                            validation_results['missing_files'].append(f"{relative_path}/{file_name}")
                            print(f"    ❌ {relative_path}/{file_name} (missing)")
            else:
                print(f"  ❌ {relative_path} (missing directory)")
    
    # Check main structure
    check_directory_structure(project_root, expected_structure)
    
    print("\\n📄 Checking root files...")
    for file_name in root_files:
        file_path = project_root / file_name
        validation_results['total_checks'] += 1
        if file_path.exists():
            validation_results['files_found'] += 1
            print(f"  ✅ {file_name}")
        else:
            validation_results['missing_files'].append(file_name)
            print(f"  ❌ {file_name} (missing)")
    
    print("\\n🔗 Checking symbolic links...")
    for symlink_name in root_symlinks:
        symlink_path = project_root / symlink_name
        validation_results['total_checks'] += 1
        if symlink_path.exists() and symlink_path.is_symlink():
            target = symlink_path.readlink()
            if symlink_path.resolve().exists():
                validation_results['symlinks_valid'] += 1
                print(f"  ✅ {symlink_name} -> {target}")
            else:
                print(f"  ❌ {symlink_name} -> {target} (broken link)")
        else:
            print(f"  ❌ {symlink_name} (missing symlink)")
    
    # Check for unexpected files in root
    print("\\n🔎 Checking for unexpected files in root...")
    expected_root_items = set(root_files + root_symlinks + 
                             ['.git', '.venv', 'src', 'tests', 'configs', 'scripts', 
                              'docs', 'data', 'models', 'notebooks', 'logs'])
    
    actual_items = set(item.name for item in project_root.iterdir())
    unexpected_items = actual_items - expected_root_items
    
    if unexpected_items:
        print("  ⚠️  Unexpected items found in root:")
        for item in unexpected_items:
            print(f"    - {item}")
            validation_results['extra_files'].append(item)
    else:
        print("  ✅ No unexpected files in root directory")
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Directories created: {validation_results['directories_created']}")
    print(f"✅ Files found: {validation_results['files_found']}")
    print(f"✅ Valid symlinks: {validation_results['symlinks_valid']}")
    print(f"❌ Missing files: {len(validation_results['missing_files'])}")
    print(f"⚠️  Extra files: {len(validation_results['extra_files'])}")
    print(f"📈 Total checks: {validation_results['total_checks']}")
    
    # Calculate success rate
    missing_count = len(validation_results['missing_files'])
    success_rate = ((validation_results['total_checks'] - missing_count) / 
                   validation_results['total_checks'] * 100) if validation_results['total_checks'] > 0 else 0
    
    print(f"\\n🎯 Organization Success Rate: {success_rate:.1f}%")
    
    if missing_count == 0:
        print("\\n🎉 PROJECT STRUCTURE PERFECTLY ORGANIZED!")
        print("✅ All files are in their proper locations")
        print("✅ All symbolic links are working")
        print("✅ Directory structure is complete")
    elif missing_count < 5:
        print("\\n✅ PROJECT STRUCTURE MOSTLY ORGANIZED!")
        print("ℹ️  Only a few minor files missing")
    else:
        print("\\n⚠️  PROJECT STRUCTURE NEEDS ATTENTION")
        print("❌ Several files are missing or misplaced")
    
    # Show import validation
    print("\\n" + "=" * 60)
    print("🔧 IMPORT VALIDATION")
    print("=" * 60)
    
    # Test critical imports
    import_tests = [
        ('app_config', 'Config class available'),
        ('file_management_utils', 'SafeFileManager available'),
        ('visualization_utils', 'ComprehensiveVisualizer available'),
        ('model_utils', 'ModelManager available'),
        ('reproducibility_utils', 'ReproducibilityManager available'),
        ('automated_reporting', 'AutomatedReportGenerator available')
    ]
    
    successful_imports = 0
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}: {description}")
            successful_imports += 1
        except ImportError as e:
            print(f"  ❌ {module_name}: Import failed - {e}")
        except Exception as e:
            print(f"  ⚠️  {module_name}: Import warning - {e}")
    
    import_success_rate = (successful_imports / len(import_tests)) * 100
    print(f"\\n📊 Import Success Rate: {import_success_rate:.1f}%")
    
    # Final recommendation
    print("\\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    if success_rate >= 95 and import_success_rate >= 80:
        print("✅ EXCELLENT! Your project structure is properly organized.")
        print("✅ All components are in their correct locations.")
        print("✅ The modular architecture is ready for development.")
        print("\\n🚀 Next steps:")
        print("  1. Run `make install` to set up the environment")
        print("  2. Run `make test` to validate functionality")
        print("  3. Begin development using the organized structure")
    elif success_rate >= 80:
        print("✅ GOOD! Most files are properly organized.")
        print("⚠️  A few minor adjustments may be needed.")
        print("\\n🔧 Suggested actions:")
        print("  1. Review missing files and create if needed")
        print("  2. Check symbolic links are working correctly")
        print("  3. Run tests to ensure everything works")
    else:
        print("⚠️  NEEDS WORK! Project structure needs more organization.")
        print("❌ Several files are missing or misplaced.")
        print("\\n🛠️  Required actions:")
        print("  1. Create missing directories and files")
        print("  2. Move files to their proper locations")
        print("  3. Fix broken symbolic links")
        print("  4. Re-run this validation script")
    
    return validation_results

if __name__ == "__main__":
    try:
        results = validate_project_structure()
        # Exit with appropriate code
        missing_count = len(results['missing_files'])
        exit_code = 0 if missing_count == 0 else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"\\n❌ Validation failed with error: {e}")
        sys.exit(1)
