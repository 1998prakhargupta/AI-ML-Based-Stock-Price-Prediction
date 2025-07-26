#!/usr/bin/env python3
"""
Import Migration Script
======================

This script migrates all import statements from symbolic links to the new organized structure.
Systematically updates all Python files to use proper src.* imports.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImportMigrator:
    """Handles systematic migration of import statements."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Define import mappings from old to new
        self.import_mappings = {
            'from src.utils.app_config import': 'from src.utils.app_config import',
            'from src.utils.file_management_utils import': 'from src.utils.file_management_utils import',
            'from src.visualization.visualization_utils import': 'from src.visualization.visualization_utils import',
            'from src.models.model_utils import': 'from src.models.model_utils import',
            'from src.utils.reproducibility_utils import': 'from src.utils.reproducibility_utils import',
            'from src.visualization.automated_reporting import': 'from src.visualization.automated_reporting import',
            
            # Handle specific class imports
            'from src.utils import app_config': 'from src.utils from src.utils import app_config',
            'from src.utils import file_management_utils': 'from src.utils from src.utils import file_management_utils',
            'from src.visualization import visualization_utils': 'from src.visualization from src.visualization import visualization_utils',
            'from src.models import model_utils': 'from src.models from src.models import model_utils',
            'from src.utils import reproducibility_utils': 'from src.utils from src.utils import reproducibility_utils',
            'from src.visualization import automated_reporting': 'from src.visualization from src.visualization import automated_reporting',
        }
        
        # Additional relative import fixes for files already in src/
        self.relative_mappings = {
            'from src.utils.file_management_utils import': 'from src.utils.file_management_utils import',
            'from src.utils.app_config import': 'from src.utils.app_config import',
            'from src.utils.reproducibility_utils import': 'from src.utils.reproducibility_utils import',
            'from src.visualization.visualization_utils import': 'from src.visualization.visualization_utils import',
            'from src.visualization.automated_reporting import': 'from src.visualization.automated_reporting import',
        }
        
        # Files to skip (already organized or don't need migration)
        self.skip_files = {
            'src/utils/__init__.py',
            'src/visualization/__init__.py',
            'src/models/__init__.py',
            'src/utils/app_config.py',
            'src/utils/file_management_utils.py',
            'src/visualization/visualization_utils.py',
            'src/models/model_utils.py',
            'src/utils/reproducibility_utils.py',
            'src/visualization/automated_reporting.py'
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files that need migration."""
        python_files = []
        
        # Scan all directories except __pycache__
        for path in self.project_root.rglob("*.py"):
            relative_path = path.relative_to(self.project_root)
            
            # Skip __pycache__ and other build directories
            if '__pycache__' in str(relative_path):
                continue
                
            # Skip files that don't need migration
            if str(relative_path) in self.skip_files:
                continue
                
            python_files.append(path)
        
        return python_files
    
    def analyze_file_imports(self, file_path: Path) -> List[Tuple[str, str]]:
        """Analyze a file and find imports that need migration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return []
        
        migrations_needed = []
        
        for old_import, new_import in self.import_mappings.items():
            if old_import in content:
                migrations_needed.append((old_import, new_import))
        
        # Check relative imports for files in src/
        if str(file_path).startswith(str(self.project_root / 'src')):
            for old_import, new_import in self.relative_mappings.items():
                if old_import in content:
                    migrations_needed.append((old_import, new_import))
        
        return migrations_needed
    
    def migrate_file_imports(self, file_path: Path, dry_run: bool = True) -> bool:
        """Migrate imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return False
        
        original_content = content
        migrations_made = []
        
        # Apply all mappings
        all_mappings = {**self.import_mappings, **self.relative_mappings}
        
        for old_import, new_import in all_mappings.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                migrations_made.append((old_import, new_import))
        
        # If changes were made
        if content != original_content:
            if not dry_run:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"✅ Migrated imports in {file_path}")
                    for old, new in migrations_made:
                        logger.info(f"   {old} → {new}")
                    return True
                except Exception as e:
                    logger.error(f"Could not write {file_path}: {e}")
                    return False
            else:
                logger.info(f"🔍 Would migrate imports in {file_path}:")
                for old, new in migrations_made:
                    logger.info(f"   {old} → {new}")
                return True
        
        return False
    
    def run_migration(self, dry_run: bool = True) -> Dict[str, int]:
        """Run the complete import migration."""
        logger.info(f"Starting import migration (dry_run={dry_run})")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to check")
        
        stats = {
            'files_checked': 0,
            'files_migrated': 0,
            'migrations_applied': 0
        }
        
        for file_path in python_files:
            stats['files_checked'] += 1
            
            # Analyze what needs migration
            migrations_needed = self.analyze_file_imports(file_path)
            
            if migrations_needed:
                if self.migrate_file_imports(file_path, dry_run):
                    stats['files_migrated'] += 1
                    stats['migrations_applied'] += len(migrations_needed)
        
        return stats


def main():
    """Main function to run import migration."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    migrator = ImportMigrator(project_root)
    
    # First run a dry run to see what would be changed
    logger.info("="*60)
    logger.info("🔍 DRY RUN - Analyzing import migration needs")
    logger.info("="*60)
    
    dry_run_stats = migrator.run_migration(dry_run=True)
    
    logger.info("="*60)
    logger.info("📊 DRY RUN RESULTS")
    logger.info("="*60)
    logger.info(f"Files checked: {dry_run_stats['files_checked']}")
    logger.info(f"Files needing migration: {dry_run_stats['files_migrated']}")
    logger.info(f"Total migrations needed: {dry_run_stats['migrations_applied']}")
    
    if dry_run_stats['files_migrated'] > 0:
        response = input("\n✅ Proceed with actual migration? (y/N): ")
        if response.lower().startswith('y'):
            logger.info("="*60)
            logger.info("🚀 EXECUTING IMPORT MIGRATION")
            logger.info("="*60)
            
            actual_stats = migrator.run_migration(dry_run=False)
            
            logger.info("="*60)
            logger.info("✅ MIGRATION COMPLETE")
            logger.info("="*60)
            logger.info(f"Files migrated: {actual_stats['files_migrated']}")
            logger.info(f"Total migrations applied: {actual_stats['migrations_applied']}")
        else:
            logger.info("Migration cancelled by user")
    else:
        logger.info("No migrations needed - all imports are already up to date!")


if __name__ == "__main__":
    main()
