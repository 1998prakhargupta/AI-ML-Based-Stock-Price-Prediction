#!/usr/bin/env python3
"""
Version Control Optimization Script
==================================

Updates .gitignore and creates .gitkeep files based on the organized project structure.
Ensures proper version control for the professional project layout.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VersionControlOptimizer:
    """Optimizes .gitignore and .gitkeep files for the organized project structure."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.gitignore_path = self.project_root / '.gitignore'
        
        # Define directory structure that needs .gitkeep files
        self.required_gitkeep_dirs = [
            'notebooks/analysis',
            'notebooks/demo', 
            'notebooks/exploration',
            'notebooks/modeling',
            'tests/fixtures',
            'tests/compliance',
            'tests/integration',
            'tests/unit'
        ]
        
        # Additional .gitignore patterns for organized structure
        self.additional_patterns = [
            "\n# =============================================================================\n",
            "# 🗂️ ORGANIZED PROJECT STRUCTURE SPECIFIC\n",
            "# =============================================================================\n",
            "# Source code cache and compiled files\n",
            "src/**/__pycache__/\n",
            "src/**/*.pyc\n",
            "src/**/*.pyo\n",
            "src/**/*.pyd\n",
            "\n",
            "# Import migration backup files\n", 
            "*.py.bak\n",
            "*.py.backup\n",
            "*_backup.py\n",
            "\n",
            "# Symbolic link validation\n",
            "import_validation_*.log\n",
            "migration_*.log\n",
            "\n",
            "# Configuration backups\n",
            "src/__init__.py.bak\n",
            "**/__init__.py.bak\n",
            "\n",
            "# Development and debugging\n",
            "debug_*.py\n",
            "test_import.py\n",
            "temp_*.py\n",
            "\n",
            "# File management metadata (specific to organized structure)\n",
            "data/**/.file_metadata/\n",
            "models/**/.file_metadata/\n",
            "\n",
            "# Experiment tracking\n",
            "experiments/\n",
            "experiment_*/\n",
            "mlruns/\n",
            "wandb/\n",
            "\n",
            "# Reproducibility artifacts\n",
            "reproducibility_state_*.json\n",
            "seed_*.json\n",
            "environment_*.yml\n",
            "\n"
        ]
    
    def analyze_current_gitignore(self) -> Dict[str, bool]:
        """Analyze current .gitignore coverage."""
        logger.info("🔍 Analyzing current .gitignore coverage...")
        
        coverage = {
            'credentials': False,
            'data_files': False,
            'ml_models': False,
            'logs': False,
            'python_cache': False,
            'virtual_envs': False,
            'jupyter': False,
            'testing': False,
            'organized_structure': False
        }
        
        if self.gitignore_path.exists():
            content = self.gitignore_path.read_text()
            
            # Check for various patterns
            if '.env' in content and 'credentials' in content:
                coverage['credentials'] = True
            if '*.csv' in content and 'data/' in content:
                coverage['data_files'] = True
            if '*.pkl' in content and 'models/' in content:
                coverage['ml_models'] = True
            if '*.log' in content and 'logs/' in content:
                coverage['logs'] = True
            if '__pycache__' in content:
                coverage['python_cache'] = True
            if 'venv/' in content:
                coverage['virtual_envs'] = True
            if '.ipynb_checkpoints' in content:
                coverage['jupyter'] = True
            if 'tests/' in content:
                coverage['testing'] = True
            if 'ORGANIZED PROJECT STRUCTURE' in content:
                coverage['organized_structure'] = True
        
        return coverage
    
    def update_gitignore(self) -> None:
        """Update .gitignore with organized structure patterns."""
        logger.info("📝 Updating .gitignore for organized structure...")
        
        coverage = self.analyze_current_gitignore()
        
        if coverage['organized_structure']:
            logger.info("✅ .gitignore already contains organized structure patterns")
            return
        
        # Read current content
        current_content = ""
        if self.gitignore_path.exists():
            current_content = self.gitignore_path.read_text()
        
        # Add new patterns
        new_content = current_content + "".join(self.additional_patterns)
        
        # Write updated content
        self.gitignore_path.write_text(new_content)
        logger.info(f"✅ Updated .gitignore with {len(self.additional_patterns)} new patterns")
    
    def create_gitkeep_files(self) -> None:
        """Create .gitkeep files for essential directories."""
        logger.info("📁 Creating .gitkeep files for directory structure...")
        
        created = 0
        existing = 0
        
        for dir_path in self.required_gitkeep_dirs:
            full_path = self.project_root / dir_path
            gitkeep_path = full_path / '.gitkeep'
            
            # Create directory if it doesn't exist
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep if it doesn't exist
            if not gitkeep_path.exists():
                gitkeep_content = f"# Keep {dir_path.split('/')[-1]} directory structure\n# {dir_path} - Essential for project organization\n"
                gitkeep_path.write_text(gitkeep_content)
                logger.info(f"  ✅ Created {gitkeep_path}")
                created += 1
            else:
                logger.info(f"  📁 {gitkeep_path} already exists")
                existing += 1
        
        logger.info(f"📊 .gitkeep summary: {created} created, {existing} existing")
    
    def validate_version_control(self) -> Dict[str, bool]:
        """Validate version control configuration."""
        logger.info("🔍 Validating version control configuration...")
        
        validation = {
            'gitignore_exists': self.gitignore_path.exists(),
            'gitignore_comprehensive': False,
            'gitkeep_coverage': False,
            'directory_structure': False
        }
        
        # Check .gitignore comprehensiveness
        if validation['gitignore_exists']:
            coverage = self.analyze_current_gitignore()
            validation['gitignore_comprehensive'] = sum(coverage.values()) >= 7  # Most patterns covered
        
        # Check .gitkeep coverage
        gitkeep_count = len(list(self.project_root.rglob('.gitkeep')))
        validation['gitkeep_coverage'] = gitkeep_count >= 10  # Good coverage
        
        # Check directory structure
        src_dirs = ['src/api', 'src/data', 'src/models', 'src/utils', 'src/visualization']
        validation['directory_structure'] = all((self.project_root / d).exists() for d in src_dirs)
        
        return validation
    
    def generate_report(self) -> str:
        """Generate a comprehensive version control report."""
        validation = self.validate_version_control()
        coverage = self.analyze_current_gitignore()
        
        report = [
            "# 📋 VERSION CONTROL OPTIMIZATION REPORT\n",
            "## 🎯 Configuration Status\n",
            f"- .gitignore exists: {'✅' if validation['gitignore_exists'] else '❌'}\n",
            f"- Comprehensive patterns: {'✅' if validation['gitignore_comprehensive'] else '❌'}\n", 
            f"- .gitkeep coverage: {'✅' if validation['gitkeep_coverage'] else '❌'}\n",
            f"- Directory structure: {'✅' if validation['directory_structure'] else '❌'}\n",
            "\n## 📊 .gitignore Coverage\n"
        ]
        
        for category, covered in coverage.items():
            status = '✅' if covered else '❌'
            report.append(f"- {category.replace('_', ' ').title()}: {status}\n")
        
        report.extend([
            "\n## 📁 Directory Structure\n",
            f"- Required .gitkeep directories: {len(self.required_gitkeep_dirs)}\n",
            f"- .gitkeep files found: {len(list(self.project_root.rglob('.gitkeep')))}\n",
            "\n## 🎉 Recommendations\n"
        ])
        
        if all(validation.values()):
            report.append("✅ Version control is optimally configured!\n")
        else:
            report.append("⚠️ Some optimization needed - run update commands\n")
        
        return "".join(report)
    
    def run_optimization(self) -> None:
        """Run complete version control optimization."""
        logger.info("🚀 Starting version control optimization...")
        logger.info("=" * 60)
        
        # Update .gitignore
        self.update_gitignore()
        
        # Create .gitkeep files
        self.create_gitkeep_files()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_path = self.project_root / 'docs' / 'VERSION_CONTROL_OPTIMIZATION.md'
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ VERSION CONTROL OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📄 Report saved: {report_path}")
        print("\n" + report)

def main():
    """Main execution function."""
    optimizer = VersionControlOptimizer()
    
    print("🔧 VERSION CONTROL OPTIMIZATION")
    print("=" * 40)
    
    try:
        optimizer.run_optimization()
        print("\n🎉 All optimizations completed successfully!")
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
