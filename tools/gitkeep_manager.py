#!/usr/bin/env python3
"""
Enterprise .gitkeep Management Utility
======================================

Manages .gitkeep files for the Price Predictor enterprise project structure.
Ensures all essential directories are preserved in version control.

Author: 1998prakhargupta
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class GitkeepManager:
    """Manages .gitkeep files for enterprise project structure."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the GitkeepManager."""
        self.project_root = project_root or Path(__file__).parent.parent
        
        # Define all directories that need .gitkeep files
        self.required_directories = [
            # Application layer
            'app/core/cache',
            'app/core/temp', 
            'app/core/runtime',
            'app/core/uploads',
            'app/core/downloads',
            'app/logs',
            'app/tmp',
            'app/sessions',
            
            # Configuration management
            'config/generated',
            'config/compiled', 
            'config/cache',
            'config/backup',
            'config/secrets',
            'config/private',
            
            # Deployment infrastructure
            'deployments/docker/context',
            'deployments/docker/volumes',
            'deployments/docker/secrets',
            'deployments/docker/logs',
            'deployments/docker/data',
            'deployments/docker/cache',
            'deployments/kubernetes/secrets',
            'deployments/kubernetes/configmaps',
            'deployments/kubernetes/logs',
            'deployments/kubernetes/data',
            'deployments/kubernetes/backups',
            'deployments/builds',
            'deployments/releases',
            'deployments/artifacts',
            'deployments/temp',
            
            # Development tools
            'tools/temp',
            'tools/output',
            'tools/cache',
            'tools/logs',
            'tools/reports',
            'tools/builds',
            'tools/artifacts',
            'tools/generated',
            
            # External dependencies
            'external/vendor',
            'external/downloads',
            'external/cache',
            'external/temp',
            'external/logs',
            'external/apis/cache',
            'external/apis/logs',
            'external/plugins/cache',
            'external/plugins/logs',
            
            # Logging infrastructure
            'logs/application',
            'logs/api',
            'logs/background',
            'logs/workers',
            'logs/celery',
            'logs/gunicorn',
            
            # Monitoring and observability
            'metrics',
            'monitoring',
            'telemetry',
            
            # Data management
            'data/backups',
            
            # Model management
            'models/archived',
            
            # Testing infrastructure
            'tests/fixtures',
            'tests/output',
            'tests/reports',
            'tests/coverage',
            'tests/screenshots'
        ]
    
    def create_gitkeep(self, dir_path: str) -> bool:
        """Create a .gitkeep file in the specified directory."""
        full_path = self.project_root / dir_path
        gitkeep_path = full_path / '.gitkeep'
        
        # Create directory if it doesn't exist
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Skip if .gitkeep already exists
        if gitkeep_path.exists():
            return False
            
        # Create descriptive content
        dir_name = full_path.name
        relative_path = str(Path(dir_path))
        
        content = f"""# Keep {dir_name} directory structure
# {relative_path} - Essential for enterprise project organization
# This directory is part of the Price Predictor enterprise structure
# Author: 1998prakhargupta
"""
        
        gitkeep_path.write_text(content)
        return True
    
    def validate_gitkeeps(self) -> Dict[str, bool]:
        """Validate that all required .gitkeep files exist."""
        results = {}
        
        for directory in self.required_directories:
            gitkeep_path = self.project_root / directory / '.gitkeep'
            results[directory] = gitkeep_path.exists()
        
        return results
    
    def create_all_gitkeeps(self) -> Tuple[int, int]:
        """Create all required .gitkeep files."""
        created = 0
        existing = 0
        
        print("🔧 Creating .gitkeep files for enterprise structure...")
        
        for directory in self.required_directories:
            if self.create_gitkeep(directory):
                print(f"  ✅ Created .gitkeep in {directory}")
                created += 1
            else:
                print(f"  📁 .gitkeep already exists in {directory}")
                existing += 1
        
        return created, existing
    
    def cleanup_orphaned_gitkeeps(self) -> int:
        """Remove .gitkeep files from directories that shouldn't have them."""
        removed = 0
        
        # Find all .gitkeep files in the project
        gitkeep_files = list(self.project_root.rglob('.gitkeep'))
        
        for gitkeep_file in gitkeep_files:
            # Get relative directory path
            dir_path = gitkeep_file.parent.relative_to(self.project_root)
            dir_str = str(dir_path)
            
            # Check if this directory should have a .gitkeep
            if dir_str not in self.required_directories:
                # Skip if directory has content (other than .gitkeep)
                dir_contents = list(gitkeep_file.parent.iterdir())
                if len(dir_contents) > 1:  # More than just .gitkeep
                    continue
                
                print(f"  🗑️ Removing orphaned .gitkeep from {dir_str}")
                gitkeep_file.unlink()
                removed += 1
        
        return removed
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of .gitkeep status."""
        validation = self.validate_gitkeeps()
        
        report = ["", "📊 Enterprise .gitkeep Status Report", "=" * 50]
        
        # Summary statistics
        total = len(validation)
        present = sum(validation.values())
        missing = total - present
        
        report.extend([
            f"Total required directories: {total}",
            f"Directories with .gitkeep: {present}",
            f"Missing .gitkeep files: {missing}",
            ""
        ])
        
        # Category breakdown
        categories = {
            'Application': [d for d in self.required_directories if d.startswith('app/')],
            'Configuration': [d for d in self.required_directories if d.startswith('config/')],
            'Deployment': [d for d in self.required_directories if d.startswith('deployments/')],
            'Tools': [d for d in self.required_directories if d.startswith('tools/')],
            'External': [d for d in self.required_directories if d.startswith('external/')],
            'Logging': [d for d in self.required_directories if d.startswith('logs/')],
            'Monitoring': [d for d in self.required_directories if d in ['metrics', 'monitoring', 'telemetry']],
            'Data & Models': [d for d in self.required_directories if d.startswith(('data/', 'models/'))],
            'Testing': [d for d in self.required_directories if d.startswith('tests/')]
        }
        
        for category, directories in categories.items():
            if not directories:
                continue
                
            category_present = sum(1 for d in directories if validation[d])
            category_total = len(directories)
            
            report.append(f"📁 {category}: {category_present}/{category_total}")
            
            for directory in directories:
                status = "✅" if validation[directory] else "❌"
                report.append(f"  {status} {directory}")
            
            report.append("")
        
        return "\n".join(report)
    
    def run_maintenance(self) -> None:
        """Run complete .gitkeep maintenance."""
        print("🚀 Running .gitkeep maintenance for enterprise structure...")
        
        # Create missing .gitkeep files
        created, existing = self.create_all_gitkeeps()
        
        # Clean up orphaned files
        removed = self.cleanup_orphaned_gitkeeps()
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        print(f"📊 Maintenance Summary:")
        print(f"  ✅ Created: {created}")
        print(f"  📁 Existing: {existing}")
        print(f"  🗑️ Removed: {removed}")
        print("🎉 .gitkeep maintenance complete!")


def main():
    """Main entry point for the script."""
    manager = GitkeepManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "create":
            created, existing = manager.create_all_gitkeeps()
            print(f"\n📊 Summary: {created} created, {existing} existing")
            
        elif command == "validate":
            validation = manager.validate_gitkeeps()
            missing = [d for d, exists in validation.items() if not exists]
            
            if missing:
                print("❌ Missing .gitkeep files:")
                for directory in missing:
                    print(f"  - {directory}")
                sys.exit(1)
            else:
                print("✅ All required .gitkeep files are present!")
                
        elif command == "report":
            print(manager.generate_report())
            
        elif command == "maintenance":
            manager.run_maintenance()
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Usage: python gitkeep_manager.py [create|validate|report|maintenance]")
            sys.exit(1)
    else:
        # Default: run maintenance
        manager.run_maintenance()


if __name__ == "__main__":
    main()
