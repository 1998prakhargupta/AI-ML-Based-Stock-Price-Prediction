#!/usr/bin/env python3
"""
Project Setup Script
====================

Sets up the price predictor project environment and dependencies.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path(".venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

def install_dependencies():
    """Install project dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = Path(".venv/Scripts/pip")
    else:  # Linux/Mac
        pip_path = Path(".venv/bin/pip")
    
    if pip_path.exists():
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")
    else:
        print("⚠️  Virtual environment pip not found, using system pip")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def create_environment_file():
    """Create .env file template"""
    env_file = Path(".env")
    
    if not env_file.exists():
        env_content = """# Environment Configuration
# Copy this file to .env and fill in your actual values

# Breeze Connect API Credentials
BREEZE_API_KEY=your_api_key_here
BREEZE_API_SECRET=your_api_secret_here
BREEZE_SESSION_TOKEN=your_session_token_here

# Data Storage Configuration
DATA_BASE_PATH=./data
LOG_LEVEL=INFO

# Compliance Settings
COMPLIANCE_LEVEL=moderate
ENABLE_RATE_LIMITING=true

# Model Configuration
DEFAULT_MODEL=RandomForestRegressor
ENABLE_HYPERPARAMETER_TUNING=true
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env template created")
    else:
        print("✅ .env file already exists")

def verify_structure():
    """Verify project structure is correct"""
    required_dirs = [
        "src/api",
        "src/data", 
        "src/models",
        "src/utils",
        "src/compliance",
        "src/visualization",
        "tests/unit",
        "tests/integration",
        "configs",
        "docs",
        "scripts",
        "notebooks",
        "data/raw",
        "data/processed",
        "logs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {missing_dirs}")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print("✅ Missing directories created")
    else:
        print("✅ Project structure verified")

def setup_git_hooks():
    """Setup git hooks for pre-commit checks"""
    hooks_dir = Path(".git/hooks")
    
    if hooks_dir.exists():
        pre_commit_hook = hooks_dir / "pre-commit"
        
        if not pre_commit_hook.exists():
            hook_content = """#!/bin/bash
# Pre-commit hook for code quality checks

echo "🔍 Running pre-commit checks..."

# Run tests
python -m pytest tests/ -v

# Check code style
python -m flake8 src/ --max-line-length=120

echo "✅ Pre-commit checks passed"
"""
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(pre_commit_hook, 0o755)
            print("✅ Git pre-commit hook installed")
        else:
            print("✅ Git hooks already configured")

def main():
    """Main setup function"""
    print("🚀 Setting up Price Predictor Project")
    print("=" * 50)
    
    try:
        check_python_version()
        create_virtual_environment()
        install_dependencies()
        create_environment_file()
        verify_structure()
        setup_git_hooks()
        
        print("\n" + "=" * 50)
        print("🎉 Project setup complete!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API credentials")
        print("2. Activate virtual environment: source .venv/bin/activate")
        print("3. Run tests: python -m pytest tests/")
        print("4. Start developing: python scripts/data_pipeline.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
