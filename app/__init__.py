"""
Application Package
==================

Main application package for the Price Predictor project.
This package contains the application layer, including core modules,
API endpoints, services, and middleware.

Author: 1998prakhargupta
"""

__version__ = "1.0.0"
__author__ = "1998prakhargupta"
__email__ = "1998prakhargupta@gmail.com"

# Application metadata
APP_NAME = "price-predictor"
APP_DESCRIPTION = "AI-ML Based Stock Price Prediction System"

# Import core modules for easy access
try:
    from .core.app_config import Config
    from .core.file_management_utils import SafeFileManager
    from .core.reproducibility_utils import ReproducibilityManager
    
    # Initialize global configuration
    config = Config()
    
except ImportError as e:
    # Handle missing dependencies gracefully
    config = None
    print(f"Warning: Could not import core modules: {e}")

__all__ = [
    "Config",
    "SafeFileManager", 
    "ReproducibilityManager",
    "config"
]
