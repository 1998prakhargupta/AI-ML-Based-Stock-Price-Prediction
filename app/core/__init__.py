"""
Core Application Modules
=======================

Core modules for the Price Predictor application.
Contains fundamental utilities and configuration management.

Author: 1998prakhargupta
"""

__version__ = "1.0.0"
__author__ = "1998prakhargupta"

# Import core modules
try:
    from .app_config import Config
    from .file_management_utils import SafeFileManager, SaveStrategy
    from .reproducibility_utils import ReproducibilityManager
    from .model_utils import ModelManager, ModelEvaluator
    from .visualization_utils import ComprehensiveVisualizer
    from .automated_reporting import AutomatedReportGenerator
    
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Could not import some core modules: {e}")

# Core module exports
__all__ = [
    "Config",
    "SafeFileManager", 
    "SaveStrategy",
    "ReproducibilityManager",
    "ComprehensiveVisualizer",
    "ModelManager",
    "ModelEvaluator",
    "AutomatedReportGenerator"
]
