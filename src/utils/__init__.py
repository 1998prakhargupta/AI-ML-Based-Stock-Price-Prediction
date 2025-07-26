"""
Utils Module - Utility Functions and Classes
============================================

This module provides common utilities used across the project:
- Configuration management
- File management
- Logging utilities
- Validation helpers
- Reproducibility utilities
"""

# Import classes only when explicitly requested to avoid circular imports
def get_config():
    from .app_config import Config
    return Config

def get_file_manager():
    from .file_management_utils import SafeFileManager
    return SafeFileManager

def get_repro_manager():
    from .reproducibility_utils import ReproducibilityManager
    return ReproducibilityManager

__all__ = [
    'get_config',
    'get_file_manager', 
    'get_repro_manager'
]
