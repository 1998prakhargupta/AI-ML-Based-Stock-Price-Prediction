"""
Data Module - Data Processing and Management
============================================

This module handles all data-related operations including:
- Data fetching from various sources
- Data processing and transformation
- Data validation and quality checks
- Data storage and retrieval
- Lookahead bias fixes
"""

from .fetchers import IndexDataManager
from .processors import DataProcessor
from .lookahead_bias_fixes import LookaheadBiasDetector, LookaheadBiasFixer

__all__ = [
    'IndexDataManager',
    'DataProcessor',
    'LookaheadBiasDetector',
    'LookaheadBiasFixer'
]
