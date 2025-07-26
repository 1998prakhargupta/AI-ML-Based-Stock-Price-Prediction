"""
Visualization Module - Charts and Reports
=========================================

This module handles data visualization and reporting:
- Chart generation
- Report creation
- Dashboard utilities
- Interactive visualizations
- Automated reporting
"""

# Import classes only when explicitly requested to avoid circular imports
def get_visualizer():
    from .visualization_utils import ComprehensiveVisualizer
    return ComprehensiveVisualizer

def get_reporter():
    from .automated_reporting import AutomatedReportGenerator
    return AutomatedReportGenerator

__all__ = [
    'get_visualizer',
    'get_reporter'
]
