"""
File Management Utilities Module
===============================

Safe file management utilities for the Stock Price Predictor project.
Handles file operations with versioning, backup, and error handling.
"""

import os
import json
import pickle
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Setup logging
logger = logging.getLogger(__name__)

class SaveStrategy(Enum):
    """File save strategy options."""
    OVERWRITE = "overwrite"
    VERSION = "version"
    TIMESTAMP = "timestamp"
    BACKUP = "backup"

@dataclass
class SaveResult:
    """Result of a file save operation."""
    success: bool
    filepath: str
    original_filepath: str = ""
    backup_filepath: str = ""
    version: int = 0
    timestamp: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class SafeFileManager:
    """
    Safe file management with versioning, backup, and error handling.
    Provides robust file operations for data science workflows.
    """
    
    def __init__(self, base_path: str = "data", default_strategy: SaveStrategy = SaveStrategy.VERSION):
        """
        Initialize file manager.
        
        Args:
            base_path: Base directory for file operations
            default_strategy: Default save strategy
        """
        self.base_path = Path(base_path)
        self.default_strategy = default_strategy
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SafeFileManager initialized with base_path: {self.base_path}")
    
    def save_dataframe(self, 
                      df: pd.DataFrame,
                      filepath: str,
                      strategy: Optional[SaveStrategy] = None) -> SaveResult:
        """
        Save DataFrame with specified strategy.
        
        Args:
            df: DataFrame to save
            filepath: Target file path
            strategy: Save strategy to use
            
        Returns:
            SaveResult with operation details
        """
        strategy = strategy or self.default_strategy
        filepath = self._resolve_path(filepath)
        
        try:
            # Apply save strategy
            actual_filepath = self._apply_strategy(filepath, strategy)
            
            # Save DataFrame
            if actual_filepath.suffix.lower() == '.csv':
                df.to_csv(actual_filepath, index=False)
            elif actual_filepath.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(actual_filepath, index=False)
            else:
                df.to_csv(actual_filepath, index=False)
            
            result = SaveResult(
                success=True,
                filepath=str(actual_filepath),
                original_filepath=str(filepath)
            )
            
            logger.info(f"DataFrame saved successfully: {actual_filepath}")
            return result
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filepath}: {e}")
            return SaveResult(
                success=False,
                filepath=str(filepath),
                error_message=str(e)
            )
    
    def _resolve_path(self, filepath: str) -> Path:
        """Resolve file path relative to base path."""
        path = Path(filepath)
        if not path.is_absolute():
            path = self.base_path / path
        return path
    
    def _apply_strategy(self, filepath: Path, strategy: SaveStrategy) -> Path:
        """Apply save strategy to determine actual save path."""
        if strategy == SaveStrategy.OVERWRITE:
            actual_filepath = filepath
            
        elif strategy == SaveStrategy.VERSION:
            actual_filepath = self._get_versioned_path(filepath)
            
        elif strategy == SaveStrategy.TIMESTAMP:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = filepath.stem
            suffix = filepath.suffix
            actual_filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
            
        elif strategy == SaveStrategy.BACKUP:
            if filepath.exists():
                self._create_backup(filepath)
            actual_filepath = filepath
            
        else:
            actual_filepath = filepath
        
        # Ensure directory exists
        actual_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        return actual_filepath
    
    def _get_versioned_path(self, filepath: Path) -> Path:
        """Get next version path for a file."""
        base_name = filepath.stem
        extension = filepath.suffix
        directory = filepath.parent
        
        # Find existing versions
        version = 1
        pattern = f"{base_name}_v*{extension}"
        existing_versions = []
        
        for version_file in directory.glob(pattern):
            try:
                v_num = int(version_file.stem.split('_v')[1])
                existing_versions.append(v_num)
            except (IndexError, ValueError):
                continue
        
        if existing_versions:
            version = max(existing_versions) + 1
        
        versioned_path = directory / f"{base_name}_v{version:03d}{extension}"
        return versioned_path
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create backup of existing file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        backup_path = filepath.parent / backup_name
        
        shutil.copy2(filepath, backup_path)
        logger.info(f"Backup created: {backup_path}")
        return backup_path


# Convenience functions for common operations
def safe_save_dataframe(df: pd.DataFrame, 
                       filepath: str, 
                       base_path: str = "data",
                       strategy: SaveStrategy = SaveStrategy.VERSION) -> SaveResult:
    """
    Safely save a DataFrame with versioning.
    
    Args:
        df: DataFrame to save
        filepath: Target file path
        base_path: Base directory
        strategy: Save strategy
        
    Returns:
        SaveResult with operation details
    """
    file_manager = SafeFileManager(base_path, strategy)
    return file_manager.save_dataframe(df, filepath, strategy)


logger.info("File management utilities loaded successfully")
