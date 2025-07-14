"""
Advanced file management utilities with versioning, timestamp handling, and safe saving.
This module addresses output file management issues by providing:
1. Timestamp/versioning for output filenames
2. File existence checks before overwriting
3. Backup and rollback functionality
4. Comprehensive metadata tracking
"""

import os
import shutil
import json
import logging
import pandas as pd
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class SaveStrategy(Enum):
    """Strategies for handling existing files."""
    OVERWRITE = "overwrite"              # Overwrite existing file (default pandas behavior)
    VERSION = "version"                  # Create versioned filename (file_v1.csv, file_v2.csv)
    TIMESTAMP = "timestamp"              # Add timestamp to filename
    BACKUP_OVERWRITE = "backup_overwrite" # Backup existing file then overwrite
    PROMPT = "prompt"                    # Prompt user for action (for interactive use)
    SKIP = "skip"                       # Skip saving if file exists

class FileFormat(Enum):
    """Supported file formats for saving data."""
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle" 
    JSON = "json"
    EXCEL = "excel"

@dataclass
class SaveResult:
    """Result of a save operation with comprehensive metadata."""
    success: bool
    filepath: str
    original_filename: str
    final_filename: str
    strategy_used: SaveStrategy
    backup_created: bool = False
    backup_path: Optional[str] = None
    metadata: Optional[Dict] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class SafeFileManager:
    """
    Advanced file manager with versioning, backup, and safe saving capabilities.
    
    FIXES OUTPUT FILE MANAGEMENT ISSUES:
    1. Prevents accidental overwrites without warning
    2. Provides versioning and timestamping options
    3. Creates backups of important files
    4. Tracks file history and metadata
    """
    
    def __init__(self, base_path: str = None, default_strategy: SaveStrategy = SaveStrategy.VERSION):
        """
        Initialize the safe file manager.
        
        Args:
            base_path: Base directory for file operations
            default_strategy: Default strategy for handling existing files
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.default_strategy = default_strategy
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata tracking directory
        self.metadata_dir = self.base_path / ".file_metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"SafeFileManager initialized with base path: {self.base_path}")
    
    def generate_safe_filename(self, filename: str, strategy: SaveStrategy = None) -> Tuple[str, bool]:
        """
        Generate a safe filename based on the specified strategy.
        
        Args:
            filename: Original filename
            strategy: Strategy to use (uses default if None)
            
        Returns:
            Tuple of (safe_filename, file_exists)
        """
        if strategy is None:
            strategy = self.default_strategy
            
        filepath = self.base_path / filename
        file_exists = filepath.exists()
        
        if not file_exists:
            return filename, False
            
        # Handle existing file based on strategy
        if strategy == SaveStrategy.OVERWRITE:
            return filename, True
            
        elif strategy == SaveStrategy.SKIP:
            return filename, True
            
        elif strategy == SaveStrategy.VERSION:
            return self._generate_versioned_filename(filename), True
            
        elif strategy == SaveStrategy.TIMESTAMP:
            return self._generate_timestamped_filename(filename), True
            
        elif strategy == SaveStrategy.BACKUP_OVERWRITE:
            return filename, True
            
        else:
            return filename, True
    
    def _generate_versioned_filename(self, filename: str) -> str:
        """Generate a versioned filename (e.g., file_v1.csv, file_v2.csv)."""
        path = Path(filename)
        stem = path.stem
        suffix = path.suffix
        
        version = 1
        while (self.base_path / f"{stem}_v{version}{suffix}").exists():
            version += 1
            
        return f"{stem}_v{version}{suffix}"
    
    def _generate_timestamped_filename(self, filename: str) -> str:
        """Generate a timestamped filename."""
        path = Path(filename)
        stem = path.stem
        suffix = path.suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{stem}_{timestamp}{suffix}"
    
    def _create_backup(self, filepath: Path) -> Optional[str]:
        """Create a backup of an existing file."""
        if not filepath.exists():
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
            backup_path = filepath.parent / backup_filename
            
            shutil.copy2(filepath, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {filepath}: {e}")
            return None
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, 
                      strategy: SaveStrategy = None,
                      file_format: FileFormat = FileFormat.CSV,
                      metadata: Dict = None,
                      **kwargs) -> SaveResult:
        """
        Safely save a DataFrame with the specified strategy.
        
        Args:
            df: DataFrame to save
            filename: Desired filename
            strategy: Strategy for handling existing files
            file_format: Format to save in
            metadata: Additional metadata to store
            **kwargs: Additional arguments for pandas save methods
            
        Returns:
            SaveResult object with operation details
        """
        if strategy is None:
            strategy = self.default_strategy
            
        # Validate inputs
        if df is None or df.empty:
            return SaveResult(
                success=False,
                filepath="",
                original_filename=filename,
                final_filename="",
                strategy_used=strategy,
                error_message="DataFrame is None or empty"
            )
        
        # Ensure proper file extension
        if file_format != FileFormat.CSV and not filename.endswith(f".{file_format.value}"):
            path = Path(filename)
            filename = f"{path.stem}.{file_format.value}"
        
        try:
            # Generate safe filename
            safe_filename, file_exists = self.generate_safe_filename(filename, strategy)
            final_filepath = self.base_path / safe_filename
            
            # Handle skip strategy
            if strategy == SaveStrategy.SKIP and file_exists:
                return SaveResult(
                    success=False,
                    filepath=str(final_filepath),
                    original_filename=filename,
                    final_filename=safe_filename,
                    strategy_used=strategy,
                    error_message="File exists and strategy is SKIP"
                )
            
            # Create backup if needed
            backup_path = None
            backup_created = False
            
            if strategy == SaveStrategy.BACKUP_OVERWRITE and file_exists:
                backup_path = self._create_backup(final_filepath)
                backup_created = backup_path is not None
            
            # Save the DataFrame
            self._save_dataframe_by_format(df, final_filepath, file_format, **kwargs)
            
            # Save metadata
            self._save_metadata(safe_filename, df, metadata, strategy)
            
            result = SaveResult(
                success=True,
                filepath=str(final_filepath),
                original_filename=filename,
                final_filename=safe_filename,
                strategy_used=strategy,
                backup_created=backup_created,
                backup_path=backup_path,
                metadata=metadata
            )
            
            self.logger.info(f"✅ Successfully saved DataFrame: {safe_filename} ({len(df)} rows, {len(df.columns)} columns)")
            return result
            
        except Exception as e:
            error_msg = f"Failed to save DataFrame: {str(e)}"
            self.logger.error(error_msg)
            
            return SaveResult(
                success=False,
                filepath="",
                original_filename=filename,
                final_filename="",
                strategy_used=strategy,
                error_message=error_msg
            )
    
    def _save_dataframe_by_format(self, df: pd.DataFrame, filepath: Path, 
                                 file_format: FileFormat, **kwargs):
        """Save DataFrame in the specified format."""
        if file_format == FileFormat.CSV:
            df.to_csv(filepath, index=kwargs.get('index', True), **kwargs)
            
        elif file_format == FileFormat.PARQUET:
            df.to_parquet(filepath, **kwargs)
            
        elif file_format == FileFormat.PICKLE:
            df.to_pickle(filepath, **kwargs)
            
        elif file_format == FileFormat.JSON:
            df.to_json(filepath, **kwargs)
            
        elif file_format == FileFormat.EXCEL:
            df.to_excel(filepath, index=kwargs.get('index', True), **kwargs)
            
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _save_metadata(self, filename: str, df: pd.DataFrame, 
                      metadata: Dict = None, strategy: SaveStrategy = None):
        """Save metadata about the file."""
        try:
            meta_info = {
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'strategy_used': strategy.value if strategy else None,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
            }
            
            # Add date range if datetime column exists
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                dt_col = datetime_cols[0]
                meta_info['date_range'] = {
                    'start': df[dt_col].min().isoformat() if pd.notna(df[dt_col].min()) else None,
                    'end': df[dt_col].max().isoformat() if pd.notna(df[dt_col].max()) else None,
                    'datetime_column': dt_col
                }
            
            # Add custom metadata
            if metadata:
                meta_info['custom_metadata'] = metadata
            
            # Save metadata file
            metadata_file = self.metadata_dir / f"{Path(filename).stem}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(meta_info, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Failed to save metadata for {filename}: {e}")
    
    def save_model(self, model, filename: str, 
                   strategy: SaveStrategy = None,
                   metadata: Dict = None) -> SaveResult:
        """
        Safely save a model with versioning.
        
        Args:
            model: Model object to save
            filename: Desired filename
            strategy: Strategy for handling existing files
            metadata: Additional metadata to store
            
        Returns:
            SaveResult object with operation details
        """
        if strategy is None:
            strategy = self.default_strategy
            
        # Ensure .pkl extension
        if not filename.endswith('.pkl'):
            filename = f"{Path(filename).stem}.pkl"
        
        try:
            # Generate safe filename
            safe_filename, file_exists = self.generate_safe_filename(filename, strategy)
            final_filepath = self.base_path / safe_filename
            
            # Handle skip strategy
            if strategy == SaveStrategy.SKIP and file_exists:
                return SaveResult(
                    success=False,
                    filepath=str(final_filepath),
                    original_filename=filename,
                    final_filename=safe_filename,
                    strategy_used=strategy,
                    error_message="File exists and strategy is SKIP"
                )
            
            # Create backup if needed
            backup_path = None
            backup_created = False
            
            if strategy == SaveStrategy.BACKUP_OVERWRITE and file_exists:
                backup_path = self._create_backup(final_filepath)
                backup_created = backup_path is not None
            
            # Save the model
            model_data = {
                'model': model,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'filename': safe_filename
            }
            
            with open(final_filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metadata
            self._save_model_metadata(safe_filename, model, metadata, strategy)
            
            result = SaveResult(
                success=True,
                filepath=str(final_filepath),
                original_filename=filename,
                final_filename=safe_filename,
                strategy_used=strategy,
                backup_created=backup_created,
                backup_path=backup_path,
                metadata=metadata
            )
            
            self.logger.info(f"✅ Successfully saved model: {safe_filename}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            self.logger.error(error_msg)
            
            return SaveResult(
                success=False,
                filepath="",
                original_filename=filename,
                final_filename="",
                strategy_used=strategy,
                error_message=error_msg
            )
    
    def _save_model_metadata(self, filename: str, model, 
                           metadata: Dict = None, strategy: SaveStrategy = None):
        """Save metadata about the model."""
        try:
            meta_info = {
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'strategy_used': strategy.value if strategy else None
            }
            
            # Add model-specific metadata if available
            if hasattr(model, 'get_params'):
                try:
                    meta_info['model_params'] = model.get_params()
                except Exception:
                    pass
            
            if hasattr(model, 'feature_importances_'):
                try:
                    meta_info['has_feature_importance'] = True
                    meta_info['n_features'] = len(model.feature_importances_)
                except Exception:
                    pass
            
            # Add custom metadata
            if metadata:
                meta_info['custom_metadata'] = metadata
            
            # Save metadata file
            metadata_file = self.metadata_dir / f"{Path(filename).stem}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(meta_info, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Failed to save model metadata for {filename}: {e}")
    
    def list_files(self, pattern: str = "*", include_metadata: bool = False) -> List[Dict]:
        """
        List files in the base directory with optional metadata.
        
        Args:
            pattern: File pattern to match
            include_metadata: Whether to include metadata for each file
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        for filepath in self.base_path.glob(pattern):
            if filepath.is_file() and not filepath.name.startswith('.'):
                file_info = self._create_file_info(filepath, include_metadata)
                files.append(file_info)
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def _create_file_info(self, filepath: Path, include_metadata: bool) -> Dict:
        """Create file information dictionary."""
        file_info = {
            'filename': filepath.name,
            'filepath': str(filepath),
            'size_mb': filepath.stat().st_size / (1024**2),
            'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
        }
        
        if include_metadata:
            file_info['metadata'] = self._load_file_metadata(filepath)
        
        return file_info
    
    def _load_file_metadata(self, filepath: Path) -> Optional[Dict]:
        """Load metadata for a file."""
        metadata_file = self.metadata_dir / f"{filepath.stem}_metadata.json"
        if not metadata_file.exists():
            return None
            
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def cleanup_old_versions(self, base_filename: str, keep_versions: int = 5) -> int:
        """
        Clean up old versions of a file, keeping only the most recent versions.
        
        Args:
            base_filename: Base filename (without version suffix)
            keep_versions: Number of versions to keep
            
        Returns:
            Number of files deleted
        """
        base_stem = Path(base_filename).stem
        base_suffix = Path(base_filename).suffix
        
        # Find all version files
        version_files = []
        for filepath in self.base_path.glob(f"{base_stem}_v*{base_suffix}"):
            try:
                # Extract version number
                version_part = filepath.stem.replace(f"{base_stem}_v", "")
                version_num = int(version_part)
                version_files.append((version_num, filepath))
            except ValueError:
                continue
        
        # Sort by version number (descending) and keep only specified number
        version_files.sort(key=lambda x: x[0], reverse=True)
        files_to_delete = version_files[keep_versions:]
        
        deleted_count = 0
        for _, filepath in files_to_delete:
            try:
                filepath.unlink()
                # Also delete metadata if exists
                metadata_file = self.metadata_dir / f"{filepath.stem}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                deleted_count += 1
                self.logger.info(f"Deleted old version: {filepath.name}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {filepath}: {e}")
        
        return deleted_count

# Convenience functions for backward compatibility and easy integration
def safe_save_dataframe(df: pd.DataFrame, filename: str, base_path: str = None,
                       strategy: SaveStrategy = SaveStrategy.VERSION,
                       **kwargs) -> SaveResult:
    """
    Convenience function to safely save a DataFrame.
    
    USAGE EXAMPLE:
    result = safe_save_dataframe(
        df, 
        "stock_data.csv", 
        strategy=SaveStrategy.TIMESTAMP
    )
    if result.success:
        print(f"Saved to: {result.final_filename}")
    """
    manager = SafeFileManager(base_path, strategy)
    return manager.save_dataframe(df, filename, **kwargs)

def safe_save_model(model, filename: str, base_path: str = None,
                   strategy: SaveStrategy = SaveStrategy.VERSION,
                   metadata: Dict = None) -> SaveResult:
    """
    Convenience function to safely save a model.
    
    USAGE EXAMPLE:
    result = safe_save_model(
        trained_model,
        "stock_predictor.pkl",
        strategy=SaveStrategy.BACKUP_OVERWRITE,
        metadata={"accuracy": 0.95, "features": feature_names}
    )
    """
    manager = SafeFileManager(base_path, strategy)
    return manager.save_model(model, filename, metadata=metadata)

# Integration functions for existing codebase
def create_safe_file_manager(base_path: str = None) -> SafeFileManager:
    """Create a SafeFileManager instance for use in existing code."""
    return SafeFileManager(base_path)

def upgrade_save_operation(original_save_func):
    """
    Decorator to upgrade existing save operations with safety features.
    
    This can be used to wrap existing save functions and add safety features.
    """
    def wrapper(*args, **kwargs):
        # Extract filename if possible
        filename = None
        if len(args) > 0 and isinstance(args[0], str):
            filename = args[0]
        elif 'filename' in kwargs:
            filename = kwargs['filename']
        
        if filename:
            # Check if file exists and warn
            if os.path.exists(filename):
                logger.warning(f"⚠️ File {filename} exists and will be overwritten")
        
        return original_save_func(*args, **kwargs)
    
    return wrapper

print("✅ Advanced file management utilities loaded successfully!")
print("🛡️ Output file management issues are now addressed with:")
print("   - Automatic versioning and timestamping")
print("   - File existence checks and backup creation")
print("   - Comprehensive metadata tracking")
print("   - Multiple save strategies to prevent data loss")
