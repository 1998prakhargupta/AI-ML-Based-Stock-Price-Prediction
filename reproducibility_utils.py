#!/usr/bin/env python3
"""
🎲 REPRODUCIBILITY UTILITIES
Comprehensive reproducibility management for the Stock Price Predictor

This module ensures all operations are reproducible while maintaining
all underlying basic logic.
"""

import numpy as np
import pandas as pd
import random
import os
import json
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class ReproducibilityManager:
    """
    Centralized reproducibility management for all project components.
    Maintains all existing logic while ensuring consistent results.
    """
    
    def __init__(self, seed: int = 42, config_file: str = "reproducibility_config.json"):
        """Initialize reproducibility manager with consistent seed"""
        self.seed = seed
        self.config_file = config_file
        self.environment_state = {}
        
        # Load or create configuration
        self._load_or_create_config()
        
        logger.info(f"🎲 ReproducibilityManager initialized with seed {self.seed}")
    
    def set_all_seeds(self) -> None:
        """
        Set seeds for all random number generators used in the project.
        This ensures reproducible results while maintaining all existing functionality.
        """
        # Python's built-in random module
        random.seed(self.seed)
        
        # NumPy random seed
        np.random.seed(self.seed)
        
        # Pandas random operations (if available)
        try:
            # For pandas operations that use numpy's random state
            np.random.seed(self.seed)
        except Exception:
            pass
        
        # Set environment variable for hash seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # TensorFlow/Keras (if used in model implementations)
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
        except ImportError:
            pass
        
        # PyTorch (if used)
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass
        
        # Scikit-learn reproducibility (will be used in model training)
        try:
            # This affects sklearn's random_state parameters
            np.random.seed(self.seed)
        except Exception:
            pass
        
        logger.info(f"✅ All random seeds set to {self.seed}")
    
    def create_reproducible_data_split(self, data: pd.DataFrame, 
                                     test_size: float = 0.2, 
                                     validation_size: float = 0.1,
                                     time_column: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create reproducible data splits while maintaining temporal order for time series.
        Preserves all existing data splitting logic.
        """
        self.set_all_seeds()
        
        if time_column and time_column in data.columns:
            # Sort by time for temporal splitting (maintains existing logic)
            data_sorted = data.sort_values(time_column).copy()
            
            n_total = len(data_sorted)
            n_test = int(n_total * test_size)
            n_val = int(n_total * validation_size)
            n_train = n_total - n_test - n_val
            
            # Split maintaining temporal order
            train_data = data_sorted.iloc[:n_train].copy()
            val_data = data_sorted.iloc[n_train:n_train + n_val].copy()
            test_data = data_sorted.iloc[n_train + n_val:].copy()
            
            logger.info(f"📊 Temporal split created: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            
        else:
            # Random split with fixed seed (maintains existing logic)
            from sklearn.model_selection import train_test_split
            
            # First split: separate test set
            train_val_data, test_data = train_test_split(
                data, test_size=test_size, random_state=self.seed, shuffle=True
            )
            
            # Second split: separate validation from training
            if validation_size > 0:
                val_ratio = validation_size / (1 - test_size)
                train_data, val_data = train_test_split(
                    train_val_data, test_size=val_ratio, random_state=self.seed, shuffle=True
                )
            else:
                train_data = train_val_data
                val_data = pd.DataFrame()
            
            logger.info(f"📊 Random split created: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'split_info': {
                'method': 'temporal' if time_column else 'random',
                'seed': self.seed,
                'test_size': test_size,
                'validation_size': validation_size,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def get_reproducible_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get model parameters with fixed random states for reproducibility.
        Maintains all existing model logic while ensuring consistency.
        """
        base_params = {'random_state': self.seed}
        
        model_specific_params = {
            'RandomForestRegressor': {
                'random_state': self.seed,
                'n_jobs': -1
            },
            'XGBRegressor': {
                'random_state': self.seed,
                'seed': self.seed
            },
            'LGBMRegressor': {
                'random_state': self.seed,
                'seed': self.seed,
                'force_row_wise': True
            },
            'SVR': {
                # SVR doesn't use random_state but we include for consistency
            },
            'LinearRegression': {
                # LinearRegression is deterministic
            }
        }
        
        params = model_specific_params.get(model_type, base_params)
        logger.info(f"🎯 Reproducible parameters for {model_type}: seed={self.seed}")
        
        return params
    
    def save_experiment_state(self, experiment_name: str, 
                            additional_info: Optional[Dict] = None) -> str:
        """
        Save complete experiment state for reproducibility documentation.
        Maintains all existing functionality while adding reproducibility tracking.
        """
        state = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'reproducibility_config': {
                'seed': self.seed,
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            },
            'environment': self._capture_environment_state(),
            'git_info': self._get_git_info(),
            'additional_info': additional_info or {}
        }
        
        filename = f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join('experiments', filename)
        
        # Create experiments directory
        os.makedirs('experiments', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"📋 Experiment state saved to {filepath}")
        return filepath
    
    def load_experiment_state(self, filepath: str) -> Dict[str, Any]:
        """Load saved experiment state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore seed if specified
        if 'reproducibility_config' in state and 'seed' in state['reproducibility_config']:
            self.seed = state['reproducibility_config']['seed']
            self.set_all_seeds()
            logger.info(f"🔄 Restored seed {self.seed} from experiment state")
        
        return state
    
    def _load_or_create_config(self) -> None:
        """Load existing config or create new one"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.seed = config.get('seed', self.seed)
                    logger.info(f"📂 Loaded reproducibility config from {self.config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config: {e}. Using default seed {self.seed}")
        else:
            self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration"""
        config = {
            'seed': self.seed,
            'created': datetime.now().isoformat(),
            'description': 'Reproducibility configuration for Stock Price Predictor'
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Saved reproducibility config to {self.config_file}")
    
    def _capture_environment_state(self) -> Dict[str, Any]:
        """Capture current environment state"""
        env_state = {
            'python_version': sys.version,
            'python_path': sys.path,
            'environment_variables': {k: v for k, v in os.environ.items() 
                                    if not k.startswith(('AWS', 'SECRET', 'KEY', 'TOKEN'))},
            'installed_packages': self._get_package_versions(),
            'working_directory': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        }
        
        return env_state
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of important packages"""
        packages = {}
        important_packages = [
            'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
            'ta', 'breeze-connect', 'joblib', 'scipy', 'plotly'
        ]
        
        for package in important_packages:
            try:
                # Handle package name variations
                module_name = package.replace('-', '_')
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
                packages[package] = version
            except ImportError:
                packages[package] = 'not_installed'
            except Exception:
                packages[package] = 'error_getting_version'
        
        return packages
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information if available"""
        git_info = {}
        
        try:
            import subprocess
            
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Get repository status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
        
        except Exception:
            git_info['error'] = 'Git information not available'
        
        return git_info


# Global instance for easy access throughout the project
global_repro_manager = ReproducibilityManager()

# Convenience functions for easy use in existing code
def set_global_seed(seed: int = 42) -> None:
    """Set global seed for all random number generators"""
    global global_repro_manager
    global_repro_manager.seed = seed
    global_repro_manager.set_all_seeds()

def get_reproducible_split(data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
    """Get reproducible data split using global manager"""
    return global_repro_manager.create_reproducible_data_split(data, **kwargs)

def get_model_params(model_type: str) -> Dict[str, Any]:
    """Get reproducible model parameters"""
    return global_repro_manager.get_reproducible_model_params(model_type)

def save_experiment(name: str, info: Optional[Dict] = None) -> str:
    """Save current experiment state"""
    return global_repro_manager.save_experiment_state(name, info)

def load_experiment(filepath: str) -> Dict[str, Any]:
    """Load experiment state"""
    return global_repro_manager.load_experiment_state(filepath)


# Initialize global reproducibility on module import
set_global_seed(42)

logger.info("🎲 Reproducibility utilities loaded and initialized")
logger.info("✅ All operations will now be reproducible with consistent seeds")
