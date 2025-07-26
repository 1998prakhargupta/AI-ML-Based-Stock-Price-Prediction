"""
Logging Configuration Utilities
===============================

Centralized logging setup for the price predictor project.
"""

import logging
import logging.config
import os
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    log_dir: str = "logs"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to logging configuration file
        default_level: Default logging level
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Default logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'application.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'api_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'api_requests.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'compliance_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'compliance.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.api': {
                'handlers': ['console', 'api_file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.compliance': {
                'handlers': ['console', 'compliance_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    # Apply configuration
    if config_path and os.path.exists(config_path):
        logging.config.fileConfig(config_path)
    else:
        logging.config.dictConfig(config)
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("✅ Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper
