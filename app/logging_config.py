"""
Module logging_config.py contains centralized logging configuration.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for entire application.
    
    Args:
        log_level: Logging level (default INFO).
        log_file: Path to file for writing logs (optional).
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create improved formatter with support for additional fields
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Add type field if it doesn't exist
            if not hasattr(record, 'type'):
                record.type = 'general'
            return super().format(record)
    
    formatter = CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(type)s] %(message)s'
    )
    
    # Add console output handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file write handler if path specified
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set logging level for loggers in other modules
    logging.getLogger('app').setLevel(log_level)
    logging.getLogger('app.request').setLevel(log_level)
    
    return root_logger
