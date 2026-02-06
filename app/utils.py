"""
Module utils.py contains utilities and helper functions.
"""

import logging
import functools
from flask import request

# Get logger from centralized configuration
logger = logging.getLogger('app.utils')


def log_invalid_file_request(func):
    """
    Decorator for logging requests with invalid files.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Wrapped function with file validation error logging.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get request information
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
            endpoint = request.endpoint or 'unknown'
            method = request.method or 'unknown'
            
            # Get filename from request
            filename = 'unknown'
            if 'file' in request.files:
                filename = request.files['file'].filename
            elif request.is_json:
                data = request.get_json()
                if data and 'file' in data:
                    filename = data.get('filename', 'base64_data')
            
            # Log API access with invalid file
            logger.warning(f"Endpoint accessed {method} {endpoint} with invalid file '{filename}' "
                          f"from client {client_ip}. Error: {str(e)}")
            
            # Re-raise exception
            raise
    return wrapper
