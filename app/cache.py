"""
Module cache.py contains functions for caching data.
"""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from .utils import logger


class SimpleCache:
    """
    Simple dictionary-based cache with TTL (Time To Live) support.
    
    Attributes:
        cache (Dict): Dictionary for storing cached data.
        ttl (int): Cache time to live in seconds.
    """
    
    def __init__(self, ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl: Cache time to live in seconds (default 5 minutes).
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Key to get value.
            
        Returns:
            Cached value or None if key not found or expired.
        """
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl:
                logger.debug(f"Cache hit for key: {key}")
                return item["value"]
            else:
                # Remove expired item
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key}")
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Key to store value.
            value: Value to cache.
        """
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        logger.debug(f"Value cached for key: {key}")
    
    def clear(self) -> None:
        """
        Clear cache.
        """
        self.cache.clear()
        logger.debug("Cache cleared")
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Key to delete.
            
        Returns:
            True if key was deleted, otherwise False.
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Value deleted from cache for key: {key}")
            return True
        return False


# Global cache instances
model_cache = SimpleCache(ttl=3600)  # Cache for model metadata (1 hour)
config_cache = SimpleCache(ttl=300)   # Cache for configuration (5 minutes)


def cache_result(cache_instance: SimpleCache, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        cache_instance: Cache instance.
        key_prefix: Prefix for cache key.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key based on function name and arguments
            cache_key = f"{key_prefix}{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # Try to get result from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # If result not in cache, call function
            result = func(*args, **kwargs)
            
            # Save result to cache
            cache_instance.set(cache_key, result)
            
            return result
        return wrapper
    return decorator
