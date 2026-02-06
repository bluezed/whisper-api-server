"""
Module context_managers.py contains context managers for resource management.
"""

import os
import contextlib
from typing import Generator, BinaryIO
from .utils import logger


@contextlib.contextmanager
def open_file(file_path: str, mode: str = 'rb') -> Generator[BinaryIO, None, None]:
    """
    Context manager for safely opening and closing files.
    
    Args:
        file_path: Path to file.
        mode: File open mode.
        
    Yields:
        File object.
    """
    file_obj = None
    try:
        file_obj = open(file_path, mode)
        yield file_obj
    except Exception as e:
        logger.error(f"Error working with file {file_path}: {e}")
        raise
    finally:
        if file_obj:
            file_obj.close()


@contextlib.contextmanager
def audio_file(file_path: str) -> Generator[BinaryIO, None, None]:
    """
    Context manager for working with audio files.
    
    Args:
        file_path: Path to audio file.
        
    Yields:
        File object in binary mode.
    """
    with open_file(file_path, 'rb') as f:
        yield f
