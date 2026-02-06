"""
Module file_manager.py contains classes for centralized temporary file management.
Provides unified interface for creating, tracking, and cleaning up temporary files.
"""

import os
import uuid
import tempfile
import contextlib
from typing import List, Tuple, Optional, Generator
from .utils import logger


class TempFileManager:
    """
    Class for centralized temporary file management.
    
    Provides methods for creating temporary files and their subsequent cleanup.
    Uses context managers for automatic resource cleanup.
    """
    
    def __init__(self):
        """
        Initialize temporary file manager.
        """
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, suffix: str = ".wav") -> Tuple[str, str]:
        """
        Creates temporary file with unique name.
        
        Args:
            suffix: Temporary file extension.
            
        Returns:
            Tuple (file path, temp directory path).
        """
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix}")
        
        self.temp_files.append(temp_file)
        self.temp_dirs.append(temp_dir)
        
        logger.debug(f"Created temporary file: {temp_file}")
        return temp_file, temp_dir
    
    def cleanup_temp_files(self, file_paths: Optional[List[str]] = None) -> None:
        """
        Cleans up temporary files and directories.
        
        Args:
            file_paths: List of file paths to clean up. If None, cleans up all tracked files.
        """
        paths_to_clean = file_paths if file_paths is not None else self.temp_files
        
        for path in paths_to_clean:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
                    
                    # Attempt to remove directory if it's empty
                    temp_dir = os.path.dirname(path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        logger.debug(f"Removed temporary directory: {temp_dir}")
                        
                        # Remove from tracked directories list
                        if temp_dir in self.temp_dirs:
                            self.temp_dirs.remove(temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temporary file {path}: {e}")
        
        # Remove files from tracked list
        if file_paths is None:
            self.temp_files.clear()
        else:
            for path in file_paths:
                if path in self.temp_files:
                    self.temp_files.remove(path)
    
    @contextlib.contextmanager
    def temp_file(self, suffix: str = ".wav") -> Generator[str, None, None]:
        """
        Context manager for creating and automatically cleaning up temporary file.
        
        Args:
            suffix: Temporary file extension.
            
        Yields:
            Path to temporary file.
        """
        temp_file, _ = self.create_temp_file(suffix)
        try:
            yield temp_file
        finally:
            self.cleanup_temp_files([temp_file])
    
    def cleanup_all(self) -> None:
        """
        Cleans up all tracked temporary files and directories.
        """
        self.cleanup_temp_files()
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    logger.debug(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory {temp_dir}: {e}")
        self.temp_dirs.clear()


# Global instance of temporary file manager
temp_file_manager = TempFileManager()
