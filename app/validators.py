"""
Module validators.py contains classes and functions for input data validation.
"""

import os
import magic
from typing import Dict, List, BinaryIO, Optional
import logging

# Get logger from centralized configuration
logger = logging.getLogger('app.validators')


class ValidationError(Exception):
    """Exception for validation errors."""
    pass


class FileValidator:
    """
    Class for file validation.
    
    Validates file type, size and other parameters based on configuration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize file validator.
        
        Args:
            config: Dictionary with configuration parameters.
        """
        self.validation_config = config.get("file_validation", {})
        self.max_file_size_mb = self.validation_config.get("max_file_size_mb", 100)
        self.allowed_extensions = self.validation_config.get("allowed_extensions", 
                                                             [".wav", ".mp3", ".ogg", ".flac", ".m4a"])
        self.allowed_mime_types = self.validation_config.get("allowed_mime_types", 
                                                            ["audio/wav", "audio/mpeg", "audio/ogg", 
                                                             "audio/flac", "audio/mp4"])
    
    def validate_file(self, file: BinaryIO, filename: str) -> bool:
        """
        Validates file based on configuration.
        
        Args:
            file: File object.
            filename: Filename.
            
        Returns:
            True if file passed validation.
            
        Raises:
            ValidationError: If file failed validation.
        """
        try:
            # Validate file size
            self._validate_file_size(file)
            
            # Validate file extension
            self._validate_file_extension(filename)
            
            # Validate file MIME type
            self._validate_file_mime_type(file)
            
            return True
        except ValidationError as e:
            # Log general validation error
            logger.warning(f"File validation error '{filename}': {str(e)}")
            raise
    
    def _validate_file_size(self, file: BinaryIO) -> None:
        """
        Validates file size.
        
        Args:
            file: File object.
            
        Raises:
            ValidationError: If file size exceeds maximum allowed.
        """
        # Save current position
        current_position = file.tell()
        
        # Move to end of file to determine size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        
        # Return to original position
        file.seek(current_position)
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_length > max_size_bytes:
            logger.warning(f"Attempted to upload file of size {file_length / (1024*1024):.2f} MB, "
                          f"which exceeds maximum allowed size of {self.max_file_size_mb} MB")
            
            raise ValidationError(f"File size ({file_length / (1024*1024):.2f} MB) "
                                 f"exceeds maximum allowed ({self.max_file_size_mb} MB)")
    
    def _validate_file_extension(self, filename: str) -> None:
        """
        Validates file extension.
        
        Args:
            filename: Filename.
            
        Raises:
            ValidationError: If file extension is not in allowed list.
        """
        if not any(filename.lower().endswith(ext.lower()) for ext in self.allowed_extensions):
            # Log attempt to upload file with disallowed extension
            file_extension = os.path.splitext(filename)[1]
            logger.warning(f"Attempted to upload file with disallowed extension '{file_extension}'. "
                          f"Filename: {filename}. Allowed extensions: {', '.join(self.allowed_extensions)}")
            
            raise ValidationError(f"File extension not allowed. "
                                 f"Allowed extensions: {', '.join(self.allowed_extensions)}")
    
    def _validate_file_mime_type(self, file: BinaryIO) -> None:
        """
        Validates file MIME type.
        
        Args:
            file: File object.
            
        Raises:
            ValidationError: If file MIME type is not in allowed list.
        """
        # Save current position
        current_position = file.tell()
        
        try:
            # Read first bytes to determine MIME type
            header = file.read(1024)
            mime_type = magic.from_buffer(header, mime=True)
            
            # Return to original position
            file.seek(current_position)
            
            if mime_type not in self.allowed_mime_types:
                # Log attempt to upload file with disallowed MIME type
                logger.warning(f"Attempted to upload file with disallowed MIME type '{mime_type}'. "
                              f"Allowed MIME types: {', '.join(self.allowed_mime_types)}")
                
                raise ValidationError(f"File MIME type ({mime_type}) not allowed. "
                                     f"Allowed MIME types: {', '.join(self.allowed_mime_types)}")
        except Exception as e:
            # Return to original position on error
            file.seek(current_position)
            logger.warning(f"Could not determine file MIME type: {e}")
            # Don't fail validation if MIME type couldn't be determined
    
    @staticmethod
    def validate_local_file_path(file_path: str, allowed_directories: Optional[List[str]] = None) -> str:
        """
        Validates local file path to prevent path traversal attacks.
        
        Args:
            file_path: Path to file.
            allowed_directories: List of allowed directories.
            
        Returns:
            Normalized and validated file path.
            
        Raises:
            ValidationError: If file path is unsafe.
        """
        # Normalize path
        normalized_path = os.path.normpath(file_path)
        
        # If allowed directories specified, check that path is within one of them
        if allowed_directories:
            for allowed_dir in allowed_directories:
                full_allowed_path = os.path.abspath(allowed_dir)
                full_file_path = os.path.abspath(os.path.join(full_allowed_path, normalized_path))
                
                if full_file_path.startswith(full_allowed_path):
                    return full_file_path
            
            logger.warning(f"Attempted to access file outside allowed directories: {file_path}")
            raise ValidationError("File path is not within allowed directories")
        
        # If no allowed directories specified, just return normalized path
        return normalized_path
