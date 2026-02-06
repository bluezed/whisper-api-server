"""
Module audio_sources.py contains abstract class AudioSource and its concrete implementations
for handling various audio file sources (uploaded files, URLs, base64, local files).
"""

import os
import uuid
import tempfile
import base64
import requests
import abc
from typing import Dict, Tuple, Optional, BinaryIO

from .utils import logger

class AudioSource(abc.ABC):
    """Abstract class for various audio file sources.
    
    Defines interface for different audio sources and provides common
    methods for working with audio files, such as checking file size.
    """
    
    def __init__(self, max_file_size_mb: int = 100):
        """
        Initialize audio source.
        
        Args:
            max_file_size_mb: Maximum file size in MB.
        """
        self.max_file_size_mb = max_file_size_mb
        
    @abc.abstractmethod
    def get_audio_file(self) -> Tuple[Optional[BinaryIO], Optional[str], Optional[str]]:
        """
        Gets audio file from source.
        
        Returns:
            Tuple (file object, filename, error message).
            On error, returns (None, None, error message).
        """
        pass
        
    def check_file_size(self, file: BinaryIO) -> Tuple[bool, Optional[str]]:
        """
        Checks file size.
        
        Args:
            file: File object to check.
            
        Returns:
            Tuple (check result, error message).
            If check passed, error message will be None.
        """
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)  # Reset file pointer after size check
        
        if file_length > self.max_file_size_mb * 1024 * 1024:
            return False, f"File exceeds maximum size of {self.max_file_size_mb}MB"
        
        return True, None


class FakeFile:
    """Simulates file object for unifying processing from different sources.
    
    Allows processing files from various sources (local path, URL, base64)
    as standard Flask file objects, ensuring compatibility with existing 
    file processing logic.
    """
    
    def __init__(self, file: BinaryIO, filename: str):
        """
        Initialize FakeFile object.
        
        Args:
            file: Original file object or stream.
            filename: Filename for metadata.
        """
        self.file = file
        self.filename = filename

    def read(self):
        """Read file contents."""
        return self.file.read()

    def seek(self, offset: int, whence: int = 0):
        """Move read position."""
        self.file.seek(offset, whence)

    def tell(self):
        """Get current read position."""
        return self.file.tell()

    def save(self, destination: str):
        """
        Save file contents to specified destination.
        
        Args:
            destination: Path to save file.
        """
        with open(destination, 'wb') as f:
            content = self.file.read()
            f.write(content)
            self.file.seek(0)  # Reset pointer after reading

    @property
    def name(self):
        """Returns filename."""
        return self.filename


class UploadedFileSource(AudioSource):
    """Audio source for files uploaded via HTTP request."""
    
    def __init__(self, request_files, max_file_size_mb: int = 100):
        """
        Initialize source for uploaded files.
        
        Args:
            request_files: request.files object from Flask.
            max_file_size_mb: Maximum file size in MB.
        """
        super().__init__(max_file_size_mb)
        self.request_files = request_files
        
    def get_audio_file(self) -> Tuple[Optional[BinaryIO], Optional[str], Optional[str]]:
        """
        Gets audio file from uploaded files.
        
        Returns:
            Tuple (file object, filename, error message).
        """
        if 'file' not in self.request_files:
            return None, None, "No file part"
            
        file = self.request_files['file']
        
        if file.filename == '':
            return None, None, "No selected file"
            
        # Check file size
        is_valid, error_message = self.check_file_size(file)
        if not is_valid:
            return None, None, error_message
            
        return file, file.filename, None


class URLSource(AudioSource):
    """Audio source for files available via URL."""
    
    def __init__(self, url: str, max_file_size_mb: int = 100):
        """
        Initialize source for URL files.
        
        Args:
            url: URL of audio file.
            max_file_size_mb: Maximum file size in MB.
        """
        super().__init__(max_file_size_mb)
        self.url = url
        self.temp_file_path = None
        self.temp_dir = None
        
    def get_audio_file(self) -> Tuple[Optional[BinaryIO], Optional[str], Optional[str]]:
        """
        Gets audio file from URL.
        
        Returns:
            Tuple (file object, filename, error message).
        """
        try:
            # Download file from URL
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            
            # Check file size (if server provided content length)
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > self.max_file_size_mb * 1024 * 1024:
                return None, None, f"File exceeds maximum size of {self.max_file_size_mb}MB"
                
            # Save to temporary file
            self.temp_dir = tempfile.mkdtemp()
            self.temp_file_path = os.path.join(self.temp_dir, str(uuid.uuid4()) + ".wav")
            
            with open(self.temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Open file for processing
            file = open(self.temp_file_path, 'rb')
            
            # Create file object as if from request.files
            fake_file = FakeFile(file, os.path.basename(self.temp_file_path))
            
            return fake_file, fake_file.filename, None
            
        except Exception as e:
            logger.error(f"Error retrieving file from URL {self.url}: {e}")
            self.cleanup()
            return None, None, f"Error retrieving file from URL: {str(e)}"
            
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
        if self.temp_dir and os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)


class Base64Source(AudioSource):
    """Audio source for files encoded in base64."""
    
    def __init__(self, base64_data: str, max_file_size_mb: int = 100):
        """
        Initialize source for base64 files.
        
        Args:
            base64_data: Audio file data in base64 format.
            max_file_size_mb: Maximum file size in MB.
        """
        super().__init__(max_file_size_mb)
        self.base64_data = base64_data
        self.temp_file_path = None
        self.temp_dir = None
        
    def get_audio_file(self) -> Tuple[Optional[BinaryIO], Optional[str], Optional[str]]:
        """
        Gets audio file from base64 data.
        
        Returns:
            Tuple (file object, filename, error message).
        """
        try:
            # Decode base64
            audio_data = base64.b64decode(self.base64_data)
            
            # Check file size
            if len(audio_data) > self.max_file_size_mb * 1024 * 1024:
                return None, None, f"File exceeds maximum size of {self.max_file_size_mb}MB"
                
            # Save to temporary file
            self.temp_dir = tempfile.mkdtemp()
            self.temp_file_path = os.path.join(self.temp_dir, str(uuid.uuid4()) + ".wav")
            
            with open(self.temp_file_path, 'wb') as f:
                f.write(audio_data)
                
            # Open file for processing
            file = open(self.temp_file_path, 'rb')
            
            # Create file object as if from request.files
            fake_file = FakeFile(file, os.path.basename(self.temp_file_path))
            
            return fake_file, fake_file.filename, None
            
        except Exception as e:
            logger.error(f"Error decoding base64 data: {e}")
            self.cleanup()
            return None, None, f"Error decoding base64 data: {str(e)}"
            
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
        if self.temp_dir and os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)


class LocalFileSource(AudioSource):
    """Audio source for local files on server."""
    
    def __init__(self, file_path: str, max_file_size_mb: int = 100):
        """
        Initialize source for local files.
        
        Args:
            file_path: Path to local file.
            max_file_size_mb: Maximum file size in MB.
        """
        super().__init__(max_file_size_mb)
        self.file_path = file_path
        
    def get_audio_file(self) -> Tuple[Optional[BinaryIO], Optional[str], Optional[str]]:
        """
        Gets local audio file.
        
        Returns:
            Tuple (file object, filename, error message).
        """
        if not os.path.exists(self.file_path):
            return None, None, f"File not found: {self.file_path}"
            
        try:
            # Check file size
            file_size = os.path.getsize(self.file_path)
            if file_size > self.max_file_size_mb * 1024 * 1024:
                return None, None, f"File exceeds maximum size of {self.max_file_size_mb}MB"
                
            # Open file for processing
            file = open(self.file_path, 'rb')
            
            # Create file object as if from request.files
            fake_file = FakeFile(file, os.path.basename(self.file_path))
            
            return fake_file, fake_file.filename, None
            
        except Exception as e:
            logger.error(f"Error opening local file {self.file_path}: {e}")
            return None, None, f"Error opening local file: {str(e)}"
