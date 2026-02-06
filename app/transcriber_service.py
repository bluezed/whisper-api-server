"""
Module transcriber_service.py contains TranscriptionService class,
which is responsible for processing and transcribing audio files.
"""

import os
import uuid
import tempfile
import time
import traceback
from typing import Dict, Tuple

from .utils import logger
from .audio_utils import AudioUtils
from .history_logger import HistoryLogger
from .audio_sources import AudioSource
from .validators import FileValidator, ValidationError


class TranscriptionService:
    """
    Service for processing and transcribing audio files.
    
    Attributes:
        transcriber: Transcriber instance.
        config (Dict): Dictionary with configuration.
        max_file_size_mb (int): Maximum file size in MB.
        history (HistoryLogger): Logging object.
    """

    def __init__(self, transcriber, config: Dict):
        """
        Initialize transcription service.

        Args:
            transcriber: Transcriber instance.
            config: Dictionary with configuration.
        """
        self.transcriber = transcriber
        self.config = config
        self.max_file_size_mb = self.config.get("file_validation", {}).get("max_file_size_mb", 100)

        # Logging object
        self.history = HistoryLogger(config)

    # Method get_audio_duration removed, functionality moved to AudioUtils

    def transcribe_from_source(self, source: AudioSource, params: Dict = None, file_validator: FileValidator = None) -> Tuple[Dict, int]:
        """
        Transcribes audio file from specified source.

        Args:
            source: Audio file source.
            params: Additional transcription parameters.
            file_validator: File validator.

        Returns:
            Tuple (JSON response, HTTP code).
        """
        # Get file from source
        file, filename, error = source.get_audio_file()

        # Handle file retrieval errors
        if error:
            logger.warning(f"Error getting file from source: {error}")
            return {"error": error}, 400

        if not file:
            logger.warning("Could not get audio file from source")
            return {"error": "Failed to get audio file"}, 400
        
        # Validate file if validator provided
        if file_validator:
            try:
                file_validator.validate_file(file, filename)
            except ValidationError as e:
                # Log validation error
                logger.warning(f"File validation error '{filename}': {str(e)}")
                return {"error": str(e)}, 400

        # Extract parameters from request if present
        params = params or {}
        language = params.get('language', self.config.get('language', 'en'))
        temperature = float(params.get('temperature', 0.0))
        prompt = params.get('prompt', '')

        # Check if timestamps are requested
        return_timestamps = params.get('return_timestamps', self.config.get('return_timestamps', False))
        # Convert string value to boolean if needed
        if isinstance(return_timestamps, str):
            return_timestamps = return_timestamps.lower() in ('true', 't', 'yes', 'y', '1')

        # Temporarily modify return_timestamps setting in transcriber
        original_return_timestamps = self.transcriber.return_timestamps
        self.transcriber.return_timestamps = return_timestamps

        # Save file to temporary file
        from .file_manager import temp_file_manager
        with temp_file_manager.temp_file() as temp_file_path:
            file.save(temp_file_path)

            # Determine audio file duration
            try:
                duration = AudioUtils.get_audio_duration(temp_file_path)
            except Exception as e:
                logger.error(f"Error determining file duration: {e}")
                return {"error": f"Could not determine audio file duration: {e}"}, 500

            # For files from external sources (URL, base64), close and cleanup
            if hasattr(source, 'cleanup'):
                file.file.close()  # Close file object
                source.cleanup()  # Cleanup source temp files

            try:
                start_time = time.time()
                result = self.transcriber.process_file(temp_file_path)
                processing_time = time.time() - start_time

                # Format response depending on return_timestamps
                if return_timestamps:
                    response = {
                        "segments": result.get("segments", []),
                        "text": result.get("text", ""),
                        "processing_time": processing_time,
                        "response_size_bytes": len(str(result).encode('utf-8')),
                        "duration_seconds": duration,
                        "model": os.path.basename(self.config["model_path"])
                    }
                else:
                    # If timestamps not requested, result is a string
                    response = {
                        "text": result,
                        "processing_time": processing_time,
                        "response_size_bytes": len(str(result).encode('utf-8')),
                        "duration_seconds": duration,
                        "model": os.path.basename(self.config["model_path"])
                    }

                # Log result
                self.history.save(response, filename)

                return response, 200

            except Exception as e:
                logger.error(f"Error transcribing file '{filename}': {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"error": str(e)}, 500

            finally:
                # Restore original return_timestamps value
                self.transcriber.return_timestamps = original_return_timestamps
