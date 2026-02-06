"""
Module audio_utils.py contains utility functions for working with audio.
"""

import os
import subprocess
import wave
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger('app.audio_utils')


class AudioUtils:
    """Utility class for working with audio."""
    
    @staticmethod
    def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Loading audio file using built-in wave library.

        Args:
            file_path: Path to audio file.
            sr: Target sampling rate.

        Returns:
            Tuple (numpy array, sampling rate).
            
        Raises:
            Exception: If failed to load audio file.
        """
        try:
            # Open WAV file
            with wave.open(file_path, 'rb') as wav_file:
                # Check if it's mono audio
                if wav_file.getnchannels() != 1:
                    logger.warning(f"File {file_path} is not mono, converting to mono")
                
                # Read audio data
                frames = wav_file.readframes(-1)
                # Convert 16-bit integers to float32 in range [-1.0, 1.0]
                # 32768.0 is 2^15, max value for 16-bit signed integer
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Get sampling rate
                sampling_rate = wav_file.getframerate()
                
                # If sampling rate doesn't match target, resample
                if sampling_rate != sr:
                    from scipy.signal import resample
                    num_samples = int(len(audio_array) * sr / sampling_rate)
                    audio_array = resample(audio_array, num_samples)
                    sampling_rate = sr
                
                return audio_array, sampling_rate
                
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """
        Determines audio file duration using ffprobe.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Duration in seconds.
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                raise Exception(f"File does not exist: {file_path}")
                
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10  # Execution timeout limit
            )
            
            duration = float(result.stdout.strip())
            return duration
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout determining file duration {file_path}")
            raise Exception(f"Timeout determining file duration {file_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running ffprobe for file {file_path}: {e.stderr}")
            raise Exception(f"Error running ffprobe for file {file_path}: {e.stderr}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting duration for file {file_path}: {e}")
            raise Exception(f"Error converting duration for file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error determining file duration {file_path}: {e}")
            raise Exception(f"Unexpected error determining file duration {file_path}: {e}")
