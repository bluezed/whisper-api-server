"""
Module audio_processor.py contains the AudioProcessor class for preprocessing audio files 
before using them in speech recognition systems. The class provides methods for converting 
audio to WAV format with 16 kHz sampling rate, normalizing volume level, 
adding silence at the beginning of recordings, and removing temporary files created during processing. 
"""

import os
import subprocess
import uuid
from typing import Dict, Tuple

from .file_manager import temp_file_manager
from .context_managers import open_file
from .utils import logger


class AudioProcessor:
    """
    Class for preprocessing audio files before recognition.
    
    Attributes:
        config (Dict): Dictionary with configuration parameters.
        norm_level (str): Audio normalization level.
        compand_params (str): Audio compressor parameters.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize audio processor.
        
        Args:
            config: Dictionary with configuration parameters.
        """
        self.config = config
        self.norm_level = config.get("norm_level", "-0.5")
        self.compand_params = config.get("compand_params", "0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2")
        self.audio_speed_factor = config.get("audio_speed_factor", 1.25)
    
    def convert_to_wav(self, input_path: str) -> str:
        """
        Convert input audio file to WAV format with 16 kHz sampling rate.
        
        Args:
            input_path: Path to source audio file.
            
        Returns:
            Path to converted WAV file.
            
        Raises:
            subprocess.CalledProcessError: If an error occurred during conversion.
        """
        audio_rate = self.config["audio_rate"]

        # Check file extension
        if input_path.lower().endswith('.wav'):
            # Check if WAV conversion is needed (e.g., if sample rate is not 16 kHz)
            try:
                info = subprocess.check_output(['soxi', input_path]).decode()
                if f'{audio_rate} Hz' in info:
                    logger.info(f"File {input_path} is already WAV with {audio_rate} Hz")
                    return input_path
            except subprocess.CalledProcessError:
                logger.warning(f"Could not get WAV file info for {input_path}")
                # Continue with conversion to ensure correct format

        # Create temporary file for WAV
        output_path, _ = temp_file_manager.create_temp_file(".wav")
        
        # Command for conversion
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", input_path,
            "-ar", f"{audio_rate}",
            "-ac", "1",  # Mono audio
            output_path
        ]
        
        logger.info(f"Converting to WAV: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"File converted to WAV: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting to WAV: {e.stderr.decode()}")
            raise
    
    def normalize_audio(self, input_path: str) -> str:
        """
        Normalize audio file using sox.
        
        Args:
            input_path: Path to WAV file.
            
        Returns:
            Path to normalized WAV file.
            
        Raises:
            subprocess.CalledProcessError: If an error occurred during normalization.
        """
        # Create temporary file for normalized audio
        output_path, _ = temp_file_manager.create_temp_file("_normalized.wav")
        
        # Command for audio normalization using sox
        cmd = [
            "sox", 
            input_path, 
            output_path, 
            "norm", self.norm_level,
            "compand"
        ] + self.compand_params.split()
        
        logger.info(f"Normalizing audio: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio normalized: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error normalizing audio: {e.stderr.decode()}")
            raise
    
    def speed_up_audio(self, input_path: str) -> str:
        """
        Speeds up audio file playback using FFmpeg.
        
        Args:
            input_path: Path to WAV file.
            
        Returns:
            Path to sped-up WAV file.
            
        Raises:
            subprocess.CalledProcessError: If an error occurred during speed-up.
        """
        # If speed-up is not required (factor = 1.0), return original file
        if float(self.audio_speed_factor) == 1.0:
            logger.info(f"Speed-up not required (factor = {self.audio_speed_factor})")
            return input_path
        
        # Create temporary file for sped-up audio
        output_path, _ = temp_file_manager.create_temp_file("_speedup.wav")
        
        # Command for audio speed-up using FFmpeg
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", input_path,
            "-filter:a", f"atempo={self.audio_speed_factor}",
            output_path
        ]
        
        logger.info(f"Speeding up audio at {self.audio_speed_factor}x: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio sped up: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error speeding up audio: {e.stderr.decode()}")
            raise
    
    def add_silence(self, input_path: str) -> str:
        """
        Adds silence at the beginning of audio file.
        
        Args:
            input_path: Path to audio file.
            
        Returns:
            Path to audio file with silence added.
            
        Raises:
            subprocess.CalledProcessError: If an error occurred during silence addition.
        """
        # Create temporary file
        output_path, _ = temp_file_manager.create_temp_file("_silence.wav")
        
        # Command for adding silence at the beginning
        cmd = [
            "sox",
            input_path,
            output_path,
            "pad", "2.0", "1.0"  # Add silence at beginning and end (seconds)
        ]
        
        logger.info(f"Adding silence: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Silence added: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error adding silence: {e.stderr.decode()}")
            raise
    
    def process_audio(self, input_path: str) -> Tuple[str, list]:
        """
        Complete audio processing: conversion, normalization, and silence addition.
        
        Args:
            input_path: Path to source audio file.
            
        Returns:
            Tuple: (path to processed file, list of temporary files for deletion)
            
        Raises:
            Exception: If an error occurred during audio processing.
        """
        temp_files = []
        
        try:
            # Convert to WAV
            wav_path = self.convert_to_wav(input_path)
            if wav_path != input_path:  # If temporary file was created
                temp_files.append(wav_path)
            
            # Normalize
            normalized_path = self.normalize_audio(wav_path)
            temp_files.append(normalized_path)
            
            # SPEED UP AUDIO (NEW STEP)
            speedup_path = self.speed_up_audio(normalized_path)
            if speedup_path != normalized_path:  # If temporary file was created
                temp_files.append(speedup_path)
            
            # Add silence
            silence_path = self.add_silence(speedup_path)
            temp_files.append(silence_path)
            
            return silence_path, temp_files
        
        except Exception as e:
            logger.error(f"Error processing audio {input_path}: {e}")
            temp_file_manager.cleanup_temp_files(temp_files)
            raise
