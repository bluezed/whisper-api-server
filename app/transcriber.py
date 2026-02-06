"""
Module transcriber.py contains WhisperTranscriber class, which uses OpenAI's Whisper model 
for transcribing audio files to text. The class includes methods for loading the model, 
processing audio (using AudioProcessor class), and performing transcription. It handles 
device selection (CPU, CUDA, MPS) for computations and provides the ability to use 
Flash Attention 2 to speed up the model on supported GPUs.
"""

import time
import traceback
from typing import Dict, Tuple, Union

import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from .audio_processor import AudioProcessor
from .audio_utils import AudioUtils
from .file_manager import temp_file_manager
from .utils import logger


class WhisperTranscriber:
    """
    Class for speech recognition using Whisper model.
    
    Attributes:
        config (Dict): Dictionary with configuration parameters.
        model_path (str): Model path for Whisper.
        language (str): Recognition language.
        chunk_length_s (int): Audio chunk length in seconds.
        batch_size (int): Batch size for processing.
        max_new_tokens (int): Maximum number of new tokens for generation.
        return_timestamps (bool): Flag for returning timestamps.
        temperature (float): Temperature parameter for generation.
        torch_dtype (torch.dtype): Optimal data type for tensors.
        audio_processor (AudioProcessor): Object for audio processing.
        device (torch.device): Device for computations.
        model (WhisperForConditionalGeneration): Loaded Whisper model.
        processor (WhisperProcessor): Processor for Whisper model.
        asr_pipeline (pipeline): Pipeline for automatic speech recognition.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize transcriber.

        Args:
            config: Dictionary with configuration parameters.
        """
        self.config = config
        self.model_path = config["model_path"]
        self.language = config["language"]
        self.chunk_length_s = config["chunk_length_s"]
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]
        self.return_timestamps = config["return_timestamps"]
        self.temperature = config["temperature"]

        # Optimal type for tensors
        self.torch_dtype = torch.bfloat16

        # Create object for audio processing
        self.audio_processor = AudioProcessor(config)

        # Determine device for computations
        self.device = self._get_device()

        # Load model during initialization
        self._load_model()

    def _get_device(self) -> torch.device:
        """
        Determine available device for computations.
        
        Returns:
            PyTorch device object.
        """
        if torch.cuda.is_available():
            # Check if GPU with index 1 is available
            if torch.cuda.device_count() > 1:
                logger.info("Using CUDA GPU index 1 for computations")
                return torch.device("cuda:1")
            else:
                logger.info("Only one CUDA GPU available, using GPU index 0")
                return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) for computations")
            # MPS workaround
            setattr(torch.distributed, "is_initialized", lambda: False)
            return torch.device("mps")
        else:
            logger.info("Using CPU for computations")
            return torch.device("cpu")

    def _load_model(self) -> None:
        """
        Load model and processor.
        
        Raises:
            Exception: If model failed to load.
        """
        logger.info(f"Loading model from {self.model_path}")

        use_flash_attn = False
        if self.device.type == "cuda":
            # Check GPU for Flash Attention support (requires Ampere architecture or newer, ie >= 8)
            capability = torch.cuda.get_device_capability(self.device.index)
            if capability[0] >= 8:
                use_flash_attn = True
                logger.info(f"GPU {self.device} supports Flash Attention 2 (compute capability: {capability[0]}.{capability[1]})")
            else:
                logger.info(f"GPU {self.device} does not support Flash Attention 2 (compute capability: {capability[0]}.{capability[1]}), falling back")
        try:
            if use_flash_attn:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="flash_attention_2"
                ).to(self.device)
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(self.device)
        except Exception as e:
            logger.warning(f"Could not load model with Flash Attention: {e}")
            # Fallback to regular version
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)

        self.processor = WhisperProcessor.from_pretrained(self.model_path)

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=self.return_timestamps,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Model loaded successfully and ready for use")

    # Method _load_audio removed, functionality moved to AudioUtils

    def transcribe(self, audio_path: str) -> Union[str, Dict]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to processed audio file.

        Returns:
            Depending on return_timestamps parameter:
            - If return_timestamps=False: string with recognized text
            - If return_timestamps=True: dictionary with "segments" (list of dictionaries with start_time_ms, end_time_ms, text keys) and "text" (full text)
        """
        logger.info(f"Starting transcription of file: {audio_path}")
        
        try:
            # Loading audio as numpy array
            audio_array, sampling_rate = AudioUtils.load_audio(audio_path, sr=16000)
            
            # Transcription with correct data format
            result = self.asr_pipeline(
                {"raw": audio_array, "sampling_rate": sampling_rate}, 
                generate_kwargs={
                    "language": self.language, 
                    "max_new_tokens": self.max_new_tokens, 
                    "temperature": self.temperature
                },
                return_timestamps=self.return_timestamps
            )
            
            # If timestamps not requested, return only text
            if not self.return_timestamps:
                transcribed_text = result.get("text", "")
                logger.info(f"Transcription completed: received {len(transcribed_text)} characters of text")
                return transcribed_text
            
            # If timestamps requested, process and format result
            segments = []
            full_text = result.get("text", "")
            
            if "chunks" in result:
                # For newer Whisper model versions
                for chunk in result["chunks"]:
                    start_time = chunk.get("timestamp", [0, 0])[0]
                    end_time = chunk.get("timestamp", [0, 0])[1]
                    text = chunk.get("text", "").strip()
                    
                    segments.append({
                        "start_time_ms": int(start_time * 1000),
                        "end_time_ms": int(end_time * 1000),
                        "text": text
                    })
            elif hasattr(result, "get") and "segments" in result:
                # For older Whisper model versions
                for segment in result["segments"]:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    
                    segments.append({
                        "start_time_ms": int(start_time * 1000),
                        "end_time_ms": int(end_time * 1000),
                        "text": text
                    })
            else:
                logger.warning("Timestamps requested but not found in transcription result")
            
            logger.info(f"Transcription with timestamps completed: received {len(segments)} segments")
            
            # Return dictionary with segments and full text
            return {
                "segments": segments,
                "text": full_text
            }
            
        except Exception as e:
            logger.error(f"Error during audio transcription '{audio_path}': {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def process_file(self, input_path: str) -> Union[str, Dict]:
        """
        Complete audio file processing and transcription.
        
        Args:
            input_path: Path to source audio file.
            
        Returns:
            Depending on return_timestamps parameter:
            - If return_timestamps=False: string with recognized text
            - If return_timestamps=True: dictionary with "segments" and "text" keys
        """
        start_time = time.time()
        logger.info(f"Starting file processing: {input_path}")
        
        temp_files = []
        
        try:
            # Audio processing (conversion, normalization, silence addition)
            processed_path, temp_files = self.audio_processor.process_audio(input_path)
            
            # Transcription
            result = self.transcribe(processed_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processing and transcription completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error processing file '{input_path}' in {elapsed_time:.2f} seconds: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
        finally:
            # Cleanup temporary files
            temp_file_manager.cleanup_temp_files(temp_files)
