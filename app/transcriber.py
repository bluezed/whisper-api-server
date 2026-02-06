"""
Модуль transcriber.py содержит класс WhisperTranscriber, который использует модель Whisper от 
OpenAI для транскрибации аудиофайлов в текст. Класс включает в себя методы для загрузки модели, 
обработки аудио (с использованием класса AudioProcessor), и выполнения транскрибации. 
Обрабатывает выбор устройства (CPU, CUDA, MPS) для выполнения вычислений и обеспечивает 
возможность использования Flash Attention 2 для ускорения работы модели на поддерживаемых GPU.
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
    Класс для распознавания речи с помощью модели Whisper.
    
    Attributes:
        config (Dict): Словарь с параметрами конфигурации.
        model_path (str): Model path Whisper.
        language (str): Язык распознавания.
        chunk_length_s (int): Длина аудиочанка в секундах.
        batch_size (int): Размер пакета для обработки.
        max_new_tokens (int): Максимальное количество новых токенов для генерации.
        return_timestamps (bool): Флаг возврата временных меток.
        temperature (float): Параметр температуры для генерации.
        torch_dtype (torch.dtype): Оптимальный тип данных для тензоров.
        audio_processor (AudioProcessor): Объект для обработки аудио.
        device (torch.device): Device for computations.
        model (WhisperForConditionalGeneration): Loaded model Whisper.
        processor (WhisperProcessor): Процессор для модели Whisper.
        asr_pipeline (pipeline): Пайплайн для автоматического распознавания речи.
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация транскрайбера.

        Args:
            config: Словарь с параметрами конфигурации.
        """
        self.config = config
        self.model_path = config["model_path"]
        self.language = config["language"]
        self.chunk_length_s = config["chunk_length_s"]
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]
        self.return_timestamps = config["return_timestamps"]
        self.temperature = config["temperature"]

        # Оптимальный тип для тензоров
        self.torch_dtype = torch.bfloat16

        # Создаем объект для обработки аудио
        self.audio_processor = AudioProcessor(config)

        # Определяем устройство для вычислений
        self.device = self._get_device()

        # Загружаем модель при инициализации
        self._load_model()

    def _get_device(self) -> torch.device:
        """
        Определение доступного устройства для вычислений.
        
        Returns:
            Объект устройства PyTorch.
        """
        if torch.cuda.is_available():
            # Проверяем, доступна ли GPU с индексом 1
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
        Loading model and processor.
        
        Raises:
            Exception: If model failed to load.
        """
        logger.info(f"Loading model from {self.model_path}")

        use_flash_attn = False
        if self.device.type == "cuda":
            # Checking GPU for Flash Attention support (requires Ampere architecture or newer, ie >= 8)
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
            logger.warning(f"Не удалось загрузить модель с Flash Attention: {e}")
            # Fallback к обычной версии
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
        Транскрибация аудиофайла.
        
        Args:
            audio_path: Путь к обработанному аудиофайлу.

        Returns:
            В зависимости от параметра return_timestamps:
            - Если return_timestamps=False: строка с распознанным текстом
            - Если return_timestamps=True: словарь с ключами "segments" (список словарей с ключами start_time_ms, end_time_ms, text) и "text" (полный текст)
        """
        logger.info(f"Starting transcription of file: {audio_path}")
        
        try:
            # Loading audio as numpy array
            audio_array, sampling_rate = AudioUtils.load_audio(audio_path, sr=16000)
            
            # Транскрибация с корректным форматом данных
            result = self.asr_pipeline(
                {"raw": audio_array, "sampling_rate": sampling_rate}, 
                generate_kwargs={
                    "language": self.language, 
                    "max_new_tokens": self.max_new_tokens, 
                    "temperature": self.temperature
                },
                return_timestamps=self.return_timestamps
            )
            
            # Если временные метки не запрошены, возвращаем только текст
            if not self.return_timestamps:
                transcribed_text = result.get("text", "")
                logger.info(f"Транскрибация завершена: получено {len(transcribed_text)} символов текста")
                return transcribed_text
            
            # Если временные метки запрошены, обрабатываем и форматируем результат
            segments = []
            full_text = result.get("text", "")
            
            if "chunks" in result:
                # Для новых версий модели Whisper
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
                # Для старых версий модели Whisper
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
                logger.warning("Временные метки запрошены, но не найдены в результате транскрибации")
            
            logger.info(f"Транскрибация с временными метками завершена: получено {len(segments)} сегментов")
            
            # Возвращаем словарь с сегментами и полным текстом
            return {
                "segments": segments,
                "text": full_text
            }
            
        except Exception as e:
            logger.error(f"Error during audio transcription '{audio_path}': {str(e)}")
            logger.error(f"Тип исключения: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def process_file(self, input_path: str) -> Union[str, Dict]:
        """
        Полный процесс обработки и транскрибации аудиофайла.
        
        Args:
            input_path: Путь к исходному аудиофайлу.
            
        Returns:
            В зависимости от параметра return_timestamps:
            - Если return_timestamps=False: строка с распознанным текстом
            - Если return_timestamps=True: словарь с ключами "segments" и "text"
        """
        start_time = time.time()
        logger.info(f"Начало обработки файла: {input_path}")
        
        temp_files = []
        
        try:
            # Обработка аудио (конвертация, нормализация, добавление тишины)
            processed_path, temp_files = self.audio_processor.process_audio(input_path)
            
            # Транскрибация
            result = self.transcribe(processed_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Обработка и транскрибация завершены за {elapsed_time:.2f} секунд")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error processing file '{input_path}' in {str(e)}")
            logger.error(f"Тип исключения: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
        finally:
            # Очистка временных файлов
            temp_file_manager.cleanup_temp_files(temp_files)
