import json
import os
import logging
from typing import Dict
from flask import Flask
from flask_cors import CORS
import waitress

# Import classes and functions from other modules
from .transcriber import WhisperTranscriber
from .routes import Routes
from .validators import FileValidator
from .file_manager import temp_file_manager
from .logging_config import setup_logging
from .request_logger import RequestLogger  # New import


class WhisperServiceAPI:
    """
    Class for speech recognition API service.
    
    Attributes:
        config (Dict): Dictionary with configuration parameters.
        port (int): Port for service.
        transcriber (WhisperTranscriber): Transcriber instance.
        app (Flask): Flask application.
        file_validator (FileValidator): File validator.
    """

    def __init__(self, config_path: str):
        """
        Initialize API service.

        Args:
            config_path: Path to configuration file.
        """
        # Loading configuration
        self.config = self._load_config(config_path)

        # Configure logging
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        log_file = self.config.get("log_file", "logs/whisper_api.log")
        setup_logging(log_level=log_level, log_file=log_file)
        
        # Get logger
        self.logger = logging.getLogger('app')
        self.logger.info("Initializing API service")

        # Port for service
        self.port = self.config["service_port"]

        # Create transcriber instance
        self.transcriber = WhisperTranscriber(self.config)
        
        # Create file validator
        self.file_validator = FileValidator(self.config)

        # Determine static folder path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_folder_path = os.path.join(current_dir, 'static')
        
        # Create Flask application with explicit static path
        self.app = Flask("whisper-service", static_folder=static_folder_path)

        # Configure CORS with explicit permission for all methods, headers, and origins
        CORS(self.app)

        # Initialize request logging
        request_logging_config = self.config.get("request_logging", {})
        RequestLogger(self.app, request_logging_config)
        self.logger.info("Request logging activated")

        # Register routes
        Routes(self.app, self.transcriber, self.config, self.file_validator)

        self.logger.info(f"API service initialized, port: {self.port}")
        self.logger.info(f"Static files will be served from: {static_folder_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Loading configuration from JSON file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary with configuration parameters.

        Raises:
            FileNotFoundError: If configuration file not found.
            json.JSONDecodeError: If configuration file contains invalid JSON.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def run(self) -> None:
        """
        Start service.
        """
        self.logger.info(f"Starting service on port {self.port}")
        
        # Use waitress for production-ready server
        waitress.serve(
            self.app, 
            host='0.0.0.0', 
            port=self.port, 
            # Increase the time the server will wait for a response from the application
            # before terminating the connection due to lack of network activity.
            channel_timeout=600  # 10 minutes
        )
    
    def cleanup(self) -> None:
        """
        Cleanup resources before shutdown.
        """
        self.logger.info("Cleaning up resources before shutdown")
        temp_file_manager.cleanup_all()
