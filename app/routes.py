"""
Module routes.py contains classes for registering API routes
for speech recognition service.
"""

import os
from flask import request, jsonify
from typing import Dict

from .transcriber_service import TranscriptionService
from .audio_sources import (
    UploadedFileSource,
    URLSource,
    Base64Source,
    LocalFileSource
)
from .validators import ValidationError
from .async_tasks import transcribe_audio_async, task_manager
from .cache import model_cache
from .utils import logger, log_invalid_file_request


class Routes:
    """
    Class for registering all API endpoints.
    
    Attributes:
        app (Flask): Flask application.
        config (Dict): Dictionary with configuration.
        transcription_service (TranscriptionService): Transcription service.
        file_validator (FileValidator): File validator.
    """

    def __init__(self, app, transcriber, config: Dict, file_validator):
        """
        Initialize routes.

        Args:
            app: Flask application.
            transcriber: Transcriber instance.
            config: Dictionary with configuration.
            file_validator: File validator.
        """
        self.app = app
        self.config = config
        self.transcription_service = TranscriptionService(transcriber, config)
        self.file_validator = file_validator

        # Register routes
        self._register_routes()

    def _register_routes(self) -> None:
        """
        Register all endpoints.
        """
        @self.app.route('/', methods=['GET'])
        def index():
            """Root. Returns HTML client."""
            return self.app.send_static_file('index.html')

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Endpoint for service status check."""
            return jsonify({
                "status": "ok",
                "version": self.config.get("version", "1.0.0")
            }), 200

        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Endpoint for getting service configuration."""
            return jsonify(self.config), 200

        @self.app.route('/local/transcriptions', methods=['POST'])
        def local_transcribe():
            """Endpoint for local transcription of file by path on server."""
            data = request.json

            if not data or "file_path" not in data:
                return jsonify({"error": "No file_path provided"}), 400

            file_path = data["file_path"]
            
            # Validate file path
            try:
                validated_path = self.file_validator.validate_local_file_path(
                    file_path, 
                    allowed_directories=self.config.get("allowed_directories", [])
                )
            except ValidationError as e:
                # Log API access with invalid file path
                client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
                logger.warning(f"Endpoint accessed /local/transcriptions with invalid file path '{file_path}' "
                              f"from client {client_ip}. Error: {str(e)}")
                return jsonify({"error": str(e)}), 400
            
            source = LocalFileSource(validated_path, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, data)
            return jsonify(response), status_code

        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            """Endpoint for getting list of available models."""
            return jsonify({
                "data": [
                    {
                        "id": os.path.basename(self.config["model_path"]),
                        "object": "model",
                        "owned_by": "openai",
                        "permissions": []
                    }
                ],
                "object": "list"
            }), 200

        @self.app.route('/v1/models/<model_id>', methods=['GET'])
        def retrieve_model(model_id):
            """Endpoint for getting information about specific model."""
            if model_id == os.path.basename(self.config["model_path"]):
                return jsonify({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "openai",
                    "permissions": []
                }), 200
            else:
                return jsonify({
                    "error": "Model not found",
                    "details": f"Model '{model_id}' does not exist"
                }), 404

        def _handle_transcription_request():
            """Common function for handling transcription requests."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions', methods=['POST'])
        @log_invalid_file_request
        def openai_transcribe_endpoint():
            """Endpoint for audio transcription (multipart form)."""
            return _handle_transcription_request()

        @self.app.route('/v1/audio/transcriptions/url', methods=['POST'])
        @log_invalid_file_request
        def transcribe_from_url():
            """Endpoint for transcription of audio file by URL."""
            data = request.json

            if not data or "url" not in data:
                return jsonify({
                    "error": "No URL provided",
                    "details": "Please provide 'url' in the JSON request"
                }), 400

            url = data["url"]
            # Extract transcription parameters if present
            params = {k: v for k, v in data.items() if k != "url"}

            source = URLSource(url, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/base64', methods=['POST'])
        @log_invalid_file_request
        def transcribe_from_base64():
            """Endpoint for transcription of base64-encoded audio."""
            data = request.json

            if not data or "file" not in data:
                return jsonify({
                    "error": "No base64 file provided",
                    "details": "Please provide 'file' in the JSON request"
                }), 400

            base64_data = data["file"]
            # Extract transcription parameters if present
            params = {k: v for k, v in data.items() if k != "file"}

            source = Base64Source(base64_data, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
        @log_invalid_file_request
        def transcribe_multipart():
            """Endpoint for transcription of audio file uploaded via form."""
            return _handle_transcription_request()
        
        @self.app.route('/v1/audio/transcriptions/async', methods=['POST'])
        @log_invalid_file_request
        def transcribe_async():
            """Endpoint for asynchronous audio file transcription."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            
            # Get file
            file, filename, error = source.get_audio_file()
            
            if error:
                return jsonify({"error": error}), 400
            
            if not file:
                return jsonify({"error": "Failed to get audio file"}), 400
            
            # Validate file
            try:
                self.file_validator.validate_file(file, filename)
            except ValidationError as e:
                return jsonify({"error": str(e)}), 400
            
            # Save file to temporary file
            from .file_manager import temp_file_manager
            with temp_file_manager.temp_file() as temp_path:
                file.save(temp_path)
                
                # Start async transcription
                task_id = transcribe_audio_async(temp_path, self.transcription_service.transcriber)
                
                return jsonify({"task_id": task_id}), 202
        
        @self.app.route('/v1/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """Endpoint for getting async task status."""
            task_info = task_manager.get_task_status(task_id)
            
            if not task_info:
                return jsonify({"error": "Task not found"}), 404
            
            response = {
                "task_id": task_id,
                "status": task_info["status"]
            }
            
            if task_info["status"] == "completed":
                response["result"] = task_info["result"]
            elif task_info["status"] == "failed":
                response["error"] = task_info["error"]
            
            return jsonify(response)
