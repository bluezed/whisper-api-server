"""
Module request_logger.py contains middleware for logging incoming requests and responses.
"""

import time
import json
import logging
from flask import request, g
from typing import Dict, Any, Optional


class RequestLogger:
    """
    Middleware for logging incoming requests and responses.
    """
    
    def __init__(self, app=None, config: Optional[Dict] = None):
        self.app = app
        self.config = config or {}
        self.logger = logging.getLogger('app.request')
        
        # Sensitive headers for filtering
        self.sensitive_headers = set(self.config.get(
            'sensitive_headers',
            ['authorization', 'cookie', 'set-cookie', 'proxy-authorization', 'x-api-key']
        ))
        
        # Endpoints to exclude from logging
        self.exclude_endpoints = set(self.config.get('exclude_endpoints', ['/health', '/static']))
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask application."""
        app.before_request(self._before_request)
        app.after_request(self._after_request)
    
    def _should_log_request(self) -> bool:
        """Check if current request should be logged."""
        # Check if endpoint is excluded
        path = request.path
        for excluded in self.exclude_endpoints:
            if path.startswith(excluded):
                return False
        
        return True
    
    def _before_request(self):
        """Log incoming request."""
        if not self._should_log_request():
            return
        
        g.start_time = time.time()
        
        # Determine logging mode
        debug_mode = self.config.get('log_debug', False)
        
        # Collect request information
        request_info = self._extract_request_info(debug=debug_mode)
        
        # Log depending on mode
        if debug_mode:
            self._log_debug_request(request_info)
        else:
            message = self._format_request_message(request_info)
            self.logger.info(
                message,
                extra={"type": "request"}
            )
    
    def _after_request(self, response):
        """Log response."""
        if not self._should_log_request():
            return response
        
        # Calculate processing time
        processing_time = time.time() - getattr(g, 'start_time', time.time())
        
        # Determine logging mode
        debug_mode = self.config.get('log_debug', False)
        
        # Log depending on mode
        if debug_mode:
            self._log_debug_response(response, processing_time)
        else:
            message = self._format_response_message(response, processing_time)
            self.logger.info(
                message,
                extra={"type": "response"}
            )
        
        return response
    
    def _extract_request_info(self, debug: bool = False) -> Dict[str, Any]:
        """Extract request information."""
        # Basic information
        info = {
            "endpoint": request.endpoint or str(request.url_rule),
            "method": request.method,
            "path": request.path,
            "client_ip": self._get_client_ip(),
            "user_agent": request.headers.get('User-Agent', 'Unknown')
        }
        
        # Request parameters
        if request.args:
            info["query_params"] = dict(request.args)
        
        # Form data (excluding files)
        if request.form:
            info["form_data"] = dict(request.form)
        
        # JSON data
        if request.is_json:
            try:
                info["json_data"] = request.get_json()
            except Exception:
                info["json_data"] = "Invalid JSON"
        
        # File information
        if request.files:
            file_info = {}
            for key, file in request.files.items():
                file_info[key] = {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "content_length": len(file.read()) if file else 0
                }
                file.seek(0)  # Reset file pointer
            info["files"] = file_info
        
        # Headers
        if debug:
            # In debug mode, log all headers
            headers = dict(request.headers)
        else:
            # In normal mode, filter sensitive headers
            headers = {}
            for key, value in request.headers:
                if key.lower() not in self.sensitive_headers:
                    headers[key] = value
        info["headers"] = headers
        
        return info
    
    def _log_debug_request(self, request_info: Dict[str, Any]):
        """Log full request data in debug mode."""
        debug_data = {
            "timestamp": time.time(),
            "type": "request",
            "data": request_info
        }
        self.logger.info(
            "DEBUG REQUEST: %s",
            json.dumps(debug_data, ensure_ascii=False, default=str)
        )
    
    def _log_debug_response(self, response, processing_time: float):
        """Log full response data in debug mode."""
        response_info = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content_length": response.content_length,
            "processing_time": round(processing_time, 3)
        }
        debug_data = {
            "timestamp": time.time(),
            "type": "response",
            "data": response_info
        }
        self.logger.info(
            "DEBUG RESPONSE: %s",
            json.dumps(debug_data, ensure_ascii=False, default=str)
        )
    
    def _format_request_message(self, request_info: Dict[str, Any]) -> str:
        """Format message with request details."""
        # Basic information
        method = request_info.get("method", "UNKNOWN")
        path = request_info.get("path", "/")
        client_ip = request_info.get("client_ip", "unknown")
        user_agent = request_info.get("user_agent", "Unknown")
        
        # File information
        file_info = ""
        if "files" in request_info and request_info["files"]:
            file_details = []
            for file_key, file_data in request_info["files"].items():
                filename = file_data.get("filename", "unknown")
                size = file_data.get("content_length", 0)
                file_details.append(f"{filename} ({size} bytes)")
            file_info = f" files: {', '.join(file_details)}"
        
        # Parameter information (only names for security)
        param_info = ""
        if "query_params" in request_info and request_info["query_params"]:
            param_names = list(request_info["query_params"].keys())
            param_info = f" params: {', '.join(param_names)}"
        elif "form_data" in request_info and request_info["form_data"]:
            param_names = list(request_info["form_data"].keys())
            param_info = f" params: {', '.join(param_names)}"
        elif "json_data" in request_info and isinstance(request_info["json_data"], dict):
            param_names = list(request_info["json_data"].keys())
            param_info = f" params: {', '.join(param_names)}"
        
        # Format full message
        message = f"{method} {path} from {client_ip} ({user_agent}){file_info}{param_info}"
        
        return message.strip()
    
    def _format_response_message(self, response, processing_time: float) -> str:
        """Format message with response details."""
        status_code = response.status_code
        content_length = response.content_length or 0
        processing_time_rounded = round(processing_time, 3)
        
        return f"{status_code} in {processing_time_rounded} sec, {content_length} bytes"
    
    def _get_client_ip(self) -> str:
        """Get real client IP address."""
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0]
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        else:
            return request.remote_addr or 'unknown'
