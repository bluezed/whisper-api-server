"""
Module history_logger.py contains HistoryLogger class for logging transcription
results.
"""

import os
import json
import datetime
import random
import string
from typing import Dict, Any, Optional

from .utils import logger

class HistoryLogger:
    """Class for saving transcription history."""
    
    def __init__(self, config: Dict):
        """
        Initialize history logger.
        
        Args:
            config: Dictionary with configuration.
        """
        self.config = config
        self.history_enabled = config.get("enable_history", False)
        self.history_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history")
        
        # Create history root directory if it doesn't exist
        if self.history_enabled and not os.path.exists(self.history_root):
            os.makedirs(self.history_root)
            logger.info(f"Created transcription history directory: {self.history_root}")
    
    def save(self, result: Dict[str, Any], original_filename: str) -> Optional[str]:
        """
        Saves transcription result to history file.
        
        Args:
            result: Transcription result.
            original_filename: Original audio filename.
            
        Returns:
            Path to saved history file or None if saving is disabled.
        """
        if not self.history_enabled:
            logger.debug("Transcription history disabled in configuration")
            return None
            
        try:
            # Get current date and time
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")

            # Get current timestamp in milliseconds
            timestamp_ms = int(datetime.datetime.now().timestamp() * 1000)
            
            # Generate 4-character random tag
            random_tag = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            
            # Get only filename without path
            base_filename = os.path.basename(original_filename)
            
            # Create history filename
            history_filename = f"{timestamp_ms}_{base_filename}_{random_tag}.json"
            
            # Path to directory for current date
            date_dir = os.path.join(self.history_root, date_str)
            
            # Create directory for current date if it doesn't exist
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
                
            # Full path to history file
            history_path = os.path.join(date_dir, history_filename)
            
            # Save result to JSON file
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Transcription result saved to history: {history_path}")
            return history_path
            
        except Exception as e:
            logger.error(f"Error saving transcription history: {e}")
            return None
