"""
Module async_tasks.py contains functions for asynchronous task processing.
"""

import uuid
import time
from typing import Dict, Any, Callable, Optional
from threading import Thread
from .utils import logger


class AsyncTaskManager:
    """
    Thread-based asynchronous task manager.
    
    Attributes:
        tasks (Dict): Dictionary for storing task information.
    """
    
    def __init__(self):
        """
        Initialize asynchronous task manager.
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def run_task(self, func: Callable, *args, **kwargs) -> str:
        """
        Run task in a separate thread.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Named arguments for the function.
            
        Returns:
            Task ID.
        """
        task_id = str(uuid.uuid4())
        
        # Create task information
        self.tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None
        }
        
        # Create and start thread
        thread = Thread(target=self._run_task_thread, args=(task_id, func, args, kwargs))
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _run_task_thread(self, task_id: str, func: Callable, args: tuple, kwargs: dict) -> None:
        """
        Function to execute task in thread.
        
        Args:
            task_id: Task ID.
            func: Function to execute.
            args: Positional arguments for the function.
            kwargs: Named arguments for the function.
        """
        try:
            # Update task status
            self.tasks[task_id]["status"] = "running"
            self.tasks[task_id]["started_at"] = time.time()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Save result
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["completed_at"] = time.time()
            
            logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            # Handle error
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
            self.tasks[task_id]["completed_at"] = time.time()
            
            logger.error(f"Task {task_id} failed with error: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.
        
        Args:
            task_id: Task ID.
            
        Returns:
            Task information or None if task not found.
        """
        return self.tasks.get(task_id)
    
    def cleanup_completed_tasks(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up completed tasks older than specified age.
        
        Args:
            max_age_seconds: Maximum task age in seconds.
        """
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, task_info in self.tasks.items():
            if (task_info["status"] in ["completed", "failed"] and 
                "completed_at" in task_info and 
                current_time - task_info["completed_at"] > max_age_seconds):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            logger.debug(f"Task {task_id} removed due to age")


# Global instance of async task manager
task_manager = AsyncTaskManager()


def transcribe_audio_async(file_path: str, transcriber) -> str:
    """
    Asynchronous audio file transcription.
    
    Args:
        file_path: Path to audio file.
        transcriber: Transcriber instance.
        
    Returns:
        Task ID.
    """
    return task_manager.run_task(transcriber.process_file, file_path)
