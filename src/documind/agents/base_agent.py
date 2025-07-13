"""
Base agent class for DocBridgeGuard 2.0 AI agents
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..models import ProcessingResult, Bridge

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all DocBridgeGuard AI agents
    """
    
    def __init__(
        self, 
        name: str,
        api_key: str,
        config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name/identifier
            api_key: API key for the service
            config: Agent-specific configuration
            capabilities: List of agent capabilities
        """
        self.name = name
        self.api_key = api_key
        self.config = config or {}
        self.capabilities = capabilities or []
        self.last_execution_time = None
        self.execution_count = 0
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "success_rate": 0.0,
            "average_confidence": 0.0
        }
        
        logger.info(f"Initialized {self.name} agent with capabilities: {self.capabilities}")
    
    @abstractmethod
    def process_document(
        self, 
        file_path: Union[str, Path],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a document and return results
        
        Args:
            file_path: Path to document file
            task_context: Context information for processing
            
        Returns:
            Processing results dictionary
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities
        
        Returns:
            List of capability strings
        """
        pass
    
    def can_handle_task(self, task_type: str) -> bool:
        """
        Check if agent can handle a specific task type
        
        Args:
            task_type: Type of task to check
            
        Returns:
            True if agent can handle the task
        """
        return task_type in self.capabilities
    
    def validate_inputs(self, file_path: Union[str, Path], task_context: Dict[str, Any]) -> bool:
        """
        Validate inputs before processing
        
        Args:
            file_path: Path to document file
            task_context: Task context information
            
        Returns:
            True if inputs are valid
        """
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Check file size
        max_size = self.config.get("max_file_size_mb", 100) * 1024 * 1024
        if file_path.stat().st_size > max_size:
            logger.error(f"File too large: {file_path.stat().st_size} bytes")
            return False
        
        # Check supported extensions
        supported_ext = self.config.get("supported_extensions", [".pdf", ".png", ".jpg", ".jpeg"])
        if file_path.suffix.lower() not in supported_ext:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return False
        
        return True
    
    def record_execution(self, start_time: datetime, success: bool, confidence: float = 0.0) -> None:
        """
        Record execution metrics
        
        Args:
            start_time: When execution started
            success: Whether execution was successful
            confidence: Confidence score (0-1)
        """
        self.execution_count += 1
        execution_time = (datetime.now() - start_time).total_seconds()
        self.last_execution_time = execution_time
        
        # Update performance metrics
        self.performance_metrics["total_processing_time"] += execution_time
        
        # Update success rate
        current_success_rate = self.performance_metrics["success_rate"]
        new_success_rate = (
            (current_success_rate * (self.execution_count - 1) + (1.0 if success else 0.0)) 
            / self.execution_count
        )
        self.performance_metrics["success_rate"] = new_success_rate
        
        # Update average confidence
        current_avg_confidence = self.performance_metrics["average_confidence"]
        new_avg_confidence = (
            (current_avg_confidence * (self.execution_count - 1) + confidence) 
            / self.execution_count
        )
        self.performance_metrics["average_confidence"] = new_avg_confidence
        
        logger.info(
            f"{self.name} execution #{self.execution_count}: "
            f"{execution_time:.2f}s, success={success}, confidence={confidence:.3f}"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for this agent
        
        Returns:
            Performance metrics dictionary
        """
        avg_time = (
            self.performance_metrics["total_processing_time"] / self.execution_count
            if self.execution_count > 0 else 0.0
        )
        
        return {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "average_processing_time": avg_time,
            "last_execution_time": self.last_execution_time,
            "success_rate": self.performance_metrics["success_rate"],
            "average_confidence": self.performance_metrics["average_confidence"],
            "capabilities": self.capabilities
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.execution_count = 0
        self.last_execution_time = None
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "success_rate": 0.0,
            "average_confidence": 0.0
        }
        logger.info(f"Reset metrics for {self.name} agent")
    
    def prepare_task_context(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare task context with default values
        
        Args:
            **kwargs: Additional context parameters
            
        Returns:
            Task context dictionary
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.name,
            "execution_id": f"{self.name}_{self.execution_count + 1}",
            **kwargs
        }
        
        return context
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle processing errors gracefully
        
        Args:
            error: Exception that occurred
            context: Processing context
            
        Returns:
            Error result dictionary
        """
        error_result = {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "agent": self.name,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"{self.name} agent error: {error}", exc_info=True)
        return error_result
    
    def __str__(self) -> str:
        """String representation of agent"""
        return f"{self.name}Agent(capabilities={self.capabilities}, executions={self.execution_count})"
    
    def __repr__(self) -> str:
        """Detailed representation of agent"""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"capabilities={self.capabilities}, "
            f"executions={self.execution_count}, "
            f"success_rate={self.performance_metrics['success_rate']:.3f}"
            f")"
        )