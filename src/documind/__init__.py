"""
DocuMind: AI-Powered Document Processing with Multi-Provider Intelligence

An intelligent document processing system that uses multiple AI providers
to extract, analyze, and process documents with enterprise-grade compliance.
"""

__version__ = "1.0.0"
__author__ = "DocuMind Team"
__email__ = "team@documind.ai"

from .models import ProcessingResult, ComplianceMetadata, Bridge
from .automation.intelligent_processor import IntelligentProcessor

# Main processor class for easy imports
DocumentProcessor = IntelligentProcessor

__all__ = [
    "DocumentProcessor",
    "IntelligentProcessor", 
    "ProcessingResult",
    "ComplianceMetadata",
    "Bridge",
]