"""
DocuMind Enhanced: Multi-Model AI Comparison Platform

Revolutionary document processing platform that compares OpenAI, Mistral, and Claude
in real-time for cost optimization, compliance analysis, and performance benchmarking.

🔄 Multi-Model Comparison: OpenAI GPT-4 + Mistral Large + Claude 3.5 Sonnet
📊 Real-Time Analytics: Interactive dashboards with cost savings calculator
🛡️ Regional Compliance: GDPR, CCPA, NDPR, Mozambique Lei nº 3/2022
💰 Cost Optimization: 60-80% savings through intelligent model selection
📁 Smart Batch Processing: 10-document limit for cost-conscious businesses
🖥️ ChatGPT-Inspired Interface: Modern dark theme with familiar design
"""

__version__ = "2.0.0"
__author__ = "DocuMind Enhanced Team"
__email__ = "enhanced@documind.ai"

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