"""
Document classification system
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Union

from ..models import DocumentType

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Classifies documents by type using heuristic rules and content analysis
    """
    
    def __init__(self):
        """Initialize document classifier"""
        self.classification_rules = self._load_classification_rules()
        logger.info("Document classifier initialized")
    
    def classify(self, file_path: Union[str, Path]) -> DocumentType:
        """
        Classify document type based on filename and content
        
        Args:
            file_path: Path to document file
            
        Returns:
            Classified document type
        """
        file_path = Path(file_path)
        
        # Step 1: Filename-based classification
        filename_type = self._classify_by_filename(file_path.name)
        if filename_type != DocumentType.UNKNOWN:
            logger.info(f"Document classified by filename as: {filename_type}")
            return filename_type
        
        # Step 2: Content-based classification (if we can read text)
        try:
            # For now, we'll do a simple text extraction for classification
            # In a full implementation, this would use OCR preview
            content_type = self._classify_by_content_heuristics(file_path)
            if content_type != DocumentType.UNKNOWN:
                logger.info(f"Document classified by content as: {content_type}")
                return content_type
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
        
        # Step 3: Default classification
        logger.info(f"Document classification defaulted to: {DocumentType.UNKNOWN}")
        return DocumentType.UNKNOWN
    
    def _load_classification_rules(self) -> Dict[str, Any]:
        """Load classification rules"""
        return {
            "filename_patterns": {
                DocumentType.CONTRACT: [
                    r".*contract.*",
                    r".*agreement.*",
                    r".*terms.*",
                    r".*proposal.*",
                    r".*quote.*",
                    r".*invoice.*",
                    r".*purchase.*",
                    r".*order.*"
                ],
                DocumentType.MEDICAL: [
                    r".*medical.*",
                    r".*health.*",
                    r".*patient.*",
                    r".*clinical.*",
                    r".*diagnosis.*",
                    r".*prescription.*",
                    r".*lab.*result.*",
                    r".*discharge.*"
                ],
                DocumentType.FINANCIAL: [
                    r".*financial.*",
                    r".*bank.*",
                    r".*statement.*",
                    r".*receipt.*",
                    r".*payment.*",
                    r".*tax.*",
                    r".*audit.*",
                    r".*budget.*"
                ],
                DocumentType.RESEARCH: [
                    r".*research.*",
                    r".*study.*",
                    r".*paper.*",
                    r".*thesis.*",
                    r".*dissertation.*",
                    r".*journal.*",
                    r".*publication.*",
                    r".*experiment.*"
                ],
                DocumentType.LEGAL: [
                    r".*legal.*",
                    r".*court.*",
                    r".*lawsuit.*",
                    r".*brief.*",
                    r".*motion.*",
                    r".*judgment.*",
                    r".*subpoena.*",
                    r".*deposition.*"
                ]
            },
            "content_keywords": {
                DocumentType.CONTRACT: [
                    "whereas", "party", "parties", "agreement", "contract",
                    "terms", "conditions", "signature", "hereby", "consideration",
                    "obligations", "liability", "termination", "breach"
                ],
                DocumentType.MEDICAL: [
                    "patient", "diagnosis", "treatment", "medication", "symptoms",
                    "doctor", "physician", "medical", "health", "clinical",
                    "laboratory", "test", "result", "prescription"
                ],
                DocumentType.FINANCIAL: [
                    "amount", "payment", "invoice", "receipt", "transaction",
                    "account", "balance", "credit", "debit", "financial",
                    "bank", "currency", "tax", "revenue"
                ],
                DocumentType.RESEARCH: [
                    "abstract", "methodology", "results", "conclusion", "hypothesis",
                    "experiment", "data", "analysis", "research", "study",
                    "participants", "findings", "literature", "references"
                ],
                DocumentType.LEGAL: [
                    "plaintiff", "defendant", "court", "judge", "attorney",
                    "legal", "law", "statute", "regulation", "litigation",
                    "evidence", "testimony", "ruling", "jurisdiction"
                ]
            }
        }
    
    def _classify_by_filename(self, filename: str) -> DocumentType:
        """Classify document based on filename patterns"""
        filename_lower = filename.lower()
        
        patterns = self.classification_rules["filename_patterns"]
        
        for doc_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, filename_lower):
                    return doc_type
        
        return DocumentType.UNKNOWN
    
    def _classify_by_content_heuristics(self, file_path: Path) -> DocumentType:
        """
        Classify document based on content analysis
        This is a simplified heuristic approach
        """
        # For PDF files, we'd need to extract text first
        # For now, this is a placeholder that would integrate with OCR
        
        # Simulate content extraction (in real implementation, use OCR preview)
        content_preview = self._get_content_preview(file_path)
        
        if not content_preview:
            return DocumentType.UNKNOWN
        
        content_lower = content_preview.lower()
        
        # Score each document type based on keyword frequency
        scores = {}
        keywords = self.classification_rules["content_keywords"]
        
        for doc_type, keyword_list in keywords.items():
            score = 0
            for keyword in keyword_list:
                score += content_lower.count(keyword.lower())
            scores[doc_type] = score
        
        # Return the document type with highest score
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:  # Must have at least one keyword match
                return best_type
        
        return DocumentType.UNKNOWN
    
    def _get_content_preview(self, file_path: Path) -> str:
        """
        Get content preview for classification
        This is a simplified implementation
        """
        try:
            # For text files, read directly
            if file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(1000)  # First 1000 characters
            
            # For PDFs and images, we'd need OCR
            # For now, return empty string (would be replaced with actual OCR preview)
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to get content preview: {e}")
            return ""
    
    def get_classification_confidence(
        self, 
        file_path: Union[str, Path], 
        doc_type: DocumentType
    ) -> float:
        """
        Get confidence score for a given classification
        
        Args:
            file_path: Path to document
            doc_type: Document type to check confidence for
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        file_path = Path(file_path)
        
        # Filename confidence
        filename_confidence = self._get_filename_confidence(file_path.name, doc_type)
        
        # Content confidence (simplified)
        content_confidence = 0.5  # Placeholder
        
        # Combined confidence
        combined_confidence = (filename_confidence * 0.4 + content_confidence * 0.6)
        
        return min(1.0, max(0.0, combined_confidence))
    
    def _get_filename_confidence(self, filename: str, doc_type: DocumentType) -> float:
        """Get confidence score based on filename match"""
        filename_lower = filename.lower()
        
        if doc_type not in self.classification_rules["filename_patterns"]:
            return 0.0
        
        patterns = self.classification_rules["filename_patterns"][doc_type]
        
        for pattern in patterns:
            if re.search(pattern, filename_lower):
                return 0.8  # High confidence for filename match
        
        return 0.0