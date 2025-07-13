"""
Core functionality for DocBridgeGuard 2.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import yaml
from pydantic import ValidationError

from .models import (
    ProcessingResult, ComplianceMetadata, ProcessingConfig, Bridge,
    DocumentType, Jurisdiction, RedactionLevel, PrivacyImpact,
    ProcessingStatus, AuditLogEntry
)
from .compliance.classifier import DocumentClassifier
from .compliance.jurisdiction_detector import JurisdictionDetector
from .compliance.pii_detector import PIIDetector
from .bridges.extractor import BridgeExtractor
from .utils.audit_logger import AuditLogger
from .utils.encryption import FieldEncryption


logger = logging.getLogger(__name__)


class ComplianceOCR:
    """
    Main class for compliance-first OCR processing
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        profile: str = "eu_gdpr",
        audit_level: str = "full",
        encryption: str = "field_level"
    ):
        """
        Initialize ComplianceOCR
        
        Args:
            config_path: Path to configuration file
            profile: Compliance profile to use
            audit_level: Level of audit logging
            encryption: Encryption level
        """
        self.profile = profile
        self.audit_level = audit_level
        self.encryption_level = encryption
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.classifier = DocumentClassifier()
        self.jurisdiction_detector = JurisdictionDetector(self.config)
        self.pii_detector = PIIDetector(self.config)
        self.bridge_extractor = BridgeExtractor(self.config)
        self.audit_logger = AuditLogger(audit_level)
        self.field_encryption = FieldEncryption()
        
        logger.info(f"ComplianceOCR initialized with profile: {profile}")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "compliance_policies": {
                "eu_gdpr": {
                    "contract": {
                        "redaction_level": "moderate",
                        "audit_logging": "basic",
                        "encryption": "field_level",
                        "retention_policy": "7_years"
                    },
                    "medical": {
                        "redaction_level": "strict",
                        "audit_logging": "full",
                        "encryption": "end_to_end",
                        "retention_policy": "gdpr_article_17"
                    }
                }
            },
            "language_map": {
                "en": "eu_gdpr",
                "fr": "eu_gdpr",
                "sw": "africa_ndpr"
            }
        }
    
    def process_document(
        self,
        file_path: Union[str, Path],
        provider: str = "mistral",
        consent_reference: Optional[str] = None,
        retention_policy: Optional[str] = None,
        custom_config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """
        Process a document with compliance-first approach
        
        Args:
            file_path: Path to document file
            provider: OCR provider ('openai' or 'mistral')
            consent_reference: Reference to consent record
            retention_policy: Custom retention policy
            custom_config: Custom processing configuration
            
        Returns:
            ProcessingResult with extracted text, bridges, and compliance metadata
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        file_path = Path(file_path)
        
        # Log processing start
        self.audit_logger.log_action(
            document_id=document_id,
            action="processing_started",
            details={"file_path": str(file_path), "provider": provider}
        )
        
        try:
            # Step 1: Document Classification
            doc_type = self.classifier.classify(file_path)
            logger.info(f"Document {document_id} classified as: {doc_type}")
            
            # Step 2: Jurisdiction Detection
            jurisdiction = self.jurisdiction_detector.detect(file_path)
            logger.info(f"Document {document_id} jurisdiction: {jurisdiction}")
            
            # Step 3: Get processing configuration
            processing_config = custom_config or self._get_processing_config(
                doc_type, jurisdiction, retention_policy
            )
            
            # Step 4: Risk-aware OCR processing
            from .providers.factory import OCRProviderFactory
            ocr_provider = OCRProviderFactory.create(provider)
            
            raw_text, tables = ocr_provider.extract(file_path)
            
            # Step 5: Apply guardrails (PII detection and redaction)
            processed_text, redaction_info = self.pii_detector.process(
                raw_text, processing_config.redaction_level
            )
            
            # Step 6: Bridge extraction
            bridges = []
            if processing_config.enable_bridge_extraction:
                bridges = self.bridge_extractor.extract(
                    processed_text,
                    provider=provider,
                    min_confidence=processing_config.min_confidence_threshold
                )
            
            # Step 7: Generate compliance metadata
            compliance_metadata = self._generate_compliance_metadata(
                document_id=document_id,
                doc_type=doc_type,
                jurisdiction=jurisdiction,
                redaction_info=redaction_info,
                processing_config=processing_config
            )
            
            # Step 8: Create processing result
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                document_id=document_id,
                original_filename=file_path.name,
                extracted_text=processed_text,
                raw_text=raw_text if self.audit_level == "full" else None,
                tables=tables,
                bridges=bridges,
                compliance_metadata=compliance_metadata,
                processing_config=processing_config,
                status=ProcessingStatus.COMPLETED,
                processing_time_seconds=processing_time,
                provider_used=provider
            )
            
            # Log successful completion
            self.audit_logger.log_action(
                document_id=document_id,
                action="processing_completed",
                details={
                    "processing_time": processing_time,
                    "bridges_count": len(bridges),
                    "redactions_count": redaction_info.get("count", 0)
                }
            )
            
            return result
            
        except Exception as e:
            # Log error
            self.audit_logger.log_action(
                document_id=document_id,
                action="processing_failed",
                details={"error": str(e)}
            )
            
            # Return failed result
            processing_time = time.time() - start_time
            return ProcessingResult(
                document_id=document_id,
                original_filename=file_path.name,
                extracted_text="",
                tables=[],
                bridges=[],
                compliance_metadata=ComplianceMetadata(
                    document_type=DocumentType.UNKNOWN,
                    jurisdiction=Jurisdiction.UNKNOWN,
                    redaction_level=RedactionLevel.NONE,
                    redactions_count=0,
                    legal_basis="processing_failed",
                    audit_reference=document_id,
                    processor_version="2.0.0",
                    compliance_score=0.0
                ),
                processing_config=ProcessingConfig(),
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time_seconds=processing_time,
                provider_used=provider
            )
    
    def _get_processing_config(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        retention_policy: Optional[str] = None
    ) -> ProcessingConfig:
        """Get processing configuration based on document type and jurisdiction"""
        
        # Get policy from config
        policies = self.config.get("compliance_policies", {})
        jurisdiction_policies = policies.get(jurisdiction.value, {})
        doc_type_policy = jurisdiction_policies.get(doc_type.value, {})
        
        # Create configuration
        config = ProcessingConfig(
            document_type=doc_type,
            jurisdiction=jurisdiction,
            redaction_level=RedactionLevel(
                doc_type_policy.get("redaction_level", "moderate")
            ),
            retention_policy=retention_policy or doc_type_policy.get("retention_policy", "7_years")
        )
        
        return config
    
    def _generate_compliance_metadata(
        self,
        document_id: str,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        redaction_info: Dict[str, Any],
        processing_config: ProcessingConfig
    ) -> ComplianceMetadata:
        """Generate compliance metadata for audit trail"""
        
        # Calculate retention deadline
        retention_until = None
        if processing_config.retention_policy == "7_years":
            retention_until = datetime.utcnow() + timedelta(days=365 * 7)
        elif processing_config.retention_policy == "5_years":
            retention_until = datetime.utcnow() + timedelta(days=365 * 5)
        
        # Determine legal basis
        legal_basis = "GDPR_Art6_1b"  # Contract performance
        if doc_type == DocumentType.MEDICAL:
            legal_basis = "GDPR_Art9_2h"  # Healthcare
        elif doc_type == DocumentType.RESEARCH:
            legal_basis = "GDPR_Art6_1f"  # Legitimate interest
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            doc_type, jurisdiction, redaction_info
        )
        
        return ComplianceMetadata(
            document_type=doc_type,
            jurisdiction=jurisdiction,
            redaction_level=processing_config.redaction_level,
            redactions_count=redaction_info.get("count", 0),
            legal_basis=legal_basis,
            retention_until=retention_until,
            audit_reference=f"DBG_2025_{document_id[:8]}",
            processor_version="2.0.0",
            compliance_score=compliance_score,
            risk_flags=redaction_info.get("risk_flags", [])
        )
    
    def _calculate_compliance_score(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        redaction_info: Dict[str, Any]
    ) -> float:
        """Calculate overall compliance score"""
        base_score = 0.8
        
        # Bonus for proper redaction
        if redaction_info.get("count", 0) > 0:
            base_score += 0.1
        
        # Penalty for high-risk documents without strict processing
        if doc_type == DocumentType.MEDICAL and redaction_info.get("count", 0) == 0:
            base_score -= 0.3
        
        # Jurisdiction-specific adjustments
        if jurisdiction == Jurisdiction.EU_GDPR:
            base_score += 0.05  # Bonus for GDPR compliance
        
        return min(1.0, max(0.0, base_score))


class DocumentProcessor:
    """
    Simplified interface for document processing
    """
    
    def __init__(self, compliance_ocr: Optional[ComplianceOCR] = None):
        """Initialize with optional ComplianceOCR instance"""
        self.compliance_ocr = compliance_ocr or ComplianceOCR()
    
    def process(
        self,
        file_path: Union[str, Path],
        provider: str = "mistral"
    ) -> ProcessingResult:
        """Simple document processing interface"""
        return self.compliance_ocr.process_document(file_path, provider)
    
    def compare_providers(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[ProcessingResult, ProcessingResult]:
        """Compare OpenAI and Mistral providers on same document"""
        openai_result = self.compliance_ocr.process_document(file_path, "openai")
        mistral_result = self.compliance_ocr.process_document(file_path, "mistral")
        
        return openai_result, mistral_result