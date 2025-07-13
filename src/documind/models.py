"""
Data models for DocBridgeGuard 2.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Document classification types"""
    CONTRACT = "contract"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    RESEARCH = "research"
    LEGAL = "legal"
    UNKNOWN = "unknown"


class Jurisdiction(str, Enum):
    """Regulatory jurisdictions"""
    EU_GDPR = "eu_gdpr"
    AFRICA_NDPR = "africa_ndpr"
    US_HIPAA = "us_hipaa"
    APAC_PDPA = "apac_pdpa"
    UK_GDPR = "uk_gdpr"
    UNKNOWN = "unknown"


class RedactionLevel(str, Enum):
    """PII redaction levels"""
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    STRICT = "strict"
    MAXIMUM = "maximum"


class PrivacyImpact(str, Enum):
    """Privacy impact assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class EntityType(str, Enum):
    """Entity types for bridge extraction"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PRODUCT = "product"
    CONTRACT_ID = "contract_id"
    OTHER = "other"


class Bridge(BaseModel):
    """Represents a relationship between two entities"""
    entity_1: str = Field(..., description="First entity in the relationship")
    entity_2: str = Field(..., description="Second entity in the relationship")
    entity_1_type: EntityType = Field(..., description="Type of first entity")
    entity_2_type: EntityType = Field(..., description="Type of second entity")
    relationship: str = Field(..., description="Type of relationship")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in relationship")
    privacy_impact: PrivacyImpact = Field(..., description="Privacy impact of this relationship")
    legal_basis: Optional[str] = Field(None, description="Legal basis for processing this relationship")
    source_document: Optional[str] = Field(None, description="Source document reference")
    extraction_method: str = Field(..., description="Method used to extract this bridge")


class ComplianceMetadata(BaseModel):
    """Compliance and audit metadata"""
    document_type: DocumentType = Field(..., description="Classified document type")
    jurisdiction: Jurisdiction = Field(..., description="Applicable jurisdiction")
    redaction_level: RedactionLevel = Field(..., description="Applied redaction level")
    redactions_count: int = Field(..., ge=0, description="Number of redactions performed")
    legal_basis: str = Field(..., description="Legal basis for processing")
    retention_until: Optional[datetime] = Field(None, description="Data retention deadline")
    audit_reference: str = Field(..., description="Unique audit trail reference")
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processor_version: str = Field(..., description="Version of processing system")
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Overall compliance score")
    risk_flags: List[str] = Field(default_factory=list, description="Identified risk flags")


class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    document_type: Optional[DocumentType] = None
    jurisdiction: Optional[Jurisdiction] = None
    redaction_level: RedactionLevel = RedactionLevel.MODERATE
    audit_logging: str = "basic"
    encryption: str = "field_level"
    retention_policy: str = "7_years"
    enable_bridge_extraction: bool = True
    min_confidence_threshold: float = 0.7
    max_file_size_mb: int = 100
    timeout_seconds: int = 300


class ProcessingResult(BaseModel):
    """Complete result of document processing"""
    document_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original file name")
    extracted_text: str = Field(..., description="Extracted and processed text")
    raw_text: Optional[str] = Field(None, description="Raw unprocessed text")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    bridges: List[Bridge] = Field(default_factory=list, description="Extracted entity relationships")
    compliance_metadata: ComplianceMetadata = Field(..., description="Compliance and audit data")
    processing_config: ProcessingConfig = Field(..., description="Processing configuration used")
    status: ProcessingStatus = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    processing_time_seconds: float = Field(..., ge=0.0, description="Total processing time")
    provider_used: str = Field(..., description="OCR provider used (openai/mistral)")
    automation_metadata: Optional[Dict[str, Any]] = Field(None, description="Automation processing metadata")
    quality_metadata: Optional[Dict[str, Any]] = Field(None, description="Quality assessment metadata")
    
    @validator('extracted_text')
    def validate_extracted_text(cls, v: str) -> str:
        """Validate extracted text is not empty"""
        if not v.strip():
            raise ValueError("Extracted text cannot be empty")
        return v


class ComparisonResult(BaseModel):
    """Result of comparing two OCR providers"""
    document_id: str = Field(..., description="Document identifier")
    openai_result: Optional[ProcessingResult] = Field(None, description="OpenAI processing result")
    mistral_result: Optional[ProcessingResult] = Field(None, description="Mistral processing result")
    comparison_metrics: Dict[str, float] = Field(default_factory=dict, description="Comparison metrics")
    winner: Optional[str] = Field(None, description="Better performing provider")
    confidence_in_winner: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in winner selection")
    detailed_analysis: Dict[str, Any] = Field(default_factory=dict, description="Detailed comparison analysis")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    document_id: str = Field(..., description="Document identifier")
    action: str = Field(..., description="Action performed")
    user_id: Optional[str] = Field(None, description="User identifier")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    compliance_impact: PrivacyImpact = Field(..., description="Privacy impact of action")
    retention_category: str = Field(..., description="Data retention category")