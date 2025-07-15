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
    MOZAMBIQUE_DPL = "mozambique_dpl"
    US_HIPAA = "us_hipaa"
    US_CCPA = "us_ccpa"
    CALIFORNIA = "california"
    SOUTH_AFRICA = "south_africa"
    NIGERIA = "nigeria"
    APAC_PDPA = "apac_pdpa"
    UK_GDPR = "uk_gdpr"
    COHERE = "cohere"
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


class AIProvider(str, Enum):
    """AI model providers"""
    OPENAI = "openai"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    GROK = "grok" 
    COHERE = "cohere"
    CLAUDE = "claude"
    AUTO = "auto"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    UNKNOWN = "unknown"


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


class RegionalCompliance(BaseModel):
    """Regional data protection compliance analysis"""
    jurisdiction: Jurisdiction = Field(..., description="Regulatory jurisdiction")
    compliance_status: ComplianceStatus = Field(..., description="Overall compliance status")
    risk_level: RiskLevel = Field(..., description="Privacy breach risk level")
    applicable_laws: List[str] = Field(default_factory=list, description="Applicable laws and regulations")
    violations: List[str] = Field(default_factory=list, description="Identified violations")
    recommendations: List[str] = Field(default_factory=list, description="Compliance recommendations")
    processing_lawful_basis: Optional[str] = Field(None, description="Lawful basis for processing")
    consent_required: bool = Field(False, description="Whether explicit consent is required")
    cross_border_restrictions: bool = Field(False, description="Cross-border transfer restrictions")
    data_subject_rights: List[str] = Field(default_factory=list, description="Applicable data subject rights")
    retention_limits: Optional[str] = Field(None, description="Data retention limits")
    breach_notification_required: bool = Field(False, description="Breach notification requirements")


class ProviderComplianceResult(BaseModel):
    """Compliance result for a specific AI provider"""
    provider: AIProvider = Field(..., description="AI provider")
    processing_result: Optional[ProcessingResult] = Field(None, description="Document processing result")
    regional_compliance: Dict[Jurisdiction, RegionalCompliance] = Field(default_factory=dict)
    overall_risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall risk score")
    data_protection_violations: List[str] = Field(default_factory=list)
    provider_available: bool = Field(True, description="Whether provider is available/configured")
    error_message: Optional[str] = Field(None, description="Error if provider unavailable")


class MultiProviderAnalysis(BaseModel):
    """Comprehensive analysis across multiple AI providers"""
    document_id: str = Field(..., description="Document identifier")
    document_filename: str = Field(..., description="Original document filename")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    selected_jurisdictions: List[Jurisdiction] = Field(..., description="Selected regulatory jurisdictions")
    provider_results: Dict[AIProvider, ProviderComplianceResult] = Field(default_factory=dict)
    cross_provider_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cross-provider comparison")
    highest_risk_provider: Optional[AIProvider] = Field(None, description="Provider with highest risk")
    safest_provider: Optional[AIProvider] = Field(None, description="Provider with lowest risk")
    compliance_summary: Dict[str, Any] = Field(default_factory=dict, description="Overall compliance summary")
    recommendations: List[str] = Field(default_factory=list, description="Overall recommendations")


class AnalyticsDashboard(BaseModel):
    """Analytics dashboard data"""
    total_documents_processed: int = Field(0, description="Total documents processed")
    risk_distribution: Dict[RiskLevel, int] = Field(default_factory=dict)
    compliance_distribution: Dict[ComplianceStatus, int] = Field(default_factory=dict)
    jurisdiction_analysis: Dict[Jurisdiction, Dict[str, Any]] = Field(default_factory=dict)
    provider_performance: Dict[AIProvider, Dict[str, float]] = Field(default_factory=dict)
    privacy_breach_incidents: int = Field(0, description="Number of privacy breach incidents detected")
    top_risk_factors: List[str] = Field(default_factory=list, description="Most common risk factors")
    processing_trends: Dict[str, List[float]] = Field(default_factory=dict, description="Processing trends over time")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class GuardrailsCheckResult(BaseModel):
    """Result of guardrails compliance check"""
    jurisdiction: Jurisdiction = Field(..., description="Jurisdiction checked")
    passed: bool = Field(..., description="Whether guardrails check passed")
    violations: List[str] = Field(default_factory=list, description="Detected violations")
    risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Risk score")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    blocked_content: List[str] = Field(default_factory=list, description="Content blocked by guardrails")
    processing_allowed: bool = Field(True, description="Whether processing should be allowed")
    
    
class DocumentComparisonMatrix(BaseModel):
    """Document comparison across multiple providers and jurisdictions"""
    document_id: str = Field(..., description="Document identifier")
    providers_tested: List[AIProvider] = Field(..., description="AI providers tested")
    jurisdictions_tested: List[Jurisdiction] = Field(..., description="Jurisdictions tested")
    compliance_matrix: Dict[str, Dict[str, ComplianceStatus]] = Field(default_factory=dict)
    risk_matrix: Dict[str, Dict[str, RiskLevel]] = Field(default_factory=dict)
    processing_success_matrix: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    best_provider_per_jurisdiction: Dict[Jurisdiction, AIProvider] = Field(default_factory=dict)
    worst_provider_per_jurisdiction: Dict[Jurisdiction, AIProvider] = Field(default_factory=dict)
    overall_safest_provider: Optional[AIProvider] = Field(None)
    analysis_summary: str = Field("", description="Human-readable analysis summary")