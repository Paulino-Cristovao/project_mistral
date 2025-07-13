"""
Compliance agent for automated regulatory compliance
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .base_agent import BaseAgent
from ..models import DocumentType, Jurisdiction, RedactionLevel, PrivacyImpact
from ..compliance.classifier import DocumentClassifier
from ..compliance.jurisdiction_detector import JurisdictionDetector
from ..compliance.pii_detector import PIIDetector

logger = logging.getLogger(__name__)


class ComplianceAgent(BaseAgent):
    """
    Specialized agent for regulatory compliance automation
    """
    
    def __init__(
        self,
        compliance_profile: str = "eu_gdpr",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize compliance agent
        
        Args:
            compliance_profile: Default compliance profile
            config: Agent configuration
        """
        capabilities = [
            "document_classification",
            "jurisdiction_detection",
            "pii_detection",
            "gdpr_compliance",
            "ndpr_compliance",
            "hipaa_compliance",
            "risk_assessment",
            "audit_trail_generation",
            "retention_policy_management",
            "consent_validation"
        ]
        
        super().__init__(
            name="Compliance",
            api_key="internal",  # No external API needed
            config=config or {},
            capabilities=capabilities
        )
        
        self.compliance_profile = compliance_profile
        
        # Initialize compliance components
        self.document_classifier = DocumentClassifier()
        self.jurisdiction_detector = JurisdictionDetector(self.config)
        self.pii_detector = PIIDetector(self.config)
        
        # Load compliance policies
        self.policies = self._load_compliance_policies()
        
        logger.info(f"Compliance agent initialized with profile: {compliance_profile}")
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities
    
    def process_document(
        self, 
        file_path: Union[str, Path],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform compliance analysis on document
        
        Args:
            file_path: Path to document file
            task_context: Task context including text content
            
        Returns:
            Compliance analysis results
        """
        start_time = datetime.now()
        
        try:
            file_path = Path(file_path)
            
            # Extract text from task context or use empty string
            text_content = task_context.get("extracted_text", "")
            
            # Step 1: Document classification
            doc_type = self.document_classifier.classify(file_path)
            classification_confidence = self.document_classifier.get_classification_confidence(
                file_path, doc_type
            )
            
            # Step 2: Jurisdiction detection
            jurisdiction = self.jurisdiction_detector.detect(file_path, text_content)
            jurisdiction_confidence = self.jurisdiction_detector.get_jurisdiction_confidence(
                file_path, text_content
            )
            
            # Step 3: Get compliance requirements
            compliance_requirements = self._get_compliance_requirements(doc_type, jurisdiction)
            
            # Step 4: PII detection and analysis
            redaction_level = RedactionLevel(
                task_context.get("redaction_level", 
                compliance_requirements.get("redaction_level", "moderate"))
            )
            
            if text_content:
                processed_text, pii_info = self.pii_detector.process(text_content, redaction_level)
                
                # Detect special categories (GDPR Article 9)
                special_categories = self.pii_detector.detect_special_categories(text_content)
            else:
                processed_text = ""
                pii_info = {"count": 0, "types": [], "risk_flags": []}
                special_categories = {}
            
            # Step 5: Risk assessment
            risk_assessment = self._perform_risk_assessment(
                doc_type, jurisdiction, pii_info, special_categories
            )
            
            # Step 6: Generate compliance metadata
            compliance_metadata = self._generate_compliance_metadata(
                doc_type, jurisdiction, compliance_requirements, 
                pii_info, risk_assessment
            )
            
            # Step 7: Determine legal basis and retention
            legal_basis = self._determine_legal_basis(doc_type, jurisdiction, special_categories)
            retention_policy = self._determine_retention_policy(doc_type, jurisdiction, legal_basis)
            
            # Calculate overall confidence
            confidence = (classification_confidence + jurisdiction_confidence) / 2
            
            # Record execution
            self.record_execution(start_time, True, confidence)
            
            result = {
                "document_classification": {
                    "type": doc_type.value,
                    "confidence": classification_confidence
                },
                "jurisdiction": {
                    "detected": jurisdiction.value,
                    "confidence": jurisdiction_confidence
                },
                "compliance_requirements": compliance_requirements,
                "pii_analysis": pii_info,
                "special_categories": special_categories,
                "risk_assessment": risk_assessment,
                "legal_basis": legal_basis,
                "retention_policy": retention_policy,
                "processed_text": processed_text,
                "compliance_metadata": compliance_metadata,
                "recommendations": self._generate_recommendations(
                    doc_type, jurisdiction, risk_assessment
                )
            }
            
            return {
                "success": True,
                "agent": self.name,
                "result": result,
                "confidence": confidence,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            self.record_execution(start_time, False, 0.0)
            return self.handle_error(e, task_context)
    
    def _load_compliance_policies(self) -> Dict[str, Any]:
        """Load compliance policies from configuration"""
        
        # Default policies - in production this would load from YAML files
        return {
            "eu_gdpr": {
                "contract": {
                    "redaction_level": "moderate",
                    "legal_basis": "GDPR_Art6_1b",
                    "retention_years": 7,
                    "special_categories": False,
                    "consent_required": False
                },
                "medical": {
                    "redaction_level": "strict",
                    "legal_basis": "GDPR_Art9_2h",
                    "retention_years": 10,
                    "special_categories": True,
                    "consent_required": True
                },
                "financial": {
                    "redaction_level": "strict",
                    "legal_basis": "GDPR_Art6_1c",
                    "retention_years": 10,
                    "special_categories": False,
                    "consent_required": False
                }
            },
            "africa_ndpr": {
                "contract": {
                    "redaction_level": "strict",
                    "legal_basis": "NDPR_Sec2_3",
                    "retention_years": 5,
                    "special_categories": False,
                    "consent_required": True
                },
                "medical": {
                    "redaction_level": "maximum",
                    "legal_basis": "NDPR_Sec2_3",
                    "retention_years": 7,
                    "special_categories": True,
                    "consent_required": True
                }
            },
            "us_hipaa": {
                "medical": {
                    "redaction_level": "strict",
                    "legal_basis": "HIPAA_164_506",
                    "retention_years": 6,
                    "special_categories": True,
                    "consent_required": True
                }
            }
        }
    
    def _get_compliance_requirements(
        self, 
        doc_type: DocumentType, 
        jurisdiction: Jurisdiction
    ) -> Dict[str, Any]:
        """Get compliance requirements for document type and jurisdiction"""
        
        policy_key = jurisdiction.value
        doc_key = doc_type.value
        
        if policy_key in self.policies and doc_key in self.policies[policy_key]:
            return self.policies[policy_key][doc_key]
        
        # Fallback to most restrictive requirements
        return {
            "redaction_level": "strict",
            "legal_basis": "consent_required",
            "retention_years": 5,
            "special_categories": True,
            "consent_required": True
        }
    
    def _perform_risk_assessment(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        pii_info: Dict[str, Any],
        special_categories: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        
        risk_score = 0.0
        risk_factors = []
        
        # Base risk by document type
        type_risk = {
            DocumentType.MEDICAL: 0.8,
            DocumentType.FINANCIAL: 0.7,
            DocumentType.LEGAL: 0.6,
            DocumentType.CONTRACT: 0.5,
            DocumentType.RESEARCH: 0.6,
            DocumentType.UNKNOWN: 0.4
        }
        risk_score += type_risk.get(doc_type, 0.4)
        
        # PII risk
        pii_count = pii_info.get("count", 0)
        if pii_count > 10:
            risk_score += 0.3
            risk_factors.append("high_pii_density")
        elif pii_count > 5:
            risk_score += 0.2
            risk_factors.append("moderate_pii_density")
        
        # Special categories risk (GDPR Article 9)
        if special_categories:
            risk_score += 0.4
            risk_factors.extend([f"special_category_{cat}" for cat in special_categories.keys()])
        
        # Risk flags from PII detection
        risk_flags = pii_info.get("risk_flags", [])
        if risk_flags:
            risk_score += len(risk_flags) * 0.1
            risk_factors.extend(risk_flags)
        
        # Jurisdiction-specific risks
        if jurisdiction == Jurisdiction.AFRICA_NDPR:
            risk_score += 0.1  # Higher scrutiny
            risk_factors.append("ndpr_jurisdiction")
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = PrivacyImpact.CRITICAL
        elif risk_score >= 0.6:
            risk_level = PrivacyImpact.HIGH
        elif risk_score >= 0.4:
            risk_level = PrivacyImpact.MEDIUM
        else:
            risk_level = PrivacyImpact.LOW
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level.value,
            "risk_factors": risk_factors,
            "mitigation_required": risk_score >= 0.6
        }
    
    def _determine_legal_basis(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        special_categories: Dict[str, Any]
    ) -> str:
        """Determine legal basis for processing"""
        
        # GDPR legal bases
        if jurisdiction == Jurisdiction.EU_GDPR:
            if special_categories:
                if doc_type == DocumentType.MEDICAL:
                    return "GDPR_Art9_2h"  # Healthcare
                elif doc_type == DocumentType.RESEARCH:
                    return "GDPR_Art9_2j"  # Research
                else:
                    return "GDPR_Art9_2a"  # Explicit consent
            else:
                if doc_type in [DocumentType.CONTRACT, DocumentType.LEGAL]:
                    return "GDPR_Art6_1b"  # Contract performance
                elif doc_type == DocumentType.FINANCIAL:
                    return "GDPR_Art6_1c"  # Legal obligation
                else:
                    return "GDPR_Art6_1a"  # Consent
        
        # NDPR legal bases
        elif jurisdiction == Jurisdiction.AFRICA_NDPR:
            return "NDPR_Sec2_3"  # Consent (primary basis in NDPR)
        
        # HIPAA
        elif jurisdiction == Jurisdiction.US_HIPAA:
            if doc_type == DocumentType.MEDICAL:
                return "HIPAA_164_506"  # Treatment, payment, operations
            else:
                return "HIPAA_164_508"  # Authorization required
        
        # Default
        return "consent_required"
    
    def _determine_retention_policy(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        legal_basis: str
    ) -> Dict[str, Any]:
        """Determine data retention policy"""
        
        compliance_req = self._get_compliance_requirements(doc_type, jurisdiction)
        retention_years = compliance_req.get("retention_years", 7)
        
        retention_until = datetime.now() + timedelta(days=365 * retention_years)
        
        return {
            "retention_period": f"{retention_years}_years",
            "retention_until": retention_until.isoformat(),
            "auto_delete": jurisdiction in [Jurisdiction.EU_GDPR, Jurisdiction.AFRICA_NDPR],
            "deletion_method": "secure_overwrite",
            "review_required": doc_type == DocumentType.MEDICAL
        }
    
    def _generate_compliance_metadata(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        compliance_req: Dict[str, Any],
        pii_info: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance metadata"""
        
        # Calculate compliance score
        compliance_score = 1.0
        
        # Deduct for high PII count without proper redaction
        if pii_info.get("count", 0) > 5:
            compliance_score -= 0.2
        
        # Deduct for high risk without mitigation
        if risk_assessment["risk_score"] > 0.7:
            compliance_score -= 0.3
        
        # Deduct for missing consent in high-risk scenarios
        if compliance_req.get("consent_required") and risk_assessment["risk_score"] > 0.5:
            compliance_score -= 0.2
        
        compliance_score = max(0.0, compliance_score)
        
        return {
            "compliance_score": compliance_score,
            "document_type": doc_type.value,
            "jurisdiction": jurisdiction.value,
            "redaction_level": compliance_req.get("redaction_level", "moderate"),
            "redactions_count": pii_info.get("count", 0),
            "legal_basis": compliance_req.get("legal_basis", "consent_required"),
            "processor_version": "2.0.0",
            "risk_flags": pii_info.get("risk_flags", []) + risk_assessment.get("risk_factors", []),
            "audit_reference": f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def _generate_recommendations(
        self,
        doc_type: DocumentType,
        jurisdiction: Jurisdiction,
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        # High-risk recommendations
        if risk_assessment["risk_score"] >= 0.8:
            recommendations.append("Implement additional access controls and monitoring")
            recommendations.append("Consider data minimization techniques")
            recommendations.append("Ensure explicit consent documentation")
        
        # Document-specific recommendations
        if doc_type == DocumentType.MEDICAL:
            recommendations.append("Verify healthcare professional authorization")
            recommendations.append("Implement patient consent tracking")
        
        elif doc_type == DocumentType.FINANCIAL:
            recommendations.append("Ensure PCI-DSS compliance for payment data")
            recommendations.append("Implement transaction monitoring")
        
        # Jurisdiction-specific recommendations
        if jurisdiction == Jurisdiction.EU_GDPR:
            recommendations.append("Maintain GDPR Article 30 processing records")
            recommendations.append("Implement data subject rights procedures")
        
        elif jurisdiction == Jurisdiction.AFRICA_NDPR:
            recommendations.append("Register with NDPR data protection authority")
            recommendations.append("Implement local data residency requirements")
        
        # Mitigation recommendations
        if risk_assessment.get("mitigation_required"):
            recommendations.append("Schedule compliance review within 30 days")
            recommendations.append("Consider additional encryption measures")
        
        return recommendations