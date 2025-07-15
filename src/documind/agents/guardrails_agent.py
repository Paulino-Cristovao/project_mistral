"""
Guardrails Agent for Regional Compliance Checking
Proactive compliance monitoring and violation prevention
"""

import re
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from .base_agent import BaseAgent
from ..models import (
    GuardrailsCheckResult, Jurisdiction, DocumentType, RiskLevel,
    ComplianceStatus, RegionalCompliance, AIProvider
)
from ..compliance.mozambique_analyzer import MozambiqueComplianceAnalyzer

logger = logging.getLogger(__name__)


class GuardrailsAgent(BaseAgent):
    """
    Proactive compliance checking agent that prevents violations
    before they occur across multiple jurisdictions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            "proactive_compliance_checking",
            "multi_jurisdiction_analysis", 
            "violation_prevention",
            "real_time_monitoring",
            "cross_border_assessment",
            "privacy_impact_prediction",
            "regulatory_updates",
            "automated_blocking"
        ]
        
        super().__init__(
            name="Guardrails Compliance Agent",
            api_key="internal",
            config=config or {},
            capabilities=capabilities
        )
        
        # Load compliance policies
        self.policies = self._load_compliance_policies()
        
        # Initialize specialized analyzers
        self.mozambique_analyzer = MozambiqueComplianceAnalyzer()
        
        # Violation patterns and rules
        self.violation_patterns = self._load_violation_patterns()
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        }
        
        # Jurisdiction-specific guardrails
        self.jurisdiction_guardrails = self._initialize_jurisdiction_guardrails()
        
        logger.info("Guardrails Agent initialized with multi-jurisdiction support")
    
    def check_pre_processing_compliance(
        self,
        document_content: str,
        jurisdiction: Jurisdiction,
        ai_provider: AIProvider,
        document_type: Optional[DocumentType] = None
    ) -> GuardrailsCheckResult:
        """
        Check compliance before processing begins
        """
        try:
            violations = []
            risk_score = 0.0
            blocked_content = []
            recommendations = []
            
            # Jurisdiction-specific checks
            if jurisdiction == Jurisdiction.MOZAMBIQUE_DPL:
                result = self._check_mozambique_guardrails(
                    document_content, ai_provider, document_type
                )
                violations.extend(result["violations"])
                risk_score = max(risk_score, result["risk_score"])
                blocked_content.extend(result["blocked_content"])
                recommendations.extend(result["recommendations"])
                
            elif jurisdiction == Jurisdiction.EU_GDPR:
                result = self._check_gdpr_guardrails(
                    document_content, ai_provider, document_type
                )
                violations.extend(result["violations"])
                risk_score = max(risk_score, result["risk_score"])
                blocked_content.extend(result["blocked_content"])
                recommendations.extend(result["recommendations"])
                
            elif jurisdiction == Jurisdiction.SOUTH_AFRICA:
                result = self._check_popia_guardrails(
                    document_content, ai_provider, document_type
                )
                violations.extend(result["violations"])
                risk_score = max(risk_score, result["risk_score"])
                blocked_content.extend(result["blocked_content"])
                recommendations.extend(result["recommendations"])
                
            elif jurisdiction == Jurisdiction.NIGERIA:
                result = self._check_ndpr_guardrails(
                    document_content, ai_provider, document_type
                )
                violations.extend(result["violations"])
                risk_score = max(risk_score, result["risk_score"])
                blocked_content.extend(result["blocked_content"])
                recommendations.extend(result["recommendations"])
                
            elif jurisdiction == Jurisdiction.CALIFORNIA:
                result = self._check_ccpa_guardrails(
                    document_content, ai_provider, document_type
                )
                violations.extend(result["violations"])
                risk_score = max(risk_score, result["risk_score"])
                blocked_content.extend(result["blocked_content"])
                recommendations.extend(result["recommendations"])
            
            # AI Provider-specific checks
            provider_result = self._check_provider_guardrails(
                document_content, ai_provider, jurisdiction
            )
            violations.extend(provider_result["violations"])
            risk_score = max(risk_score, provider_result["risk_score"])
            recommendations.extend(provider_result["recommendations"])
            
            # Determine if processing should be allowed
            processing_allowed = risk_score < self.risk_thresholds["critical"] and len(violations) == 0
            
            return GuardrailsCheckResult(
                jurisdiction=jurisdiction,
                passed=len(violations) == 0,
                violations=violations,
                risk_score=risk_score,
                recommendations=recommendations,
                blocked_content=blocked_content,
                processing_allowed=processing_allowed
            )
            
        except Exception as e:
            logger.error(f"Guardrails check failed: {e}")
            return GuardrailsCheckResult(
                jurisdiction=jurisdiction,
                passed=False,
                violations=[f"Guardrails check failed: {str(e)}"],
                risk_score=1.0,
                recommendations=["Manual review required due to guardrails failure"],
                blocked_content=[],
                processing_allowed=False
            )
    
    def _check_mozambique_guardrails(
        self,
        content: str,
        provider: AIProvider,
        doc_type: Optional[DocumentType]
    ) -> Dict[str, Any]:
        """Mozambique-specific guardrails"""
        
        violations = []
        blocked_content = []
        recommendations = []
        risk_score = 0.0
        
        # Check for Mozambican PII
        mozambican_pii = self._detect_mozambican_pii(content)
        if mozambican_pii["found"]:
            risk_score = max(risk_score, 0.7)
            
            # Check consent documentation
            if not self._check_consent_indicators(content):
                violations.append("MDPL Art12: No consent indicators found for Mozambican personal data")
                risk_score = max(risk_score, 0.8)
                recommendations.append("Document explicit consent before processing")
            
            # Check data localization requirements
            if provider in [AIProvider.OPENAI, AIProvider.CLAUDE]:  # US-based providers
                violations.append("MDPL Art42: Cross-border transfer to non-adequate jurisdiction")
                risk_score = max(risk_score, 0.9)
                recommendations.append("Consider data localization within Mozambique")
                blocked_content.extend(mozambican_pii["content"])
        
        # Check for special categories
        special_categories = self._detect_special_categories_mz(content)
        if special_categories["found"]:
            risk_score = max(risk_score, 0.8)
            
            if not self._check_explicit_consent_indicators(content):
                violations.append("MDPL Art15: Special category data requires explicit consent")
                risk_score = max(risk_score, 0.9)
                recommendations.append("Obtain explicit consent for special category processing")
        
        # Check medical data specific requirements
        if doc_type == DocumentType.MEDICAL:
            if not self._check_medical_consent_mozambique(content):
                violations.append("MDPL Medical: Healthcare data requires specific consent")
                risk_score = max(risk_score, 0.85)
                recommendations.append("Ensure healthcare-specific consent mechanisms")
        
        return {
            "violations": violations,
            "risk_score": risk_score,
            "blocked_content": blocked_content,
            "recommendations": recommendations
        }
    
    def _check_gdpr_guardrails(
        self,
        content: str,
        provider: AIProvider,
        doc_type: Optional[DocumentType]
    ) -> Dict[str, Any]:
        """GDPR-specific guardrails"""
        
        violations = []
        blocked_content = []
        recommendations = []
        risk_score = 0.0
        
        # Check for EU personal data
        eu_pii = self._detect_eu_pii(content)
        if eu_pii["found"]:
            risk_score = max(risk_score, 0.6)
            
            # Check lawful basis indicators
            if not self._check_lawful_basis_indicators(content):
                violations.append("GDPR Art6: No lawful basis identified for processing")
                risk_score = max(risk_score, 0.7)
                recommendations.append("Identify and document lawful basis for processing")
            
            # Check data subject rights notices
            if not self._check_data_subject_rights_notice(content):
                violations.append("GDPR Art13: Data subject rights not adequately communicated")
                risk_score = max(risk_score, 0.6)
                recommendations.append("Include data subject rights information")
        
        # Special category data checks
        if self._detect_gdpr_special_categories(content):
            risk_score = max(risk_score, 0.8)
            if not self._check_explicit_consent_indicators(content):
                violations.append("GDPR Art9: Special category data requires explicit consent")
                risk_score = max(risk_score, 0.9)
        
        # Cross-border transfer checks
        if provider in [AIProvider.OPENAI, AIProvider.GROK]:  # US providers
            if not self._check_adequacy_decision_or_safeguards(content):
                violations.append("GDPR Art44: Cross-border transfer lacks adequate safeguards")
                risk_score = max(risk_score, 0.8)
                recommendations.append("Implement Standard Contractual Clauses or other safeguards")
        
        return {
            "violations": violations,
            "risk_score": risk_score,
            "blocked_content": blocked_content,
            "recommendations": recommendations
        }
    
    def _check_popia_guardrails(
        self,
        content: str,
        provider: AIProvider,
        doc_type: Optional[DocumentType]
    ) -> Dict[str, Any]:
        """South Africa POPIA guardrails"""
        
        violations = []
        blocked_content = []
        recommendations = []
        risk_score = 0.0
        
        # South African ID detection
        sa_ids = re.findall(r'\b\d{13}\b', content)
        if sa_ids:
            risk_score = max(risk_score, 0.7)
            if not self._check_consent_indicators(content):
                violations.append("POPIA: Processing SA ID numbers requires consent")
                recommendations.append("Obtain consent for ID number processing")
        
        # Cross-border restrictions
        if provider not in [AIProvider.MISTRAL]:  # Non-European providers
            violations.append("POPIA: Cross-border transfer restrictions apply")
            risk_score = max(risk_score, 0.8)
            recommendations.append("Ensure adequate protection for cross-border transfers")
        
        return {
            "violations": violations,
            "risk_score": risk_score,
            "blocked_content": blocked_content,
            "recommendations": recommendations
        }
    
    def _check_ndpr_guardrails(
        self,
        content: str,
        provider: AIProvider,
        doc_type: Optional[DocumentType]
    ) -> Dict[str, Any]:
        """Nigeria NDPR guardrails"""
        
        violations = []
        blocked_content = []
        recommendations = []
        risk_score = 0.0
        
        # Nigerian phone numbers and BVN detection
        nigerian_phones = re.findall(r'\+234[\s-]?[789][01][\s-]?\d{4}[\s-]?\d{4}', content)
        bvn_numbers = re.findall(r'\bBVN[\s:]*\d{11}\b', content)
        
        if nigerian_phones or bvn_numbers:
            risk_score = max(risk_score, 0.8)
            
            # Data localization requirements
            if provider in [AIProvider.OPENAI, AIProvider.CLAUDE, AIProvider.GROK]:
                violations.append("NDPR: Nigerian data must be processed within Nigeria")
                risk_score = max(risk_score, 0.9)
                blocked_content.extend(nigerian_phones + bvn_numbers)
                recommendations.append("Use Nigeria-localized data processing")
        
        return {
            "violations": violations,
            "risk_score": risk_score,
            "blocked_content": blocked_content,
            "recommendations": recommendations
        }
    
    def _check_ccpa_guardrails(
        self,
        content: str,
        provider: AIProvider,
        doc_type: Optional[DocumentType]
    ) -> Dict[str, Any]:
        """California CCPA guardrails"""
        
        violations = []
        blocked_content = []
        recommendations = []
        risk_score = 0.0
        
        # California-specific PII detection
        ca_drivers_license = re.findall(r'\b[A-Z]\d{7}\b', content)
        ssn_numbers = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', content)
        
        if ca_drivers_license or ssn_numbers:
            risk_score = max(risk_score, 0.6)
            
            # Check for privacy notice
            if not self._check_privacy_notice_indicators(content):
                violations.append("CCPA: Privacy notice requirements not met")
                risk_score = max(risk_score, 0.7)
                recommendations.append("Include CCPA-compliant privacy notice")
            
            # Check opt-out mechanisms
            if not self._check_opt_out_mechanisms(content):
                violations.append("CCPA: Right to opt-out not adequately provided")
                recommendations.append("Implement clear opt-out mechanisms")
        
        return {
            "violations": violations,
            "risk_score": risk_score,
            "blocked_content": blocked_content,
            "recommendations": recommendations
        }
    
    def _check_provider_guardrails(
        self,
        content: str,
        provider: AIProvider,
        jurisdiction: Jurisdiction
    ) -> Dict[str, Any]:
        """Provider-specific compliance checks"""
        
        violations = []
        recommendations = []
        risk_score = 0.0
        
        # OpenAI specific checks
        if provider == AIProvider.OPENAI:
            # US-based provider considerations
            if jurisdiction in [Jurisdiction.MOZAMBIQUE_DPL, Jurisdiction.NIGERIA]:
                risk_score = max(risk_score, 0.6)
                recommendations.append("Consider data residency requirements for US-based processing")
        
        # Mistral specific checks
        elif provider == AIProvider.MISTRAL:
            # EU-based provider advantages
            if jurisdiction == Jurisdiction.EU_GDPR:
                risk_score = max(risk_score, 0.2)  # Lower risk for EU processing
                recommendations.append("EU-based processing provides GDPR advantages")
        
        # Placeholder providers
        elif provider in [AIProvider.CLAUDE, AIProvider.GEMINI, AIProvider.GROK, AIProvider.COHERE]:
            violations.append(f"{provider.value}: Provider not yet available")
            risk_score = max(risk_score, 0.5)
            recommendations.append(f"Consider alternative providers while {provider.value} integration is pending")
        
        return {
            "violations": violations,
            "risk_score": risk_score,
            "recommendations": recommendations
        }
    
    # Helper methods for PII and consent detection
    def _detect_mozambican_pii(self, content: str) -> Dict[str, Any]:
        """Detect Mozambican-specific PII"""
        patterns = {
            "mozambican_id": r'\b\d{13}[A-Z]\b',
            "bi_number": r'\bBI[\s]*\d{8}[A-Z]\b',
            "mozambican_phone": r'\+258[\s-]?[82][0-9][\s-]?\d{3}[\s-]?\d{3}'
        }
        
        found_items = []
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            found_items.extend(matches)
        
        return {
            "found": len(found_items) > 0,
            "content": found_items,
            "types": list(patterns.keys())
        }
    
    def _detect_eu_pii(self, content: str) -> Dict[str, Any]:
        """Detect EU personal data"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\+[1-9]\d{1,14}'
        
        emails = re.findall(email_pattern, content)
        phones = re.findall(phone_pattern, content)
        
        return {
            "found": len(emails) > 0 or len(phones) > 0,
            "content": emails + phones
        }
    
    def _check_consent_indicators(self, content: str) -> bool:
        """Check for consent indicators in content"""
        consent_keywords = [
            "consent", "agree", "authorize", "permit", "allow",
            "consentimento", "autorizo", "concordo"  # Portuguese
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in consent_keywords)
    
    def _check_explicit_consent_indicators(self, content: str) -> bool:
        """Check for explicit consent indicators"""
        explicit_keywords = [
            "explicit consent", "expressly agree", "explicitly authorize",
            "consentimento explícito", "autorização expressa"  # Portuguese
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in explicit_keywords)
    
    def _detect_special_categories_mz(self, content: str) -> Dict[str, Any]:
        """Detect special categories under Mozambique law"""
        special_keywords = [
            "health", "medical", "religion", "political", "trade union",
            "criminal", "sexual orientation", "saúde", "médico", "religião",
            "política", "sindical", "criminal", "orientação sexual"
        ]
        
        content_lower = content.lower()
        found_categories = [kw for kw in special_keywords if kw in content_lower]
        
        return {
            "found": len(found_categories) > 0,
            "categories": found_categories
        }
    
    def _detect_gdpr_special_categories(self, content: str) -> bool:
        """Detect GDPR Article 9 special categories"""
        special_keywords = [
            "racial", "ethnic", "political", "religious", "philosophical",
            "trade union", "genetic", "biometric", "health", "sex life",
            "sexual orientation"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in special_keywords)
    
    def _check_lawful_basis_indicators(self, content: str) -> bool:
        """Check for GDPR lawful basis indicators"""
        lawful_basis_keywords = [
            "lawful basis", "legitimate interest", "legal obligation",
            "contract", "vital interest", "public task"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in lawful_basis_keywords)
    
    def _check_data_subject_rights_notice(self, content: str) -> bool:
        """Check for data subject rights notice"""
        rights_keywords = [
            "right to access", "right to rectification", "right to erasure",
            "right to portability", "right to object", "data protection rights"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in rights_keywords)
    
    def _check_adequacy_decision_or_safeguards(self, content: str) -> bool:
        """Check for adequacy decision or safeguards indicators"""
        safeguard_keywords = [
            "adequacy decision", "standard contractual clauses",
            "binding corporate rules", "safeguards", "appropriate safeguards"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in safeguard_keywords)
    
    def _check_medical_consent_mozambique(self, content: str) -> bool:
        """Check for Mozambique-specific medical consent"""
        medical_consent_keywords = [
            "medical consent", "healthcare authorization", "patient consent",
            "consentimento médico", "autorização de saúde"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in medical_consent_keywords)
    
    def _check_privacy_notice_indicators(self, content: str) -> bool:
        """Check for privacy notice indicators (CCPA)"""
        notice_keywords = [
            "privacy notice", "privacy policy", "data collection notice",
            "california privacy rights", "do not sell"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in notice_keywords)
    
    def _check_opt_out_mechanisms(self, content: str) -> bool:
        """Check for opt-out mechanisms (CCPA)"""
        opt_out_keywords = [
            "opt out", "do not sell", "unsubscribe", "withdraw consent"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in opt_out_keywords)
    
    def _load_compliance_policies(self) -> Dict[str, Any]:
        """Load compliance policies from configuration"""
        try:
            policies_path = Path(__file__).parent.parent / "compliance" / "policies.yaml"
            with open(policies_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load compliance policies: {e}")
            return {}
    
    def _load_violation_patterns(self) -> Dict[str, List[str]]:
        """Load common violation patterns"""
        return {
            "high_risk_pii": [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{13,19}\b',  # Credit card
                r'\b\d{13}[A-Z]\b'  # Mozambican ID
            ],
            "medical_data": [
                r'\bMRN[\s:]*\d+\b',
                r'\bpatient[\s]+id[\s:]*\d+\b',
                r'\bmedical[\s]+record\b'
            ],
            "financial_data": [
                r'\baccount[\s]+number[\s:]*\d+\b',
                r'\bbank[\s]+account[\s:]*\d+\b',
                r'\brouting[\s]+number\b'
            ]
        }
    
    def _initialize_jurisdiction_guardrails(self) -> Dict[Jurisdiction, Dict[str, Any]]:
        """Initialize jurisdiction-specific guardrails configuration"""
        return {
            Jurisdiction.MOZAMBIQUE_DPL: {
                "data_localization_required": True,
                "explicit_consent_for_special": True,
                "cross_border_restrictions": True,
                "retention_limits": "5 years",
                "breach_notification": "72 hours"
            },
            Jurisdiction.EU_GDPR: {
                "lawful_basis_required": True,
                "data_subject_rights": True,
                "cross_border_safeguards": True,
                "retention_principle": "storage limitation",
                "breach_notification": "72 hours"
            },
            Jurisdiction.SOUTH_AFRICA: {
                "consent_required": True,
                "cross_border_restrictions": True,
                "data_subject_rights": True,
                "retention_limits": "reasonable period",
                "breach_notification": "immediately"
            },
            Jurisdiction.NIGERIA: {
                "data_localization_required": True,
                "consent_required": True,
                "audit_requirements": True,
                "retention_limits": "as specified",
                "breach_notification": "72 hours"
            },
            Jurisdiction.CALIFORNIA: {
                "privacy_notice_required": True,
                "opt_out_mechanisms": True,
                "data_subject_rights": True,
                "sale_of_data_restrictions": True,
                "breach_notification": "without unreasonable delay"
            }
        }