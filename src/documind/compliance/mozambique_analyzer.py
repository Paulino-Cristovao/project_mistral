"""
Mozambique Data Protection Law (MDPL) Compliance Analyzer
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..models import (
    RegionalCompliance, ComplianceStatus, RiskLevel, 
    Jurisdiction, DocumentType
)


class MozambiqueComplianceAnalyzer:
    """
    Specialized analyzer for Mozambique Data Protection Law compliance
    """
    
    def __init__(self):
        self.jurisdiction = Jurisdiction.MOZAMBIQUE_DPL
        self.mdpl_articles = {
            "Art12": "Lawful basis for processing",
            "Art15": "Special categories of personal data",
            "Art22": "Data subject rights",
            "Art28": "Data retention limitations",
            "Art35": "Cross-border data transfers",
            "Art42": "Data localization requirements"
        }
        
        # Mozambique-specific risk factors
        self.high_risk_factors = [
            "cross_border_transfer",
            "special_category_data",
            "lack_of_consent",
            "inadequate_security",
            "data_localization_violation",
            "excessive_retention"
        ]
        
    def analyze_document_compliance(
        self,
        document_text: str,
        document_type: DocumentType,
        extracted_entities: List[Dict],
        bridges: List[Dict]
    ) -> RegionalCompliance:
        """
        Analyze document for MDPL compliance
        """
        violations = []
        recommendations = []
        risk_level = RiskLevel.LOW
        
        # Check for Mozambican PII
        mozambican_pii = self._detect_mozambican_pii(document_text)
        
        # Check consent indicators using helper method
        consent_analysis = self._check_consent_indicators(document_text, document_type)
        
        # Assess data categories
        special_categories = self._detect_special_categories(document_text, extracted_entities)
        
        # Check cross-border implications
        cross_border_risk = self._assess_cross_border_risk(extracted_entities, bridges)
        
        # Evaluate consent requirements
        consent_analysis = self._analyze_consent_requirements(
            document_type, special_categories, mozambican_pii
        )
        
        # Determine compliance status
        compliance_status = ComplianceStatus.COMPLIANT
        
        if mozambican_pii:
            if not consent_analysis["consent_documented"]:
                violations.append("MDPL Art12: Lack of documented consent for Mozambican personal data")
                compliance_status = ComplianceStatus.NON_COMPLIANT
                risk_level = RiskLevel.HIGH
                
            if special_categories and not consent_analysis["explicit_consent"]:
                violations.append("MDPL Art15: Explicit consent required for special category data")
                compliance_status = ComplianceStatus.NON_COMPLIANT
                risk_level = RiskLevel.CRITICAL
                
        if cross_border_risk["high_risk"]:
            violations.append("MDPL Art35: Cross-border transfer restrictions apply")
            recommendations.append("Ensure adequate data protection measures for international transfers")
            if compliance_status == ComplianceStatus.COMPLIANT:
                compliance_status = ComplianceStatus.NEEDS_REVIEW
            risk_level = max(risk_level, RiskLevel.HIGH)
            
        # Data localization requirements
        if self._requires_data_localization(document_type, special_categories):
            recommendations.append("MDPL Art42: Consider data localization requirements")
            
        # Retention compliance
        retention_analysis = self._analyze_retention_compliance(document_type)
        if retention_analysis["excessive_retention_risk"]:
            violations.append("MDPL Art28: Potential excessive data retention")
            recommendations.append(f"Review data retention period: {retention_analysis['recommended_period']}")
            
        # Generate recommendations
        if not violations:
            recommendations.append("Document appears compliant with MDPL requirements")
        else:
            recommendations.extend(self._generate_remediation_recommendations(violations))
            
        return RegionalCompliance(
            jurisdiction=self.jurisdiction,
            compliance_status=compliance_status,
            risk_level=risk_level,
            applicable_laws=[
                "Lei de Protecção de Dados Pessoais (MDPL)",
                "Decreto nº 32/2021 (Regulamento da LPDP)",
                "Constituição da República de Moçambique - Art35"
            ],
            violations=violations,
            recommendations=recommendations,
            processing_lawful_basis=self._determine_lawful_basis(document_type, consent_analysis),
            consent_required=consent_analysis["consent_required"],
            cross_border_restrictions=cross_border_risk["restrictions_apply"],
            data_subject_rights=[
                "Direito de acesso",
                "Direito de rectificação",
                "Direito de eliminação",
                "Direito de portabilidade",
                "Direito de oposição",
                "Direito de limitação do tratamento"
            ],
            retention_limits=retention_analysis["recommended_period"],
            breach_notification_required=True
        )
    
    def _detect_mozambican_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect Mozambican-specific PII patterns"""
        pii_found = {
            "mozambican_ids": [],
            "mozambican_phones": [],
            "portuguese_names": []
        }
        
        # Mozambican ID patterns
        id_patterns = [
            r'\b\d{13}[A-Z]\b',  # National ID format
            r'\bBI[\s]*\d{8}[A-Z]\b'  # BI format
        ]
        for pattern in id_patterns:
            matches = re.findall(pattern, text)
            pii_found["mozambican_ids"].extend(matches)
            
        # Mozambican phone numbers
        phone_patterns = [
            r'\+258[\s-]?[82][0-9][\s-]?\d{3}[\s-]?\d{3}\b',
            r'\b[82][0-9][\s-]?\d{3}[\s-]?\d{3}\b'
        ]
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            pii_found["mozambican_phones"].extend(matches)
            
        # Portuguese names (common in Mozambique)
        name_pattern = r'\b[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ][a-záàâãéèêíìîóòôõúùû]+(\s+[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ][a-záàâãéèêíìîóòôõúùû]+)*\b'
        names = re.findall(name_pattern, text)
        pii_found["portuguese_names"] = [name[0] if isinstance(name, tuple) else name for name in names[:5]]  # Limit to 5
        
        return pii_found
    
    def _detect_special_categories(self, text: str, entities: List[Dict]) -> bool:
        """Detect special categories of data under MDPL"""
        special_keywords = [
            "saúde", "medical", "médico", "hospital", "clínica",
            "religião", "religion", "religiosa", "church", "igreja",
            "política", "political", "partido", "eleição",
            "sindical", "union", "sindicato",
            "criminal", "crime", "prisão", "tribunal",
            "sexual", "orientação sexual", "vida sexual"
        ]
        
        text_lower = text.lower()
        for keyword in special_keywords:
            if keyword in text_lower:
                return True
                
        # Check entity types for health-related entities
        for entity in entities:
            if entity.get("label") in ["MEDICAL", "HOSPITAL", "CLINIC"]:
                return True
                
        return False
    
    def _assess_cross_border_risk(self, entities: List[Dict], bridges: List[Dict]) -> Dict[str, bool]:
        """Assess cross-border data transfer risks"""
        international_entities = []
        foreign_countries = [
            "south africa", "zimbabwe", "malawi", "tanzania", "zambia",
            "swaziland", "botswana", "europe", "america", "usa", "china",
            "brazil", "portugal", "uk", "france", "germany"
        ]
        
        # Check for international entity mentions
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            for country in foreign_countries:
                if country in entity_text:
                    international_entities.append(entity_text)
                    
        has_cross_border_risk = len(international_entities) > 0
        
        return {
            "high_risk": has_cross_border_risk,
            "restrictions_apply": has_cross_border_risk,
            "international_entities": international_entities
        }
    
    def _analyze_consent_requirements(
        self, 
        document_type: DocumentType, 
        has_special_categories: bool,
        mozambican_pii: Dict[str, List[str]]
    ) -> Dict[str, bool]:
        """Analyze consent requirements under MDPL"""
        
        has_mozambican_data = any(
            len(pii_list) > 0 for pii_list in mozambican_pii.values()
        )
        
        consent_required = has_mozambican_data  # MDPL requires consent for processing
        explicit_consent_required = has_special_categories  # Art15 requirement
        
        # For research and medical documents, explicit consent is always required
        if document_type in [DocumentType.MEDICAL, DocumentType.RESEARCH]:
            explicit_consent_required = True
            
        return {
            "consent_required": consent_required,
            "explicit_consent": explicit_consent_required,
            "consent_documented": False  # Would need document analysis to determine
        }
    
    def _requires_data_localization(self, document_type: DocumentType, has_special_categories: bool) -> bool:
        """Determine if data localization is required"""
        # Medical and financial data often require localization
        if document_type in [DocumentType.MEDICAL, DocumentType.FINANCIAL]:
            return True
            
        # Special categories of data may require localization
        if has_special_categories:
            return True
            
        return False
    
    def _analyze_retention_compliance(self, document_type: DocumentType) -> Dict[str, any]:
        """Analyze data retention compliance"""
        retention_periods = {
            DocumentType.MEDICAL: "6 years",
            DocumentType.FINANCIAL: "7 years", 
            DocumentType.CONTRACT: "5 years",
            DocumentType.RESEARCH: "Ethics committee approval period",
            DocumentType.LEGAL: "10 years (statutory limitation)"
        }
        
        recommended_period = retention_periods.get(document_type, "5 years")
        
        # Check for excessive retention risk (placeholder logic)
        excessive_retention_risk = False  # Would implement based on document content
        
        return {
            "recommended_period": recommended_period,
            "excessive_retention_risk": excessive_retention_risk
        }
    
    def _determine_lawful_basis(self, document_type: DocumentType, consent_analysis: Dict) -> str:
        """Determine the lawful basis for processing under MDPL Art12"""
        
        if consent_analysis["consent_required"]:
            return "MDPL Art12(1)(a) - Consent of the data subject"
            
        lawful_basis_map = {
            DocumentType.CONTRACT: "MDPL Art12(1)(b) - Performance of contract",
            DocumentType.LEGAL: "MDPL Art12(1)(c) - Legal obligation",
            DocumentType.MEDICAL: "MDPL Art15(2)(h) - Healthcare provision",
            DocumentType.RESEARCH: "MDPL Art12(1)(f) - Legitimate interests (with ethics approval)",
            DocumentType.FINANCIAL: "MDPL Art12(1)(c) - Legal obligation"
        }
        
        return lawful_basis_map.get(document_type, "MDPL Art12(1)(f) - Legitimate interests")
    
    def _generate_remediation_recommendations(self, violations: List[str]) -> List[str]:
        """Generate specific remediation recommendations"""
        recommendations = []
        
        for violation in violations:
            if "consent" in violation.lower():
                recommendations.append("Implement clear consent mechanisms with opt-in/opt-out options")
                recommendations.append("Document consent records with timestamps and purposes")
                
            if "cross-border" in violation.lower():
                recommendations.append("Implement adequate safeguards for international transfers")
                recommendations.append("Consider Standard Contractual Clauses or adequacy decisions")
                
            if "retention" in violation.lower():
                recommendations.append("Implement automated data retention policies")
                recommendations.append("Regular review and deletion of outdated personal data")
                
            if "localization" in violation.lower():
                recommendations.append("Consider data localization within Mozambique borders")
                recommendations.append("Evaluate cloud storage providers with local presence")
                
        return recommendations
    
    def _check_consent_indicators(self, content: str, doc_type: DocumentType) -> Dict[str, bool]:
        """Check for consent indicators in document content"""
        consent_keywords = [
            "consent", "consentimento", "autorização", "autorizo", 
            "concordo", "aceito", "permito", "autorizar", "BI:"
        ]
        
        content_lower = content.lower()
        consent_found = any(keyword in content_lower for keyword in consent_keywords)
        
        return {
            "consent_documented": consent_found,
            "explicit_consent_required": doc_type in [DocumentType.MEDICAL, DocumentType.RESEARCH]
        }