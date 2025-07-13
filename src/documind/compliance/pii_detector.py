"""
PII detection and redaction system
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional

import spacy
from spacy.lang.en import English

from ..models import RedactionLevel

logger = logging.getLogger(__name__)


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information (PII)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PII detector
        
        Args:
            config: Configuration dictionary containing PII patterns
        """
        self.config = config
        self.pii_patterns = config.get("pii_patterns", {})
        
        # Initialize spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found, using basic NER")
            self.nlp = English()
            
        logger.info("PII detector initialized")
    
    def process(
        self, 
        text: str, 
        redaction_level: RedactionLevel
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process text to detect and redact PII
        
        Args:
            text: Input text to process
            redaction_level: Level of redaction to apply
            
        Returns:
            Tuple of (redacted_text, redaction_info)
        """
        if not text or redaction_level == RedactionLevel.NONE:
            return text, {"count": 0, "types": [], "risk_flags": []}
        
        redacted_text = text
        redaction_info = {
            "count": 0,
            "types": [],
            "risk_flags": [],
            "entities_found": []
        }
        
        # Step 1: Named Entity Recognition using spaCy
        spacy_redactions = self._detect_entities_spacy(text, redaction_level)
        redacted_text, spacy_info = self._apply_redactions(redacted_text, spacy_redactions)
        
        # Step 2: Pattern-based detection
        pattern_redactions = self._detect_patterns(redacted_text, redaction_level)
        redacted_text, pattern_info = self._apply_redactions(redacted_text, pattern_redactions)
        
        # Step 3: Combine redaction information
        redaction_info["count"] = spacy_info["count"] + pattern_info["count"]
        redaction_info["types"] = list(set(spacy_info["types"] + pattern_info["types"]))
        redaction_info["entities_found"] = spacy_info["entities"] + pattern_info["entities"]
        
        # Step 4: Risk assessment
        redaction_info["risk_flags"] = self._assess_privacy_risks(redaction_info["entities_found"])
        
        logger.info(f"PII detection complete: {redaction_info['count']} redactions, "
                   f"{len(redaction_info['risk_flags'])} risk flags")
        
        return redacted_text, redaction_info
    
    def _detect_entities_spacy(
        self, 
        text: str, 
        redaction_level: RedactionLevel
    ) -> List[Dict[str, Any]]:
        """Detect entities using spaCy NER"""
        redactions = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Determine if entity should be redacted based on level
                if self._should_redact_entity(ent.label_, redaction_level):
                    redactions.append({
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "label": ent.label_,
                        "type": "spacy_ner",
                        "confidence": 0.8,  # Default spaCy confidence
                        "redaction": self._get_redaction_text(ent.label_)
                    })
        
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")
        
        return redactions
    
    def _detect_patterns(
        self, 
        text: str, 
        redaction_level: RedactionLevel
    ) -> List[Dict[str, Any]]:
        """Detect PII using regex patterns"""
        redactions = []
        
        for pii_type, pattern_config in self.pii_patterns.items():
            if not self._should_check_pattern(pii_type, redaction_level):
                continue
            
            patterns = pattern_config.get("patterns", [])
            confidence = pattern_config.get("confidence", 0.7)
            redaction_text = pattern_config.get("redaction", "[REDACTED]")
            
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        redactions.append({
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "label": pii_type,
                            "type": "pattern_match",
                            "confidence": confidence,
                            "redaction": redaction_text
                        })
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {pii_type}: {e}")
        
        return redactions
    
    def _apply_redactions(
        self, 
        text: str, 
        redactions: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply redactions to text"""
        if not redactions:
            return text, {"count": 0, "types": [], "entities": []}
        
        # Sort redactions by start position (reverse order for easier replacement)
        sorted_redactions = sorted(redactions, key=lambda x: x["start"], reverse=True)
        
        redacted_text = text
        redaction_types = set()
        entities = []
        
        for redaction in sorted_redactions:
            start = redaction["start"]
            end = redaction["end"]
            replacement = redaction["redaction"]
            
            # Apply redaction
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            
            # Track information
            redaction_types.add(redaction["label"])
            entities.append({
                "original_text": redaction["text"],
                "label": redaction["label"],
                "confidence": redaction["confidence"],
                "type": redaction["type"]
            })
        
        return redacted_text, {
            "count": len(redactions),
            "types": list(redaction_types),
            "entities": entities
        }
    
    def _should_redact_entity(self, entity_label: str, redaction_level: RedactionLevel) -> bool:
        """Determine if entity should be redacted based on level"""
        if redaction_level == RedactionLevel.NONE:
            return False
        elif redaction_level == RedactionLevel.BASIC:
            return entity_label in ["PERSON"]
        elif redaction_level == RedactionLevel.MODERATE:
            return entity_label in ["PERSON", "ORG", "GPE"]
        elif redaction_level == RedactionLevel.STRICT:
            return entity_label in ["PERSON", "ORG", "GPE", "DATE", "MONEY"]
        elif redaction_level == RedactionLevel.MAXIMUM:
            return True  # Redact all entities
        
        return False
    
    def _should_check_pattern(self, pii_type: str, redaction_level: RedactionLevel) -> bool:
        """Determine if pattern should be checked based on redaction level"""
        if redaction_level == RedactionLevel.NONE:
            return False
        elif redaction_level == RedactionLevel.BASIC:
            return pii_type in ["person_names", "email"]
        elif redaction_level == RedactionLevel.MODERATE:
            return pii_type in ["person_names", "email", "phone", "address"]
        elif redaction_level == RedactionLevel.STRICT:
            return pii_type in [
                "person_names", "email", "phone", "address", 
                "ssn", "credit_card", "bank_account"
            ]
        elif redaction_level == RedactionLevel.MAXIMUM:
            return True  # Check all patterns
        
        return False
    
    def _get_redaction_text(self, entity_label: str) -> str:
        """Get appropriate redaction text for entity type"""
        redaction_map = {
            "PERSON": "[NAME_REDACTED]",
            "ORG": "[ORG_REDACTED]",
            "GPE": "[LOCATION_REDACTED]",
            "MONEY": "[AMOUNT_REDACTED]",
            "DATE": "[DATE_REDACTED]",
            "PRODUCT": "[PRODUCT_REDACTED]",
            "EVENT": "[EVENT_REDACTED]",
            "WORK_OF_ART": "[WORK_REDACTED]",
            "LAW": "[LAW_REDACTED]",
            "LANGUAGE": "[LANGUAGE_REDACTED]",
            "NORP": "[GROUP_REDACTED]",
            "FACILITY": "[FACILITY_REDACTED]",
            "LOC": "[LOCATION_REDACTED]"
        }
        
        return redaction_map.get(entity_label, "[REDACTED]")
    
    def _assess_privacy_risks(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Assess privacy risks based on detected entities"""
        risk_flags = []
        
        # Get entity labels
        labels = [entity["label"] for entity in entities]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        # High-risk combinations
        if "PERSON" in labels and "MONEY" in labels:
            risk_flags.append("person_financial_data_combination")
        
        if "PERSON" in labels and "GPE" in labels:
            risk_flags.append("person_location_combination")
        
        if "PERSON" in labels and "ORG" in labels and "MONEY" in labels:
            risk_flags.append("person_org_financial_combination")
        
        # High entity density
        if len(entities) > 10:
            risk_flags.append("high_pii_density")
        
        # Sensitive PII types
        sensitive_types = ["ssn", "credit_card", "medical_record", "bank_account"]
        for entity in entities:
            if entity["label"] in sensitive_types:
                risk_flags.append(f"sensitive_data_{entity['label']}")
        
        # Multiple person names (potential data subject lists)
        if label_counts.get("PERSON", 0) > 3:
            risk_flags.append("multiple_data_subjects")
        
        return list(set(risk_flags))  # Remove duplicates
    
    def detect_special_categories(self, text: str) -> Dict[str, Any]:
        """
        Detect GDPR Article 9 special categories of personal data
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with special category indicators
        """
        special_categories = {
            "racial_ethnic": [],
            "political_opinions": [],
            "religious_beliefs": [],
            "trade_union": [],
            "genetic_data": [],
            "biometric_data": [],
            "health_data": [],
            "sex_life": []
        }
        
        text_lower = text.lower()
        
        # Health data indicators
        health_keywords = [
            "diagnosis", "treatment", "medication", "medical", "health",
            "patient", "doctor", "hospital", "clinic", "disease",
            "symptoms", "therapy", "prescription", "laboratory"
        ]
        for keyword in health_keywords:
            if keyword in text_lower:
                special_categories["health_data"].append(keyword)
        
        # Genetic/biometric indicators
        genetic_keywords = [
            "dna", "genetic", "genome", "chromosome", "hereditary",
            "fingerprint", "biometric", "retina", "facial recognition"
        ]
        for keyword in genetic_keywords:
            if keyword in text_lower:
                if "genetic" in keyword or "dna" in keyword:
                    special_categories["genetic_data"].append(keyword)
                else:
                    special_categories["biometric_data"].append(keyword)
        
        # Religious indicators
        religious_keywords = [
            "religion", "religious", "faith", "church", "mosque",
            "temple", "synagogue", "prayer", "worship", "belief"
        ]
        for keyword in religious_keywords:
            if keyword in text_lower:
                special_categories["religious_beliefs"].append(keyword)
        
        # Political indicators
        political_keywords = [
            "political", "politics", "party", "election", "vote",
            "candidate", "government", "democracy", "republican", "democrat"
        ]
        for keyword in political_keywords:
            if keyword in text_lower:
                special_categories["political_opinions"].append(keyword)
        
        # Remove empty categories
        return {k: v for k, v in special_categories.items() if v}