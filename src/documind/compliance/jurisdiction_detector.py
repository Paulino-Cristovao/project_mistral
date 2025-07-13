"""
Jurisdiction detection system
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

import spacy
from spacy.lang.en import English

from ..models import Jurisdiction

logger = logging.getLogger(__name__)


class JurisdictionDetector:
    """
    Detects regulatory jurisdiction based on language and content analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize jurisdiction detector
        
        Args:
            config: Configuration dictionary containing language mappings
        """
        self.config = config
        self.language_map = config.get("language_map", {})
        
        # Initialize spaCy for language detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found, using basic language detection")
            self.nlp = English()
        
        logger.info("Jurisdiction detector initialized")
    
    def detect(self, file_path: Union[str, Path], text_preview: Optional[str] = None) -> Jurisdiction:
        """
        Detect jurisdiction based on document content and metadata
        
        Args:
            file_path: Path to document file
            text_preview: Optional text preview for analysis
            
        Returns:
            Detected jurisdiction
        """
        file_path = Path(file_path)
        
        # Step 1: Language-based detection
        if text_preview:
            detected_language = self._detect_language(text_preview)
        else:
            detected_language = self._detect_language_from_filename(file_path)
        
        # Step 2: Map language to jurisdiction
        jurisdiction = self._map_language_to_jurisdiction(detected_language)
        
        # Step 3: Content-based jurisdiction hints
        if text_preview and jurisdiction == Jurisdiction.UNKNOWN:
            jurisdiction = self._detect_jurisdiction_from_content(text_preview)
        
        logger.info(f"Detected jurisdiction: {jurisdiction} (language: {detected_language})")
        return jurisdiction
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language from text content
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO language code (e.g., 'en', 'fr', 'de')
        """
        if not text or len(text.strip()) < 10:
            return "en"  # Default to English
        
        try:
            # Use spaCy for language detection
            doc = self.nlp(text[:500])  # Analyze first 500 characters
            
            # Simple heuristic: count language-specific patterns
            language_scores = {
                "en": self._score_english(text),
                "fr": self._score_french(text),
                "de": self._score_german(text),
                "es": self._score_spanish(text),
                "sw": self._score_swahili(text),
                "ha": self._score_hausa(text)
            }
            
            # Return language with highest score
            detected_lang = max(language_scores, key=language_scores.get)
            
            # Only return if score is above threshold
            if language_scores[detected_lang] > 0.3:
                return detected_lang
            
            return "en"  # Default fallback
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"
    
    def _detect_language_from_filename(self, file_path: Path) -> str:
        """Detect language hints from filename"""
        filename_lower = file_path.name.lower()
        
        # Language indicators in filename
        if any(indicator in filename_lower for indicator in ["_fr", "_french", "francais"]):
            return "fr"
        elif any(indicator in filename_lower for indicator in ["_de", "_german", "deutsch"]):
            return "de"
        elif any(indicator in filename_lower for indicator in ["_es", "_spanish", "espanol"]):
            return "es"
        elif any(indicator in filename_lower for indicator in ["_sw", "_swahili"]):
            return "sw"
        
        return "en"  # Default
    
    def _map_language_to_jurisdiction(self, language: str) -> Jurisdiction:
        """Map detected language to jurisdiction"""
        jurisdiction_str = self.language_map.get(language, "unknown")
        
        try:
            return Jurisdiction(jurisdiction_str)
        except ValueError:
            return Jurisdiction.UNKNOWN
    
    def _detect_jurisdiction_from_content(self, text: str) -> Jurisdiction:
        """
        Detect jurisdiction from content patterns and legal references
        
        Args:
            text: Text content to analyze
            
        Returns:
            Detected jurisdiction
        """
        text_lower = text.lower()
        
        # EU/GDPR indicators
        eu_indicators = [
            "gdpr", "general data protection regulation", "dsgvo",
            "article 6", "article 9", "data protection officer",
            "european union", "eu regulation", "directive 95/46/ec"
        ]
        
        # Africa/NDPR indicators
        africa_indicators = [
            "ndpr", "nigeria data protection regulation",
            "nitda", "national information technology development agency",
            "data protection compliance organisation", "dpco"
        ]
        
        # US/HIPAA indicators
        us_indicators = [
            "hipaa", "health insurance portability",
            "phi", "protected health information",
            "hhs", "department of health", "cfr"
        ]
        
        # Score each jurisdiction
        eu_score = sum(1 for indicator in eu_indicators if indicator in text_lower)
        africa_score = sum(1 for indicator in africa_indicators if indicator in text_lower)
        us_score = sum(1 for indicator in us_indicators if indicator in text_lower)
        
        # Return jurisdiction with highest score
        if eu_score > 0 and eu_score >= africa_score and eu_score >= us_score:
            return Jurisdiction.EU_GDPR
        elif africa_score > 0 and africa_score >= us_score:
            return Jurisdiction.AFRICA_NDPR
        elif us_score > 0:
            return Jurisdiction.US_HIPAA
        
        return Jurisdiction.UNKNOWN
    
    def _score_english(self, text: str) -> float:
        """Score likelihood of English language"""
        english_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through"
        ]
        return self._score_language_words(text, english_words)
    
    def _score_french(self, text: str) -> float:
        """Score likelihood of French language"""
        french_words = [
            "le", "la", "les", "de", "du", "des", "et", "ou", "dans",
            "sur", "avec", "pour", "par", "entre", "sous", "vers"
        ]
        return self._score_language_words(text, french_words)
    
    def _score_german(self, text: str) -> float:
        """Score likelihood of German language"""
        german_words = [
            "der", "die", "das", "und", "oder", "aber", "in", "auf",
            "mit", "für", "von", "zu", "bei", "nach", "über"
        ]
        return self._score_language_words(text, german_words)
    
    def _score_spanish(self, text: str) -> float:
        """Score likelihood of Spanish language"""
        spanish_words = [
            "el", "la", "los", "las", "de", "del", "y", "o", "en",
            "con", "por", "para", "desde", "hasta", "sobre"
        ]
        return self._score_language_words(text, spanish_words)
    
    def _score_swahili(self, text: str) -> float:
        """Score likelihood of Swahili language"""
        swahili_words = [
            "na", "ya", "wa", "za", "la", "kwa", "katika", "hadi",
            "kutoka", "pamoja", "bila", "baada", "kabla"
        ]
        return self._score_language_words(text, swahili_words)
    
    def _score_hausa(self, text: str) -> float:
        """Score likelihood of Hausa language"""
        hausa_words = [
            "da", "na", "ta", "a", "ga", "daga", "zuwa", "cikin",
            "akan", "tare", "ba", "ko", "amma", "don"
        ]
        return self._score_language_words(text, hausa_words)
    
    def _score_language_words(self, text: str, words: list[str]) -> float:
        """Score text based on frequency of language-specific words"""
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return 0.0
        
        matches = sum(1 for word in words if f" {word} " in f" {text_lower} ")
        return matches / word_count
    
    def get_jurisdiction_confidence(
        self, 
        file_path: Union[str, Path], 
        text_preview: Optional[str] = None
    ) -> float:
        """
        Get confidence score for jurisdiction detection
        
        Args:
            file_path: Path to document
            text_preview: Optional text preview
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple confidence calculation based on available information
        confidence = 0.5  # Base confidence
        
        if text_preview and len(text_preview.strip()) > 100:
            confidence += 0.3  # Bonus for substantial text content
        
        file_path = Path(file_path)
        if any(lang in file_path.name.lower() for lang in ["_fr", "_de", "_es"]):
            confidence += 0.2  # Bonus for language hints in filename
        
        return min(1.0, confidence)