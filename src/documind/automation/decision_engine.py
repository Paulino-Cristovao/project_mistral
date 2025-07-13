"""
Decision engine for intelligent document processing decisions
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Makes intelligent decisions about document processing parameters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decision engine
        
        Args:
            config: Decision engine configuration
        """
        self.config = config or {}
        
        # Decision rules and weights
        self.decision_rules = self._load_decision_rules()
        self.confidence_thresholds = self._load_confidence_thresholds()
        
        logger.info("Decision engine initialized")
    
    def make_processing_decisions(
        self,
        doc_characteristics: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        processing_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Make intelligent decisions about document processing
        
        Args:
            doc_characteristics: Document analysis results
            user_preferences: Optional user preferences
            context_hints: Optional context hints
            processing_history: Historical processing data
            
        Returns:
            Processing decisions and parameters
        """
        try:
            # Step 1: Analyze document type and complexity
            doc_analysis = self._analyze_document_type(doc_characteristics)
            
            # Step 2: Determine optimal processing parameters
            processing_params = self._determine_processing_parameters(
                doc_analysis, doc_characteristics
            )
            
            # Step 3: Select optimal providers and strategies
            provider_selection = self._select_optimal_providers(
                doc_analysis, processing_params, processing_history
            )
            
            # Step 4: Determine compliance requirements
            compliance_decisions = self._determine_compliance_requirements(
                doc_analysis, user_preferences
            )
            
            # Step 5: Configure advanced features
            feature_config = self._configure_advanced_features(
                doc_analysis, context_hints
            )
            
            # Step 6: Apply user preferences
            final_decisions = self._apply_user_preferences(
                {
                    **processing_params,
                    **provider_selection,
                    **compliance_decisions,
                    **feature_config
                },
                user_preferences
            )
            
            # Step 7: Calculate decision confidence
            decision_confidence = self._calculate_decision_confidence(
                doc_analysis, final_decisions
            )
            
            final_decisions.update({
                "decision_confidence": decision_confidence,
                "decision_timestamp": datetime.now().isoformat(),
                "document_analysis": doc_analysis,
                "reasoning": self._generate_decision_reasoning(final_decisions)
            })
            
            logger.info(f"Processing decisions made with {decision_confidence:.2f} confidence")
            return final_decisions
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return self._create_fallback_decisions(doc_characteristics)
    
    def _load_decision_rules(self) -> Dict[str, Any]:
        """Load decision rules and weights"""
        return {
            "document_type_rules": {
                "contract": {
                    "preferred_provider": "openai",
                    "confidence_threshold": 0.8,
                    "enable_bridge_extraction": True,
                    "redaction_level": "strict"
                },
                "medical": {
                    "preferred_provider": "mistral",
                    "confidence_threshold": 0.9,
                    "enable_bridge_extraction": True,
                    "redaction_level": "maximum"
                },
                "financial": {
                    "preferred_provider": "openai",
                    "confidence_threshold": 0.85,
                    "enable_bridge_extraction": True,
                    "redaction_level": "strict"
                },
                "legal": {
                    "preferred_provider": "openai",
                    "confidence_threshold": 0.8,
                    "enable_bridge_extraction": True,
                    "redaction_level": "moderate"
                },
                "research": {
                    "preferred_provider": "mistral",
                    "confidence_threshold": 0.7,
                    "enable_bridge_extraction": False,
                    "redaction_level": "basic"
                }
            },
            "complexity_rules": {
                "high": {
                    "strategy": "parallel",
                    "timeout_seconds": 600,
                    "max_file_size_mb": 200
                },
                "medium": {
                    "strategy": "sequential",
                    "timeout_seconds": 300,
                    "max_file_size_mb": 100
                },
                "low": {
                    "strategy": "best_agent",
                    "timeout_seconds": 120,
                    "max_file_size_mb": 50
                }
            },
            "language_rules": {
                "french": {"preferred_provider": "mistral", "confidence_boost": 0.1},
                "german": {"preferred_provider": "mistral", "confidence_boost": 0.1},
                "spanish": {"preferred_provider": "mistral", "confidence_boost": 0.1},
                "english": {"preferred_provider": "openai", "confidence_boost": 0.05}
            }
        }
    
    def _load_confidence_thresholds(self) -> Dict[str, float]:
        """Load confidence thresholds for different scenarios"""
        return {
            "document_type_prediction": 0.7,
            "complexity_assessment": 0.6,
            "provider_selection": 0.8,
            "compliance_classification": 0.75,
            "overall_decision": 0.7
        }
    
    def _analyze_document_type(self, doc_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document type and properties"""
        
        analysis = {
            "predicted_document_type": "unknown",
            "type_prediction_confidence": 0.0,
            "complexity_assessment": doc_characteristics.get("complexity_estimate", "medium"),
            "processing_requirements": []
        }
        
        # Extract filename hints
        filename_hints = doc_characteristics.get("file_name_hints", {})
        type_hints = filename_hints.get("document_type_hints", [])
        
        # Predict document type based on hints
        if type_hints:
            # Use the first hint as primary type
            predicted_type = type_hints[0]
            analysis["predicted_document_type"] = predicted_type
            analysis["type_prediction_confidence"] = 0.8  # High confidence from filename
        else:
            # Use file size and complexity for prediction
            file_size = doc_characteristics.get("file_size", 0)
            if file_size > 5 * 1024 * 1024:  # > 5MB
                analysis["predicted_document_type"] = "legal"
                analysis["type_prediction_confidence"] = 0.6
            elif file_size < 1024 * 1024:  # < 1MB
                analysis["predicted_document_type"] = "research"
                analysis["type_prediction_confidence"] = 0.5
            else:
                analysis["predicted_document_type"] = "contract"
                analysis["type_prediction_confidence"] = 0.5
        
        # Determine processing requirements
        compliance_hints = filename_hints.get("compliance_hints", [])
        processing_hints = filename_hints.get("processing_hints", [])
        
        if "high_sensitivity" in compliance_hints:
            analysis["processing_requirements"].append("enhanced_privacy")
        if "privacy_relevant" in compliance_hints:
            analysis["processing_requirements"].append("gdpr_compliance")
        if "scanned_document" in processing_hints:
            analysis["processing_requirements"].append("ocr_optimization")
        if "structured_form" in processing_hints:
            analysis["processing_requirements"].append("table_extraction")
        
        return analysis
    
    def _determine_processing_parameters(
        self,
        doc_analysis: Dict[str, Any],
        doc_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine optimal processing parameters"""
        
        doc_type = doc_analysis["predicted_document_type"]
        complexity = doc_analysis["complexity_assessment"]
        
        # Get base parameters from rules
        type_rules = self.decision_rules["document_type_rules"].get(doc_type, {})
        complexity_rules = self.decision_rules["complexity_rules"].get(complexity, {})
        
        return {
            "optimal_strategy": complexity_rules.get("strategy", "sequential"),
            "confidence_threshold": type_rules.get("confidence_threshold", 0.7),
            "timeout_seconds": complexity_rules.get("timeout_seconds", 300),
            "max_file_size_mb": complexity_rules.get("max_file_size_mb", 100),
            "enable_vision_capabilities": doc_characteristics.get("file_extension") in [".pdf", ".png", ".jpg", ".jpeg"],
            "requires_vision_capabilities": complexity == "high" or "scanned_document" in doc_analysis.get("processing_requirements", [])
        }
    
    def _select_optimal_providers(
        self,
        doc_analysis: Dict[str, Any],
        processing_params: Dict[str, Any],
        processing_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Select optimal providers based on analysis"""
        
        doc_type = doc_analysis["predicted_document_type"]
        
        # Get provider preference from rules
        type_rules = self.decision_rules["document_type_rules"].get(doc_type, {})
        preferred_provider = type_rules.get("preferred_provider", "mistral")
        
        # Consider language preferences
        language = doc_analysis.get("likely_language", "english")
        language_rules = self.decision_rules["language_rules"].get(language, {})
        language_preferred = language_rules.get("preferred_provider")
        
        if language_preferred:
            preferred_provider = language_preferred
        
        # Consider vision requirements
        if processing_params.get("requires_vision_capabilities") and preferred_provider != "openai":
            preferred_provider = "openai"  # OpenAI has better vision capabilities
        
        # Consider historical performance
        if processing_history:
            historical_preference = self._analyze_historical_performance(
                processing_history, doc_analysis
            )
            if historical_preference:
                preferred_provider = historical_preference
        
        return {
            "preferred_provider": preferred_provider,
            "provider_selection_confidence": 0.8,
            "fallback_provider": "mistral" if preferred_provider == "openai" else "openai",
            "enable_provider_comparison": doc_type in ["contract", "legal", "financial"]
        }
    
    def _determine_compliance_requirements(
        self,
        doc_analysis: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Determine compliance requirements"""
        
        doc_type = doc_analysis["predicted_document_type"]
        processing_requirements = doc_analysis.get("processing_requirements", [])
        
        # Default compliance settings
        compliance_config = {
            "predicted_jurisdiction": "eu_gdpr",
            "optimal_redaction_level": "moderate",
            "enable_bridge_extraction": True,
            "compliance_profile": "standard"
        }
        
        # Adjust based on document type
        type_rules = self.decision_rules["document_type_rules"].get(doc_type, {})
        if "redaction_level" in type_rules:
            compliance_config["optimal_redaction_level"] = type_rules["redaction_level"]
        if "enable_bridge_extraction" in type_rules:
            compliance_config["enable_bridge_extraction"] = type_rules["enable_bridge_extraction"]
        
        # Adjust based on processing requirements
        if "enhanced_privacy" in processing_requirements:
            compliance_config["optimal_redaction_level"] = "maximum"
            compliance_config["compliance_profile"] = "enhanced"
        
        if "gdpr_compliance" in processing_requirements:
            compliance_config["predicted_jurisdiction"] = "eu_gdpr"
        
        # Apply user preferences
        if user_preferences:
            if "jurisdiction" in user_preferences:
                compliance_config["predicted_jurisdiction"] = user_preferences["jurisdiction"]
            if "redaction_level" in user_preferences:
                compliance_config["optimal_redaction_level"] = user_preferences["redaction_level"]
        
        return compliance_config
    
    def _configure_advanced_features(
        self,
        doc_analysis: Dict[str, Any],
        context_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Configure advanced processing features"""
        
        processing_requirements = doc_analysis.get("processing_requirements", [])
        
        config = {
            "enable_table_extraction": "table_extraction" in processing_requirements,
            "enable_entity_linking": True,
            "enable_semantic_analysis": doc_analysis["predicted_document_type"] in ["contract", "legal"],
            "enable_relationship_analysis": doc_analysis["predicted_document_type"] != "research",
            "optimize_for_accuracy": doc_analysis["complexity_assessment"] == "high",
            "optimize_for_speed": doc_analysis["complexity_assessment"] == "low"
        }
        
        # Apply context hints
        if context_hints:
            if context_hints.get("focus_on_relationships"):
                config["enable_relationship_analysis"] = True
                config["enable_bridge_extraction"] = True
            if context_hints.get("speed_priority"):
                config["optimize_for_speed"] = True
                config["optimize_for_accuracy"] = False
        
        return config
    
    def _apply_user_preferences(
        self,
        decisions: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply user preferences to override decisions"""
        
        if not user_preferences:
            return decisions
        
        # Apply explicit overrides
        override_fields = [
            "preferred_provider", "optimal_strategy", "confidence_threshold",
            "optimal_redaction_level", "enable_bridge_extraction"
        ]
        
        for field in override_fields:
            if field in user_preferences:
                decisions[field] = user_preferences[field]
                decisions[f"{field}_override"] = True
        
        return decisions
    
    def _analyze_historical_performance(
        self,
        processing_history: List[Dict[str, Any]],
        doc_analysis: Dict[str, Any]
    ) -> Optional[str]:
        """Analyze historical performance to select provider"""
        
        if not processing_history:
            return None
        
        # Find similar documents in history
        similar_docs = []
        target_type = doc_analysis["predicted_document_type"]
        
        for entry in processing_history[-20:]:  # Last 20 entries
            if entry.get("characteristics", {}).get("predicted_document_type") == target_type:
                similar_docs.append(entry)
        
        if len(similar_docs) < 3:  # Not enough data
            return None
        
        # Calculate performance by provider
        provider_performance: Dict[str, Dict[str, List[float]]] = {}
        
        for doc in similar_docs:
            provider = doc.get("decisions", {}).get("preferred_provider")
            quality = doc.get("result_quality", 0.0)
            processing_time = doc.get("processing_time", 0.0)
            
            if provider:
                if provider not in provider_performance:
                    provider_performance[provider] = {"quality": [], "time": []}
                
                provider_performance[provider]["quality"].append(quality)
                provider_performance[provider]["time"].append(processing_time)
        
        # Select best performing provider
        best_provider = None
        best_score = 0.0
        
        for provider, metrics in provider_performance.items():
            avg_quality = sum(metrics["quality"]) / len(metrics["quality"])
            avg_time = sum(metrics["time"]) / len(metrics["time"])
            
            # Score: quality is more important than speed
            score = avg_quality * 0.7 + (1.0 / max(avg_time, 1.0)) * 0.3
            
            if score > best_score:
                best_score = score
                best_provider = provider
        
        return best_provider if best_provider else None
    
    def _calculate_decision_confidence(
        self,
        doc_analysis: Dict[str, Any],
        decisions: Dict[str, Any]
    ) -> float:
        """Calculate confidence in decision making"""
        
        confidence_factors = []
        
        # Document type confidence
        type_confidence = doc_analysis.get("type_prediction_confidence", 0.0)
        confidence_factors.append(type_confidence)
        
        # Provider selection confidence
        provider_confidence = decisions.get("provider_selection_confidence", 0.0)
        confidence_factors.append(provider_confidence)
        
        # Rule-based confidence (high if decisions match rules)
        doc_type = doc_analysis["predicted_document_type"]
        if doc_type in self.decision_rules["document_type_rules"]:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # User override penalty (lower confidence if user overrides)
        override_count = sum(1 for key in decisions.keys() if key.endswith("_override"))
        override_penalty = override_count * 0.1
        
        # Calculate weighted average
        base_confidence = sum(confidence_factors) / len(confidence_factors)
        final_confidence = max(0.1, base_confidence - override_penalty)
        
        return min(1.0, final_confidence)
    
    def _generate_decision_reasoning(self, decisions: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for decisions"""
        
        reasoning = []
        
        # Provider selection reasoning
        provider = decisions.get("preferred_provider", "unknown")
        doc_type = decisions.get("document_analysis", {}).get("predicted_document_type", "unknown")
        
        reasoning.append(f"Selected {provider} as primary provider for {doc_type} documents")
        
        # Strategy reasoning
        strategy = decisions.get("optimal_strategy", "sequential")
        complexity = decisions.get("document_analysis", {}).get("complexity_assessment", "medium")
        
        reasoning.append(f"Using {strategy} strategy due to {complexity} document complexity")
        
        # Compliance reasoning
        redaction_level = decisions.get("optimal_redaction_level", "moderate")
        reasoning.append(f"Applied {redaction_level} redaction level for privacy protection")
        
        # Feature configuration reasoning
        if decisions.get("enable_bridge_extraction"):
            reasoning.append("Enabled relationship extraction for entity analysis")
        
        if decisions.get("requires_vision_capabilities"):
            reasoning.append("Vision capabilities required for document processing")
        
        return reasoning
    
    def _create_fallback_decisions(self, doc_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback decisions when decision making fails"""
        
        return {
            "preferred_provider": "mistral",
            "optimal_strategy": "sequential",
            "confidence_threshold": 0.7,
            "optimal_redaction_level": "moderate",
            "enable_bridge_extraction": True,
            "predicted_jurisdiction": "eu_gdpr",
            "predicted_document_type": "unknown",
            "timeout_seconds": 300,
            "max_file_size_mb": 100,
            "decision_confidence": 0.5,
            "fallback_mode": True,
            "reasoning": ["Using fallback decisions due to analysis failure"]
        }
    
    def update_decision_rules(self, new_rules: Dict[str, Any]) -> None:
        """Update decision rules based on feedback"""
        
        for rule_type, rules in new_rules.items():
            if rule_type in self.decision_rules:
                self.decision_rules[rule_type].update(rules)
            else:
                self.decision_rules[rule_type] = rules
        
        logger.info("Decision rules updated based on feedback")
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of decision engine configuration"""
        
        return {
            "total_document_types": len(self.decision_rules["document_type_rules"]),
            "supported_complexities": list(self.decision_rules["complexity_rules"].keys()),
            "supported_languages": list(self.decision_rules["language_rules"].keys()),
            "confidence_thresholds": self.confidence_thresholds,
            "last_updated": datetime.now().isoformat()
        }