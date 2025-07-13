"""
Intelligent processor for hands-off document processing
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..agents import AgentCoordinator, OpenAIAgent, MistralAgent, ComplianceAgent, BridgeAgent
from ..models import ProcessingResult, ProcessingConfig, DocumentType, Jurisdiction, RedactionLevel
from .decision_engine import DecisionEngine
from .adaptive_agent import AdaptiveAgent

logger = logging.getLogger(__name__)


class IntelligentProcessor:
    """
    Hands-off intelligent processor that makes autonomous decisions
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        learning_enabled: bool = True
    ):
        """
        Initialize intelligent processor
        
        Args:
            openai_api_key: OpenAI API key
            mistral_api_key: Mistral API key  
            learning_enabled: Enable adaptive learning
        """
        self.learning_enabled = learning_enabled
        
        # Initialize agents
        self.agents: List[Any] = []
        
        if openai_api_key:
            self.openai_agent: Optional[OpenAIAgent] = OpenAIAgent(openai_api_key)
            self.agents.append(self.openai_agent)
        else:
            self.openai_agent: Optional[OpenAIAgent] = None
        
        if mistral_api_key:
            self.mistral_agent: Optional[MistralAgent] = MistralAgent(mistral_api_key)
            self.agents.append(self.mistral_agent)
        else:
            self.mistral_agent: Optional[MistralAgent] = None
        
        # Specialized agents
        self.compliance_agent = ComplianceAgent()
        self.bridge_agent = BridgeAgent()
        self.agents.extend([self.compliance_agent, self.bridge_agent])
        
        # Coordinator
        self.coordinator = AgentCoordinator(self.agents)
        
        # Decision engine
        self.decision_engine = DecisionEngine()
        
        # Adaptive learning
        if learning_enabled:
            self.adaptive_agent: Optional[AdaptiveAgent] = AdaptiveAgent()
        else:
            self.adaptive_agent: Optional[AdaptiveAgent] = None
        
        # Processing history
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info(f"Intelligent processor initialized with {len(self.agents)} agents")
    
    def process_intelligently(
        self,
        file_path: Union[str, Path],
        user_preferences: Optional[Dict[str, Any]] = None,
        context_hints: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process document with full automation and intelligent decision making
        
        Args:
            file_path: Path to document file
            user_preferences: Optional user preferences
            context_hints: Optional context hints about the document
            
        Returns:
            Processing result with automated decisions
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # Step 1: Analyze document characteristics
            doc_characteristics = self._analyze_document_characteristics(file_path)
            
            # Step 2: Make intelligent decisions about processing
            processing_decisions = self.decision_engine.make_processing_decisions(
                doc_characteristics, user_preferences, context_hints, self.processing_history
            )
            
            # Step 3: Create optimized processing configuration
            processing_config = self._create_optimized_config(processing_decisions)
            
            # Step 4: Select optimal agent and strategy
            agent_selection = self._select_optimal_agent(doc_characteristics, processing_decisions)
            
            # Step 5: Execute processing
            result = self.coordinator.process_document(
                file_path=file_path,
                strategy=agent_selection["strategy"],
                primary_agent=agent_selection["primary_agent"],
                processing_config=processing_config
            )
            
            # Step 6: Post-process and validate results
            validated_result = self._validate_and_enhance_results(result, processing_decisions)
            
            # Step 7: Learn from this processing (if enabled)
            if self.adaptive_agent:
                self.adaptive_agent.learn_from_processing(
                    doc_characteristics, processing_decisions, validated_result
                )
            
            # Step 8: Update processing history
            self._update_processing_history(
                file_path, doc_characteristics, processing_decisions, validated_result
            )
            
            # Step 9: Generate intelligent insights
            insights = self._generate_processing_insights(validated_result, processing_decisions)
            
            # Add automation metadata
            validated_result.automation_metadata = {
                "intelligent_processing": True,
                "decisions_made": processing_decisions,
                "agent_selection": agent_selection,
                "processing_insights": insights,
                "total_automation_time": (datetime.now() - start_time).total_seconds()
            }
            
            logger.info(f"Intelligent processing completed for {file_path.name}")
            return validated_result
            
        except Exception as e:
            logger.error(f"Intelligent processing failed: {e}")
            
            # Fallback to basic processing
            return self._fallback_processing(file_path, str(e))
    
    def _analyze_document_characteristics(self, file_path: Path) -> Dict[str, Any]:
        """Analyze document to understand its characteristics"""
        
        characteristics = {
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
            "estimated_pages": self._estimate_page_count(file_path),
            "file_name_hints": self._extract_filename_hints(file_path.name),
            "complexity_estimate": "medium"  # Will be refined by analysis
        }
        
        # Determine document complexity
        if characteristics["file_size"] > 10 * 1024 * 1024:  # > 10MB
            characteristics["complexity_estimate"] = "high"
        elif characteristics["file_size"] < 1024 * 1024:  # < 1MB
            characteristics["complexity_estimate"] = "low"
        
        # Language hints from filename
        if any(lang in file_path.name.lower() for lang in ["fr", "french", "francais"]):
            characteristics["likely_language"] = "french"
        elif any(lang in file_path.name.lower() for lang in ["de", "german", "deutsch"]):
            characteristics["likely_language"] = "german"
        elif any(lang in file_path.name.lower() for lang in ["es", "spanish", "espanol"]):
            characteristics["likely_language"] = "spanish"
        else:
            characteristics["likely_language"] = "english"
        
        return characteristics
    
    def _estimate_page_count(self, file_path: Path) -> int:
        """Estimate number of pages in document"""
        
        file_size = file_path.stat().st_size
        
        if file_path.suffix.lower() == ".pdf":
            # Rough estimate: 100KB per page for PDF
            return max(1, file_size // (100 * 1024))
        else:
            # For images, assume 1 page
            return 1
    
    def _extract_filename_hints(self, filename: str) -> Dict[str, Any]:
        """Extract hints from filename about document type and content"""
        
        filename_lower = filename.lower()
        hints = {
            "document_type_hints": [],
            "compliance_hints": [],
            "processing_hints": []
        }
        
        # Document type hints
        type_keywords = {
            "contract": ["contract", "agreement", "terms", "proposal"],
            "medical": ["medical", "patient", "health", "clinical"],
            "financial": ["invoice", "receipt", "financial", "payment"],
            "legal": ["legal", "court", "lawsuit", "brief"],
            "research": ["research", "study", "paper", "thesis"]
        }
        
        for doc_type, keywords in type_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                hints["document_type_hints"].append(doc_type)
        
        # Compliance hints
        if any(term in filename_lower for term in ["confidential", "private", "restricted"]):
            hints["compliance_hints"].append("high_sensitivity")
        
        if any(term in filename_lower for term in ["gdpr", "personal", "pii"]):
            hints["compliance_hints"].append("privacy_relevant")
        
        # Processing hints
        if any(term in filename_lower for term in ["scan", "scanned", "copy"]):
            hints["processing_hints"].append("scanned_document")
        
        if any(term in filename_lower for term in ["form", "application", "questionnaire"]):
            hints["processing_hints"].append("structured_form")
        
        return hints
    
    def _create_optimized_config(self, processing_decisions: Dict[str, Any]) -> ProcessingConfig:
        """Create optimized processing configuration based on decisions"""
        
        return ProcessingConfig(
            document_type=DocumentType(processing_decisions.get("predicted_document_type", "unknown")),
            jurisdiction=Jurisdiction(processing_decisions.get("predicted_jurisdiction", "unknown")),
            redaction_level=RedactionLevel(processing_decisions.get("optimal_redaction_level", "moderate")),
            enable_bridge_extraction=processing_decisions.get("enable_bridge_extraction", True),
            min_confidence_threshold=processing_decisions.get("confidence_threshold", 0.7),
            max_file_size_mb=processing_decisions.get("max_file_size_mb", 100),
            timeout_seconds=processing_decisions.get("timeout_seconds", 300)
        )
    
    def _select_optimal_agent(
        self,
        doc_characteristics: Dict[str, Any],
        processing_decisions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select optimal agent and strategy based on analysis"""
        
        # Default selection
        selection = {
            "primary_agent": "mistral",  # Default to Mistral
            "strategy": "sequential",
            "reasoning": []
        }
        
        # Agent selection logic
        if processing_decisions.get("requires_vision_capabilities"):
            if self.openai_agent:
                selection["primary_agent"] = "openai"
                selection["reasoning"].append("Selected OpenAI for vision capabilities")
            else:
                selection["reasoning"].append("Vision capabilities needed but OpenAI not available")
        
        # Language-specific selection
        likely_language = doc_characteristics.get("likely_language", "english")
        if likely_language in ["french", "german", "spanish"] and self.mistral_agent:
            selection["primary_agent"] = "mistral"
            selection["reasoning"].append(f"Selected Mistral for {likely_language} processing")
        
        # Complexity-based strategy selection
        complexity = doc_characteristics.get("complexity_estimate", "medium")
        if complexity == "high":
            selection["strategy"] = "parallel"
            selection["reasoning"].append("Selected parallel strategy for high complexity")
        elif complexity == "low":
            selection["strategy"] = "best_agent"
            selection["reasoning"].append("Selected best agent strategy for simple document")
        
        # Historical performance consideration
        if self.adaptive_agent:
            historical_preference = self.adaptive_agent.get_agent_preference(doc_characteristics)
            if historical_preference:
                selection["primary_agent"] = historical_preference
                selection["reasoning"].append("Selected based on historical performance")
        
        return selection
    
    def _validate_and_enhance_results(
        self,
        result: ProcessingResult,
        processing_decisions: Dict[str, Any]
    ) -> ProcessingResult:
        """Validate and enhance processing results"""
        
        # Quality validation
        quality_score = self._calculate_quality_score(result)
        
        # If quality is low, attempt enhancement
        if quality_score < 0.6:
            enhanced_result = self._attempt_result_enhancement(result, processing_decisions)
            if enhanced_result:
                result = enhanced_result
        
        # Add quality metadata
        result.quality_metadata = {
            "quality_score": quality_score,
            "validation_passed": quality_score >= 0.6,
            "enhancement_applied": quality_score < 0.6
        }
        
        return result
    
    def _calculate_quality_score(self, result: ProcessingResult) -> float:
        """Calculate quality score for processing result"""
        
        score = 0.0
        components = 0
        
        # Text extraction quality
        if result.extracted_text:
            text_quality = min(1.0, len(result.extracted_text) / 1000)  # Longer text = better
            score += text_quality
            components += 1
        
        # Compliance score
        score += result.compliance_metadata.compliance_score
        components += 1
        
        # Bridge extraction quality
        if result.bridges:
            avg_bridge_confidence = sum(b.confidence_score for b in result.bridges) / len(result.bridges)
            score += avg_bridge_confidence
            components += 1
        
        # Overall success
        if result.status.value == "completed":
            score += 1.0
            components += 1
        
        return score / components if components > 0 else 0.0
    
    def _attempt_result_enhancement(
        self,
        result: ProcessingResult,
        processing_decisions: Dict[str, Any]
    ) -> Optional[ProcessingResult]:
        """Attempt to enhance low-quality results"""
        
        # Could implement retry with different agent or parameters
        # For now, return original result
        logger.warning("Result enhancement not yet implemented")
        return None
    
    def _generate_processing_insights(
        self,
        result: ProcessingResult,
        processing_decisions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intelligent insights about the processing"""
        
        insights = {
            "document_analysis": {
                "predicted_type": processing_decisions.get("predicted_document_type"),
                "confidence": processing_decisions.get("type_prediction_confidence", 0.0),
                "complexity": processing_decisions.get("complexity_assessment")
            },
            "processing_efficiency": {
                "time_to_process": result.processing_time_seconds,
                "tokens_processed": getattr(result, "token_count", 0),
                "processing_speed": "fast" if result.processing_time_seconds < 5 else "moderate"
            },
            "compliance_assessment": {
                "risk_level": "high" if result.compliance_metadata.compliance_score < 0.7 else "low",
                "recommendations": self._generate_compliance_recommendations(result)
            },
            "relationship_insights": {
                "network_complexity": len(result.bridges),
                "high_risk_relationships": len([
                    b for b in result.bridges 
                    if b.privacy_impact.value in ["high", "critical"]
                ])
            }
        }
        
        return insights
    
    def _generate_compliance_recommendations(self, result: ProcessingResult) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        if result.compliance_metadata.compliance_score < 0.7:
            recommendations.append("Consider additional privacy controls")
        
        if result.compliance_metadata.redactions_count == 0:
            recommendations.append("Review for potential PII that may need redaction")
        
        if len(result.compliance_metadata.risk_flags) > 0:
            recommendations.append("Address identified risk flags before processing")
        
        return recommendations
    
    def _update_processing_history(
        self,
        file_path: Path,
        characteristics: Dict[str, Any],
        decisions: Dict[str, Any],
        result: ProcessingResult
    ) -> None:
        """Update processing history for learning"""
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": file_path.name,
            "characteristics": characteristics,
            "decisions": decisions,
            "result_quality": getattr(result, "quality_metadata", {}).get("quality_score", 0.0),
            "processing_time": result.processing_time_seconds,
            "success": result.status.value == "completed"
        }
        
        self.processing_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]
    
    def _fallback_processing(self, file_path: Path, error_message: str) -> ProcessingResult:
        """Fallback processing when intelligent processing fails"""
        
        try:
            # Simple fallback with basic configuration
            basic_config = ProcessingConfig()
            
            result = self.coordinator.process_document(
                file_path=file_path,
                strategy="best_agent",
                processing_config=basic_config
            )
            
            # Add fallback metadata
            result.automation_metadata = {
                "intelligent_processing": False,
                "fallback_reason": error_message,
                "fallback_processing": True
            }
            
            return result
            
        except Exception as e:
            # Create error result
            from ..models import ComplianceMetadata, DocumentType, Jurisdiction, RedactionLevel, ProcessingStatus
            
            return ProcessingResult(
                document_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
                    audit_reference=f"FALLBACK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    processor_version="2.0.0",
                    compliance_score=0.0
                ),
                processing_config=ProcessingConfig(),
                status=ProcessingStatus.FAILED,
                error_message=f"Intelligent processing failed: {error_message}, fallback also failed: {str(e)}",
                processing_time_seconds=0.0,
                provider_used="fallback"
            )
    
    def get_processing_analytics(self) -> Dict[str, Any]:
        """Get analytics about processing performance"""
        
        if not self.processing_history:
            return {"message": "No processing history available"}
        
        successful_processes = [h for h in self.processing_history if h["success"]]
        
        analytics = {
            "total_documents_processed": len(self.processing_history),
            "successful_processes": len(successful_processes),
            "success_rate": len(successful_processes) / len(self.processing_history),
            "average_processing_time": sum(h["processing_time"] for h in successful_processes) / len(successful_processes) if successful_processes else 0,
            "average_quality_score": sum(h["result_quality"] for h in successful_processes) / len(successful_processes) if successful_processes else 0
        }
        
        # Document type distribution
        type_distribution = {}
        for entry in self.processing_history:
            doc_type = entry["decisions"].get("predicted_document_type", "unknown")
            type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
        
        analytics["document_type_distribution"] = type_distribution
        
        return analytics