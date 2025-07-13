"""
Agent coordinator for orchestrating DocBridgeGuard AI agents
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from .base_agent import BaseAgent
from .openai_agent import OpenAIAgent
from .mistral_agent import MistralAgent
from .compliance_agent import ComplianceAgent
from .bridge_agent import BridgeAgent
from ..models import ProcessingResult, ComplianceMetadata, ProcessingConfig, ProcessingStatus
from ..utils.comparison import ComparisonEngine

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates multiple AI agents for comprehensive document processing
    """
    
    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent coordinator
        
        Args:
            agents: List of agents to coordinate
            config: Coordinator configuration
        """
        self.agents = agents or []
        self.config = config or {}
        
        # Organize agents by type
        self.ocr_agents = []
        self.specialized_agents = []
        
        for agent in self.agents:
            if isinstance(agent, (OpenAIAgent, MistralAgent)):
                self.ocr_agents.append(agent)
            else:
                self.specialized_agents.append(agent)
        
        # Initialize comparison engine
        self.comparison_engine = ComparisonEngine()
        
        # Coordination strategies
        self.strategies = {
            "sequential": self._process_sequential,
            "parallel": self._process_parallel,
            "consensus": self._process_consensus,
            "best_agent": self._process_best_agent
        }
        
        logger.info(f"Agent coordinator initialized with {len(self.agents)} agents")
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the coordinator"""
        self.agents.append(agent)
        
        if isinstance(agent, (OpenAIAgent, MistralAgent)):
            self.ocr_agents.append(agent)
        else:
            self.specialized_agents.append(agent)
        
        logger.info(f"Added {agent.name} agent to coordinator")
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent by name"""
        for agent in self.agents:
            if agent.name == agent_name:
                self.agents.remove(agent)
                
                if agent in self.ocr_agents:
                    self.ocr_agents.remove(agent)
                if agent in self.specialized_agents:
                    self.specialized_agents.remove(agent)
                
                logger.info(f"Removed {agent_name} agent from coordinator")
                return True
        
        return False
    
    def process_document(
        self,
        file_path: Union[str, Path],
        strategy: str = "sequential",
        primary_agent: Optional[str] = None,
        enable_comparison: bool = False,
        processing_config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """
        Process document using coordinated agents
        
        Args:
            file_path: Path to document file
            strategy: Processing strategy ('sequential', 'parallel', 'consensus', 'best_agent')
            primary_agent: Primary agent to use (for best_agent strategy)
            enable_comparison: Whether to compare OCR agents
            processing_config: Processing configuration
            
        Returns:
            Comprehensive processing result
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # Validate file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Select processing strategy
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy '{strategy}', defaulting to 'sequential'")
                strategy = "sequential"
            
            # Execute processing strategy
            result = self.strategies[strategy](
                file_path, primary_agent, enable_comparison, processing_config
            )
            
            # Add coordinator metadata
            result["coordinator_metadata"] = {
                "strategy_used": strategy,
                "agents_involved": [agent.name for agent in self.agents],
                "total_processing_time": (datetime.now() - start_time).total_seconds(),
                "coordination_successful": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            
            # Return error result
            return self._create_error_result(file_path, str(e), start_time)
    
    def _process_sequential(
        self,
        file_path: Path,
        primary_agent: Optional[str],
        enable_comparison: bool,
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Process document sequentially through agents"""
        
        # Step 1: Choose primary OCR agent
        ocr_agent = self._select_primary_ocr_agent(primary_agent)
        if not ocr_agent:
            raise ValueError("No OCR agent available")
        
        # Step 2: Initial OCR processing
        task_context = self._create_initial_context(file_path, processing_config)
        ocr_result = ocr_agent.process_document(file_path, task_context)
        
        if not ocr_result.get("success"):
            raise RuntimeError(f"OCR processing failed: {ocr_result.get('error')}")
        
        # Step 3: Extract OCR results
        ocr_data = ocr_result["result"]
        extracted_text = ocr_data.get("text", "")
        tables = ocr_data.get("tables", [])
        entities = ocr_data.get("entities", [])
        
        # Step 4: Compliance processing
        compliance_agent = self._find_agent_by_type(ComplianceAgent)
        compliance_result = None
        
        if compliance_agent:
            compliance_context = task_context.copy()
            compliance_context["extracted_text"] = extracted_text
            compliance_result = compliance_agent.process_document(file_path, compliance_context)
        
        # Step 5: Bridge extraction
        bridge_agent = self._find_agent_by_type(BridgeAgent)
        bridge_result = None
        
        if bridge_agent:
            bridge_context = task_context.copy()
            bridge_context["extracted_text"] = extracted_text
            bridge_context["entities"] = entities
            bridge_context["provider"] = ocr_agent.name.lower()
            bridge_result = bridge_agent.process_document(file_path, bridge_context)
        
        # Step 6: Compile results
        return self._compile_sequential_results(
            file_path, ocr_result, compliance_result, bridge_result, 
            enable_comparison, processing_config
        )
    
    def _process_parallel(
        self,
        file_path: Path,
        primary_agent: Optional[str],
        enable_comparison: bool,
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Process document in parallel with multiple agents"""
        
        # Create tasks for parallel execution
        tasks = []
        task_context = self._create_initial_context(file_path, processing_config)
        
        # OCR agents
        for agent in self.ocr_agents:
            tasks.append(self._async_process_agent(agent, file_path, task_context))
        
        # Run parallel processing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        ocr_results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        # Select best OCR result
        best_ocr_result = self._select_best_ocr_result(ocr_results)
        
        # Continue with specialized agents using best OCR result
        if best_ocr_result and best_ocr_result.get("success"):
            extracted_text = best_ocr_result["result"].get("text", "")
            
            # Process with specialized agents
            specialized_results = self._process_specialized_agents(
                file_path, extracted_text, task_context
            )
            
            return self._compile_parallel_results(
                file_path, ocr_results, specialized_results, processing_config
            )
        else:
            raise RuntimeError("All OCR agents failed")
    
    def _process_consensus(
        self,
        file_path: Path,
        primary_agent: Optional[str],
        enable_comparison: bool,
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Process document using consensus from multiple agents"""
        
        # Run parallel processing first
        parallel_result = self._process_parallel(
            file_path, primary_agent, enable_comparison, processing_config
        )
        
        # Apply consensus algorithms to merge results
        consensus_text = self._build_consensus_text(parallel_result)
        consensus_entities = self._build_consensus_entities(parallel_result)
        consensus_bridges = self._build_consensus_bridges(parallel_result)
        
        # Update result with consensus
        parallel_result.extracted_text = consensus_text
        # Update other fields as needed
        
        return parallel_result
    
    def _process_best_agent(
        self,
        file_path: Path,
        primary_agent: Optional[str],
        enable_comparison: bool,
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Process using the best available agent for the task"""
        
        # Determine best agent based on historical performance or configuration
        best_agent = self._determine_best_agent(file_path, primary_agent)
        
        if not best_agent:
            raise ValueError("No suitable agent found")
        
        # Process with the best agent
        task_context = self._create_initial_context(file_path, processing_config)
        result = best_agent.process_document(file_path, task_context)
        
        # Continue with specialized processing
        if result.get("success"):
            extracted_text = result["result"].get("text", "")
            specialized_results = self._process_specialized_agents(
                file_path, extracted_text, task_context
            )
            
            return self._compile_best_agent_results(
                file_path, result, specialized_results, processing_config
            )
        else:
            raise RuntimeError(f"Best agent processing failed: {result.get('error')}")
    
    async def _async_process_agent(
        self,
        agent: BaseAgent,
        file_path: Path,
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document with agent asynchronously"""
        try:
            return agent.process_document(file_path, task_context)
        except Exception as e:
            return {
                "success": False,
                "agent": agent.name,
                "error": str(e)
            }
    
    def _select_primary_ocr_agent(self, primary_agent: Optional[str]) -> Optional[BaseAgent]:
        """Select primary OCR agent"""
        
        if primary_agent:
            for agent in self.ocr_agents:
                if agent.name.lower() == primary_agent.lower():
                    return agent
        
        # Default to first available OCR agent
        return self.ocr_agents[0] if self.ocr_agents else None
    
    def _find_agent_by_type(self, agent_type: type) -> Optional[BaseAgent]:
        """Find agent by type"""
        for agent in self.specialized_agents:
            if isinstance(agent, agent_type):
                return agent
        return None
    
    def _create_initial_context(
        self,
        file_path: Path,
        processing_config: Optional[ProcessingConfig]
    ) -> Dict[str, Any]:
        """Create initial task context"""
        
        config = processing_config or ProcessingConfig()
        
        return {
            "file_path": str(file_path),
            "file_extension": file_path.suffix,
            "task_type": "full_processing",
            "compliance_profile": config.jurisdiction.value if config.jurisdiction else "eu_gdpr",
            "redaction_level": config.redaction_level.value,
            "enable_bridge_extraction": config.enable_bridge_extraction,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_specialized_agents(
        self,
        file_path: Path,
        extracted_text: str,
        base_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process with specialized agents"""
        
        results = {}
        
        # Compliance agent
        compliance_agent = self._find_agent_by_type(ComplianceAgent)
        if compliance_agent:
            compliance_context = base_context.copy()
            compliance_context["extracted_text"] = extracted_text
            results["compliance"] = compliance_agent.process_document(file_path, compliance_context)
        
        # Bridge agent
        bridge_agent = self._find_agent_by_type(BridgeAgent)
        if bridge_agent:
            bridge_context = base_context.copy()
            bridge_context["extracted_text"] = extracted_text
            results["bridge"] = bridge_agent.process_document(file_path, bridge_context)
        
        return results
    
    def _select_best_ocr_result(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best OCR result from multiple agents"""
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        
        if not successful_results:
            return None
        
        # Score results based on confidence and text length
        scored_results = []
        for result in successful_results:
            confidence = result.get("confidence", 0.0)
            text_length = len(result.get("result", {}).get("text", ""))
            score = confidence * 0.7 + min(1.0, text_length / 1000) * 0.3
            scored_results.append((score, result))
        
        # Return best result
        return max(scored_results, key=lambda x: x[0])[1]
    
    def _compile_sequential_results(
        self,
        file_path: Path,
        ocr_result: Dict[str, Any],
        compliance_result: Optional[Dict[str, Any]],
        bridge_result: Optional[Dict[str, Any]],
        enable_comparison: bool,
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Compile results from sequential processing"""
        
        # Extract data from results
        ocr_data = ocr_result.get("result", {})
        compliance_data = compliance_result.get("result", {}) if compliance_result else {}
        bridge_data = bridge_result.get("result", {}) if bridge_result else {}
        
        # Create compliance metadata
        compliance_metadata = self._create_compliance_metadata(compliance_data, file_path)
        
        # Create processing result
        return ProcessingResult(
            document_id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_filename=file_path.name,
            extracted_text=compliance_data.get("processed_text", ocr_data.get("text", "")),
            raw_text=ocr_data.get("text", "") if compliance_data.get("processed_text") else None,
            tables=ocr_data.get("tables", []),
            bridges=self._convert_bridges(bridge_data.get("bridges", [])),
            compliance_metadata=compliance_metadata,
            processing_config=processing_config or ProcessingConfig(),
            status=ProcessingStatus.COMPLETED,
            processing_time_seconds=sum([
                ocr_result.get("processing_time", 0),
                compliance_result.get("processing_time", 0) if compliance_result else 0,
                bridge_result.get("processing_time", 0) if bridge_result else 0
            ]),
            provider_used=ocr_result.get("agent", "unknown")
        )
    
    def _compile_parallel_results(
        self,
        file_path: Path,
        ocr_results: List[Dict[str, Any]],
        specialized_results: Dict[str, Any],
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Compile results from parallel processing"""
        
        # Get best OCR result
        best_ocr = self._select_best_ocr_result(ocr_results)
        
        if not best_ocr:
            raise RuntimeError("No successful OCR results")
        
        # Continue with compilation similar to sequential
        return self._compile_sequential_results(
            file_path, best_ocr, 
            specialized_results.get("compliance"),
            specialized_results.get("bridge"),
            False, processing_config
        )
    
    def _compile_best_agent_results(
        self,
        file_path: Path,
        primary_result: Dict[str, Any],
        specialized_results: Dict[str, Any],
        processing_config: Optional[ProcessingConfig]
    ) -> ProcessingResult:
        """Compile results from best agent processing"""
        
        return self._compile_sequential_results(
            file_path, primary_result,
            specialized_results.get("compliance"),
            specialized_results.get("bridge"),
            False, processing_config
        )
    
    def _create_compliance_metadata(
        self,
        compliance_data: Dict[str, Any],
        file_path: Path
    ) -> ComplianceMetadata:
        """Create compliance metadata from agent results"""
        
        from ..models import DocumentType, Jurisdiction, RedactionLevel
        
        # Extract or use defaults
        doc_classification = compliance_data.get("document_classification", {})
        jurisdiction_data = compliance_data.get("jurisdiction", {})
        
        return ComplianceMetadata(
            document_type=DocumentType(doc_classification.get("type", "unknown")),
            jurisdiction=Jurisdiction(jurisdiction_data.get("detected", "unknown")),
            redaction_level=RedactionLevel.MODERATE,
            redactions_count=compliance_data.get("pii_analysis", {}).get("count", 0),
            legal_basis=compliance_data.get("legal_basis", "consent_required"),
            audit_reference=f"COORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processor_version="2.0.0",
            compliance_score=compliance_data.get("compliance_metadata", {}).get("compliance_score", 0.7),
            risk_flags=compliance_data.get("pii_analysis", {}).get("risk_flags", [])
        )
    
    def _convert_bridges(self, bridge_data: List[Dict[str, Any]]) -> List:
        """Convert bridge dictionaries to Bridge objects"""
        from ..models import Bridge, EntityType, PrivacyImpact
        
        bridges = []
        for bridge_dict in bridge_data:
            try:
                bridge = Bridge(
                    entity_1=bridge_dict["entity_1"],
                    entity_2=bridge_dict["entity_2"],
                    entity_1_type=EntityType(bridge_dict["entity_1_type"]),
                    entity_2_type=EntityType(bridge_dict["entity_2_type"]),
                    relationship=bridge_dict["relationship"],
                    confidence_score=bridge_dict["confidence_score"],
                    privacy_impact=PrivacyImpact(bridge_dict["privacy_impact"]),
                    legal_basis=bridge_dict.get("legal_basis", ""),
                    extraction_method=bridge_dict.get("extraction_method", "agent_coordination")
                )
                bridges.append(bridge)
            except Exception as e:
                logger.warning(f"Failed to convert bridge: {e}")
        
        return bridges
    
    def _determine_best_agent(
        self,
        file_path: Path,
        primary_agent: Optional[str]
    ) -> Optional[BaseAgent]:
        """Determine best agent for the task"""
        
        if primary_agent:
            for agent in self.ocr_agents:
                if agent.name.lower() == primary_agent.lower():
                    return agent
        
        # Choose based on performance metrics
        best_agent = None
        best_score = 0.0
        
        for agent in self.ocr_agents:
            performance = agent.get_performance_summary()
            
            # Simple scoring: success rate * average confidence
            score = performance["success_rate"] * performance["average_confidence"]
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent or (self.ocr_agents[0] if self.ocr_agents else None)
    
    def _build_consensus_text(self, result: ProcessingResult) -> str:
        """Build consensus text from multiple agent results"""
        # Placeholder for consensus algorithm
        return result.extracted_text
    
    def _build_consensus_entities(self, result: ProcessingResult) -> List[Dict[str, Any]]:
        """Build consensus entities from multiple agent results"""
        # Placeholder for consensus algorithm
        return []
    
    def _build_consensus_bridges(self, result: ProcessingResult) -> List:
        """Build consensus bridges from multiple agent results"""
        # Placeholder for consensus algorithm
        return result.bridges
    
    def _create_error_result(
        self,
        file_path: Path,
        error_message: str,
        start_time: datetime
    ) -> ProcessingResult:
        """Create error result when processing fails"""
        
        from ..models import ComplianceMetadata, DocumentType, Jurisdiction, RedactionLevel
        
        return ProcessingResult(
            document_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
                audit_reference=f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                processor_version="2.0.0",
                compliance_score=0.0
            ),
            processing_config=ProcessingConfig(),
            status=ProcessingStatus.FAILED,
            error_message=error_message,
            processing_time_seconds=(datetime.now() - start_time).total_seconds(),
            provider_used="coordinator"
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        status = {
            "total_agents": len(self.agents),
            "ocr_agents": len(self.ocr_agents),
            "specialized_agents": len(self.specialized_agents),
            "agent_details": []
        }
        
        for agent in self.agents:
            performance = agent.get_performance_summary()
            status["agent_details"].append({
                "name": agent.name,
                "type": type(agent).__name__,
                "capabilities": agent.capabilities,
                "performance": performance
            })
        
        return status