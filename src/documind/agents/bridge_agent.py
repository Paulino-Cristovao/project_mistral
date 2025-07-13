"""
Bridge agent for intelligent entity relationship extraction
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import networkx as nx

from .base_agent import BaseAgent
from ..models import Bridge, EntityType, PrivacyImpact
from ..bridges.extractor import BridgeExtractor
from ..bridges.analyzer import RelationshipAnalyzer

logger = logging.getLogger(__name__)


class BridgeAgent(BaseAgent):
    """
    Specialized agent for entity relationship extraction and analysis
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize bridge agent
        
        Args:
            confidence_threshold: Minimum confidence for relationship extraction
            config: Agent configuration
        """
        capabilities = [
            "entity_extraction",
            "relationship_mapping",
            "semantic_analysis",
            "network_analysis",
            "privacy_assessment",
            "bridge_validation",
            "graph_construction",
            "centrality_analysis"
        ]
        
        super().__init__(
            name="Bridge",
            api_key="internal",  # No external API needed
            config=config or {},
            capabilities=capabilities
        )
        
        self.confidence_threshold = confidence_threshold
        
        # Initialize bridge extraction components
        self.bridge_extractor = BridgeExtractor(self.config)
        self.relationship_analyzer = RelationshipAnalyzer()
        
        # Relationship patterns and rules
        self.relationship_patterns = self._load_relationship_patterns()
        self.privacy_rules = self._load_privacy_rules()
        
        logger.info(f"Bridge agent initialized with confidence threshold: {confidence_threshold}")
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities
    
    def process_document(
        self, 
        file_path: Union[str, Path],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and analyze entity relationships
        
        Args:
            file_path: Path to document file
            task_context: Task context with extracted content
            
        Returns:
            Bridge extraction and analysis results
        """
        start_time = datetime.now()
        
        try:
            # Extract text and entities from context
            text_content = task_context.get("extracted_text", "")
            entities = task_context.get("entities", [])
            provider = task_context.get("provider", "mistral")
            
            if not text_content:
                logger.warning("No text content provided for bridge extraction")
                return self._create_empty_result()
            
            # Step 1: Extract bridges using the bridge extractor
            if entities:
                bridges = self._extract_bridges_from_entities(entities, text_content, provider)
            else:
                bridges = self.bridge_extractor.extract(
                    text_content, provider, self.confidence_threshold
                )
            
            # Step 2: Validate and enhance bridges
            validated_bridges = self._validate_bridges(bridges, text_content)
            
            # Step 3: Assess privacy impact for each bridge
            privacy_assessed_bridges = self._assess_bridge_privacy(validated_bridges)
            
            # Step 4: Perform network analysis
            network_analysis = self._perform_network_analysis(privacy_assessed_bridges)
            
            # Step 5: Generate relationship insights
            insights = self._generate_relationship_insights(
                privacy_assessed_bridges, network_analysis
            )
            
            # Step 6: Create relationship graph
            graph_data = self._create_graph_representation(privacy_assessed_bridges)
            
            # Calculate confidence
            confidence = self._calculate_overall_confidence(privacy_assessed_bridges)
            
            # Record execution
            self.record_execution(start_time, True, confidence)
            
            result = {
                "bridges": [bridge.model_dump() for bridge in privacy_assessed_bridges],
                "bridge_count": len(privacy_assessed_bridges),
                "network_analysis": network_analysis,
                "relationship_insights": insights,
                "graph_representation": graph_data,
                "privacy_summary": self._summarize_privacy_impact(privacy_assessed_bridges),
                "quality_metrics": self._calculate_quality_metrics(privacy_assessed_bridges)
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
    
    def _load_relationship_patterns(self) -> Dict[str, Any]:
        """Load relationship patterns for entity linking"""
        return {
            "high_confidence_patterns": [
                ("PERSON", "ORG", ["employed_by", "member_of", "affiliated_with"]),
                ("PERSON", "MONEY", ["payment_to", "salary_of", "owns"]),
                ("ORG", "LOCATION", ["based_in", "operates_in", "located_at"]),
                ("CONTRACT_ID", "PERSON", ["signatory", "party_to", "beneficiary"]),
                ("CONTRACT_ID", "ORG", ["contracting_party", "vendor", "client"])
            ],
            "contextual_patterns": [
                ("PERSON", "DATE", ["associated_with", "active_on", "employed_from"]),
                ("ORG", "PRODUCT", ["manufactures", "sells", "distributes"]),
                ("LOCATION", "EVENT", ["hosted", "location_of", "venue_for"])
            ],
            "temporal_patterns": [
                ("PERSON", "ORG", "DATE", "employment_period"),
                ("CONTRACT_ID", "PERSON", "DATE", "signing_date"),
                ("ORG", "LOCATION", "DATE", "establishment_date")
            ]
        }
    
    def _load_privacy_rules(self) -> Dict[str, Any]:
        """Load privacy assessment rules"""
        return {
            "high_risk_combinations": [
                ("PERSON", "MONEY"),
                ("PERSON", "LOCATION"),
                ("PERSON", "MEDICAL"),
                ("PERSON", "SENSITIVE_DATA")
            ],
            "medium_risk_combinations": [
                ("PERSON", "ORG"),
                ("PERSON", "DATE"),
                ("ORG", "FINANCIAL_DATA"),
                ("LOCATION", "PERSONAL_DATA")
            ],
            "privacy_impact_rules": {
                PrivacyImpact.CRITICAL: ["health_data", "biometric_data", "genetic_data"],
                PrivacyImpact.HIGH: ["financial_data", "identity_documents", "location_tracking"],
                PrivacyImpact.MEDIUM: ["employment_data", "contact_information", "preferences"],
                PrivacyImpact.LOW: ["public_information", "business_data", "anonymized_data"]
            }
        }
    
    def _extract_bridges_from_entities(
        self,
        entities: List[Dict[str, Any]],
        text_content: str,
        provider: str
    ) -> List[Bridge]:
        """Extract bridges from pre-identified entities"""
        
        bridges = []
        
        # Convert entities to standardized format
        standardized_entities = []
        for entity in entities:
            standardized_entities.append({
                "text": entity.get("text", ""),
                "entity_type": EntityType(entity.get("type", "other")),
                "confidence": entity.get("confidence", 0.8),
                "context": self._get_entity_context(entity["text"], text_content)
            })
        
        # Extract relationships between entity pairs
        for i, entity1 in enumerate(standardized_entities):
            for entity2 in standardized_entities[i+1:]:
                
                # Calculate semantic similarity if provider available
                similarity = self._calculate_entity_similarity(
                    entity1, entity2, text_content, provider
                )
                
                if similarity >= self.confidence_threshold:
                    
                    # Determine relationship type
                    relationship_type = self._determine_relationship_type(
                        entity1, entity2, text_content
                    )
                    
                    # Assess privacy impact
                    privacy_impact = self._assess_entity_pair_privacy(entity1, entity2)
                    
                    # Create bridge
                    bridge = Bridge(
                        entity_1=entity1["text"],
                        entity_2=entity2["text"],
                        entity_1_type=entity1["entity_type"],
                        entity_2_type=entity2["entity_type"],
                        relationship=relationship_type,
                        confidence_score=similarity,
                        privacy_impact=privacy_impact,
                        legal_basis=self._determine_legal_basis(privacy_impact),
                        extraction_method="bridge_agent_entity_analysis"
                    )
                    
                    bridges.append(bridge)
        
        return bridges
    
    def _get_entity_context(self, entity_text: str, full_text: str) -> str:
        """Get contextual text around an entity"""
        
        # Find entity position
        entity_pos = full_text.lower().find(entity_text.lower())
        if entity_pos == -1:
            return entity_text
        
        # Extract context (50 chars before and after)
        start = max(0, entity_pos - 50)
        end = min(len(full_text), entity_pos + len(entity_text) + 50)
        
        return full_text[start:end]
    
    def _calculate_entity_similarity(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        text_content: str,
        provider: str
    ) -> float:
        """Calculate semantic similarity between entities"""
        
        # Use context and relationship patterns for similarity
        context1 = entity1.get("context", entity1["text"])
        context2 = entity2.get("context", entity2["text"])
        
        # Check for direct relationship patterns
        pattern_score = self._check_relationship_patterns(entity1, entity2, text_content)
        
        # Check for proximity in text
        proximity_score = self._calculate_proximity_score(
            entity1["text"], entity2["text"], text_content
        )
        
        # Combine scores
        combined_score = (pattern_score * 0.6 + proximity_score * 0.4)
        
        return combined_score
    
    def _check_relationship_patterns(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        text_content: str
    ) -> float:
        """Check for known relationship patterns"""
        
        type1 = entity1["entity_type"]
        type2 = entity2["entity_type"]
        
        # Check high confidence patterns
        high_patterns = self.relationship_patterns["high_confidence_patterns"]
        for pattern_type1, pattern_type2, relationships in high_patterns:
            if ((type1.value == pattern_type1.lower() and type2.value == pattern_type2.lower()) or
                (type1.value == pattern_type2.lower() and type2.value == pattern_type1.lower())):
                
                # Check if any relationship keywords are nearby
                entity1_text = entity1["text"].lower()
                entity2_text = entity2["text"].lower()
                
                for relationship in relationships:
                    # Simple proximity check for relationship words
                    if relationship.replace("_", " ") in text_content.lower():
                        # Check if entities are mentioned near the relationship word
                        rel_pos = text_content.lower().find(relationship.replace("_", " "))
                        entity1_pos = text_content.lower().find(entity1_text)
                        entity2_pos = text_content.lower().find(entity2_text)
                        
                        if (abs(rel_pos - entity1_pos) < 100 and 
                            abs(rel_pos - entity2_pos) < 100):
                            return 0.9
                
                return 0.7  # Pattern match without explicit relationship word
        
        # Check contextual patterns
        contextual_patterns = self.relationship_patterns["contextual_patterns"]
        for pattern_type1, pattern_type2, relationships in contextual_patterns:
            if ((type1.value == pattern_type1.lower() and type2.value == pattern_type2.lower()) or
                (type1.value == pattern_type2.lower() and type2.value == pattern_type1.lower())):
                return 0.6
        
        return 0.3  # Default low score
    
    def _calculate_proximity_score(
        self,
        entity1_text: str,
        entity2_text: str,
        full_text: str
    ) -> float:
        """Calculate proximity score based on text distance"""
        
        entity1_pos = full_text.lower().find(entity1_text.lower())
        entity2_pos = full_text.lower().find(entity2_text.lower())
        
        if entity1_pos == -1 or entity2_pos == -1:
            return 0.1
        
        distance = abs(entity1_pos - entity2_pos)
        
        # Closer entities have higher scores
        if distance < 50:
            return 0.9
        elif distance < 100:
            return 0.7
        elif distance < 200:
            return 0.5
        elif distance < 500:
            return 0.3
        else:
            return 0.1
    
    def _determine_relationship_type(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        text_content: str
    ) -> str:
        """Determine the type of relationship between entities"""
        
        type1 = entity1["entity_type"]
        type2 = entity2["entity_type"]
        
        # Use the same logic as BridgeExtractor but with enhanced patterns
        if type1 == EntityType.PERSON and type2 == EntityType.ORGANIZATION:
            return "employed_by"
        elif type1 == EntityType.PERSON and type2 == EntityType.MONEY:
            return "financial_relationship"
        elif type1 == EntityType.ORGANIZATION and type2 == EntityType.LOCATION:
            return "based_in"
        elif type1 == EntityType.CONTRACT_ID and type2 == EntityType.PERSON:
            return "signatory"
        elif type1 == EntityType.PERSON and type2 == EntityType.DATE:
            return "temporal_association"
        else:
            return "related_to"
    
    def _assess_entity_pair_privacy(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> PrivacyImpact:
        """Assess privacy impact of entity pair"""
        
        type1 = entity1["entity_type"]
        type2 = entity2["entity_type"]
        
        # Check high-risk combinations
        high_risk = self.privacy_rules["high_risk_combinations"]
        for risk_type1, risk_type2 in high_risk:
            if ((type1.value == risk_type1.lower() and type2.value == risk_type2.lower()) or
                (type1.value == risk_type2.lower() and type2.value == risk_type1.lower())):
                return PrivacyImpact.HIGH
        
        # Check medium-risk combinations
        medium_risk = self.privacy_rules["medium_risk_combinations"]
        for risk_type1, risk_type2 in medium_risk:
            if ((type1.value == risk_type1.lower() and type2.value == risk_type2.lower()) or
                (type1.value == risk_type2.lower() and type2.value == risk_type1.lower())):
                return PrivacyImpact.MEDIUM
        
        return PrivacyImpact.LOW
    
    def _determine_legal_basis(self, privacy_impact: PrivacyImpact) -> str:
        """Determine legal basis based on privacy impact"""
        
        if privacy_impact == PrivacyImpact.CRITICAL:
            return "explicit_consent_required"
        elif privacy_impact == PrivacyImpact.HIGH:
            return "consent_or_legal_obligation"
        elif privacy_impact == PrivacyImpact.MEDIUM:
            return "legitimate_interest"
        else:
            return "contract_performance"
    
    def _validate_bridges(self, bridges: List[Bridge], text_content: str) -> List[Bridge]:
        """Validate and filter bridges based on quality criteria"""
        
        validated_bridges = []
        
        for bridge in bridges:
            
            # Check confidence threshold
            if bridge.confidence_score < self.confidence_threshold:
                continue
            
            # Check for meaningful relationships (not just entity co-occurrence)
            if self._is_meaningful_relationship(bridge, text_content):
                validated_bridges.append(bridge)
        
        # Remove duplicate relationships
        deduplicated_bridges = self._deduplicate_bridges(validated_bridges)
        
        return deduplicated_bridges
    
    def _is_meaningful_relationship(self, bridge: Bridge, text_content: str) -> bool:
        """Check if relationship is meaningful beyond co-occurrence"""
        
        # Check if entities appear in same sentence
        sentences = text_content.split('.')
        same_sentence = False
        
        for sentence in sentences:
            if (bridge.entity_1.lower() in sentence.lower() and 
                bridge.entity_2.lower() in sentence.lower()):
                same_sentence = True
                break
        
        # Meaningful if in same sentence and high confidence
        return same_sentence and bridge.confidence_score >= 0.7
    
    def _deduplicate_bridges(self, bridges: List[Bridge]) -> List[Bridge]:
        """Remove duplicate bridges"""
        
        seen_pairs = set()
        deduplicated = []
        
        for bridge in bridges:
            
            # Create normalized pair key
            pair_key = tuple(sorted([bridge.entity_1.lower(), bridge.entity_2.lower()]))
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                deduplicated.append(bridge)
        
        return deduplicated
    
    def _assess_bridge_privacy(self, bridges: List[Bridge]) -> List[Bridge]:
        """Enhance privacy assessment for bridges"""
        
        enhanced_bridges = []
        
        for bridge in bridges:
            
            # Re-assess privacy impact with additional context
            enhanced_impact = self._enhanced_privacy_assessment(bridge)
            
            # Update bridge with enhanced assessment
            enhanced_bridge = Bridge(
                entity_1=bridge.entity_1,
                entity_2=bridge.entity_2,
                entity_1_type=bridge.entity_1_type,
                entity_2_type=bridge.entity_2_type,
                relationship=bridge.relationship,
                confidence_score=bridge.confidence_score,
                privacy_impact=enhanced_impact,
                legal_basis=self._determine_legal_basis(enhanced_impact),
                source_document=bridge.source_document,
                extraction_method=bridge.extraction_method
            )
            
            enhanced_bridges.append(enhanced_bridge)
        
        return enhanced_bridges
    
    def _enhanced_privacy_assessment(self, bridge: Bridge) -> PrivacyImpact:
        """Enhanced privacy impact assessment"""
        
        # Start with existing assessment
        current_impact = bridge.privacy_impact
        
        # Enhance based on relationship type
        high_risk_relationships = [
            "financial_relationship", "payment_to", "employed_by", 
            "medical_relationship", "location_tracking"
        ]
        
        if bridge.relationship in high_risk_relationships:
            if current_impact == PrivacyImpact.LOW:
                return PrivacyImpact.MEDIUM
            elif current_impact == PrivacyImpact.MEDIUM:
                return PrivacyImpact.HIGH
        
        return current_impact
    
    def _perform_network_analysis(self, bridges: List[Bridge]) -> Dict[str, Any]:
        """Perform network analysis on bridges"""
        
        if not bridges:
            return {"node_count": 0, "edge_count": 0}
        
        # Build graph
        graph = nx.Graph()
        
        for bridge in bridges:
            graph.add_edge(
                bridge.entity_1, 
                bridge.entity_2,
                relationship=bridge.relationship,
                confidence=bridge.confidence_score,
                privacy_impact=bridge.privacy_impact.value
            )
        
        # Calculate network metrics
        analysis = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
            "connected_components": nx.number_connected_components(graph),
            "clustering_coefficient": nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0.0
        }
        
        # Find central nodes
        if graph.number_of_nodes() > 1:
            centrality = nx.degree_centrality(graph)
            analysis["most_central_entity"] = max(centrality, key=centrality.get)
            analysis["centrality_scores"] = dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return analysis
    
    def _generate_relationship_insights(
        self,
        bridges: List[Bridge],
        network_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights about relationships"""
        
        if not bridges:
            return {}
        
        # Analyze relationship patterns
        relationship_types = {}
        privacy_distribution = {}
        entity_frequency = {}
        
        for bridge in bridges:
            # Count relationship types
            rel_type = bridge.relationship
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            # Count privacy impacts
            privacy_level = bridge.privacy_impact.value
            privacy_distribution[privacy_level] = privacy_distribution.get(privacy_level, 0) + 1
            
            # Count entity frequencies
            entity_frequency[bridge.entity_1] = entity_frequency.get(bridge.entity_1, 0) + 1
            entity_frequency[bridge.entity_2] = entity_frequency.get(bridge.entity_2, 0) + 1
        
        # Find most connected entities
        most_connected = sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_relationships": len(bridges),
            "relationship_type_distribution": relationship_types,
            "privacy_impact_distribution": privacy_distribution,
            "most_connected_entities": dict(most_connected),
            "network_complexity": network_analysis.get("clustering_coefficient", 0.0),
            "data_flow_intensity": network_analysis.get("density", 0.0)
        }
    
    def _create_graph_representation(self, bridges: List[Bridge]) -> Dict[str, Any]:
        """Create graph representation for visualization"""
        
        nodes = []
        edges = []
        node_set = set()
        
        for bridge in bridges:
            
            # Add nodes
            if bridge.entity_1 not in node_set:
                nodes.append({
                    "id": bridge.entity_1,
                    "label": bridge.entity_1,
                    "type": bridge.entity_1_type.value,
                    "size": 10
                })
                node_set.add(bridge.entity_1)
            
            if bridge.entity_2 not in node_set:
                nodes.append({
                    "id": bridge.entity_2,
                    "label": bridge.entity_2,
                    "type": bridge.entity_2_type.value,
                    "size": 10
                })
                node_set.add(bridge.entity_2)
            
            # Add edge
            edges.append({
                "source": bridge.entity_1,
                "target": bridge.entity_2,
                "relationship": bridge.relationship,
                "confidence": bridge.confidence_score,
                "privacy_impact": bridge.privacy_impact.value,
                "weight": bridge.confidence_score
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "force-directed"
        }
    
    def _summarize_privacy_impact(self, bridges: List[Bridge]) -> Dict[str, Any]:
        """Summarize privacy impact across all bridges"""
        
        if not bridges:
            return {"total_bridges": 0, "risk_summary": "no_data"}
        
        impact_counts = {impact.value: 0 for impact in PrivacyImpact}
        
        for bridge in bridges:
            impact_counts[bridge.privacy_impact.value] += 1
        
        # Determine overall risk level
        total_bridges = len(bridges)
        high_risk_ratio = (impact_counts["high"] + impact_counts["critical"]) / total_bridges
        
        if high_risk_ratio >= 0.5:
            risk_summary = "high_risk"
        elif high_risk_ratio >= 0.3:
            risk_summary = "moderate_risk"
        else:
            risk_summary = "low_risk"
        
        return {
            "total_bridges": total_bridges,
            "impact_distribution": impact_counts,
            "high_risk_ratio": high_risk_ratio,
            "risk_summary": risk_summary
        }
    
    def _calculate_quality_metrics(self, bridges: List[Bridge]) -> Dict[str, float]:
        """Calculate quality metrics for extracted bridges"""
        
        if not bridges:
            return {"average_confidence": 0.0, "quality_score": 0.0}
        
        confidences = [bridge.confidence_score for bridge in bridges]
        average_confidence = sum(confidences) / len(confidences)
        
        # Quality score based on confidence and meaningful relationships
        high_confidence_count = sum(1 for conf in confidences if conf >= 0.8)
        quality_score = (high_confidence_count / len(bridges)) * average_confidence
        
        return {
            "average_confidence": average_confidence,
            "high_confidence_ratio": high_confidence_count / len(bridges),
            "quality_score": quality_score,
            "total_bridges": len(bridges)
        }
    
    def _calculate_overall_confidence(self, bridges: List[Bridge]) -> float:
        """Calculate overall confidence in bridge extraction"""
        
        if not bridges:
            return 0.0
        
        # Average confidence weighted by quality
        confidences = [bridge.confidence_score for bridge in bridges]
        average_confidence = sum(confidences) / len(confidences)
        
        # Adjust for number of bridges (more bridges = higher confidence in process)
        bridge_count_factor = min(1.0, len(bridges) / 10)
        
        return average_confidence * (0.7 + 0.3 * bridge_count_factor)
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no content is available"""
        return {
            "success": True,
            "agent": self.name,
            "result": {
                "bridges": [],
                "bridge_count": 0,
                "network_analysis": {"node_count": 0, "edge_count": 0},
                "relationship_insights": {},
                "graph_representation": {"nodes": [], "edges": []},
                "privacy_summary": {"total_bridges": 0, "risk_summary": "no_data"},
                "quality_metrics": {"average_confidence": 0.0, "quality_score": 0.0}
            },
            "confidence": 0.0,
            "processing_time": 0.0
        }