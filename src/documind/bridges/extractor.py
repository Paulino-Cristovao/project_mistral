"""
Bridge extraction system for entity relationships
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import networkx as nx
import spacy
from spacy.lang.en import English

from ..models import Bridge, EntityType, PrivacyImpact
from ..providers.factory import OCRProviderFactory

logger = logging.getLogger(__name__)


class BridgeExtractor:
    """
    Extracts relationships (bridges) between entities in documents
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bridge extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.entity_types = config.get("entity_types", {})
        
        # Initialize spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found, using basic NER")
            self.nlp = English()
        
        # Initialize relationship graph
        self.relationship_graph = nx.Graph()
        
        logger.info("Bridge extractor initialized")
    
    def extract(
        self,
        text: str,
        provider: str = "mistral",
        min_confidence: float = 0.7
    ) -> List[Bridge]:
        """
        Extract entity relationships from text
        
        Args:
            text: Input text to analyze
            provider: OCR provider for embeddings ('openai' or 'mistral')
            min_confidence: Minimum confidence threshold for relationships
            
        Returns:
            List of extracted bridges (relationships)
        """
        if not text or len(text.strip()) < 10:
            return []
        
        try:
            # Step 1: Extract entities using spaCy
            entities = self._extract_entities(text)
            
            if len(entities) < 2:
                logger.info("Not enough entities found for bridge extraction")
                return []
            
            # Step 2: Calculate entity relationships using embeddings
            bridges = self._calculate_relationships(
                entities, text, provider, min_confidence
            )
            
            # Step 3: Build relationship graph
            self._update_relationship_graph(bridges)
            
            logger.info(f"Extracted {len(bridges)} bridges from {len(entities)} entities")
            return bridges
            
        except Exception as e:
            logger.error(f"Bridge extraction failed: {e}")
            return []
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label_to_entity_type(ent.label_)
                if entity_type != EntityType.OTHER:  # Filter relevant entities
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "entity_type": entity_type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.8  # Default spaCy confidence
                    })
        
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        return entities
    
    def _calculate_relationships(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        provider: str,
        min_confidence: float
    ) -> List[Bridge]:
        """Calculate relationships between entities using embeddings"""
        bridges = []
        
        try:
            # Get embeddings for entity contexts
            entity_contexts = self._get_entity_contexts(entities, text)
            
            if not entity_contexts:
                return bridges
            
            # Get OCR provider for embeddings
            ocr_provider = OCRProviderFactory.create(provider)
            embeddings = ocr_provider.get_embeddings([ctx["context"] for ctx in entity_contexts])
            
            # Calculate pairwise relationships
            for i, entity1 in enumerate(entity_contexts):
                for j, entity2 in enumerate(entity_contexts[i+1:], i+1):
                    # Calculate semantic similarity
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    
                    if similarity >= min_confidence:
                        # Determine relationship type
                        relationship_type = self._determine_relationship_type(
                            entity1, entity2, text
                        )
                        
                        # Assess privacy impact
                        privacy_impact = self._assess_privacy_impact(
                            entity1["entity"], entity2["entity"]
                        )
                        
                        bridge = Bridge(
                            entity_1=entity1["entity"]["text"],
                            entity_2=entity2["entity"]["text"],
                            entity_1_type=entity1["entity"]["entity_type"],
                            entity_2_type=entity2["entity"]["entity_type"],
                            relationship=relationship_type,
                            confidence_score=float(similarity),
                            privacy_impact=privacy_impact,
                            legal_basis=self._determine_legal_basis(privacy_impact),
                            extraction_method="embedding_similarity"
                        )
                        
                        bridges.append(bridge)
        
        except Exception as e:
            logger.error(f"Relationship calculation failed: {e}")
        
        return bridges
    
    def _get_entity_contexts(
        self, 
        entities: List[Dict[str, Any]], 
        text: str
    ) -> List[Dict[str, Any]]:
        """Get contextual text around each entity for better relationship detection"""
        contexts = []
        
        for entity in entities:
            start = max(0, entity["start"] - 50)  # 50 chars before
            end = min(len(text), entity["end"] + 50)  # 50 chars after
            
            context = text[start:end].strip()
            
            contexts.append({
                "entity": entity,
                "context": context
            })
        
        return contexts
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _determine_relationship_type(
        self,
        entity1_ctx: Dict[str, Any],
        entity2_ctx: Dict[str, Any],
        full_text: str
    ) -> str:
        """Determine the type of relationship between two entities"""
        
        ent1_type = entity1_ctx["entity"]["entity_type"]
        ent2_type = entity2_ctx["entity"]["entity_type"]
        
        # Define relationship patterns based on entity types
        if ent1_type == EntityType.PERSON and ent2_type == EntityType.ORGANIZATION:
            return "employed_by"
        elif ent1_type == EntityType.PERSON and ent2_type == EntityType.MONEY:
            return "payment_to"
        elif ent1_type == EntityType.ORGANIZATION and ent2_type == EntityType.MONEY:
            return "financial_transaction"
        elif ent1_type == EntityType.PERSON and ent2_type == EntityType.LOCATION:
            return "located_at"
        elif ent1_type == EntityType.ORGANIZATION and ent2_type == EntityType.LOCATION:
            return "based_in"
        elif ent1_type == EntityType.PERSON and ent2_type == EntityType.DATE:
            return "associated_with_date"
        elif ent1_type == EntityType.CONTRACT_ID and ent2_type == EntityType.PERSON:
            return "signatory"
        elif ent1_type == EntityType.CONTRACT_ID and ent2_type == EntityType.ORGANIZATION:
            return "contracting_party"
        else:
            # Generic relationship
            return "related_to"
    
    def _assess_privacy_impact(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> PrivacyImpact:
        """Assess privacy impact of entity relationship"""
        
        type1 = entity1["entity_type"]
        type2 = entity2["entity_type"]
        
        # High privacy impact combinations
        high_impact_combinations = [
            (EntityType.PERSON, EntityType.MONEY),
            (EntityType.PERSON, EntityType.LOCATION),
            (EntityType.PERSON, EntityType.ORGANIZATION)
        ]
        
        # Medium privacy impact combinations
        medium_impact_combinations = [
            (EntityType.ORGANIZATION, EntityType.MONEY),
            (EntityType.ORGANIZATION, EntityType.LOCATION),
            (EntityType.PERSON, EntityType.DATE)
        ]
        
        combination = (type1, type2)
        reverse_combination = (type2, type1)
        
        if combination in high_impact_combinations or reverse_combination in high_impact_combinations:
            return PrivacyImpact.HIGH
        elif combination in medium_impact_combinations or reverse_combination in medium_impact_combinations:
            return PrivacyImpact.MEDIUM
        else:
            return PrivacyImpact.LOW
    
    def _determine_legal_basis(self, privacy_impact: PrivacyImpact) -> str:
        """Determine legal basis for processing relationship data"""
        if privacy_impact == PrivacyImpact.HIGH:
            return "consent_required"
        elif privacy_impact == PrivacyImpact.MEDIUM:
            return "legitimate_interest"
        else:
            return "contract_performance"
    
    def _map_spacy_label_to_entity_type(self, spacy_label: str) -> EntityType:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "MONEY": EntityType.MONEY,
            "DATE": EntityType.DATE,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.OTHER,
            "FAC": EntityType.LOCATION,
            "NORP": EntityType.OTHER,
            "WORK_OF_ART": EntityType.PRODUCT,
            "LAW": EntityType.OTHER,
            "LANGUAGE": EntityType.OTHER
        }
        
        return mapping.get(spacy_label, EntityType.OTHER)
    
    def _update_relationship_graph(self, bridges: List[Bridge]) -> None:
        """Update the relationship graph with new bridges"""
        for bridge in bridges:
            self.relationship_graph.add_edge(
                bridge.entity_1,
                bridge.entity_2,
                relationship=bridge.relationship,
                confidence=bridge.confidence_score,
                privacy_impact=bridge.privacy_impact.value
            )
    
    def get_relationship_graph(self) -> nx.Graph:
        """Get the current relationship graph"""
        return self.relationship_graph.copy()
    
    def find_entity_connections(self, entity: str, max_depth: int = 2) -> List[str]:
        """Find all entities connected to a given entity within max_depth"""
        if entity not in self.relationship_graph:
            return []
        
        connected = []
        
        # BFS to find connections within max_depth
        visited = {entity}
        queue = [(entity, 0)]
        
        while queue:
            current_entity, depth = queue.pop(0)
            
            if depth < max_depth:
                for neighbor in self.relationship_graph.neighbors(current_entity):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        connected.append(neighbor)
                        queue.append((neighbor, depth + 1))
        
        return connected
    
    def analyze_entity_centrality(self) -> Dict[str, float]:
        """Analyze entity centrality in the relationship graph"""
        if len(self.relationship_graph) == 0:
            return {}
        
        try:
            centrality = nx.betweenness_centrality(self.relationship_graph)
            return centrality
        except Exception as e:
            logger.warning(f"Centrality analysis failed: {e}")
            return {}