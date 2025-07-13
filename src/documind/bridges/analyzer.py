"""
Relationship analysis and visualization
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import pandas as pd

from ..models import Bridge, PrivacyImpact

logger = logging.getLogger(__name__)


class RelationshipAnalyzer:
    """
    Analyzes and visualizes entity relationships
    """
    
    def __init__(self):
        """Initialize relationship analyzer"""
        self.graph = nx.Graph()
        logger.info("Relationship analyzer initialized")
    
    def analyze_bridges(self, bridges: List[Bridge]) -> Dict[str, Any]:
        """
        Analyze a list of bridges to extract insights
        
        Args:
            bridges: List of Bridge objects to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not bridges:
            return {
                "total_bridges": 0,
                "entity_count": 0,
                "privacy_distribution": {},
                "relationship_types": {},
                "high_risk_entities": [],
                "network_metrics": {}
            }
        
        # Build graph from bridges
        self._build_graph_from_bridges(bridges)
        
        # Perform analysis
        analysis = {
            "total_bridges": len(bridges),
            "entity_count": self._count_unique_entities(bridges),
            "privacy_distribution": self._analyze_privacy_distribution(bridges),
            "relationship_types": self._analyze_relationship_types(bridges),
            "high_risk_entities": self._identify_high_risk_entities(bridges),
            "network_metrics": self._calculate_network_metrics(),
            "central_entities": self._find_central_entities(),
            "privacy_clusters": self._identify_privacy_clusters(bridges),
            "compliance_score": self._calculate_compliance_score(bridges)
        }
        
        return analysis
    
    def _build_graph_from_bridges(self, bridges: List[Bridge]) -> None:
        """Build NetworkX graph from bridges"""
        self.graph.clear()
        
        for bridge in bridges:
            self.graph.add_edge(
                bridge.entity_1,
                bridge.entity_2,
                relationship=bridge.relationship,
                confidence=bridge.confidence_score,
                privacy_impact=bridge.privacy_impact.value,
                legal_basis=bridge.legal_basis
            )
    
    def _count_unique_entities(self, bridges: List[Bridge]) -> int:
        """Count unique entities across all bridges"""
        entities = set()
        for bridge in bridges:
            entities.add(bridge.entity_1)
            entities.add(bridge.entity_2)
        return len(entities)
    
    def _analyze_privacy_distribution(self, bridges: List[Bridge]) -> Dict[str, int]:
        """Analyze distribution of privacy impact levels"""
        distribution = {impact.value: 0 for impact in PrivacyImpact}
        
        for bridge in bridges:
            distribution[bridge.privacy_impact.value] += 1
        
        return distribution
    
    def _analyze_relationship_types(self, bridges: List[Bridge]) -> Dict[str, int]:
        """Analyze distribution of relationship types"""
        types = {}
        
        for bridge in bridges:
            rel_type = bridge.relationship
            types[rel_type] = types.get(rel_type, 0) + 1
        
        return types
    
    def _identify_high_risk_entities(self, bridges: List[Bridge]) -> List[Dict[str, Any]]:
        """Identify entities involved in high-risk relationships"""
        entity_risk_scores = {}
        
        # Calculate risk scores for each entity
        for bridge in bridges:
            # Risk score based on privacy impact
            risk_value = {
                PrivacyImpact.LOW: 1,
                PrivacyImpact.MEDIUM: 3,
                PrivacyImpact.HIGH: 5,
                PrivacyImpact.CRITICAL: 10
            }.get(bridge.privacy_impact, 1)
            
            # Add risk to both entities
            entity_risk_scores[bridge.entity_1] = entity_risk_scores.get(bridge.entity_1, 0) + risk_value
            entity_risk_scores[bridge.entity_2] = entity_risk_scores.get(bridge.entity_2, 0) + risk_value
        
        # Sort by risk score and return top entities
        sorted_entities = sorted(
            entity_risk_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        high_risk_entities = []
        for entity, risk_score in sorted_entities[:10]:  # Top 10
            if risk_score >= 5:  # High risk threshold
                high_risk_entities.append({
                    "entity": entity,
                    "risk_score": risk_score,
                    "connection_count": self.graph.degree(entity) if entity in self.graph else 0
                })
        
        return high_risk_entities
    
    def _calculate_network_metrics(self) -> Dict[str, Any]:
        """Calculate network analysis metrics"""
        if len(self.graph) == 0:
            return {}
        
        try:
            metrics = {
                "node_count": self.graph.number_of_nodes(),
                "edge_count": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
                "is_connected": nx.is_connected(self.graph)
            }
            
            # Add connected components info
            if not metrics["is_connected"]:
                components = list(nx.connected_components(self.graph))
                metrics["connected_components"] = len(components)
                metrics["largest_component_size"] = len(max(components, key=len))
            
            # Add centrality measures for top nodes
            if len(self.graph) > 1:
                degree_centrality = nx.degree_centrality(self.graph)
                metrics["most_connected_entity"] = max(degree_centrality, key=degree_centrality.get)
                metrics["max_degree_centrality"] = max(degree_centrality.values())
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Network metrics calculation failed: {e}")
            return {}
    
    def _find_central_entities(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Find most central entities in the network"""
        if len(self.graph) <= 1:
            return []
        
        try:
            # Calculate different centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # Combine centrality measures
            entities = list(self.graph.nodes())
            central_entities = []
            
            for entity in entities:
                combined_score = (
                    degree_centrality.get(entity, 0) * 0.4 +
                    betweenness_centrality.get(entity, 0) * 0.4 +
                    closeness_centrality.get(entity, 0) * 0.2
                )
                
                central_entities.append({
                    "entity": entity,
                    "centrality_score": combined_score,
                    "degree_centrality": degree_centrality.get(entity, 0),
                    "betweenness_centrality": betweenness_centrality.get(entity, 0),
                    "closeness_centrality": closeness_centrality.get(entity, 0),
                    "connections": self.graph.degree(entity)
                })
            
            # Sort by combined centrality score
            central_entities.sort(key=lambda x: x["centrality_score"], reverse=True)
            
            return central_entities[:top_n]
            
        except Exception as e:
            logger.warning(f"Central entities calculation failed: {e}")
            return []
    
    def _identify_privacy_clusters(self, bridges: List[Bridge]) -> List[Dict[str, Any]]:
        """Identify clusters of entities with similar privacy characteristics"""
        if not bridges:
            return []
        
        # Group bridges by privacy impact
        privacy_groups = {}
        for bridge in bridges:
            impact = bridge.privacy_impact.value
            if impact not in privacy_groups:
                privacy_groups[impact] = []
            privacy_groups[impact].append(bridge)
        
        clusters = []
        for impact_level, group_bridges in privacy_groups.items():
            if len(group_bridges) >= 2:  # Minimum cluster size
                entities = set()
                for bridge in group_bridges:
                    entities.add(bridge.entity_1)
                    entities.add(bridge.entity_2)
                
                clusters.append({
                    "privacy_impact": impact_level,
                    "entity_count": len(entities),
                    "bridge_count": len(group_bridges),
                    "entities": list(entities)[:10]  # Limit for readability
                })
        
        # Sort by entity count (largest clusters first)
        clusters.sort(key=lambda x: x["entity_count"], reverse=True)
        
        return clusters
    
    def _calculate_compliance_score(self, bridges: List[Bridge]) -> float:
        """Calculate overall compliance score based on bridge analysis"""
        if not bridges:
            return 0.0
        
        total_score = 0.0
        
        for bridge in bridges:
            # Score based on privacy impact and legal basis
            privacy_penalty = {
                PrivacyImpact.LOW: 0.0,
                PrivacyImpact.MEDIUM: 0.1,
                PrivacyImpact.HIGH: 0.3,
                PrivacyImpact.CRITICAL: 0.5
            }.get(bridge.privacy_impact, 0.2)
            
            # Score based on legal basis
            legal_basis_score = 1.0
            if bridge.legal_basis == "consent_required":
                legal_basis_score = 0.7  # Requires additional consent management
            elif bridge.legal_basis == "legitimate_interest":
                legal_basis_score = 0.8
            
            # Bridge score
            bridge_score = (legal_basis_score - privacy_penalty) * bridge.confidence_score
            total_score += max(0.0, bridge_score)
        
        # Average score
        average_score = total_score / len(bridges)
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, average_score))
    
    def generate_report(self, bridges: List[Bridge]) -> str:
        """Generate a text report of the relationship analysis"""
        analysis = self.analyze_bridges(bridges)
        
        report = f"""
DocBridgeGuard 2.0 - Relationship Analysis Report
================================================

Overview:
- Total Bridges: {analysis['total_bridges']}
- Unique Entities: {analysis['entity_count']}
- Compliance Score: {analysis['compliance_score']:.2f}/1.0

Privacy Impact Distribution:
"""
        
        for impact, count in analysis['privacy_distribution'].items():
            percentage = (count / analysis['total_bridges'] * 100) if analysis['total_bridges'] > 0 else 0
            report += f"- {impact.title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"\nRelationship Types:\n"
        for rel_type, count in analysis['relationship_types'].items():
            report += f"- {rel_type.replace('_', ' ').title()}: {count}\n"
        
        if analysis['high_risk_entities']:
            report += f"\nHigh-Risk Entities:\n"
            for entity_info in analysis['high_risk_entities'][:5]:
                report += f"- {entity_info['entity']} (Risk Score: {entity_info['risk_score']})\n"
        
        if analysis['central_entities']:
            report += f"\nMost Central Entities:\n"
            for entity_info in analysis['central_entities'][:3]:
                report += f"- {entity_info['entity']} (Centrality: {entity_info['centrality_score']:.3f})\n"
        
        network_metrics = analysis['network_metrics']
        if network_metrics:
            report += f"\nNetwork Analysis:\n"
            report += f"- Network Density: {network_metrics.get('density', 0):.3f}\n"
            report += f"- Average Clustering: {network_metrics.get('average_clustering', 0):.3f}\n"
            report += f"- Connected: {network_metrics.get('is_connected', False)}\n"
        
        return report
    
    def export_to_dataframe(self, bridges: List[Bridge]) -> pd.DataFrame:
        """Export bridges to pandas DataFrame for further analysis"""
        if not bridges:
            return pd.DataFrame()
        
        data = []
        for bridge in bridges:
            data.append({
                "entity_1": bridge.entity_1,
                "entity_2": bridge.entity_2,
                "entity_1_type": bridge.entity_1_type.value,
                "entity_2_type": bridge.entity_2_type.value,
                "relationship": bridge.relationship,
                "confidence_score": bridge.confidence_score,
                "privacy_impact": bridge.privacy_impact.value,
                "legal_basis": bridge.legal_basis,
                "extraction_method": bridge.extraction_method
            })
        
        return pd.DataFrame(data)