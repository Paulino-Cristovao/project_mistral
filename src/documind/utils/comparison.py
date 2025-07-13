"""
Comparison engine for evaluating OCR provider performance
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import difflib
import numpy as np

from ..models import ProcessingResult, ComparisonResult

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """
    Compares OCR provider performance and generates detailed analysis
    """
    
    def __init__(self):
        """Initialize comparison engine"""
        logger.info("Comparison engine initialized")
    
    def compare_results(
        self,
        openai_result: ProcessingResult,
        mistral_result: ProcessingResult
    ) -> ComparisonResult:
        """
        Compare results from OpenAI and Mistral providers
        
        Args:
            openai_result: Processing result from OpenAI
            mistral_result: Processing result from Mistral
            
        Returns:
            Detailed comparison result
        """
        try:
            # Calculate comparison metrics
            metrics = self._calculate_metrics(openai_result, mistral_result)
            
            # Determine winner
            winner, confidence = self._determine_winner(metrics)
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(
                openai_result, mistral_result, metrics
            )
            
            comparison = ComparisonResult(
                document_id=openai_result.document_id,
                openai_result=openai_result,
                mistral_result=mistral_result,
                comparison_metrics=metrics,
                winner=winner,
                confidence_in_winner=confidence,
                detailed_analysis=detailed_analysis
            )
            
            logger.info(f"Comparison complete: {winner} wins with {confidence:.2f} confidence")
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            
            # Return failed comparison
            return ComparisonResult(
                document_id=openai_result.document_id or mistral_result.document_id,
                openai_result=openai_result if openai_result else None,
                mistral_result=mistral_result if mistral_result else None,
                comparison_metrics={"error": str(e)},
                winner=None,
                confidence_in_winner=0.0,
                detailed_analysis={"error": "Comparison failed", "details": str(e)}
            )
    
    def _calculate_metrics(
        self,
        openai_result: ProcessingResult,
        mistral_result: ProcessingResult
    ) -> Dict[str, float]:
        """Calculate comparison metrics between two results"""
        metrics = {}
        
        # Text extraction metrics
        if openai_result.extracted_text and mistral_result.extracted_text:
            metrics.update(self._compare_text_extraction(
                openai_result.extracted_text,
                mistral_result.extracted_text
            ))
        
        # Bridge extraction metrics
        metrics.update(self._compare_bridge_extraction(
            openai_result.bridges,
            mistral_result.bridges
        ))
        
        # Performance metrics
        metrics.update(self._compare_performance(
            openai_result,
            mistral_result
        ))
        
        # Compliance metrics
        metrics.update(self._compare_compliance(
            openai_result.compliance_metadata,
            mistral_result.compliance_metadata
        ))
        
        # Table extraction metrics
        metrics.update(self._compare_table_extraction(
            openai_result.tables,
            mistral_result.tables
        ))
        
        return metrics
    
    def _compare_text_extraction(self, text1: str, text2: str) -> Dict[str, float]:
        """Compare text extraction quality"""
        metrics = {}
        
        # Text similarity using sequence matcher
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        metrics["text_similarity"] = similarity
        
        # Length comparison
        len1, len2 = len(text1), len(text2)
        metrics["openai_text_length"] = len1
        metrics["mistral_text_length"] = len2
        metrics["length_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # Word count comparison
        words1 = text1.split()
        words2 = text2.split()
        metrics["openai_word_count"] = len(words1)
        metrics["mistral_word_count"] = len(words2)
        
        # Unique words analysis
        set1, set2 = set(words1), set(words2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        metrics["word_overlap"] = len(intersection) / len(union) if union else 0
        metrics["openai_unique_words"] = len(set1 - set2)
        metrics["mistral_unique_words"] = len(set2 - set1)
        
        return metrics
    
    def _compare_bridge_extraction(self, bridges1: List, bridges2: List) -> Dict[str, float]:
        """Compare bridge extraction results"""
        metrics = {}
        
        metrics["openai_bridge_count"] = len(bridges1)
        metrics["mistral_bridge_count"] = len(bridges2)
        
        if not bridges1 and not bridges2:
            metrics["bridge_overlap"] = 1.0
            return metrics
        
        # Extract entity pairs for comparison
        pairs1 = {(b.entity_1, b.entity_2) for b in bridges1}
        pairs2 = {(b.entity_1, b.entity_2) for b in bridges2}
        
        # Add reverse pairs for undirected comparison
        pairs1.update({(b.entity_2, b.entity_1) for b in bridges1})
        pairs2.update({(b.entity_2, b.entity_1) for b in bridges2})
        
        # Calculate overlap
        intersection = pairs1.intersection(pairs2)
        union = pairs1.union(pairs2)
        
        metrics["bridge_overlap"] = len(intersection) / len(union) if union else 0
        metrics["bridge_precision_openai"] = len(intersection) / len(pairs1) if pairs1 else 0
        metrics["bridge_precision_mistral"] = len(intersection) / len(pairs2) if pairs2 else 0
        
        # Average confidence scores
        if bridges1:
            metrics["openai_avg_bridge_confidence"] = sum(b.confidence_score for b in bridges1) / len(bridges1)
        if bridges2:
            metrics["mistral_avg_bridge_confidence"] = sum(b.confidence_score for b in bridges2) / len(bridges2)
        
        return metrics
    
    def _compare_performance(
        self,
        result1: ProcessingResult,
        result2: ProcessingResult
    ) -> Dict[str, float]:
        """Compare performance metrics"""
        return {
            "openai_processing_time": result1.processing_time_seconds,
            "mistral_processing_time": result2.processing_time_seconds,
            "speed_ratio": result1.processing_time_seconds / result2.processing_time_seconds 
                         if result2.processing_time_seconds > 0 else float('inf')
        }
    
    def _compare_compliance(self, metadata1, metadata2) -> Dict[str, float]:
        """Compare compliance metadata"""
        metrics = {}
        
        metrics["openai_compliance_score"] = metadata1.compliance_score
        metrics["mistral_compliance_score"] = metadata2.compliance_score
        metrics["compliance_score_diff"] = abs(metadata1.compliance_score - metadata2.compliance_score)
        
        metrics["openai_redactions"] = metadata1.redactions_count
        metrics["mistral_redactions"] = metadata2.redactions_count
        
        # Risk flags comparison
        flags1 = set(metadata1.risk_flags)
        flags2 = set(metadata2.risk_flags)
        
        if flags1 or flags2:
            intersection = flags1.intersection(flags2)
            union = flags1.union(flags2)
            metrics["risk_flags_overlap"] = len(intersection) / len(union)
        else:
            metrics["risk_flags_overlap"] = 1.0
        
        return metrics
    
    def _compare_table_extraction(self, tables1: List, tables2: List) -> Dict[str, float]:
        """Compare table extraction results"""
        return {
            "openai_table_count": len(tables1),
            "mistral_table_count": len(tables2),
            "table_count_diff": abs(len(tables1) - len(tables2))
        }
    
    def _determine_winner(self, metrics: Dict[str, float]) -> tuple[Optional[str], float]:
        """Determine the winning provider based on metrics"""
        
        # Scoring weights for different aspects
        weights = {
            "text_quality": 0.3,
            "bridge_quality": 0.25,
            "performance": 0.2,
            "compliance": 0.15,
            "tables": 0.1
        }
        
        openai_score = 0.0
        mistral_score = 0.0
        
        # Text quality scoring
        text_sim = metrics.get("text_similarity", 0.5)
        length_ratio = metrics.get("length_ratio", 0.5)
        word_overlap = metrics.get("word_overlap", 0.5)
        
        # Favor longer, more detailed extractions with good overlap
        openai_text_score = (
            text_sim * 0.4 + 
            length_ratio * 0.3 + 
            word_overlap * 0.3
        )
        mistral_text_score = openai_text_score  # Same baseline, differentiate with other factors
        
        # Adjust based on length (favor longer text if quality is similar)
        openai_len = metrics.get("openai_text_length", 0)
        mistral_len = metrics.get("mistral_text_length", 0)
        
        if openai_len > mistral_len:
            openai_text_score += 0.1
        elif mistral_len > openai_len:
            mistral_text_score += 0.1
        
        openai_score += openai_text_score * weights["text_quality"]
        mistral_score += mistral_text_score * weights["text_quality"]
        
        # Bridge quality scoring
        bridge_overlap = metrics.get("bridge_overlap", 0.5)
        openai_bridge_precision = metrics.get("bridge_precision_openai", 0)
        mistral_bridge_precision = metrics.get("bridge_precision_mistral", 0)
        
        openai_bridge_score = (bridge_overlap * 0.5 + openai_bridge_precision * 0.5)
        mistral_bridge_score = (bridge_overlap * 0.5 + mistral_bridge_precision * 0.5)
        
        openai_score += openai_bridge_score * weights["bridge_quality"]
        mistral_score += mistral_bridge_score * weights["bridge_quality"]
        
        # Performance scoring (favor faster processing)
        openai_time = metrics.get("openai_processing_time", 1.0)
        mistral_time = metrics.get("mistral_processing_time", 1.0)
        
        if openai_time < mistral_time:
            openai_score += weights["performance"]
        else:
            mistral_score += weights["performance"]
        
        # Compliance scoring
        openai_compliance = metrics.get("openai_compliance_score", 0.5)
        mistral_compliance = metrics.get("mistral_compliance_score", 0.5)
        
        if openai_compliance > mistral_compliance:
            openai_score += weights["compliance"]
        elif mistral_compliance > openai_compliance:
            mistral_score += weights["compliance"]
        else:
            # Equal compliance scores
            openai_score += weights["compliance"] * 0.5
            mistral_score += weights["compliance"] * 0.5
        
        # Table extraction scoring
        openai_tables = metrics.get("openai_table_count", 0)
        mistral_tables = metrics.get("mistral_table_count", 0)
        
        if openai_tables > mistral_tables:
            openai_score += weights["tables"]
        elif mistral_tables > openai_tables:
            mistral_score += weights["tables"]
        else:
            openai_score += weights["tables"] * 0.5
            mistral_score += weights["tables"] * 0.5
        
        # Determine winner
        if openai_score > mistral_score:
            winner = "openai"
            confidence = openai_score / (openai_score + mistral_score)
        elif mistral_score > openai_score:
            winner = "mistral"
            confidence = mistral_score / (openai_score + mistral_score)
        else:
            winner = "tie"
            confidence = 0.5
        
        return winner, confidence
    
    def _generate_detailed_analysis(
        self,
        openai_result: ProcessingResult,
        mistral_result: ProcessingResult,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate detailed analysis of the comparison"""
        
        analysis = {
            "summary": {},
            "text_analysis": {},
            "bridge_analysis": {},
            "performance_analysis": {},
            "compliance_analysis": {},
            "recommendations": []
        }
        
        # Summary
        analysis["summary"] = {
            "openai_status": openai_result.status.value,
            "mistral_status": mistral_result.status.value,
            "text_similarity": metrics.get("text_similarity", 0),
            "bridge_overlap": metrics.get("bridge_overlap", 0),
            "performance_difference": abs(
                metrics.get("openai_processing_time", 0) - 
                metrics.get("mistral_processing_time", 0)
            )
        }
        
        # Text analysis
        analysis["text_analysis"] = {
            "length_comparison": {
                "openai": metrics.get("openai_text_length", 0),
                "mistral": metrics.get("mistral_text_length", 0),
                "ratio": metrics.get("length_ratio", 0)
            },
            "word_analysis": {
                "openai_words": metrics.get("openai_word_count", 0),
                "mistral_words": metrics.get("mistral_word_count", 0),
                "overlap": metrics.get("word_overlap", 0)
            }
        }
        
        # Bridge analysis
        analysis["bridge_analysis"] = {
            "counts": {
                "openai": metrics.get("openai_bridge_count", 0),
                "mistral": metrics.get("mistral_bridge_count", 0)
            },
            "quality": {
                "overlap": metrics.get("bridge_overlap", 0),
                "openai_precision": metrics.get("bridge_precision_openai", 0),
                "mistral_precision": metrics.get("bridge_precision_mistral", 0)
            }
        }
        
        # Performance analysis
        analysis["performance_analysis"] = {
            "processing_times": {
                "openai": metrics.get("openai_processing_time", 0),
                "mistral": metrics.get("mistral_processing_time", 0)
            },
            "speed_comparison": {
                "ratio": metrics.get("speed_ratio", 1),
                "faster_provider": "openai" if metrics.get("openai_processing_time", 0) < metrics.get("mistral_processing_time", 0) else "mistral"
            }
        }
        
        # Compliance analysis
        analysis["compliance_analysis"] = {
            "scores": {
                "openai": metrics.get("openai_compliance_score", 0),
                "mistral": metrics.get("mistral_compliance_score", 0)
            },
            "redactions": {
                "openai": metrics.get("openai_redactions", 0),
                "mistral": metrics.get("mistral_redactions", 0)
            },
            "risk_flags_overlap": metrics.get("risk_flags_overlap", 0)
        }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(metrics)
        
        return analysis
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on comparison metrics"""
        recommendations = []
        
        # Text extraction recommendations
        text_sim = metrics.get("text_similarity", 0)
        if text_sim < 0.7:
            recommendations.append(
                "Low text similarity detected. Consider manual review of extraction accuracy."
            )
        
        # Bridge extraction recommendations
        bridge_overlap = metrics.get("bridge_overlap", 0)
        if bridge_overlap < 0.5:
            recommendations.append(
                "Low bridge overlap suggests different relationship extraction. Review entity relationships manually."
            )
        
        # Performance recommendations
        openai_time = metrics.get("openai_processing_time", 0)
        mistral_time = metrics.get("mistral_processing_time", 0)
        
        if abs(openai_time - mistral_time) > 5.0:  # More than 5 seconds difference
            faster = "OpenAI" if openai_time < mistral_time else "Mistral"
            recommendations.append(
                f"{faster} significantly faster for this document type. Consider for time-sensitive processing."
            )
        
        # Compliance recommendations
        openai_compliance = metrics.get("openai_compliance_score", 0)
        mistral_compliance = metrics.get("mistral_compliance_score", 0)
        
        if max(openai_compliance, mistral_compliance) < 0.8:
            recommendations.append(
                "Low compliance scores detected. Review privacy settings and redaction levels."
            )
        
        # Table recommendations
        openai_tables = metrics.get("openai_table_count", 0)
        mistral_tables = metrics.get("mistral_table_count", 0)
        
        if abs(openai_tables - mistral_tables) > 2:
            better_provider = "OpenAI" if openai_tables > mistral_tables else "Mistral"
            recommendations.append(
                f"{better_provider} extracted more tables. Consider for table-heavy documents."
            )
        
        return recommendations