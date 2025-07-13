"""
Adaptive agent for learning from processing history and improving decisions
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class AdaptiveAgent:
    """
    Learns from processing history to improve future decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive agent
        
        Args:
            config: Adaptive agent configuration
        """
        self.config = config or {}
        
        # Learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.memory_size = self.config.get("memory_size", 1000)
        self.min_samples_for_learning = self.config.get("min_samples_for_learning", 5)
        
        # Learning data structures
        self.provider_performance: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.document_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.strategy_effectiveness: Dict[str, List[Dict[str, Any]]] = defaultdict(lambda: defaultdict(list))
        self.user_preference_patterns: Dict[str, int] = defaultdict(int)
        self.temporal_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Model weights and biases
        self.provider_weights = {
            "openai": {"quality": 0.5, "speed": 0.5, "compliance": 0.5},
            "mistral": {"quality": 0.5, "speed": 0.5, "compliance": 0.5}
        }
        
        self.strategy_weights = {
            "sequential": {"accuracy": 0.5, "efficiency": 0.5},
            "parallel": {"accuracy": 0.5, "efficiency": 0.5},
            "best_agent": {"accuracy": 0.5, "efficiency": 0.5},
            "consensus": {"accuracy": 0.5, "efficiency": 0.5}
        }
        
        # Performance thresholds
        self.quality_threshold = 0.7
        self.speed_threshold = 30.0  # seconds
        self.confidence_threshold = 0.8
        
        logger.info("Adaptive agent initialized for continuous learning")
    
    def learn_from_processing(
        self,
        doc_characteristics: Dict[str, Any],
        processing_decisions: Dict[str, Any],
        processing_result: Any  # ProcessingResult object
    ) -> None:
        """
        Learn from a processing session
        
        Args:
            doc_characteristics: Document characteristics
            processing_decisions: Decisions made
            processing_result: Processing result
        """
        try:
            # Extract learning features
            features = self._extract_learning_features(
                doc_characteristics, processing_decisions, processing_result
            )
            
            # Update provider performance tracking
            self._update_provider_performance(features)
            
            # Update strategy effectiveness tracking
            self._update_strategy_effectiveness(features)
            
            # Update document pattern recognition
            self._update_document_patterns(features)
            
            # Update temporal patterns
            self._update_temporal_patterns(features)
            
            # Adapt model weights based on performance
            self._adapt_model_weights(features)
            
            # Prune old data if needed
            self._prune_memory()
            
            logger.debug(f"Learned from processing session: {features['session_id']}")
            
        except Exception as e:
            logger.error(f"Learning from processing failed: {e}")
    
    def get_agent_preference(
        self,
        doc_characteristics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get agent preference based on learned patterns
        
        Args:
            doc_characteristics: Document characteristics
            context: Additional context
            
        Returns:
            Preferred agent name or None
        """
        try:
            # Find similar documents in history
            similar_docs = self._find_similar_documents(doc_characteristics)
            
            if len(similar_docs) < self.min_samples_for_learning:
                return None
            
            # Calculate provider scores based on historical performance
            provider_scores = self._calculate_provider_scores(similar_docs, doc_characteristics)
            
            # Select best performing provider
            if provider_scores:
                best_provider = max(provider_scores.keys(), key=lambda k: provider_scores[k])
                confidence = provider_scores[best_provider]
                
                if confidence >= self.confidence_threshold:
                    logger.info(f"Adaptive agent recommends {best_provider} with {confidence:.2f} confidence")
                    return best_provider
            
            return None
            
        except Exception as e:
            logger.error(f"Agent preference calculation failed: {e}")
            return None
    
    def get_strategy_recommendation(
        self,
        doc_characteristics: Dict[str, Any],
        processing_decisions: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get strategy recommendation based on learned patterns
        
        Args:
            doc_characteristics: Document characteristics
            processing_decisions: Current processing decisions
            
        Returns:
            Recommended strategy or None
        """
        try:
            doc_type = doc_characteristics.get("predicted_document_type", "unknown")
            complexity = doc_characteristics.get("complexity_estimate", "medium")
            
            # Get strategy performance for similar documents
            strategy_key = f"{doc_type}_{complexity}"
            strategy_performance = self.strategy_effectiveness[strategy_key]
            
            if not strategy_performance:
                return None
            
            # Calculate strategy scores
            strategy_scores: Dict[str, List[float]] = {}
            for strategy_data in strategy_performance:
                strategy = strategy_data["strategy"]
                quality = strategy_data["quality_score"]
                efficiency = strategy_data["efficiency_score"]
                
                # Weighted score
                score = (quality * self.strategy_weights[strategy]["accuracy"] + 
                        efficiency * self.strategy_weights[strategy]["efficiency"])
                
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(score)
            
            # Average scores and select best
            avg_scores = {
                strategy: np.mean(scores) 
                for strategy, scores in strategy_scores.items()
            }
            
            if avg_scores:
                best_strategy = max(avg_scores.keys(), key=lambda k: avg_scores[k])
                confidence = avg_scores[best_strategy]
                
                if confidence >= 0.7:  # High confidence threshold for strategy
                    return best_strategy
            
            return None
            
        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            return None
    
    def get_optimization_suggestions(
        self,
        doc_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimization suggestions based on learned patterns
        
        Args:
            doc_characteristics: Document characteristics
            
        Returns:
            Optimization suggestions
        """
        suggestions = {
            "parameter_adjustments": {},
            "feature_recommendations": {},
            "performance_insights": {},
            "confidence_level": 0.0
        }
        
        try:
            # Analyze confidence threshold optimization
            confidence_suggestions = self._analyze_confidence_thresholds(doc_characteristics)
            suggestions["parameter_adjustments"]["confidence_threshold"] = confidence_suggestions
            
            # Analyze feature enablement patterns
            feature_suggestions = self._analyze_feature_patterns(doc_characteristics)
            suggestions["feature_recommendations"] = feature_suggestions
            
            # Generate performance insights
            performance_insights = self._generate_performance_insights(doc_characteristics)
            suggestions["performance_insights"] = performance_insights
            
            # Calculate overall confidence in suggestions
            suggestions["confidence_level"] = self._calculate_suggestion_confidence(doc_characteristics)
            
        except Exception as e:
            logger.error(f"Optimization suggestions failed: {e}")
        
        return suggestions
    
    def _extract_learning_features(
        self,
        doc_characteristics: Dict[str, Any],
        processing_decisions: Dict[str, Any],
        processing_result: Any
    ) -> Dict[str, Any]:
        """Extract features for learning"""
        
        # Extract result metrics
        quality_score = getattr(processing_result, "quality_metadata", {}).get("quality_score", 0.0)
        processing_time = getattr(processing_result, "processing_time_seconds", 0.0)
        compliance_score = getattr(processing_result, "compliance_metadata", None)
        compliance_score = compliance_score.compliance_score if compliance_score else 0.0
        
        return {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(),
            "document_type": doc_characteristics.get("predicted_document_type", "unknown"),
            "complexity": doc_characteristics.get("complexity_estimate", "medium"),
            "file_size": doc_characteristics.get("file_size", 0),
            "language": doc_characteristics.get("likely_language", "english"),
            "provider_used": processing_decisions.get("preferred_provider", "unknown"),
            "strategy_used": processing_decisions.get("optimal_strategy", "sequential"),
            "bridge_extraction_enabled": processing_decisions.get("enable_bridge_extraction", False),
            "redaction_level": processing_decisions.get("optimal_redaction_level", "moderate"),
            "quality_score": quality_score,
            "processing_time": processing_time,
            "compliance_score": compliance_score,
            "success": getattr(processing_result, "status", None) and processing_result.status.value == "completed",
            "bridge_count": len(getattr(processing_result, "bridges", [])),
            "confidence_threshold_used": processing_decisions.get("confidence_threshold", 0.7)
        }
    
    def _update_provider_performance(self, features: Dict[str, Any]) -> None:
        """Update provider performance tracking"""
        
        provider = features["provider_used"]
        doc_type = features["document_type"]
        
        performance_data = {
            "timestamp": features["timestamp"],
            "quality_score": features["quality_score"],
            "processing_time": features["processing_time"],
            "compliance_score": features["compliance_score"],
            "success": features["success"],
            "document_complexity": features["complexity"]
        }
        
        self.provider_performance[provider][doc_type].append(performance_data)
    
    def _update_strategy_effectiveness(self, features: Dict[str, Any]) -> None:
        """Update strategy effectiveness tracking"""
        
        strategy = features["strategy_used"]
        doc_type = features["document_type"]
        complexity = features["complexity"]
        
        # Calculate efficiency score (inverse of processing time, normalized)
        efficiency_score = min(1.0, self.speed_threshold / max(features["processing_time"], 1.0))
        
        strategy_key = f"{doc_type}_{complexity}"
        strategy_data = {
            "timestamp": features["timestamp"],
            "strategy": strategy,
            "quality_score": features["quality_score"],
            "efficiency_score": efficiency_score,
            "compliance_score": features["compliance_score"],
            "success": features["success"]
        }
        
        self.strategy_effectiveness[strategy_key].append(strategy_data)
    
    def _update_document_patterns(self, features: Dict[str, Any]) -> None:
        """Update document pattern recognition"""
        
        pattern = {
            "timestamp": features["timestamp"],
            "document_type": features["document_type"],
            "complexity": features["complexity"],
            "file_size": features["file_size"],
            "language": features["language"],
            "optimal_provider": features["provider_used"] if features["success"] else None,
            "optimal_strategy": features["strategy_used"] if features["success"] else None,
            "performance_score": (features["quality_score"] + features["compliance_score"]) / 2
        }
        
        pattern_key = f"{features['document_type']}_{features['complexity']}_{features['language']}"
        self.document_patterns[pattern_key].append(pattern)
    
    def _update_temporal_patterns(self, features: Dict[str, Any]) -> None:
        """Update temporal patterns"""
        
        hour = features["timestamp"].hour
        day_of_week = features["timestamp"].weekday()
        
        temporal_data = {
            "hour": hour,
            "day_of_week": day_of_week,
            "processing_time": features["processing_time"],
            "quality_score": features["quality_score"],
            "provider": features["provider_used"]
        }
        
        self.temporal_patterns["hourly"].append(temporal_data)
    
    def _adapt_model_weights(self, features: Dict[str, Any]) -> None:
        """Adapt model weights based on performance feedback"""
        
        provider = features["provider_used"]
        strategy = features["strategy_used"]
        
        # Calculate performance indicators
        quality_good = features["quality_score"] >= self.quality_threshold
        speed_good = features["processing_time"] <= self.speed_threshold
        compliance_good = features["compliance_score"] >= self.quality_threshold
        
        # Update provider weights
        if provider in self.provider_weights:
            weights = self.provider_weights[provider]
            
            # Adjust weights based on performance
            if quality_good:
                weights["quality"] = min(1.0, weights["quality"] + self.learning_rate)
            else:
                weights["quality"] = max(0.1, weights["quality"] - self.learning_rate)
            
            if speed_good:
                weights["speed"] = min(1.0, weights["speed"] + self.learning_rate)
            else:
                weights["speed"] = max(0.1, weights["speed"] - self.learning_rate)
            
            if compliance_good:
                weights["compliance"] = min(1.0, weights["compliance"] + self.learning_rate)
            else:
                weights["compliance"] = max(0.1, weights["compliance"] - self.learning_rate)
        
        # Update strategy weights
        if strategy in self.strategy_weights:
            weights = self.strategy_weights[strategy]
            
            # Accuracy based on quality and compliance
            accuracy_good = quality_good and compliance_good
            if accuracy_good:
                weights["accuracy"] = min(1.0, weights["accuracy"] + self.learning_rate)
            else:
                weights["accuracy"] = max(0.1, weights["accuracy"] - self.learning_rate)
            
            # Efficiency based on speed
            if speed_good:
                weights["efficiency"] = min(1.0, weights["efficiency"] + self.learning_rate)
            else:
                weights["efficiency"] = max(0.1, weights["efficiency"] - self.learning_rate)
    
    def _find_similar_documents(self, doc_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar documents in processing history"""
        
        target_type = doc_characteristics.get("predicted_document_type", "unknown")
        target_complexity = doc_characteristics.get("complexity_estimate", "medium")
        target_language = doc_characteristics.get("likely_language", "english")
        
        similar_docs = []
        
        # Search through document patterns
        for pattern_key, patterns in self.document_patterns.items():
            if target_type in pattern_key and target_complexity in pattern_key:
                similar_docs.extend(patterns)
        
        # Sort by recency (more recent documents are more relevant)
        similar_docs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return similar_docs[:50]  # Return top 50 most recent similar documents
    
    def _calculate_provider_scores(
        self,
        similar_docs: List[Dict[str, Any]],
        doc_characteristics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate provider scores based on similar documents"""
        
        provider_scores = defaultdict(list)
        
        for doc in similar_docs:
            if doc["optimal_provider"] and doc["performance_score"]:
                provider_scores[doc["optimal_provider"]].append(doc["performance_score"])
        
        # Calculate average scores
        avg_scores = {}
        for provider, scores in provider_scores.items():
            if len(scores) >= self.min_samples_for_learning:
                avg_score = np.mean(scores)
                # Apply provider weights
                if provider in self.provider_weights:
                    weights = self.provider_weights[provider]
                    weighted_score = avg_score * np.mean(list(weights.values()))
                    avg_scores[provider] = weighted_score
                else:
                    avg_scores[provider] = avg_score
        
        return avg_scores
    
    def _analyze_confidence_thresholds(self, doc_characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Analyze optimal confidence thresholds"""
        
        doc_type = doc_characteristics.get("predicted_document_type", "unknown")
        
        # Find confidence threshold performance
        threshold_performance: Dict[str, List[float]] = defaultdict(list)
        
        for pattern_key, patterns in self.document_patterns.items():
            if doc_type in pattern_key:
                for pattern in patterns:
                    # This would need to be tracked in features
                    # For now, return default suggestions
                    pass
        
        return {
            "recommended_threshold": 0.75,
            "confidence_in_recommendation": 0.6
        }
    
    def _analyze_feature_patterns(self, doc_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature enablement patterns"""
        
        return {
            "bridge_extraction": {
                "recommended": True,
                "confidence": 0.8,
                "reasoning": "Historical data shows improved compliance scores"
            },
            "table_extraction": {
                "recommended": doc_characteristics.get("complexity_estimate") != "low",
                "confidence": 0.7,
                "reasoning": "Complex documents benefit from table extraction"
            }
        }
    
    def _generate_performance_insights(self, doc_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance insights"""
        
        insights = {}
        
        # Provider performance insights
        if self.provider_performance:
            insights["provider_trends"] = self._analyze_provider_trends()
        
        # Strategy effectiveness insights
        if self.strategy_effectiveness:
            insights["strategy_trends"] = self._analyze_strategy_trends()
        
        # Temporal insights
        if self.temporal_patterns:
            insights["temporal_patterns"] = self._analyze_temporal_trends()
        
        return insights
    
    def _analyze_provider_trends(self) -> Dict[str, Any]:
        """Analyze provider performance trends"""
        
        trends = {}
        
        for provider, doc_types in self.provider_performance.items():
            provider_quality = []
            provider_speed = []
            
            for doc_type, performances in doc_types.items():
                for perf in performances[-10:]:  # Last 10 sessions
                    provider_quality.append(perf["quality_score"])
                    provider_speed.append(perf["processing_time"])
            
            if provider_quality:
                trends[provider] = {
                    "avg_quality": np.mean(provider_quality),
                    "avg_speed": np.mean(provider_speed),
                    "consistency": 1.0 - np.std(provider_quality),  # Lower std = higher consistency
                    "sample_size": len(provider_quality)
                }
        
        return trends
    
    def _analyze_strategy_trends(self) -> Dict[str, Any]:
        """Analyze strategy effectiveness trends"""
        
        trends = {}
        
        for strategy_key, performances in self.strategy_effectiveness.items():
            strategy_scores = defaultdict(list)
            
            for perf in performances[-10:]:  # Last 10 sessions
                strategy = perf["strategy"]
                strategy_scores[strategy].append(perf["quality_score"])
            
            for strategy, scores in strategy_scores.items():
                if len(scores) >= 3:  # Minimum sample size
                    trends[f"{strategy_key}_{strategy}"] = {
                        "avg_performance": np.mean(scores),
                        "consistency": 1.0 - np.std(scores),
                        "trend": "improving" if scores[-1] > scores[0] else "declining"
                    }
        
        return trends
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze temporal processing patterns"""
        
        if not self.temporal_patterns["hourly"]:
            return {}
        
        hourly_performance = defaultdict(list)
        
        for entry in self.temporal_patterns["hourly"][-100:]:  # Last 100 entries
            hour = entry["hour"]
            hourly_performance[hour].append(entry["processing_time"])
        
        trends = {}
        for hour, times in hourly_performance.items():
            if len(times) >= 3:
                trends[f"hour_{hour}"] = {
                    "avg_processing_time": np.mean(times),
                    "performance_rating": "fast" if np.mean(times) < self.speed_threshold else "slow"
                }
        
        return trends
    
    def _calculate_suggestion_confidence(self, doc_characteristics: Dict[str, Any]) -> float:
        """Calculate confidence in optimization suggestions"""
        
        doc_type = doc_characteristics.get("predicted_document_type", "unknown")
        
        # Count similar documents in history
        similar_count = 0
        for pattern_key in self.document_patterns.keys():
            if doc_type in pattern_key:
                similar_count += len(self.document_patterns[pattern_key])
        
        # Confidence based on sample size
        if similar_count >= 20:
            return 0.9
        elif similar_count >= 10:
            return 0.7
        elif similar_count >= 5:
            return 0.5
        else:
            return 0.3
    
    def _prune_memory(self) -> None:
        """Prune old data to maintain memory limits"""
        
        cutoff_date = datetime.now() - timedelta(days=30)  # Keep last 30 days
        
        # Prune provider performance data
        for provider in self.provider_performance:
            for doc_type in self.provider_performance[provider]:
                self.provider_performance[provider][doc_type] = [
                    perf for perf in self.provider_performance[provider][doc_type]
                    if perf["timestamp"] > cutoff_date
                ]
        
        # Prune document patterns
        for pattern_key in self.document_patterns:
            self.document_patterns[pattern_key] = [
                pattern for pattern in self.document_patterns[pattern_key]
                if pattern["timestamp"] > cutoff_date
            ]
        
        # Prune strategy effectiveness data
        for strategy_key in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_key] = [
                perf for perf in self.strategy_effectiveness[strategy_key]
                if perf["timestamp"] > cutoff_date
            ]
        
        # Prune temporal patterns
        if "hourly" in self.temporal_patterns:
            self.temporal_patterns["hourly"] = [
                entry for entry in self.temporal_patterns["hourly"]
                if entry.get("timestamp", datetime.now()) > cutoff_date
            ]
    
    def export_learning_data(self, file_path: str) -> None:
        """Export learning data for backup or analysis"""
        
        export_data = {
            "provider_performance": dict(self.provider_performance),
            "document_patterns": dict(self.document_patterns),
            "strategy_effectiveness": dict(self.strategy_effectiveness),
            "provider_weights": self.provider_weights,
            "strategy_weights": self.strategy_weights,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Learning data exported to {file_path}")
    
    def import_learning_data(self, file_path: str) -> None:
        """Import learning data from backup"""
        
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            if "provider_performance" in import_data:
                self.provider_performance.update(import_data["provider_performance"])
            
            if "document_patterns" in import_data:
                self.document_patterns.update(import_data["document_patterns"])
            
            if "strategy_effectiveness" in import_data:
                self.strategy_effectiveness.update(import_data["strategy_effectiveness"])
            
            if "provider_weights" in import_data:
                self.provider_weights.update(import_data["provider_weights"])
            
            if "strategy_weights" in import_data:
                self.strategy_weights.update(import_data["strategy_weights"])
            
            logger.info(f"Learning data imported from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to import learning data: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        
        total_sessions = sum(
            len(patterns) for patterns in self.document_patterns.values()
        )
        
        provider_data_points = sum(
            len(doc_data) for provider_data in self.provider_performance.values()
            for doc_data in provider_data.values()
        )
        
        strategy_data_points = sum(
            len(strategy_data) for strategy_data in self.strategy_effectiveness.values()
        )
        
        return {
            "total_learning_sessions": total_sessions,
            "provider_data_points": provider_data_points,
            "strategy_data_points": strategy_data_points,
            "document_types_seen": len(set(
                pattern["document_type"] for patterns in self.document_patterns.values()
                for pattern in patterns
            )),
            "learning_active": total_sessions >= self.min_samples_for_learning,
            "learning_rate": self.learning_rate,
            "last_updated": datetime.now().isoformat()
        }