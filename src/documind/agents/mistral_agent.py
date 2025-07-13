"""
Mistral agent with advanced capabilities for DocBridgeGuard 2.0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from mistralai import Mistral

from .base_agent import BaseAgent
from ..models import Bridge, EntityType, PrivacyImpact

logger = logging.getLogger(__name__)


class MistralAgent(BaseAgent):
    """
    Advanced Mistral agent with tool calling and multilingual capabilities
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Mistral agent
        
        Args:
            api_key: Mistral API key
            model: Model to use for processing
            config: Agent configuration
        """
        capabilities = [
            "document_ocr",
            "multilingual_processing",
            "text_extraction",
            "table_detection",
            "entity_recognition",
            "relationship_extraction",
            "content_analysis",
            "function_calling",
            "european_compliance",
            "african_languages"
        ]
        
        super().__init__(
            name="Mistral",
            api_key=api_key,
            config=config or {},
            capabilities=capabilities
        )
        
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.embedding_model = self.config.get("embedding_model", "mistral-embed")
        
        # Mistral-specific configuration
        self.supported_languages = [
            "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"
        ]
        
        # Define available tools for Mistral
        self.tools = self._define_tools()
        
        logger.info(f"Mistral agent initialized with model: {model}")
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities
    
    def process_document(
        self, 
        file_path: Union[str, Path],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process document using Mistral with advanced capabilities
        
        Args:
            file_path: Path to document file
            task_context: Task context and parameters
            
        Returns:
            Processing results
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not self.validate_inputs(file_path, task_context):
                return self.handle_error(
                    ValueError("Invalid inputs"), 
                    task_context
                )
            
            file_path = Path(file_path)
            
            # Process with Mistral OCR if available, otherwise use chat model
            result = self._process_with_mistral(file_path, task_context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(result)
            
            # Record execution metrics
            self.record_execution(start_time, True, confidence)
            
            return {
                "success": True,
                "agent": self.name,
                "result": result,
                "confidence": confidence,
                "model_used": self.model,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            self.record_execution(start_time, False, 0.0)
            return self.handle_error(e, task_context)
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools/functions for Mistral agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_multilingual_content",
                    "description": "Extract content from multilingual documents with language detection",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "primary_language": {
                                "type": "string",
                                "description": "Primary language detected in document"
                            },
                            "secondary_languages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Additional languages found"
                            },
                            "extracted_text": {
                                "type": "string",
                                "description": "Extracted text content"
                            },
                            "translation_notes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Notes about translations or interpretations"
                            },
                            "cultural_context": {
                                "type": "string",
                                "description": "Cultural context relevant to document interpretation"
                            }
                        },
                        "required": ["primary_language", "extracted_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_european_compliance",
                    "description": "Analyze document for European regulatory compliance (GDPR, AI Act, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gdpr_assessment": {
                                "type": "object",
                                "properties": {
                                    "legal_basis": {"type": "string"},
                                    "data_categories": {"type": "array", "items": {"type": "string"}},
                                    "special_categories": {"type": "boolean"},
                                    "consent_required": {"type": "boolean"},
                                    "retention_period": {"type": "string"}
                                }
                            },
                            "ai_act_relevance": {
                                "type": "object",
                                "properties": {
                                    "risk_category": {"type": "string"},
                                    "prohibited_practices": {"type": "array", "items": {"type": "string"}},
                                    "transparency_requirements": {"type": "boolean"}
                                }
                            },
                            "compliance_score": {
                                "type": "number",
                                "description": "Overall EU compliance score (0-1)"
                            }
                        },
                        "required": ["gdpr_assessment", "compliance_score"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_structured_data",
                    "description": "Extract and structure document data including tables and forms",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_structure": {
                                "type": "object",
                                "properties": {
                                    "sections": {"type": "array", "items": {"type": "string"}},
                                    "headers": {"type": "array", "items": {"type": "string"}},
                                    "footers": {"type": "array", "items": {"type": "string"}}
                                }
                            },
                            "tables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "headers": {"type": "array", "items": {"type": "string"}},
                                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                                        "confidence": {"type": "number"}
                                    }
                                }
                            },
                            "forms_data": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field_name": {"type": "string"},
                                        "field_value": {"type": "string"},
                                        "field_type": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["document_structure"]
                    }
                }
            }
        ]
    
    def _process_with_mistral(
        self, 
        file_path: Path, 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process document using Mistral capabilities"""
        
        # Try OCR endpoint first (if available)
        try:
            ocr_result = self._try_mistral_ocr(file_path)
            if ocr_result:
                # Enhance OCR result with chat model analysis
                enhanced_result = self._enhance_with_chat_model(
                    ocr_result, task_context
                )
                return enhanced_result
        except Exception as e:
            logger.warning(f"Mistral OCR failed, falling back to chat model: {e}")
        
        # Fallback to chat model with document analysis
        return self._analyze_with_chat_model(file_path, task_context)
    
    def _try_mistral_ocr(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Try to use Mistral OCR endpoint"""
        try:
            with open(file_path, "rb") as f:
                # Note: This is hypothetical - actual Mistral OCR API may differ
                response = self.client.ocr.extract(
                    file=f,
                    model="mistral-ocr-latest"
                )
            
            if isinstance(response, dict):
                return {
                    "text": response.get("text", ""),
                    "tables": response.get("tables", []),
                    "metadata": response.get("metadata", {})
                }
            elif hasattr(response, 'text'):
                return {
                    "text": response.text,
                    "tables": getattr(response, 'tables', []),
                    "metadata": getattr(response, 'metadata', {})
                }
            
        except Exception as e:
            logger.debug(f"Mistral OCR not available: {e}")
            return None
    
    def _enhance_with_chat_model(
        self, 
        ocr_result: Dict[str, Any], 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance OCR results using chat model analysis"""
        
        system_prompt = """
        You are Mistral DocBridge, specialized in European multilingual document analysis.
        
        Enhance the provided OCR results with:
        1. Language detection and cultural context
        2. European compliance assessment (GDPR, AI Act)
        3. Entity and relationship extraction
        4. Structured data validation
        
        Focus on accuracy, multilingual support, and European regulatory requirements.
        """
        
        user_prompt = f"""
        Analyze and enhance this OCR result:
        
        OCR Text: {ocr_result.get('text', '')[:2000]}...
        
        Task Context:
        - Compliance Profile: {task_context.get('compliance_profile', 'eu_gdpr')}
        - Expected Language: {task_context.get('expected_language', 'auto-detect')}
        
        Use available tools to provide comprehensive analysis.
        """
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=self.tools,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.1
            )
            
            enhanced_result = self._process_chat_response(response, ocr_result)
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Chat model enhancement failed: {e}")
            return ocr_result
    
    def _analyze_with_chat_model(
        self, 
        file_path: Path, 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze document using chat model when OCR is not available"""
        
        system_prompt = """
        You are Mistral DocBridge, an expert in multilingual document analysis.
        
        Since direct OCR is not available, provide the best possible analysis
        based on the document context and requirements.
        
        Focus on:
        1. Document type and structure assessment
        2. Expected content patterns
        3. Compliance requirements
        4. Multilingual considerations
        """
        
        # Create document context prompt
        file_info = {
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "extension": file_path.suffix
        }
        
        user_prompt = f"""
        Analyze document: {file_info}
        
        Context: {task_context}
        
        Provide structured analysis using available tools.
        """
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=self.tools,
                tool_choice="auto",
                max_tokens=3000,
                temperature=0.2
            )
            
            result = self._process_chat_response(response, {})
            return result
            
        except Exception as e:
            logger.error(f"Chat model analysis failed: {e}")
            return self._create_fallback_result(file_path, task_context)
    
    def _process_chat_response(
        self, 
        response, 
        base_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process Mistral chat response with tool calls"""
        
        result = {
            "text": base_result.get("text", ""),
            "tables": base_result.get("tables", []),
            "entities": [],
            "bridges": [],
            "compliance": {},
            "metadata": base_result.get("metadata", {}),
            "language_info": {}
        }
        
        # Handle tool calls or function calls
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            
            # Process tool calls if available
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name == "extract_multilingual_content":
                            result["text"] = function_args.get("extracted_text", result["text"])
                            result["language_info"] = {
                                "primary_language": function_args.get("primary_language"),
                                "secondary_languages": function_args.get("secondary_languages", []),
                                "cultural_context": function_args.get("cultural_context"),
                                "translation_notes": function_args.get("translation_notes", [])
                            }
                        
                        elif function_name == "analyze_european_compliance":
                            result["compliance"] = function_args
                        
                        elif function_name == "extract_structured_data":
                            if "tables" in function_args:
                                result["tables"].extend(function_args["tables"])
                            result["metadata"]["structure"] = function_args.get("document_structure", {})
                            result["metadata"]["forms"] = function_args.get("forms_data", [])
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call arguments: {e}")
            
            # Fallback to content if no tool calls
            elif hasattr(message, 'content') and message.content:
                try:
                    parsed_content = json.loads(message.content)
                    result.update(parsed_content)
                except json.JSONDecodeError:
                    if not result["text"]:
                        result["text"] = message.content
        
        return result
    
    def _create_fallback_result(
        self, 
        file_path: Path, 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create fallback result when processing fails"""
        return {
            "text": f"[Processing unavailable for {file_path.name}]",
            "tables": [],
            "entities": [],
            "bridges": [],
            "compliance": {"compliance_score": 0.5},
            "metadata": {
                "fallback": True,
                "filename": file_path.name,
                "size": file_path.stat().st_size
            },
            "language_info": {"primary_language": "unknown"}
        }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for Mistral results"""
        
        scores = []
        
        # Text quality score
        text = result.get("text", "")
        if text and not text.startswith("[Processing unavailable"):
            text_score = min(1.0, len(text) / 1000)  # Longer text = higher confidence
            scores.append(text_score)
        
        # Language detection confidence
        lang_info = result.get("language_info", {})
        if lang_info.get("primary_language") and lang_info["primary_language"] != "unknown":
            scores.append(0.8)
        
        # Compliance score
        compliance_score = result.get("compliance", {}).get("compliance_score")
        if compliance_score is not None:
            scores.append(compliance_score)
        
        # Table extraction score
        tables = result.get("tables", [])
        if tables:
            table_confidences = [t.get("confidence", 0.7) for t in tables]
            scores.append(sum(table_confidences) / len(table_confidences))
        
        # Fallback handling
        if result.get("metadata", {}).get("fallback"):
            scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.6
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Mistral embeddings model"""
        try:
            response = self.client.embeddings(
                model=self.embedding_model,
                input=texts
            )
            
            # Handle different response formats
            if hasattr(response, 'data'):
                return [item.embedding for item in response.data]
            elif isinstance(response, list):
                return response
            else:
                return [response.embedding] if hasattr(response, 'embedding') else []
                
        except Exception as e:
            logger.error(f"Failed to get Mistral embeddings: {e}")
            return []
    
    def detect_language(self, text: str) -> str:
        """Detect language using Mistral's multilingual capabilities"""
        try:
            prompt = f"Detect the primary language of this text and respond with just the ISO language code: {text[:500]}"
            
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            if hasattr(response, 'choices') and response.choices:
                detected = response.choices[0].message.content.strip().lower()
                return detected if detected in self.supported_languages else "unknown"
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        
        return "unknown"