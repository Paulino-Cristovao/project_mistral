"""
OpenAI agent with advanced capabilities for DocBridgeGuard 2.0
"""

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from openai import OpenAI
import tiktoken

from .base_agent import BaseAgent
from ..models import Bridge, EntityType, PrivacyImpact

logger = logging.getLogger(__name__)


class OpenAIAgent(BaseAgent):
    """
    Advanced OpenAI agent with function calling and tool use capabilities
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-vision-preview",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OpenAI agent
        
        Args:
            api_key: OpenAI API key
            model: Model to use for processing
            config: Agent configuration
        """
        capabilities = [
            "document_ocr",
            "image_analysis",
            "text_extraction",
            "table_detection",
            "entity_recognition",
            "relationship_extraction",
            "content_analysis",
            "function_calling",
            "tool_usage"
        ]
        
        super().__init__(
            name="OpenAI",
            api_key=api_key,
            config=config or {},
            capabilities=capabilities
        )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Define available tools/functions
        self.tools = self._define_tools()
        
        logger.info(f"OpenAI agent initialized with model: {model}")
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities
    
    def process_document(
        self, 
        file_path: Union[str, Path],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process document using OpenAI with advanced capabilities
        
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
            
            # Encode file for processing
            encoded_file = self._encode_file(file_path)
            
            # Create processing prompt with tools
            messages = self._create_processing_messages(encoded_file, task_context)
            
            # Process with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.1
            )
            
            # Process response and tool calls
            result = self._process_response(response, task_context)
            
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
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "token_usage": response.usage.model_dump() if response.usage else {}
            }
            
        except Exception as e:
            self.record_execution(start_time, False, 0.0)
            return self.handle_error(e, task_context)
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools/functions for the agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_document_structure",
                    "description": "Extract structured information from document including text, tables, and metadata",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_type": {
                                "type": "string",
                                "enum": ["contract", "medical", "financial", "legal", "research", "other"],
                                "description": "Type of document being processed"
                            },
                            "extracted_text": {
                                "type": "string",
                                "description": "Main text content extracted from the document"
                            },
                            "tables": {
                                "type": "array",
                                "description": "Tables found in the document",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "headers": {"type": "array", "items": {"type": "string"}},
                                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                                    }
                                }
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Document metadata",
                                "properties": {
                                    "language": {"type": "string"},
                                    "page_count": {"type": "integer"},
                                    "confidence_score": {"type": "number"}
                                }
                            }
                        },
                        "required": ["document_type", "extracted_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "identify_entities_and_relationships",
                    "description": "Identify entities and their relationships (bridges) in the document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "description": "Entities found in the document",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "type": {"type": "string", "enum": ["person", "organization", "location", "date", "money", "product", "contract_id", "other"]},
                                        "confidence": {"type": "number"}
                                    }
                                }
                            },
                            "relationships": {
                                "type": "array",
                                "description": "Relationships between entities",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity_1": {"type": "string"},
                                        "entity_2": {"type": "string"},
                                        "relationship_type": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "privacy_impact": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                                    }
                                }
                            }
                        },
                        "required": ["entities", "relationships"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assess_compliance_requirements",
                    "description": "Assess document for compliance requirements and risks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "jurisdiction": {
                                "type": "string",
                                "enum": ["eu_gdpr", "africa_ndpr", "us_hipaa", "apac_pdpa", "unknown"],
                                "description": "Applicable regulatory jurisdiction"
                            },
                            "pii_detected": {
                                "type": "array",
                                "description": "PII elements detected",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "content": {"type": "string"},
                                        "risk_level": {"type": "string"}
                                    }
                                }
                            },
                            "compliance_score": {
                                "type": "number",
                                "description": "Overall compliance score (0-1)"
                            },
                            "risk_flags": {
                                "type": "array",
                                "description": "Identified risk flags",
                                "items": {"type": "string"}
                            },
                            "recommendations": {
                                "type": "array",
                                "description": "Compliance recommendations",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["jurisdiction", "compliance_score"]
                    }
                }
            }
        ]
    
    def _encode_file(self, file_path: Path) -> str:
        """Encode file for API processing"""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _create_processing_messages(
        self, 
        encoded_file: str, 
        task_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create messages for document processing"""
        
        # Determine file type
        file_ext = task_context.get("file_extension", ".pdf").lower()
        is_image = file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
        
        system_prompt = """
        You are DocBridgeGuard 2.0, an advanced AI agent specialized in compliance-first document processing.
        
        Your capabilities include:
        1. Intelligent OCR and text extraction
        2. Table detection and structure preservation
        3. Entity recognition and relationship mapping (bridge extraction)
        4. Compliance assessment and risk analysis
        5. PII detection and privacy impact evaluation
        
        Process the document thoroughly and use the available tools to:
        1. Extract all text content and tables accurately
        2. Identify entities and their relationships (bridges)
        3. Assess compliance requirements and risks
        
        Maintain high accuracy and provide confidence scores for all extractions.
        Consider privacy implications and regulatory requirements throughout.
        """
        
        user_prompt = f"""
        Please process this document comprehensively:
        
        Document Context:
        - Task: {task_context.get('task_type', 'full_processing')}
        - Compliance Profile: {task_context.get('compliance_profile', 'eu_gdpr')}
        - Processing Requirements: {task_context.get('requirements', 'standard')}
        
        Use all available tools to:
        1. Extract text and structural elements
        2. Identify entities and relationships
        3. Assess compliance and privacy implications
        
        Provide detailed, accurate results with confidence scores.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Add file content
        if is_image:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_file}",
                    "detail": "high"
                }
            })
        else:
            # For PDFs, we'd need to convert to images first
            # For now, treat as text-based processing
            messages[1]["content"].append({
                "type": "text",
                "text": f"Document file provided (base64 encoded, {len(encoded_file)} chars)"
            })
        
        return messages
    
    def _process_response(
        self, 
        response, 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process OpenAI response and tool calls"""
        
        result = {
            "text": "",
            "tables": [],
            "entities": [],
            "bridges": [],
            "compliance": {},
            "metadata": {}
        }
        
        message = response.choices[0].message
        
        # Process tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "extract_document_structure":
                    result["text"] = function_args.get("extracted_text", "")
                    result["tables"] = function_args.get("tables", [])
                    result["metadata"].update(function_args.get("metadata", {}))
                    result["document_type"] = function_args.get("document_type", "other")
                
                elif function_name == "identify_entities_and_relationships":
                    result["entities"] = function_args.get("entities", [])
                    
                    # Convert relationships to Bridge objects
                    relationships = function_args.get("relationships", [])
                    bridges = []
                    
                    for rel in relationships:
                        try:
                            bridge = Bridge(
                                entity_1=rel["entity_1"],
                                entity_2=rel["entity_2"],
                                entity_1_type=EntityType(rel.get("entity_1_type", "other")),
                                entity_2_type=EntityType(rel.get("entity_2_type", "other")),
                                relationship=rel["relationship_type"],
                                confidence_score=rel.get("confidence", 0.8),
                                privacy_impact=PrivacyImpact(rel.get("privacy_impact", "low")),
                                extraction_method="openai_agent_function_call"
                            )
                            bridges.append(bridge)
                        except Exception as e:
                            logger.warning(f"Failed to create bridge: {e}")
                    
                    result["bridges"] = bridges
                
                elif function_name == "assess_compliance_requirements":
                    result["compliance"] = function_args
        
        # If no tool calls, try to parse content directly
        elif message.content:
            try:
                # Try to parse as JSON
                parsed_content = json.loads(message.content)
                result.update(parsed_content)
            except json.JSONDecodeError:
                # Treat as plain text
                result["text"] = message.content
        
        return result
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        
        scores = []
        
        # Text extraction confidence
        if result.get("metadata", {}).get("confidence_score"):
            scores.append(result["metadata"]["confidence_score"])
        else:
            # Estimate based on text length
            text_length = len(result.get("text", ""))
            if text_length > 100:
                scores.append(0.8)
            elif text_length > 10:
                scores.append(0.6)
            else:
                scores.append(0.3)
        
        # Bridge extraction confidence
        bridges = result.get("bridges", [])
        if bridges:
            bridge_confidences = [b.confidence_score for b in bridges]
            scores.append(sum(bridge_confidences) / len(bridge_confidences))
        
        # Compliance assessment confidence
        compliance_score = result.get("compliance", {}).get("compliance_score")
        if compliance_score is not None:
            scores.append(compliance_score)
        
        # Return average if we have scores, otherwise default
        return sum(scores) / len(scores) if scores else 0.7
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return len(text.split()) * 1.3  # Rough estimate