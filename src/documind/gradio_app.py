"""
Gradio web interface for DocBridgeGuard 2.0 with AI Agents
"""

import os
import json
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .agents import (
    OpenAIAgent, MistralAgent, ComplianceAgent, 
    BridgeAgent, AgentCoordinator
)
from .utils.comparison import ComparisonEngine
from .utils.report_generator import ReportGenerator
from .models import ProcessingConfig, RedactionLevel


# Global variables for agents
coordinator = None
report_generator = None


def initialize_agents():
    """Initialize AI agents with API keys"""
    global coordinator, report_generator
    
    agents = []
    
    # Initialize OpenAI agent if API key available
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        openai_agent = OpenAIAgent(
            api_key=openai_key,
            model="gpt-4-vision-preview"
        )
        agents.append(openai_agent)
    
    # Initialize Mistral agent if API key available
    mistral_key = os.getenv('MISTRAL_API_KEY')
    if mistral_key:
        mistral_agent = MistralAgent(
            api_key=mistral_key,
            model="mistral-large-latest"
        )
        agents.append(mistral_agent)
    
    # Initialize specialized agents
    compliance_agent = ComplianceAgent(compliance_profile="eu_gdpr")
    bridge_agent = BridgeAgent(confidence_threshold=0.7)
    
    agents.extend([compliance_agent, bridge_agent])
    
    # Initialize coordinator
    coordinator = AgentCoordinator(agents=agents)
    
    # Initialize report generator
    report_generator = ReportGenerator()
    
    return len(agents)


def process_single_document(
    file,
    provider_choice: str,
    compliance_profile: str,
    redaction_level: str,
    enable_bridges: bool,
    strategy: str
) -> Tuple[str, str, str, str, str]:
    """
    Process a single document with selected configuration
    
    Returns:
        Tuple of (status, results_json, compliance_info, bridges_info, processing_time)
    """
    global coordinator
    
    if not coordinator:
        num_agents = initialize_agents()
        if num_agents == 0:
            return (
                "❌ Error: No agents available. Please check API keys.",
                "{}", "", "", ""
            )
    
    if file is None:
        return (
            "⚠️ Please upload a document first.",
            "{}", "", "", ""
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        # Create processing configuration
        config = ProcessingConfig(
            redaction_level=RedactionLevel(redaction_level),
            enable_bridge_extraction=enable_bridges
        )
        
        # Determine primary agent
        primary_agent = provider_choice.lower() if provider_choice != "Auto" else None
        
        # Process document
        result = coordinator.process_document(
            file_path=tmp_file_path,
            strategy=strategy.lower(),
            primary_agent=primary_agent,
            processing_config=config
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Format results
        status = f"✅ Processing completed successfully!"
        
        # Extract key information
        results_summary = {
            "document_id": result.document_id,
            "filename": result.original_filename,
            "provider_used": result.provider_used,
            "processing_time": f"{result.processing_time_seconds:.2f}s",
            "status": result.status.value,
            "extracted_text_length": len(result.extracted_text),
            "tables_found": len(result.tables),
            "bridges_extracted": len(result.bridges)
        }
        
        # Compliance information
        compliance_info = f"""
**Compliance Analysis:**
- Document Type: {result.compliance_metadata.document_type.value}
- Jurisdiction: {result.compliance_metadata.jurisdiction.value}
- Compliance Score: {result.compliance_metadata.compliance_score:.2f}/1.0
- PII Redactions: {result.compliance_metadata.redactions_count}
- Legal Basis: {result.compliance_metadata.legal_basis}
- Risk Flags: {len(result.compliance_metadata.risk_flags)}
        """
        
        # Bridge information
        if result.bridges:
            bridge_summary = []
            privacy_counts = {}
            
            for bridge in result.bridges:
                bridge_summary.append(f"• {bridge.entity_1} ↔ {bridge.entity_2} ({bridge.relationship})")
                privacy_level = bridge.privacy_impact.value
                privacy_counts[privacy_level] = privacy_counts.get(privacy_level, 0) + 1
            
            bridges_info = f"""
**Bridge Extraction Results:**
- Total Bridges: {len(result.bridges)}
- Privacy Impact Distribution:
"""
            for impact, count in privacy_counts.items():
                bridges_info += f"  - {impact.title()}: {count}\n"
            
            bridges_info += f"\n**Sample Relationships:**\n" + "\n".join(bridge_summary[:5])
            
            if len(bridge_summary) > 5:
                bridges_info += f"\n... and {len(bridge_summary) - 5} more"
        else:
            bridges_info = "**Bridge Extraction Results:**\nNo relationships detected in document."
        
        processing_time = f"**Processing Time:** {result.processing_time_seconds:.2f} seconds"
        
        return (
            status,
            json.dumps(results_summary, indent=2),
            compliance_info,
            bridges_info,
            processing_time
        )
        
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "{}",
            "Compliance analysis unavailable due to processing error.",
            "Bridge extraction unavailable due to processing error.",
            "Processing failed"
        )


def compare_providers(
    file,
    compliance_profile: str,
    redaction_level: str,
    enable_bridges: bool
) -> Tuple[str, str, str, str]:
    """
    Compare OpenAI and Mistral providers on the same document
    
    Returns:
        Tuple of (status, comparison_results, visualization, recommendations)
    """
    global coordinator
    
    if not coordinator:
        initialize_agents()
    
    if file is None:
        return (
            "⚠️ Please upload a document for comparison.",
            "{}", "", ""
        )
    
    # Check if both providers are available
    openai_available = os.getenv('OPENAI_API_KEY') is not None
    mistral_available = os.getenv('MISTRAL_API_KEY') is not None
    
    if not (openai_available and mistral_available):
        missing = []
        if not openai_available:
            missing.append("OpenAI")
        if not mistral_available:
            missing.append("Mistral")
        
        return (
            f"❌ Missing API keys for: {', '.join(missing)}",
            "{}", "", ""
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        # Create processing configuration
        config = ProcessingConfig(
            redaction_level=RedactionLevel(redaction_level),
            enable_bridge_extraction=enable_bridges
        )
        
        # Process with OpenAI
        openai_result = coordinator.process_document(
            file_path=tmp_file_path,
            strategy="best_agent",
            primary_agent="openai",
            processing_config=config
        )
        
        # Process with Mistral
        mistral_result = coordinator.process_document(
            file_path=tmp_file_path,
            strategy="best_agent",
            primary_agent="mistral",
            processing_config=config
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Compare results
        comparison_engine = ComparisonEngine()
        comparison = comparison_engine.compare_results(openai_result, mistral_result)
        
        # Format comparison results
        status = f"✅ Comparison completed! Winner: {comparison.winner or 'Tie'}"
        
        comparison_summary = {
            "winner": comparison.winner or "Tie",
            "confidence_in_winner": f"{comparison.confidence_in_winner:.2f}",
            "openai_processing_time": f"{openai_result.processing_time_seconds:.2f}s",
            "mistral_processing_time": f"{mistral_result.processing_time_seconds:.2f}s",
            "text_similarity": f"{comparison.comparison_metrics.get('text_similarity', 0):.3f}",
            "bridge_overlap": f"{comparison.comparison_metrics.get('bridge_overlap', 0):.3f}",
            "openai_compliance_score": f"{openai_result.compliance_metadata.compliance_score:.2f}",
            "mistral_compliance_score": f"{mistral_result.compliance_metadata.compliance_score:.2f}"
        }
        
        # Create visualization
        visualization = create_comparison_visualization(comparison)
        
        # Format recommendations
        recommendations = "\n".join([
            f"• {rec}" for rec in comparison.detailed_analysis.get("recommendations", [])
        ])
        
        return (
            status,
            json.dumps(comparison_summary, indent=2),
            visualization,
            f"**Recommendations:**\n{recommendations}" if recommendations else "No specific recommendations."
        )
        
    except Exception as e:
        return (
            f"❌ Comparison failed: {str(e)}",
            "{}",
            "",
            "Comparison unavailable due to error."
        )


def create_comparison_visualization(comparison) -> str:
    """Create visualization for comparison results"""
    
    try:
        metrics = comparison.comparison_metrics
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Processing Time", "Compliance Scores", "Text Quality", "Bridge Quality"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Processing time comparison
        fig.add_trace(
            go.Bar(
                x=["OpenAI", "Mistral"],
                y=[metrics.get("openai_processing_time", 0), metrics.get("mistral_processing_time", 0)],
                name="Processing Time (s)",
                marker_color=["#1f77b4", "#ff7f0e"]
            ),
            row=1, col=1
        )
        
        # Compliance scores
        fig.add_trace(
            go.Bar(
                x=["OpenAI", "Mistral"],
                y=[metrics.get("openai_compliance_score", 0), metrics.get("mistral_compliance_score", 0)],
                name="Compliance Score",
                marker_color=["#2ca02c", "#d62728"]
            ),
            row=1, col=2
        )
        
        # Text quality (similarity and length)
        fig.add_trace(
            go.Bar(
                x=["Text Similarity", "Length Ratio"],
                y=[metrics.get("text_similarity", 0), metrics.get("length_ratio", 0)],
                name="Text Quality",
                marker_color=["#9467bd", "#8c564b"]
            ),
            row=2, col=1
        )
        
        # Bridge quality
        fig.add_trace(
            go.Bar(
                x=["Bridge Overlap", "Avg Confidence"],
                y=[metrics.get("bridge_overlap", 0), 
                   (metrics.get("openai_avg_bridge_confidence", 0) + metrics.get("mistral_avg_bridge_confidence", 0)) / 2],
                name="Bridge Quality",
                marker_color=["#e377c2", "#7f7f7f"]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Provider Comparison Results",
            height=600,
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs="cdn")
        
    except Exception as e:
        return f"<p>Visualization error: {str(e)}</p>"


def batch_process_documents(
    files,
    provider_choice: str,
    compliance_profile: str,
    redaction_level: str,
    enable_bridges: bool
) -> Tuple[str, str, str]:
    """
    Process multiple documents in batch
    
    Returns:
        Tuple of (status, results_table, summary_stats)
    """
    global coordinator
    
    if not coordinator:
        initialize_agents()
    
    if not files:
        return (
            "⚠️ Please upload documents for batch processing.",
            "", ""
        )
    
    try:
        results = []
        failed_files = []
        
        # Create processing configuration
        config = ProcessingConfig(
            redaction_level=RedactionLevel(redaction_level),
            enable_bridge_extraction=enable_bridges
        )
        
        # Process each file
        for file in files:
            try:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                    tmp_file.write(file.read())
                    tmp_file_path = tmp_file.name
                
                # Process document
                result = coordinator.process_document(
                    file_path=tmp_file_path,
                    strategy="best_agent",
                    primary_agent=provider_choice.lower() if provider_choice != "Auto" else None,
                    processing_config=config
                )
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                # Extract results for table
                results.append({
                    "Filename": result.original_filename,
                    "Status": result.status.value,
                    "Provider": result.provider_used,
                    "Processing Time (s)": f"{result.processing_time_seconds:.2f}",
                    "Compliance Score": f"{result.compliance_metadata.compliance_score:.2f}",
                    "Bridges": len(result.bridges),
                    "PII Redactions": result.compliance_metadata.redactions_count,
                    "Text Length": len(result.extracted_text)
                })
                
            except Exception as e:
                failed_files.append(f"{file.name}: {str(e)}")
        
        # Create results table
        if results:
            df = pd.DataFrame(results)
            results_table = df.to_html(index=False, classes="table table-striped")
            
            # Calculate summary statistics
            successful_count = len(results)
            failed_count = len(failed_files)
            total_time = sum(float(r["Processing Time (s)"]) for r in results)
            avg_compliance = sum(float(r["Compliance Score"]) for r in results) / len(results)
            total_bridges = sum(int(r["Bridges"]) for r in results)
            
            summary_stats = f"""
**Batch Processing Summary:**
- Total Files: {successful_count + failed_count}
- Successful: {successful_count}
- Failed: {failed_count}
- Total Processing Time: {total_time:.2f}s
- Average Compliance Score: {avg_compliance:.2f}
- Total Bridges Extracted: {total_bridges}
            """
            
            if failed_files:
                summary_stats += f"\n\n**Failed Files:**\n" + "\n".join([f"• {f}" for f in failed_files])
            
            status = f"✅ Batch processing completed! {successful_count}/{successful_count + failed_count} files processed successfully."
            
        else:
            results_table = "<p>No files processed successfully.</p>"
            summary_stats = "All files failed to process."
            status = "❌ Batch processing failed for all files."
        
        return (status, results_table, summary_stats)
        
    except Exception as e:
        return (
            f"❌ Batch processing error: {str(e)}",
            "", ""
        )


def analyze_document_relationships(
    file,
    confidence_threshold: float
) -> Tuple[str, str, str]:
    """
    Analyze relationships in document with visualization
    
    Returns:
        Tuple of (status, network_analysis, relationship_insights)
    """
    global coordinator
    
    if not coordinator:
        initialize_agents()
    
    if file is None:
        return (
            "⚠️ Please upload a document for relationship analysis.",
            "", ""
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        # Process with focus on bridge extraction
        config = ProcessingConfig(
            enable_bridge_extraction=True,
            min_confidence_threshold=confidence_threshold
        )
        
        result = coordinator.process_document(
            file_path=tmp_file_path,
            strategy="best_agent",
            processing_config=config
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        if not result.bridges:
            return (
                "ℹ️ No relationships found in document.",
                "No network to visualize.",
                "No relationship insights available."
            )
        
        # Analyze relationships
        from .bridges.analyzer import RelationshipAnalyzer
        analyzer = RelationshipAnalyzer()
        analysis = analyzer.analyze_bridges(result.bridges)
        
        # Create network visualization
        network_viz = create_network_visualization(result.bridges)
        
        # Format insights
        insights = f"""
**Relationship Analysis:**
- Total Relationships: {analysis['total_bridges']}
- Unique Entities: {analysis['entity_count']}
- Network Density: {analysis['network_metrics'].get('density', 0):.3f}
- Compliance Score: {analysis['compliance_score']:.2f}

**Privacy Impact Distribution:**
"""
        for impact, count in analysis['privacy_distribution'].items():
            percentage = (count / analysis['total_bridges'] * 100) if analysis['total_bridges'] > 0 else 0
            insights += f"- {impact.title()}: {count} ({percentage:.1f}%)\n"
        
        insights += f"\n**Relationship Types:**\n"
        for rel_type, count in list(analysis['relationship_types'].items())[:5]:
            insights += f"- {rel_type.replace('_', ' ').title()}: {count}\n"
        
        if analysis['high_risk_entities']:
            insights += f"\n**High-Risk Entities:**\n"
            for entity_info in analysis['high_risk_entities'][:3]:
                insights += f"- {entity_info['entity']} (Risk Score: {entity_info['risk_score']})\n"
        
        status = f"✅ Relationship analysis completed! Found {len(result.bridges)} relationships."
        
        return (status, network_viz, insights)
        
    except Exception as e:
        return (
            f"❌ Relationship analysis failed: {str(e)}",
            "", ""
        )


def create_network_visualization(bridges) -> str:
    """Create network visualization of relationships"""
    
    try:
        import networkx as nx
        
        # Build graph
        G = nx.Graph()
        edge_data = []
        
        for bridge in bridges:
            G.add_edge(bridge.entity_1, bridge.entity_2)
            edge_data.append({
                "source": bridge.entity_1,
                "target": bridge.entity_2,
                "relationship": bridge.relationship,
                "confidence": bridge.confidence_score,
                "privacy_impact": bridge.privacy_impact.value
            })
        
        # Get node positions
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plotly figure
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Count connections
            connections = len(list(G.neighbors(node)))
            node_info.append(f"{node}<br>Connections: {connections}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_info,
            textposition="middle center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Document Relationship Network",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Relationship network visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="grey", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig.to_html(include_plotlyjs="cdn")
        
    except Exception as e:
        return f"<p>Network visualization error: {str(e)}</p>"


def create_gradio_interface():
    """Create the main Gradio interface"""
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .tab-nav {
        border-bottom: 2px solid #e1e5e9;
    }
    .selected {
        border-bottom: 2px solid #007bff !important;
    }
    """
    
    with gr.Blocks(css=css, title="DocBridgeGuard 2.0 - AI Agents") as demo:
        
        gr.Markdown("""
        # 🛡️ DocBridgeGuard 2.0: AI Agents
        ### Enterprise Compliance-First OCR Pipeline with Intelligent Agents
        
        Process documents with AI agents, extract relationships, and ensure regulatory compliance.
        """)
        
        # API Key Status
        with gr.Row():
            with gr.Column():
                openai_status = "✅ OpenAI Available" if os.getenv('OPENAI_API_KEY') else "❌ OpenAI Key Missing"
                mistral_status = "✅ Mistral Available" if os.getenv('MISTRAL_API_KEY') else "❌ Mistral Key Missing"
                gr.Markdown(f"**API Status:** {openai_status} | {mistral_status}")
        
        with gr.Tabs():
            
            # Single Document Processing Tab
            with gr.TabItem("📄 Process Document"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload & Configuration")
                        
                        file_input = gr.File(
                            label="Upload Document",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                            type="binary"
                        )
                        
                        provider_choice = gr.Dropdown(
                            choices=["Auto", "OpenAI", "Mistral"],
                            value="Auto",
                            label="Primary Agent"
                        )
                        
                        compliance_profile = gr.Dropdown(
                            choices=["eu_gdpr", "africa_ndpr", "us_hipaa"],
                            value="eu_gdpr",
                            label="Compliance Profile"
                        )
                        
                        redaction_level = gr.Dropdown(
                            choices=["none", "basic", "moderate", "strict", "maximum"],
                            value="moderate",
                            label="PII Redaction Level"
                        )
                        
                        enable_bridges = gr.Checkbox(
                            value=True,
                            label="Enable Bridge Extraction"
                        )
                        
                        strategy = gr.Dropdown(
                            choices=["Sequential", "Parallel", "Best_Agent"],
                            value="Sequential",
                            label="Processing Strategy"
                        )
                        
                        process_btn = gr.Button("🚀 Process Document", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        
                        status_output = gr.Markdown()
                        
                        with gr.Tabs():
                            with gr.TabItem("Summary"):
                                results_json = gr.Code(language="json")
                            
                            with gr.TabItem("Compliance"):
                                compliance_info = gr.Markdown()
                            
                            with gr.TabItem("Relationships"):
                                bridges_info = gr.Markdown()
                            
                            with gr.TabItem("Performance"):
                                processing_time = gr.Markdown()
                
                process_btn.click(
                    fn=process_single_document,
                    inputs=[
                        file_input, provider_choice, compliance_profile,
                        redaction_level, enable_bridges, strategy
                    ],
                    outputs=[
                        status_output, results_json, compliance_info,
                        bridges_info, processing_time
                    ]
                )
            
            # Provider Comparison Tab
            with gr.TabItem("⚖️ Compare Providers"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Compare OpenAI vs Mistral")
                        
                        compare_file_input = gr.File(
                            label="Upload Document",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                            type="binary"
                        )
                        
                        compare_compliance_profile = gr.Dropdown(
                            choices=["eu_gdpr", "africa_ndpr", "us_hipaa"],
                            value="eu_gdpr",
                            label="Compliance Profile"
                        )
                        
                        compare_redaction_level = gr.Dropdown(
                            choices=["none", "basic", "moderate", "strict", "maximum"],
                            value="moderate",
                            label="PII Redaction Level"
                        )
                        
                        compare_enable_bridges = gr.Checkbox(
                            value=True,
                            label="Enable Bridge Extraction"
                        )
                        
                        compare_btn = gr.Button("🥊 Compare Providers", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Comparison Results")
                        
                        compare_status = gr.Markdown()
                        
                        with gr.Tabs():
                            with gr.TabItem("Results"):
                                comparison_results = gr.Code(language="json")
                            
                            with gr.TabItem("Visualization"):
                                comparison_viz = gr.HTML()
                            
                            with gr.TabItem("Recommendations"):
                                recommendations = gr.Markdown()
                
                compare_btn.click(
                    fn=compare_providers,
                    inputs=[
                        compare_file_input, compare_compliance_profile,
                        compare_redaction_level, compare_enable_bridges
                    ],
                    outputs=[
                        compare_status, comparison_results,
                        comparison_viz, recommendations
                    ]
                )
            
            # Batch Processing Tab
            with gr.TabItem("📊 Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Configuration")
                        
                        batch_files_input = gr.File(
                            label="Upload Multiple Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                            type="binary"
                        )
                        
                        batch_provider_choice = gr.Dropdown(
                            choices=["Auto", "OpenAI", "Mistral"],
                            value="Auto",
                            label="Primary Agent"
                        )
                        
                        batch_compliance_profile = gr.Dropdown(
                            choices=["eu_gdpr", "africa_ndpr", "us_hipaa"],
                            value="eu_gdpr",
                            label="Compliance Profile"
                        )
                        
                        batch_redaction_level = gr.Dropdown(
                            choices=["none", "basic", "moderate", "strict", "maximum"],
                            value="moderate",
                            label="PII Redaction Level"
                        )
                        
                        batch_enable_bridges = gr.Checkbox(
                            value=True,
                            label="Enable Bridge Extraction"
                        )
                        
                        batch_process_btn = gr.Button("🚀 Process Batch", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Batch Results")
                        
                        batch_status = gr.Markdown()
                        
                        with gr.Tabs():
                            with gr.TabItem("Results Table"):
                                batch_results_table = gr.HTML()
                            
                            with gr.TabItem("Summary"):
                                batch_summary = gr.Markdown()
                
                batch_process_btn.click(
                    fn=batch_process_documents,
                    inputs=[
                        batch_files_input, batch_provider_choice,
                        batch_compliance_profile, batch_redaction_level,
                        batch_enable_bridges
                    ],
                    outputs=[batch_status, batch_results_table, batch_summary]
                )
            
            # Relationship Analysis Tab
            with gr.TabItem("🔍 Relationship Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Network Analysis Configuration")
                        
                        analysis_file_input = gr.File(
                            label="Upload Document",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                            type="binary"
                        )
                        
                        confidence_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Confidence Threshold"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Relationships", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Relationship Network")
                        
                        analysis_status = gr.Markdown()
                        
                        with gr.Tabs():
                            with gr.TabItem("Network Visualization"):
                                network_viz = gr.HTML()
                            
                            with gr.TabItem("Insights"):
                                relationship_insights = gr.Markdown()
                
                analyze_btn.click(
                    fn=analyze_document_relationships,
                    inputs=[analysis_file_input, confidence_threshold],
                    outputs=[analysis_status, network_viz, relationship_insights]
                )
        
        gr.Markdown("""
        ---
        **DocBridgeGuard 2.0** - Built with ❤️ for enterprise compliance and data privacy.
        
        Features AI agents powered by OpenAI and Mistral for intelligent document processing.
        """)
    
    return demo


def launch_gradio(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False
):
    """Launch the Gradio interface"""
    
    # Initialize agents
    try:
        num_agents = initialize_agents()
        print(f"Initialized {num_agents} agents")
    except Exception as e:
        print(f"Warning: Failed to initialize agents: {e}")
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_tips=True,
        enable_queue=True
    )


if __name__ == "__main__":
    launch_gradio()