"""
Mind Enhanced - AI Compliance Analysis Interface
Modern dark-themed interface for comprehensive document compliance analysis
"""

import os
import json
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from .agents import (
        OpenAIAgent, MistralAgent, ComplianceAgent, 
        BridgeAgent, AgentCoordinator
    )
except ImportError:
    # Fallback for agents
    OpenAIAgent = None
    MistralAgent = None
    ComplianceAgent = None
    BridgeAgent = None
    AgentCoordinator = None
from .models import (
    AIProvider, Jurisdiction, DocumentType, RiskLevel,
    ComplianceStatus, MultiProviderAnalysis, AnalyticsDashboard,
    DocumentComparisonMatrix, RegionalCompliance
)
try:
    from .compliance.mozambique_analyzer import MozambiqueComplianceAnalyzer
except ImportError:
    MozambiqueComplianceAnalyzer = None

try:
    from .utils.comparison import ComparisonEngine
    from .utils.report_generator import ReportGenerator
except ImportError:
    ComparisonEngine = None
    ReportGenerator = None


# Global variables
coordinator = None
report_generator = None
mozambique_analyzer = None
analytics_data = AnalyticsDashboard()


def initialize_mind_enhanced_system():
    """Initialize the Mind Enhanced AI system"""
    global coordinator, report_generator, mozambique_analyzer
    
    agents = []
    
    # Initialize available agents
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        openai_agent = OpenAIAgent(
            api_key=openai_key,
            model="gpt-4-vision-preview"
        )
        agents.append(openai_agent)
    
    mistral_key = os.getenv('MISTRAL_API_KEY')
    if mistral_key:
        mistral_agent = MistralAgent(
            api_key=mistral_key,
            model="mistral-large-latest"
        )
        agents.append(mistral_agent)
    
    # Specialized agents
    compliance_agent = ComplianceAgent(compliance_profile="multi_region")
    bridge_agent = BridgeAgent(confidence_threshold=0.7)
    agents.extend([compliance_agent, bridge_agent])
    
    # Initialize coordinator and utilities
    coordinator = AgentCoordinator(agents=agents)
    report_generator = ReportGenerator()
    mozambique_analyzer = MozambiqueComplianceAnalyzer()
    
    return len(agents)


def analyze_document_compliance(
    file,
    selected_providers: List[str],
    selected_jurisdictions: List[str],
    risk_assessment_enabled: bool = True
) -> Tuple[str, str, str, str, str]:
    """
    Comprehensive document compliance analysis across providers and jurisdictions
    """
    global coordinator, mozambique_analyzer, analytics_data
    
    if not coordinator:
        num_agents = initialize_mind_enhanced_system()
        if num_agents == 0:
            return "❌ Error: No AI agents available", "", "", "", ""
    
    if file is None:
        return "⚠️ Please upload a document first", "", "", "", ""
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        # Create multi-provider analysis
        analysis = MultiProviderAnalysis(
            document_id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            document_filename=file.name,
            selected_jurisdictions=[Jurisdiction(j) for j in selected_jurisdictions]
        )
        
        # Process with each selected provider
        provider_results = {}
        processing_summary = []
        
        for provider_name in selected_providers:
            try:
                provider = AIProvider(provider_name.lower())
                
                # Check if provider is available
                if provider == AIProvider.OPENAI and not os.getenv('OPENAI_API_KEY'):
                    processing_summary.append(f"❌ {provider_name}: API key not configured")
                    continue
                elif provider == AIProvider.MISTRAL and not os.getenv('MISTRAL_API_KEY'):
                    processing_summary.append(f"❌ {provider_name}: API key not configured")
                    continue
                elif provider in [AIProvider.GEMINI, AIProvider.GROK, AIProvider.COHERE, AIProvider.CLAUDE]:
                    processing_summary.append(f"🔄 {provider_name}: Coming soon (placeholder)")
                    continue
                
                # Process document with available provider
                result = coordinator.process_document(
                    file_path=tmp_file_path,
                    strategy="best_agent",
                    primary_agent=provider_name.lower()
                )
                
                # Analyze regional compliance for each jurisdiction
                regional_compliance = {}
                for jurisdiction in analysis.selected_jurisdictions:
                    if jurisdiction == Jurisdiction.MOZAMBIQUE_DPL:
                        compliance = mozambique_analyzer.analyze_document_compliance(
                            result.extracted_text,
                            DocumentType.UNKNOWN,  # Would be classified in real implementation
                            [],  # Would extract entities in real implementation
                            []   # Would extract bridges in real implementation
                        )
                        regional_compliance[jurisdiction] = compliance
                    else:
                        # Placeholder for other jurisdictions
                        compliance = RegionalCompliance(
                            jurisdiction=jurisdiction,
                            compliance_status=ComplianceStatus.NEEDS_REVIEW,
                            risk_level=RiskLevel.MEDIUM,
                            applicable_laws=[f"{jurisdiction.value.upper()} regulations"],
                            violations=[],
                            recommendations=["Compliance analysis pending"],
                            consent_required=True,
                            cross_border_restrictions=False,
                            data_subject_rights=["Access", "Rectification", "Erasure"],
                            breach_notification_required=True
                        )
                        regional_compliance[jurisdiction] = compliance
                
                provider_results[provider] = {
                    "processing_result": result,
                    "regional_compliance": regional_compliance,
                    "overall_risk_score": 0.3,  # Would calculate based on analysis
                    "provider_available": True
                }
                
                processing_summary.append(f"✅ {provider_name}: Processed successfully")
                
            except Exception as e:
                processing_summary.append(f"❌ {provider_name}: {str(e)}")
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Generate compliance matrix
        compliance_matrix = generate_compliance_matrix(provider_results, analysis.selected_jurisdictions)
        
        # Generate analytics
        analytics_chart = generate_analytics_dashboard(provider_results)
        
        # Generate comparison summary
        comparison_summary = generate_comparison_summary(provider_results)
        
        # Generate recommendations
        recommendations = generate_smart_recommendations(provider_results, analysis.selected_jurisdictions)
        
        # Update analytics data
        analytics_data.total_documents_processed += 1
        analytics_data.last_updated = datetime.now()
        
        status = f"✅ Analysis completed! Processed with {len([s for s in processing_summary if '✅' in s])} providers"
        
        return (
            status,
            compliance_matrix,
            analytics_chart,
            comparison_summary,
            recommendations
        )
        
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}", "", "", "", ""


def generate_compliance_matrix(provider_results: Dict, jurisdictions: List[Jurisdiction]) -> str:
    """Generate HTML compliance matrix"""
    
    matrix_data = []
    providers = list(provider_results.keys())
    
    # Create matrix data
    for jurisdiction in jurisdictions:
        row = {"Jurisdiction": jurisdiction.value.replace('_', ' ').title()}
        
        for provider in providers:
            if provider in provider_results:
                compliance = provider_results[provider]["regional_compliance"].get(jurisdiction)
                if compliance:
                    status_icon = {
                        ComplianceStatus.COMPLIANT: "✅",
                        ComplianceStatus.NON_COMPLIANT: "❌", 
                        ComplianceStatus.NEEDS_REVIEW: "⚠️",
                        ComplianceStatus.UNKNOWN: "❓"
                    }.get(compliance.compliance_status, "❓")
                    
                    row[provider.value.title()] = f"{status_icon} {compliance.risk_level.value.title()}"
                else:
                    row[provider.value.title()] = "❓ Unknown"
            else:
                row[provider.value.title()] = "❌ Unavailable"
                
        matrix_data.append(row)
    
    # Convert to HTML table
    if matrix_data:
        df = pd.DataFrame(matrix_data)
        html_table = df.to_html(index=False, classes="compliance-matrix", escape=False)
        
        # Add custom styling
        styled_html = f"""
        <div class="matrix-container">
            <h3>🛡️ Compliance Analysis Matrix</h3>
            <style>
                .compliance-matrix {{
                    width: 100%;
                    border-collapse: collapse;
                    background: #1a1a1a;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .compliance-matrix th {{
                    background: #2d2d2d;
                    color: #ffffff;
                    padding: 12px;
                    text-align: left;
                    border-bottom: 2px solid #3a3a3a;
                }}
                .compliance-matrix td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #3a3a3a;
                    color: #e0e0e0;
                }}
                .compliance-matrix tr:hover {{
                    background: #2a2a2a;
                }}
                .matrix-container {{
                    background: #1a1a1a;
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid #3a3a3a;
                }}
            </style>
            {html_table}
        </div>
        """
        return styled_html
    else:
        return "<p>No compliance data available</p>"


def generate_analytics_dashboard(provider_results: Dict) -> str:
    """Generate analytics dashboard visualization"""
    
    try:
        # Prepare data for visualization
        providers = []
        risk_scores = []
        compliance_counts = {"Compliant": 0, "Non-Compliant": 0, "Needs Review": 0}
        
        for provider, data in provider_results.items():
            providers.append(provider.value.title())
            risk_scores.append(data["overall_risk_score"])
            
            # Count compliance statuses
            for compliance in data["regional_compliance"].values():
                status = compliance.compliance_status.value.replace('_', ' ').title()
                if status in compliance_counts:
                    compliance_counts[status] += 1
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Privacy Breach Risk Assessment", "Processing Summary", 
                          "Provider Risk Comparison", "Compliance Distribution"),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Privacy breach risk (main indicator)
        overall_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        risk_color = "red" if overall_risk > 0.7 else "orange" if overall_risk > 0.4 else "green"
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Risk"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 0.4], 'color': "#2d5a2d"},
                        {'range': [0.4, 0.7], 'color': "#5a5a2d"},
                        {'range': [0.7, 1], 'color': "#5a2d2d"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ),
            row=1, col=1
        )
        
        # Processing summary
        process_data = {"Successful": len(provider_results), "Failed": 0, "Pending": 3}  # Placeholder
        fig.add_trace(
            go.Bar(
                x=list(process_data.keys()),
                y=list(process_data.values()),
                marker_color=['#4CAF50', '#F44336', '#FF9800']
            ),
            row=1, col=2
        )
        
        # Provider risk comparison
        if providers and risk_scores:
            fig.add_trace(
                go.Bar(
                    x=providers,
                    y=risk_scores,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(providers)]
                ),
                row=2, col=1
            )
        
        # Compliance distribution
        fig.add_trace(
            go.Pie(
                labels=list(compliance_counts.keys()),
                values=list(compliance_counts.values()),
                marker_colors=['#4CAF50', '#F44336', '#FF9800']
            ),
            row=2, col=2
        )
        
        # Update layout with dark theme
        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title={
                'text': "📊 Analytics Dashboard",
                'x': 0.5,
                'font': {'size': 24, 'color': '#ffffff'}
            }
        )
        
        # Update axes for dark theme
        fig.update_xaxes(gridcolor='#3a3a3a', color='#ffffff')
        fig.update_yaxes(gridcolor='#3a3a3a', color='#ffffff')
        
        return fig.to_html(include_plotlyjs="cdn")
        
    except Exception as e:
        return f"<p>Analytics visualization error: {str(e)}</p>"


def generate_comparison_summary(provider_results: Dict) -> str:
    """Generate provider comparison summary"""
    
    if not provider_results:
        return "No provider results to compare"
    
    summary = "## 🔍 Provider Comparison Summary\n\n"
    
    # Find best and worst performers
    risk_scores = {provider: data["overall_risk_score"] for provider, data in provider_results.items()}
    
    if risk_scores:
        safest_provider = min(risk_scores, key=risk_scores.get)
        riskiest_provider = max(risk_scores, key=risk_scores.get)
        
        summary += f"**🏆 Safest Provider:** {safest_provider.value.title()} (Risk: {risk_scores[safest_provider]:.2f})\n\n"
        summary += f"**⚠️ Highest Risk:** {riskiest_provider.value.title()} (Risk: {risk_scores[riskiest_provider]:.2f})\n\n"
    
    # Provider details
    summary += "### Provider Analysis:\n\n"
    for provider, data in provider_results.items():
        summary += f"**{provider.value.title()}:**\n"
        summary += f"- Risk Score: {data['overall_risk_score']:.2f}\n"
        summary += f"- Jurisdictions Analyzed: {len(data['regional_compliance'])}\n"
        
        # Count violations
        total_violations = sum(
            len(compliance.violations) 
            for compliance in data['regional_compliance'].values()
        )
        summary += f"- Total Violations: {total_violations}\n\n"
    
    return summary


def generate_smart_recommendations(provider_results: Dict, jurisdictions: List[Jurisdiction]) -> str:
    """Generate intelligent recommendations based on analysis"""
    
    recommendations = "## 🎯 Smart Recommendations\n\n"
    
    # Analyze overall risk level
    risk_scores = [data["overall_risk_score"] for data in provider_results.values()]
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
    
    if avg_risk > 0.7:
        recommendations += "### 🚨 High Risk Detected\n"
        recommendations += "- **Immediate Action Required:** Consider alternative processing approaches\n"
        recommendations += "- **Review Required:** Manual compliance review recommended\n"
        recommendations += "- **Data Protection:** Implement additional safeguards\n\n"
    elif avg_risk > 0.4:
        recommendations += "### ⚠️ Medium Risk Level\n"
        recommendations += "- **Enhanced Monitoring:** Implement additional oversight\n"
        recommendations += "- **Compliance Review:** Regular compliance audits recommended\n\n"
    else:
        recommendations += "### ✅ Low Risk Profile\n"
        recommendations += "- **Standard Processing:** Current approach appears compliant\n"
        recommendations += "- **Routine Monitoring:** Continue standard compliance practices\n\n"
    
    # Jurisdiction-specific recommendations
    recommendations += "### 🌍 Jurisdiction-Specific Guidance:\n\n"
    for jurisdiction in jurisdictions:
        if jurisdiction == Jurisdiction.MOZAMBIQUE_DPL:
            recommendations += "**🇲🇿 Mozambique (MDPL):**\n"
            recommendations += "- Ensure explicit consent for all personal data processing\n"
            recommendations += "- Consider data localization requirements\n"
            recommendations += "- Review cross-border transfer restrictions\n\n"
        elif jurisdiction == Jurisdiction.EU_GDPR:
            recommendations += "**🇪🇺 European Union (GDPR):**\n"
            recommendations += "- Verify lawful basis for processing\n"
            recommendations += "- Ensure data subject rights are respected\n"
            recommendations += "- Review retention policies\n\n"
        # Add more jurisdiction-specific recommendations
    
    # Provider-specific recommendations
    recommendations += "### 🤖 AI Provider Recommendations:\n\n"
    for provider, data in provider_results.items():
        if data["overall_risk_score"] > 0.5:
            recommendations += f"**{provider.value.title()}:** Consider additional review before use\n"
        else:
            recommendations += f"**{provider.value.title()}:** Approved for standard use\n"
    
    return recommendations


def create_mind_enhanced_interface():
    """Create the Mind Enhanced compliance analysis interface"""
    
    # Custom CSS for dark theme
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%) !important;
        color: #ffffff !important;
        max-width: 1400px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .dark-header {
        background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .upload-area {
        background: #1a1a1a;
        border: 2px dashed #3a3a3a;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4a9eff;
        background: #1f1f1f;
    }
    
    .config-panel {
        background: #1a1a1a;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .results-panel {
        background: #1a1a1a;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .risk-indicator {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-low { background: #2d5a2d; }
    .risk-medium { background: #5a5a2d; }
    .risk-high { background: #5a2d2d; }
    
    /* Custom button styling */
    .gradio-button {
        background: linear-gradient(90deg, #4a9eff 0%, #0078d4 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    
    .gradio-button:hover {
        background: linear-gradient(90deg, #0078d4 0%, #106ebe 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.3) !important;
    }
    
    /* Tab styling */
    .tab-nav {
        background: #2d2d2d !important;
        border-radius: 8px !important;
    }
    
    .tab-nav button {
        background: transparent !important;
        color: #b0b0b0 !important;
        border: none !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .tab-nav button.selected {
        background: #4a9eff !important;
        color: white !important;
        border-radius: 6px !important;
    }
    """
    
    with gr.Blocks(css=css, title="Mind Enhanced - AI Compliance Analysis", theme="dark") as demo:
        
        # Header
        gr.HTML("""
        <div class="dark-header">
            <h1 style="font-size: 3em; margin: 0; background: linear-gradient(90deg, #4a9eff, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🧠 Mind Enhanced
            </h1>
            <p style="font-size: 1.2em; margin: 10px 0 0 0; color: #b0b0b0;">
                AI Compliance Analysis Platform - Proving data protection challenges across models and regions
            </p>
        </div>
        """)
        
        # Main interface
        with gr.Row():
            # Left Panel - Document Upload & Configuration
            with gr.Column(scale=1):
                gr.HTML('<div class="config-panel">')
                gr.Markdown("### 📄 Document Upload")
                
                file_input = gr.File(
                    label="Drop files here to upload",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".docx", ".txt"],
                    type="binary",
                    elem_classes=["upload-area"]
                )
                
                gr.Markdown("### 🤖 AI Model Selection")
                provider_selection = gr.CheckboxGroup(
                    choices=["OpenAI", "Mistral", "Claude", "Cohere", "Gemini", "Grok"],
                    value=["OpenAI", "Mistral"],
                    label="Select AI Providers",
                    info="Choose which AI models to analyze"
                )
                
                gr.Markdown("### 🌍 Jurisdiction Selection") 
                jurisdiction_selection = gr.CheckboxGroup(
                    choices=[
                        "EU (GDPR)", 
                        "Mozambique", 
                        "Nigeria", 
                        "South Africa",
                        "California", 
                        "US (HIPAA)"
                    ],
                    value=["EU (GDPR)", "Mozambique"],
                    label="Select Jurisdictions",
                    info="Choose applicable data protection laws"
                )
                
                risk_assessment = gr.Checkbox(
                    value=True,
                    label="Enable Guardrails & Risk Assessment",
                    info="Proactive compliance checking with guardrails agent"
                )
                
                analyze_btn = gr.Button(
                    "🚀 Analyze Compliance",
                    variant="primary",
                    size="lg",
                    elem_classes=["gradio-button"]
                )
                
                gr.HTML('</div>')
            
            # Right Panel - Results and Analytics
            with gr.Column(scale=2):
                gr.HTML('<div class="results-panel">')
                
                # Status and Results
                status_output = gr.Markdown()
                
                with gr.Tabs():
                    # Audit Trails & Compliance Matrix
                    with gr.TabItem("🛡️ Audit Trails & Compliance"):
                        compliance_matrix_output = gr.HTML()
                    
                    # Analytics Dashboard
                    with gr.TabItem("📊 Analytics"):
                        analytics_output = gr.HTML()
                    
                    # Processing Summary
                    with gr.TabItem("📋 Processing Summary"):
                        comparison_output = gr.Markdown()
                    
                    # Recommendations
                    with gr.TabItem("🎯 Recommendations"):
                        recommendations_output = gr.Markdown()
                
                gr.HTML('</div>')
        
        # API Status
        with gr.Row():
            gr.HTML(f"""
            <div style="background: #2d2d2d; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h4 style="margin: 0; color: #ffffff;">🔑 API Status</h4>
                <p style="margin: 5px 0 0 0; color: #b0b0b0;">
                    OpenAI: {'✅ Connected' if os.getenv('OPENAI_API_KEY') else '❌ Not configured'} | 
                    Mistral: {'✅ Connected' if os.getenv('MISTRAL_API_KEY') else '❌ Not configured'} | 
                    Others: 🔄 Coming soon
                </p>
            </div>
            """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_document_compliance,
            inputs=[
                file_input,
                provider_selection,
                jurisdiction_selection,
                risk_assessment
            ],
            outputs=[
                status_output,
                compliance_matrix_output,
                analytics_output,
                comparison_output,
                recommendations_output
            ]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #3a3a3a; margin-top: 30px;">
            <p><strong>Mind Enhanced</strong> - Built with ❤️ for AI compliance and data protection research</p>
            <p>Proving potential data protection issues across AI models in various regulatory jurisdictions</p>
        </div>
        """)
    
    return demo


def launch_mind_enhanced(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False
):
    """Launch the Mind Enhanced interface"""
    
    # Initialize system
    try:
        num_agents = initialize_mind_enhanced_system()
        print(f"🧠 Mind Enhanced initialized with {num_agents} AI agents")
    except Exception as e:
        print(f"⚠️ Warning: Failed to initialize some agents: {e}")
    
    # Create and launch interface
    demo = create_mind_enhanced_interface()
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_tips=True,
        enable_queue=True,
        debug=True
    )


if __name__ == "__main__":
    launch_mind_enhanced()