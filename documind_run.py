#!/usr/bin/env python3
"""
Simple launcher for Mind Enhanced Platform
Runs a simplified version with core functionality
"""

import os
import sys
import gradio as gr
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_api_keys():
    """Check available API keys"""
    openai_available = bool(os.getenv('OPENAI_API_KEY'))
    mistral_available = bool(os.getenv('MISTRAL_API_KEY'))
    
    return {
        "openai": openai_available,
        "mistral": mistral_available,
        "any_available": openai_available or mistral_available
    }

def analyze_document_simple(
    files,
    selected_providers: List[str],
    selected_jurisdictions: List[str],
    risk_assessment_enabled: bool = True
) -> Tuple[str, str, str, str, str]:
    """
    Simplified document analysis with demo functionality
    """
    
    if not files or len(files) == 0:
        return "⚠️ Please upload at least one document", "", "", "", ""
    
    # Handle multiple files
    all_content = []
    all_filenames = []
    
    try:
        # Process each uploaded file
        for file in files:
            if isinstance(file, str):
                # This is a file path from Gradio
                filename = Path(file).name
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                all_content.append(content)
                all_filenames.append(filename)
            else:
                # Fallback for other types
                all_filenames.append('uploaded_document')
                all_content.append(str(file)[:1000])
                
    except Exception as e:
        return f"❌ Error reading files: {str(e)}", "", "", "", ""
    
    # Combine content for analysis
    combined_content = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(all_content)
    
    # Simple analysis on combined content
    analysis_results = perform_simple_analysis(combined_content, selected_providers, selected_jurisdictions)
    
    # Add file information to results
    analysis_results["files_processed"] = len(all_filenames)
    analysis_results["filenames"] = all_filenames
    
    # Generate outputs
    if len(all_filenames) == 1:
        status = f"✅ Analysis completed for {all_filenames[0]}"
    else:
        status = f"✅ Batch analysis completed for {len(all_filenames)} documents: {', '.join(all_filenames[:3])}{'...' if len(all_filenames) > 3 else ''}"
    
    # Compliance matrix
    compliance_matrix = generate_simple_compliance_matrix(analysis_results)
    
    # Analytics dashboard
    analytics = generate_simple_analytics(analysis_results)
    
    # Comparison summary
    comparison = generate_simple_comparison(analysis_results)
    
    # Recommendations
    recommendations = generate_simple_recommendations(analysis_results)
    
    return status, compliance_matrix, analytics, comparison, recommendations

def perform_simple_analysis(content: str, providers: List[str], jurisdictions: List[str]) -> Dict:
    """Perform simplified compliance analysis"""
    
    # Simple PII detection
    pii_detected = {
        "emails": len([x for x in content.split() if '@' in x]),
        "phones": len([x for x in content.split() if '+' in x or '(' in x]),
        "mozambican_ids": content.count('BI:') + content.count('BI '),
        "names": len([x for x in content.split() if x.istitle() and len(x) > 2])
    }
    
    # Simple risk assessment
    risk_factors = []
    risk_score = 0.0
    
    if pii_detected["emails"] > 0:
        risk_factors.append("Email addresses detected")
        risk_score += 0.2
    
    if pii_detected["phones"] > 0:
        risk_factors.append("Phone numbers detected")
        risk_score += 0.2
    
    if pii_detected["mozambican_ids"] > 0:
        risk_factors.append("Mozambican ID numbers detected")
        risk_score += 0.4
    
    if "médico" in content.lower() or "health" in content.lower():
        risk_factors.append("Medical/health data detected")
        risk_score += 0.3
    
    if "consent" not in content.lower() and "consentimento" not in content.lower():
        risk_factors.append("No consent indicators found")
        risk_score += 0.3
    
    # Jurisdiction-specific analysis
    jurisdiction_analysis = {}
    for jurisdiction in jurisdictions:
        if jurisdiction == "Mozambique":
            jurisdiction_analysis[jurisdiction] = {
                "status": "❌ Non-Compliant" if risk_score > 0.6 else "⚠️ Needs Review" if risk_score > 0.3 else "✅ Compliant",
                "risk_level": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low",
                "violations": risk_factors if risk_score > 0.6 else [],
                "requirements": ["Explicit consent required", "Data localization preferred", "Cross-border restrictions apply"]
            }
        elif jurisdiction == "EU (GDPR)":
            jurisdiction_analysis[jurisdiction] = {
                "status": "⚠️ Needs Review" if risk_score > 0.5 else "✅ Compliant",
                "risk_level": "High" if risk_score > 0.6 else "Medium" if risk_score > 0.3 else "Low",
                "violations": risk_factors if risk_score > 0.5 else [],
                "requirements": ["Lawful basis required", "Data subject rights", "Retention limitations"]
            }
        else:
            jurisdiction_analysis[jurisdiction] = {
                "status": "⚠️ Needs Review",
                "risk_level": "Medium",
                "violations": [],
                "requirements": ["Review applicable regulations"]
            }
    
    # Provider analysis
    provider_analysis = {}
    api_keys = check_api_keys()
    
    for provider in providers:
        if provider == "OpenAI":
            provider_analysis[provider] = {
                "available": api_keys["openai"],
                "status": "✅ Available" if api_keys["openai"] else "❌ API key not configured",
                "strengths": ["High accuracy OCR", "Vision analysis", "Function calling"],
                "compliance_notes": ["US-based provider", "Cross-border considerations"]
            }
        elif provider == "Mistral":
            provider_analysis[provider] = {
                "available": api_keys["mistral"],
                "status": "✅ Available" if api_keys["mistral"] else "❌ API key not configured",
                "strengths": ["European compliance", "Multilingual", "GDPR focused"],
                "compliance_notes": ["EU-based provider", "GDPR advantages"]
            }
        else:
            provider_analysis[provider] = {
                "available": False,
                "status": "🔄 Coming soon",
                "strengths": ["Placeholder functionality"],
                "compliance_notes": ["Integration pending"]
            }
    
    return {
        "content_length": len(content),
        "pii_detected": pii_detected,
        "risk_score": min(risk_score, 1.0),
        "risk_factors": risk_factors,
        "jurisdiction_analysis": jurisdiction_analysis,
        "provider_analysis": provider_analysis,
        "timestamp": datetime.now().isoformat()
    }

def generate_simple_compliance_matrix(analysis: Dict) -> str:
    """Generate HTML compliance matrix"""
    
    matrix_html = """
    <div style="background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #3a3a3a;">
        <h3 style="color: #ffffff; margin-top: 0;">🛡️ Compliance Analysis Matrix</h3>
        <table style="width: 100%; border-collapse: collapse; background: #1a1a1a; color: #ffffff;">
            <thead>
                <tr style="background: #2d2d2d;">
                    <th style="padding: 12px; border: 1px solid #3a3a3a;">Jurisdiction</th>
                    <th style="padding: 12px; border: 1px solid #3a3a3a;">OpenAI</th>
                    <th style="padding: 12px; border: 1px solid #3a3a3a;">Mistral</th>
                    <th style="padding: 12px; border: 1px solid #3a3a3a;">Claude</th>
                    <th style="padding: 12px; border: 1px solid #3a3a3a;">Cohere</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for jurisdiction, data in analysis["jurisdiction_analysis"].items():
        matrix_html += f"""
                <tr style="border-bottom: 1px solid #3a3a3a;">
                    <td style="padding: 10px; border: 1px solid #3a3a3a; font-weight: bold;">{jurisdiction}</td>
                    <td style="padding: 10px; border: 1px solid #3a3a3a;">{data['status']} {data['risk_level']}</td>
                    <td style="padding: 10px; border: 1px solid #3a3a3a;">{data['status']} {data['risk_level']}</td>
                    <td style="padding: 10px; border: 1px solid #3a3a3a;">🔄 Coming Soon</td>
                    <td style="padding: 10px; border: 1px solid #3a3a3a;">🔄 Coming Soon</td>
                </tr>
        """
    
    matrix_html += """
            </tbody>
        </table>
        <p style="color: #b0b0b0; margin-top: 15px; font-size: 0.9em;">
            ✅ Compliant | ⚠️ Needs Review | ❌ Non-Compliant | 🔄 Coming Soon
        </p>
    </div>
    """
    
    return matrix_html

def generate_simple_analytics(analysis: Dict) -> str:
    """Generate analytics HTML"""
    
    risk_score = analysis["risk_score"]
    risk_color = "#ff4444" if risk_score > 0.7 else "#ffaa00" if risk_score > 0.4 else "#44ff44"
    
    analytics_html = f"""
    <div style="background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #3a3a3a;">
        <h3 style="color: #ffffff; margin-top: 0;">📊 Analytics Dashboard</h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: #2d2d2d; padding: 15px; border-radius: 8px;">
                <h4 style="color: #ffffff; margin: 0 0 10px 0;">Privacy Breach Risk</h4>
                <div style="background: #333; border-radius: 10px; height: 20px; position: relative;">
                    <div style="background: {risk_color}; width: {risk_score * 100}%; height: 100%; border-radius: 10px;"></div>
                </div>
                <p style="color: #ffffff; margin: 10px 0 0 0; font-size: 1.2em; font-weight: bold;">
                    {risk_score:.1%} Risk Level
                </p>
            </div>
            
            <div style="background: #2d2d2d; padding: 15px; border-radius: 8px;">
                <h4 style="color: #ffffff; margin: 0 0 10px 0;">PII Detection Summary</h4>
                <ul style="color: #e0e0e0; margin: 0; padding-left: 20px;">
                    <li>Email addresses: {analysis['pii_detected']['emails']}</li>
                    <li>Phone numbers: {analysis['pii_detected']['phones']}</li>
                    <li>Mozambican IDs: {analysis['pii_detected']['mozambican_ids']}</li>
                    <li>Potential names: {analysis['pii_detected']['names']}</li>
                </ul>
            </div>
        </div>
        
        <div style="background: #2d2d2d; padding: 15px; border-radius: 8px;">
            <h4 style="color: #ffffff; margin: 0 0 10px 0;">Risk Factors Identified</h4>
    """
    
    if analysis["risk_factors"]:
        for factor in analysis["risk_factors"]:
            analytics_html += f'<p style="color: #ffaa00; margin: 5px 0;">⚠️ {factor}</p>'
    else:
        analytics_html += '<p style="color: #44ff44;">✅ No significant risk factors detected</p>'
    
    analytics_html += """
        </div>
    </div>
    """
    
    return analytics_html

def generate_simple_comparison(analysis: Dict) -> str:
    """Generate comparison summary"""
    
    comparison_md = "## 🔍 Provider Comparison Summary\n\n"
    
    api_keys = check_api_keys()
    
    if api_keys["any_available"]:
        if api_keys["openai"]:
            comparison_md += "**🤖 OpenAI GPT-4:**\n"
            comparison_md += "- Status: ✅ Available\n"
            comparison_md += "- Strengths: High accuracy, vision analysis, function calling\n"
            comparison_md += "- Compliance: US-based, cross-border considerations\n\n"
        
        if api_keys["mistral"]:
            comparison_md += "**🇪🇺 Mistral Large:**\n"
            comparison_md += "- Status: ✅ Available\n"
            comparison_md += "- Strengths: European compliance, multilingual, GDPR focus\n"
            comparison_md += "- Compliance: EU-based, GDPR advantages\n\n"
    
    comparison_md += "**📋 Analysis Results:**\n"
    comparison_md += f"- Document length: {analysis['content_length']} characters\n"
    comparison_md += f"- Risk score: {analysis['risk_score']:.1%}\n"
    comparison_md += f"- Jurisdictions analyzed: {len(analysis['jurisdiction_analysis'])}\n"
    comparison_md += f"- PII elements found: {sum(analysis['pii_detected'].values())}\n\n"
    
    # Recommendations based on analysis
    if analysis['risk_score'] > 0.7:
        comparison_md += "**⚠️ High Risk Detected:**\n"
        comparison_md += "- Consider additional privacy safeguards\n"
        comparison_md += "- Review consent mechanisms\n"
        comparison_md += "- Evaluate data minimization\n"
    elif analysis['risk_score'] > 0.4:
        comparison_md += "**📋 Medium Risk Level:**\n"
        comparison_md += "- Standard compliance review recommended\n"
        comparison_md += "- Monitor for regulatory changes\n"
    else:
        comparison_md += "**✅ Low Risk Profile:**\n"
        comparison_md += "- Current approach appears compliant\n"
        comparison_md += "- Continue standard practices\n"
    
    return comparison_md

def generate_simple_recommendations(analysis: Dict) -> str:
    """Generate smart recommendations"""
    
    recommendations_md = "## 🎯 Smart Recommendations\n\n"
    
    # Jurisdiction-specific recommendations
    for jurisdiction, data in analysis["jurisdiction_analysis"].items():
        recommendations_md += f"### {jurisdiction}\n"
        
        if data["status"].startswith("❌"):
            recommendations_md += "🚨 **Immediate Action Required:**\n"
        elif data["status"].startswith("⚠️"):
            recommendations_md += "⚠️ **Review Recommended:**\n"
        else:
            recommendations_md += "✅ **Compliant Status:**\n"
        
        for req in data["requirements"]:
            recommendations_md += f"- {req}\n"
        
        if data["violations"]:
            recommendations_md += "\n**Violations to Address:**\n"
            for violation in data["violations"]:
                recommendations_md += f"- {violation}\n"
        
        recommendations_md += "\n"
    
    # Provider recommendations
    recommendations_md += "### 🤖 AI Provider Recommendations\n\n"
    
    api_keys = check_api_keys()
    
    if not api_keys["any_available"]:
        recommendations_md += "**⚠️ No API Keys Configured:**\n"
        recommendations_md += "- Set `OPENAI_API_KEY` for OpenAI access\n"
        recommendations_md += "- Set `MISTRAL_API_KEY` for Mistral access\n"
        recommendations_md += "- Platform will work in demo mode without API keys\n\n"
    
    # Mozambique-specific recommendations
    if "Mozambique" in analysis["jurisdiction_analysis"]:
        recommendations_md += "### 🇲🇿 Mozambique-Specific Guidance\n\n"
        recommendations_md += "**Data Protection Law (MDPL) Requirements:**\n"
        recommendations_md += "- Explicit consent required for all personal data processing\n"
        recommendations_md += "- Special categories require additional safeguards\n"
        recommendations_md += "- Consider data localization within Mozambique\n"
        recommendations_md += "- Review cross-border transfer restrictions\n"
        recommendations_md += "- Implement 72-hour breach notification procedures\n"
    
    return recommendations_md

def create_simple_interface():
    """Create simplified Mind Enhanced interface"""
    
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
    """
    
    with gr.Blocks(css=css, title="Mind Enhanced - AI Compliance Analysis") as demo:
        
        # Header
        gr.HTML("""
        <div class="dark-header">
            <h1 style="font-size: 3em; margin: 0; background: linear-gradient(90deg, #4a9eff, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🧠 Mind Enhanced
            </h1>
            <p style="font-size: 1.2em; margin: 10px 0 0 0; color: #b0b0b0;">
                AI Compliance Analysis Platform - Demo Mode
            </p>
        </div>
        """)
        
        # Main interface
        with gr.Row():
            # Left Panel - Configuration
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Upload")
                
                file_input = gr.File(
                    label="Drop files here to upload",
                    file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".docx"],
                    type="filepath",
                    file_count="multiple"
                )
                
                clear_btn = gr.Button(
                    "🗑️ Clear Files",
                    variant="secondary",
                    size="sm"
                )
                
                gr.Markdown("### 🤖 AI Model Selection")
                provider_selection = gr.CheckboxGroup(
                    choices=["OpenAI", "Mistral", "Claude", "Cohere", "Gemini", "Grok"],
                    value=["OpenAI", "Mistral"],
                    label="Select AI Providers"
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
                    label="Select Jurisdictions"
                )
                
                risk_assessment = gr.Checkbox(
                    value=True,
                    label="Enable Guardrails & Risk Assessment"
                )
                
                analyze_btn = gr.Button(
                    "🚀 Analyze Compliance",
                    variant="primary",
                    size="lg"
                )
            
            # Right Panel - Results
            with gr.Column(scale=2):
                status_output = gr.Markdown()
                
                with gr.Tabs():
                    with gr.TabItem("🛡️ Compliance Matrix"):
                        compliance_matrix_output = gr.HTML()
                    
                    with gr.TabItem("📊 Analytics"):
                        analytics_output = gr.HTML()
                    
                    with gr.TabItem("📋 Comparison"):
                        comparison_output = gr.Markdown()
                    
                    with gr.TabItem("🎯 Recommendations"):
                        recommendations_output = gr.Markdown()
        
        # API Status
        api_keys = check_api_keys()
        with gr.Row():
            gr.HTML(f"""
            <div style="background: #2d2d2d; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h4 style="margin: 0; color: #ffffff;">🔑 API Status</h4>
                <p style="margin: 5px 0 0 0; color: #b0b0b0;">
                    OpenAI: {'✅ Connected' if api_keys['openai'] else '❌ Not configured'} | 
                    Mistral: {'✅ Connected' if api_keys['mistral'] else '❌ Not configured'} | 
                    Others: 🔄 Coming soon
                </p>
                <p style="margin: 5px 0 0 0; color: #888; font-size: 0.9em;">
                    Demo mode active - Platform works without API keys for testing
                </p>
            </div>
            """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_document_simple,
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
        
        # Clear button handler
        clear_btn.click(
            fn=lambda: (None, "", "", "", "", ""),
            outputs=[
                file_input,
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
            <p><strong>Mind Enhanced</strong> - AI Compliance Analysis Platform (Demo Mode)</p>
            <p>Demonstrating potential data protection issues across AI models and regulatory jurisdictions</p>
        </div>
        """)
    
    return demo

def main():
    """Main function to launch the platform"""
    
    print("🚀 LAUNCHING MIND ENHANCED PLATFORM")
    print("=" * 50)
    
    # Check API keys
    api_keys = check_api_keys()
    print(f"🔑 OpenAI API Key: {'✅ Available' if api_keys['openai'] else '❌ Not configured'}")
    print(f"🔑 Mistral API Key: {'✅ Available' if api_keys['mistral'] else '❌ Not configured'}")
    
    if not api_keys["any_available"]:
        print("⚠️ Running in demo mode - no API keys configured")
        print("Set OPENAI_API_KEY and/or MISTRAL_API_KEY for full functionality")
    
    print("\n🧠 Creating Mind Enhanced interface...")
    
    try:
        demo = create_simple_interface()
        
        print("✅ Interface created successfully")
        print("\n🌐 Starting web interface...")
        print("📝 Interface will be available at: http://localhost:7860")
        
        # Launch the interface - try multiple ports
        ports_to_try = [7860, 7861, 7862, 7863, 7864]
        launched = False
        
        for port in ports_to_try:
            try:
                print(f"🔄 Trying port {port}...")
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=False,
                    prevent_thread_lock=False
                )
                print(f"✅ Successfully launched on port {port}")
                print(f"🌐 Access your platform at: http://localhost:{port}")
                launched = True
                break
            except Exception as e:
                print(f"❌ Port {port} failed: {str(e)}")
                continue
        
        if not launched:
            print("❌ Could not find an available port. Try manually:")
            print("GRADIO_SERVER_PORT=7865 python documind_run.py")
        
    except Exception as e:
        print(f"❌ Launch error: {e}")
        print("🔧 Check the error details above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)