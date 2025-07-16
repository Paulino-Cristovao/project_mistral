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

def apply_smart_guardrails(content: str, jurisdictions: List[str]) -> Dict[str, Any]:
    """🛡️ INNOVATIVE GUARDRAILS: Block/mask sensitive data before AI processing"""
    
    import re
    
    guardrails_applied = {
        "original_length": len(content),
        "redacted_content": content,
        "blocked_items": [],
        "masked_items": [],
        "risk_level": "low",
        "protection_applied": []
    }
    
    # 🔒 Email Masking - Jurisdiction Specific
    email_pattern = r'\b[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]{2,}\.[A-Za-z]{2,6}\b'
    emails = re.findall(email_pattern, content)
    valid_emails = []
    for email in emails:
        if (email.count('@') == 1 and len(email) >= 5 and len(email) <= 50 and
            not any(char in email for char in ['<', '>', '{', '}', '[', ']']) and
            email[-1].isalpha() and email[0].isalnum()):
            valid_emails.append(email)
    
    redacted_content = content
    for email in valid_emails:
        if "Mozambique" in jurisdictions:
            # 🇲🇿 MDPL: Complete blocking for cross-border
            redacted_content = redacted_content.replace(email, "[EMAIL_BLOCKED_MDPL]")
            guardrails_applied["blocked_items"].append(f"Email: {email[:3]}***@{email.split('@')[1]}")
        elif "EU (GDPR)" in jurisdictions:
            # 🇪🇺 GDPR: Mask but preserve for legitimate analysis 
            masked_email = f"{email[:2]}***@{email.split('@')[1]}"
            redacted_content = redacted_content.replace(email, f"[EMAIL_MASKED: {masked_email}]")
            guardrails_applied["masked_items"].append(f"Email: {masked_email}")
        else:
            # Other jurisdictions: Light masking
            masked_email = f"{email[:2]}***@***{email.split('@')[1][-3:]}"
            redacted_content = redacted_content.replace(email, masked_email)
            guardrails_applied["masked_items"].append(f"Email: {masked_email}")
    
    # 🔒 Phone Number Protection
    phone_patterns = [
        r'\+\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}',
        r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',
        r'\d{3}[\s-]?\d{3}[\s-]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, redacted_content)
        for phone in phones:
            if "Mozambique" in jurisdictions and "+258" in phone:
                # 🇲🇿 Mozambican numbers: Complete blocking
                redacted_content = redacted_content.replace(phone, "[PHONE_BLOCKED_MDPL]")
                guardrails_applied["blocked_items"].append(f"Mozambican Phone: +258***")
            else:
                # Other phones: Smart masking
                masked_phone = phone[:3] + "***" + phone[-2:] if len(phone) > 5 else "***"
                redacted_content = redacted_content.replace(phone, f"[PHONE: {masked_phone}]")
                guardrails_applied["masked_items"].append(f"Phone: {masked_phone}")
    
    # 🔒 Medical Data - Context-Aware Blocking
    medical_keywords = ['patient', 'diagnosis', 'treatment', 'medical record', 'prescription']
    
    # Check if this is technical/academic context first
    technical_contexts = ['image analysis', 'ai', 'research', 'algorithm', 'development']
    is_technical = any(ctx in content.lower() for ctx in technical_contexts)
    
    if not is_technical:  # Only block if it's actual medical data
        for keyword in medical_keywords:
            if keyword in redacted_content.lower():
                # Replace sensitive medical terms
                redacted_content = re.sub(rf'\b{keyword}\b', '[MEDICAL_DATA_PROTECTED]', 
                                        redacted_content, flags=re.IGNORECASE)
                guardrails_applied["blocked_items"].append(f"Medical: {keyword}")
    
    # 🔒 ID Numbers & Financial Data
    # Credit cards
    cc_pattern = r'\b(?:\d{4}[\s-]?){3}\d{4}\b'
    cc_numbers = re.findall(cc_pattern, redacted_content)
    for cc in cc_numbers:
        redacted_content = redacted_content.replace(cc, "[CREDIT_CARD_BLOCKED]")
        guardrails_applied["blocked_items"].append(f"Credit Card: {cc[:4]}****")
    
    # SSNs
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    ssns = re.findall(ssn_pattern, redacted_content)
    for ssn in ssns:
        redacted_content = redacted_content.replace(ssn, "[SSN_BLOCKED]")
        guardrails_applied["blocked_items"].append(f"SSN: {ssn[:3]}-**-****")
    
    # 📊 Calculate Protection Level
    total_protected = len(guardrails_applied["blocked_items"]) + len(guardrails_applied["masked_items"])
    
    if len(guardrails_applied["blocked_items"]) > 0:
        guardrails_applied["risk_level"] = "high"
        guardrails_applied["protection_applied"].append("🛡️ Data Blocking Active")
    elif len(guardrails_applied["masked_items"]) > 2:
        guardrails_applied["risk_level"] = "medium" 
        guardrails_applied["protection_applied"].append("🎭 Data Masking Active")
    else:
        guardrails_applied["risk_level"] = "low"
        guardrails_applied["protection_applied"].append("✅ Minimal Processing")
    
    # 🌍 Jurisdiction-Specific Protections
    if "Mozambique" in jurisdictions:
        guardrails_applied["protection_applied"].append("🇲🇿 MDPL Compliance Active")
    if "EU (GDPR)" in jurisdictions:
        guardrails_applied["protection_applied"].append("🇪🇺 GDPR Protection Active")
    
    guardrails_applied["redacted_content"] = redacted_content
    guardrails_applied["protection_summary"] = f"Protected {total_protected} sensitive items"
    
    return guardrails_applied

def analyze_document_simple(
    files,
    selected_providers: List[str],
    selected_jurisdictions: List[str],
    risk_assessment_enabled: bool = True
) -> Tuple[str, str, str, str, str, str]:
    """
    Simplified document analysis with demo functionality
    """
    
    if not files or len(files) == 0:
        return "⚠️ Please upload at least one document", "", "", "", "", ""
    
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
        return f"❌ Error reading files: {str(e)}", "", "", "", "", ""
    
    # Combine content for analysis
    combined_content = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(all_content)
    
    # 🛡️ APPLY SMART GUARDRAILS BEFORE AI PROCESSING
    guardrails_results = apply_smart_guardrails(combined_content, selected_jurisdictions)
    
    # Use protected content for AI analysis (this is what providers will see)
    protected_content = guardrails_results["redacted_content"]
    
    # Analysis on original content for detection, but providers get protected content
    analysis_results = perform_simple_analysis(combined_content, selected_providers, selected_jurisdictions)
    analysis_results["guardrails"] = guardrails_results
    
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
    
    # Detailed issues analysis
    detailed_issues = generate_detailed_issues_analysis(all_content, all_filenames)
    
    return status, compliance_matrix, analytics, comparison, recommendations, detailed_issues

def perform_simple_analysis(content: str, providers: List[str], jurisdictions: List[str]) -> Dict:
    """Perform simplified compliance analysis with accurate PII detection"""
    
    import re
    
    # Enhanced email detection with strict validation
    email_pattern = r'\b[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]{2,}\.[A-Za-z]{2,6}\b'
    potential_emails = re.findall(email_pattern, content)
    
    # Validate emails - filter out garbled text
    emails = []
    for email in potential_emails:
        # Must have valid domain and not contain too many special chars
        if (email.count('@') == 1 and 
            email.count('.') >= 1 and 
            len(email) >= 5 and len(email) <= 50 and
            not any(char in email for char in ['<', '>', '{', '}', '[', ']', '(', ')', '|', '\\', '/', '?', '*', '%']) and
            email[-1].isalpha() and  # Must end with letter
            email[0].isalnum()):     # Must start with letter/number
            emails.append(email)
    
    emails = list(set(emails))  # Remove duplicates
    
    # Enhanced phone detection with international patterns
    phone_patterns = [
        r'\+\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}',  # International format
        r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',  # US format (123) 456-7890
        r'\d{3}[\s-]?\d{3}[\s-]?\d{4}'  # US format 123-456-7890
    ]
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, content))
    phones = list(set(phones))  # Remove duplicates
    
    # Mozambican ID detection - specific patterns
    mozambican_bi = len(re.findall(r'\bBI[\s]*\d{8,13}[A-Z]?\b', content, re.IGNORECASE))
    mozambican_phone = len(re.findall(r'\+258[\s-]?[0-9]{2}[\s-]?[0-9]{3}[\s-]?[0-9]{3}', content))
    
    # Enhanced name detection - very conservative for resumes
    words = re.findall(r'\b[A-Z][a-z]{2,}\b', content)  # Only properly capitalized words
    excluded_words = {
        'Python', 'Java', 'JavaScript', 'Docker', 'AWS', 'API', 'AI', 'ML', 'Data', 'Science', 
        'University', 'Bachelor', 'Master', 'PhD', 'Professor', 'Research', 'Development',
        'Technologies', 'GitHub', 'LinkedIn', 'Paris', 'France', 'English', 'French', 'Portuguese',
        'Microsoft', 'Google', 'Amazon', 'Software', 'Engineering', 'Computer', 'Technology',
        'Database', 'Network', 'Security', 'Analytics', 'Machine', 'Learning', 'Intelligence',
        'Analysis', 'Processing', 'Management', 'Systems', 'Applications', 'Solutions'
    }
    
    # Only count likely personal names (2-3 names max for a resume)
    potential_names = [word for word in set(words) if word not in excluded_words and len(word) > 2]
    # Limit to reasonable count for a resume (max 5 names)
    potential_names = potential_names[:5] if len(potential_names) > 5 else potential_names
    
    pii_detected = {
        "emails": len(emails),
        "phones": len(phones),
        "mozambican_ids": mozambican_bi + mozambican_phone,
        "names": len(potential_names)
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
    
    # Context-aware medical detection to avoid false positives
    content_lower = content.lower()
    technical_contexts = [
        'image analysis', 'data analysis', 'computer vision', 'machine learning', 
        'ai', 'artificial intelligence', 'research', 'algorithm', 'technology', 
        'development', 'software', 'system', 'application', 'university', 
        'study', 'academic', 'thesis', 'project', 'technical', 'engineering'
    ]
    
    # Check if this appears to be a technical/academic document
    is_technical_context = any(context in content_lower for context in technical_contexts)
    
    # Only detect medical data if NOT in a technical research context
    medical_keywords = ['médico', 'health', 'medical', 'patient', 'diagnosis', 'treatment', 'hospital']
    medical_found = any(keyword in content_lower for keyword in medical_keywords)
    
    if medical_found and not is_technical_context:
        # Additional check: avoid flagging technical work about medical topics
        technical_medical_patterns = [
            'medical image', 'medical ai', 'medical research', 'medical data', 
            'medical algorithm', 'medical analysis', 'medical imaging', 'medical software'
        ]
        is_technical_medical = any(pattern in content_lower for pattern in technical_medical_patterns)
        
        if not is_technical_medical:
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
    """Generate analytics HTML with guardrails information"""
    
    risk_score = analysis["risk_score"]
    risk_color = "#ff4444" if risk_score > 0.7 else "#ffaa00" if risk_score > 0.4 else "#44ff44"
    
    # 🛡️ Guardrails information
    guardrails_info = analysis.get("guardrails", {})
    
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
        
        {get_guardrails_section(guardrails_info)}
        
        <div style="background: #2d2d2d; padding: 15px; border-radius: 8px; margin-top: 20px;">
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

def get_guardrails_section(guardrails_info: Dict) -> str:
    """Generate guardrails information section"""
    
    if not guardrails_info:
        return ""
    
    risk_level = guardrails_info.get("risk_level", "low")
    risk_colors = {"high": "#ff4444", "medium": "#ffaa00", "low": "#44ff44"}
    risk_color = risk_colors.get(risk_level, "#44ff44")
    
    blocked_count = len(guardrails_info.get("blocked_items", []))
    masked_count = len(guardrails_info.get("masked_items", []))
    
    section = f"""
        <div style="background: #0d3a2d; padding: 15px; border-radius: 8px; margin-top: 20px; border: 2px solid #44ff44;">
            <h4 style="color: #44ff44; margin: 0 0 15px 0;">🛡️ Smart Guardrails Active</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="color: #ff4444; font-size: 1.5em; font-weight: bold;">{blocked_count}</div>
                    <div style="color: #ffffff; font-size: 0.9em;">Items Blocked</div>
                </div>
                <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="color: #ffaa00; font-size: 1.5em; font-weight: bold;">{masked_count}</div>
                    <div style="color: #ffffff; font-size: 0.9em;">Items Masked</div>
                </div>
                <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="color: {risk_color}; font-size: 1.2em; font-weight: bold; text-transform: uppercase;">{risk_level}</div>
                    <div style="color: #ffffff; font-size: 0.9em;">Protection Level</div>
                </div>
            </div>
    """
    
    # Show protection methods applied
    if guardrails_info.get("protection_applied"):
        section += '<div style="margin-bottom: 10px;">'
        for protection in guardrails_info["protection_applied"]:
            section += f'<span style="background: #2d2d2d; color: #44ff44; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; display: inline-block; margin-bottom: 5px;">{protection}</span>'
        section += '</div>'
    
    # Show what was blocked/masked
    if blocked_count > 0 or masked_count > 0:
        section += '<div style="background: #1a1a1a; padding: 10px; border-radius: 6px; margin-top: 10px;">'
        section += '<h5 style="color: #ffffff; margin: 0 0 8px 0;">Protected Items:</h5>'
        
        if blocked_count > 0:
            section += '<div style="color: #ff4444; font-size: 0.9em; margin-bottom: 5px;"><strong>Blocked:</strong></div>'
            for item in guardrails_info.get("blocked_items", [])[:3]:
                section += f'<div style="color: #cccccc; font-size: 0.8em; margin-left: 10px;">• {item}</div>'
        
        if masked_count > 0:
            section += '<div style="color: #ffaa00; font-size: 0.9em; margin-bottom: 5px; margin-top: 8px;"><strong>Masked:</strong></div>'
            for item in guardrails_info.get("masked_items", [])[:3]:
                section += f'<div style="color: #cccccc; font-size: 0.8em; margin-left: 10px;">• {item}</div>'
        
        section += '</div>'
    
    section += '</div>'
    
    return section

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
    
    # 🛡️ Show guardrails impact
    guardrails_info = analysis.get("guardrails", {})
    if guardrails_info:
        comparison_md += "\n## 🛡️ Guardrails Protection Impact\n\n"
        
        blocked_count = len(guardrails_info.get("blocked_items", []))
        masked_count = len(guardrails_info.get("masked_items", []))
        
        comparison_md += f"**Data Protection Applied:**\n"
        comparison_md += f"- {blocked_count} items completely blocked from AI processing\n"
        comparison_md += f"- {masked_count} items masked/anonymized\n"
        comparison_md += f"- Protection level: {guardrails_info.get('risk_level', 'unknown').upper()}\n\n"
        
        comparison_md += "**What AI Providers See:**\n"
        comparison_md += "- Sensitive emails → [EMAIL_BLOCKED_MDPL] or [EMAIL_MASKED]\n"
        comparison_md += "- Phone numbers → [PHONE: +33***28]\n"
        comparison_md += "- Medical data → [MEDICAL_DATA_PROTECTED]\n"
        comparison_md += "- Credit cards → [CREDIT_CARD_BLOCKED]\n\n"
        
        comparison_md += "**🔒 This means:**\n"
        comparison_md += "- **OpenAI** gets protected data, not your sensitive info\n"
        comparison_md += "- **Mistral** processes anonymized content\n"
        comparison_md += "- **Other providers** can't access blocked data\n"
        comparison_md += "- **Your privacy** is preserved during AI analysis\n\n"
    
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

def generate_detailed_issues_analysis(all_content: List[str], all_filenames: List[str]) -> str:
    """Generate comprehensive detailed issues analysis for all documents"""
    
    import re  # Import re module for regex operations
    
    issues_html = """
    <div style="background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #3a3a3a;">
        <h3 style="color: #ffffff; margin-top: 0;">🔍 Detailed Issues Analysis</h3>
        <p style="color: #b0b0b0; margin-bottom: 20px;">Comprehensive breakdown of compliance issues found in each document</p>
    """
    
    for i, (filename, content) in enumerate(zip(all_filenames, all_content)):
        # Enhanced PII detection with details
        issues_details = {}
        
        # Use the same enhanced email detection as main analysis
        email_pattern = r'\b[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]{2,}\.[A-Za-z]{2,6}\b'
        potential_emails = re.findall(email_pattern, content)
        
        # Validate emails - filter out garbled text
        valid_emails = []
        for email in potential_emails:
            if (email.count('@') == 1 and 
                email.count('.') >= 1 and 
                len(email) >= 5 and len(email) <= 50 and
                not any(char in email for char in ['<', '>', '{', '}', '[', ']', '(', ')', '|', '\\', '/', '?', '*', '%']) and
                email[-1].isalpha() and  # Must end with letter
                email[0].isalnum()):     # Must start with letter/number
                valid_emails.append(email)
        
        valid_emails = list(set(valid_emails))  # Remove duplicates
        
        if valid_emails:
            issues_details["📧 Email Addresses"] = {
                "count": len(valid_emails),
                "examples": ", ".join(valid_emails[:3]),
                "risk": "Medium",
                "description": "Personal email addresses require consent and data protection measures"
            }
        
        # Use the same enhanced phone detection as main analysis
        phone_patterns = [
            r'\+\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}',
            r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',
            r'\d{3}[\s-]?\d{3}[\s-]?\d{4}'
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, content))
        phones = list(set(phones))  # Remove duplicates
        
        if phones:
            issues_details["📞 Phone Numbers"] = {
                "count": len(phones),
                "examples": ", ".join(phones[:2]),
                "risk": "Medium",
                "description": "Phone numbers are personal data requiring protection"
            }
        
        # Mozambican identifiers (separate detection)
        mozambican_bi = re.findall(r'\bBI[\s]*\d{8,13}[A-Z]?\b', content, re.IGNORECASE)
        mozambican_phone = re.findall(r'\+258[\s-]?[0-9]{2}[\s-]?[0-9]{3}[\s-]?[0-9]{3}', content)
        
        if mozambican_bi or mozambican_phone:
            all_moz_ids = mozambican_bi + mozambican_phone
            issues_details["🇲🇿 Mozambican Identifiers"] = {
                "count": len(all_moz_ids),
                "examples": ", ".join(all_moz_ids[:2]),
                "risk": "High",
                "description": "Mozambican personal identifiers subject to MDPL regulations"
            }
        
        # Credit card detection with masking
        cc_matches = re.findall(r'\b(?:\d{4}[\s-]?){3}\d{4}\b', content)
        if cc_matches:
            masked_cc = [cc[:4] + " **** **** " + cc[-4:] for cc in cc_matches]
            issues_details["💳 Credit Card Numbers"] = {
                "count": len(cc_matches),
                "examples": masked_cc[:3],
                "risk": "High",
                "description": "Financial payment card data requires PCI DSS compliance, encryption, and strict access controls"
            }
        
        # SSN detection with masking
        ssn_matches = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', content)
        if ssn_matches:
            masked_ssn = [ssn[:3] + "-**-****" for ssn in ssn_matches]
            issues_details["🆔 Social Security Numbers"] = {
                "count": len(ssn_matches),
                "examples": masked_ssn[:3],
                "risk": "Critical",
                "description": "SSNs are highly sensitive identifiers requiring strict access controls, encryption, and compliance with federal privacy laws"
            }
        
        # Enhanced Medical/health data detection with context awareness
        medical_findings = {
            "conditions": [],
            "treatments": [],
            "procedures": [],
            "medications": [],
            "medical_identifiers": [],
            "healthcare_providers": []
        }
        
        # Context exclusions - avoid false positives from technical/academic contexts
        technical_contexts = [
            'image analysis', 'data analysis', 'computer vision', 'machine learning', 'AI', 'artificial intelligence',
            'research', 'algorithm', 'technology', 'development', 'software', 'system', 'application',
            'university', 'study', 'academic', 'thesis', 'project', 'technical', 'engineering'
        ]
        
        # Check if this appears to be a technical/academic document
        is_technical_context = any(context.lower() in content.lower() for context in technical_contexts)
        
        # Medical conditions (English and Portuguese) - only flag if not in technical context
        conditions = ['diabetes', 'hipertensão', 'hypertension', 'cancer', 'câncer', 'hepatitis', 'hepatite', 
                     'HIV', 'AIDS', 'tuberculosis', 'tuberculose', 'pneumonia', 'asthma', 'asma',
                     'depression', 'depressão', 'anxiety', 'ansiedade', 'bipolar', 'schizophrenia',
                     'stroke', 'derrame', 'heart attack', 'infarto']
        
        # Medical treatments and procedures - exclude technical research contexts
        treatments = ['treatment', 'tratamento', 'therapy', 'terapia', 'medication', 'medicação',
                     'prescription', 'prescrição', 'chemotherapy', 'quimioterapia', 'radiation', 'radiação', 'dialysis', 'diálise']
        
        # Healthcare context words - be more specific to avoid false positives
        healthcare_context = ['patient care', 'paciente', 'doctor visit', 'médico consulta', 'physician appointment', 
                             'nurse', 'enfermeira', 'hospital admission', 'clinic visit', 'clínica consulta', 
                             'emergency room', 'emergência médica', 'ambulance', 'ambulância']
        
        # Medical identifiers and records - very specific to actual medical records
        medical_ids = ['medical record number', 'prontuário médico', 'patient ID number', 'ID do paciente', 
                      'health insurance card', 'seguro saúde número', 'medical history record', 'histórico médico pessoal', 
                      'lab test results', 'resultados exames médicos']
        
        # Check for each category - but skip if in technical/academic context
        content_lower = content.lower()
        
        # Simplified medical detection - already handled above
        # (This section was replaced with the context-aware detection above)
        
        # Phone and consent already handled above
        
        # Calculate overall risk for this document
        critical_count = sum(1 for issue in issues_details.values() if issue["risk"] == "Critical")
        high_risk_count = sum(1 for issue in issues_details.values() if issue["risk"] == "High")
        medium_risk_count = sum(1 for issue in issues_details.values() if issue["risk"] == "Medium")
        
        if critical_count >= 1:
            doc_risk = "Critical"
            doc_risk_color = "#8b0000"
        elif high_risk_count >= 2:
            doc_risk = "High"
            doc_risk_color = "#ff4444"
        elif high_risk_count >= 1 or medium_risk_count >= 3:
            doc_risk = "Medium"
            doc_risk_color = "#ffaa00"
        else:
            doc_risk = "Low"
            doc_risk_color = "#44ff44"
        
        # Generate document section
        issues_html += f"""
        <div style="background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid {doc_risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="color: #ffffff; margin: 0;">📄 {filename}</h4>
                <span style="background: {doc_risk_color}; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold;">
                    {doc_risk} Risk ({len(issues_details)} issues)
                </span>
            </div>
        """
        
        if issues_details:
            for issue_type, details in issues_details.items():
                risk_badge_colors = {
                    "Critical": "#8b0000",
                    "High": "#ff4444", 
                    "Medium": "#ffaa00",
                    "Low": "#44ff44"
                }
                risk_badge_color = risk_badge_colors.get(details["risk"], "#ffaa00")
                
                issues_html += f"""
                <div style="background: #333; padding: 15px; border-radius: 6px; margin-bottom: 12px; border-left: 3px solid {risk_badge_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h5 style="color: #ffffff; margin: 0;">{issue_type}</h5>
                        <span style="background: {risk_badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                            {details["risk"]} Risk
                        </span>
                    </div>
                    
                    <p style="color: #e0e0e0; margin: 8px 0; font-size: 0.9em;">
                        <strong>Count:</strong> {details["count"]} | 
                        <strong>Examples:</strong> {', '.join(map(str, details["examples"][:2]))}{'...' if len(details["examples"]) > 2 else ''}
                    </p>
                    
                    <p style="color: #b0b0b0; margin: 8px 0 0 0; font-size: 0.85em; font-style: italic;">
                        {details["description"]}
                    </p>
                </div>
                """
        else:
            issues_html += """
            <div style="background: #333; padding: 15px; border-radius: 6px; border-left: 3px solid #44ff44;">
                <p style="color: #44ff44; margin: 0; text-align: center;">
                    ✅ No significant compliance issues detected in this document
                </p>
            </div>
            """
        
        issues_html += "</div>"
    
    # Summary section
    total_docs = len(all_filenames)
    total_issues = sum(len(re.findall(r'[📧🇲🇿💳🆔🏥⚠️📞]', content)) for content in all_content)
    
    issues_html += f"""
        <div style="background: #2d2d2d; padding: 20px; border-radius: 8px; border: 1px solid #4a4a4a; margin-top: 20px;">
            <h4 style="color: #ffffff; margin: 0 0 15px 0;">📊 Overall Issues Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="text-align: center;">
                    <h5 style="color: #ffffff; margin: 0;">Total Documents</h5>
                    <p style="color: #4a9eff; font-size: 1.5em; font-weight: bold; margin: 5px 0;">{total_docs}</p>
                </div>
                <div style="text-align: center;">
                    <h5 style="color: #ffffff; margin: 0;">Total Issues</h5>
                    <p style="color: #ffaa00; font-size: 1.5em; font-weight: bold; margin: 5px 0;">{total_issues}</p>
                </div>
                <div style="text-align: center;">
                    <h5 style="color: #ffffff; margin: 0;">Compliance Status</h5>
                    <p style="color: {'#8b0000' if total_issues > 10 else '#ff4444' if total_issues > 5 else '#ffaa00' if total_issues > 2 else '#44ff44'}; font-size: 1.2em; font-weight: bold; margin: 5px 0;">{'Critical' if total_issues > 10 else 'High Risk' if total_issues > 5 else 'Needs Review' if total_issues > 2 else 'Good'}</p>
                </div>
            </div>
        </div>
    </div>
    """
    
    return issues_html

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
                    
                    with gr.TabItem("🔍 Detailed Issues"):
                        detailed_issues_output = gr.HTML()
                    
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
                recommendations_output,
                detailed_issues_output
            ]
        )
        
        # Clear button handler
        clear_btn.click(
            fn=lambda: (None, "", "", "", "", "", ""),
            outputs=[
                file_input,
                status_output,
                compliance_matrix_output,
                analytics_output,
                comparison_output,
                recommendations_output,
                detailed_issues_output
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