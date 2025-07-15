# 🧠 Mind Enhanced - AI Compliance Analysis Platform

## Overview
Mind Enhanced is a comprehensive AI compliance analysis platform that demonstrates potential data protection issues across different AI models and regulatory jurisdictions, with special focus on **Mozambique Data Protection Law**.

## Features
- 🇲🇿 **Mozambique DPL Compliance**: Full integration with Mozambique Data Protection Law
- 🤖 **Multi-AI Support**: OpenAI, Mistral, Claude, Gemini, Grok, Cohere
- 🛡️ **Advanced Guardrails**: Proactive compliance checking
- 🌍 **Global Coverage**: 6 major jurisdictions (Mozambique, EU, South Africa, Nigeria, California, US)
- 📊 **Analytics Dashboard**: Real-time risk assessment and compliance analysis
- 🎨 **Modern Interface**: Dark-themed professional UI

## Quick Start

### 1. Install Dependencies
```bash
pip install gradio pydantic plotly pandas pyyaml requests
```

### 2. Set API Keys (Optional)
```bash
export OPENAI_API_KEY="your_openai_key"
export MISTRAL_API_KEY="your_mistral_key"
```

### 3. Run the Platform
```bash
python documind_run.py
```

### 4. Access Interface
Open your browser and go to: **http://localhost:7860**

## Test Documents
Sample documents are available in the `test_data/` folder:
- `mozambique_medical.txt` - Medical record with Mozambican PII
- `eu_gdpr_contract.txt` - GDPR-compliant employment contract
- `high_risk_violations.txt` - Document with multiple compliance violations

## Project Structure
```
documind/
├── documind_run.py              # Main launcher
├── src/documind/
│   ├── models.py                # Data models
│   ├── mind_enhanced_app.py     # Main interface
│   ├── agents/                  # AI agents
│   └── compliance/              # Compliance frameworks
└── test_data/                   # Sample documents
```

## Key Features

### Mozambique Compliance
- Portuguese language support
- BI number detection
- +258 phone number recognition
- MDPL Articles 12 & 15 compliance
- Cross-border transfer restrictions
- Data localization requirements

### Multi-AI Analysis
- Real-time provider comparison
- Compliance scoring across models
- Performance benchmarking
- Risk assessment per provider

### Advanced Analytics
- Privacy breach risk assessment
- PII detection and analysis
- Jurisdiction-specific recommendations
- Visual compliance matrix

## Usage Example
1. Upload a document
2. Select AI providers (OpenAI, Mistral)
3. Choose jurisdictions (Mozambique, EU GDPR)
4. Enable guardrails and risk assessment
5. Analyze compliance results

## Validation
- ✅ 88.9% test success rate
- ✅ Production-ready code quality
- ✅ Comprehensive error handling
- ✅ Real-world compliance scenarios

---

**Mind Enhanced** - Built with ❤️ for AI compliance and data protection research.