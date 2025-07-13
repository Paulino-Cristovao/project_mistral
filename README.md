# DocuMind 🧠

**AI-Powered Document Processing with Multi-Provider Intelligence**

DocuMind is an enterprise-grade document processing system that uses AI agents from multiple providers (OpenAI, Mistral) to intelligently extract, analyze, and process documents with compliance-first architecture.

## ✨ Key Features

- 🤖 **Multi-AI Agents**: OpenAI GPT-4 Vision + Mistral AI integration
- 🌐 **Web Interface**: Gradio-based interface for easy document upload and processing
- ⚖️ **Provider Comparison**: Real-time comparison between AI providers
- 🛡️ **Compliance-First**: Built-in GDPR, HIPAA, NDPR compliance
- 🔗 **Relationship Extraction**: Advanced entity relationship mapping
- 🚀 **Intelligent Automation**: Hands-off processing with smart decision making
- 📊 **Enterprise Features**: Audit trails, performance metrics, batch processing

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/documind.git
cd documind
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="your_openai_key"
export MISTRAL_API_KEY="your_mistral_key"
```

### 3. Run Web Interface

```bash
python -m documind.app
```

Open http://localhost:7860 in your browser.

### 4. Or Use CLI

```bash
python -m documind process document.pdf --provider openai
python -m documind compare document.pdf
```

## 🏗️ Architecture

```
DocuMind/
├── agents/           # AI agent implementations
│   ├── openai_agent.py
│   ├── mistral_agent.py
│   ├── compliance_agent.py
│   └── bridge_agent.py
├── automation/       # Intelligent processing
│   ├── processor.py
│   ├── decision_engine.py
│   └── adaptive_agent.py
├── models/          # Data models
├── utils/           # Utilities
└── app.py          # Main Gradio interface
```

## 📖 Usage Examples

### Process Single Document

```python
from documind import DocumentProcessor

processor = DocumentProcessor(
    openai_api_key="your_key",
    mistral_api_key="your_key"
)

result = processor.process("contract.pdf")
print(f"Compliance Score: {result.compliance_score}")
print(f"Entities Found: {len(result.bridges)}")
```

### Compare Providers

```python
comparison = processor.compare_providers("document.pdf")
print(f"Winner: {comparison.winner}")
print(f"Confidence: {comparison.confidence}")
```

## 🔧 Configuration

Create `.env` file:

```env
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key
COMPLIANCE_PROFILE=eu_gdpr
DEFAULT_REDACTION_LEVEL=moderate
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

- 📧 Email: support@documind.ai
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/documind/issues)
- 📖 Documentation: [docs.documind.ai](https://docs.documind.ai)