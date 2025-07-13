#!/usr/bin/env python3
"""
DocuMind Simple Demo Application

A lightweight demo that showcases DocuMind capabilities using basic libraries.
"""

import os
import time
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from urllib.parse import parse_qs, urlparse
import json

class DocuMindServer(SimpleHTTPRequestHandler):
    """Simple HTTP server for DocuMind demo"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html_interface().encode())
        elif self.path.startswith('/process'):
            self.handle_processing()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_processing(self):
        """Handle document processing simulation"""
        # Parse query parameters
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        provider = params.get('provider', ['openai'])[0]
        
        # Simulate processing
        time.sleep(1)  # Simulate processing time
        
        result = self.simulate_processing(provider)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def simulate_processing(self, provider):
        """Simulate document processing"""
        if provider == 'openai':
            return {
                'provider': 'OpenAI GPT-4 Vision',
                'confidence': round(random.uniform(0.85, 0.95), 3),
                'compliance_score': round(random.uniform(0.88, 0.96), 3),
                'entities': ['John Doe', 'ABC Corporation', '2024-01-15', 'Contract #12345'],
                'processing_time': '2.1s',
                'status': 'success'
            }
        else:  # mistral
            return {
                'provider': 'Mistral AI',
                'confidence': round(random.uniform(0.82, 0.93), 3),
                'compliance_score': round(random.uniform(0.85, 0.94), 3),
                'entities': ['Jean Dupont', 'XYZ Société', '15/01/2024', 'Contrat #12345'],
                'processing_time': '1.8s',
                'status': 'success'
            }
    
    def get_html_interface(self):
        """Generate HTML interface"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuMind - AI Document Processing Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #5ba7f7 0%, #4a90e2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%), 
                        linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%);
            background-size: 20px 20px;
            background-position: 0 0, 10px 10px;
            opacity: 0.3;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.95;
            position: relative;
            z-index: 1;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
            border: 2px dashed #4a90e2;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .upload-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(74, 144, 226, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .upload-section:hover {
            border-color: #357abd;
            background: linear-gradient(135deg, #d4e8fc 0%, #e8f4fd 100%);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(74, 144, 226, 0.15);
        }
        
        .upload-section:hover::before {
            left: 100%;
        }
        
        .upload-section h3 {
            color: #357abd;
            margin-bottom: 15px;
            position: relative;
            z-index: 1;
        }
        
        .file-input {
            margin: 20px 0;
        }
        
        .file-input input[type="file"] {
            padding: 12px;
            border: 2px solid #4a90e2;
            border-radius: 8px;
            width: 100%;
            max-width: 400px;
            background: white;
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
        }
        
        .file-input input[type="file"]:focus {
            outline: none;
            border-color: #357abd;
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
        }
        
        .providers {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .provider-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%);
            border: 1px solid #bdd7f2;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(74, 144, 226, 0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .provider-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4a90e2, #5ba7f7, #357abd);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .provider-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 35px rgba(74, 144, 226, 0.2);
            border-color: #4a90e2;
        }
        
        .provider-card:hover::before {
            transform: scaleX(1);
        }
        
        .provider-card h3 {
            color: #357abd;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
            z-index: 1;
            font-weight: 600;
        }
        
        .btn {
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: all 0.3s ease;
            width: 100%;
            margin-bottom: 15px;
            position: relative;
            z-index: 1;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.6s;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(74, 144, 226, 0.4);
            background: linear-gradient(135deg, #5ba7f7 0%, #4a90e2 100%);
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn.secondary {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        }
        
        .btn.secondary:hover {
            background: linear-gradient(135deg, #81c0ff 0%, #1e90ff 100%);
            box-shadow: 0 8px 20px rgba(116, 185, 255, 0.4);
        }
        
        .btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            background: linear-gradient(135deg, #f8fcff 0%, #ecf6ff 100%);
            border: 1px solid #bdd7f2;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
            min-height: 120px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            position: relative;
            overflow: hidden;
        }
        
        .results::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #4a90e2, #74b9ff, #357abd);
        }
        
        .comparison {
            grid-column: 1 / -1;
            text-align: center;
        }
        
        .comparison .btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            max-width: 300px;
            margin: 0 auto;
        }
        
        .loading {
            color: #4a90e2;
            font-style: italic;
            position: relative;
            z-index: 1;
        }
        
        .success {
            color: #357abd;
            position: relative;
            z-index: 1;
        }
        
        .error {
            color: #e74c3c;
            position: relative;
            z-index: 1;
        }
        
        @media (max-width: 768px) {
            .providers {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 DocuMind</h1>
            <p>AI-Powered Document Processing with Multi-Provider Intelligence</p>
        </div>
        
        <div class="content">
            <div class="upload-section">
                <h3>📄 Upload Your Document</h3>
                <p>Select a PDF, image, or document file to process with AI</p>
                <div class="file-input">
                    <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.tiff,.doc,.docx">
                </div>
            </div>
            
            <div class="providers">
                <div class="provider-card">
                    <h3>🤖 OpenAI Processing</h3>
                    <p>GPT-4 Vision with advanced OCR capabilities</p>
                    <button class="btn" onclick="processWithProvider('openai')">Process with OpenAI</button>
                    <div class="results" id="openai-results">Ready to process your document...</div>
                </div>
                
                <div class="provider-card">
                    <h3>🧠 Mistral Processing</h3>
                    <p>Mistral AI with European compliance focus</p>
                    <button class="btn" onclick="processWithProvider('mistral')">Process with Mistral</button>
                    <div class="results" id="mistral-results">Ready to process your document...</div>
                </div>
                
                <div class="provider-card comparison">
                    <h3>⚖️ Provider Comparison</h3>
                    <p>Compare results from both AI providers</p>
                    <button class="btn secondary" onclick="compareProviders()">Compare Both Providers</button>
                    <div class="results" id="comparison-results">Upload a document and click compare to see provider analysis...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function processWithProvider(provider) {
            const fileInput = document.getElementById('fileInput');
            const resultsDiv = document.getElementById(provider + '-results');
            
            if (!fileInput.files.length) {
                resultsDiv.innerHTML = '<span class="error">❌ Please upload a file first</span>';
                return;
            }
            
            resultsDiv.innerHTML = '<span class="loading">🔄 Processing document with ' + provider.toUpperCase() + '...</span>';
            
            try {
                const response = await fetch('/process?provider=' + provider);
                const result = await response.json();
                
                const output = `
🚀 **${result.provider} Processing Results**

📊 Confidence Score: ${(result.confidence * 100).toFixed(1)}%
🛡️ Compliance Score: ${(result.compliance_score * 100).toFixed(1)}%
🔍 Entities Found: ${result.entities.length}
⏱️ Processing Time: ${result.processing_time}

**Extracted Entities:**
${result.entities.map(entity => '• ' + entity).join('\\n')}

**Status:** ✅ Processing Complete
                `.trim();
                
                resultsDiv.innerHTML = '<span class="success">' + output + '</span>';
            } catch (error) {
                resultsDiv.innerHTML = '<span class="error">❌ Error processing document</span>';
            }
        }
        
        async function compareProviders() {
            const fileInput = document.getElementById('fileInput');
            const resultsDiv = document.getElementById('comparison-results');
            
            if (!fileInput.files.length) {
                resultsDiv.innerHTML = '<span class="error">❌ Please upload a file first</span>';
                return;
            }
            
            resultsDiv.innerHTML = '<span class="loading">🔄 Comparing providers...</span>';
            
            try {
                const [openaiResponse, mistralResponse] = await Promise.all([
                    fetch('/process?provider=openai'),
                    fetch('/process?provider=mistral')
                ]);
                
                const openaiResult = await openaiResponse.json();
                const mistralResult = await mistralResponse.json();
                
                const winner = openaiResult.confidence > mistralResult.confidence ? 'OpenAI' : 'Mistral';
                const confidenceDiff = Math.abs(openaiResult.confidence - mistralResult.confidence);
                
                const output = `
⚖️ **Provider Comparison Results**

**OpenAI Results:**
• Confidence: ${(openaiResult.confidence * 100).toFixed(1)}%
• Compliance: ${(openaiResult.compliance_score * 100).toFixed(1)}%
• Entities: ${openaiResult.entities.length}

**Mistral Results:**
• Confidence: ${(mistralResult.confidence * 100).toFixed(1)}%
• Compliance: ${(mistralResult.compliance_score * 100).toFixed(1)}%
• Entities: ${mistralResult.entities.length}

🏆 **Winner:** ${winner}
📈 **Confidence Difference:** ${(confidenceDiff * 100).toFixed(1)}%

**Recommendation:** Use ${winner} for optimal results on this document type.
                `.trim();
                
                resultsDiv.innerHTML = '<span class="success">' + output + '</span>';
            } catch (error) {
                resultsDiv.innerHTML = '<span class="error">❌ Error comparing providers</span>';
            }
        }
    </script>
</body>
</html>
        '''

def main():
    """Main application entry point"""
    port = 8080
    
    print("🧠 DocuMind Demo Application")
    print("=" * 50)
    print(f"🌐 Starting server on http://localhost:{port}")
    print("📄 This demo simulates AI-powered document processing")
    print("🤖 Compare OpenAI and Mistral AI results")
    print("=" * 50)
    
    # Start the server
    server = HTTPServer(('localhost', port), DocuMindServer)
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}')
    
    print(f"✅ Server running! Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop the server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        server.server_close()

if __name__ == "__main__":
    main()