"""
Command-line interface for DocBridgeGuard 2.0
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .core import ComplianceOCR, DocumentProcessor
from .utils.comparison import ComparisonEngine
from .utils.report_generator import ReportGenerator
from .bridges.analyzer import RelationshipAnalyzer
from .models import ProcessingConfig, RedactionLevel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="2.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    DocBridgeGuard 2.0: Enterprise Compliance-First OCR Pipeline
    
    A compliance-native OCR pipeline with guardrails and relationship extraction.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure context object
    ctx.ensure_object(dict)
    
    # Store configuration
    ctx.obj['config_path'] = Path(config) if config else None
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--provider', '-p', 
              type=click.Choice(['openai', 'mistral']), 
              default='mistral',
              help='OCR provider to use')
@click.option('--profile', 
              default='eu_gdpr',
              help='Compliance profile (e.g., eu_gdpr, africa_ndpr)')
@click.option('--output', '-o', 
              type=click.Path(),
              help='Output file path (default: auto-generated)')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'pdf', 'docx']),
              default='json',
              help='Output format')
@click.option('--redaction-level',
              type=click.Choice(['none', 'basic', 'moderate', 'strict', 'maximum']),
              default='moderate',
              help='PII redaction level')
@click.pass_context
def process(
    ctx: click.Context,
    file_path: str,
    provider: str,
    profile: str,
    output: Optional[str],
    output_format: str,
    redaction_level: str
) -> None:
    """
    Process a single document with compliance-first OCR
    
    FILE_PATH: Path to the document to process
    """
    try:
        click.echo(f"Processing document: {file_path}")
        click.echo(f"Provider: {provider}")
        click.echo(f"Profile: {profile}")
        
        # Initialize ComplianceOCR
        config_path = ctx.obj.get('config_path')
        compliance_ocr = ComplianceOCR(
            config_path=config_path,
            profile=profile
        )
        
        # Create processing configuration
        processing_config = ProcessingConfig(
            redaction_level=RedactionLevel(redaction_level)
        )
        
        # Process document
        with click.progressbar(length=100, label='Processing document') as bar:
            bar.update(20)
            result = compliance_ocr.process_document(
                file_path=file_path,
                provider=provider,
                custom_config=processing_config
            )
            bar.update(80)
        
        click.echo(f"✅ Processing completed successfully!")
        click.echo(f"Status: {result.status.value}")
        click.echo(f"Processing time: {result.processing_time_seconds:.2f}s")
        click.echo(f"Compliance score: {result.compliance_metadata.compliance_score:.2f}")
        click.echo(f"Bridges extracted: {len(result.bridges)}")
        click.echo(f"PII redactions: {result.compliance_metadata.redactions_count}")
        
        # Generate report
        report_generator = ReportGenerator()
        
        if not output:
            output = f"processing_result_{result.document_id}"
        
        report_path = report_generator.generate_processing_report(
            result, format_type=output_format
        )
        
        click.echo(f"📄 Report saved to: {report_path}")
        
        # Show risk flags if any
        if result.compliance_metadata.risk_flags:
            click.echo("⚠️  Risk flags detected:")
            for flag in result.compliance_metadata.risk_flags:
                click.echo(f"   - {flag}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output', '-o', 
              type=click.Path(),
              help='Output file path (default: auto-generated)')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'pdf', 'docx']),
              default='pdf',
              help='Output format')
@click.option('--profile', 
              default='eu_gdpr',
              help='Compliance profile')
@click.pass_context
def compare(
    ctx: click.Context,
    file_path: str,
    output: Optional[str],
    output_format: str,
    profile: str
) -> None:
    """
    Compare OpenAI and Mistral providers on the same document
    
    FILE_PATH: Path to the document to compare
    """
    try:
        click.echo(f"Comparing providers on document: {file_path}")
        
        # Check API keys
        if not os.getenv('OPENAI_API_KEY'):
            click.echo("❌ OPENAI_API_KEY not found in environment variables", err=True)
            sys.exit(1)
        if not os.getenv('MISTRAL_API_KEY'):
            click.echo("❌ MISTRAL_API_KEY not found in environment variables", err=True)
            sys.exit(1)
        
        # Initialize document processor
        config_path = ctx.obj.get('config_path')
        compliance_ocr = ComplianceOCR(config_path=config_path, profile=profile)
        processor = DocumentProcessor(compliance_ocr)
        
        # Process with both providers
        with click.progressbar(length=100, label='Processing with OpenAI') as bar:
            openai_result = processor.process(file_path, provider="openai")
            bar.update(100)
        
        with click.progressbar(length=100, label='Processing with Mistral') as bar:
            mistral_result = processor.process(file_path, provider="mistral")
            bar.update(100)
        
        # Compare results
        click.echo("🔍 Comparing results...")
        comparison_engine = ComparisonEngine()
        comparison = comparison_engine.compare_results(openai_result, mistral_result)
        
        # Display comparison summary
        click.echo(f"✅ Comparison completed!")
        click.echo(f"Winner: {comparison.winner or 'Tie'}")
        click.echo(f"Confidence: {comparison.confidence_in_winner:.2f}")
        
        # Show key metrics
        metrics = comparison.comparison_metrics
        click.echo("\n📊 Key Metrics:")
        click.echo(f"Text similarity: {metrics.get('text_similarity', 0):.3f}")
        click.echo(f"Bridge overlap: {metrics.get('bridge_overlap', 0):.3f}")
        click.echo(f"OpenAI processing time: {metrics.get('openai_processing_time', 0):.2f}s")
        click.echo(f"Mistral processing time: {metrics.get('mistral_processing_time', 0):.2f}s")
        
        # Generate comparison report
        report_generator = ReportGenerator()
        
        if not output:
            output = f"comparison_{comparison.document_id}"
        
        report_path = report_generator.generate_comparison_report(
            comparison, format_type=output_format
        )
        
        click.echo(f"📄 Comparison report saved to: {report_path}")
        
        # Show recommendations
        analysis = comparison.detailed_analysis
        if "recommendations" in analysis and analysis["recommendations"]:
            click.echo("\n💡 Recommendations:")
            for rec in analysis["recommendations"]:
                click.echo(f"   • {rec}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--provider', '-p', 
              type=click.Choice(['openai', 'mistral']), 
              default='mistral',
              help='OCR provider to use')
@click.option('--profile', 
              default='eu_gdpr',
              help='Compliance profile')
@click.option('--output-dir', '-o', 
              type=click.Path(),
              help='Output directory (default: ./reports)')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'pdf', 'docx', 'excel']),
              default='excel',
              help='Output format')
@click.option('--pattern', 
              default='*.pdf',
              help='File pattern to match (e.g., *.pdf, *.png)')
@click.pass_context
def batch(
    ctx: click.Context,
    directory: str,
    provider: str,
    profile: str,
    output_dir: Optional[str],
    output_format: str,
    pattern: str
) -> None:
    """
    Process multiple documents in batch mode
    
    DIRECTORY: Directory containing documents to process
    """
    try:
        directory_path = Path(directory)
        click.echo(f"Processing documents in: {directory_path}")
        click.echo(f"Pattern: {pattern}")
        click.echo(f"Provider: {provider}")
        
        # Find matching files
        files = list(directory_path.glob(pattern))
        if not files:
            click.echo(f"❌ No files found matching pattern: {pattern}")
            sys.exit(1)
        
        click.echo(f"Found {len(files)} files to process")
        
        # Initialize components
        config_path = ctx.obj.get('config_path')
        compliance_ocr = ComplianceOCR(config_path=config_path, profile=profile)
        processor = DocumentProcessor(compliance_ocr)
        
        # Process files
        results = []
        failed_files = []
        
        with click.progressbar(files, label='Processing files') as file_bar:
            for file_path in file_bar:
                try:
                    result = processor.process(file_path, provider=provider)
                    results.append(result)
                except Exception as e:
                    failed_files.append((file_path, str(e)))
                    logger.error(f"Failed to process {file_path}: {e}")
        
        click.echo(f"✅ Processed {len(results)} files successfully")
        if failed_files:
            click.echo(f"❌ Failed to process {len(failed_files)} files")
        
        # Generate batch report
        if results:
            report_generator = ReportGenerator(
                output_dir=Path(output_dir) if output_dir else None
            )
            
            if output_format == 'excel':
                report_path = report_generator.export_to_excel(
                    results, f"batch_report_{provider}"
                )
            else:
                report_path = report_generator.generate_compliance_report(
                    results, format_type=output_format
                )
            
            click.echo(f"📄 Batch report saved to: {report_path}")
            
            # Show summary statistics
            avg_compliance = sum(r.compliance_metadata.compliance_score for r in results) / len(results)
            total_bridges = sum(len(r.bridges) for r in results)
            total_redactions = sum(r.compliance_metadata.redactions_count for r in results)
            
            click.echo(f"\n📊 Summary Statistics:")
            click.echo(f"Average compliance score: {avg_compliance:.2f}")
            click.echo(f"Total bridges extracted: {total_bridges}")
            click.echo(f"Total PII redactions: {total_redactions}")
        
        # Show failed files
        if failed_files:
            click.echo("\n❌ Failed files:")
            for file_path, error in failed_files:
                click.echo(f"   {file_path}: {error}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--provider', '-p', 
              type=click.Choice(['openai', 'mistral']), 
              default='mistral',
              help='OCR provider to use')
@click.option('--output', '-o', 
              type=click.Path(),
              help='Output file path (default: auto-generated)')
@click.pass_context
def analyze(
    ctx: click.Context,
    file_path: str,
    provider: str,
    output: Optional[str]
) -> None:
    """
    Analyze bridge relationships in a document
    
    FILE_PATH: Path to the document to analyze
    """
    try:
        click.echo(f"Analyzing relationships in: {file_path}")
        
        # Process document
        config_path = ctx.obj.get('config_path')
        compliance_ocr = ComplianceOCR(config_path=config_path)
        processor = DocumentProcessor(compliance_ocr)
        
        result = processor.process(file_path, provider=provider)
        
        if not result.bridges:
            click.echo("ℹ️  No bridges found in document")
            return
        
        # Analyze relationships
        analyzer = RelationshipAnalyzer()
        analysis = analyzer.analyze_bridges(result.bridges)
        
        # Display analysis
        click.echo(f"✅ Analysis completed!")
        click.echo(f"Total bridges: {analysis['total_bridges']}")
        click.echo(f"Unique entities: {analysis['entity_count']}")
        click.echo(f"Compliance score: {analysis['compliance_score']:.2f}")
        
        click.echo("\n🔍 Privacy Impact Distribution:")
        for impact, count in analysis['privacy_distribution'].items():
            percentage = (count / analysis['total_bridges'] * 100) if analysis['total_bridges'] > 0 else 0
            click.echo(f"   {impact}: {count} ({percentage:.1f}%)")
        
        click.echo("\n🔗 Relationship Types:")
        for rel_type, count in list(analysis['relationship_types'].items())[:5]:  # Top 5
            click.echo(f"   {rel_type}: {count}")
        
        if analysis['high_risk_entities']:
            click.echo("\n⚠️  High-Risk Entities:")
            for entity_info in analysis['high_risk_entities'][:3]:  # Top 3
                click.echo(f"   {entity_info['entity']} (Risk Score: {entity_info['risk_score']})")
        
        # Generate detailed report
        report = analyzer.generate_report(result.bridges)
        
        if output:
            with open(output, 'w') as f:
                f.write(report)
            click.echo(f"📄 Detailed analysis saved to: {output}")
        else:
            click.echo("\n📋 Detailed Analysis:")
            click.echo(report)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8501, help='Port to bind to')
@click.pass_context
def dashboard(ctx: click.Context, host: str, port: int) -> None:
    """
    Launch the Streamlit dashboard for interactive document processing
    """
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Get dashboard script path
        dashboard_script = Path(__file__).parent / 'dashboard.py'
        
        if not dashboard_script.exists():
            click.echo("❌ Dashboard script not found", err=True)
            sys.exit(1)
        
        # Launch Streamlit
        click.echo(f"🚀 Launching dashboard at http://{host}:{port}")
        
        sys.argv = [
            "streamlit", "run", str(dashboard_script),
            "--server.address", host,
            "--server.port", str(port)
        ]
        
        stcli.main()
        
    except ImportError:
        click.echo("❌ Streamlit not installed. Install with: pip install streamlit", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error launching dashboard: {e}", err=True)
        sys.exit(1)


@cli.command()
def version() -> None:
    """Show version information"""
    click.echo("DocBridgeGuard 2.0.0")
    click.echo("Enterprise Compliance-First OCR Pipeline")
    click.echo("Copyright (c) 2024 Lino Paulino")


def main() -> None:
    """Main CLI entry point"""
    cli()


if __name__ == '__main__':
    main()