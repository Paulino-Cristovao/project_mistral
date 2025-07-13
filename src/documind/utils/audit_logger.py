"""
Audit logging system for compliance tracking
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..models import AuditLogEntry, PrivacyImpact

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Comprehensive audit logging system for compliance tracking
    """
    
    def __init__(self, level: str = "basic", log_dir: Optional[Path] = None):
        """
        Initialize audit logger
        
        Args:
            level: Logging level ('basic', 'full')
            log_dir: Directory for audit logs
        """
        self.level = level
        self.log_dir = log_dir or Path("audit_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create audit log file
        self.audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        logger.info(f"Audit logger initialized with level: {level}")
    
    def log_action(
        self,
        document_id: str,
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        compliance_impact: PrivacyImpact = PrivacyImpact.LOW
    ) -> None:
        """
        Log an audit action
        
        Args:
            document_id: Document identifier
            action: Action performed
            user_id: User identifier (optional)
            details: Additional details
            compliance_impact: Privacy impact of the action
        """
        try:
            # Create audit entry
            entry = AuditLogEntry(
                document_id=document_id,
                action=action,
                user_id=user_id,
                details=details or {},
                compliance_impact=compliance_impact,
                retention_category=self._determine_retention_category(action, compliance_impact)
            )
            
            # Write to audit log file
            self._write_audit_entry(entry)
            
            # Log to application logger based on level
            if self.level == "full" or compliance_impact in [PrivacyImpact.HIGH, PrivacyImpact.CRITICAL]:
                logger.info(f"AUDIT: {action} on {document_id} by {user_id or 'system'}")
            
        except Exception as e:
            logger.error(f"Failed to log audit action: {e}")
    
    def _write_audit_entry(self, entry: AuditLogEntry) -> None:
        """Write audit entry to file"""
        try:
            with open(self.audit_file, 'a', encoding='utf-8') as f:
                json_line = entry.model_dump_json() + '\n'
                f.write(json_line)
        
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    def _determine_retention_category(
        self, 
        action: str, 
        compliance_impact: PrivacyImpact
    ) -> str:
        """Determine retention category for audit entry"""
        
        if compliance_impact == PrivacyImpact.CRITICAL:
            return "permanent_retention"
        elif compliance_impact == PrivacyImpact.HIGH:
            return "extended_retention"
        elif action in ["processing_failed", "compliance_violation", "access_denied"]:
            return "security_retention"
        else:
            return "standard_retention"
    
    def get_audit_trail(
        self, 
        document_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Dict[str, Any]]:
        """
        Get audit trail for a specific document
        
        Args:
            document_id: Document identifier
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            List of audit entries
        """
        audit_entries = []
        
        try:
            # Read all audit log files in date range
            log_files = self._get_log_files_in_range(start_date, end_date)
            
            for log_file in log_files:
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry_data = json.loads(line.strip())
                                if entry_data.get('document_id') == document_id:
                                    # Filter by date if specified
                                    entry_time = datetime.fromisoformat(
                                        entry_data['timestamp'].replace('Z', '+00:00')
                                    )
                                    
                                    if start_date and entry_time < start_date:
                                        continue
                                    if end_date and entry_time > end_date:
                                        continue
                                    
                                    audit_entries.append(entry_data)
                            
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
        
        # Sort by timestamp
        audit_entries.sort(key=lambda x: x['timestamp'])
        return audit_entries
    
    def _get_log_files_in_range(
        self, 
        start_date: Optional[datetime], 
        end_date: Optional[datetime]
    ) -> list[Path]:
        """Get audit log files in date range"""
        
        # If no date range specified, return current log file
        if not start_date and not end_date:
            return [self.audit_file]
        
        # Find all audit log files
        log_files = []
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                # Check if file is in range
                if start_date and file_date < start_date.replace(hour=0, minute=0, second=0):
                    continue
                if end_date and file_date > end_date.replace(hour=23, minute=59, second=59):
                    continue
                
                log_files.append(log_file)
            
            except ValueError:
                # Skip files with invalid date format
                continue
        
        return sorted(log_files)
    
    def generate_compliance_report(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report from audit logs
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report dictionary
        """
        try:
            # Get all audit entries in range
            all_entries = []
            log_files = self._get_log_files_in_range(start_date, end_date)
            
            for log_file in log_files:
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry_data = json.loads(line.strip())
                                entry_time = datetime.fromisoformat(
                                    entry_data['timestamp'].replace('Z', '+00:00')
                                )
                                
                                if start_date and entry_time < start_date:
                                    continue
                                if end_date and entry_time > end_date:
                                    continue
                                
                                all_entries.append(entry_data)
                            
                            except json.JSONDecodeError:
                                continue
            
            # Analyze audit entries
            report = {
                "report_period": {
                    "start": start_date.isoformat() if start_date else "beginning",
                    "end": end_date.isoformat() if end_date else "present"
                },
                "total_entries": len(all_entries),
                "action_summary": {},
                "compliance_impact_summary": {},
                "document_count": len(set(entry['document_id'] for entry in all_entries)),
                "security_events": [],
                "high_risk_activities": []
            }
            
            # Analyze actions
            for entry in all_entries:
                action = entry.get('action', 'unknown')
                report['action_summary'][action] = report['action_summary'].get(action, 0) + 1
                
                impact = entry.get('compliance_impact', 'low')
                report['compliance_impact_summary'][impact] = report['compliance_impact_summary'].get(impact, 0) + 1
                
                # Track security events
                if action in ['processing_failed', 'access_denied', 'compliance_violation']:
                    report['security_events'].append({
                        'timestamp': entry['timestamp'],
                        'action': action,
                        'document_id': entry['document_id'],
                        'details': entry.get('details', {})
                    })
                
                # Track high-risk activities
                if impact in ['high', 'critical']:
                    report['high_risk_activities'].append({
                        'timestamp': entry['timestamp'],
                        'action': action,
                        'document_id': entry['document_id'],
                        'impact': impact
                    })
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {"error": str(e)}
    
    def cleanup_old_logs(self, retention_days: int = 365) -> None:
        """
        Clean up old audit logs based on retention policy
        
        Args:
            retention_days: Number of days to retain logs
        """
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0) - \
                         datetime.timedelta(days=retention_days)
            
            for log_file in self.log_dir.glob("audit_*.jsonl"):
                try:
                    # Extract date from filename
                    date_str = log_file.stem.replace("audit_", "")
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        log_file.unlink()  # Delete old log file
                        logger.info(f"Deleted old audit log: {log_file}")
                
                except (ValueError, OSError) as e:
                    logger.warning(f"Failed to process log file {log_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")