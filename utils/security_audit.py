"""
Security Audit Script - Checks for remaining hardcoded secrets and credentials
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class SecurityAuditor:
    """
    Audits the codebase for security issues like hardcoded credentials
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.issues = []
        
        # Patterns to detect potential security issues
        self.credential_patterns = [
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{10,}["\']', 
            r'token\s*=\s*["\'][^"\']{10,}["\']',
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'["\'][A-Za-z0-9_]{32,}["\']',  # Long strings that could be keys
            r'Bearer\s+[A-Za-z0-9_.-]+',
            r'[Aa]pi[Kk]ey\s*:\s*["\'][^"\']+["\']',
            r'[Ss]ecret[Kk]ey\s*:\s*["\'][^"\']+["\']'
        ]
        
        # Files to skip (they should contain example credentials)
        self.skip_files = {
            'env.template',
            'migrate_secrets.py',
            'secret_manager.py',
            'security_audit.py',
            'README.md',
            '.gitignore'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git',
            '__pycache__',
            '.pytest_cache',
            'node_modules',
            'venv',
            '.venv',
            '.env',
            'site-packages'
        }
    
    def scan_file(self, file_path: Path) -> List[Dict]:
        """
        Scan a single file for potential security issues
        """
        if file_path.name in self.skip_files:
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            issues = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern in self.credential_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's a comment about storing in SecretManager
                        if 'SecretManager' in line or 'Stored in SecretManager' in line:
                            continue
                        
                        # Skip if it's empty or placeholder
                        if match.strip('\'"') in ['', 'YOUR_API_KEY', 'YOUR_TOKEN', 'your_password']:
                            continue
                            
                        issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'content': line.strip(),
                            'match': match,
                            'severity': 'HIGH' if len(match.strip('\'"')) > 20 else 'MEDIUM'
                        })
            
            return issues
            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            return []
    
    def scan_directory(self, directory: Path = None) -> List[Dict]:
        """
        Recursively scan directory for security issues
        """
        if directory is None:
            directory = self.project_root
            
        all_issues = []
        
        for item in directory.iterdir():
            if item.is_dir():
                if item.name not in self.skip_dirs:
                    all_issues.extend(self.scan_directory(item))
            elif item.is_file():
                if item.suffix in ['.py', '.js', '.ts', '.json', '.yaml', '.yml', '.env']:
                    all_issues.extend(self.scan_file(item))
        
        return all_issues
    
    def check_env_file(self) -> List[Dict]:
        """
        Check .env file for credentials that should be migrated
        """
        env_path = self.project_root / '.env'
        if not env_path.exists():
            return []
        
        issues = []
        with open(env_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('\'"')
                
                # Check for API keys and secrets
                if any(keyword in key.upper() for keyword in ['API_KEY', 'API_SECRET', 'TOKEN', 'PASSWORD']):
                    if value and value not in ['', 'YOUR_API_KEY', 'YOUR_TOKEN']:
                        issues.append({
                            'file': '.env',
                            'line': line_num,
                            'content': f"{key}={value[:10]}...",
                            'match': key,
                            'severity': 'HIGH',
                            'message': 'Credential found in .env file - should be migrated to SecretManager'
                        })
        
        return issues
    
    def check_imports(self) -> List[Dict]:
        """
        Check if files are properly importing SecretManager
        """
        issues = []
        
        # Files that should use SecretManager
        files_to_check = [
            'core/exchange.py',
            'utils/notifier.py',
            'ml_components/market_sentiment.py',
            'data_sources/coingecko_source.py'
        ]
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                if 'SecretManager' in content:
                    # Check if properly imported
                    if 'from utils.secret_manager import SecretManager' not in content:
                        issues.append({
                            'file': file_path,
                            'line': 1,
                            'content': 'Missing proper SecretManager import',
                            'match': 'import',
                            'severity': 'MEDIUM'
                        })
        
        return issues
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive security audit report
        """
        print("🔍 Running Security Audit...")
        
        # Scan for hardcoded credentials
        code_issues = self.scan_directory()
        
        # Check .env file
        env_issues = self.check_env_file()
        
        # Check imports
        import_issues = self.check_imports()
        
        all_issues = code_issues + env_issues + import_issues
        
        # Generate report
        report = "# Security Audit Report\n\n"
        report += f"**Scan Date:** {Path(__file__).stat().st_mtime}\n"
        report += f"**Project Root:** {self.project_root}\n\n"
        
        if not all_issues:
            report += "✅ **No security issues found!**\n\n"
            report += "All credentials appear to be properly stored in SecretManager.\n"
        else:
            report += f"⚠️ **{len(all_issues)} security issues found:**\n\n"
            
            # Group by severity
            high_issues = [i for i in all_issues if i['severity'] == 'HIGH']
            medium_issues = [i for i in all_issues if i['severity'] == 'MEDIUM']
            
            if high_issues:
                report += "## 🔴 HIGH SEVERITY ISSUES\n\n"
                for issue in high_issues:
                    report += f"- **{issue['file']}:{issue['line']}**\n"
                    report += f"  - Content: `{issue['content']}`\n"
                    report += f"  - Match: `{issue['match']}`\n"
                    if 'message' in issue:
                        report += f"  - Message: {issue['message']}\n"
                    report += "\n"
            
            if medium_issues:
                report += "## 🟡 MEDIUM SEVERITY ISSUES\n\n"
                for issue in medium_issues:
                    report += f"- **{issue['file']}:{issue['line']}**\n"
                    report += f"  - Content: `{issue['content']}`\n"
                    report += f"  - Match: `{issue['match']}`\n"
                    report += "\n"
        
        report += "\n## Recommendations\n\n"
        report += "1. **Remove all hardcoded credentials** from code and config files\n"
        report += "2. **Migrate remaining .env credentials** to SecretManager\n"
        report += "3. **Use SecretManager.get_secret()** instead of os.getenv() for sensitive data\n"
        report += "4. **Add .env to .gitignore** to prevent accidental commits\n"
        
        return report
    
    def run_audit(self) -> bool:
        """
        Run the complete security audit
        """
        report = self.generate_report()
        
        # Save report
        report_path = self.project_root / 'security_audit_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📋 Security audit report saved to: {report_path}")
        
        # Print summary
        all_issues = self.scan_directory() + self.check_env_file() + self.check_imports()
        
        if all_issues:
            high_count = len([i for i in all_issues if i['severity'] == 'HIGH'])
            medium_count = len([i for i in all_issues if i['severity'] == 'MEDIUM'])
            
            print(f"⚠️ Found {len(all_issues)} security issues:")
            print(f"  - {high_count} HIGH severity")
            print(f"  - {medium_count} MEDIUM severity")
            
            return False
        else:
            print("✅ No security issues found!")
            return True

if __name__ == "__main__":
    auditor = SecurityAuditor()
    success = auditor.run_audit()
    
    if not success:
        sys.exit(1)