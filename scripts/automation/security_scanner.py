#!/usr/bin/env python3
"""Security scanner for materials orchestrator codebase."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Patterns that could indicate security issues
SECURITY_PATTERNS = {
    "hardcoded_secrets": [
        r"password\s*=\s*[\"'][^\"']+[\"']",
        r"secret\s*=\s*[\"'][^\"']+[\"']",
        r"api_key\s*=\s*[\"'][^\"']+[\"']",
        r"token\s*=\s*[\"'][^\"']+[\"']",
        r"key\s*=\s*[\"'][^\"']+[\"']",
    ],
    "dangerous_functions": [
        r"eval\s*\(",
        r"exec\s*\(",
        r"pickle\.loads\s*\(",
        r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
    ],
    "insecure_random": [
        r"random\.random\s*\(",
        r"random\.choice\s*\(",
        r"random\.randint\s*\(",
    ],
    "sql_injection": [
        r"[\"'].*%s.*[\"']",
        r"\.format\s*\(",
        r"\+.*[\"'].*SELECT.*[\"']",
    ],
}

# Files/patterns to ignore
IGNORE_PATTERNS = [
    r"__pycache__",
    r"\.git",
    r"\.pyc$",
    r"venv",
    r"node_modules",
    r"test_.*\.py$",  # Test files may have mock secrets
]


def should_ignore_file(file_path: str) -> bool:
    """Check if file should be ignored."""
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, file_path):
            return True
    return False


def scan_file_for_patterns(file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
    """Scan a file for security patterns.
    
    Args:
        file_path: Path to file to scan
        
    Returns:
        Dict of pattern_type -> [(line_number, line_content)]
    """
    if should_ignore_file(str(file_path)):
        return {}
    
    findings = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return {}
    
    for pattern_type, patterns in SECURITY_PATTERNS.items():
        matches = []
        
        for line_num, line in enumerate(lines, 1):
            line_clean = line.strip()
            
            # Skip comments and docstrings for some patterns
            if pattern_type == "hardcoded_secrets":
                if line_clean.startswith('#') or '"""' in line_clean or "'''" in line_clean:
                    continue
            
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append((line_num, line_clean))
        
        if matches:
            findings[pattern_type] = matches
    
    return findings


def scan_directory(directory: Path) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
    """Scan directory for security issues.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dict of file_path -> findings
    """
    all_findings = {}
    
    for root, dirs, files in os.walk(directory):
        # Filter directories
        dirs[:] = [d for d in dirs if not should_ignore_file(d)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                findings = scan_file_for_patterns(file_path)
                
                if findings:
                    all_findings[str(file_path)] = findings
    
    return all_findings


def generate_report(findings: Dict[str, Dict[str, List[Tuple[int, str]]]]) -> str:
    """Generate security scan report.
    
    Args:
        findings: Scan findings
        
    Returns:
        Report string
    """
    report = ["# Security Scan Report", ""]
    report.append(f"**Scan Date:** {subprocess.check_output(['date']).decode().strip()}")
    report.append(f"**Total Files Scanned:** {len(findings)}")
    report.append("")
    
    if not findings:
        report.append("✅ **No security issues found!**")
        return "\n".join(report)
    
    # Summary
    total_issues = sum(len(file_findings) for file_findings in findings.values())
    report.append(f"⚠️  **Total Issues Found:** {total_issues}")
    report.append("")
    
    # Issue breakdown by type
    issue_counts = {}
    for file_findings in findings.values():
        for issue_type in file_findings:
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + len(file_findings[issue_type])
    
    report.append("## Issue Summary")
    for issue_type, count in sorted(issue_counts.items()):
        report.append(f"- **{issue_type.replace('_', ' ').title()}:** {count}")
    report.append("")
    
    # Detailed findings
    report.append("## Detailed Findings")
    report.append("")
    
    for file_path, file_findings in sorted(findings.items()):
        report.append(f"### {file_path}")
        report.append("")
        
        for issue_type, matches in file_findings.items():
            report.append(f"**{issue_type.replace('_', ' ').title()}:**")
            for line_num, line_content in matches:
                report.append(f"- Line {line_num}: `{line_content[:100]}...`")
            report.append("")
    
    return "\n".join(report)


def run_additional_security_checks() -> List[str]:
    """Run additional security checks."""
    checks = []
    
    # Check for common security files
    security_files = ['.env', '.env.local', 'secrets.txt', 'passwords.txt']
    for file in security_files:
        if os.path.exists(file):
            checks.append(f"⚠️  Found potentially sensitive file: {file}")
    
    # Check file permissions
    sensitive_files = ['scripts/deploy.sh', 'start_production.sh']
    for file in sensitive_files:
        if os.path.exists(file):
            stat = os.stat(file)
            if stat.st_mode & 0o077:
                checks.append(f"⚠️  {file} has overly permissive permissions")
    
    # Check for secrets in environment
    env_secrets = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']
    for var in env_secrets:
        if any(var.lower() in k.lower() for k in os.environ):
            checks.append(f"⚠️  Potential secret in environment: {var}")
    
    return checks


def main():
    """Main security scanner function."""
    print("🔒 Materials Orchestrator Security Scanner")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print("❌ Source directory not found")
        sys.exit(1)
    
    print(f"📁 Scanning directory: {src_dir}")
    print()
    
    # Run pattern-based scan
    findings = scan_directory(src_dir)
    
    # Run additional checks
    additional_checks = run_additional_security_checks()
    
    # Generate and display report
    report = generate_report(findings)
    print(report)
    
    if additional_checks:
        print("\n## Additional Security Checks")
        for check in additional_checks:
            print(check)
    
    # Write report to file
    report_file = project_root / "security_scan_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
        if additional_checks:
            f.write("\n\n## Additional Security Checks\n")
            for check in additional_checks:
                f.write(f"{check}\n")
    
    print(f"\n📊 Report saved to: {report_file}")
    
    # Exit with appropriate code
    total_issues = len(findings) + len(additional_checks)
    if total_issues > 0:
        print(f"\n⚠️  Found {total_issues} potential security issues")
        sys.exit(1)
    else:
        print("\n✅ No security issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()