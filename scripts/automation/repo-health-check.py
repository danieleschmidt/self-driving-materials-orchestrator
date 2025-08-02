#!/usr/bin/env python3
"""
Repository health check script for Self-Driving Materials Orchestrator.

Performs comprehensive health checks on the repository including:
- Code quality checks
- Security scans
- Documentation validation
- Dependency analysis
- Infrastructure validation
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthCheck:
    """Comprehensive repository health checker."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'overall_score': 0,
            'checks': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
    
    def run_command(self, cmd: List[str], check_return_code: bool = True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root,
                timeout=300
            )
            
            if check_return_code and result.returncode != 0:
                return False, result.stderr
            
            return True, result.stdout
            
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def check_file_exists(self, filepath: str, required: bool = False) -> bool:
        """Check if a file exists and optionally mark as required."""
        file_path = self.project_root / filepath
        exists = file_path.exists()
        
        if required and not exists:
            self.results['critical_issues'].append(f"Required file missing: {filepath}")
        elif not exists:
            self.results['warnings'].append(f"Optional file missing: {filepath}")
        
        return exists
    
    def check_project_structure(self) -> Dict[str, Any]:
        """Validate project structure and essential files."""
        logger.info("Checking project structure...")
        
        structure_check = {
            'score': 0,
            'max_score': 100,
            'details': {}
        }
        
        # Required files
        required_files = [
            'README.md',
            'LICENSE',
            'pyproject.toml',
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
            'SECURITY.md'
        ]
        
        # Optional but recommended files
        optional_files = [
            'CHANGELOG.md',
            'ARCHITECTURE.md',
            'PROJECT_CHARTER.md',
            '.env.example',
            '.gitignore',
            '.editorconfig'
        ]
        
        # Required directories
        required_dirs = [
            'src',
            'tests',
            'docs'
        ]
        
        # Check required files (60 points)
        required_score = 0
        for file in required_files:
            if self.check_file_exists(file, required=True):
                required_score += 10
                structure_check['details'][file] = '✅ Present'
            else:
                structure_check['details'][file] = '❌ Missing'
        
        # Check optional files (20 points)
        optional_score = 0
        for file in optional_files:
            if self.check_file_exists(file):
                optional_score += 3
                structure_check['details'][file] = '✅ Present'
            else:
                structure_check['details'][file] = '⚠️  Optional'
        
        # Check required directories (20 points)
        dir_score = 0
        for directory in required_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists() and dir_path.is_dir():
                dir_score += 7
                structure_check['details'][f"{directory}/"] = '✅ Present'
            else:
                structure_check['details'][f"{directory}/"] = '❌ Missing'
                self.results['critical_issues'].append(f"Required directory missing: {directory}/")
        
        structure_check['score'] = min(required_score + optional_score + dir_score, 100)
        return structure_check
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics and standards."""
        logger.info("Checking code quality...")
        
        quality_check = {
            'score': 0,
            'max_score': 100,
            'details': {}
        }
        
        # Check if linting tools are configured
        linting_configs = [
            ('ruff', 'pyproject.toml'),
            ('black', 'pyproject.toml'),
            ('mypy', 'pyproject.toml'),
            ('pre-commit', '.pre-commit-config.yaml')
        ]
        
        config_score = 0
        for tool, config_file in linting_configs:
            if self.check_file_exists(config_file):
                config_score += 5
                quality_check['details'][f"{tool}_config"] = '✅ Configured'
            else:
                quality_check['details'][f"{tool}_config"] = '❌ Missing'
        
        # Run linting tools if available
        lint_score = 0
        
        # Ruff check
        success, output = self.run_command(['ruff', 'check', 'src/', '--quiet'], check_return_code=False)
        if success and not output.strip():
            lint_score += 15
            quality_check['details']['ruff_check'] = '✅ No issues'
        else:
            quality_check['details']['ruff_check'] = f'⚠️  Issues found'
            if output:
                self.results['warnings'].append(f"Ruff issues: {len(output.splitlines())} problems")
        
        # Black formatting check
        success, output = self.run_command(['black', '--check', 'src/', 'tests/'], check_return_code=False)
        if success:
            lint_score += 15
            quality_check['details']['black_format'] = '✅ Properly formatted'
        else:
            quality_check['details']['black_format'] = '⚠️  Formatting issues'
            self.results['warnings'].append("Code formatting issues found")
        
        # MyPy type checking
        success, output = self.run_command(['mypy', 'src/'], check_return_code=False)
        if success:
            lint_score += 15
            quality_check['details']['mypy_types'] = '✅ Type check passed'
        else:
            quality_check['details']['mypy_types'] = '⚠️  Type issues'
            if 'error' in output.lower():
                self.results['warnings'].append("Type checking issues found")
        
        # Test coverage check
        success, output = self.run_command(['coverage', 'report', '--show-missing'], check_return_code=False)
        if success and 'TOTAL' in output:
            # Extract coverage percentage
            lines = output.strip().split('\n')
            for line in lines:
                if 'TOTAL' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_str = parts[-1].replace('%', '')
                        try:
                            coverage = float(coverage_str)
                            if coverage >= 80:
                                lint_score += 15
                                quality_check['details']['coverage'] = f'✅ {coverage}%'
                            elif coverage >= 60:
                                lint_score += 10
                                quality_check['details']['coverage'] = f'⚠️  {coverage}%'
                            else:
                                quality_check['details']['coverage'] = f'❌ {coverage}%'
                                self.results['warnings'].append(f"Low test coverage: {coverage}%")
                            break
                        except ValueError:
                            pass
        else:
            quality_check['details']['coverage'] = '❌ No coverage data'
        
        quality_check['score'] = min(config_score + lint_score, 100)
        return quality_check
    
    def check_security(self) -> Dict[str, Any]:
        """Perform security checks and vulnerability scanning."""
        logger.info("Checking security...")
        
        security_check = {
            'score': 0,
            'max_score': 100,
            'details': {}
        }
        
        # Check for security configuration files
        security_configs = [
            'SECURITY.md',
            '.github/dependabot.yml',
            'bandit.yml'
        ]
        
        config_score = 0
        for config in security_configs:
            if self.check_file_exists(config):
                config_score += 10
                security_check['details'][config] = '✅ Present'
            else:
                security_check['details'][config] = '⚠️  Missing'
        
        # Run security tools
        scan_score = 0
        
        # Bandit security scan
        success, output = self.run_command(['bandit', '-r', 'src/', '-f', 'json'], check_return_code=False)
        if success:
            try:
                bandit_results = json.loads(output)
                issues = bandit_results.get('results', [])
                high_severity = [r for r in issues if r.get('issue_severity') == 'HIGH']
                medium_severity = [r for r in issues if r.get('issue_severity') == 'MEDIUM']
                
                if not high_severity:
                    scan_score += 20
                    security_check['details']['bandit_scan'] = '✅ No high severity issues'
                else:
                    security_check['details']['bandit_scan'] = f'❌ {len(high_severity)} high severity issues'
                    self.results['critical_issues'].append(f"Bandit found {len(high_severity)} high severity security issues")
                
                if medium_severity:
                    self.results['warnings'].append(f"Bandit found {len(medium_severity)} medium severity issues")
                    
            except json.JSONDecodeError:
                security_check['details']['bandit_scan'] = '⚠️  Scan error'
        else:
            security_check['details']['bandit_scan'] = '❌ Tool not available'
        
        # Safety dependency check
        success, output = self.run_command(['safety', 'check', '--json'], check_return_code=False)
        if success:
            try:
                safety_results = json.loads(output)
                vulnerabilities = safety_results.get('vulnerabilities', [])
                
                if not vulnerabilities:
                    scan_score += 20
                    security_check['details']['safety_check'] = '✅ No vulnerable dependencies'
                else:
                    security_check['details']['safety_check'] = f'❌ {len(vulnerabilities)} vulnerable dependencies'
                    self.results['critical_issues'].append(f"Found {len(vulnerabilities)} vulnerable dependencies")
                    
            except json.JSONDecodeError:
                security_check['details']['safety_check'] = '⚠️  Scan error'
        else:
            security_check['details']['safety_check'] = '❌ Tool not available'
        
        # Check for exposed secrets
        patterns_to_check = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        import re
        exposed_secrets = 0
        for py_file in (self.project_root / 'src').rglob('*.py'):
            content = py_file.read_text()
            for pattern in patterns_to_check:
                if re.search(pattern, content, re.IGNORECASE):
                    exposed_secrets += 1
        
        if exposed_secrets == 0:
            scan_score += 10
            security_check['details']['secret_scan'] = '✅ No exposed secrets'
        else:
            security_check['details']['secret_scan'] = f'⚠️  {exposed_secrets} potential secrets'
            self.results['warnings'].append(f"Found {exposed_secrets} potential exposed secrets")
        
        security_check['score'] = min(config_score + scan_score, 100)
        return security_check
    
    def check_testing(self) -> Dict[str, Any]:
        """Check testing setup and coverage."""
        logger.info("Checking testing setup...")
        
        testing_check = {
            'score': 0,
            'max_score': 100,
            'details': {}
        }
        
        # Check test structure
        test_dirs = ['tests/unit', 'tests/integration', 'tests/e2e']
        structure_score = 0
        
        for test_dir in test_dirs:
            if (self.project_root / test_dir).exists():
                structure_score += 10
                testing_check['details'][test_dir] = '✅ Present'
            else:
                testing_check['details'][test_dir] = '⚠️  Missing'
        
        # Check test configuration
        if self.check_file_exists('pytest.ini') or self.check_file_exists('pyproject.toml'):
            structure_score += 10
            testing_check['details']['pytest_config'] = '✅ Configured'
        else:
            testing_check['details']['pytest_config'] = '❌ Missing'
        
        # Run tests and collect metrics
        test_score = 0
        
        # Count test files
        test_files = list((self.project_root / 'tests').rglob('test_*.py'))
        if test_files:
            test_count = len(test_files)
            if test_count >= 10:
                test_score += 15
                testing_check['details']['test_count'] = f'✅ {test_count} test files'
            elif test_count >= 5:
                test_score += 10
                testing_check['details']['test_count'] = f'⚠️  {test_count} test files'
            else:
                testing_check['details']['test_count'] = f'❌ Only {test_count} test files'
        else:
            testing_check['details']['test_count'] = '❌ No test files found'
        
        # Run pytest
        success, output = self.run_command(['pytest', '--tb=short', '-v'], check_return_code=False)
        if success:
            test_score += 25
            testing_check['details']['test_execution'] = '✅ All tests pass'
        else:
            testing_check['details']['test_execution'] = '❌ Some tests fail'
            self.results['critical_issues'].append("Test failures detected")
        
        testing_check['score'] = min(structure_score + test_score, 100)
        return testing_check
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness and quality."""
        logger.info("Checking documentation...")
        
        docs_check = {
            'score': 0,
            'max_score': 100,
            'details': {}
        }
        
        # Check documentation files
        doc_files = [
            'README.md',
            'CONTRIBUTING.md',
            'ARCHITECTURE.md',
            'docs/deployment/README.md',
            'docs/testing/README.md',
            'docs/monitoring/README.md'
        ]
        
        doc_score = 0
        for doc_file in doc_files:
            if self.check_file_exists(doc_file):
                doc_score += 10
                docs_check['details'][doc_file] = '✅ Present'
            else:
                docs_check['details'][doc_file] = '⚠️  Missing'
        
        # Check if documentation builds
        if self.check_file_exists('mkdocs.yml'):
            success, output = self.run_command(['mkdocs', 'build', '--strict'], check_return_code=False)
            if success:
                doc_score += 20
                docs_check['details']['docs_build'] = '✅ Builds successfully'
            else:
                docs_check['details']['docs_build'] = '❌ Build fails'
                self.results['warnings'].append("Documentation build fails")
        else:
            docs_check['details']['docs_build'] = '⚠️  No mkdocs.yml'
        
        # Check README quality
        readme_path = self.project_root / 'README.md'
        if readme_path.exists():
            readme_content = readme_path.read_text()
            readme_score = 0
            
            # Check for essential sections
            essential_sections = [
                'installation', 'usage', 'example', 'setup',
                'quick start', 'getting started'
            ]
            
            for section in essential_sections:
                if section.lower() in readme_content.lower():
                    readme_score += 5
                    break
            
            if len(readme_content) > 1000:
                readme_score += 5
                
            docs_check['details']['readme_quality'] = f'✅ Score: {readme_score}/10'
            doc_score += readme_score
        
        docs_check['score'] = min(doc_score, 100)
        return docs_check
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check dependency management and health."""
        logger.info("Checking dependencies...")
        
        deps_check = {
            'score': 0,
            'max_score': 100,
            'details': {}
        }
        
        # Check dependency files
        if self.check_file_exists('pyproject.toml'):
            deps_check['details']['dependency_file'] = '✅ pyproject.toml present'
            score = 30
        elif self.check_file_exists('requirements.txt'):
            deps_check['details']['dependency_file'] = '⚠️  Using requirements.txt'
            score = 20
        else:
            deps_check['details']['dependency_file'] = '❌ No dependency file'
            score = 0
            self.results['critical_issues'].append("No dependency management file found")
        
        # Check for security updates
        success, output = self.run_command(['pip', 'list', '--outdated', '--format=json'], check_return_code=False)
        if success:
            try:
                outdated_packages = json.loads(output)
                if not outdated_packages:
                    score += 30
                    deps_check['details']['outdated_packages'] = '✅ All packages up to date'
                else:
                    count = len(outdated_packages)
                    if count <= 5:
                        score += 20
                        deps_check['details']['outdated_packages'] = f'⚠️  {count} outdated packages'
                    else:
                        deps_check['details']['outdated_packages'] = f'❌ {count} outdated packages'
                        self.results['warnings'].append(f"{count} packages are outdated")
            except json.JSONDecodeError:
                deps_check['details']['outdated_packages'] = '⚠️  Could not check'
        
        # Check for lock files
        if self.check_file_exists('poetry.lock') or self.check_file_exists('Pipfile.lock'):
            score += 20
            deps_check['details']['lock_file'] = '✅ Lock file present'
        else:
            deps_check['details']['lock_file'] = '⚠️  No lock file'
        
        # Check for dev dependencies separation
        if self.check_file_exists('requirements-dev.txt') or 'dev' in Path('pyproject.toml').read_text():
            score += 20
            deps_check['details']['dev_dependencies'] = '✅ Dev deps separated'
        else:
            deps_check['details']['dev_dependencies'] = '⚠️  Dev deps not separated'
        
        deps_check['score'] = min(score, 100)
        return deps_check
    
    def generate_recommendations(self) -> None:
        """Generate actionable recommendations based on health check results."""
        recommendations = []
        
        # Analyze results and generate specific recommendations
        for check_name, check_result in self.results['checks'].items():
            score = check_result['score']
            max_score = check_result['max_score']
            percentage = (score / max_score) * 100
            
            if percentage < 70:
                if check_name == 'project_structure':
                    recommendations.append("📁 Improve project structure by adding missing files and directories")
                elif check_name == 'code_quality':
                    recommendations.append("🔧 Set up and run code quality tools (ruff, black, mypy)")
                elif check_name == 'security':
                    recommendations.append("🔒 Address security issues and set up automated security scanning")
                elif check_name == 'testing':
                    recommendations.append("🧪 Improve test coverage and add more test cases")
                elif check_name == 'documentation':
                    recommendations.append("📚 Complete documentation and ensure it builds correctly")
                elif check_name == 'dependencies':
                    recommendations.append("📦 Update dependencies and improve dependency management")
        
        # Add specific recommendations based on critical issues
        if self.results['critical_issues']:
            recommendations.append("🚨 Address critical issues immediately - they may block development")
        
        # Add general best practices
        recommendations.extend([
            "🔄 Set up automated CI/CD pipelines if not already configured",
            "📊 Implement metrics collection and monitoring",
            "🔐 Enable GitHub security features (Dependabot, code scanning)",
            "📋 Create issue and PR templates for consistent contributions",
            "🏷️  Use semantic versioning and proper release management"
        ])
        
        self.results['recommendations'] = recommendations
    
    def calculate_overall_score(self) -> int:
        """Calculate overall health score based on all checks."""
        if not self.results['checks']:
            return 0
        
        total_score = 0
        total_max_score = 0
        
        # Weight different categories
        weights = {
            'project_structure': 1.0,
            'code_quality': 1.5,
            'security': 2.0,
            'testing': 1.5,
            'documentation': 1.0,
            'dependencies': 1.0
        }
        
        for check_name, check_result in self.results['checks'].items():
            weight = weights.get(check_name, 1.0)
            weighted_score = check_result['score'] * weight
            weighted_max_score = check_result['max_score'] * weight
            
            total_score += weighted_score
            total_max_score += weighted_max_score
        
        if total_max_score == 0:
            return 0
        
        overall_percentage = (total_score / total_max_score) * 100
        return round(overall_percentage)
    
    def run_health_check(self, checks: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive health check."""
        if checks is None:
            checks = ['structure', 'quality', 'security', 'testing', 'documentation', 'dependencies']
        
        logger.info("Starting repository health check...")
        
        if 'structure' in checks:
            self.results['checks']['project_structure'] = self.check_project_structure()
        
        if 'quality' in checks:
            self.results['checks']['code_quality'] = self.check_code_quality()
        
        if 'security' in checks:
            self.results['checks']['security'] = self.check_security()
        
        if 'testing' in checks:
            self.results['checks']['testing'] = self.check_testing()
        
        if 'documentation' in checks:
            self.results['checks']['documentation'] = self.check_documentation()
        
        if 'dependencies' in checks:
            self.results['checks']['dependencies'] = self.check_dependencies()
        
        self.results['overall_score'] = self.calculate_overall_score()
        self.generate_recommendations()
        
        logger.info(f"Health check completed. Overall score: {self.results['overall_score']}/100")
        return self.results
    
    def print_report(self) -> None:
        """Print a formatted health check report."""
        print("\n" + "="*80)
        print("🏥 REPOSITORY HEALTH CHECK REPORT")
        print("="*80)
        print(f"📅 Generated: {self.results['timestamp']}")
        print(f"📊 Overall Score: {self.results['overall_score']}/100")
        
        # Score interpretation
        score = self.results['overall_score']
        if score >= 90:
            print("🎉 Excellent health! Repository is in great shape.")
        elif score >= 80:
            print("👍 Good health with room for improvement.")
        elif score >= 70:
            print("⚠️  Fair health. Several areas need attention.")
        else:
            print("🚨 Poor health. Immediate action required.")
        
        print("\n" + "-"*80)
        print("📋 DETAILED RESULTS")
        print("-"*80)
        
        for check_name, check_result in self.results['checks'].items():
            score = check_result['score']
            max_score = check_result['max_score']
            percentage = (score / max_score) * 100 if max_score > 0 else 0
            
            status_emoji = "✅" if percentage >= 80 else "⚠️ " if percentage >= 60 else "❌"
            print(f"\n{status_emoji} {check_name.replace('_', ' ').title()}: {score}/{max_score} ({percentage:.1f}%)")
            
            for item, status in check_result['details'].items():
                print(f"   • {item}: {status}")
        
        if self.results['critical_issues']:
            print("\n" + "-"*80)
            print("🚨 CRITICAL ISSUES")
            print("-"*80)
            for issue in self.results['critical_issues']:
                print(f"❌ {issue}")
        
        if self.results['warnings']:
            print("\n" + "-"*80)
            print("⚠️  WARNINGS")
            print("-"*80)
            for warning in self.results['warnings']:
                print(f"⚠️  {warning}")
        
        print("\n" + "-"*80)
        print("💡 RECOMMENDATIONS")
        print("-"*80)
        for i, recommendation in enumerate(self.results['recommendations'], 1):
            print(f"{i}. {recommendation}")
        
        print("\n" + "="*80)


def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(description='Repository health check')
    parser.add_argument('--checks', nargs='+', 
                       choices=['structure', 'quality', 'security', 'testing', 'documentation', 'dependencies'],
                       help='Specific checks to run')
    parser.add_argument('--output', choices=['text', 'json'], default='text',
                       help='Output format')
    parser.add_argument('--output-file', help='Save results to file')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        health_checker = HealthCheck(args.project_root)
        results = health_checker.run_health_check(args.checks)
        
        if args.output == 'json':
            output = json.dumps(results, indent=2)
            if args.output_file:
                Path(args.output_file).write_text(output)
                print(f"Results saved to {args.output_file}")
            else:
                print(output)
        else:
            health_checker.print_report()
            if args.output_file:
                # Save text report
                import io
                from contextlib import redirect_stdout
                
                f = io.StringIO()
                with redirect_stdout(f):
                    health_checker.print_report()
                
                Path(args.output_file).write_text(f.getvalue())
                print(f"\nReport saved to {args.output_file}")
        
        # Exit with appropriate code
        score = results['overall_score']
        if score >= 80:
            sys.exit(0)  # Success
        elif score >= 60:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Error
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()