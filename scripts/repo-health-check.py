#!/usr/bin/env python3
"""
Repository health check script for Materials Orchestrator.

This script performs comprehensive health checks on the repository
to ensure it follows best practices and maintains high quality.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthCheck:
    """Repository health check item."""
    
    def __init__(self, name: str, description: str, category: str):
        self.name = name
        self.description = description
        self.category = category
        self.status = "unknown"
        self.message = ""
        self.score = 0
        self.details = {}


class RepositoryHealthChecker:
    """Comprehensive repository health checker."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = []
        self.categories = {
            'structure': 'Repository Structure',
            'code_quality': 'Code Quality',
            'documentation': 'Documentation',
            'security': 'Security',
            'ci_cd': 'CI/CD',
            'maintenance': 'Maintenance',
            'community': 'Community Health'
        }
        self.max_score = 0
        self.total_score = 0
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        logger.info("Starting repository health check...")
        
        # Define all checks
        check_methods = [
            # Structure checks
            (self._check_readme_exists, 'structure'),
            (self._check_license_exists, 'structure'),
            (self._check_gitignore_exists, 'structure'),
            (self._check_project_structure, 'structure'),
            (self._check_config_files, 'structure'),
            
            # Code quality checks
            (self._check_code_formatting, 'code_quality'),
            (self._check_linting_config, 'code_quality'),
            (self._check_type_hints, 'code_quality'),
            (self._check_test_coverage, 'code_quality'),
            (self._check_code_complexity, 'code_quality'),
            
            # Documentation checks
            (self._check_api_documentation, 'documentation'),
            (self._check_user_documentation, 'documentation'),
            (self._check_changelog, 'documentation'),
            (self._check_contributing_guide, 'documentation'),
            (self._check_code_comments, 'documentation'),
            
            # Security checks
            (self._check_security_policy, 'security'),
            (self._check_dependency_security, 'security'),
            (self._check_secrets_exposure, 'security'),
            (self._check_permissions, 'security'),
            
            # CI/CD checks
            (self._check_github_workflows, 'ci_cd'),
            (self._check_branch_protection, 'ci_cd'),
            (self._check_automated_testing, 'ci_cd'),
            (self._check_deployment_config, 'ci_cd'),
            
            # Maintenance checks
            (self._check_dependency_freshness, 'maintenance'),
            (self._check_recent_activity, 'maintenance'),
            (self._check_issue_management, 'maintenance'),
            (self._check_pr_practices, 'maintenance'),
            
            # Community checks
            (self._check_community_files, 'community'),
            (self._check_issue_templates, 'community'),
            (self._check_pr_templates, 'community'),
            (self._check_code_of_conduct, 'community')
        ]
        
        # Run all checks
        for check_method, category in check_methods:
            try:
                check = check_method()
                check.category = category
                self.checks.append(check)
                self.max_score += 10  # Each check worth 10 points
                self.total_score += check.score
                
                status_emoji = "✅" if check.status == "pass" else "❌" if check.status == "fail" else "⚠️"
                logger.info(f"{status_emoji} {check.name}: {check.message}")
                
            except Exception as e:
                logger.error(f"Error running check {check_method.__name__}: {e}")
        
        # Calculate overall health score
        health_score = (self.total_score / self.max_score * 100) if self.max_score > 0 else 0
        
        # Generate report
        report = self._generate_report(health_score)
        logger.info(f"Repository health score: {health_score:.1f}%")
        
        return report
    
    def _generate_report(self, health_score: float) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        # Group checks by category
        categories = {}
        for check in self.checks:
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append({
                'name': check.name,
                'status': check.status,
                'message': check.message,
                'score': check.score,
                'details': check.details
            })
        
        # Calculate category scores
        category_scores = {}
        for category, checks in categories.items():
            total = len(checks) * 10
            achieved = sum(check['score'] for check in checks)
            category_scores[category] = (achieved / total * 100) if total > 0 else 0
        
        return {
            'overall_health_score': health_score,
            'total_checks': len(self.checks),
            'passed_checks': len([c for c in self.checks if c.status == 'pass']),
            'failed_checks': len([c for c in self.checks if c.status == 'fail']),
            'warning_checks': len([c for c in self.checks if c.status == 'warning']),
            'category_scores': category_scores,
            'categories': categories,
            'recommendations': self._generate_recommendations(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failed checks."""
        recommendations = []
        
        failed_checks = [c for c in self.checks if c.status == 'fail']
        warning_checks = [c for c in self.checks if c.status == 'warning']
        
        if failed_checks:
            recommendations.append("Address the following critical issues:")
            for check in failed_checks:
                recommendations.append(f"  - {check.name}: {check.message}")
        
        if warning_checks:
            recommendations.append("Consider improving the following areas:")
            for check in warning_checks:
                recommendations.append(f"  - {check.name}: {check.message}")
        
        # General recommendations based on score
        overall_score = (self.total_score / self.max_score * 100) if self.max_score > 0 else 0
        
        if overall_score < 70:
            recommendations.append("Repository needs significant improvements to meet best practices")
        elif overall_score < 85:
            recommendations.append("Repository is in good shape with some areas for improvement")
        else:
            recommendations.append("Repository follows excellent practices - keep up the good work!")
        
        return recommendations
    
    # Structure checks
    def _check_readme_exists(self) -> HealthCheck:
        """Check if README.md exists and has good content."""
        check = HealthCheck("README File", "Repository has comprehensive README", "structure")
        
        readme_path = Path("README.md")
        if readme_path.exists():
            content = readme_path.read_text()
            
            # Check for essential sections
            essential_sections = ['overview', 'installation', 'usage', 'contributing']
            found_sections = sum(1 for section in essential_sections if section.lower() in content.lower())
            
            if found_sections >= 3:
                check.status = "pass"
                check.score = 10
                check.message = f"README exists with {found_sections}/4 essential sections"
            else:
                check.status = "warning"
                check.score = 5
                check.message = f"README exists but missing some sections ({found_sections}/4)"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "README.md not found"
        
        return check
    
    def _check_license_exists(self) -> HealthCheck:
        """Check if LICENSE file exists."""
        check = HealthCheck("License File", "Repository has proper license", "structure")
        
        license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md']
        license_found = any(Path(f).exists() for f in license_files)
        
        if license_found:
            check.status = "pass"
            check.score = 10
            check.message = "License file found"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No license file found"
        
        return check
    
    def _check_gitignore_exists(self) -> HealthCheck:
        """Check if .gitignore exists and is comprehensive."""
        check = HealthCheck("Git Ignore", "Repository has comprehensive .gitignore", "structure")
        
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            
            # Check for common patterns
            important_patterns = ['__pycache__', '*.pyc', '.env', 'venv', '.pytest_cache']
            found_patterns = sum(1 for pattern in important_patterns if pattern in content)
            
            if found_patterns >= 4:
                check.status = "pass"
                check.score = 10
                check.message = f"Comprehensive .gitignore with {found_patterns}/5 important patterns"
            else:
                check.status = "warning"
                check.score = 7
                check.message = f".gitignore exists but could be more comprehensive ({found_patterns}/5)"
        else:
            check.status = "fail"
            check.score = 0
            check.message = ".gitignore not found"
        
        return check
    
    def _check_project_structure(self) -> HealthCheck:
        """Check project structure follows conventions."""
        check = HealthCheck("Project Structure", "Repository follows standard structure", "structure")
        
        expected_dirs = ['src', 'tests', 'docs']
        found_dirs = sum(1 for d in expected_dirs if Path(d).exists())
        
        # Check for source code organization
        has_src_package = any(Path('src').glob('*/')) if Path('src').exists() else False
        
        if found_dirs >= 2 and has_src_package:
            check.status = "pass"
            check.score = 10
            check.message = f"Good project structure ({found_dirs}/3 expected directories)"
        elif found_dirs >= 2:
            check.status = "warning"
            check.score = 7
            check.message = "Basic structure present, consider using src/ layout"
        else:
            check.status = "warning"
            check.score = 5
            check.message = f"Basic structure ({found_dirs}/3 expected directories)"
        
        return check
    
    def _check_config_files(self) -> HealthCheck:
        """Check for essential configuration files."""
        check = HealthCheck("Configuration Files", "Repository has proper configuration", "structure")
        
        config_files = [
            'pyproject.toml',
            '.pre-commit-config.yaml',
            '.editorconfig',
            'docker-compose.yml',
            'Dockerfile'
        ]
        
        found_configs = sum(1 for f in config_files if Path(f).exists())
        
        if found_configs >= 4:
            check.status = "pass"
            check.score = 10
            check.message = f"Excellent configuration ({found_configs}/5 files)"
        elif found_configs >= 2:
            check.status = "warning"
            check.score = 7
            check.message = f"Basic configuration ({found_configs}/5 files)"
        else:
            check.status = "fail"
            check.score = 3
            check.message = f"Minimal configuration ({found_configs}/5 files)"
        
        return check
    
    # Code quality checks
    def _check_code_formatting(self) -> HealthCheck:
        """Check if code formatting tools are configured."""
        check = HealthCheck("Code Formatting", "Code formatting tools configured", "code_quality")
        
        formatters = {
            'black': 'pyproject.toml',
            'isort': 'pyproject.toml',
            'prettier': '.prettierrc'
        }
        
        configured_formatters = []
        for formatter, config_file in formatters.items():
            if Path(config_file).exists():
                content = Path(config_file).read_text()
                if formatter in content.lower():
                    configured_formatters.append(formatter)
        
        if len(configured_formatters) >= 2:
            check.status = "pass"
            check.score = 10
            check.message = f"Formatters configured: {', '.join(configured_formatters)}"
        elif len(configured_formatters) >= 1:
            check.status = "warning"
            check.score = 7
            check.message = f"Some formatters configured: {', '.join(configured_formatters)}"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No code formatters configured"
        
        return check
    
    def _check_linting_config(self) -> HealthCheck:
        """Check if linting is configured."""
        check = HealthCheck("Code Linting", "Code linting tools configured", "code_quality")
        
        linters = ['ruff', 'flake8', 'pylint', 'mypy']
        pyproject_path = Path('pyproject.toml')
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            configured_linters = [linter for linter in linters if linter in content.lower()]
            
            if len(configured_linters) >= 2:
                check.status = "pass"
                check.score = 10
                check.message = f"Linters configured: {', '.join(configured_linters)}"
            elif len(configured_linters) >= 1:
                check.status = "warning"
                check.score = 7
                check.message = f"Basic linting: {', '.join(configured_linters)}"
            else:
                check.status = "fail"
                check.score = 0
                check.message = "No linters configured"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No pyproject.toml found"
        
        return check
    
    def _check_type_hints(self) -> HealthCheck:
        """Check for type hints usage."""
        check = HealthCheck("Type Hints", "Code uses type hints", "code_quality")
        
        python_files = list(Path('src').rglob('*.py')) if Path('src').exists() else []
        if not python_files:
            python_files = list(Path('.').glob('*.py'))
        
        if not python_files:
            check.status = "warning"
            check.score = 5
            check.message = "No Python files found to check"
            return check
        
        files_with_typing = 0
        for py_file in python_files[:10]:  # Check first 10 files
            try:
                content = py_file.read_text()
                if 'typing' in content or '->' in content or ': str' in content or ': int' in content:
                    files_with_typing += 1
            except Exception:
                continue
        
        if files_with_typing >= len(python_files) * 0.7:
            check.status = "pass"
            check.score = 10
            check.message = f"Good type hint usage ({files_with_typing}/{len(python_files[:10])} files)"
        elif files_with_typing >= len(python_files) * 0.3:
            check.status = "warning"
            check.score = 7
            check.message = f"Some type hints ({files_with_typing}/{len(python_files[:10])} files)"
        else:
            check.status = "fail"
            check.score = 3
            check.message = f"Limited type hints ({files_with_typing}/{len(python_files[:10])} files)"
        
        return check
    
    def _check_test_coverage(self) -> HealthCheck:
        """Check test coverage configuration."""
        check = HealthCheck("Test Coverage", "Test coverage tracking configured", "code_quality")
        
        coverage_indicators = [
            Path('.coverage').exists(),
            Path('coverage.xml').exists(),
            Path('htmlcov').exists(),
            'pytest-cov' in Path('pyproject.toml').read_text() if Path('pyproject.toml').exists() else False
        ]
        
        coverage_score = sum(coverage_indicators)
        
        if coverage_score >= 2:
            check.status = "pass"
            check.score = 10
            check.message = "Test coverage tracking configured"
        elif coverage_score >= 1:
            check.status = "warning"
            check.score = 7
            check.message = "Basic coverage tracking"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No test coverage tracking found"
        
        return check
    
    def _check_code_complexity(self) -> HealthCheck:
        """Check for code complexity analysis."""
        check = HealthCheck("Code Complexity", "Code complexity monitoring", "code_quality")
        
        # This is a simplified check - in reality you'd run complexity analysis tools
        pyproject_path = Path('pyproject.toml')
        has_complexity_tools = False
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            complexity_tools = ['radon', 'mccabe', 'xenon']
            has_complexity_tools = any(tool in content for tool in complexity_tools)
        
        if has_complexity_tools:
            check.status = "pass"
            check.score = 10
            check.message = "Code complexity tools configured"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "Consider adding complexity analysis tools"
        
        return check
    
    # Documentation checks
    def _check_api_documentation(self) -> HealthCheck:
        """Check for API documentation."""
        check = HealthCheck("API Documentation", "API is properly documented", "documentation")
        
        docs_indicators = [
            Path('docs/api').exists(),
            Path('docs').glob('*api*'),
            'mkdocs' in Path('pyproject.toml').read_text() if Path('pyproject.toml').exists() else False
        ]
        
        if any(docs_indicators):
            check.status = "pass"
            check.score = 10
            check.message = "API documentation found"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No API documentation found"
        
        return check
    
    def _check_user_documentation(self) -> HealthCheck:
        """Check for user documentation."""
        check = HealthCheck("User Documentation", "User documentation available", "documentation")
        
        user_docs = [
            Path('docs/quickstart').exists(),
            Path('docs/installation.md').exists(),
            Path('docs/tutorials').exists(),
            any(Path('docs').glob('*guide*'))
        ]
        
        docs_count = sum(user_docs)
        
        if docs_count >= 2:
            check.status = "pass"
            check.score = 10
            check.message = f"Good user documentation ({docs_count} sections)"
        elif docs_count >= 1:
            check.status = "warning"
            check.score = 7
            check.message = "Basic user documentation"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No user documentation found"
        
        return check
    
    def _check_changelog(self) -> HealthCheck:
        """Check for changelog."""
        check = HealthCheck("Changelog", "Project maintains changelog", "documentation")
        
        changelog_files = ['CHANGELOG.md', 'HISTORY.md', 'CHANGES.md']
        has_changelog = any(Path(f).exists() for f in changelog_files)
        
        if has_changelog:
            check.status = "pass"
            check.score = 10
            check.message = "Changelog found"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No changelog found"
        
        return check
    
    def _check_contributing_guide(self) -> HealthCheck:
        """Check for contributing guide."""
        check = HealthCheck("Contributing Guide", "Contributing guidelines available", "documentation")
        
        contributing_files = ['CONTRIBUTING.md', 'docs/CONTRIBUTING.md']
        has_contributing = any(Path(f).exists() for f in contributing_files)
        
        if has_contributing:
            check.status = "pass"
            check.score = 10
            check.message = "Contributing guide found"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No contributing guide found"
        
        return check
    
    def _check_code_comments(self) -> HealthCheck:
        """Check code comment quality."""
        check = HealthCheck("Code Comments", "Code is well commented", "documentation")
        
        # Simplified check - would need more sophisticated analysis in practice
        python_files = list(Path('src').rglob('*.py')) if Path('src').exists() else []
        
        if not python_files:
            check.status = "warning"
            check.score = 5
            check.message = "No Python files to check"
            return check
        
        files_with_docstrings = 0
        for py_file in python_files[:5]:  # Check first 5 files
            try:
                content = py_file.read_text()
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
            except Exception:
                continue
        
        if files_with_docstrings >= len(python_files[:5]) * 0.8:
            check.status = "pass"
            check.score = 10
            check.message = "Good documentation coverage"
        elif files_with_docstrings >= len(python_files[:5]) * 0.5:
            check.status = "warning"
            check.score = 7
            check.message = "Some documentation present"
        else:
            check.status = "fail"
            check.score = 3
            check.message = "Limited code documentation"
        
        return check
    
    # Security checks
    def _check_security_policy(self) -> HealthCheck:
        """Check for security policy."""
        check = HealthCheck("Security Policy", "Security policy documented", "security")
        
        security_files = ['SECURITY.md', '.github/SECURITY.md', 'docs/SECURITY.md']
        has_security_policy = any(Path(f).exists() for f in security_files)
        
        if has_security_policy:
            check.status = "pass"
            check.score = 10
            check.message = "Security policy found"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No security policy found"
        
        return check
    
    def _check_dependency_security(self) -> HealthCheck:
        """Check dependency security scanning."""
        check = HealthCheck("Dependency Security", "Dependencies scanned for vulnerabilities", "security")
        
        security_configs = [
            'safety' in Path('pyproject.toml').read_text() if Path('pyproject.toml').exists() else False,
            Path('.github/workflows').glob('*security*') if Path('.github/workflows').exists() else [],
            'bandit' in Path('pyproject.toml').read_text() if Path('pyproject.toml').exists() else False
        ]
        
        if any(security_configs):
            check.status = "pass"
            check.score = 10
            check.message = "Security scanning configured"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No security scanning found"
        
        return check
    
    def _check_secrets_exposure(self) -> HealthCheck:
        """Check for exposed secrets."""
        check = HealthCheck("Secrets Protection", "No secrets exposed in repository", "security")
        
        # Simple check for common secret patterns
        sensitive_patterns = ['password', 'api_key', 'secret', 'token']
        potential_issues = []
        
        for pattern in ['*.py', '*.yml', '*.yaml']:
            for file_path in Path('.').rglob(pattern):
                if any(exclude in str(file_path) for exclude in ['.git', '__pycache__', '.pytest_cache']):
                    continue
                try:
                    content = file_path.read_text().lower()
                    for sensitive in sensitive_patterns:
                        if f'{sensitive}=' in content or f'{sensitive}:' in content:
                            if 'example' not in str(file_path) and 'template' not in str(file_path):
                                potential_issues.append(str(file_path))
                                break
                except Exception:
                    continue
        
        if not potential_issues:
            check.status = "pass"
            check.score = 10
            check.message = "No obvious secrets found"
        else:
            check.status = "warning"
            check.score = 7
            check.message = f"Potential secrets in {len(potential_issues)} files"
            check.details = {'files': potential_issues[:5]}
        
        return check
    
    def _check_permissions(self) -> HealthCheck:
        """Check file permissions."""
        check = HealthCheck("File Permissions", "File permissions are appropriate", "security")
        
        # Check for executable files that shouldn't be
        script_extensions = ['.py', '.sh', '.bash']
        executable_scripts = []
        
        for ext in script_extensions:
            for file_path in Path('.').rglob(f'*{ext}'):
                if file_path.stat().st_mode & 0o111:  # Check if executable
                    if 'scripts/' in str(file_path) or file_path.name.startswith('run_'):
                        continue  # These should be executable
                    executable_scripts.append(str(file_path))
        
        if len(executable_scripts) == 0:
            check.status = "pass"
            check.score = 10
            check.message = "File permissions look correct"
        else:
            check.status = "warning"
            check.score = 7
            check.message = f"Some unexpected executable files: {len(executable_scripts)}"
        
        return check
    
    # CI/CD checks
    def _check_github_workflows(self) -> HealthCheck:
        """Check for GitHub Actions workflows."""
        check = HealthCheck("GitHub Workflows", "CI/CD workflows configured", "ci_cd")
        
        workflows_dir = Path('.github/workflows')
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))
            
            if len(workflow_files) >= 2:
                check.status = "pass"
                check.score = 10
                check.message = f"Good workflow coverage ({len(workflow_files)} workflows)"
            elif len(workflow_files) >= 1:
                check.status = "warning"
                check.score = 7
                check.message = "Basic CI/CD configured"
            else:
                check.status = "fail"
                check.score = 0
                check.message = "No workflows found"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No .github/workflows directory"
        
        return check
    
    def _check_branch_protection(self) -> HealthCheck:
        """Check branch protection (simulated)."""
        check = HealthCheck("Branch Protection", "Main branch is protected", "ci_cd")
        
        # This would require GitHub API access to check properly
        # For now, we'll check for workflow files that suggest protection
        workflows_dir = Path('.github/workflows')
        has_pr_workflow = False
        
        if workflows_dir.exists():
            for workflow in workflows_dir.glob('*.yml'):
                content = workflow.read_text()
                if 'pull_request' in content:
                    has_pr_workflow = True
                    break
        
        if has_pr_workflow:
            check.status = "pass"
            check.score = 10
            check.message = "PR workflows suggest branch protection"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "Branch protection status unknown"
        
        return check
    
    def _check_automated_testing(self) -> HealthCheck:
        """Check for automated testing setup."""
        check = HealthCheck("Automated Testing", "Automated testing configured", "ci_cd")
        
        test_indicators = [
            Path('tests').exists(),
            'pytest' in Path('pyproject.toml').read_text() if Path('pyproject.toml').exists() else False,
            any(Path('.github/workflows').glob('*test*')) if Path('.github/workflows').exists() else False
        ]
        
        test_score = sum(test_indicators)
        
        if test_score >= 3:
            check.status = "pass"
            check.score = 10
            check.message = "Comprehensive testing setup"
        elif test_score >= 2:
            check.status = "warning"
            check.score = 7
            check.message = "Basic testing setup"
        else:
            check.status = "fail"
            check.score = 3
            check.message = "Limited testing setup"
        
        return check
    
    def _check_deployment_config(self) -> HealthCheck:
        """Check deployment configuration."""
        check = HealthCheck("Deployment Config", "Deployment configuration present", "ci_cd")
        
        deployment_files = [
            'Dockerfile',
            'docker-compose.yml',
            'k8s',
            '.github/workflows'
        ]
        
        found_configs = sum(1 for f in deployment_files if Path(f).exists())
        
        if found_configs >= 3:
            check.status = "pass"
            check.score = 10
            check.message = "Good deployment configuration"
        elif found_configs >= 2:
            check.status = "warning"
            check.score = 7
            check.message = "Basic deployment configuration"
        else:
            check.status = "fail"
            check.score = 3
            check.message = "Limited deployment configuration"
        
        return check
    
    # Maintenance checks
    def _check_dependency_freshness(self) -> HealthCheck:
        """Check dependency freshness."""
        check = HealthCheck("Dependency Freshness", "Dependencies are up to date", "maintenance")
        
        # This would require checking actual package versions
        # For now, check if dependency management is configured
        dep_management = [
            'dependabot' in Path('.github').read_text() if Path('.github').exists() else False,
            any(Path('.github/workflows').glob('*dependency*')) if Path('.github/workflows').exists() else False,
            Path('requirements.txt').exists(),
            'dependencies' in Path('pyproject.toml').read_text() if Path('pyproject.toml').exists() else False
        ]
        
        if sum(dep_management) >= 2:
            check.status = "pass"
            check.score = 10
            check.message = "Dependency management configured"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "Basic dependency management"
        
        return check
    
    def _check_recent_activity(self) -> HealthCheck:
        """Check for recent activity."""
        check = HealthCheck("Recent Activity", "Repository is actively maintained", "maintenance")
        
        # Check git log for recent commits (simplified)
        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', '-10'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                commit_count = len(result.stdout.strip().split('\n'))
                if commit_count >= 5:
                    check.status = "pass"
                    check.score = 10
                    check.message = f"Active development ({commit_count} recent commits)"
                else:
                    check.status = "warning"
                    check.score = 7
                    check.message = "Some recent activity"
            else:
                check.status = "warning"
                check.score = 5
                check.message = "Limited recent activity"
        except Exception:
            check.status = "warning"
            check.score = 5
            check.message = "Cannot determine activity level"
        
        return check
    
    def _check_issue_management(self) -> HealthCheck:
        """Check issue management setup."""
        check = HealthCheck("Issue Management", "Issue management is configured", "maintenance")
        
        issue_templates_dir = Path('.github/ISSUE_TEMPLATE')
        has_issue_templates = issue_templates_dir.exists() and any(issue_templates_dir.iterdir())
        
        if has_issue_templates:
            check.status = "pass"
            check.score = 10
            check.message = "Issue templates configured"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No issue templates found"
        
        return check
    
    def _check_pr_practices(self) -> HealthCheck:
        """Check PR practices."""
        check = HealthCheck("PR Practices", "Pull request practices configured", "maintenance")
        
        pr_template = Path('.github/PULL_REQUEST_TEMPLATE.md')
        
        if pr_template.exists():
            check.status = "pass"
            check.score = 10
            check.message = "PR template configured"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No PR template found"
        
        return check
    
    # Community checks
    def _check_community_files(self) -> HealthCheck:
        """Check for community health files."""
        check = HealthCheck("Community Files", "Community health files present", "community")
        
        community_files = [
            'CODE_OF_CONDUCT.md',
            'CONTRIBUTING.md',
            'SECURITY.md',
            '.github/SUPPORT.md'
        ]
        
        found_files = sum(1 for f in community_files if Path(f).exists())
        
        if found_files >= 3:
            check.status = "pass"
            check.score = 10
            check.message = f"Good community health ({found_files}/4 files)"
        elif found_files >= 2:
            check.status = "warning"
            check.score = 7
            check.message = f"Basic community health ({found_files}/4 files)"
        else:
            check.status = "fail"
            check.score = 3
            check.message = f"Limited community health ({found_files}/4 files)"
        
        return check
    
    def _check_issue_templates(self) -> HealthCheck:
        """Check for issue templates."""
        check = HealthCheck("Issue Templates", "Issue templates configured", "community")
        
        templates_dir = Path('.github/ISSUE_TEMPLATE')
        if templates_dir.exists():
            templates = list(templates_dir.glob('*.md')) + list(templates_dir.glob('*.yml'))
            
            if len(templates) >= 2:
                check.status = "pass"
                check.score = 10
                check.message = f"Multiple issue templates ({len(templates)})"
            elif len(templates) >= 1:
                check.status = "warning"
                check.score = 7
                check.message = "Basic issue template"
            else:
                check.status = "fail"
                check.score = 0
                check.message = "No issue templates"
        else:
            check.status = "fail"
            check.score = 0
            check.message = "No issue templates directory"
        
        return check
    
    def _check_pr_templates(self) -> HealthCheck:
        """Check for PR templates."""
        check = HealthCheck("PR Templates", "Pull request templates configured", "community")
        
        pr_template_files = [
            '.github/PULL_REQUEST_TEMPLATE.md',
            '.github/pull_request_template.md'
        ]
        
        has_pr_template = any(Path(f).exists() for f in pr_template_files)
        
        if has_pr_template:
            check.status = "pass"
            check.score = 10
            check.message = "PR template configured"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No PR template found"
        
        return check
    
    def _check_code_of_conduct(self) -> HealthCheck:
        """Check for code of conduct."""
        check = HealthCheck("Code of Conduct", "Code of conduct present", "community")
        
        coc_files = ['CODE_OF_CONDUCT.md', '.github/CODE_OF_CONDUCT.md']
        has_coc = any(Path(f).exists() for f in coc_files)
        
        if has_coc:
            check.status = "pass"
            check.score = 10
            check.message = "Code of conduct found"
        else:
            check.status = "warning"
            check.score = 5
            check.message = "No code of conduct found"
        
        return check


def main():
    """Main health check function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repository health check for Materials Orchestrator')
    parser.add_argument('--output', '-o', default='health-report.json',
                       help='Output file for detailed report')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    checker = RepositoryHealthChecker()
    report = checker.run_all_checks()
    
    # Save detailed report
    if args.format == 'json':
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    else:
        # Generate markdown report
        markdown_report = _generate_markdown_report(report)
        output_file = args.output.replace('.json', '.md')
        with open(output_file, 'w') as f:
            f.write(markdown_report)
    
    print(f"Health report saved to {args.output}")
    
    # Print summary
    print(f"\nRepository Health Score: {report['overall_health_score']:.1f}%")
    print(f"Checks: {report['passed_checks']} passed, {report['failed_checks']} failed, {report['warning_checks']} warnings")
    
    # Return appropriate exit code
    if report['overall_health_score'] >= 80:
        sys.exit(0)  # Excellent health
    elif report['overall_health_score'] >= 60:
        sys.exit(1)  # Good health with warnings
    else:
        sys.exit(2)  # Poor health


def _generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate markdown health report."""
    lines = []
    lines.append("# Repository Health Report")
    lines.append(f"**Generated**: {report['timestamp']}")
    lines.append(f"**Overall Score**: {report['overall_health_score']:.1f}%")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append(f"- Total Checks: {report['total_checks']}")
    lines.append(f"- Passed: {report['passed_checks']} ✅")
    lines.append(f"- Failed: {report['failed_checks']} ❌")
    lines.append(f"- Warnings: {report['warning_checks']} ⚠️")
    lines.append("")
    
    # Category scores
    lines.append("## Category Scores")
    for category, score in report['category_scores'].items():
        emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        lines.append(f"- {category}: {score:.1f}% {emoji}")
    lines.append("")
    
    # Detailed results by category
    for category, checks in report['categories'].items():
        lines.append(f"## {category}")
        for check in checks:
            status_emoji = "✅" if check['status'] == 'pass' else "❌" if check['status'] == 'fail' else "⚠️"
            lines.append(f"- {status_emoji} **{check['name']}**: {check['message']}")
        lines.append("")
    
    # Recommendations
    if report['recommendations']:
        lines.append("## Recommendations")
        for rec in report['recommendations']:
            lines.append(f"- {rec}")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    main()