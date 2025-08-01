#!/usr/bin/env python3
"""
Integration Validation Script for Materials Orchestrator SDLC

This script validates that all SDLC components are properly integrated
and functional before final deployment.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationValidator:
    """Comprehensive integration validation for SDLC setup."""
    
    def __init__(self):
        """Initialize the integration validator."""
        self.repo_root = Path.cwd()
        self.validation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': 'unknown',
            'phases': {},
            'issues': [],
            'recommendations': []
        }
        
    def run_validation(self) -> Dict[str, Any]:
        """Run complete integration validation."""
        logger.info("🚀 Starting SDLC Integration Validation...")
        
        # Define validation phases
        validation_phases = [
            ('foundation', self._validate_foundation),
            ('development_env', self._validate_development_environment),
            ('testing', self._validate_testing_infrastructure),
            ('containerization', self._validate_containerization),
            ('monitoring', self._validate_monitoring),
            ('automation', self._validate_automation),
            ('security', self._validate_security),
            ('documentation', self._validate_documentation)
        ]
        
        # Execute validation phases
        total_score = 0
        max_score = 0
        
        for phase_name, validator_func in validation_phases:
            logger.info(f"📋 Validating {phase_name.replace('_', ' ').title()}...")
            
            try:
                phase_result = validator_func()
                self.validation_results['phases'][phase_name] = phase_result
                
                total_score += phase_result['score']
                max_score += phase_result['max_score']
                
                status_emoji = "✅" if phase_result['status'] == 'pass' else "❌" if phase_result['status'] == 'fail' else "⚠️"
                logger.info(f"{status_emoji} {phase_name}: {phase_result['message']}")
                
            except Exception as e:
                logger.error(f"❌ Error validating {phase_name}: {e}")
                self.validation_results['phases'][phase_name] = {
                    'status': 'error',
                    'score': 0,
                    'max_score': 100,
                    'message': f'Validation error: {str(e)}',
                    'details': []
                }
                max_score += 100
        
        # Calculate overall status
        overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        if overall_percentage >= 90:
            self.validation_results['overall_status'] = 'excellent'
        elif overall_percentage >= 80:
            self.validation_results['overall_status'] = 'good'
        elif overall_percentage >= 60:
            self.validation_results['overall_status'] = 'acceptable'
        else:
            self.validation_results['overall_status'] = 'needs_improvement'
        
        self.validation_results['overall_score'] = round(overall_percentage, 1)
        self.validation_results['total_score'] = f"{total_score}/{max_score}"
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"🎯 Integration Validation Complete: {overall_percentage:.1f}% ({self.validation_results['overall_status']})")
        
        return self.validation_results
    
    def _validate_foundation(self) -> Dict[str, Any]:
        """Validate foundation documentation and structure."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check essential documentation files
        essential_docs = [
            ('README.md', 20, 'Main project documentation'),
            ('ARCHITECTURE.md', 20, 'System architecture documentation'),
            ('PROJECT_CHARTER.md', 15, 'Project charter and objectives'),
            ('docs/ROADMAP.md', 15, 'Development roadmap'),
            ('docs/adr/', 15, 'Architecture decision records'),
            ('.gitignore', 10, 'Git ignore configuration'),
            ('LICENSE', 5, 'Project license')
        ]
        
        for doc_path, points, description in essential_docs:
            path = self.repo_root / doc_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
            else:
                result['details'].append(f"❌ Missing {description}: {doc_path}")
        
        # Validate documentation quality
        readme_path = self.repo_root / 'README.md'
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text()
                if len(readme_content) > 1000:
                    result['details'].append("✅ README has substantial content")
                else:
                    result['details'].append("⚠️ README could be more comprehensive")
            except Exception as e:
                result['details'].append(f"❌ Error reading README: {e}")
        
        # Set status based on score
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Foundation documentation is excellent'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Foundation documentation is good with minor gaps'
        else:
            result['status'] = 'fail'
            result['message'] = 'Foundation documentation needs significant improvement'
        
        return result
    
    def _validate_development_environment(self) -> Dict[str, Any]:
        """Validate development environment configuration."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check configuration files
        config_files = [
            ('.env.example', 25, 'Environment variables template'),
            ('requirements.txt', 25, 'Python dependencies'),
            ('scripts/setup-dev.sh', 20, 'Development setup script'),
            ('scripts/init-mongo.js', 15, 'Database initialization'),
            ('pyproject.toml', 15, 'Python project configuration')
        ]
        
        for config_path, points, description in config_files:
            path = self.repo_root / config_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
                
                # Validate file contents
                if config_path == 'requirements.txt':
                    try:
                        content = path.read_text()
                        if 'flask' in content.lower() and 'pymongo' in content.lower():
                            result['details'].append("✅ Core dependencies present")
                        else:
                            result['details'].append("⚠️ Some expected dependencies missing")
                    except Exception:
                        pass
                        
            else:
                result['details'].append(f"❌ Missing {description}: {config_path}")
        
        # Check if setup script is executable
        setup_script = self.repo_root / 'scripts/setup-dev.sh'
        if setup_script.exists():
            if setup_script.stat().st_mode & 0o111:
                result['details'].append("✅ Setup script is executable")
            else:
                result['details'].append("⚠️ Setup script should be executable")
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Development environment is well configured'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Development environment has minor issues'
        else:
            result['status'] = 'fail'
            result['message'] = 'Development environment needs configuration'
        
        return result
    
    def _validate_testing_infrastructure(self) -> Dict[str, Any]:
        """Validate testing infrastructure setup."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check test structure
        test_components = [
            ('tests/', 20, 'Test directory structure'),
            ('tests/fixtures/', 20, 'Test fixtures directory'),
            ('tests/fixtures/materials_data.py', 15, 'Materials test data'),
            ('tests/test_materials_orchestrator.py', 15, 'Core integration tests'),
            ('tests/e2e/', 15, 'End-to-end test directory'),
            ('tests/performance/', 15, 'Performance test directory')
        ]
        
        for test_path, points, description in test_components:
            path = self.repo_root / test_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
            else:
                result['details'].append(f"❌ Missing {description}: {test_path}")
        
        # Try to run tests (if pytest is available)
        try:
            test_result = subprocess.run(
                ['python', '-m', 'pytest', '--collect-only', '-q'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if test_result.returncode == 0:
                result['details'].append("✅ Tests can be collected successfully")
            else:
                result['details'].append("⚠️ Test collection has issues")
                
        except subprocess.TimeoutExpired:
            result['details'].append("⚠️ Test collection timed out")
        except Exception as e:
            result['details'].append(f"⚠️ Cannot run test collection: {e}")
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Testing infrastructure is comprehensive'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Testing infrastructure is functional'
        else:
            result['status'] = 'fail'
            result['message'] = 'Testing infrastructure needs work'
        
        return result
    
    def _validate_containerization(self) -> Dict[str, Any]:
        """Validate containerization setup."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check Docker files
        docker_components = [
            ('Dockerfile.production', 30, 'Production Docker configuration'),
            ('Dockerfile.development', 25, 'Development Docker configuration'),
            ('docker-compose.yml', 25, 'Docker Compose configuration'),
            ('.dockerignore', 20, 'Docker ignore file')
        ]
        
        for docker_path, points, description in docker_components:
            path = self.repo_root / docker_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
                
                # Validate Dockerfile content
                if 'Dockerfile' in docker_path:
                    try:
                        content = path.read_text()
                        if 'FROM python:' in content:
                            result['details'].append(f"✅ {docker_path} uses Python base image")
                        if 'USER ' in content and 'root' not in content.split('USER ')[-1].split('\n')[0]:
                            result['details'].append(f"✅ {docker_path} uses non-root user")
                        else:
                            result['details'].append(f"⚠️ {docker_path} might be running as root")
                    except Exception:
                        pass
                        
            else:
                result['details'].append(f"❌ Missing {description}: {docker_path}")
        
        # Try to validate Docker syntax (if Docker is available)
        production_dockerfile = self.repo_root / 'Dockerfile.production'
        if production_dockerfile.exists():
            try:
                docker_result = subprocess.run(
                    ['docker', 'build', '-f', 'Dockerfile.production', '--dry-run', '.'],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if docker_result.returncode == 0:
                    result['details'].append("✅ Production Dockerfile syntax is valid")
                else:
                    result['details'].append("⚠️ Production Dockerfile may have syntax issues")
                    
            except subprocess.TimeoutExpired:
                result['details'].append("⚠️ Docker validation timed out")
            except FileNotFoundError:
                result['details'].append("ℹ️ Docker not available for validation")
            except Exception:
                result['details'].append("ℹ️ Cannot validate Docker syntax")
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Containerization is well configured'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Containerization has minor issues'
        else:
            result['status'] = 'fail'
            result['message'] = 'Containerization needs configuration'
        
        return result
    
    def _validate_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring and observability setup."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check monitoring components
        monitoring_components = [
            ('src/health_check.py', 25, 'Health check endpoints'),
            ('config/logging.yaml', 20, 'Logging configuration'),
            ('docs/monitoring/', 20, 'Monitoring documentation'),
            ('docs/monitoring/runbooks/', 15, 'Operational runbooks'),
            ('monitoring/', 10, 'Monitoring configuration directory'),
            ('scripts/setup-monitoring.sh', 10, 'Monitoring setup script')
        ]
        
        for mon_path, points, description in monitoring_components:
            path = self.repo_root / mon_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
            else:
                result['details'].append(f"❌ Missing {description}: {mon_path}")
        
        # Check health check implementation
        health_check_path = self.repo_root / 'src/health_check.py'
        if health_check_path.exists():
            try:
                content = health_check_path.read_text()
                if 'health' in content.lower() and 'status' in content.lower():
                    result['details'].append("✅ Health check implementation looks functional")
                else:
                    result['details'].append("⚠️ Health check implementation may be incomplete")
            except Exception:
                pass
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Monitoring setup is comprehensive'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Monitoring setup is functional'
        else:
            result['status'] = 'fail'
            result['message'] = 'Monitoring setup needs work'
        
        return result
    
    def _validate_automation(self) -> Dict[str, Any]:
        """Validate automation and metrics setup."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check automation components
        automation_components = [
            ('.github/project-metrics.json', 25, 'Project metrics configuration'),
            ('scripts/collect-metrics.py', 25, 'Metrics collection script'),
            ('scripts/automated-maintenance.py', 25, 'Automated maintenance'),
            ('scripts/repo-health-check.py', 25, 'Repository health monitoring')
        ]
        
        for auto_path, points, description in automation_components:
            path = self.repo_root / auto_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
                
                # Validate script executability
                if path.suffix == '.py' and auto_path.startswith('scripts/'):
                    if path.stat().st_mode & 0o111:
                        result['details'].append(f"✅ {auto_path} is executable")
                    else:
                        result['details'].append(f"⚠️ {auto_path} should be executable")
                        
            else:
                result['details'].append(f"❌ Missing {description}: {auto_path}")
        
        # Test metrics configuration validity
        metrics_config = self.repo_root / '.github/project-metrics.json'
        if metrics_config.exists():
            try:
                with open(metrics_config, 'r') as f:
                    metrics_data = json.load(f)
                    if 'categories' in metrics_data and 'thresholds' in metrics_data:
                        result['details'].append("✅ Metrics configuration is well-structured")
                    else:
                        result['details'].append("⚠️ Metrics configuration may be incomplete")
            except json.JSONDecodeError:
                result['details'].append("❌ Metrics configuration has invalid JSON")
            except Exception:
                result['details'].append("⚠️ Cannot validate metrics configuration")
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Automation setup is excellent'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Automation setup is functional'
        else:
            result['status'] = 'fail'
            result['message'] = 'Automation setup needs work'
        
        return result
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security configuration."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check security files
        security_components = [
            ('SECURITY.md', 20, 'Security policy documentation'),
            ('.github/workflows/security-scan.yml', 25, 'Security scanning workflow'),
            ('docs/security/', 15, 'Security documentation'),
            ('.bandit', 10, 'Bandit security scanner configuration'),
            ('.safety-policy.json', 10, 'Safety scanner policy'),
            ('scripts/security-scan.sh', 20, 'Security scanning script')
        ]
        
        for sec_path, points, description in security_components:
            path = self.repo_root / sec_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
            else:
                result['details'].append(f"❌ Missing {description}: {sec_path}")
        
        # Check for sensitive data exposure
        sensitive_patterns = ['password', 'api_key', 'secret', 'token']
        potential_issues = []
        
        for pattern in ['*.py', '*.yml', '*.yaml', '*.json']:
            for file_path in self.repo_root.rglob(pattern):
                if any(exclude in str(file_path) for exclude in ['.git', '__pycache__', 'node_modules']):
                    continue
                try:
                    content = file_path.read_text().lower()
                    for sensitive in sensitive_patterns:
                        if f'{sensitive}=' in content or f'{sensitive}:' in content:
                            if 'example' not in str(file_path) and 'template' not in str(file_path):
                                potential_issues.append(str(file_path.relative_to(self.repo_root)))
                                break
                except Exception:
                    continue
        
        if not potential_issues:
            result['details'].append("✅ No obvious secrets found in repository")
        else:
            result['details'].append(f"⚠️ Potential secrets found in {len(potential_issues)} files")
            for issue in potential_issues[:3]:  # Show first 3
                result['details'].append(f"  - {issue}")
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Security setup is comprehensive'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Security setup is functional'
        else:
            result['status'] = 'fail'
            result['message'] = 'Security setup needs improvement'
        
        return result
    
    def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness and quality."""
        result = {
            'status': 'unknown',
            'score': 0,
            'max_score': 100,
            'message': '',
            'details': []
        }
        
        # Check documentation files
        doc_components = [
            ('README.md', 20, 'Main project documentation'),
            ('CONTRIBUTING.md', 15, 'Contributing guidelines'),
            ('CHANGELOG.md', 10, 'Change log'),
            ('docs/', 15, 'Documentation directory'),
            ('docs/api/', 10, 'API documentation'),
            ('docs/tutorials/', 10, 'User tutorials'),
            ('docs/deployment/', 10, 'Deployment guides'),
            ('docs/troubleshooting/', 10, 'Troubleshooting guides')
        ]
        
        for doc_path, points, description in doc_components:
            path = self.repo_root / doc_path
            if path.exists():
                result['score'] += points
                result['details'].append(f"✅ {description}")
            else:
                result['details'].append(f"❌ Missing {description}: {doc_path}")
        
        # Analyze README quality
        readme_path = self.repo_root / 'README.md'
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text().lower()
                sections = ['installation', 'usage', 'contributing', 'license']
                found_sections = [s for s in sections if s in readme_content]
                
                if len(found_sections) >= 3:
                    result['details'].append(f"✅ README has {len(found_sections)}/4 essential sections")
                else:
                    result['details'].append(f"⚠️ README missing sections: {set(sections) - set(found_sections)}")
                    
            except Exception:
                result['details'].append("⚠️ Cannot analyze README content")
        
        # Set status
        if result['score'] >= 85:
            result['status'] = 'pass'
            result['message'] = 'Documentation is comprehensive'
        elif result['score'] >= 70:
            result['status'] = 'warning'
            result['message'] = 'Documentation is adequate'
        else:
            result['status'] = 'fail'
            result['message'] = 'Documentation needs significant improvement'
        
        return result
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze failed validations
        for phase_name, phase_result in self.validation_results['phases'].items():
            if phase_result['status'] == 'fail':
                recommendations.append(f"🔴 Critical: Fix {phase_name.replace('_', ' ')} issues")
            elif phase_result['status'] == 'warning':
                recommendations.append(f"🟡 Improve: Address {phase_name.replace('_', ' ')} warnings")
        
        # Overall recommendations
        overall_score = self.validation_results.get('overall_score', 0)
        
        if overall_score < 70:
            recommendations.append("🚨 Repository requires significant work before production use")
            recommendations.append("📋 Focus on fixing critical issues first")
        elif overall_score < 85:
            recommendations.append("⚡ Repository is functional but has areas for improvement")
            recommendations.append("🎯 Address warning items to reach excellence")
        else:
            recommendations.append("🎉 Repository is in excellent condition!")
            recommendations.append("🔄 Continue regular maintenance and monitoring")
        
        # Specific recommendations
        recommendations.extend([
            "📊 Run health checks regularly using scripts/repo-health-check.py",
            "🔐 Ensure all secrets are properly managed and not committed",
            "🧪 Execute full test suite before any production deployment",
            "📚 Keep documentation updated as the system evolves",
            "⚙️ Set up automated dependency updates and security scanning"
        ])
        
        self.validation_results['recommendations'] = recommendations
    
    def save_report(self, output_file: str = 'integration-validation-report.json'):
        """Save validation report to file."""
        report_path = self.repo_root / output_file
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"📄 Validation report saved to: {report_path}")
        return report_path
    
    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "="*60)
        print("🎯 SDLC INTEGRATION VALIDATION SUMMARY")
        print("="*60)
        
        print(f"📊 Overall Score: {self.validation_results['overall_score']}%")
        print(f"🎭 Status: {self.validation_results['overall_status'].upper()}")
        print(f"📅 Validation Time: {self.validation_results['timestamp']}")
        print()
        
        print("📋 PHASE RESULTS:")
        print("-" * 40)
        
        for phase_name, phase_result in self.validation_results['phases'].items():
            status_emoji = "✅" if phase_result['status'] == 'pass' else "❌" if phase_result['status'] == 'fail' else "⚠️"
            score_pct = (phase_result['score'] / phase_result['max_score'] * 100) if phase_result['max_score'] > 0 else 0
            
            print(f"{status_emoji} {phase_name.replace('_', ' ').title()}: {score_pct:.1f}%")
            print(f"   {phase_result['message']}")
            
            if phase_result['details']:
                for detail in phase_result['details'][:3]:  # Show first 3 details
                    print(f"   {detail}")
                if len(phase_result['details']) > 3:
                    print(f"   ... and {len(phase_result['details']) - 3} more")
            print()
        
        print("💡 KEY RECOMMENDATIONS:")
        print("-" * 40)
        for rec in self.validation_results['recommendations'][:5]:  # Show top 5
            print(f"   {rec}")
        
        if len(self.validation_results['recommendations']) > 5:
            print(f"   ... and {len(self.validation_results['recommendations']) - 5} more")
        
        print("\n" + "="*60)


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SDLC Integration Validation')
    parser.add_argument('--output', '-o', default='integration-validation-report.json',
                       help='Output file for detailed report')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Run validation
    validator = IntegrationValidator()
    results = validator.run_validation()
    
    # Save report
    validator.save_report(args.output)
    
    # Print summary
    if not args.quiet:
        validator.print_summary()
    
    # Exit with appropriate code
    overall_score = results['overall_score']
    if overall_score >= 85:
        sys.exit(0)  # Excellent
    elif overall_score >= 70:
        sys.exit(1)  # Good with warnings
    else:
        sys.exit(2)  # Needs improvement


if __name__ == '__main__':
    main()