#!/usr/bin/env python3
"""
Automated metrics collection script for Materials Orchestrator project.

This script collects various metrics from different sources and updates
the project metrics dashboard and reporting systems.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
import subprocess
from github import Github


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect metrics from various sources."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize metrics collector."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.github_client = self._init_github_client()
        self.metrics = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from project-metrics.json."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
            
    def _init_github_client(self) -> Optional[Github]:
        """Initialize GitHub API client."""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            logger.warning("GITHUB_TOKEN not found, GitHub metrics will be limited")
            return None
        return Github(token)
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        logger.info("Starting metrics collection...")
        
        # Collect metrics from different sources
        tasks = [
            self._collect_github_metrics(),
            self._collect_code_quality_metrics(),
            self._collect_security_metrics(),
            self._collect_performance_metrics(),
            self._collect_business_metrics(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in metrics collection task {i}: {result}")
            else:
                self.metrics.update(result)
        
        # Add collection metadata
        self.metrics['collection_info'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'collector_version': '1.0.0',
            'sources_collected': len([r for r in results if not isinstance(r, Exception)])
        }
        
        logger.info(f"Metrics collection completed. Collected {len(self.metrics)} metric categories.")
        return self.metrics
    
    async def _collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        if not self.github_client:
            return {'github': {'error': 'GitHub client not available'}}
        
        try:
            repo_name = self.config['project']['repository']
            repo = self.github_client.get_repo(repo_name)
            
            # Get repository statistics
            stats = {
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'watchers': repo.watchers_count,
                'open_issues': repo.open_issues_count,
                'contributors': repo.get_contributors().totalCount,
                'languages': dict(repo.get_languages()),
                'size_kb': repo.size,
                'default_branch': repo.default_branch,
                'created_at': repo.created_at.isoformat(),
                'updated_at': repo.updated_at.isoformat(),
                'pushed_at': repo.pushed_at.isoformat(),
            }
            
            # Get recent activity
            since = datetime.utcnow() - timedelta(days=30)
            
            # Pull requests
            prs = list(repo.get_pulls(state='all', sort='updated', direction='desc'))
            recent_prs = [pr for pr in prs if pr.updated_at > since]
            
            pr_stats = {
                'total_prs': len(prs),
                'recent_prs': len(recent_prs),
                'open_prs': len([pr for pr in prs if pr.state == 'open']),
                'merged_prs_last_30d': len([pr for pr in recent_prs if pr.merged_at]),
                'avg_pr_age_days': self._calculate_avg_pr_age(prs[:50])  # Limit for performance
            }
            
            # Issues
            issues = list(repo.get_issues(state='all', since=since))
            issue_stats = {
                'total_issues': repo.open_issues_count,
                'recent_issues': len(issues),
                'open_issues': len([i for i in issues if i.state == 'open']),
                'closed_issues_last_30d': len([i for i in issues if i.state == 'closed'])
            }
            
            # Commits
            commits = list(repo.get_commits(since=since))
            commit_stats = {
                'commits_last_30d': len(commits),
                'unique_authors_last_30d': len(set(c.author.login for c in commits if c.author)),
                'avg_commits_per_day': len(commits) / 30
            }
            
            # Releases
            releases = list(repo.get_releases())
            release_stats = {
                'total_releases': len(releases),
                'latest_release': releases[0].tag_name if releases else None,
                'latest_release_date': releases[0].published_at.isoformat() if releases else None
            }
            
            return {
                'github': {
                    'repository': stats,
                    'pull_requests': pr_stats,
                    'issues': issue_stats,
                    'commits': commit_stats,
                    'releases': release_stats,
                    'collected_at': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting GitHub metrics: {e}")
            return {'github': {'error': str(e)}}
    
    def _calculate_avg_pr_age(self, prs: List) -> float:
        """Calculate average age of pull requests in days."""
        if not prs:
            return 0.0
        
        now = datetime.utcnow()
        total_age = sum((now - pr.created_at).days for pr in prs if pr.created_at)
        return total_age / len(prs) if prs else 0.0
    
    async def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        try:
            metrics = {}
            
            # Lines of code (using cloc if available)
            try:
                result = subprocess.run(
                    ['cloc', '--json', 'src/', 'tests/'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    cloc_data = json.loads(result.stdout)
                    metrics['lines_of_code'] = {
                        'total': cloc_data.get('SUM', {}).get('code', 0),
                        'by_language': {
                            lang: data.get('code', 0) 
                            for lang, data in cloc_data.items() 
                            if isinstance(data, dict) and 'code' in data
                        }
                    }
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
                logger.warning("Could not collect lines of code metrics")
            
            # Test coverage (from coverage.xml if exists)
            coverage_file = Path('coverage.xml')
            if coverage_file.exists():
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(coverage_file)
                    root = tree.getroot()
                    
                    # Extract coverage percentage
                    coverage_elem = root.find('.//coverage')
                    if coverage_elem is not None:
                        line_rate = float(coverage_elem.get('line-rate', 0))
                        branch_rate = float(coverage_elem.get('branch-rate', 0))
                        
                        metrics['test_coverage'] = {
                            'line_coverage': line_rate * 100,
                            'branch_coverage': branch_rate * 100,
                            'overall_coverage': (line_rate + branch_rate) / 2 * 100
                        }
                except Exception as e:
                    logger.warning(f"Could not parse coverage file: {e}")
            
            # Complexity metrics (using radon if available)
            try:
                result = subprocess.run(
                    ['radon', 'cc', 'src/', '--json'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    complexity_data = json.loads(result.stdout)
                    
                    # Calculate average complexity
                    all_complexities = []
                    for file_data in complexity_data.values():
                        for item in file_data:
                            if isinstance(item, dict) and 'complexity' in item:
                                all_complexities.append(item['complexity'])
                    
                    if all_complexities:
                        metrics['complexity'] = {
                            'average_complexity': sum(all_complexities) / len(all_complexities),
                            'max_complexity': max(all_complexities),
                            'high_complexity_count': len([c for c in all_complexities if c > 10])
                        }
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
                logger.warning("Could not collect complexity metrics")
            
            return {'code_quality': metrics}
            
        except Exception as e:
            logger.error(f"Error collecting code quality metrics: {e}")
            return {'code_quality': {'error': str(e)}}
    
    async def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        try:
            metrics = {}
            
            # Safety check for known vulnerabilities
            try:
                result = subprocess.run(
                    ['safety', 'check', '--json'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = safety_data if isinstance(safety_data, list) else []
                    
                    # Count by severity
                    severity_counts = {'high': 0, 'medium': 0, 'low': 0}
                    for vuln in vulnerabilities:
                        severity = vuln.get('severity', 'unknown').lower()
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                    
                    metrics['vulnerabilities'] = {
                        'total': len(vulnerabilities),
                        **severity_counts,
                        'details': vulnerabilities[:10]  # Limit details
                    }
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
                logger.warning("Could not collect safety metrics")
            
            # Bandit security scan
            try:
                result = subprocess.run(
                    ['bandit', '-r', 'src/', '-f', 'json'],
                    capture_output=True, text=True, timeout=60
                )
                # Bandit returns non-zero for findings, so check output instead
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    results = bandit_data.get('results', [])
                    
                    # Count by severity
                    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                    for finding in results:
                        severity = finding.get('issue_severity', 'UNKNOWN')
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                    
                    metrics['static_analysis'] = {
                        'total_findings': len(results),
                        'high_severity': severity_counts['HIGH'],
                        'medium_severity': severity_counts['MEDIUM'],
                        'low_severity': severity_counts['LOW']
                    }
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
                logger.warning("Could not collect Bandit metrics")
            
            return {'security': metrics}
            
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
            return {'security': {'error': str(e)}}
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            metrics = {}
            
            # Docker image size
            try:
                result = subprocess.run(
                    ['docker', 'images', '--format', 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if 'materials-orchestrator' in line:
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                metrics['docker_image_size'] = parts[2]
                                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                logger.warning("Could not collect Docker metrics")
            
            # Test execution times (from pytest cache if available)
            cache_dir = Path('.pytest_cache')
            if cache_dir.exists():
                try:
                    # Look for test timing information
                    # This is a simplified example - real implementation would parse pytest output
                    metrics['test_performance'] = {
                        'last_run': 'metrics_collection_time',
                        'note': 'Detailed test timing requires pytest-benchmark integration'
                    }
                except Exception:
                    logger.warning("Could not collect test performance metrics")
            
            return {'performance': metrics}
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {'performance': {'error': str(e)}}
    
    async def _collect_business_metrics(self) -> Dict[str, Any]:
        """Collect business/application-specific metrics."""
        try:
            metrics = {}
            
            # Check if application is running and collect metrics
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Try to get metrics from running application
                    response = await client.get('http://localhost:8000/metrics')
                    if response.status_code == 200:
                        # Parse Prometheus metrics format
                        metrics_text = response.text
                        
                        # Simple parsing for key metrics
                        # In production, you'd use a proper Prometheus client
                        for line in metrics_text.split('\n'):
                            if line.startswith('#'):
                                continue
                            if 'experiments_total' in line:
                                try:
                                    value = float(line.split()[-1])
                                    metrics['experiments_total'] = value
                                except (IndexError, ValueError):
                                    pass
                            elif 'campaigns_active_total' in line:
                                try:
                                    value = float(line.split()[-1])
                                    metrics['active_campaigns'] = value
                                except (IndexError, ValueError):
                                    pass
                    
                    # Get health status
                    health_response = await client.get('http://localhost:8000/health')
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        metrics['service_health'] = health_data.get('status', 'unknown')
                        
            except (httpx.RequestError, httpx.TimeoutException):
                logger.info("Application not running or not accessible for metrics collection")
                metrics['service_status'] = 'not_running'
            
            # File system metrics
            if Path('data/').exists():
                try:
                    # Count files in data directory
                    data_files = list(Path('data/').rglob('*'))
                    metrics['data_files_count'] = len([f for f in data_files if f.is_file()])
                    
                    # Calculate total size
                    total_size = sum(f.stat().st_size for f in data_files if f.is_file())
                    metrics['data_size_bytes'] = total_size
                except Exception:
                    logger.warning("Could not collect data directory metrics")
            
            return {'business': metrics}
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
            return {'business': {'error': str(e)}}
    
    def save_metrics(self, output_file: str = 'metrics-report.json') -> None:
        """Save collected metrics to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        report_lines = []
        report_lines.append("# Materials Orchestrator Metrics Summary")
        report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # GitHub metrics
        if 'github' in self.metrics and 'repository' in self.metrics['github']:
            gh = self.metrics['github']['repository']
            report_lines.append("## Repository Statistics")
            report_lines.append(f"- Stars: {gh.get('stars', 'N/A')}")
            report_lines.append(f"- Forks: {gh.get('forks', 'N/A')}")
            report_lines.append(f"- Open Issues: {gh.get('open_issues', 'N/A')}")
            report_lines.append(f"- Contributors: {gh.get('contributors', 'N/A')}")
            report_lines.append("")
        
        # Code quality
        if 'code_quality' in self.metrics:
            cq = self.metrics['code_quality']
            report_lines.append("## Code Quality")
            if 'lines_of_code' in cq:
                report_lines.append(f"- Total Lines of Code: {cq['lines_of_code'].get('total', 'N/A')}")
            if 'test_coverage' in cq:
                report_lines.append(f"- Test Coverage: {cq['test_coverage'].get('overall_coverage', 'N/A'):.1f}%")
            if 'complexity' in cq:
                report_lines.append(f"- Average Complexity: {cq['complexity'].get('average_complexity', 'N/A'):.2f}")
            report_lines.append("")
        
        # Security
        if 'security' in self.metrics:
            sec = self.metrics['security']
            report_lines.append("## Security")
            if 'vulnerabilities' in sec:
                vuln = sec['vulnerabilities']
                report_lines.append(f"- Total Vulnerabilities: {vuln.get('total', 'N/A')}")
                report_lines.append(f"- High Severity: {vuln.get('high', 'N/A')}")
                report_lines.append(f"- Medium Severity: {vuln.get('medium', 'N/A')}")
            if 'static_analysis' in sec:
                sa = sec['static_analysis']
                report_lines.append(f"- Static Analysis Findings: {sa.get('total_findings', 'N/A')}")
            report_lines.append("")
        
        # Business metrics
        if 'business' in self.metrics:
            biz = self.metrics['business']
            report_lines.append("## Application Status")
            report_lines.append(f"- Service Health: {biz.get('service_health', 'N/A')}")
            if 'experiments_total' in biz:
                report_lines.append(f"- Total Experiments: {biz['experiments_total']}")
            if 'active_campaigns' in biz:
                report_lines.append(f"- Active Campaigns: {biz['active_campaigns']}")
            report_lines.append("")
        
        return '\n'.join(report_lines)


async def main():
    """Main metrics collection function."""
    collector = MetricsCollector()
    
    # Collect all metrics
    metrics = await collector.collect_all_metrics()
    
    # Save detailed metrics
    collector.save_metrics('metrics-report.json')
    
    # Generate and save summary
    summary = collector.generate_summary_report()
    with open('metrics-summary.md', 'w') as f:
        f.write(summary)
    
    # Print summary to console
    print(summary)
    
    # Check for alerts
    config = collector.config
    alerts = []
    
    # Check test coverage
    if 'code_quality' in metrics and 'test_coverage' in metrics['code_quality']:
        coverage = metrics['code_quality']['test_coverage'].get('overall_coverage', 0)
        min_coverage = config.get('alerts', {}).get('thresholds', {}).get('test_coverage_min', 80)
        if coverage < min_coverage:
            alerts.append(f"Test coverage ({coverage:.1f}%) below threshold ({min_coverage}%)")
    
    # Check vulnerabilities
    if 'security' in metrics and 'vulnerabilities' in metrics['security']:
        vuln_count = metrics['security']['vulnerabilities'].get('total', 0)
        max_vulns = config.get('alerts', {}).get('thresholds', {}).get('vulnerability_count_max', 5)
        if vuln_count > max_vulns:
            alerts.append(f"Vulnerability count ({vuln_count}) above threshold ({max_vulns})")
    
    # Print alerts
    if alerts:
        print("\n🚨 ALERTS:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\n✅ All metrics within acceptable thresholds")
    
    return len(alerts) == 0


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)