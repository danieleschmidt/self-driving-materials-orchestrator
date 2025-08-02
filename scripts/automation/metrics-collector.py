#!/usr/bin/env python3
"""
Automated metrics collection script for Self-Driving Materials Orchestrator.

This script collects metrics from various sources and updates the project metrics file.
Can be run manually or scheduled as a cron job.
"""

import json
import os
import sys
import argparse
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and updates project metrics from various sources."""
    
    def __init__(self, metrics_file: str = ".github/project-metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics_data = self._load_metrics()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'terragonlabs/self-driving-materials-orchestrator')
        
    def _load_metrics(self) -> Dict[str, Any]:
        """Load current metrics from JSON file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Metrics file not found: {self.metrics_file}")
            sys.exit(1)
    
    def _save_metrics(self) -> None:
        """Save updated metrics to JSON file."""
        self.metrics_data['project']['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
        logger.info(f"Metrics updated in {self.metrics_file}")
    
    def _github_api_request(self, endpoint: str) -> Optional[Dict]:
        """Make authenticated GitHub API request."""
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not provided, skipping GitHub metrics")
            return None
            
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        url = f"https://api.github.com/repos/{self.github_repo}/{endpoint}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            return None
    
    def collect_repository_metrics(self) -> None:
        """Collect repository health metrics from GitHub API."""
        logger.info("Collecting repository metrics...")
        
        # Get repository info
        repo_info = self._github_api_request("")
        if repo_info:
            self.metrics_data['metrics']['development']['repository_health'].update({
                'open_issues': repo_info.get('open_issues_count', 0),
                'stargazers_count': repo_info.get('stargazers_count', 0),
                'forks_count': repo_info.get('forks_count', 0)
            })
        
        # Get pull requests
        prs = self._github_api_request("pulls?state=open")
        if prs:
            self.metrics_data['metrics']['development']['repository_health']['open_prs'] = len(prs)
        
        # Get recent commits
        since = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'
        commits = self._github_api_request(f"commits?since={since}")
        if commits:
            self.metrics_data['metrics']['development']['repository_health']['commits_last_30_days'] = len(commits)
        
        # Get contributors
        contributors = self._github_api_request("contributors")
        if contributors:
            self.metrics_data['metrics']['development']['repository_health']['contributor_count'] = len(contributors)
    
    def collect_code_quality_metrics(self) -> None:
        """Collect code quality metrics from coverage reports and analysis tools."""
        logger.info("Collecting code quality metrics...")
        
        # Try to get coverage from coverage.xml
        coverage_file = Path("coverage.xml")
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage = float(root.attrib.get('line-rate', 0)) * 100
                self.metrics_data['metrics']['development']['code_quality']['coverage_current'] = round(coverage, 2)
            except Exception as e:
                logger.warning(f"Failed to parse coverage.xml: {e}")
        
        # Run pytest to get test counts
        try:
            result = subprocess.run(['pytest', '--collect-only', '-q'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                test_count = 0
                for line in lines:
                    if 'test session starts' in line:
                        continue
                    if line.strip().endswith('tests collected'):
                        test_count = int(line.strip().split()[0])
                        break
                
                self.metrics_data['metrics']['development']['testing']['unit_test_count'] = test_count
        except Exception as e:
            logger.warning(f"Failed to collect test metrics: {e}")
    
    def collect_security_metrics(self) -> None:
        """Collect security metrics from various security tools."""
        logger.info("Collecting security metrics...")
        
        # Check for security scan results
        security_files = [
            ("bandit-results.json", "bandit"),
            ("safety-results.json", "safety"),
            ("trivy-results.sarif", "trivy")
        ]
        
        total_vulnerabilities = 0
        critical_count = 0
        high_count = 0
        
        for filename, tool in security_files:
            filepath = Path(filename)
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        
                    if tool == "bandit":
                        results = data.get('results', [])
                        for result in results:
                            severity = result.get('issue_severity', '').lower()
                            if severity == 'high':
                                high_count += 1
                                total_vulnerabilities += 1
                            elif severity == 'medium':
                                total_vulnerabilities += 1
                                
                    elif tool == "safety":
                        vulnerabilities = data.get('vulnerabilities', [])
                        total_vulnerabilities += len(vulnerabilities)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {filename}: {e}")
        
        self.metrics_data['metrics']['development']['security'].update({
            'vulnerability_count': total_vulnerabilities,
            'critical_vulnerabilities': critical_count,
            'high_vulnerabilities': high_count,
            'last_security_scan': datetime.utcnow().isoformat() + 'Z'
        })
    
    def collect_deployment_metrics(self) -> None:
        """Collect deployment and infrastructure metrics."""
        logger.info("Collecting deployment metrics...")
        
        # Get deployment info from GitHub deployments API
        deployments = self._github_api_request("deployments")
        if deployments:
            recent_deployments = [d for d in deployments if 
                                datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')) > 
                                datetime.utcnow().replace(tzinfo=None) - timedelta(days=30)]
            
            self.metrics_data['metrics']['deployment']['reliability']['deployment_frequency'] = len(recent_deployments)
            
            # Calculate success rate
            successful = 0
            for deployment in recent_deployments:
                statuses = self._github_api_request(f"deployments/{deployment['id']}/statuses")
                if statuses and any(s['state'] == 'success' for s in statuses):
                    successful += 1
            
            if recent_deployments:
                success_rate = (successful / len(recent_deployments)) * 100
                self.metrics_data['metrics']['deployment']['reliability']['deployment_success_rate'] = round(success_rate, 2)
    
    def collect_application_metrics(self) -> None:
        """Collect application performance metrics from Prometheus or logs."""
        logger.info("Collecting application metrics...")
        
        # Try to get metrics from Prometheus
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        
        metrics_queries = {
            'api_response_time_p95_ms': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) * 1000',
            'api_error_rate': 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100',
            'experiments_per_day': 'increase(experiments_total[24h])',
            'experiment_success_rate': 'rate(experiments_total{status="success"}[24h]) / rate(experiments_total[24h]) * 100'
        }
        
        try:
            for metric_name, query in metrics_queries.items():
                response = requests.get(f"{prometheus_url}/api/v1/query", 
                                      params={'query': query}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success' and data['data']['result']:
                        value = float(data['data']['result'][0]['value'][1])
                        
                        # Update the appropriate metric
                        if metric_name in ['api_response_time_p95_ms', 'api_error_rate']:
                            self.metrics_data['metrics']['application']['performance'][metric_name] = round(value, 2)
                        elif metric_name in ['experiments_per_day', 'experiment_success_rate']:
                            self.metrics_data['metrics']['application']['business'][metric_name] = round(value, 2)
                            
        except Exception as e:
            logger.warning(f"Failed to collect Prometheus metrics: {e}")
    
    def update_historical_data(self) -> None:
        """Add current metrics to historical tracking."""
        logger.info("Updating historical data...")
        
        current_entry = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "coverage": self.metrics_data['metrics']['development']['code_quality']['coverage_current'],
            "experiments_per_day": self.metrics_data['metrics']['application']['business']['experiments_per_day'],
            "deployment_frequency": self.metrics_data['metrics']['deployment']['reliability']['deployment_frequency'],
            "vulnerability_count": self.metrics_data['metrics']['development']['security']['vulnerability_count']
        }
        
        # Add to history if not already present for today
        history = self.metrics_data.get('history', [])
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Remove any existing entry for today
        history = [entry for entry in history if entry['date'] != today]
        
        # Add new entry
        history.append(current_entry)
        
        # Keep only last 90 days
        history = sorted(history, key=lambda x: x['date'])[-90:]
        
        self.metrics_data['history'] = history
    
    def generate_alerts(self) -> None:
        """Check metrics against thresholds and generate alerts."""
        logger.info("Checking alert conditions...")
        
        alerts = []
        
        # Check coverage threshold
        coverage = self.metrics_data['metrics']['development']['code_quality']['coverage_current']
        coverage_threshold = self.metrics_data['tracking']['alerts']['coverage_below_threshold']['threshold']
        if coverage < coverage_threshold:
            alerts.append({
                'type': 'coverage_below_threshold',
                'severity': 'warning',
                'message': f"Code coverage ({coverage}%) is below threshold ({coverage_threshold}%)",
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        # Check critical vulnerabilities
        critical_vulns = self.metrics_data['metrics']['development']['security']['critical_vulnerabilities']
        if critical_vulns > 0:
            alerts.append({
                'type': 'critical_vulnerabilities',
                'severity': 'critical',
                'message': f"Found {critical_vulns} critical vulnerabilities",
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        # Store alerts
        self.metrics_data['alerts'] = alerts
        
        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                logger.warning(f"ALERT: {alert['message']}")
    
    def run_collection(self, sources: list = None) -> None:
        """Run metrics collection from specified sources."""
        if sources is None:
            sources = ['repository', 'code_quality', 'security', 'deployment', 'application']
        
        logger.info(f"Starting metrics collection for sources: {sources}")
        
        if 'repository' in sources:
            self.collect_repository_metrics()
        
        if 'code_quality' in sources:
            self.collect_code_quality_metrics()
        
        if 'security' in sources:
            self.collect_security_metrics()
        
        if 'deployment' in sources:
            self.collect_deployment_metrics()
        
        if 'application' in sources:
            self.collect_application_metrics()
        
        self.update_historical_data()
        self.generate_alerts()
        self._save_metrics()
        
        logger.info("Metrics collection completed successfully")


def main():
    """Main entry point for the metrics collector."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--sources', nargs='+', 
                       choices=['repository', 'code_quality', 'security', 'deployment', 'application'],
                       help='Specific sources to collect metrics from')
    parser.add_argument('--metrics-file', default='.github/project-metrics.json',
                       help='Path to metrics file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        collector = MetricsCollector(args.metrics_file)
        collector.run_collection(args.sources)
        
        print("✅ Metrics collection completed successfully")
        
        # Print summary
        metrics = collector.metrics_data['metrics']
        print(f"📊 Current Coverage: {metrics['development']['code_quality']['coverage_current']}%")
        print(f"🔒 Security Issues: {metrics['development']['security']['vulnerability_count']}")
        print(f"🚀 Deployments (30d): {metrics['deployment']['reliability']['deployment_frequency']}")
        print(f"⚗️  Experiments/day: {metrics['application']['business']['experiments_per_day']}")
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()