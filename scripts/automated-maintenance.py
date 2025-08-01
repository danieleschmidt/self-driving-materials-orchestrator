#!/usr/bin/env python3
"""
Automated maintenance script for Materials Orchestrator project.

This script performs routine maintenance tasks to keep the repository
healthy and up-to-date.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaintenanceRunner:
    """Automated maintenance task runner."""
    
    def __init__(self, dry_run: bool = False):
        """Initialize maintenance runner."""
        self.dry_run = dry_run
        self.tasks_completed = []
        self.tasks_failed = []
        
    def run_all_tasks(self) -> bool:
        """Run all maintenance tasks."""
        logger.info("Starting automated maintenance tasks...")
        if self.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
        
        tasks = [
            ("Clean temporary files", self._clean_temp_files),
            ("Clean Python cache", self._clean_python_cache),
            ("Clean build artifacts", self._clean_build_artifacts),
            ("Update pre-commit hooks", self._update_precommit_hooks),
            ("Clean Docker resources", self._clean_docker_resources),
            ("Organize log files", self._organize_log_files),
            ("Clean test artifacts", self._clean_test_artifacts),
            ("Update documentation timestamps", self._update_doc_timestamps),
            ("Check disk usage", self._check_disk_usage),
            ("Generate maintenance report", self._generate_maintenance_report),
        ]
        
        for task_name, task_func in tasks:
            try:
                logger.info(f"Running task: {task_name}")
                if not self.dry_run:
                    task_func()
                else:
                    logger.info(f"[DRY RUN] Would execute: {task_name}")
                self.tasks_completed.append(task_name)
                logger.info(f"✅ Completed: {task_name}")
            except Exception as e:
                logger.error(f"❌ Failed: {task_name} - {e}")
                self.tasks_failed.append((task_name, str(e)))
        
        # Print summary
        self._print_summary()
        
        return len(self.tasks_failed) == 0
    
    def _clean_temp_files(self) -> None:
        """Clean temporary files and directories."""
        temp_patterns = [
            '**/*.tmp',
            '**/*.temp',
            '**/tmp/*',
            '**/.DS_Store',
            '**/Thumbs.db',
            '**/*.swp',
            '**/*.swo',
            '**/.*~',
        ]
        
        removed_count = 0
        for pattern in temp_patterns:
            for path in Path('.').rglob('*'):
                if path.is_file() and self._matches_pattern(str(path), pattern):
                    if not self.dry_run:
                        path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed temp file: {path}")
        
        logger.info(f"Cleaned {removed_count} temporary files")
    
    def _clean_python_cache(self) -> None:
        """Clean Python cache files and directories."""
        cache_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/*.pyd',
            '.pytest_cache',
            '.mypy_cache',
            '.ruff_cache',
        ]
        
        removed_count = 0
        for pattern in cache_patterns:
            for path in Path('.').rglob('*'):
                if self._matches_pattern(str(path), pattern):
                    if path.is_dir():
                        if not self.dry_run:
                            shutil.rmtree(path)
                        removed_count += 1
                        logger.debug(f"Removed cache directory: {path}")
                    elif path.is_file():
                        if not self.dry_run:
                            path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed cache file: {path}")
        
        logger.info(f"Cleaned {removed_count} Python cache items")
    
    def _clean_build_artifacts(self) -> None:
        """Clean build artifacts and output directories."""
        build_dirs = [
            'build',
            'dist',
            '*.egg-info',
            '.eggs',
            'htmlcov',
            'site',  # MkDocs output
        ]
        
        removed_count = 0
        for pattern in build_dirs:
            for path in Path('.').glob(pattern):
                if path.is_dir():
                    if not self.dry_run:
                        shutil.rmtree(path)
                    removed_count += 1
                    logger.debug(f"Removed build directory: {path}")
                elif path.is_file():
                    if not self.dry_run:
                        path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed build file: {path}")
        
        logger.info(f"Cleaned {removed_count} build artifacts")
    
    def _update_precommit_hooks(self) -> None:
        """Update pre-commit hooks to latest versions."""
        try:
            if not Path('.pre-commit-config.yaml').exists():
                logger.info("No pre-commit configuration found, skipping")
                return
            
            if not self.dry_run:
                # Update pre-commit hooks
                result = subprocess.run(
                    ['pre-commit', 'autoupdate'],
                    capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    logger.info("Pre-commit hooks updated successfully")
                    if result.stdout:
                        logger.debug(f"Pre-commit output: {result.stdout}")
                else:
                    logger.warning(f"Pre-commit update failed: {result.stderr}")
            else:
                logger.info("[DRY RUN] Would update pre-commit hooks")
                
        except subprocess.TimeoutExpired:
            logger.error("Pre-commit update timed out")
        except FileNotFoundError:
            logger.warning("Pre-commit not installed, skipping hook update")
    
    def _clean_docker_resources(self) -> None:
        """Clean unused Docker resources."""
        try:
            if not self.dry_run:
                # Clean unused images
                result = subprocess.run(
                    ['docker', 'image', 'prune', '-f'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    logger.info("Docker images pruned")
                
                # Clean unused containers
                result = subprocess.run(
                    ['docker', 'container', 'prune', '-f'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    logger.info("Docker containers pruned")
                
                # Clean unused volumes
                result = subprocess.run(
                    ['docker', 'volume', 'prune', '-f'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    logger.info("Docker volumes pruned")
            else:
                logger.info("[DRY RUN] Would clean Docker resources")
                
        except subprocess.TimeoutExpired:
            logger.warning("Docker cleanup timed out")
        except FileNotFoundError:
            logger.info("Docker not installed, skipping Docker cleanup")
    
    def _organize_log_files(self) -> None:
        """Organize and archive old log files."""
        logs_dir = Path('logs')
        if not logs_dir.exists():
            logger.info("No logs directory found, skipping log organization")
            return
        
        # Create archive directory for old logs
        archive_dir = logs_dir / 'archive'
        if not self.dry_run:
            archive_dir.mkdir(exist_ok=True)
        
        # Move logs older than 30 days to archive
        cutoff_date = datetime.now() - timedelta(days=30)
        archived_count = 0
        
        for log_file in logs_dir.glob('*.log'):
            if log_file.is_file():
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    archive_path = archive_dir / f"{file_time.strftime('%Y%m%d')}_{log_file.name}"
                    if not self.dry_run:
                        log_file.rename(archive_path)
                    archived_count += 1
                    logger.debug(f"Archived log file: {log_file} -> {archive_path}")
        
        logger.info(f"Archived {archived_count} old log files")
    
    def _clean_test_artifacts(self) -> None:
        """Clean test artifacts and reports."""
        test_artifacts = [
            '.coverage',
            'coverage.xml',
            'htmlcov',
            'pytest-results.xml',
            '**/*.pytest_cache',
            'test-results',
            'test-reports',
        ]
        
        removed_count = 0
        for pattern in test_artifacts:
            for path in Path('.').glob(pattern):
                if path.is_dir():
                    if not self.dry_run:
                        shutil.rmtree(path)
                    removed_count += 1
                    logger.debug(f"Removed test directory: {path}")
                elif path.is_file():
                    if not self.dry_run:
                        path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed test file: {path}")
        
        logger.info(f"Cleaned {removed_count} test artifacts")
    
    def _update_doc_timestamps(self) -> None:
        """Update documentation timestamps."""
        docs_dir = Path('docs')
        if not docs_dir.exists():
            logger.info("No docs directory found, skipping timestamp updates")
            return
        
        updated_count = 0
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Update README files with last updated timestamps
        for readme in docs_dir.rglob('README.md'):
            if not self.dry_run:
                content = readme.read_text()
                
                # Look for timestamp patterns and update them
                timestamp_patterns = [
                    r'Last Updated: \d{4}-\d{2}-\d{2}',
                    r'Updated: \d{4}-\d{2}-\d{2}',
                ]
                
                updated = False
                for pattern in timestamp_patterns:
                    import re
                    if re.search(pattern, content):
                        content = re.sub(pattern, f'Last Updated: {current_date}', content)
                        updated = True
                
                if updated:
                    readme.write_text(content)
                    updated_count += 1
                    logger.debug(f"Updated timestamps in: {readme}")
            else:
                updated_count += 1  # Simulate for dry run
        
        logger.info(f"Updated timestamps in {updated_count} documentation files")
    
    def _check_disk_usage(self) -> None:
        """Check disk usage and warn if space is low."""
        try:
            if not self.dry_run:
                result = subprocess.run(
                    ['df', '-h', '.'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        # Parse disk usage line
                        parts = lines[1].split()
                        if len(parts) >= 5:
                            usage_percent = parts[4].rstrip('%')
                            try:
                                usage = int(usage_percent)
                                if usage > 90:
                                    logger.warning(f"Disk usage is high: {usage}%")
                                elif usage > 80:
                                    logger.info(f"Disk usage: {usage}%")
                                else:
                                    logger.info(f"Disk usage is acceptable: {usage}%")
                            except ValueError:
                                logger.info("Could not parse disk usage percentage")
            else:
                logger.info("[DRY RUN] Would check disk usage")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("Could not check disk usage (df command not available)")
    
    def _generate_maintenance_report(self) -> None:
        """Generate maintenance report."""
        report = {
            'maintenance_date': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': [{'task': task, 'error': error} for task, error in self.tasks_failed],
            'summary': {
                'total_tasks': len(self.tasks_completed) + len(self.tasks_failed),
                'successful_tasks': len(self.tasks_completed),
                'failed_tasks': len(self.tasks_failed),
                'success_rate': len(self.tasks_completed) / (len(self.tasks_completed) + len(self.tasks_failed)) * 100
            }
        }
        
        if not self.dry_run:
            # Save to file
            report_file = Path('maintenance-report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Maintenance report saved to {report_file}")
            
            # Also create a markdown summary
            markdown_report = self._create_markdown_report(report)
            with open('maintenance-summary.md', 'w') as f:
                f.write(markdown_report)
        else:
            logger.info("[DRY RUN] Would generate maintenance report")
    
    def _create_markdown_report(self, report: Dict) -> str:
        """Create markdown formatted maintenance report."""
        lines = []
        lines.append("# Maintenance Report")
        lines.append(f"**Date**: {report['maintenance_date']}")
        lines.append(f"**Dry Run**: {report['dry_run']}")
        lines.append("")
        
        lines.append("## Summary")
        summary = report['summary']
        lines.append(f"- Total Tasks: {summary['total_tasks']}")
        lines.append(f"- Successful: {summary['successful_tasks']}")
        lines.append(f"- Failed: {summary['failed_tasks']}")
        lines.append(f"- Success Rate: {summary['success_rate']:.1f}%")
        lines.append("")
        
        if report['tasks_completed']:
            lines.append("## Completed Tasks")
            for task in report['tasks_completed']:
                lines.append(f"- ✅ {task}")
            lines.append("")
        
        if report['tasks_failed']:
            lines.append("## Failed Tasks")
            for task_info in report['tasks_failed']:
                lines.append(f"- ❌ {task_info['task']}: {task_info['error']}")
            lines.append("")
        
        lines.append("## Next Steps")
        if report['tasks_failed']:
            lines.append("- Review failed tasks and resolve underlying issues")
            lines.append("- Re-run maintenance after fixes")
        else:
            lines.append("- All maintenance tasks completed successfully")
            lines.append("- Schedule next maintenance run")
        
        return '\n'.join(lines)
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches glob pattern."""
        from fnmatch import fnmatch
        return fnmatch(path, pattern)
    
    def _print_summary(self) -> None:
        """Print maintenance summary."""
        print("\n" + "="*60)
        print("MAINTENANCE SUMMARY")
        print("="*60)
        print(f"Total Tasks: {len(self.tasks_completed) + len(self.tasks_failed)}")
        print(f"Completed: {len(self.tasks_completed)}")
        print(f"Failed: {len(self.tasks_failed)}")
        
        if self.tasks_failed:
            print("\nFailed Tasks:")
            for task, error in self.tasks_failed:
                print(f"  ❌ {task}: {error}")
        
        if self.tasks_completed:
            print("\nCompleted Tasks:")
            for task in self.tasks_completed:
                print(f"  ✅ {task}")
        
        print("="*60)


def main():
    """Main maintenance function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated maintenance for Materials Orchestrator')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = MaintenanceRunner(dry_run=args.dry_run)
    success = runner.run_all_tasks()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()