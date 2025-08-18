#!/usr/bin/env python3
"""Automated metrics collection for materials orchestrator."""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects various metrics about the materials orchestrator system."""

    def __init__(self, project_root: Path):
        """Initialize metrics collector.
        
        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "collection_version": "1.0.0"
        }

    def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code-related metrics."""
        logger.info("Collecting code metrics...")
        
        metrics = {}
        src_dir = self.project_root / "src"
        
        if src_dir.exists():
            # Count files and lines
            python_files = list(src_dir.rglob("*.py"))
            total_lines = 0
            total_functions = 0
            total_classes = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        
                        # Count functions and classes
                        content = ''.join(lines)
                        total_functions += content.count('def ')
                        total_classes += content.count('class ')
                        
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
            
            metrics.update({
                "total_python_files": len(python_files),
                "total_lines_of_code": total_lines,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "avg_lines_per_file": total_lines / len(python_files) if python_files else 0,
            })
        
        return metrics

    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect test-related metrics."""
        logger.info("Collecting test metrics...")
        
        metrics = {}
        tests_dir = self.project_root / "tests"
        
        if tests_dir.exists():
            test_files = list(tests_dir.rglob("test_*.py"))
            metrics["total_test_files"] = len(test_files)
            
            # Try to run pytest and collect coverage
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", 
                    "--collect-only", "-q", str(tests_dir)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse test collection output
                    output_lines = result.stdout.split('\n')
                    test_count = 0
                    for line in output_lines:
                        if 'collected' in line and 'item' in line:
                            # Extract number from "collected X items"
                            words = line.split()
                            for i, word in enumerate(words):
                                if word == 'collected' and i + 1 < len(words):
                                    try:
                                        test_count = int(words[i + 1])
                                        break
                                    except ValueError:
                                        pass
                    
                    metrics["total_test_cases"] = test_count
                else:
                    metrics["test_collection_error"] = result.stderr[:500]
                    
            except Exception as e:
                metrics["test_metrics_error"] = str(e)
        
        return metrics

    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        logger.info("Collecting dependency metrics...")
        
        metrics = {}
        
        # Check pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)
                
                dependencies = data.get("project", {}).get("dependencies", [])
                optional_deps = data.get("project", {}).get("optional-dependencies", {})
                
                metrics.update({
                    "core_dependencies": len(dependencies),
                    "optional_dependency_groups": len(optional_deps),
                    "total_optional_dependencies": sum(len(deps) for deps in optional_deps.values()),
                })
                
            except ImportError:
                # Fallback: count lines in pyproject.toml
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                    metrics["pyproject_lines"] = len(content.split('\n'))
                    metrics["has_pyproject"] = True
        
        # Check requirements files
        req_files = ["requirements.txt", "requirements-dev.txt"]
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                with open(req_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    metrics[f"{req_file.replace('.txt', '_count')}"] = len(lines)
        
        return metrics

    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        logger.info("Collecting documentation metrics...")
        
        metrics = {}
        
        # Count markdown files
        md_files = list(self.project_root.rglob("*.md"))
        metrics["total_markdown_files"] = len(md_files)
        
        # Check for key documentation files
        key_docs = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"]
        for doc in key_docs:
            doc_path = self.project_root / doc
            metrics[f"has_{doc.lower().replace('.', '_')}"] = doc_path.exists()
            if doc_path.exists():
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    metrics[f"{doc.lower().replace('.', '_')}_lines"] = len(f.readlines())
        
        # Check docs directory
        docs_dir = self.project_root / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.md"))
            metrics["docs_directory_files"] = len(doc_files)
        
        return metrics

    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        logger.info("Collecting Git metrics...")
        
        metrics = {}
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"], 
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.returncode == 0:
                metrics["is_git_repo"] = True
                
                # Get commit count
                result = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD"], 
                    capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    metrics["total_commits"] = int(result.stdout.strip())
                
                # Get branch info
                result = subprocess.run(
                    ["git", "branch", "--show-current"], 
                    capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    metrics["current_branch"] = result.stdout.strip()
                
                # Get last commit info
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%H %s"], 
                    capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    commit_info = result.stdout.strip()
                    if commit_info:
                        parts = commit_info.split(' ', 1)
                        metrics["last_commit_hash"] = parts[0][:8]
                        if len(parts) > 1:
                            metrics["last_commit_message"] = parts[1][:100]
                
            else:
                metrics["is_git_repo"] = False
                
        except Exception as e:
            metrics["git_metrics_error"] = str(e)
        
        return metrics

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-related metrics."""
        logger.info("Collecting system metrics...")
        
        metrics = {}
        
        # Python version
        metrics["python_version"] = sys.version.split()[0]
        
        # System info
        try:
            import platform
            metrics.update({
                "platform_system": platform.system(),
                "platform_version": platform.version(),
                "platform_machine": platform.machine(),
                "platform_processor": platform.processor(),
            })
        except Exception:
            pass
        
        # Directory sizes
        try:
            def get_dir_size(path: Path) -> int:
                total = 0
                for file in path.rglob('*'):
                    if file.is_file():
                        try:
                            total += file.stat().st_size
                        except Exception:
                            pass
                return total
            
            src_dir = self.project_root / "src"
            if src_dir.exists():
                metrics["src_directory_size_bytes"] = get_dir_size(src_dir)
            
            tests_dir = self.project_root / "tests"
            if tests_dir.exists():
                metrics["tests_directory_size_bytes"] = get_dir_size(tests_dir)
                
        except Exception as e:
            metrics["directory_size_error"] = str(e)
        
        return metrics

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        logger.info("Starting comprehensive metrics collection...")
        
        all_metrics = {
            "metadata": self.metrics,
            "code": self.collect_code_metrics(),
            "tests": self.collect_test_metrics(),
            "dependencies": self.collect_dependency_metrics(),
            "documentation": self.collect_documentation_metrics(),
            "git": self.collect_git_metrics(),
            "system": self.collect_system_metrics(),
        }
        
        return all_metrics

    def save_metrics(self, metrics: Dict[str, Any], output_file: Path) -> None:
        """Save metrics to file.
        
        Args:
            metrics: Metrics data
            output_file: Output file path
        """
        logger.info(f"Saving metrics to {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print metrics summary.
        
        Args:
            metrics: Metrics data
        """
        print("\n📊 Metrics Collection Summary")
        print("=" * 40)
        
        code = metrics.get("code", {})
        tests = metrics.get("tests", {})
        deps = metrics.get("dependencies", {})
        docs = metrics.get("documentation", {})
        git = metrics.get("git", {})
        system = metrics.get("system", {})
        
        print(f"📝 Code Files: {code.get('total_python_files', 0)}")
        print(f"📏 Lines of Code: {code.get('total_lines_of_code', 0):,}")
        print(f"🔧 Functions: {code.get('total_functions', 0)}")
        print(f"🏗️  Classes: {code.get('total_classes', 0)}")
        print(f"🧪 Test Files: {tests.get('total_test_files', 0)}")
        print(f"📋 Test Cases: {tests.get('total_test_cases', 'N/A')}")
        print(f"📦 Core Dependencies: {deps.get('core_dependencies', 0)}")
        print(f"📚 Documentation Files: {docs.get('total_markdown_files', 0)}")
        print(f"🌿 Git Commits: {git.get('total_commits', 'N/A')}")
        print(f"🐍 Python Version: {system.get('python_version', 'Unknown')}")
        print(f"💻 Platform: {system.get('platform_system', 'Unknown')}")


def main():
    """Main metrics collection function."""
    print("📊 Materials Orchestrator Metrics Collector")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize collector
    collector = MetricsCollector(project_root)
    
    # Collect all metrics
    start_time = time.time()
    metrics = collector.collect_all_metrics()
    collection_time = time.time() - start_time
    
    # Add collection performance info
    metrics["metadata"]["collection_duration_seconds"] = round(collection_time, 2)
    
    # Save metrics
    output_file = project_root / "metrics_report.json"
    collector.save_metrics(metrics, output_file)
    
    # Print summary
    collector.print_summary(metrics)
    
    print(f"\n⏱️  Collection completed in {collection_time:.2f} seconds")
    print(f"💾 Full report saved to: {output_file}")


if __name__ == "__main__":
    main()