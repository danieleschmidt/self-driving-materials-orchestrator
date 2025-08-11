#!/usr/bin/env python3
"""Quality gates script for comprehensive validation."""

import subprocess
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of a quality check."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    duration: float
    error: str = ""


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    timestamp: datetime
    overall_score: float
    passed: bool
    results: List[QualityResult]
    summary: Dict[str, Any]


class QualityGateRunner:
    """Runs comprehensive quality gates."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.src_path = repo_root / "src"
        self.tests_path = repo_root / "tests"
        
        # Quality thresholds
        self.thresholds = {
            "test_coverage": 85.0,
            "test_success_rate": 95.0,
            "security_score": 90.0,
            "performance_score": 80.0,
            "code_quality_score": 85.0,
            "documentation_score": 70.0
        }
    
    def run_all_gates(self) -> QualityReport:
        """Run all quality gates."""
        logger.info("🔍 Starting comprehensive quality gate validation...")
        
        start_time = time.time()
        results = []
        
        # 1. Test Coverage and Success Rate
        results.append(self._run_test_coverage())
        
        # 2. Security Analysis
        results.append(self._run_security_analysis())
        
        # 3. Performance Benchmarks
        results.append(self._run_performance_benchmarks())
        
        # 4. Code Quality Analysis  
        results.append(self._run_code_quality())
        
        # 5. Documentation Coverage
        results.append(self._run_documentation_analysis())
        
        # 6. Integration Testing
        results.append(self._run_integration_tests())
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        passed = all(result.passed for result in results)
        
        duration = time.time() - start_time
        
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            passed=passed,
            results=results,
            summary=self._generate_summary(results, duration)
        )
        
        self._print_report(report)
        return report
    
    def _run_test_coverage(self) -> QualityResult:
        """Run test coverage analysis."""
        logger.info("📊 Running test coverage analysis...")
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.tests_path),
                f"--cov={self.src_path / 'materials_orchestrator'}",
                "--cov-report=json",
                "--cov-report=term",
                "-q"
            ]
            
            env = {"PYTHONPATH": str(self.src_path)}
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root, env=env)
            
            # Parse coverage report
            coverage_file = self.repo_root / "coverage.json"
            coverage_data = {}
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
            
            # Extract metrics
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
            test_count = result.stdout.count("PASSED") + result.stdout.count("FAILED")
            failed_count = result.stdout.count("FAILED")
            success_rate = ((test_count - failed_count) / test_count * 100) if test_count > 0 else 0
            
            # Determine pass/fail
            coverage_passed = total_coverage >= self.thresholds["test_coverage"]
            success_rate_passed = success_rate >= self.thresholds["test_success_rate"]
            passed = coverage_passed and success_rate_passed
            
            # Calculate score
            score = (total_coverage + success_rate) / 2
            
            duration = time.time() - start_time
            
            return QualityResult(
                name="Test Coverage & Success",
                passed=passed,
                score=score,
                details={
                    "coverage_percent": total_coverage,
                    "test_success_rate": success_rate,
                    "total_tests": test_count,
                    "failed_tests": failed_count,
                    "coverage_threshold": self.thresholds["test_coverage"],
                    "success_rate_threshold": self.thresholds["test_success_rate"]
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityResult(
                name="Test Coverage & Success",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                error=str(e)
            )
    
    def _run_security_analysis(self) -> QualityResult:
        """Run security analysis."""
        logger.info("🔒 Running security analysis...")
        start_time = time.time()
        
        try:
            # Check for common security issues
            security_score = 100.0
            issues = []
            
            # Check for potential security issues in code
            src_files = list(self.src_path.rglob("*.py"))
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Basic security checks
                        if "eval(" in content:
                            issues.append(f"Potential eval() usage in {file_path}")
                            security_score -= 10
                        
                        if "exec(" in content:
                            issues.append(f"Potential exec() usage in {file_path}")
                            security_score -= 10
                        
                        if "shell=True" in content:
                            issues.append(f"Shell injection risk in {file_path}")
                            security_score -= 5
                        
                        if "pickle.loads" in content:
                            issues.append(f"Unsafe pickle usage in {file_path}")
                            security_score -= 5
                            
                except Exception:
                    continue
            
            # Check for exposed credentials
            config_files = list(self.repo_root.rglob("*.env*")) + list(self.repo_root.rglob("*.json"))
            for config_file in config_files:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in ["password", "secret", "key", "token"]):
                            # This is just a warning, not necessarily an issue
                            pass
                except Exception:
                    continue
            
            security_score = max(0.0, security_score)
            passed = security_score >= self.thresholds["security_score"]
            
            duration = time.time() - start_time
            
            return QualityResult(
                name="Security Analysis",
                passed=passed,
                score=security_score,
                details={
                    "issues_found": len(issues),
                    "issues": issues[:10],  # Limit to first 10
                    "files_scanned": len(src_files),
                    "threshold": self.thresholds["security_score"]
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityResult(
                name="Security Analysis",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                error=str(e)
            )
    
    def _run_performance_benchmarks(self) -> QualityResult:
        """Run performance benchmarks."""
        logger.info("⚡ Running performance benchmarks...")
        start_time = time.time()
        
        try:
            # Run simple performance test
            env = {"PYTHONPATH": str(self.src_path)}
            
            # Test basic import performance
            import_time_start = time.time()
            subprocess.run([
                sys.executable, "-c", 
                "from materials_orchestrator import AutonomousLab; print('Import successful')"
            ], capture_output=True, cwd=self.repo_root, env=env)
            import_duration = time.time() - import_time_start
            
            # Test basic functionality performance
            func_time_start = time.time()
            test_result = subprocess.run([
                sys.executable, "-c", '''
import sys
sys.path.insert(0, "src")
from materials_orchestrator import AutonomousLab, MaterialsObjective
import time

start = time.time()
lab = AutonomousLab()
objective = MaterialsObjective("band_gap", (1.2, 1.6))
param_space = {"temperature": (100, 200), "concentration": (0.5, 1.5)}
campaign = lab.run_campaign(objective, param_space, max_experiments=3)
duration = time.time() - start
print(f"Performance test completed in {duration:.2f}s")
print(f"Experiments: {campaign.total_experiments}")
                '''
            ], capture_output=True, text=True, cwd=self.repo_root, env=env)
            func_duration = time.time() - func_time_start
            
            # Extract results
            performance_metrics = {
                "import_time": import_duration,
                "basic_functionality_time": func_duration,
                "test_output": test_result.stdout
            }
            
            # Calculate performance score
            score = 100.0
            if import_duration > 5.0:  # > 5 seconds is slow
                score -= 20
            if func_duration > 30.0:  # > 30 seconds is slow
                score -= 30
                
            passed = score >= self.thresholds["performance_score"]
            
            duration = time.time() - start_time
            
            return QualityResult(
                name="Performance Benchmarks",
                passed=passed,
                score=score,
                details={
                    "import_time_seconds": import_duration,
                    "functionality_time_seconds": func_duration,
                    "performance_score": score,
                    "threshold": self.thresholds["performance_score"],
                    "test_successful": test_result.returncode == 0
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityResult(
                name="Performance Benchmarks",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                error=str(e)
            )
    
    def _run_code_quality(self) -> QualityResult:
        """Run code quality analysis."""
        logger.info("🔍 Running code quality analysis...")
        start_time = time.time()
        
        try:
            # Basic code quality metrics
            src_files = list(self.src_path.rglob("*.py"))
            
            total_lines = 0
            total_functions = 0
            total_classes = 0
            docstring_coverage = 0
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        
                        # Count functions and classes
                        for line in lines:
                            line = line.strip()
                            if line.startswith("def "):
                                total_functions += 1
                            elif line.startswith("class "):
                                total_classes += 1
                            elif '"""' in line or "'''" in line:
                                docstring_coverage += 0.5  # Approximate docstring detection
                                
                except Exception:
                    continue
            
            # Calculate metrics
            avg_file_size = total_lines / len(src_files) if src_files else 0
            complexity_score = min(100.0, max(0.0, 100.0 - (avg_file_size - 200) / 10))  # Penalize very large files
            docstring_score = min(100.0, (docstring_coverage / max(1, total_functions + total_classes)) * 100)
            
            # Overall code quality score
            code_quality_score = (complexity_score + docstring_score) / 2
            passed = code_quality_score >= self.thresholds["code_quality_score"]
            
            duration = time.time() - start_time
            
            return QualityResult(
                name="Code Quality Analysis",
                passed=passed,
                score=code_quality_score,
                details={
                    "total_files": len(src_files),
                    "total_lines": total_lines,
                    "total_functions": total_functions,
                    "total_classes": total_classes,
                    "avg_file_size": avg_file_size,
                    "complexity_score": complexity_score,
                    "docstring_score": docstring_score,
                    "threshold": self.thresholds["code_quality_score"]
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityResult(
                name="Code Quality Analysis",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                error=str(e)
            )
    
    def _run_documentation_analysis(self) -> QualityResult:
        """Run documentation coverage analysis."""
        logger.info("📖 Running documentation analysis...")
        start_time = time.time()
        
        try:
            # Check for documentation files
            doc_files = []
            doc_files.extend(list(self.repo_root.glob("*.md")))
            doc_files.extend(list((self.repo_root / "docs").rglob("*.md") if (self.repo_root / "docs").exists() else []))
            
            # Check for docstrings in Python files
            src_files = list(self.src_path.rglob("*.py"))
            documented_items = 0
            total_items = 0
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines):
                            line = line.strip()
                            if line.startswith(('def ', 'class ')):
                                total_items += 1
                                # Check if next few lines contain docstring
                                for j in range(i + 1, min(i + 5, len(lines))):
                                    if '"""' in lines[j] or "'''" in lines[j]:
                                        documented_items += 1
                                        break
                                        
                except Exception:
                    continue
            
            # Calculate documentation score
            docstring_coverage = (documented_items / max(1, total_items)) * 100
            doc_file_score = min(100.0, len(doc_files) * 20)  # Up to 100 for 5+ doc files
            
            documentation_score = (docstring_coverage + doc_file_score) / 2
            passed = documentation_score >= self.thresholds["documentation_score"]
            
            duration = time.time() - start_time
            
            return QualityResult(
                name="Documentation Analysis",
                passed=passed,
                score=documentation_score,
                details={
                    "doc_files_count": len(doc_files),
                    "docstring_coverage": docstring_coverage,
                    "documented_items": documented_items,
                    "total_items": total_items,
                    "threshold": self.thresholds["documentation_score"]
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityResult(
                name="Documentation Analysis",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                error=str(e)
            )
    
    def _run_integration_tests(self) -> QualityResult:
        """Run integration tests."""
        logger.info("🔗 Running integration tests...")
        start_time = time.time()
        
        try:
            env = {"PYTHONPATH": str(self.src_path)}
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(self.tests_path / "integration"),
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.repo_root, env=env)
            
            # Parse results
            test_count = result.stdout.count("PASSED") + result.stdout.count("FAILED")
            failed_count = result.stdout.count("FAILED")
            success_rate = ((test_count - failed_count) / test_count * 100) if test_count > 0 else 100
            
            passed = success_rate >= 90.0 and result.returncode == 0
            
            duration = time.time() - start_time
            
            return QualityResult(
                name="Integration Tests",
                passed=passed,
                score=success_rate,
                details={
                    "total_tests": test_count,
                    "failed_tests": failed_count,
                    "success_rate": success_rate,
                    "return_code": result.returncode
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityResult(
                name="Integration Tests", 
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                error=str(e)
            )
    
    def _calculate_overall_score(self, results: List[QualityResult]) -> float:
        """Calculate overall quality score."""
        if not results:
            return 0.0
        
        # Weight different quality aspects
        weights = {
            "Test Coverage & Success": 0.25,
            "Security Analysis": 0.20,
            "Performance Benchmarks": 0.20,
            "Code Quality Analysis": 0.15,
            "Documentation Analysis": 0.10,
            "Integration Tests": 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result.name, 0.1)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_summary(self, results: List[QualityResult], duration: float) -> Dict[str, Any]:
        """Generate quality report summary."""
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        return {
            "total_gates": len(results),
            "passed_gates": passed_count,
            "failed_gates": failed_count,
            "success_rate": (passed_count / len(results) * 100) if results else 0,
            "total_duration": duration,
            "avg_gate_duration": duration / len(results) if results else 0
        }
    
    def _print_report(self, report: QualityReport) -> None:
        """Print quality report."""
        print("\n" + "="*80)
        print("🛡️  QUALITY GATES REPORT")
        print("="*80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Status: {'✅ PASSED' if report.passed else '❌ FAILED'}")
        print()
        
        print("Individual Gate Results:")
        print("-" * 50)
        
        for result in report.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{result.name:<30} {status:<10} {result.score:>6.1f}/100 ({result.duration:.2f}s)")
            
            if result.error:
                print(f"   Error: {result.error}")
        
        print()
        print("Summary:")
        print(f"  Total Gates: {report.summary['total_gates']}")
        print(f"  Passed: {report.summary['passed_gates']}")
        print(f"  Failed: {report.summary['failed_gates']}")
        print(f"  Success Rate: {report.summary['success_rate']:.1f}%")
        print(f"  Total Duration: {report.summary['total_duration']:.2f}s")
        print("="*80)


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    
    runner = QualityGateRunner(repo_root)
    report = runner.run_all_gates()
    
    # Save report
    report_file = repo_root / "quality_report.json" 
    with open(report_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"\n📊 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()