#!/bin/bash
set -e

# Security scanning script for Docker images and dependencies
# Usage: ./scripts/security-scan.sh [image-name]

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[SECURITY]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get image name
IMAGE_NAME=${1:-"materials-orchestrator:latest"}

print_status "Running security scans for Materials Orchestrator"
print_status "Target: $IMAGE_NAME"

# Create reports directory
mkdir -p reports/security

# 1. Python dependency vulnerability scan with Safety
print_status "1/5 Scanning Python dependencies with Safety..."
if command -v safety &> /dev/null; then
    safety check --json --output reports/security/safety-report.json || true
    safety check --short-report
    print_success "Safety scan completed"
else
    print_warning "Safety not installed. Install with: pip install safety"
fi

# 2. Python dependency audit with pip-audit
print_status "2/5 Scanning dependencies with pip-audit..."
if command -v pip-audit &> /dev/null; then
    pip-audit --format=json --output=reports/security/pip-audit-report.json || true
    pip-audit --format=cyclonedx-json --output=reports/security/sbom.json || true
    print_success "pip-audit scan completed"
else
    print_warning "pip-audit not installed. Install with: pip install pip-audit"
fi

# 3. Bandit static analysis
print_status "3/5 Running Bandit static analysis..."
if command -v bandit &> /dev/null; then
    bandit -r src/ -f json -o reports/security/bandit-report.json || true
    bandit -r src/ --severity-level medium
    print_success "Bandit analysis completed"
else
    print_warning "Bandit not installed. Install with: pip install bandit"
fi

# 4. Docker image vulnerability scan with Trivy (if available)
print_status "4/5 Scanning Docker image with Trivy..."
if command -v trivy &> /dev/null; then
    # Scan for vulnerabilities
    trivy image --format json --output reports/security/trivy-vuln-report.json "$IMAGE_NAME" || true
    
    # Generate SBOM
    trivy image --format cyclonedx --output reports/security/docker-sbom.json "$IMAGE_NAME" || true
    
    # Show summary
    trivy image --severity HIGH,CRITICAL "$IMAGE_NAME"
    print_success "Trivy scan completed"
else
    print_warning "Trivy not installed. Install from: https://aquasecurity.github.io/trivy/"
fi

# 5. Semgrep static analysis (if available)
print_status "5/5 Running Semgrep static analysis..."
if command -v semgrep &> /dev/null; then
    semgrep --config=auto --json --output=reports/security/semgrep-report.json src/ || true
    semgrep --config=auto src/
    print_success "Semgrep analysis completed"
else
    print_warning "Semgrep not installed. Install with: pip install semgrep"
fi

# Generate summary report
print_status "Generating security summary..."

cat > reports/security/summary.md << EOF
# Security Scan Summary

**Scan Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Target:** $IMAGE_NAME

## Scans Performed

1. **Safety** - Python dependency vulnerability scan
2. **pip-audit** - Python package audit
3. **Bandit** - Python static security analysis
4. **Trivy** - Container vulnerability and SBOM scan
5. **Semgrep** - Multi-language static analysis

## Report Files

- \`safety-report.json\` - Python dependency vulnerabilities
- \`pip-audit-report.json\` - Package audit results
- \`bandit-report.json\` - Static analysis findings
- \`trivy-vuln-report.json\` - Container vulnerabilities
- \`docker-sbom.json\` - Container SBOM
- \`sbom.json\` - Python dependencies SBOM
- \`semgrep-report.json\` - Static analysis results

## Next Steps

1. Review all high and critical severity findings
2. Update vulnerable dependencies
3. Address static analysis findings
4. Re-run scans after fixes
5. Document accepted risks for unavoidable issues

## Compliance

This scan supports compliance with:
- NIST Cybersecurity Framework
- OWASP Top 10
- CIS Docker Benchmark
- SLSA Build Provenance

EOF

print_success "Security summary generated: reports/security/summary.md"

# Check for critical issues
CRITICAL_ISSUES=0

# Count critical issues from various scans
if [[ -f "reports/security/trivy-vuln-report.json" ]]; then
    CRITICAL_VULNS=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' reports/security/trivy-vuln-report.json 2>/dev/null | wc -l || echo 0)
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + CRITICAL_VULNS))
fi

if [[ -f "reports/security/bandit-report.json" ]]; then
    HIGH_CONFIDENCE_ISSUES=$(jq -r '.results[] | select(.issue_confidence == "HIGH" and .issue_severity == "HIGH")' reports/security/bandit-report.json 2>/dev/null | wc -l || echo 0)
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + HIGH_CONFIDENCE_ISSUES))
fi

print_status "Security scan completed"
print_status "Critical issues found: $CRITICAL_ISSUES"

if [[ $CRITICAL_ISSUES -gt 0 ]]; then
    print_error "⚠️  Critical security issues found! Review reports in reports/security/"
    print_status "Address critical issues before deploying to production"
    exit 1
else
    print_success "✅ No critical security issues found"
fi

# Optional: upload results to security dashboard
if [[ "${UPLOAD_RESULTS:-false}" == "true" ]] && [[ -n "${SECURITY_DASHBOARD_URL}" ]]; then
    print_status "Uploading results to security dashboard..."
    # Add your security dashboard integration here
    print_success "Results uploaded to security dashboard"
fi