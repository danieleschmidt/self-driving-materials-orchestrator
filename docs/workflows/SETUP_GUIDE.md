# Workflow Setup Guide

## Overview

This guide provides step-by-step instructions for setting up GitHub Actions workflows for the Materials Orchestrator project. Due to GitHub App permission limitations, workflows must be manually created by repository maintainers.

## Quick Setup

### 1. Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Templates

Copy the workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Core workflows (recommended)
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml

# Optional workflows
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

### 3. Configure Repository Secrets

Set up the following secrets in your repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets
- `GITHUB_TOKEN` (automatically provided)

#### Docker Registry Secrets
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or access token

#### Deployment Secrets
- `STAGING_KUBECONFIG` - Base64 encoded kubeconfig for staging cluster
- `PRODUCTION_KUBECONFIG` - Base64 encoded kubeconfig for production cluster

#### Notification Secrets
- `SLACK_WEBHOOK` - Slack webhook URL for notifications
- `PAGERDUTY_INTEGRATION_KEY` - PagerDuty integration key

#### Security Scanning Secrets
- `SNYK_TOKEN` - Snyk API token for vulnerability scanning
- `DEPENDENCY_UPDATE_TOKEN` - GitHub personal access token for dependency updates

### 4. Configure Branch Protection

Set up branch protection rules for the `main` branch:

1. Go to `Settings > Branches`
2. Add rule for `main` branch
3. Enable:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Include administrators

Required status checks:
- `Lint and Code Quality`
- `Unit Tests`
- `Integration Tests`
- `Security Scan`
- `Docker Build Test`

## Detailed Setup Instructions

### CI Workflow (ci.yml)

The Continuous Integration workflow runs on every push and pull request.

#### Features
- **Multi-Python Version Testing**: Tests against Python 3.9, 3.10, 3.11, 3.12
- **Comprehensive Testing**: Unit, integration, and E2E tests
- **Code Quality**: Linting, formatting, and type checking
- **Security Scanning**: Bandit, Safety, and pip-audit
- **Documentation**: MkDocs build and link checking
- **Docker Testing**: Container build verification

#### Configuration

1. **Python Versions**: Modify the matrix in the `test-unit` job to test against your supported Python versions
2. **Test Services**: The integration tests use MongoDB and Redis services
3. **Coverage**: Results are uploaded to Codecov (requires `CODECOV_TOKEN` if repository is private)

#### Customization

```yaml
# Modify Python versions
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]

# Adjust test timeouts
timeout-minutes: 30

# Modify test markers
pytest tests/e2e/ -m "not slow"
```

### CD Workflow (cd.yml)

The Continuous Deployment workflow handles deployments to staging and production.

#### Features
- **Multi-Stage Deployment**: Staging → Production with approval gates
- **Security Scanning**: Container vulnerability scanning with Trivy
- **SBOM Generation**: Software Bill of Materials for supply chain security
- **Rollback Capability**: Automatic rollback on deployment failure
- **Smoke Testing**: Post-deployment verification

#### Environment Setup

1. **Create Environments**: Set up `staging` and `production` environments in repository settings
2. **Protection Rules**: Configure protection rules for production environment
3. **Reviewers**: Add required reviewers for production deployments

#### Kubernetes Configuration

The workflow assumes Kubernetes deployment. Create the following files:

```
k8s/
├── staging/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
└── production/
    ├── deployment.yaml
    ├── service.yaml
    └── ingress.yaml
```

Example deployment template:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: materials-orchestrator
  namespace: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: materials-orchestrator
  template:
    metadata:
      labels:
        app: materials-orchestrator
    spec:
      containers:
      - name: materials-orchestrator
        image: IMAGE_TAG  # This will be replaced by the workflow
        ports:
        - containerPort: 8000
        - containerPort: 8501
        env:
        - name: ENVIRONMENT
          value: "staging"
        - name: MONGODB_URL
          valueFrom:
            secretKeyRef:
              name: mongodb-credentials
              key: url
```

### Security Scan Workflow (security-scan.yml)

Comprehensive security scanning that runs daily and on code changes.

#### Features
- **SAST**: Static application security testing with Bandit and Semgrep
- **Dependency Scanning**: Vulnerability scanning with Safety, pip-audit, and Snyk
- **Container Scanning**: Trivy container vulnerability scanning
- **Infrastructure Scanning**: Checkov for infrastructure as code
- **Secrets Detection**: GitLeaks and TruffleHog for secret scanning
- **License Compliance**: License compatibility checking
- **SLSA Provenance**: Supply chain security attestation

#### Setup Requirements

1. **Snyk Token**: Sign up for Snyk and add `SNYK_TOKEN` secret
2. **Security Tab**: Ensure GitHub Security tab is enabled for SARIF uploads
3. **Notification Channel**: Configure Slack webhook for security notifications

### Dependency Update Workflow (dependency-update.yml)

Automated dependency management with security prioritization.

#### Features
- **Security Priority**: Immediate updates for vulnerable dependencies
- **Graduated Updates**: Patch, minor, and major version handling
- **Automated PRs**: Creates pull requests with detailed change information
- **Testing Integration**: Runs tests before creating update PRs
- **Lockfile Maintenance**: Regular lockfile updates for consistency

#### Setup Requirements

1. **Personal Access Token**: Create a fine-grained personal access token with:
   - Repository permissions: Contents (write), Pull requests (write), Issues (write)
   - Add as `DEPENDENCY_UPDATE_TOKEN` secret

2. **Review Teams**: Configure review teams for different update types:
   - Security updates: `security-team`
   - Regular updates: `maintainers`
   - Major updates: Manual review required

## Repository Configuration

### Branch Protection Rules

Configure the following branch protection rules:

#### Main Branch
```yaml
Protection Rules:
  - Require pull request reviews: 2 reviewers
  - Dismiss stale reviews: true
  - Require review from code owners: true
  - Require status checks: true
  - Require branches up to date: true
  - Include administrators: true
  - Allow force pushes: false
  - Allow deletions: false

Required Status Checks:
  - "Lint and Code Quality"
  - "Unit Tests (3.11)"  # At minimum
  - "Integration Tests"
  - "Security Scan / SAST"
  - "Docker Build Test"
```

#### Development Branch (if used)
```yaml
Protection Rules:
  - Require pull request reviews: 1 reviewer
  - Require status checks: true
  - Include administrators: false
```

### Environments

#### Staging Environment
```yaml
Environment: staging
Protection Rules:
  - Required reviewers: 0
  - Deployment timeout: 10 minutes
  - Environment secrets:
    - STAGING_KUBECONFIG
    - STAGING_DATABASE_URL
```

#### Production Environment
```yaml
Environment: production
Protection Rules:
  - Required reviewers: 2 (from maintainers team)
  - Deployment timeout: 30 minutes
  - Environment secrets:
    - PRODUCTION_KUBECONFIG
    - PRODUCTION_DATABASE_URL
    - PRODUCTION_SLACK_WEBHOOK
```

### Repository Settings

#### General Settings
- Allow merge commits: false
- Allow squash merging: true
- Allow rebase merging: true
- Automatically delete head branches: true
- Always suggest updating pull request branches: true

#### Security Settings
- Enable vulnerability alerts: true
- Enable security updates: true
- Enable secret scanning: true
- Enable push protection: true

## Monitoring and Maintenance

### Workflow Health Monitoring

Monitor workflow health with the following metrics:
- Success rate of CI runs
- Average build time
- Deployment frequency
- Mean time to recovery (MTTR)

### Regular Maintenance Tasks

#### Monthly
- Review workflow run history for failures
- Update workflow templates if needed
- Review and update secrets rotation schedule
- Check dependency update PR backlog

#### Quarterly
- Review branch protection rules
- Update required status checks
- Review environment protection rules
- Update notification channels

#### Annually
- Review and update security scanning tools
- Update supported Python versions
- Review deployment strategy
- Update documentation

## Troubleshooting

### Common Issues

#### CI Failures
```bash
# Check logs
gh run list --workflow=ci.yml --limit=5
gh run view <run-id> --log-failed

# Common fixes
- Update Python version in matrix
- Check service health in integration tests
- Verify test database connectivity
```

#### Deployment Failures
```bash
# Check deployment status
kubectl get pods -n staging
kubectl describe deployment materials-orchestrator -n staging

# Common fixes
- Verify kubeconfig secrets
- Check image availability
- Validate Kubernetes manifests
```

#### Security Scan Issues
```bash
# Check SARIF upload issues
- Verify security tab is enabled
- Check SARIF file format
- Validate scanner versions
```

### Getting Help

1. **GitHub Actions Documentation**: https://docs.github.com/en/actions
2. **Workflow Syntax**: https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions
3. **Security Scanning**: https://docs.github.com/en/code-security
4. **Community Support**: GitHub Actions community forum

## Best Practices

### Workflow Design
1. **Fail Fast**: Run quick checks (linting) before expensive tests
2. **Parallel Execution**: Use job dependencies efficiently
3. **Caching**: Cache dependencies to reduce build times
4. **Conditional Execution**: Use conditions to skip unnecessary runs

### Security
1. **Least Privilege**: Use minimal required permissions
2. **Secret Management**: Rotate secrets regularly
3. **SARIF Integration**: Upload security results to GitHub Security tab
4. **Supply Chain**: Generate and verify SBOMs

### Maintenance
1. **Version Pinning**: Pin action versions for reproducibility
2. **Regular Updates**: Keep actions and dependencies updated
3. **Documentation**: Keep workflow documentation current
4. **Monitoring**: Monitor workflow performance and reliability

## Migration from Existing CI/CD

If you're migrating from another CI/CD system:

1. **Audit Current Pipelines**: Document existing build and deployment processes
2. **Map Workflows**: Map existing stages to GitHub Actions jobs
3. **Migrate Secrets**: Transfer secrets and environment variables
4. **Test Thoroughly**: Run workflows in parallel during migration period
5. **Update Documentation**: Update deployment and development documentation

## Custom Modifications

### Adding New Test Types
```yaml
# Add performance tests
performance-test:
  name: Performance Tests
  runs-on: ubuntu-latest
  steps:
    - name: Run performance tests
      run: pytest tests/performance/ --benchmark-only
```

### Custom Deployment Targets
```yaml
# Add staging-2 environment
deploy-staging-2:
  name: Deploy to Staging 2
  runs-on: ubuntu-latest
  environment: staging-2
  # ... deployment steps
```

### Additional Security Scans
```yaml
# Add CodeQL analysis
codeql:
  name: CodeQL Analysis
  runs-on: ubuntu-latest
  steps:
    - uses: github/codeql-action/init@v2
      with:
        languages: python
    - uses: github/codeql-action/analyze@v2
```