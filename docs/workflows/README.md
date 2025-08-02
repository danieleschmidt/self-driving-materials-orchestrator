# CI/CD Workflows Documentation

This directory contains comprehensive CI/CD workflow templates and documentation for the Self-Driving Materials Orchestrator project.

## 🚨 Important Notice

**Due to GitHub App permission limitations, workflow files cannot be created automatically. Repository maintainers must manually create the workflow files from the templates provided in this directory.**

## Quick Setup

### 1. Copy Workflow Templates
```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy templates from docs/workflows/examples/
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Configure Secrets
```bash
# Required secrets for CI/CD workflows
gh secret set STAGING_KUBECONFIG --body "$(base64 -w 0 ~/.kube/staging-config)"
gh secret set PRODUCTION_KUBECONFIG --body "$(base64 -w 0 ~/.kube/prod-config)"
gh secret set API_SECRET_KEY --body "$(openssl rand -hex 32)"
gh secret set GRAFANA_API_KEY --body "your-grafana-api-key"
gh secret set SLACK_WEBHOOK_URL --body "your-slack-webhook-url"
```

### 3. Enable Workflows
```bash
# Enable GitHub Actions in repository settings
gh repo edit --enable-actions

# Enable security features
gh repo edit --enable-vulnerability-alerts
gh repo edit --enable-automated-security-fixes
```

## Available Workflows

### 1. Continuous Integration (`ci.yml`)
**Triggers**: Push to main/develop, Pull Requests
**Purpose**: Comprehensive testing and quality assurance

#### Jobs:
- **Code Quality**: Linting, formatting, type checking
- **Test Suite**: Unit, integration, and performance tests
- **Security Scanning**: Vulnerability detection, secrets scanning
- **Documentation**: Build and validate documentation
- **Container Build**: Docker image building and scanning
- **Dependency Audit**: Security and license compliance

#### Key Features:
- Multi-Python version testing (3.9-3.12)
- Parallel job execution for speed
- Comprehensive coverage reporting
- Security vulnerability scanning
- Automated dependency checks

### 2. Continuous Deployment (`cd.yml`)
**Triggers**: Push to main, Tags (v*), Manual dispatch
**Purpose**: Automated deployment to staging and production

#### Jobs:
- **Build & Push**: Multi-architecture container images
- **Security Scan**: Container vulnerability assessment
- **Staging Deployment**: Automated staging environment deployment
- **Production Deployment**: Blue-green production deployment
- **Rollback**: Automatic rollback on failure

#### Key Features:
- SLSA Level 3 provenance generation
- Multi-architecture builds (AMD64, ARM64)
- Blue-green deployment strategy
- Automated rollback capability
- Post-deployment validation

### 3. Security Scanning (`security-scan.yml`)
**Triggers**: Daily schedule, Push, Pull Requests
**Purpose**: Comprehensive security analysis

#### Jobs:
- **SAST**: Static application security testing
- **Dependency Scan**: Vulnerability detection in dependencies
- **Secret Scan**: Detect exposed secrets and credentials
- **Container Scan**: Docker image vulnerability assessment
- **IaC Scan**: Infrastructure as Code security analysis
- **License Compliance**: License compatibility checking
- **DAST**: Dynamic application security testing
- **Compliance**: SLSA and governance checks

#### Key Features:
- Multiple security tools integration
- SARIF format results for GitHub Security tab
- Automated security reporting
- Compliance verification
- License auditing

### 4. Dependency Updates (`dependency-update.yml`)
**Triggers**: Weekly schedule, Manual dispatch
**Purpose**: Automated dependency management

#### Jobs:
- **Python Dependencies**: Update Python packages
- **Docker Images**: Update base Docker images
- **GitHub Actions**: Update action versions
- **Security Advisories**: Monitor and alert on vulnerabilities
- **Dependency Analysis**: Health and compatibility analysis

#### Key Features:
- Automated PR creation for updates
- Security vulnerability monitoring
- Compatibility testing
- Dependency health analysis
- Automated maintenance tasks

## Configuration Requirements

### Environment Variables
```yaml
# Required for all workflows
PYTHON_VERSION: '3.11'
NODE_VERSION: '18'
REGISTRY: 'ghcr.io'

# CI-specific
MONGODB_URL: 'mongodb://localhost:27017/test_materials_discovery'
REDIS_URL: 'redis://localhost:6379/1'

# CD-specific
IMAGE_NAME: '${{ github.repository }}'
```

### Required Secrets
```yaml
# Deployment
STAGING_KUBECONFIG: 'Base64 encoded kubeconfig for staging'
PRODUCTION_KUBECONFIG: 'Base64 encoded kubeconfig for production'

# Security
API_SECRET_KEY: 'Application secret key'
GRAFANA_API_KEY: 'Grafana API key for annotations'
SNYK_TOKEN: 'Snyk security scanning token'
GITLEAKS_LICENSE: 'GitLeaks license key (optional)'

# Notifications
SLACK_WEBHOOK_URL: 'Slack webhook for notifications'
DOCS_DEPLOY_TOKEN: 'Token for documentation deployment'

# Database (Production)
MONGODB_ROOT_PASSWORD: 'MongoDB root password'
MONGODB_USERNAME: 'Application database user'
MONGODB_PASSWORD: 'Application database password'
REDIS_PASSWORD: 'Redis password'
POSTGRES_USER: 'PostgreSQL user for Grafana'
POSTGRES_PASSWORD: 'PostgreSQL password'
GRAFANA_ADMIN_PASSWORD: 'Grafana admin password'
GRAFANA_SECRET_KEY: 'Grafana secret key'
```

### Branch Protection Rules
```yaml
# Configure in GitHub repository settings
Required status checks:
  - Code Quality
  - Test Suite (ubuntu-latest, 3.11)
  - Security Scan
  - Container Build & Scan

Require branches to be up to date: true
Require review from code owners: true
Dismiss stale PR approvals: true
Require status checks to pass: true
Restrict pushes to matching branches: true
```

## Workflow Customization

### Modifying Test Configuration
```yaml
# In ci.yml - adjust test matrix
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']
    os: [ubuntu-latest, windows-latest, macos-latest]  # Add OS matrix
```

### Customizing Security Scans
```yaml
# In security-scan.yml - adjust scan frequency
schedule:
  - cron: '0 3 * * *'  # Daily at 3 AM
  - cron: '0 15 * * 5'  # Weekly on Friday at 3 PM
```

### Deployment Environment Configuration
```yaml
# In cd.yml - add new environment
deploy-qa:
  name: Deploy to QA
  runs-on: ubuntu-latest
  needs: [build, security-scan]
  environment:
    name: qa
    url: https://qa.materials-orchestrator.com
```

## Advanced Features

### Multi-Environment Deployment
```yaml
# Matrix deployment to multiple environments
strategy:
  matrix:
    environment: [dev, staging, qa]
    include:
      - environment: dev
        namespace: materials-dev
      - environment: staging
        namespace: materials-staging
```

### Conditional Workflow Execution
```yaml
# Run only on specific conditions
if: |
  github.event_name == 'push' && 
  contains(github.event.head_commit.message, '[deploy]')
```

### Workflow Dependencies
```yaml
# Chain workflows together
workflow_run:
  workflows: ["Continuous Integration"]
  types: [completed]
  branches: [main]
```

## Performance Optimization

### Caching Strategies
```yaml
# Cache Python dependencies
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}

# Cache Docker layers
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Parallel Execution
```yaml
# Run jobs in parallel
jobs:
  test-unit:
    # Unit tests
  test-integration:
    # Integration tests  
  test-performance:
    # Performance tests
```

### Resource Optimization
```yaml
# Use faster runners for CI
runs-on: ubuntu-latest-8-cores  # Larger runners for faster builds

# Limit concurrent workflows
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## Monitoring and Alerting

### Workflow Monitoring
```yaml
# Add workflow status to external monitoring
- name: Report workflow status
  if: always()
  run: |
    curl -X POST "${{ secrets.MONITORING_WEBHOOK }}" \
      -H "Content-Type: application/json" \
      -d '{"workflow": "${{ github.workflow }}", "status": "${{ job.status }}"}'
```

### Failed Workflow Alerts
```yaml
# Send alerts on workflow failures
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
```bash
# Check workflow syntax
yamllint .github/workflows/ci.yml

# Verify branch protection rules
gh api repos/:owner/:repo/branches/main/protection
```

#### Permission Errors
```bash
# Check workflow permissions
permissions:
  contents: read
  packages: write
  security-events: write
```

#### Secret Not Found
```bash
# List repository secrets
gh secret list

# Set missing secret
gh secret set SECRET_NAME --body "secret-value"
```

#### Failed Deployment
```bash
# Check deployment logs
kubectl logs -n materials-production deployment/materials-orchestrator

# Rollback deployment
kubectl rollout undo deployment/materials-orchestrator -n materials-production
```

### Debugging Workflows

#### Enable Debug Logging
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

#### Add Debug Steps
```yaml
- name: Debug environment
  run: |
    echo "GitHub Event: ${{ github.event_name }}"
    echo "GitHub Ref: ${{ github.ref }}"
    echo "Runner OS: ${{ runner.os }}"
    env | sort
```

## Best Practices

### Security
- Store sensitive data in GitHub Secrets
- Use least privilege principle for permissions
- Regularly rotate secrets and tokens
- Enable vulnerability alerts and security updates
- Use SARIF format for security scan results

### Performance
- Cache dependencies and build artifacts
- Use matrix builds for parallel execution
- Cancel redundant workflow runs
- Optimize Docker builds with multi-stage builds
- Use appropriate runner sizes

### Reliability
- Implement proper error handling
- Add retry logic for flaky operations
- Use health checks for deployments
- Implement rollback mechanisms
- Monitor workflow success rates

### Maintenance
- Keep workflows and actions up to date
- Review and update branch protection rules
- Monitor workflow execution times
- Clean up old artifacts and caches
- Document workflow changes

## Integration with External Services

### Slack Notifications
```yaml
- uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#materials-orchestrator'
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Jira Integration
```yaml
- uses: atlassian/gajira-transition@v3
  with:
    issue: ${{ github.event.issue.number }}
    transition: "In Progress"
```

### Email Notifications
```yaml
- uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.MAIL_USERNAME }}
    password: ${{ secrets.MAIL_PASSWORD }}
    subject: "Deployment Complete: ${{ github.ref }}"
    body: "Deployment to production completed successfully."
```

## Workflow Templates

The `examples/` directory contains complete workflow templates that can be copied directly to `.github/workflows/`. Each template includes:

- Comprehensive job definitions
- Proper error handling
- Security best practices
- Performance optimizations
- Detailed documentation

### Template Usage
1. Copy template to `.github/workflows/`
2. Configure required secrets
3. Adjust settings for your environment
4. Test with a pull request
5. Monitor and iterate

For specific implementation examples and advanced configurations, see the individual workflow files in the `examples/` directory.