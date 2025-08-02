# Manual Setup Required

Due to GitHub App permission limitations, some repository configuration must be completed manually by repository maintainers.

## 🚨 Critical Setup Items

### 1. GitHub Workflows
**Location**: Copy from `docs/workflows/examples/` to `.github/workflows/`

```bash
mkdir -p .github/workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. GitHub Secrets Configuration
Required secrets for CI/CD workflows:

```bash
# Deployment secrets
gh secret set STAGING_KUBECONFIG --body "$(base64 -w 0 ~/.kube/staging-config)"
gh secret set PRODUCTION_KUBECONFIG --body "$(base64 -w 0 ~/.kube/prod-config)"

# Security secrets
gh secret set API_SECRET_KEY --body "$(openssl rand -hex 32)"
gh secret set GRAFANA_API_KEY --body "your-grafana-api-key"
gh secret set SNYK_TOKEN --body "your-snyk-token"

# Notification secrets
gh secret set SLACK_WEBHOOK_URL --body "your-slack-webhook-url"
gh secret set DOCS_DEPLOY_TOKEN --body "your-docs-deploy-token"

# Database secrets (Production)
gh secret set MONGODB_ROOT_PASSWORD --body "secure-mongodb-password"
gh secret set MONGODB_USERNAME --body "materials_user"
gh secret set MONGODB_PASSWORD --body "secure-user-password"
gh secret set REDIS_PASSWORD --body "secure-redis-password"
gh secret set POSTGRES_USER --body "grafana_user"
gh secret set POSTGRES_PASSWORD --body "secure-postgres-password"
gh secret set GRAFANA_ADMIN_PASSWORD --body "secure-grafana-password"
gh secret set GRAFANA_SECRET_KEY --body "$(openssl rand -hex 32)"
```

### 3. Branch Protection Rules
Configure in GitHub repository settings:

```yaml
Branch: main
Required status checks:
  - Code Quality
  - Test Suite (ubuntu-latest, 3.11)
  - Security Scan
  - Container Build & Scan

Options:
☑️ Require branches to be up to date before merging
☑️ Require review from code owners
☑️ Dismiss stale PR approvals when new commits are pushed
☑️ Require status checks to pass before merging
☑️ Restrict pushes that create files larger than 100MB
☑️ Include administrators
```

### 4. GitHub Security Features
Enable in repository settings:

```bash
# Enable security features via GitHub CLI
gh repo edit --enable-vulnerability-alerts
gh repo edit --enable-automated-security-fixes
gh repo edit --enable-private-vulnerability-reporting

# Or via web interface:
# Settings → Security and analysis → Enable all features
```

### 5. Repository Settings
Configure repository description and metadata:

```bash
gh repo edit \
  --description "Autonomous laboratory platform for materials discovery" \
  --homepage "https://materials-orchestrator.com" \
  --add-topic "materials-science" \
  --add-topic "autonomous-systems" \
  --add-topic "machine-learning" \
  --add-topic "laboratory-automation" \
  --add-topic "bayesian-optimization"
```

## 📋 Optional Enhancements

### 1. Issue and PR Templates
Create `.github/ISSUE_TEMPLATE/` directory with templates:

```bash
mkdir -p .github/ISSUE_TEMPLATE

# Bug report template
cat > .github/ISSUE_TEMPLATE/bug_report.md << 'EOF'
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python version: [e.g. 3.11]
 - Materials Orchestrator version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
EOF

# Feature request template
cat > .github/ISSUE_TEMPLATE/feature_request.md << 'EOF'
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions.

**Additional context**
Add any other context or screenshots about the feature request here.
EOF

# Pull request template
cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Added new tests for changes

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] New and existing unit tests pass locally with my changes
EOF
```

### 2. Dependabot Configuration
Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    reviewers:
      - "team-materials-orchestrator"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "deps"
      prefix-development: "deps(dev)"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "tuesday"
      time: "09:00"
    reviewers:
      - "team-materials-orchestrator"
    labels:
      - "dependencies"
      - "docker"
    commit-message:
      prefix: "deps"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "wednesday"
      time: "09:00"
    reviewers:
      - "team-materials-orchestrator"
    labels:
      - "dependencies"
      - "github-actions"
    commit-message:
      prefix: "deps"
```

### 3. CODEOWNERS File
Create `.github/CODEOWNERS`:

```bash
# Global owners
* @terragonlabs/materials-orchestrator-team

# Core code
/src/ @terragonlabs/core-team

# Documentation
/docs/ @terragonlabs/documentation-team
README.md @terragonlabs/documentation-team

# CI/CD and infrastructure
/.github/ @terragonlabs/devops-team
/docker-compose*.yml @terragonlabs/devops-team
/Dockerfile* @terragonlabs/devops-team
/monitoring/ @terragonlabs/devops-team

# Security
/SECURITY.md @terragonlabs/security-team
/.github/workflows/security-scan.yml @terragonlabs/security-team

# Tests
/tests/ @terragonlabs/qa-team

# Dependencies
/pyproject.toml @terragonlabs/core-team
/requirements*.txt @terragonlabs/core-team
```

### 4. Repository Labels
Create useful labels for issue and PR management:

```bash
# Priority labels
gh label create "priority:critical" --color "d73a4a" --description "Critical priority"
gh label create "priority:high" --color "f85149" --description "High priority"
gh label create "priority:medium" --color "fb8500" --description "Medium priority"
gh label create "priority:low" --color "0969da" --description "Low priority"

# Type labels
gh label create "type:bug" --color "d73a4a" --description "Something isn't working"
gh label create "type:feature" --color "a2eeef" --description "New feature or request"
gh label create "type:enhancement" --color "7cb342" --description "Enhancement to existing functionality"
gh label create "type:documentation" --color "0075ca" --description "Improvements or additions to documentation"
gh label create "type:security" --color "ee0701" --description "Security related issue"

# Component labels
gh label create "component:core" --color "5319e7" --description "Core orchestration system"
gh label create "component:robots" --color "5319e7" --description "Robot integration"
gh label create "component:optimization" --color "5319e7" --description "Bayesian optimization"
gh label create "component:monitoring" --color "5319e7" --description "Monitoring and observability"
gh label create "component:testing" --color "5319e7" --description "Testing infrastructure"

# Status labels
gh label create "status:blocked" --color "d73a4a" --description "Blocked by external dependency"
gh label create "status:in-progress" --color "fbca04" --description "Currently being worked on"
gh label create "status:needs-review" --color "0052cc" --description "Needs code review"
gh label create "status:needs-testing" --color "1d76db" --description "Needs testing"
```

## 🔧 Validation Steps

After completing the manual setup, validate the configuration:

### 1. Test Workflows
```bash
# Trigger CI workflow
git push origin main

# Check workflow status
gh run list --limit 5

# View workflow details
gh run view --log
```

### 2. Test Security Features
```bash
# Check security advisories
gh api repos/:owner/:repo/security-advisories

# Verify secret scanning
gh api repos/:owner/:repo/secret-scanning/alerts
```

### 3. Validate Branch Protection
```bash
# Try to push directly to main (should fail)
git push origin main --force

# Check protection status
gh api repos/:owner/:repo/branches/main/protection
```

### 4. Test Repository Health
```bash
# Run health check script
python scripts/automation/repo-health-check.py

# Collect metrics
python scripts/automation/metrics-collector.py
```

## 📞 Support

If you encounter issues during setup:

1. **Check the documentation**: Each component has detailed setup instructions
2. **Review the logs**: Workflow and script logs provide detailed error information
3. **Validate permissions**: Ensure you have the necessary repository permissions
4. **Test incrementally**: Set up one component at a time and test before proceeding

## ✅ Completion Checklist

- [ ] GitHub workflows copied and configured
- [ ] All required secrets configured
- [ ] Branch protection rules enabled
- [ ] Security features activated
- [ ] Repository metadata configured
- [ ] Issue and PR templates created
- [ ] Dependabot configured
- [ ] CODEOWNERS file created
- [ ] Repository labels created
- [ ] Configuration validated with test runs

Once all items are completed, the Self-Driving Materials Orchestrator will have a fully configured SDLC implementation!