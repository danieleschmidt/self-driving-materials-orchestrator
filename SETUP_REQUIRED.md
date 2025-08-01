# Manual Setup Required

## Overview

Due to GitHub App permission limitations, the following manual setup steps are required to complete the SDLC implementation. These tasks must be performed by a repository maintainer with appropriate permissions.

## 🚨 Critical Setup Tasks

### 1. GitHub Workflows (REQUIRED)

**Location**: `.github/workflows/`
**Priority**: HIGH
**Estimated Time**: 30 minutes

The following workflow files need to be manually created from the templates in `docs/workflows/examples/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

**Documentation**: See `docs/workflows/SETUP_GUIDE.md` for detailed instructions.

### 2. Repository Secrets Configuration (REQUIRED)

**Location**: Repository Settings > Secrets and variables > Actions
**Priority**: HIGH
**Estimated Time**: 15 minutes

Required secrets for workflows to function:

#### Core Secrets
- `GITHUB_TOKEN` - Automatically provided by GitHub
- `DEPENDENCY_UPDATE_TOKEN` - Personal access token for dependency updates

#### Docker & Deployment
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- `STAGING_KUBECONFIG` - Base64 encoded kubeconfig for staging
- `PRODUCTION_KUBECONFIG` - Base64 encoded kubeconfig for production

#### Security & Monitoring
- `SNYK_TOKEN` - Snyk API token for vulnerability scanning
- `SLACK_WEBHOOK` - Slack webhook URL for notifications
- `PAGERDUTY_INTEGRATION_KEY` - PagerDuty integration key

### 3. Branch Protection Rules (REQUIRED)

**Location**: Repository Settings > Branches
**Priority**: HIGH
**Estimated Time**: 10 minutes

Configure branch protection for `main` branch:
- Require pull request reviews (2 reviewers)
- Require status checks to pass
- Require branches to be up to date
- Include administrators
- Dismiss stale reviews

Required status checks:
- `Lint and Code Quality`
- `Unit Tests`
- `Integration Tests`
- `Security Scan`
- `Docker Build Test`

### 4. Environment Configuration (REQUIRED)

**Location**: Repository Settings > Environments
**Priority**: HIGH
**Estimated Time**: 10 minutes

Create environments:
- **staging**: No protection rules
- **production**: Require 2 reviewers from maintainers team
- **production-rollback**: Emergency rollback environment

## 📋 Recommended Setup Tasks

### 5. Security Settings (RECOMMENDED)

**Location**: Repository Settings > Security
**Priority**: MEDIUM
**Estimated Time**: 5 minutes

Enable:
- Vulnerability alerts
- Dependabot security updates
- Secret scanning
- Push protection for secrets
- Code scanning (CodeQL)

### 6. Issue and PR Templates (RECOMMENDED)

**Location**: `.github/` directory
**Priority**: MEDIUM
**Estimated Time**: 10 minutes

Templates are provided but need to be activated:

```bash
# These directories already exist with templates
.github/ISSUE_TEMPLATE/
.github/PULL_REQUEST_TEMPLATE.md
```

No action required - templates are already in place.

### 7. Repository Settings (RECOMMENDED)

**Location**: Repository Settings > General
**Priority**: LOW
**Estimated Time**: 5 minutes

Recommended settings:
- Disable merge commits (use squash and rebase only)
- Enable "Always suggest updating pull request branches"
- Enable "Automatically delete head branches"
- Set default branch to `main`

## 📝 Optional Enhancement Tasks

### 8. External Integrations (OPTIONAL)

**Priority**: LOW
**Estimated Time**: 30 minutes per integration

Consider integrating with:
- **Codecov**: Code coverage reporting
- **Sonarcloud**: Code quality analysis
- **Sentry**: Error tracking and monitoring
- **DataDog**: Application performance monitoring

### 9. Custom Domain (OPTIONAL)

**Priority**: LOW
**Estimated Time**: 15 minutes

If using GitHub Pages for documentation:
- Configure custom domain in Pages settings
- Set up DNS CNAME record
- Enable HTTPS enforcement

### 10. Team Permissions (OPTIONAL)

**Location**: Repository Settings > Manage access
**Priority**: LOW
**Estimated Time**: 10 minutes

Set up teams with appropriate permissions:
- **Maintainers**: Admin access
- **Developers**: Write access
- **Security Team**: Read access + security alerts
- **External Contributors**: No access (rely on forks)

## 🔧 Technical Requirements

### Docker Registry Setup

If using a private Docker registry:

1. **GitHub Container Registry (ghcr.io)**:
   - Already configured in workflows
   - Uses GITHUB_TOKEN automatically
   - No additional setup required

2. **Docker Hub**:
   - Create account at https://hub.docker.com
   - Generate access token
   - Add DOCKER_USERNAME and DOCKER_PASSWORD secrets

3. **Private Registry**:
   - Update registry URL in CD workflow
   - Add appropriate authentication secrets

### Kubernetes Cluster Setup

For deployment workflows to function:

1. **Staging Cluster**:
   - Create kubeconfig with appropriate permissions
   - Base64 encode: `base64 -w 0 kubeconfig-staging.yaml`
   - Add as STAGING_KUBECONFIG secret

2. **Production Cluster**:
   - Create kubeconfig with appropriate permissions
   - Base64 encode: `base64 -w 0 kubeconfig-production.yaml`
   - Add as PRODUCTION_KUBECONFIG secret

3. **Kubernetes Manifests**:
   - Create `k8s/staging/` and `k8s/production/` directories
   - Add deployment, service, and ingress manifests
   - See SETUP_GUIDE.md for examples

### Monitoring Integration

1. **Slack Integration**:
   - Create Slack app with webhook permissions
   - Add webhook URL as SLACK_WEBHOOK secret
   - Configure appropriate channels

2. **PagerDuty Integration**:
   - Create PagerDuty service
   - Generate integration key
   - Add as PAGERDUTY_INTEGRATION_KEY secret

## ✅ Verification Steps

After completing the setup:

### 1. Test CI Workflow
```bash
# Create a test branch and PR
git checkout -b test-ci-setup
echo "# Test" > TEST.md
git add TEST.md
git commit -m "test: verify CI workflow"
git push origin test-ci-setup
# Create PR and verify all checks pass
```

### 2. Test Security Scanning
```bash
# Manually trigger security scan workflow
gh workflow run security-scan.yml
# Verify scan completes and uploads results
```

### 3. Test Dependency Updates
```bash
# Manually trigger dependency update workflow
gh workflow run dependency-update.yml
# Verify it creates appropriate PRs if updates available
```

### 4. Verify Branch Protection
```bash
# Try to push directly to main (should fail)
git checkout main
echo "# Direct push test" >> README.md
git add README.md
git commit -m "test: direct push"
git push origin main  # This should fail
```

## 🆘 Support and Documentation

### Complete Documentation
- **Workflow Setup**: `docs/workflows/SETUP_GUIDE.md`
- **Docker Deployment**: `docs/deployment/docker.md`
- **Monitoring Guide**: `docs/monitoring/README.md`
- **Testing Documentation**: `docs/testing/README.md`

### Getting Help
1. **GitHub Actions**: https://docs.github.com/en/actions
2. **Repository Issues**: Create issue with "setup" label
3. **Community Support**: GitHub Actions community forum

### Troubleshooting
If workflows fail after setup:
1. Check repository secrets are correctly set
2. Verify branch protection rules don't conflict
3. Ensure environment permissions are correct
4. Review workflow logs for specific errors

## 📊 Setup Checklist

Copy and use this checklist to track setup progress:

### Critical Tasks (Must Complete)
- [ ] Copy workflow files from `docs/workflows/examples/` to `.github/workflows/`
- [ ] Configure repository secrets (GITHUB_TOKEN, DOCKER_*, SLACK_WEBHOOK, etc.)
- [ ] Set up branch protection rules for `main` branch
- [ ] Create staging and production environments
- [ ] Test CI workflow with a test PR

### Recommended Tasks (Should Complete)
- [ ] Enable security settings (vulnerability alerts, secret scanning)
- [ ] Configure repository settings (merge policies, branch deletion)
- [ ] Set up external integrations (Codecov, monitoring)
- [ ] Create team permissions and access controls

### Optional Tasks (Nice to Have)
- [ ] Set up custom domain for documentation
- [ ] Configure additional security integrations
- [ ] Set up performance monitoring
- [ ] Create custom deployment environments

### Verification Tasks
- [ ] All CI checks pass on test PR
- [ ] Security scans run successfully
- [ ] Branch protection prevents direct pushes to main
- [ ] Notifications work correctly (Slack, email)
- [ ] Documentation builds and deploys correctly

## 🎯 Success Criteria

Setup is complete when:
1. ✅ All workflow files are in place and functioning
2. ✅ Repository secrets are configured correctly
3. ✅ Branch protection prevents unauthorized changes
4. ✅ CI/CD pipelines run successfully on PRs
5. ✅ Security scanning is active and reporting
6. ✅ Notifications reach appropriate channels
7. ✅ Documentation is accessible and up-to-date

**Estimated Total Setup Time**: 2-3 hours for complete implementation

---

**Note**: This setup is required due to GitHub App permission limitations. Once completed manually, all automation will function as designed and documented in the comprehensive SDLC implementation.