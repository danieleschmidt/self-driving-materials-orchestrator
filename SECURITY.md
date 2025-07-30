# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you believe you have found a security vulnerability in self-driving-materials-orchestrator, please report it to us as described below.

### Please do NOT report security vulnerabilities through public GitHub issues.

Instead, please report them via email to: **security@terragonlabs.com**

Include the following information in your report:
- Type of issue (buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Initial Assessment**: We will provide an initial assessment within 5 business days.
- **Progress Updates**: We will keep you informed of our progress throughout the investigation.
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days.

### Safe Harbor

We support safe harbor for security researchers who:
- Make a good faith effort to avoid privacy violations and disruptions to others
- Only interact with accounts you own or with explicit permission of the account holder
- Do not access or modify data that does not belong to you
- Report vulnerabilities as soon as possible after discovery
- Do not run automated vulnerability scanners against our infrastructure

### Recognition

We believe in recognizing the contributions of security researchers who help keep our users safe. With your permission, we will:
- Acknowledge your contribution in our security advisories
- Include you in our Security Hall of Fame
- Provide appropriate recognition for your responsible disclosure

### Scope

This security policy applies to:
- The main self-driving-materials-orchestrator repository
- Official container images
- Production deployments and infrastructure
- Third-party integrations and plugins

### Out of Scope

The following are typically out of scope:
- Issues in third-party applications or libraries (report to respective maintainers)
- Social engineering attacks
- Physical attacks against facilities or personnel
- Denial of service attacks
- Issues requiring physical access to laboratory equipment

### Security Best Practices

When deploying self-driving-materials-orchestrator:

1. **Network Security**
   - Use TLS/SSL for all communications
   - Implement proper network segmentation
   - Restrict access to laboratory networks

2. **Authentication & Authorization**
   - Use strong, unique passwords
   - Implement multi-factor authentication
   - Follow principle of least privilege

3. **System Security**
   - Keep all dependencies up to date
   - Regular security audits
   - Monitor for suspicious activity

4. **Laboratory Safety**
   - Implement emergency stops on all robotic systems
   - Maintain human oversight of autonomous operations
   - Follow all laboratory safety protocols

For more information about security practices, see our [Security Documentation](docs/security/).

## Contact

For any questions about this security policy, please contact: security@terragonlabs.com