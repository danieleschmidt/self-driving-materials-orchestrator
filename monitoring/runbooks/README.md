# Operational Runbooks

This directory contains operational runbooks for responding to common alerts and incidents in the Materials Orchestrator system.

## Runbook Structure

Each runbook follows a standard structure:

1. **Alert Description**: What triggered this runbook
2. **Severity Level**: Critical/High/Medium/Low
3. **Impact**: What services or functionality are affected
4. **Initial Assessment**: Quick steps to understand the issue
5. **Resolution Steps**: Detailed step-by-step resolution process
6. **Escalation**: When and how to escalate
7. **Post-Incident**: Follow-up actions and prevention

## Available Runbooks

### System Health
- [Service Down](service-down.md) - When core services become unavailable
- [High Memory Usage](high-memory.md) - Memory consumption alerts
- [High CPU Usage](high-cpu.md) - CPU utilization issues
- [Disk Space Low](disk-space.md) - Storage capacity problems

### Database Issues
- [Database Connection Failed](database-connection.md) - MongoDB connectivity issues
- [Slow Database Queries](slow-queries.md) - Database performance problems
- [Database Replication Lag](replication-lag.md) - MongoDB replica set issues

### Application Issues
- [High Error Rate](high-error-rate.md) - Increased application errors
- [API Latency High](api-latency.md) - Response time degradation
- [Experiment Failures](experiment-failures.md) - High experiment failure rates
- [Optimization Stalled](optimization-stalled.md) - Optimization engine issues

### Robot and Hardware
- [Robot Offline](robot-offline.md) - Robot connectivity problems
- [Robot Command Failures](robot-command-failures.md) - Robot execution issues
- [Safety System Violation](safety-violation.md) - Safety protocol breaches

### Security Incidents
- [Unauthorized Access](unauthorized-access.md) - Security breach responses
- [Authentication Failures](auth-failures.md) - Login and auth issues
- [Data Export Anomaly](data-export-anomaly.md) - Unusual data access patterns

## Alert Response Framework

### Severity Levels

#### Critical (P0)
- **Response Time**: Immediate (< 5 minutes)
- **Examples**: Service completely down, data corruption, security breach
- **Actions**: Page on-call engineer, emergency response

#### High (P1)
- **Response Time**: Within 30 minutes
- **Examples**: Degraded service, high error rates, robot failures
- **Actions**: Alert on-call team, investigate immediately

#### Medium (P2)
- **Response Time**: Within 2 hours
- **Examples**: Performance degradation, non-critical failures
- **Actions**: Create ticket, investigate during business hours

#### Low (P3)
- **Response Time**: Within 24 hours
- **Examples**: Minor issues, informational alerts
- **Actions**: Log for trending, address in regular maintenance

### Escalation Matrix

| Time Since Alert | Action |
|------------------|--------|
| 0 minutes | On-call engineer notified |
| 15 minutes | Team lead notified (for P0/P1) |
| 30 minutes | Management notified (for P0) |
| 60 minutes | Incident commander assigned (for P0) |

## General Response Procedures

### Initial Response (First 5 minutes)
1. **Acknowledge** the alert to stop notifications
2. **Assess** the scope and impact
3. **Check** if it's a known issue or maintenance window
4. **Triage** severity level
5. **Begin** initial troubleshooting

### Investigation Phase
1. **Gather** relevant logs and metrics
2. **Check** recent changes or deployments
3. **Review** system health dashboards
4. **Identify** root cause or contributing factors
5. **Document** findings in incident tracking

### Resolution Phase
1. **Implement** fix or workaround
2. **Verify** system recovery
3. **Monitor** for recurring issues
4. **Update** incident tracking
5. **Communicate** status to stakeholders

### Post-Incident
1. **Document** timeline and actions taken
2. **Conduct** post-mortem if needed
3. **Identify** prevention measures
4. **Update** monitoring and alerts
5. **Share** learnings with team

## Tools and Resources

### Monitoring Dashboards
- **System Overview**: http://grafana:3000/d/system-overview
- **Application Metrics**: http://grafana:3000/d/app-metrics
- **Infrastructure**: http://grafana:3000/d/infrastructure
- **Business Metrics**: http://grafana:3000/d/business-metrics

### Log Analysis
- **Grafana Loki**: http://grafana:3000/explore
- **Kibana**: http://kibana:5601
- **Application Logs**: `docker logs materials-orchestrator`

### Command Line Tools
```bash
# Check service status
curl http://materials-orchestrator:8000/health

# View real-time logs
docker logs -f materials-orchestrator

# Check container resources
docker stats materials-orchestrator

# MongoDB operations
docker exec mongodb mongo --eval "db.stats()"

# Redis operations
docker exec redis redis-cli info
```

### Communication Channels
- **Slack**: #materials-alerts, #materials-ops
- **Email**: ops@materials-lab.com
- **PagerDuty**: Materials Orchestrator service
- **Incident Management**: https://company.pagerduty.com

## Runbook Maintenance

### Regular Reviews
- **Monthly**: Review runbook accuracy and completeness
- **After Incidents**: Update based on lessons learned
- **Quarterly**: Validate contact information and procedures
- **Annually**: Comprehensive review and process improvements

### Testing
- **Alert Testing**: Regularly test alert firing and notifications
- **Procedure Validation**: Practice runbook procedures
- **Tool Access**: Verify access to all monitoring tools
- **Communication**: Test notification channels

### Updates
- **Version Control**: Track all runbook changes
- **Team Review**: Get team input on procedure changes
- **Training**: Ensure team is trained on updated procedures
- **Documentation**: Keep related documentation in sync

## Contact Information

### On-Call Rotation
- **Primary**: Check PagerDuty schedule
- **Secondary**: Check PagerDuty schedule
- **Escalation**: Team lead and management

### Subject Matter Experts
- **Systems**: DevOps team
- **Database**: Database administrator
- **Application**: Development team
- **Security**: Security team
- **Robots**: Laboratory operations team

### Emergency Contacts
- **IT Emergency**: +1-555-0199
- **Facilities**: +1-555-0299
- **Security**: +1-555-0399
- **Management**: +1-555-0499