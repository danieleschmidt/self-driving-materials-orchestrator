# Monitoring & Observability

This guide covers the comprehensive monitoring and observability setup for the Self-Driving Materials Orchestrator.

## Overview

The monitoring stack provides complete visibility into:
- **Application Performance**: API response times, throughput, errors
- **Optimization Metrics**: Campaign progress, convergence rates, experiment success
- **Infrastructure Health**: CPU, memory, disk, network usage
- **Database Performance**: MongoDB query performance, connection pools
- **Business Metrics**: Materials discovered, experiments per day, cost savings

## Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        A[Materials Orchestrator API]
        B[Streamlit Dashboard]
        C[Robot Controllers]
    end
    
    subgraph "Metrics Collection"
        D[Prometheus]
        E[Custom Metrics Exporter]
        F[Node Exporter]
        G[MongoDB Exporter]
    end
    
    subgraph "Visualization"
        H[Grafana]
        I[Pre-built Dashboards]
        J[Custom Dashboards]
    end
    
    subgraph "Alerting"
        K[AlertManager]
        L[Alert Rules]
        M[Notification Channels]
    end
    
    A --> D
    B --> D
    C --> E
    F --> D
    G --> D
    D --> H
    D --> K
    H --> I
    H --> J
    K --> L
    L --> M
```

## Metrics Categories

### 1. Application Metrics

#### API Performance
- `http_requests_total` - Total HTTP requests by method and status
- `http_request_duration_seconds` - Request duration histogram
- `api_errors_total` - API errors by type and endpoint
- `active_connections` - Current active connections

#### Experiment Metrics
- `experiments_total` - Total experiments run
- `experiments_success_rate` - Success rate percentage
- `campaign_progress` - Current campaign progress
- `optimization_convergence_time` - Time to convergence

#### Robot & Instrument Metrics
- `robot_status` - Robot availability and status
- `robot_utilization` - Robot utilization percentage
- `instrument_queue_length` - Current instrument queue length
- `synthesis_time_seconds` - Synthesis duration

### 2. Business Metrics

#### Discovery Metrics
- `materials_discovered_total` - Number of novel materials found
- `target_achievements` - Materials meeting target criteria
- `cost_savings_total` - Total cost savings vs manual methods
- `time_acceleration_factor` - Speed improvement over traditional methods

#### Resource Efficiency
- `resource_utilization` - Lab resource utilization
- `experiment_cost_per_material` - Cost efficiency metrics
- `energy_consumption` - Energy usage tracking
- `waste_reduction` - Environmental impact metrics

### 3. Infrastructure Metrics

#### System Resources
- `cpu_usage_percent` - CPU utilization
- `memory_usage_bytes` - Memory consumption
- `disk_usage_bytes` - Disk space utilization
- `network_io_bytes` - Network traffic

#### Database Performance
- `mongodb_operations_total` - Database operations count
- `mongodb_query_duration_seconds` - Query performance
- `mongodb_connections_current` - Active connections
- `mongodb_cache_hit_ratio` - Cache efficiency

## Dashboard Configuration

### 1. Executive Dashboard
**Target Audience**: Lab managers, executives
**Key Metrics**:
- Daily experiment count
- Success rate trends
- Cost savings achieved
- Novel materials discovered
- Resource utilization summary

### 2. Operations Dashboard
**Target Audience**: Lab technicians, operators
**Key Metrics**:
- Current experiment status
- Robot and instrument availability
- Queue lengths and wait times
- Recent alerts and issues
- Real-time system health

### 3. Scientific Dashboard
**Target Audience**: Researchers, scientists
**Key Metrics**:
- Optimization progress
- Parameter space exploration
- Property prediction accuracy
- Campaign comparison metrics
- Statistical analysis views

### 4. Infrastructure Dashboard
**Target Audience**: DevOps, system administrators
**Key Metrics**:
- System resource utilization
- Application performance
- Error rates and latencies
- Database performance
- Security metrics

## Alert Configuration

### Critical Alerts
- **System Down**: Any core service unavailable
- **High Error Rate**: >5% API error rate
- **Resource Exhaustion**: >90% CPU/memory usage
- **Database Issues**: Connection failures or slow queries
- **Security Incidents**: Unauthorized access attempts

### Warning Alerts
- **Performance Degradation**: Response time >2x baseline
- **Queue Backup**: Experiment queue >100 items
- **Resource Pressure**: >80% resource utilization
- **Optimization Stalling**: No improvement for 50+ experiments
- **Hardware Issues**: Robot or instrument errors

### Informational Alerts
- **Campaign Completion**: Optimization campaign finished
- **Target Achievement**: Material meets success criteria
- **Milestone Reached**: Experiment count milestones
- **Maintenance Reminders**: Scheduled maintenance due
- **Discovery Notifications**: Novel material discovered

## Setup Instructions

### 1. Basic Monitoring Stack

#### Start Monitoring Services
```bash
# Development environment
docker-compose up -d prometheus grafana

# Production environment
docker-compose -f docker-compose.production.yml up -d prometheus grafana alertmanager
```

#### Access Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### 2. Custom Metrics Integration

#### Add Metrics to Application
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
EXPERIMENTS_TOTAL = Counter('experiments_total', 'Total experiments run', ['status'])
EXPERIMENT_DURATION = Histogram('experiment_duration_seconds', 'Experiment duration')
ACTIVE_CAMPAIGNS = Gauge('active_campaigns', 'Number of active campaigns')

# Use in code
EXPERIMENTS_TOTAL.labels(status='success').inc()
EXPERIMENT_DURATION.observe(experiment_time)
ACTIVE_CAMPAIGNS.set(len(active_campaigns))

# Start metrics server
start_http_server(9090)
```

#### Configure Prometheus Scraping
```yaml
# Add to prometheus.yml
scrape_configs:
  - job_name: 'custom-metrics'
    static_configs:
      - targets: ['app:9090']
    scrape_interval: 30s
```

### 3. Dashboard Import

#### Import Pre-built Dashboards
```bash
# Import all dashboards
for dashboard in monitoring/dashboards/*.json; do
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @"$dashboard" \
        http://admin:admin123@localhost:3000/api/dashboards/db
done
```

#### Dashboard IDs
- **Executive Dashboard**: 1001
- **Operations Dashboard**: 1002  
- **Scientific Dashboard**: 1003
- **Infrastructure Dashboard**: 1004

### 4. Alert Configuration

#### Configure Alert Rules
```yaml
# monitoring/alert_rules.yml
groups:
  - name: materials_orchestrator
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}% over the last 5 minutes"
```

#### Configure Notification Channels
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'alerts@yourcompany.com'
        subject: 'Materials Orchestrator Alert: {{ .GroupLabels.alertname }}'
```

## Performance Monitoring

### 1. Application Performance Monitoring (APM)

#### Response Time Tracking
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            RESPONSE_TIME.observe(duration)
            return result
        except Exception as e:
            ERROR_COUNT.inc()
            raise
    return wrapper

@monitor_performance
def run_experiment(parameters):
    # Experiment logic here
    pass
```

#### Database Query Monitoring
```python
from pymongo import monitoring

class CommandLogger(monitoring.CommandListener):
    def started(self, event):
        self.start_time = time.time()
    
    def succeeded(self, event):
        duration = time.time() - self.start_time
        DB_QUERY_DURATION.observe(duration)
    
    def failed(self, event):
        DB_ERRORS.inc()

monitoring.register(CommandLogger())
```

### 2. Business Metrics Tracking

#### Campaign Progress Monitoring
```python
def update_campaign_metrics(campaign):
    """Update campaign-specific metrics."""
    CAMPAIGN_PROGRESS.labels(
        campaign_id=campaign.id,
        objective=campaign.objective.target_property
    ).set(campaign.progress_percentage)
    
    if campaign.best_result:
        BEST_RESULT_VALUE.labels(
            campaign_id=campaign.id,
            property=campaign.objective.target_property
        ).set(campaign.best_result.value)
```

#### Cost and Efficiency Tracking
```python
def track_efficiency_metrics(experiment):
    """Track cost and time efficiency."""
    manual_cost = estimate_manual_cost(experiment)
    automated_cost = experiment.actual_cost
    savings = manual_cost - automated_cost
    
    COST_SAVINGS.inc(savings)
    EFFICIENCY_RATIO.set(manual_cost / automated_cost)
```

## Alerting Best Practices

### 1. Alert Hierarchy
```
CRITICAL: Immediate action required (page on-call)
WARNING: Action required within 24 hours
INFO: Awareness, no immediate action
```

### 2. Alert Fatigue Prevention
- Use appropriate thresholds
- Implement alert suppression for known issues
- Group related alerts
- Include actionable information in alerts

### 3. Runbooks
Each alert should include:
- Clear description of the problem
- Impact assessment
- Step-by-step troubleshooting guide
- Escalation procedures

## Advanced Monitoring

### 1. Distributed Tracing

#### Jaeger Integration
```python
from jaeger_client import Config

def init_tracer(service_name):
    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
        },
        service_name=service_name,
    )
    return config.initialize_tracer()

tracer = init_tracer('materials-orchestrator')

@tracer.start_span('experiment_execution')
def execute_experiment(parameters):
    # Experiment logic with tracing
    pass
```

### 2. Log Aggregation

#### ELK Stack Integration
```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    paths:
      - /app/logs/*.log
    fields:
      service: materials-orchestrator
    
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### 3. Real-time Monitoring

#### WebSocket Metrics Streaming
```python
import asyncio
import websockets
import json

async def metrics_streamer(websocket, path):
    while True:
        metrics = {
            'experiments_running': get_running_experiments(),
            'success_rate': get_current_success_rate(),
            'queue_length': get_queue_length()
        }
        await websocket.send(json.dumps(metrics))
        await asyncio.sleep(5)

start_server = websockets.serve(metrics_streamer, "localhost", 8765)
```

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus logs
docker logs prometheus
```

#### Dashboard Not Loading
```bash
# Check Grafana logs
docker logs grafana

# Verify datasource configuration
curl -u admin:admin123 http://localhost:3000/api/datasources
```

#### Alerts Not Firing
```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Verify AlertManager configuration
curl http://localhost:9093/api/v1/status
```

### Performance Optimization

#### Reduce Metrics Cardinality
```python
# Bad: High cardinality
REQUEST_COUNT.labels(
    method=method,
    path=full_path,  # Can be thousands of unique values
    user_id=user_id  # Thousands of users
)

# Good: Low cardinality
REQUEST_COUNT.labels(
    method=method,
    endpoint=normalize_path(full_path),  # Limited set of endpoints
    status_class=get_status_class(status)  # 2xx, 3xx, 4xx, 5xx
)
```

#### Optimize Query Performance
```promql
# Use recording rules for expensive queries
groups:
  - name: materials_orchestrator_recording
    rules:
      - record: materials:success_rate_5m
        expr: rate(experiments_total{status="success"}[5m]) / rate(experiments_total[5m])
```

## Security Monitoring

### 1. Authentication Metrics
```python
LOGIN_ATTEMPTS = Counter('login_attempts_total', 'Login attempts', ['status'])
FAILED_LOGINS = Counter('failed_logins_total', 'Failed login attempts', ['source_ip'])

def track_login_attempt(username, success, source_ip):
    status = 'success' if success else 'failure'
    LOGIN_ATTEMPTS.labels(status=status).inc()
    
    if not success:
        FAILED_LOGINS.labels(source_ip=source_ip).inc()
```

### 2. Security Alerts
```yaml
- alert: SuspiciousLoginActivity
  expr: rate(failed_logins_total[5m]) > 5
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "High rate of failed logins detected"
```

## Cost Monitoring

### 1. Resource Cost Tracking
```python
def calculate_experiment_cost(experiment):
    """Calculate cost of running an experiment."""
    robot_time_cost = experiment.duration * ROBOT_HOURLY_RATE
    reagent_cost = sum(reagent.cost for reagent in experiment.reagents)
    instrument_cost = experiment.analysis_time * INSTRUMENT_HOURLY_RATE
    
    total_cost = robot_time_cost + reagent_cost + instrument_cost
    
    EXPERIMENT_COST.observe(total_cost)
    return total_cost
```

### 2. ROI Tracking
```python
def track_roi_metrics(campaign):
    """Track return on investment for campaigns."""
    total_cost = sum(exp.cost for exp in campaign.experiments)
    value_created = estimate_discovery_value(campaign.best_materials)
    roi = (value_created - total_cost) / total_cost
    
    CAMPAIGN_ROI.labels(campaign_id=campaign.id).set(roi)
```

For detailed monitoring setup and configuration examples, see the configuration files in the `monitoring/` directory.