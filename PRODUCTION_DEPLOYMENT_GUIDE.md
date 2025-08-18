# 🚀 Production Deployment Guide

## Quick Start (5 minutes)

### Option 1: Immediate Deployment
```bash
# Start production services
./start_production.sh

# Verify deployment
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Run example
python examples/perovskite_discovery_example.py
```

### Option 2: Docker Deployment
```bash
# Production Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Check services
docker-compose ps
docker logs materials-orchestrator-api
```

### Option 3: Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check deployment
kubectl get pods -l app=materials-orchestrator
kubectl port-forward svc/materials-orchestrator-api 8000:80
```

## Service Endpoints

- **API**: http://localhost:8000 (Documentation: /docs)
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Dashboard**: http://localhost:8501 (if enabled)
- **Monitoring**: http://localhost:9090 (Prometheus)
- **Analytics**: http://localhost:3000 (Grafana)

## Environment Configuration

### Required Environment Variables
```bash
# Core Configuration
export MATERIALS_ORCHESTRATOR_ENV=production
export API_HOST=0.0.0.0
export API_PORT=8000

# Database (optional - will use file storage as fallback)
export MONGODB_URL=mongodb://localhost:27017/materials
export DATABASE_NAME=materials_discovery

# Security
export JWT_SECRET_KEY=your_secure_key_here
export API_KEY_SALT=your_salt_here

# Monitoring
export PROMETHEUS_PORT=9090
export GRAFANA_PORT=3000

# Performance
export MAX_CONCURRENT_EXPERIMENTS=10
export CACHE_SIZE=10000
export WORKER_THREADS=4
```

### Production Configuration File
Create `.env.production`:
```bash
# Copy template and customize
cp configs/development.env .env.production

# Edit with production values
vim .env.production
```

## Monitoring & Health Checks

### Health Check Endpoints
```bash
# Basic health check
curl http://localhost:8000/health

# Kubernetes readiness probe
curl http://localhost:8000/health/ready

# Kubernetes liveness probe  
curl http://localhost:8000/health/live

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Monitoring Setup
```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Configure Grafana
# 1. Access http://localhost:3000 (admin/admin)
# 2. Add Prometheus datasource: http://prometheus:9090
# 3. Import dashboard from monitoring/grafana-dashboards.json
```

## Security Configuration

### SSL/TLS Setup
```bash
# Generate certificates (or use your CA)
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes

# Update configuration
export SSL_CERT_FILE=certs/cert.pem
export SSL_KEY_FILE=certs/key.pem
export USE_SSL=true
```

### Authentication Setup
```bash
# Generate secure keys
export JWT_SECRET_KEY=$(openssl rand -base64 32)
export API_KEY_SALT=$(openssl rand -base64 16)

# Configure authentication method
export AUTH_METHOD=jwt  # Options: none, jwt, api_key
export TOKEN_EXPIRY_HOURS=24
```

## Scaling Configuration

### Auto-Scaling Setup
```python
# In your deployment script
from materials_orchestrator.auto_scaling import setup_default_scaling_targets

# Configure scaling targets
setup_default_scaling_targets()

# Start auto-scaling engine
from materials_orchestrator.auto_scaling import get_auto_scaling_engine
engine = get_auto_scaling_engine()
engine.start_monitoring()
```

### Performance Tuning
```bash
# CPU-bound workloads
export WORKER_THREADS=8  # Set to number of CPU cores

# Memory optimization
export MAX_MEMORY_MB=4096
export CACHE_SIZE=5000

# Database optimization
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=30
```

## Backup & Recovery

### Data Backup
```bash
# MongoDB backup (if using)
mongodump --uri="$MONGODB_URL" --out=backups/$(date +%Y%m%d)

# File-based backup
tar -czf backups/data-$(date +%Y%m%d).tar.gz data/

# Configuration backup
tar -czf backups/config-$(date +%Y%m%d).tar.gz configs/ .env*
```

### Recovery Procedures
```bash
# Restore from backup
tar -xzf backups/data-YYYYMMDD.tar.gz

# Restart services
./start_production.sh

# Verify system health
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker logs materials-orchestrator-api
tail -f logs/production.log

# Check dependencies
python -c "from materials_orchestrator import AutonomousLab"

# Verify ports
netstat -tlnp | grep 8000
```

#### Performance Issues
```bash
# Check system resources
htop
df -h

# Monitor application metrics
curl http://localhost:8000/metrics | grep response_time

# Check database connection
python scripts/automation/repo-health-check.py
```

#### Database Connectivity
```bash
# Test MongoDB connection
mongo $MONGODB_URL --eval "db.stats()"

# Check file-based storage fallback
ls -la data/
tail -f logs/database.log
```

### Log Locations
- Application logs: `logs/production.log`
- Error logs: `logs/error.log`
- Access logs: `logs/access.log`
- Docker logs: `docker logs <container_name>`

## Maintenance Tasks

### Daily Tasks
- Check system health: `curl http://localhost:8000/health`
- Monitor disk space: `df -h`
- Review error logs: `tail -f logs/error.log`

### Weekly Tasks
- Update security scans: `python scripts/automation/security_scanner.py`
- Collect metrics: `python scripts/automation/metrics_collector.py`
- Backup data: Follow backup procedures above

### Monthly Tasks
- Review performance trends
- Update dependencies
- Security audit
- Capacity planning review

## Support & Contact

### Documentation
- API Documentation: http://localhost:8000/docs
- Architecture: `ARCHITECTURE.md`
- Development Guide: `docs/DEVELOPMENT.md`

### Monitoring Dashboards
- System Health: Grafana dashboard
- Performance Metrics: Prometheus queries
- Application Metrics: `/metrics` endpoint

### Emergency Contacts
- Technical Support: [Your support team]
- On-call Engineer: [On-call contact]
- Escalation: [Management contact]

## Performance Benchmarks

### Expected Performance
- API Response Time: <100ms (95th percentile)
- Experiment Throughput: 50-100 experiments/hour
- Memory Usage: <2GB baseline
- CPU Usage: <50% under normal load

### Load Testing
```bash
# Simple load test
curl -X POST http://localhost:8000/experiment \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"temp": 150, "time": 2}}'

# Batch testing
python scripts/load_test.py --experiments=100 --concurrent=10
```

## Compliance & Governance

### Data Retention
- Experiment data: 7 years
- Logs: 90 days
- Backups: 1 year

### Security Compliance
- Regular security scans
- Vulnerability assessments
- Access audit logs
- Encrypted data transmission

### Change Management
- All changes through version control
- Staged deployment process
- Rollback procedures documented
- Change approval workflows

---

**Ready for Production Deployment!** 🚀

This system has been validated and is ready for enterprise deployment. Follow the quick start guide for immediate deployment or the full configuration sections for production hardening.