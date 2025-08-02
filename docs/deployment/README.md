# Deployment Guide

This guide covers deployment options for the Self-Driving Materials Orchestrator in different environments.

## Quick Start Deployment

### Local Development
```bash
# Clone repository
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator

# Start with Docker Compose
docker-compose up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# MongoDB: localhost:27017
```

### Production Deployment
```bash
# Set environment variables
export MONGODB_ROOT_PASSWORD="secure_password"
export API_SECRET_KEY="your-secret-key"
export GRAFANA_ADMIN_PASSWORD="admin_password"

# Deploy production stack
docker-compose -f docker-compose.production.yml up -d

# Access services
# API: https://your-domain.com
# Dashboard: https://your-domain.com/dashboard
# Monitoring: https://your-domain.com/grafana
```

## Deployment Options

### 1. Docker Compose (Recommended for Development)

**Pros:**
- Easy setup and management
- All services preconfigured
- Suitable for single-machine deployments
- Great for development and testing

**Cons:**
- Limited scalability
- Single point of failure
- Not suitable for high-availability production

#### Development Stack
```bash
# Start development environment
docker-compose up -d

# Services included:
# - Materials Orchestrator (port 8000, 8501)
# - MongoDB (port 27017)
# - Redis (port 6379)
# - Prometheus (port 9090)
# - Grafana (port 3000)
```

#### Production Stack
```bash
# Production deployment with SSL and monitoring
docker-compose -f docker-compose.production.yml up -d

# Additional services:
# - Nginx reverse proxy with SSL
# - PostgreSQL for Grafana
# - Automated backups
```

### 2. Kubernetes Deployment

**Pros:**
- High availability and scalability
- Automatic failover and recovery
- Resource management and scheduling
- Suitable for enterprise production

**Cons:**
- Complex setup and management
- Requires Kubernetes expertise
- Higher resource overhead

#### Prerequisites
```bash
# Kubernetes cluster (1.19+)
# kubectl configured
# Helm 3.x installed
```

#### Basic Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace materials-orchestrator

# Deploy using Helm chart
helm install materials-orchestrator ./k8s/helm-chart \
  --namespace materials-orchestrator \
  --values values.production.yaml

# Check deployment
kubectl get pods -n materials-orchestrator
```

### 3. Cloud Platform Deployment

#### AWS ECS Deployment
```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker build -f Dockerfile.production -t materials-orchestrator .
docker tag materials-orchestrator:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/materials-orchestrator:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/materials-orchestrator:latest

# Deploy ECS service
aws ecs create-service --cli-input-json file://aws/ecs-service.json
```

#### Google Cloud Run Deployment
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/materials-orchestrator .
gcloud run deploy materials-orchestrator \
  --image gcr.io/PROJECT-ID/materials-orchestrator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances
```bash
# Create resource group
az group create --name materials-orchestrator --location eastus

# Deploy container
az container create \
  --resource-group materials-orchestrator \
  --name materials-orchestrator \
  --image materials-orchestrator:latest \
  --dns-name-label materials-orchestrator \
  --ports 8000 8501
```

## Environment Configuration

### Development Environment
```bash
# .env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true
SIMULATION_MODE=true
MONGODB_URL=mongodb://localhost:27017/materials_discovery_dev
```

### Staging Environment
```bash
# .env.staging
ENVIRONMENT=staging
LOG_LEVEL=INFO
DEBUG=false
SIMULATION_MODE=false
MONGODB_URL=mongodb://staging-db:27017/materials_discovery_staging
API_SECRET_KEY=staging-secret-key
```

### Production Environment
```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=WARNING
DEBUG=false
SIMULATION_MODE=false
MONGODB_URL=mongodb://prod-cluster:27017/materials_discovery
API_SECRET_KEY=production-secret-key
MONGODB_USERNAME=materials_user
MONGODB_PASSWORD=secure_database_password
REDIS_PASSWORD=secure_redis_password
```

## Security Configuration

### SSL/TLS Setup
```bash
# Generate SSL certificates (Let's Encrypt)
certbot certonly --standalone -d your-domain.com

# Configure Nginx
cp nginx/nginx-ssl.conf nginx/nginx.conf
# Edit nginx.conf with your domain and certificate paths

# Restart with SSL
docker-compose -f docker-compose.production.yml up -d nginx
```

### Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable

# Block direct access to services
sudo ufw deny 8000   # API (should go through nginx)
sudo ufw deny 27017  # MongoDB
sudo ufw deny 6379   # Redis
```

### Authentication Setup
```bash
# Configure API authentication
export API_SECRET_KEY="$(openssl rand -hex 32)"

# Set up MongoDB authentication
mongo --eval "
db.createUser({
  user: 'materials_user',
  pwd: 'secure_password',
  roles: [{ role: 'readWrite', db: 'materials_discovery' }]
})
"
```

## Monitoring and Logging

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'materials-orchestrator'
    static_configs:
      - targets: ['materials-orchestrator:9090']
  
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']
```

### Grafana Dashboards
```bash
# Import pre-built dashboards
curl -X POST \
  http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana-dashboards.json
```

### Log Aggregation
```bash
# Configure log shipping to ELK stack
docker run -d \
  --name filebeat \
  --volume /var/log:/var/log:ro \
  --volume /var/lib/docker/containers:/var/lib/docker/containers:ro \
  elastic/filebeat:7.15.0
```

## Backup and Recovery

### Database Backup
```bash
# Automated MongoDB backup
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mongodump --host mongodb:27017 --db materials_discovery --out $BACKUP_DIR

# Compress backup
tar -czf "${BACKUP_DIR}.tar.gz" $BACKUP_DIR
rm -rf $BACKUP_DIR

# Upload to cloud storage (optional)
aws s3 cp "${BACKUP_DIR}.tar.gz" s3://materials-backups/
```

### Application Data Backup
```bash
# Backup application data
docker run --rm \
  -v materials_app_data:/data \
  -v /host/backup:/backup \
  alpine tar czf /backup/app_data_$(date +%Y%m%d).tar.gz /data
```

### Disaster Recovery
```bash
# Restore from backup
mongorestore --host mongodb:27017 --db materials_discovery /backup/materials_discovery

# Restore application data
docker run --rm \
  -v materials_app_data:/data \
  -v /host/backup:/backup \
  alpine tar xzf /backup/app_data_20250130.tar.gz -C /
```

## Performance Tuning

### Database Optimization
```javascript
// MongoDB indexes for performance
db.experiments.createIndex({ "campaign_id": 1, "timestamp": -1 });
db.experiments.createIndex({ "parameters": 1 });
db.experiments.createIndex({ "results.band_gap": 1 });
db.campaigns.createIndex({ "status": 1, "created_at": -1 });
```

### Application Scaling
```yaml
# Docker Compose scaling
services:
  materials-orchestrator:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats

# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs materials-orchestrator

# Check resource usage
docker system df
docker system prune  # Clean up if needed
```

#### Database Connection Issues
```bash
# Test MongoDB connection
docker exec -it mongodb mongo --eval "db.adminCommand('ping')"

# Check network connectivity
docker network inspect materials-network
```

#### Performance Issues
```bash
# Check resource limits
docker stats --no-stream

# Monitor application metrics
curl http://localhost:9090/metrics
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in /path/to/cert.pem -text -noout

# Renew Let's Encrypt certificate
certbot renew --nginx
```

### Health Checks

#### API Health Check
```bash
curl -f http://localhost:8000/health || echo "API unhealthy"
```

#### Database Health Check
```bash
docker exec mongodb mongo --eval "db.runCommand('ping')" || echo "Database unhealthy"
```

#### Complete System Health Check
```bash
#!/bin/bash
# scripts/health-check.sh

echo "Checking system health..."

# Check containers
if ! docker-compose ps | grep -q "Up"; then
    echo "ERROR: Some containers are not running"
    exit 1
fi

# Check API
if ! curl -sf http://localhost:8000/health > /dev/null; then
    echo "ERROR: API is not responding"
    exit 1
fi

# Check database
if ! docker exec mongodb mongo --quiet --eval "db.adminCommand('ping')" > /dev/null; then
    echo "ERROR: Database is not responding"
    exit 1
fi

echo "System is healthy!"
```

## Maintenance

### Regular Maintenance Tasks
```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Clean up old images
docker image prune -f

# Backup before updates
./scripts/backup.sh

# Check logs for errors
docker-compose logs --tail=100 | grep ERROR
```

### Security Updates
```bash
# Update base images
docker pull python:3.11-slim
docker pull mongo:5.0
docker pull redis:7-alpine

# Rebuild with updated base images
docker-compose build --no-cache
```

### Performance Monitoring
```bash
# Weekly performance report
docker exec prometheus promtool query instant \
  'rate(http_requests_total[7d])'

# Database performance metrics
docker exec mongodb mongo --eval "
  db.runCommand({serverStatus: 1}).metrics
"
```

For more detailed deployment scenarios and advanced configurations, see the specific deployment guides in the `docs/deployment/` directory.