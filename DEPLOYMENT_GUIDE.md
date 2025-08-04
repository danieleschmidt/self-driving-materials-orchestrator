# Production Deployment Guide

## Overview

This guide covers deploying the Self-Driving Materials Orchestrator in a production environment using Docker Compose with comprehensive monitoring, security, and backup strategies.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: Minimum 4 cores, 8+ cores recommended
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: Minimum 100GB free space, SSD recommended
- **Network**: Stable internet connection for Docker images

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- OpenSSL (for SSL certificate generation)
- curl (for health checks)

### Installation Commands

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose git openssl curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# CentOS/RHEL
sudo yum install -y docker docker-compose git openssl curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

**Note**: Log out and back in after adding user to docker group.

## Quick Start Deployment

### 1. Clone Repository

```bash
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator
```

### 2. Configure Environment

```bash
# Copy production environment template
cp .env.production .env

# Edit configuration (IMPORTANT: Change all passwords and secrets!)
nano .env
```

### 3. Deploy

```bash
# Run automated deployment
./scripts/deploy.sh
```

The deployment script will:
- Check prerequisites
- Set up directories and permissions
- Generate SSL certificates (self-signed)
- Build Docker images
- Run health checks
- Deploy all services
- Verify deployment

### 4. Access Services

After successful deployment:

- **Main Application**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Manual Deployment

If you prefer manual control over the deployment process:

### 1. Environment Setup

```bash
# Create required directories
mkdir -p data logs configs ssl

# Set permissions
chmod 755 data logs
chmod 644 configs

# Copy environment file
cp .env.production .env
```

### 2. SSL Certificates

```bash
# Generate self-signed certificates (for testing)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
    -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Set permissions
chmod 600 ssl/key.pem
chmod 644 ssl/cert.pem
```

### 3. Build and Deploy

```bash
# Build images
docker-compose -f docker-compose.production.yml build

# Start services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
```

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Security (CHANGE THESE!)
API_SECRET_KEY=your-super-secret-key
MONGODB_ROOT_PASSWORD=secure-mongodb-password
REDIS_PASSWORD=secure-redis-password
GRAFANA_ADMIN_PASSWORD=secure-grafana-password

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_CONCURRENT_EXPERIMENTS=10
MAX_QUEUE_SIZE=100

# Feature Flags
ENABLE_ML_ACCELERATION=true
ENABLE_AUTO_SCALING=true
ENABLE_SECURITY_AUDIT=true

# Paths
DATA_PATH=./data
LOGS_PATH=./logs
CONFIGS_PATH=./configs
```

### SSL/TLS Configuration

For production deployments with HTTPS:

1. **Obtain SSL Certificates**:
   - From a Certificate Authority (Let's Encrypt, etc.)
   - Place `cert.pem` and `key.pem` in `ssl/` directory

2. **Update Nginx Configuration**:
   - Edit `nginx/nginx-prod.conf`
   - Configure SSL settings and redirects

### Database Configuration

**MongoDB**:
- Configured with authentication
- Data persisted in Docker volume
- Connection string: `mongodb://username:password@mongodb:27017/materials_discovery`

**Redis**:
- Password-protected
- Persistent storage enabled
- Used for caching and session storage

**PostgreSQL**:
- Used by Grafana for metadata storage
- Separate from main application data

## Monitoring and Observability

### Prometheus Metrics

Available at `http://localhost:9090`

**Key Metrics**:
- Experiment throughput and success rates
- System resource utilization
- Application health and performance
- Custom business metrics

### Grafana Dashboards

Available at `http://localhost:3000`

**Default Dashboards**:
- Materials Discovery Overview
- System Performance
- Application Health
- Experiment Analytics

**Login**: admin / [configured password]

### Log Management

**Centralized Logging**:
- All containers log to JSON format
- Logs rotated automatically (10MB max, 3 files)
- Stored in `logs/` directory

**View Logs**:
```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker-compose -f docker-compose.production.yml logs -f materials-orchestrator

# Application logs
tail -f logs/application.log
```

## Health Checks and Monitoring

### Built-in Health Checks

All services include health checks:
- HTTP endpoints for web services
- Database connectivity checks
- Custom application health indicators

### Manual Health Check

```bash
# Check all service health
docker-compose -f docker-compose.production.yml ps

# Application health endpoint
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/status
```

### Alerting

**Prometheus Alerting**:
- Configured in `monitoring/alert_rules.yml`
- Covers system and application metrics
- Integrates with email/Slack notifications

**Health Check Script**:
```bash
# Run comprehensive health check
python3 scripts/automation/repo-health-check.py
```

## Backup and Recovery

### Automated Backups

The deployment includes automated backup service:
- Runs daily by default
- Backs up MongoDB data and application files
- Configurable retention (30 days default)
- Stored in `backup_storage` volume

### Manual Backup

```bash
# Create backup
./scripts/backup.sh

# MongoDB backup
docker exec materials-mongodb-prod mongodump --out /backup/$(date +%Y%m%d)

# Application data backup
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/
```

### Recovery

```bash
# Stop services
docker-compose -f docker-compose.production.yml down

# Restore MongoDB
docker exec materials-mongodb-prod mongorestore /backup/20231207

# Restore application data
tar -xzf data-backup-20231207.tar.gz

# Restart services
docker-compose -f docker-compose.production.yml up -d
```

## Security Considerations

### Network Security

- All services run in isolated Docker network
- Only necessary ports exposed to host
- Configure firewall rules:

```bash
# UFW firewall example
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 8000/tcp # API (if needed externally)
sudo ufw enable
```

### Application Security

- Authentication required for all operations
- API key-based access control
- Input validation and sanitization
- Security audit logging enabled
- Regular security updates via Docker images

### Data Security

- Database authentication enabled
- Encrypted connections (SSL/TLS)
- Sensitive data encrypted at rest
- Regular backup encryption
- Access logging and monitoring

### Security Checklist

- [ ] Changed all default passwords
- [ ] Configured proper SSL certificates
- [ ] Set up firewall rules
- [ ] Enabled security audit logging
- [ ] Configured backup encryption
- [ ] Set up monitoring alerts
- [ ] Reviewed user access permissions
- [ ] Enabled automatic security updates

## Scaling and Performance

### Horizontal Scaling

**Load Balancer Configuration**:
- Multiple application instances behind Nginx
- Database connection pooling
- Redis clustering for cache scaling
- Auto-scaling based on queue depth

**Scale Services**:
```bash
# Scale application instances
docker-compose -f docker-compose.production.yml up -d --scale materials-orchestrator=3

# Scale with load balancer
docker-compose -f docker-compose.production.yml -f docker-compose.scale.yml up -d
```

### Performance Tuning

**Application Tuning**:
- Adjust `MAX_CONCURRENT_EXPERIMENTS`
- Configure cache sizes
- Optimize database indexes
- Enable ML acceleration features

**Resource Limits**:
- Set in `docker-compose.production.yml`
- Monitor resource usage via Grafana
- Adjust based on workload patterns

### Database Optimization

**MongoDB**:
- Enable sharding for large datasets
- Configure replica sets for high availability
- Optimize indexes for query patterns
- Regular maintenance and compaction

**Redis**:
- Configure memory policies
- Enable persistence for critical data
- Monitor memory usage and eviction
- Set up clustering if needed

## Troubleshooting

### Common Issues

**Services Won't Start**:
```bash
# Check logs
docker-compose -f docker-compose.production.yml logs

# Check system resources
docker system df
free -h
df -h

# Reset deployment
docker-compose -f docker-compose.production.yml down -v
docker system prune -f
./scripts/deploy.sh
```

**Database Connection Issues**:
```bash
# Check MongoDB status
docker exec materials-mongodb-prod mongo --eval "db.adminCommand('ping')"

# Check Redis status
docker exec materials-redis-prod redis-cli ping

# Test connectivity
docker exec materials-orchestrator-prod python3 -c "import pymongo; print('MongoDB OK')"
```

**Performance Issues**:
```bash
# Monitor resource usage
docker stats

# Check application logs
docker-compose -f docker-compose.production.yml logs materials-orchestrator

# Profile application
curl http://localhost:8000/profile
```

### Debug Mode

```bash
# Enable debug logging
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose -f docker-compose.production.yml restart materials-orchestrator

# Run integration tests
python3 scripts/validate-integration.py

# Health check
python3 scripts/automation/repo-health-check.py
```

### Getting Help

1. **Check Documentation**: Review all README files and documentation
2. **System Logs**: Examine Docker and application logs
3. **Health Checks**: Run built-in diagnostic tools
4. **Community**: Open GitHub issues for bugs or questions
5. **Professional Support**: Contact Terragon Labs for enterprise support

## Maintenance

### Regular Maintenance Tasks

**Daily**:
- Monitor system health via Grafana
- Check backup completion
- Review security alerts

**Weekly**:
- Update Docker images
- Review system logs
- Performance analysis
- Security audit review

**Monthly**:
- Update base system packages
- Review and rotate credentials
- Backup validation
- Capacity planning review

### Update Procedure

```bash
# 1. Backup current deployment
./scripts/backup.sh

# 2. Pull latest code
git pull origin main

# 3. Update images
docker-compose -f docker-compose.production.yml pull

# 4. Restart services with zero downtime
docker-compose -f docker-compose.production.yml up -d --no-deps materials-orchestrator

# 5. Verify deployment
curl http://localhost:8000/health
```

### Monitoring Maintenance

- **Prometheus**: Data retention and storage management
- **Grafana**: Dashboard updates and user management
- **Logs**: Rotation and archival policies
- **Metrics**: Performance baseline updates

## Production Checklist

### Pre-Deployment
- [ ] Hardware requirements met
- [ ] All prerequisites installed
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Firewall rules configured
- [ ] Backup strategy planned

### Post-Deployment
- [ ] All services healthy
- [ ] Application accessible
- [ ] Monitoring dashboards configured
- [ ] Backup system tested
- [ ] Alert notifications working
- [ ] Performance baseline established
- [ ] Security audit completed
- [ ] Documentation updated

### Ongoing Operations
- [ ] Regular health monitoring
- [ ] Automated backups running
- [ ] Security updates applied
- [ ] Performance optimization
- [ ] Capacity planning
- [ ] Incident response procedures

## Support and Resources

- **Documentation**: `/docs` directory
- **GitHub Issues**: https://github.com/terragonlabs/self-driving-materials-orchestrator/issues
- **Community**: Discussion forums and chat
- **Enterprise Support**: Contact Terragon Labs

---

**Note**: This guide covers production deployment. For development setup, see `DEVELOPMENT.md`.
