# Docker Deployment Guide

## Overview

The Materials Orchestrator provides multiple Docker configurations for different deployment scenarios:

- **Production**: Optimized multi-stage build for production environments
- **Development**: Full development environment with debugging tools
- **Jupyter**: Analysis environment with Jupyter notebooks
- **Standard**: Simple single-stage build for basic deployments

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the full stack
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Start with analysis tools
docker-compose --profile analysis up -d

# Production deployment
docker-compose --profile production up -d
```

### Using Docker Directly

```bash
# Build and run production image
./scripts/build-docker.sh production
docker run -p 8000:8000 -p 8501:8501 materials-orchestrator:latest

# Run development image
./scripts/build-docker.sh development
docker run -p 8000:8000 -p 8501:8501 materials-orchestrator:dev
```

## Image Variants

### Production Image (`Dockerfile.production`)

**Target**: `materials-orchestrator:latest`

Features:
- Multi-stage build for minimal size
- Security-hardened with non-root user
- Optimized Python dependencies
- Health checks included
- Uses Gunicorn for production WSGI serving

```bash
# Build production image
./scripts/build-docker.sh production

# Run with custom configuration
docker run -d \
  --name materials-orchestrator \
  -p 8000:8000 \
  -e MONGODB_URL=mongodb://mongo:27017/materials \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  materials-orchestrator:latest
```

### Development Image

**Target**: `materials-orchestrator:dev`

Features:
- Includes development tools (pytest, black, ruff, mypy)
- Auto-reload enabled
- Debug logging
- Additional debugging utilities

```bash
# Build development image
./scripts/build-docker.sh development

# Run with development features
docker run -d \
  --name materials-orchestrator-dev \
  -p 8000:8000 \
  -e DEBUG=true \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  materials-orchestrator:dev
```

### Jupyter Analysis Image (`Dockerfile.jupyter`)

**Target**: `materials-orchestrator-jupyter:latest`

Features:
- JupyterLab with scientific Python stack
- Pre-installed analysis packages
- Materials orchestrator package included
- Jupyter extensions for visualization

```bash
# Build Jupyter image
./scripts/build-docker.sh jupyter

# Run Jupyter environment
docker run -d \
  --name materials-jupyter \
  -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secure-token \
  -v $(pwd)/notebooks:/home/jovyan/work/notebooks \
  -v $(pwd)/data:/home/jovyan/work/data:ro \
  materials-orchestrator-jupyter:latest
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017/materials_discovery` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEBUG` | Enable debug mode | `false` |
| `ENVIRONMENT` | Deployment environment | `production` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `DASHBOARD_PORT` | Dashboard port | `8501` |
| `WORKERS` | Number of worker processes | `4` |

### Volume Mounts

| Container Path | Purpose | Recommended Host Path |
|----------------|---------|----------------------|
| `/app/data` | Experiment data storage | `./data` |
| `/app/logs` | Application logs | `./logs` |
| `/app/configs` | Configuration files | `./configs` |
| `/app/backups` | Database backups | `./backups` |

### Port Mapping

| Port | Service | Description |
|------|---------|-------------|
| 8000 | API Server | REST API and main application |
| 8501 | Dashboard | Streamlit dashboard interface |
| 27017 | MongoDB | Database (if using container) |
| 6379 | Redis | Cache and task queue |
| 9090 | Prometheus | Metrics collection |
| 3000 | Grafana | Monitoring dashboard |
| 8888 | Jupyter | Analysis notebooks |

## Docker Compose Profiles

### Base Profile (Default)
Includes essential services:
- Materials Orchestrator
- MongoDB
- Redis

```bash
docker-compose up -d
```

### Monitoring Profile
Adds monitoring stack:
- Prometheus
- Grafana

```bash
docker-compose --profile monitoring up -d
```

### Analysis Profile
Adds analysis tools:
- Jupyter notebooks

```bash
docker-compose --profile analysis up -d
```

### Production Profile
Adds production features:
- Nginx reverse proxy
- SSL termination
- Enhanced security

```bash
docker-compose --profile production up -d
```

## Security Considerations

### Image Security

1. **Base Image**: Uses official Python slim images
2. **Non-root User**: Runs as unprivileged `materials` user
3. **Minimal Attack Surface**: Multi-stage builds reduce image size
4. **Dependency Scanning**: Regular vulnerability scans with Trivy
5. **Security Scanning**: Automated security checks in CI/CD

### Network Security

```yaml
# Custom network configuration
networks:
  materials-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
```

### Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "your-mongodb-password" | docker secret create mongodb_password -

# Reference in compose file
services:
  mongodb:
    secrets:
      - mongodb_password
    environment:
      - MONGO_INITDB_ROOT_PASSWORD_FILE=/run/secrets/mongodb_password
```

## Health Checks

All images include comprehensive health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

Monitor health status:
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Get detailed health info
docker inspect --format='{{json .State.Health}}' materials-orchestrator
```

## Performance Optimization

### Resource Limits

```yaml
services:
  materials-orchestrator:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Build Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with cache from registry
docker build --cache-from materials-orchestrator:latest .

# Multi-platform builds
docker buildx build --platform linux/amd64,linux/arm64 .
```

## Monitoring and Logging

### Container Logs

```bash
# View real-time logs
docker-compose logs -f materials-orchestrator

# View logs from all services
docker-compose logs -f

# Export logs
docker logs materials-orchestrator > app.log 2>&1
```

### Metrics Collection

Prometheus metrics are automatically exposed at `/metrics`:

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics
```

### Log Aggregation

Configure centralized logging:

```yaml
services:
  materials-orchestrator:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker logs materials-orchestrator

# Check port conflicts
netstat -tulpn | grep :8000

# Verify environment variables
docker exec materials-orchestrator env
```

#### Database Connection Issues
```bash
# Test MongoDB connectivity
docker exec materials-orchestrator python -c "import pymongo; pymongo.MongoClient('mongodb://mongodb:27017').admin.command('ping')"

# Check network connectivity
docker exec materials-orchestrator ping mongodb
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats materials-orchestrator

# Check memory usage
docker exec materials-orchestrator free -h

# Monitor disk usage
docker exec materials-orchestrator df -h
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Run with debug logging
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG materials-orchestrator:latest

# Access container shell
docker exec -it materials-orchestrator /bin/bash

# Run specific commands
docker exec materials-orchestrator python -m materials_orchestrator.cli status
```

## Backup and Recovery

### Data Backup

```bash
# Backup MongoDB data
docker exec mongodb mongodump --out /backup --db materials_discovery

# Backup application data
docker run --rm -v materials_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .
```

### Disaster Recovery

```bash
# Restore from backup
docker run --rm -v materials_data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data

# Restore MongoDB
docker exec mongodb mongorestore /backup
```

## Scaling and Load Balancing

### Horizontal Scaling

```yaml
services:
  materials-orchestrator:
    deploy:
      replicas: 3
    depends_on:
      - mongodb
      - redis
```

### Load Balancer Configuration

```nginx
upstream materials_app {
    server materials-orchestrator-1:8000;
    server materials-orchestrator-2:8000;
    server materials-orchestrator-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://materials_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## CI/CD Integration

### Build Pipeline

```yaml
# GitHub Actions example
- name: Build Docker image
  run: |
    ./scripts/build-docker.sh production
    ./scripts/security-scan.sh materials-orchestrator:latest

- name: Push to registry
  run: |
    echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
    docker push materials-orchestrator:latest
```

### Automated Deployment

```bash
# Blue-green deployment
docker-compose -f docker-compose.blue.yml up -d
# Test new version
# Switch traffic
docker-compose -f docker-compose.green.yml down
```

## Best Practices

1. **Use specific image tags** in production (not `latest`)
2. **Implement health checks** for all services
3. **Use multi-stage builds** to minimize image size
4. **Run security scans** regularly
5. **Monitor resource usage** and set appropriate limits
6. **Backup data regularly** and test recovery procedures
7. **Use secrets management** for sensitive configuration
8. **Implement proper logging** and monitoring
9. **Test deployments** in staging environments first
10. **Document any custom configurations**