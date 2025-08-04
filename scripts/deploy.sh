#!/bin/bash
# Production deployment script for Materials Orchestrator

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.production.yml"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended for production deployments."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create environment file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating environment file from template..."
        cp "$PROJECT_ROOT/.env.production" "$ENV_FILE"
        log_warning "Please edit $ENV_FILE with your production values before continuing."
        log_warning "Especially change all passwords and secret keys!"
        read -p "Press Enter when you've configured the environment file..."
    fi
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/configs"
    mkdir -p "$PROJECT_ROOT/ssl"
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/data"
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 644 "$PROJECT_ROOT/configs" 2>/dev/null || true
    
    log_success "Environment setup complete"
}

generate_ssl_certificates() {
    log_info "Checking SSL certificates..."
    
    SSL_DIR="$PROJECT_ROOT/ssl"
    CERT_FILE="$SSL_DIR/cert.pem"
    KEY_FILE="$SSL_DIR/key.pem"
    
    if [[ ! -f "$CERT_FILE" ]] || [[ ! -f "$KEY_FILE" ]]; then
        log_info "Generating self-signed SSL certificates..."
        
        mkdir -p "$SSL_DIR"
        
        # Generate self-signed certificate
        openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            2>/dev/null || {
            log_warning "OpenSSL not available. SSL certificates not generated."
            log_warning "You may need to provide your own certificates for HTTPS."
            return 0
        }
        
        chmod 600 "$KEY_FILE"
        chmod 644 "$CERT_FILE"
        
        log_success "Self-signed SSL certificates generated"
        log_warning "For production, replace with certificates from a trusted CA"
    else
        log_success "SSL certificates already exist"
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Set build arguments
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    export VERSION=$(grep '__version__' src/materials_orchestrator/__init__.py | cut -d'"' -f2 2>/dev/null || echo "0.1.0")
    
    # Build production image
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache materials-orchestrator
    
    log_success "Docker images built successfully"
}

run_health_checks() {
    log_info "Running pre-deployment health checks..."
    
    cd "$PROJECT_ROOT"
    
    # Check configuration
    python3 scripts/validate-integration.py || {
        log_error "Integration validation failed"
        return 1
    }
    
    # Run repository health check
    python3 scripts/automation/repo-health-check.py || {
        log_warning "Repository health check failed, but continuing..."
    }
    
    log_success "Health checks passed"
}

deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "unhealthy\|Exit"; then
        log_error "Some services are not healthy. Check logs with:"
        log_error "docker-compose -f $DOCKER_COMPOSE_FILE logs"
        return 1
    fi
    
    log_success "Services deployed successfully"
}

post_deployment_checks() {
    log_info "Running post-deployment checks..."
    
    # Check if main application is responding
    max_attempts=30
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Application is responding on port 8000"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Application is not responding after $max_attempts attempts"
            return 1
        fi
        
        log_info "Waiting for application to start... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    # Check dashboard
    if curl -f -s http://localhost:8501 > /dev/null 2>&1; then
        log_success "Dashboard is responding on port 8501"
    else
        log_warning "Dashboard is not responding on port 8501"
    fi
    
    # Check monitoring
    if curl -f -s http://localhost:9090 > /dev/null 2>&1; then
        log_success "Prometheus is responding on port 9090"
    else
        log_warning "Prometheus is not responding on port 9090"
    fi
    
    if curl -f -s http://localhost:3000 > /dev/null 2>&1; then
        log_success "Grafana is responding on port 3000"
    else
        log_warning "Grafana is not responding on port 3000"
    fi
    
    log_success "Post-deployment checks completed"
}

show_deployment_info() {
    log_success "\n🎉 Deployment completed successfully!"
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Materials Orchestrator Production Deployment${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo
    echo -e "${BLUE}🔬 Main Application:${NC} http://localhost:8000"
    echo -e "${BLUE}📊 Dashboard:${NC} http://localhost:8501"
    echo -e "${BLUE}📈 Prometheus:${NC} http://localhost:9090"
    echo -e "${BLUE}📊 Grafana:${NC} http://localhost:3000"
    echo
    echo -e "${YELLOW}Default Credentials:${NC}"
    echo -e "  Grafana: admin / [check .env file]"
    echo
    echo -e "${YELLOW}Useful Commands:${NC}"
    echo -e "  View logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo -e "  Stop services: docker-compose -f $DOCKER_COMPOSE_FILE down"
    echo -e "  Restart services: docker-compose -f $DOCKER_COMPOSE_FILE restart"
    echo -e "  Update images: docker-compose -f $DOCKER_COMPOSE_FILE pull && docker-compose -f $DOCKER_COMPOSE_FILE up -d"
    echo
    echo -e "${RED}⚠️  Security Reminders:${NC}"
    echo -e "  1. Change all default passwords in .env file"
    echo -e "  2. Replace self-signed SSL certificates with trusted ones"
    echo -e "  3. Configure firewall rules appropriately"
    echo -e "  4. Set up regular backups"
    echo -e "  5. Monitor logs and system health regularly"
    echo
}

cleanup_on_error() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v 2>/dev/null || true
    exit 1
}

# Main execution
main() {
    log_info "Starting Materials Orchestrator production deployment..."
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Check command line arguments
    case "${1:-}" in
        "--help"|-h)
            echo "Usage: $0 [--skip-build] [--skip-health-checks]"
            echo "Options:"
            echo "  --skip-build         Skip Docker image building"
            echo "  --skip-health-checks Skip pre-deployment health checks"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
    esac
    
    # Execute deployment steps
    check_prerequisites
    setup_environment
    generate_ssl_certificates
    
    if [[ "$*" != *"--skip-build"* ]]; then
        build_images
    fi
    
    if [[ "$*" != *"--skip-health-checks"* ]]; then
        run_health_checks
    fi
    
    deploy_services
    post_deployment_checks
    show_deployment_info
    
    log_success "✅ Production deployment completed successfully!"
}

# Run main function with all arguments
main "$@"
