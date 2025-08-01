#!/bin/bash
set -e

# Docker build script for Materials Orchestrator
# Usage: ./scripts/build-docker.sh [production|development|jupyter]

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get build target (default: production)
BUILD_TARGET=${1:-production}

# Get version information
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "dev")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Image names
BASE_IMAGE="materials-orchestrator"
DEV_IMAGE="${BASE_IMAGE}:dev"
PROD_IMAGE="${BASE_IMAGE}:${VERSION}"
LATEST_IMAGE="${BASE_IMAGE}:latest"
JUPYTER_IMAGE="${BASE_IMAGE}-jupyter:latest"

print_status "Building Materials Orchestrator Docker images"
print_status "Version: $VERSION"
print_status "Build Date: $BUILD_DATE"
print_status "VCS Ref: $VCS_REF"
print_status "Target: $BUILD_TARGET"

case $BUILD_TARGET in
    "production"|"prod")
        print_status "Building production image..."
        
        # Build production image
        docker build \
            --file Dockerfile.production \
            --target runtime \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VERSION="$VERSION" \
            --build-arg VCS_REF="$VCS_REF" \
            --tag "$PROD_IMAGE" \
            --tag "$LATEST_IMAGE" \
            .
        
        print_success "Production image built: $PROD_IMAGE"
        
        # Test the image
        print_status "Testing production image..."
        if docker run --rm "$PROD_IMAGE" python -c "import materials_orchestrator; print('✅ Import successful')"; then
            print_success "Production image test passed"
        else
            print_error "Production image test failed"
            exit 1
        fi
        ;;
        
    "development"|"dev")
        print_status "Building development image..."
        
        # Build development image
        docker build \
            --file Dockerfile.production \
            --target development \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VERSION="$VERSION" \
            --build-arg VCS_REF="$VCS_REF" \
            --tag "$DEV_IMAGE" \
            .
        
        print_success "Development image built: $DEV_IMAGE"
        
        # Test the image
        print_status "Testing development image..."
        if docker run --rm "$DEV_IMAGE" python -c "import materials_orchestrator; print('✅ Import successful')"; then
            print_success "Development image test passed"
        else
            print_error "Development image test failed"
            exit 1
        fi
        ;;
        
    "jupyter")
        print_status "Building Jupyter analysis image..."
        
        # Build Jupyter image
        docker build \
            --file Dockerfile.jupyter \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VERSION="$VERSION" \
            --build-arg VCS_REF="$VCS_REF" \
            --tag "$JUPYTER_IMAGE" \
            .
        
        print_success "Jupyter image built: $JUPYTER_IMAGE"
        
        # Test the image
        print_status "Testing Jupyter image..."
        if docker run --rm "$JUPYTER_IMAGE" python -c "import materials_orchestrator; print('✅ Import successful')"; then
            print_success "Jupyter image test passed"
        else
            print_error "Jupyter image test failed"
            exit 1
        fi
        ;;
        
    "all")
        print_status "Building all images..."
        
        # Build production image
        print_status "1/3 Building production image..."
        docker build \
            --file Dockerfile.production \
            --target runtime \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VERSION="$VERSION" \
            --build-arg VCS_REF="$VCS_REF" \
            --tag "$PROD_IMAGE" \
            --tag "$LATEST_IMAGE" \
            .
        
        # Build development image
        print_status "2/3 Building development image..."
        docker build \
            --file Dockerfile.production \
            --target development \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VERSION="$VERSION" \
            --build-arg VCS_REF="$VCS_REF" \
            --tag "$DEV_IMAGE" \
            .
        
        # Build Jupyter image
        print_status "3/3 Building Jupyter image..."
        docker build \
            --file Dockerfile.jupyter \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VERSION="$VERSION" \
            --build-arg VCS_REF="$VCS_REF" \
            --tag "$JUPYTER_IMAGE" \
            .
        
        print_success "All images built successfully"
        ;;
        
    *)
        print_error "Unknown build target: $BUILD_TARGET"
        print_status "Available targets: production, development, jupyter, all"
        exit 1
        ;;
esac

# Show built images
print_status "Built images:"
docker images "$BASE_IMAGE*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Clean up build cache (optional)
if [[ "${CLEAN_BUILD_CACHE:-false}" == "true" ]]; then
    print_status "Cleaning build cache..."
    docker builder prune -f
    print_success "Build cache cleaned"
fi

print_success "Docker build completed successfully!"

# Optional: push to registry
if [[ "${PUSH_TO_REGISTRY:-false}" == "true" ]] && [[ -n "${DOCKER_REGISTRY}" ]]; then
    print_status "Pushing images to registry: $DOCKER_REGISTRY"
    
    case $BUILD_TARGET in
        "production"|"prod"|"all")
            docker tag "$PROD_IMAGE" "$DOCKER_REGISTRY/$PROD_IMAGE"
            docker tag "$LATEST_IMAGE" "$DOCKER_REGISTRY/$LATEST_IMAGE"
            docker push "$DOCKER_REGISTRY/$PROD_IMAGE"
            docker push "$DOCKER_REGISTRY/$LATEST_IMAGE"
            ;;
    esac
    
    if [[ "$BUILD_TARGET" == "development" || "$BUILD_TARGET" == "all" ]]; then
        docker tag "$DEV_IMAGE" "$DOCKER_REGISTRY/$DEV_IMAGE"
        docker push "$DOCKER_REGISTRY/$DEV_IMAGE"
    fi
    
    if [[ "$BUILD_TARGET" == "jupyter" || "$BUILD_TARGET" == "all" ]]; then
        docker tag "$JUPYTER_IMAGE" "$DOCKER_REGISTRY/$JUPYTER_IMAGE"
        docker push "$DOCKER_REGISTRY/$JUPYTER_IMAGE"
    fi
    
    print_success "Images pushed to registry"
fi