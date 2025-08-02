#!/bin/bash

# Self-Driving Materials Orchestrator Build Script
# Builds Docker images for different environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
VCS_REF="$(git rev-parse HEAD)"
VERSION="${VERSION:-$(git describe --tags --always --dirty)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [TARGET]

Build Docker images for Self-Driving Materials Orchestrator

TARGETS:
    development     Build development image (default)
    production      Build production image with multi-stage
    jupyter         Build Jupyter analysis image
    all             Build all images

OPTIONS:
    -h, --help      Show this help message
    -v, --version   Specify version tag (default: git describe)
    -t, --tag       Additional tag for the image
    --no-cache      Build without using cache
    --push          Push images to registry after building
    --registry      Registry to push to (default: docker.io)
    --parallel      Build images in parallel
    --platform      Target platform (e.g., linux/amd64,linux/arm64)

EXAMPLES:
    $0                          # Build development image
    $0 production               # Build production image
    $0 all --push              # Build all images and push to registry
    $0 production -v 1.0.0      # Build production with specific version

EOF
}

# Parse command line arguments
TARGETS=()
NO_CACHE=""
PUSH_IMAGES=""
REGISTRY="docker.io"
PARALLEL=""
PLATFORM=""
ADDITIONAL_TAGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -t|--tag)
            ADDITIONAL_TAGS+=("$2")
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH_IMAGES="true"
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="true"
            shift
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        development|production|jupyter|all)
            TARGETS+=("$1")
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to development if no target specified
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("development")
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not running"
    exit 1
fi

# Function to build an image
build_image() {
    local target=$1
    local dockerfile=$2
    local image_name=$3
    
    log_info "Building $target image..."
    
    # Prepare build command
    local build_cmd="docker build"
    build_cmd+=" $NO_CACHE"
    build_cmd+=" $PLATFORM"
    build_cmd+=" --build-arg BUILD_DATE='$BUILD_DATE'"
    build_cmd+=" --build-arg VERSION='$VERSION'"
    build_cmd+=" --build-arg VCS_REF='$VCS_REF'"
    build_cmd+=" -f $dockerfile"
    build_cmd+=" -t $image_name:$VERSION"
    build_cmd+=" -t $image_name:latest"
    
    # Add additional tags
    for tag in "${ADDITIONAL_TAGS[@]}"; do
        build_cmd+=" -t $image_name:$tag"
    done
    
    build_cmd+=" ."
    
    log_info "Executing: $build_cmd"
    
    if eval "$build_cmd"; then
        log_success "Successfully built $target image: $image_name:$VERSION"
        
        # Push if requested
        if [[ "$PUSH_IMAGES" == "true" ]]; then
            push_image "$image_name"
        fi
        
        return 0
    else
        log_error "Failed to build $target image"
        return 1
    fi
}

# Function to push an image
push_image() {
    local image_name=$1
    
    log_info "Pushing $image_name to registry..."
    
    # Tag for registry if needed
    if [[ "$REGISTRY" != "docker.io" ]]; then
        docker tag "$image_name:$VERSION" "$REGISTRY/$image_name:$VERSION"
        docker tag "$image_name:latest" "$REGISTRY/$image_name:latest"
        image_name="$REGISTRY/$image_name"
    fi
    
    if docker push "$image_name:$VERSION" && docker push "$image_name:latest"; then
        log_success "Successfully pushed $image_name"
        
        # Push additional tags
        for tag in "${ADDITIONAL_TAGS[@]}"; do
            docker push "$image_name:$tag"
        done
    else
        log_error "Failed to push $image_name"
        return 1
    fi
}

# Function to build development image
build_development() {
    build_image "development" "Dockerfile" "materials-orchestrator"
}

# Function to build production image
build_production() {
    build_image "production" "Dockerfile.production" "materials-orchestrator-prod"
}

# Function to build Jupyter image
build_jupyter() {
    build_image "jupyter" "Dockerfile.jupyter" "materials-orchestrator-jupyter"
}

# Function to run builds in parallel
run_parallel_builds() {
    local pids=()
    
    for target in "${TARGETS[@]}"; do
        case $target in
            development)
                build_development &
                pids+=($!)
                ;;
            production)
                build_production &
                pids+=($!)
                ;;
            jupyter)
                build_jupyter &
                pids+=($!)
                ;;
            all)
                build_development &
                pids+=($!)
                build_production &
                pids+=($!)
                build_jupyter &
                pids+=($!)
                ;;
        esac
    done
    
    # Wait for all builds to complete
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    
    return $failed
}

# Function to run builds sequentially
run_sequential_builds() {
    local failed=0
    
    for target in "${TARGETS[@]}"; do
        case $target in
            development)
                if ! build_development; then
                    failed=1
                fi
                ;;
            production)
                if ! build_production; then
                    failed=1
                fi
                ;;
            jupyter)
                if ! build_jupyter; then
                    failed=1
                fi
                ;;
            all)
                if ! build_development || ! build_production || ! build_jupyter; then
                    failed=1
                fi
                ;;
            *)
                log_error "Unknown target: $target"
                failed=1
                ;;
        esac
    done
    
    return $failed
}

# Main execution
log_info "Starting build process..."
log_info "Version: $VERSION"
log_info "Build Date: $BUILD_DATE"
log_info "VCS Ref: $VCS_REF"
log_info "Targets: ${TARGETS[*]}"

# Pre-build validation
log_info "Running pre-build validation..."

# Check if pyproject.toml exists
if [[ ! -f "pyproject.toml" ]]; then
    log_error "pyproject.toml not found. Are you in the project root?"
    exit 1
fi

# Run tests before building (optional)
if [[ "${RUN_TESTS_BEFORE_BUILD:-false}" == "true" ]]; then
    log_info "Running tests before build..."
    if ! pytest tests/ -x --tb=short; then
        log_error "Tests failed. Aborting build."
        exit 1
    fi
    log_success "Tests passed."
fi

# Execute builds
if [[ "$PARALLEL" == "true" ]]; then
    log_info "Running builds in parallel..."
    if run_parallel_builds; then
        log_success "All builds completed successfully!"
    else
        log_error "Some builds failed!"
        exit 1
    fi
else
    log_info "Running builds sequentially..."
    if run_sequential_builds; then
        log_success "All builds completed successfully!"
    else
        log_error "Some builds failed!"
        exit 1
    fi
fi

# Build summary
log_info "Build Summary:"
docker images --filter "reference=materials-orchestrator*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

log_success "Build process completed!"

# Optional cleanup
if [[ "${CLEANUP_AFTER_BUILD:-false}" == "true" ]]; then
    log_info "Cleaning up build cache..."
    docker builder prune -f
    docker system prune -f
fi