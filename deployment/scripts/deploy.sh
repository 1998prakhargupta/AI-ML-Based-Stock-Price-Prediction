#!/bin/bash

# =============================================================================
# Enterprise Stock Price Predictor - Unified Deployment Script
# =============================================================================
# This script consolidates and manages all deployment configurations
# Supports Docker Compose, Kubernetes, and local development setups

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPT_NAME="$(basename "$0")"
VERSION="1.0.0"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_ENVIRONMENT="development"
ENVIRONMENT="${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
CONFIG_DIR="$PROJECT_ROOT/config"
DEPLOYMENTS_DIR="$PROJECT_ROOT/deployments"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
${BLUE}Enterprise Stock Price Predictor - Deployment Manager${NC}

Usage: $SCRIPT_NAME [COMMAND] [OPTIONS]

Commands:
  setup             Setup project structure and dependencies
  docker            Docker deployment commands
  k8s              Kubernetes deployment commands  
  config           Configuration management
  clean            Cleanup deployment resources
  status           Show deployment status
  logs             View application logs
  help             Show this help message

Docker Commands:
  docker build     Build all Docker images
  docker up        Start Docker Compose services
  docker down      Stop Docker Compose services
  docker logs      View Docker container logs
  docker clean     Clean Docker images and volumes

Kubernetes Commands:
  k8s deploy       Deploy to Kubernetes cluster
  k8s delete       Delete Kubernetes resources
  k8s status       Check Kubernetes deployment status
  k8s logs         View Kubernetes pod logs

Config Commands:
  config init      Initialize configuration files
  config validate  Validate configuration files
  config generate  Generate environment-specific configs

Options:
  -e, --env ENV    Environment (development|staging|production) [default: development]
  -f, --force      Force operation without confirmation
  -v, --verbose    Verbose output
  -h, --help       Show this help

Examples:
  $SCRIPT_NAME setup                           # Initial project setup
  $SCRIPT_NAME docker up -e development       # Start development environment
  $SCRIPT_NAME docker up -e production        # Start production environment
  $SCRIPT_NAME k8s deploy -e production       # Deploy to Kubernetes
  $SCRIPT_NAME config validate                # Validate all configurations
  $SCRIPT_NAME logs api                       # View API service logs

Environment Variables:
  ENVIRONMENT      Target environment (development|staging|production)
  DOCKER_REGISTRY  Docker registry for images
  KUBECONFIG      Kubernetes configuration file path

EOF
}

# Function to validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            print_status "Environment: $ENVIRONMENT"
            ;;
        *)
            print_error "Invalid environment: $ENVIRONMENT"
            print_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check for required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
    
    if [[ ${#missing_tools[@]} -ne 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install the missing tools and try again"
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Function to setup project structure
setup_project() {
    print_header "Setting up project structure..."
    
    # Create necessary directories
    local directories=(
        "data/raw"
        "data/processed" 
        "data/cache"
        "data/outputs"
        "data/plots"
        "data/reports"
        "models/experiments"
        "models/checkpoints" 
        "models/production"
        "logs/api"
        "logs/training"
        "logs/pipeline"
        "config/environments"
        "config/monitoring"
        "scripts/sql"
        "notebooks/exploration"
        "notebooks/analysis"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        echo "Created directory: $dir"
    done
    
    # Create .gitkeep files to preserve directory structure
    find "$PROJECT_ROOT" -type d -empty -exec touch {}/.gitkeep \;
    
    # Copy environment template if .env doesn't exist
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        if [[ -f "$PROJECT_ROOT/.env.template" ]]; then
            cp "$PROJECT_ROOT/.env.template" "$PROJECT_ROOT/.env"
            print_warning "Created .env from template. Please update with your actual values."
        fi
    fi
    
    print_success "Project structure setup complete"
}

# Function to initialize configuration
init_config() {
    print_header "Initializing configuration files..."
    
    # Ensure unified configurations are in place
    if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        if [[ -f "$PROJECT_ROOT/docker-compose.unified.yml" ]]; then
            cp "$PROJECT_ROOT/docker-compose.unified.yml" "$PROJECT_ROOT/docker-compose.yml"
            print_status "Created docker-compose.yml from unified template"
        fi
    fi
    
    if [[ ! -f "$PROJECT_ROOT/Dockerfile" ]]; then
        if [[ -f "$PROJECT_ROOT/Dockerfile.unified" ]]; then
            cp "$PROJECT_ROOT/Dockerfile.unified" "$PROJECT_ROOT/Dockerfile"
            print_status "Created Dockerfile from unified template"
        fi
    fi
    
    # Create Kubernetes directory structure
    mkdir -p "$PROJECT_ROOT/k8s"
    if [[ ! -f "$PROJECT_ROOT/k8s/deployment.yaml" ]] && [[ -f "$PROJECT_ROOT/k8s-unified.yaml" ]]; then
        cp "$PROJECT_ROOT/k8s-unified.yaml" "$PROJECT_ROOT/k8s/deployment.yaml"
        print_status "Created k8s/deployment.yaml from unified template"
    fi
    
    print_success "Configuration initialization complete"
}

# Function to validate configuration files
validate_config() {
    print_header "Validating configuration files..."
    
    local errors=0
    
    # Validate Docker Compose
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" config >/dev/null 2>&1; then
            print_success "docker-compose.yml is valid"
        else
            print_error "docker-compose.yml has syntax errors"
            ((errors++))
        fi
    else
        print_warning "docker-compose.yml not found"
    fi
    
    # Validate Dockerfile
    if [[ -f "$PROJECT_ROOT/Dockerfile" ]]; then
        print_success "Dockerfile exists"
    else
        print_warning "Dockerfile not found"
    fi
    
    # Validate environment file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        print_success ".env file exists"
        
        # Check for placeholder values
        if grep -q "your_.*_here" "$PROJECT_ROOT/.env"; then
            print_warning ".env contains placeholder values - update with real credentials"
        fi
    else
        print_warning ".env file not found"
    fi
    
    # Validate Kubernetes manifests
    if command -v kubectl >/dev/null 2>&1; then
        if [[ -f "$PROJECT_ROOT/k8s/deployment.yaml" ]]; then
            if kubectl apply --dry-run=client -f "$PROJECT_ROOT/k8s/deployment.yaml" >/dev/null 2>&1; then
                print_success "Kubernetes manifests are valid"
            else
                print_error "Kubernetes manifests have errors"
                ((errors++))
            fi
        fi
    fi
    
    if [[ $errors -eq 0 ]]; then
        print_success "All configuration files are valid"
        return 0
    else
        print_error "Found $errors configuration errors"
        return 1
    fi
}

# Docker functions
docker_build() {
    print_header "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    if [[ -f ".env" ]]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    # Build with environment-specific target
    local target="production"
    if [[ "$ENVIRONMENT" == "development" ]]; then
        target="development"
    fi
    
    docker build \
        --target "$target" \
        --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --build-arg VERSION="${VERSION:-1.0.0}" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        -t "price-predictor:$target" \
        -t "price-predictor:${VERSION:-latest}" \
        .
    
    print_success "Docker images built successfully"
}

docker_up() {
    print_header "Starting Docker Compose services..."
    
    cd "$PROJECT_ROOT"
    
    # Use environment-specific compose file if available
    local compose_files=("-f" "docker-compose.yml")
    
    if [[ "$ENVIRONMENT" == "development" ]] && [[ -f "docker-compose.dev.yml" ]]; then
        compose_files+=("-f" "docker-compose.dev.yml")
    elif [[ "$ENVIRONMENT" == "production" ]] && [[ -f "docker-compose.prod.yml" ]]; then
        compose_files+=("-f" "docker-compose.prod.yml")
    fi
    
    # Start services
    docker-compose "${compose_files[@]}" up -d
    
    print_success "Docker services started"
    
    # Show running services
    docker-compose "${compose_files[@]}" ps
}

docker_down() {
    print_header "Stopping Docker Compose services..."
    
    cd "$PROJECT_ROOT"
    docker-compose down
    
    print_success "Docker services stopped"
}

docker_logs() {
    local service="${1:-api}"
    
    cd "$PROJECT_ROOT"
    docker-compose logs -f "$service"
}

docker_clean() {
    print_header "Cleaning Docker resources..."
    
    # Stop and remove containers
    docker-compose down -v --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    print_success "Docker cleanup complete"
}

# Kubernetes functions
k8s_deploy() {
    print_header "Deploying to Kubernetes..."
    
    if ! command -v kubectl >/dev/null 2>&1; then
        print_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Apply Kubernetes manifests
    if [[ -f "k8s/deployment.yaml" ]]; then
        kubectl apply -f k8s/deployment.yaml
        print_success "Kubernetes deployment applied"
    else
        print_error "Kubernetes deployment file not found"
        exit 1
    fi
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/api-deployment -n price-predictor
    
    print_success "Kubernetes deployment complete"
}

k8s_delete() {
    print_header "Deleting Kubernetes resources..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -f "k8s/deployment.yaml" ]]; then
        kubectl delete -f k8s/deployment.yaml
        print_success "Kubernetes resources deleted"
    fi
}

k8s_status() {
    print_header "Kubernetes deployment status..."
    
    if ! command -v kubectl >/dev/null 2>&1; then
        print_error "kubectl not found"
        exit 1
    fi
    
    echo "Namespaces:"
    kubectl get namespaces | grep price-predictor || echo "No price-predictor namespace found"
    
    echo -e "\nPods:"
    kubectl get pods -n price-predictor 2>/dev/null || echo "No pods found"
    
    echo -e "\nServices:"
    kubectl get services -n price-predictor 2>/dev/null || echo "No services found"
    
    echo -e "\nDeployments:"
    kubectl get deployments -n price-predictor 2>/dev/null || echo "No deployments found"
}

k8s_logs() {
    local service="${1:-api}"
    
    kubectl logs -f deployment/"$service"-deployment -n price-predictor
}

# Status function
show_status() {
    print_header "Deployment Status"
    
    echo "Environment: $ENVIRONMENT"
    echo "Project Root: $PROJECT_ROOT"
    
    # Docker status
    if command -v docker >/dev/null 2>&1; then
        echo -e "\n${BLUE}Docker Services:${NC}"
        if docker-compose ps 2>/dev/null | grep -q "Up"; then
            docker-compose ps
        else
            echo "No Docker services running"
        fi
    fi
    
    # Kubernetes status
    if command -v kubectl >/dev/null 2>&1; then
        echo -e "\n${BLUE}Kubernetes Services:${NC}"
        k8s_status
    fi
}

# Main command processing
main() {
    local command="${1:-help}"
    shift || true
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Validate environment
    validate_environment
    
    # Execute command
    case "$command" in
        setup)
            check_prerequisites
            setup_project
            init_config
            ;;
        docker)
            local docker_cmd="${1:-up}"
            case "$docker_cmd" in
                build) docker_build ;;
                up) docker_up ;;
                down) docker_down ;;
                logs) docker_logs "${2:-api}" ;;
                clean) docker_clean ;;
                *) print_error "Unknown docker command: $docker_cmd"; exit 1 ;;
            esac
            ;;
        k8s)
            local k8s_cmd="${1:-deploy}"
            case "$k8s_cmd" in
                deploy) k8s_deploy ;;
                delete) k8s_delete ;;
                status) k8s_status ;;
                logs) k8s_logs "${2:-api}" ;;
                *) print_error "Unknown k8s command: $k8s_cmd"; exit 1 ;;
            esac
            ;;
        config)
            local config_cmd="${1:-validate}"
            case "$config_cmd" in
                init) init_config ;;
                validate) validate_config ;;
                generate) init_config ;;
                *) print_error "Unknown config command: $config_cmd"; exit 1 ;;
            esac
            ;;
        clean)
            docker_clean
            ;;
        status)
            show_status
            ;;
        logs)
            if command -v kubectl >/dev/null 2>&1 && kubectl get namespace price-predictor >/dev/null 2>&1; then
                k8s_logs "${1:-api}"
            else
                docker_logs "${1:-api}"
            fi
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
