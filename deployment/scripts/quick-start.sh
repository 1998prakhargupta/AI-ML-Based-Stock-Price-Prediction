#!/bin/bash

# =============================================================================
# Quick Start Deployment Script for Price Predictor
# =============================================================================
# This script provides a simple interface to deploy the price predictor system

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

show_usage() {
    cat << EOF
Price Predictor Quick Start Deployment

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  dev          Start development environment
  staging      Start staging environment  
  prod         Start production environment
  stop         Stop all services
  clean        Clean up all resources
  logs         Show service logs
  status       Show deployment status

Options:
  -h, --help   Show this help

Examples:
  $0 dev       # Start development environment
  $0 prod      # Start production environment
  $0 logs api  # Show API service logs
  $0 clean     # Clean up all resources

EOF
}

start_development() {
    print_header "Starting Development Environment"
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Load development environment
    export $(grep -v '^#' "../config/environments/development.env" | xargs)
    
    # Start services
    docker-compose up -d
    
    print_status "Development environment started!"
    print_status "Services available:"
    echo "  - API: http://localhost:8000"
    echo "  - Jupyter: http://localhost:8888"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Adminer: http://localhost:8080"
}

start_production() {
    print_header "Starting Production Environment"
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Load production environment
    export $(grep -v '^#' "../config/environments/production.env" | xargs)
    
    # Start services
    docker-compose -f docker-compose.yml up -d
    
    print_status "Production environment started!"
}

stop_services() {
    print_header "Stopping All Services"
    
    cd "$DEPLOYMENT_DIR/docker"
    docker-compose down
    
    print_status "All services stopped!"
}

clean_resources() {
    print_header "Cleaning Resources"
    
    cd "$DEPLOYMENT_DIR/docker"
    docker-compose down -v
    docker system prune -f
    
    print_status "Resources cleaned!"
}

show_logs() {
    local service="${1:-api}"
    
    cd "$DEPLOYMENT_DIR/docker"
    docker-compose logs -f "$service"
}

show_status() {
    print_header "Deployment Status"
    
    cd "$DEPLOYMENT_DIR/docker"
    docker-compose ps
}

# Main execution
case "${1:-help}" in
    dev|development)
        start_development
        ;;
    staging)
        start_staging
        ;;
    prod|production)
        start_production
        ;;
    stop)
        stop_services
        ;;
    clean)
        clean_resources
        ;;
    logs)
        show_logs "${2:-api}"
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
