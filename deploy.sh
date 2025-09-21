#!/bin/bash

# DeepSight Imaging AI - Deployment Script
# For deploying to deepsightimaging.ai

set -e

echo "🚀 DeepSight Imaging AI - Deployment Script"
echo "============================================="

# Configuration
DOMAIN="deepsightimaging.ai"
APP_NAME="deepsight-imaging-ai"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed ✓"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/{dicom_incoming,dicom_processed,dicom_errors}
    mkdir -p logs
    mkdir -p ssl
    
    print_status "Directories created ✓"
}

# Build and start services
deploy_services() {
    print_status "Building and deploying services..."
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose up -d --build
    
    print_status "Services deployed ✓"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for the main application
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/api/health &> /dev/null; then
            print_status "Application is ready ✓"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Application failed to start within expected time"
            docker-compose logs deepsight-app
            exit 1
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
}

# Setup SSL certificates (Let's Encrypt)
setup_ssl() {
    print_status "Setting up SSL certificates..."
    
    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        print_warning "Certbot is not installed. SSL setup will be skipped."
        print_warning "You can install it later and run: certbot --nginx -d $DOMAIN"
        return
    fi
    
    # Request SSL certificate
    if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
        print_status "Requesting SSL certificate for $DOMAIN..."
        certbot certonly --standalone -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
    fi
    
    print_status "SSL certificates configured ✓"
}

# Configure Nginx
configure_nginx() {
    print_status "Configuring Nginx..."
    
    cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream deepsight_app {
        server deepsight-app:8000;
    }

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=upload:10m rate=5r/s;

    server {
        listen 80;
        server_name $DOMAIN www.$DOMAIN;
        
        # Redirect HTTP to HTTPS
        return 301 https://\$server_name\$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name $DOMAIN www.$DOMAIN;

        # SSL Configuration
        ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://deepsight_app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Upload endpoint with special rate limiting
        location /api/v1/upload {
            limit_req zone=upload burst=5 nodelay;
            client_max_body_size 500M;
            proxy_pass http://deepsight_app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
        }

        # Frontend static files
        location / {
            proxy_pass http://deepsight_app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

    print_status "Nginx configuration created ✓"
}

# Create systemd service for auto-start
create_systemd_service() {
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/deepsight-imaging-ai.service > /dev/null << EOF
[Unit]
Description=DeepSight Imaging AI
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable deepsight-imaging-ai.service
    
    print_status "Systemd service created and enabled ✓"
}

# Show deployment status
show_status() {
    print_status "Deployment Status:"
    echo "==================="
    
    # Show running containers
    docker-compose ps
    
    echo ""
    print_status "Application URLs:"
    echo "  - Main Application: https://$DOMAIN"
    echo "  - API Documentation: https://$DOMAIN/api/docs"
    echo "  - Health Check: https://$DOMAIN/api/health"
    
    echo ""
    print_status "Useful Commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Restart: docker-compose restart"
    echo "  - Stop: docker-compose down"
    echo "  - Update: ./deploy.sh"
}

# Main deployment function
main() {
    print_status "Starting deployment to $DOMAIN..."
    
    check_docker
    create_directories
    configure_nginx
    deploy_services
    wait_for_services
    setup_ssl
    create_systemd_service
    
    print_status "Deployment completed successfully! 🎉"
    show_status
}

# Handle command line arguments
case "${1:-}" in
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        show_status
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "update")
        print_status "Updating services..."
        docker-compose pull
        docker-compose up -d --build
        show_status
        ;;
    *)
        main
        ;;
esac
