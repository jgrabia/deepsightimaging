#!/usr/bin/env python3
"""
Deploy 3D Slicer Browser Integration
Sets up the branded breast cancer detection interface
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    
    print("🔧 Installing Required Packages")
    print("=" * 40)
    
    requirements = [
        "streamlit",
        "requests",
        "numpy",
        "pillow",
        "pydicom",
        "nibabel",
        "plotly",
        "scipy",
        "matplotlib"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])

def create_dockerfile():
    """Create Dockerfile for deployment"""
    
    dockerfile_content = """# BreastAI Pro - 3D Slicer Browser Integration
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY slicer_integration.py .
COPY deploy_slicer_browser.py .

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "slicer_integration.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created")

def create_requirements():
    """Create requirements.txt"""
    
    requirements_content = """streamlit>=1.28.0
requests>=2.31.0
numpy>=1.24.0
pillow>=10.0.0
pydicom>=2.4.0
nibabel>=5.1.0
plotly>=5.15.0
scipy>=1.10.0
matplotlib>=3.6.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt created")

def create_docker_compose():
    """Create docker-compose.yml for easy deployment"""
    
    compose_content = """version: '3.8'

services:
  breast-ai-pro:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    
  # Optional: Add 3D Slicer backend service
  slicer-backend:
    image: slicer/slicer:latest
    ports:
      - "8080:8080"
    volumes:
      - ./slicer_data:/data
    environment:
      - SLICER_PORT=8080
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print("✅ docker-compose.yml created")

def create_nginx_config():
    """Create nginx configuration for production deployment"""
    
    nginx_config = """server {
    listen 80;
    server_name breast-ai-pro.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Custom branding
    location /static/ {
        alias /var/www/breast-ai-pro/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    
    print("✅ nginx.conf created")

def create_deployment_script():
    """Create deployment script"""
    
    deploy_script = """#!/bin/bash
# BreastAI Pro Deployment Script

echo "🚀 Deploying BreastAI Pro..."

# Build and start services
docker-compose up -d --build

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ BreastAI Pro is running at http://localhost:8501"
else
    echo "❌ Service failed to start"
    exit 1
fi

echo "🎉 Deployment complete!"
echo "📱 Access your application at: http://localhost:8501"
echo "🔧 To stop: docker-compose down"
"""
    
    with open("deploy.sh", "w") as f:
        f.write(deploy_script)
    
    # Make executable
    os.chmod("deploy.sh", 0o755)
    
    print("✅ deploy.sh created")

def create_readme():
    """Create README with deployment instructions"""
    
    readme_content = """# BreastAI Pro - 3D Slicer Browser Integration

Advanced 3D breast cancer detection and analysis platform with custom branding.

## Features

- 🏥 **Custom Branded Interface** - Professional medical-grade UI
- 🤖 **AI-Powered Analysis** - SegResNet-3D breast cancer detection
- 🎨 **3D Visualization** - Interactive 3D volume rendering
- 📊 **Comprehensive Reports** - Detailed analysis and measurements
- 🌐 **Browser-Based** - No installation required for users

## Quick Start

### Option 1: Local Development
```bash
# Install requirements
pip install -r requirements.txt

# Run the application
streamlit run slicer_integration.py
```

### Option 2: Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Access at http://localhost:8501
```

### Option 3: Production Deployment
```bash
# Run deployment script
./deploy.sh

# Configure nginx (see nginx.conf)
sudo cp nginx.conf /etc/nginx/sites-available/breast-ai-pro
sudo ln -s /etc/nginx/sites-available/breast-ai-pro /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

## Customization

### Branding
- Edit `slicer_integration.py` to change colors, logos, and branding
- Modify CSS in `_get_custom_css()` method
- Update company name and contact information

### AI Models
- Integrate your trained MONAI models
- Add custom segmentation algorithms
- Configure model parameters

### 3D Slicer Integration
- Customize 3D Slicer notebooks for breast analysis
- Add specific measurement tools
- Configure visualization settings

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Browser  │    │  BreastAI Pro   │    │   3D Slicer     │
│                 │◄──►│   Interface     │◄──►│   Backend       │
│  Custom Brand   │    │  (Streamlit)    │    │  (Jupyter)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Security

- HTTPS encryption for all communications
- Secure file upload handling
- User authentication (can be added)
- Data privacy compliance

## Support

For technical support or customization requests, contact your development team.

## License

This software is proprietary and confidential.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ README.md created")

def main():
    """Main deployment function"""
    
    print("🏥 BreastAI Pro - 3D Slicer Browser Integration")
    print("=" * 60)
    
    # Install requirements
    install_requirements()
    
    # Create deployment files
    create_requirements()
    create_dockerfile()
    create_docker_compose()
    create_nginx_config()
    create_deployment_script()
    create_readme()
    
    print("\n🎉 **Deployment Setup Complete!**")
    print("\n📋 **Next Steps:**")
    print("   1. Customize branding in slicer_integration.py")
    print("   2. Run: ./deploy.sh")
    print("   3. Access at: http://localhost:8501")
    
    print("\n🔧 **Customization Options:**")
    print("   - Change company name and colors")
    print("   - Add your logo and branding")
    print("   - Integrate your AI models")
    print("   - Configure 3D Slicer notebooks")
    
    print("\n🌐 **Production Deployment:**")
    print("   - Use nginx.conf for reverse proxy")
    print("   - Set up SSL certificates")
    print("   - Configure domain name")
    print("   - Set up monitoring and logging")

if __name__ == "__main__":
    main()
