#!/usr/bin/env python3
"""
Setup script for DICOM Viewer API on AWS server
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and print the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} - Failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up DICOM Viewer API on AWS Server...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('dicom_viewer_api.py'):
        print("❌ dicom_viewer_api.py not found in current directory!")
        print("   Please run this script from the directory containing dicom_viewer_api.py")
        sys.exit(1)
    
    # Install required packages
    packages = [
        'fastapi',
        'uvicorn[standard]',
        'python-multipart',
        'pillow',
        'opencv-python',
        'pydicom',
        'numpy',
        'pandas'
    ]
    
    print("\n📦 Installing required packages...")
    for package in packages:
        if not run_command(f"pip install {package}", f"Install {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Create systemd service file
    service_content = """[Unit]
Description=DeepSight DICOM Viewer API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/mri_app
Environment=PATH=/home/ubuntu/mri_app/venv/bin
ExecStart=/home/ubuntu/mri_app/venv/bin/uvicorn dicom_viewer_api:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    print("\n🔧 Creating systemd service...")
    try:
        with open('/tmp/deepsight-dicom-api.service', 'w') as f:
            f.write(service_content)
        
        run_command("sudo mv /tmp/deepsight-dicom-api.service /etc/systemd/system/", "Move service file")
        run_command("sudo systemctl daemon-reload", "Reload systemd")
        run_command("sudo systemctl enable deepsight-dicom-api", "Enable service")
        run_command("sudo systemctl start deepsight-dicom-api", "Start service")
        
    except Exception as e:
        print(f"❌ Error creating service: {e}")
    
    # Create nginx configuration
    nginx_config = """server {
    listen 80;
    server_name your-aws-server-ip;
    
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
"""
    
    print("\n🌐 Creating nginx configuration...")
    try:
        with open('/tmp/deepsight-dicom-api.conf', 'w') as f:
            f.write(nginx_config)
        
        run_command("sudo mv /tmp/deepsight-dicom-api.conf /etc/nginx/sites-available/", "Move nginx config")
        run_command("sudo ln -sf /etc/nginx/sites-available/deepsight-dicom-api.conf /etc/nginx/sites-enabled/", "Enable nginx site")
        run_command("sudo nginx -t", "Test nginx configuration")
        run_command("sudo systemctl reload nginx", "Reload nginx")
        
    except Exception as e:
        print(f"❌ Error configuring nginx: {e}")
    
    # Check service status
    print("\n📊 Checking service status...")
    run_command("sudo systemctl status deepsight-dicom-api", "Check API service status")
    run_command("sudo systemctl status nginx", "Check nginx status")
    
    print("\n" + "=" * 60)
    print("🎉 DICOM Viewer API setup completed!")
    print("\n📋 Next steps:")
    print("1. Update your React app to use the AWS API URL")
    print("2. Test the API endpoints:")
    print("   - http://your-aws-server-ip/api/dicom/upload")
    print("   - http://your-aws-server-ip/api/dicom/slice/0")
    print("3. Update Render.com environment variables:")
    print("   - REACT_APP_AWS_API_URL=http://your-aws-server-ip")
    print("\n🔧 Useful commands:")
    print("   - sudo systemctl status deepsight-dicom-api")
    print("   - sudo systemctl restart deepsight-dicom-api")
    print("   - sudo journalctl -u deepsight-dicom-api -f")

if __name__ == "__main__":
    main()
