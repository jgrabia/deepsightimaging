#!/usr/bin/env python3
"""
DeepSight Imaging AI - Bridge Installer
Easy installation script for the DICOM folder bridge
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        "requests",
        "watchdog", 
        "pydicom"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "dicom_incoming",
        "dicom_processed", 
        "dicom_errors"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_config():
    """Create configuration file"""
    print("\n⚙️ Creating configuration file...")
    
    config = {
        "watch_folder": "./dicom_incoming",
        "processed_folder": "./dicom_processed",
        "error_folder": "./dicom_errors",
        "api_base_url": "https://api.deepsightimaging.ai",
        "api_token": "YOUR_API_TOKEN_HERE",
        "customer_id": "YOUR_CUSTOMER_ID_HERE",
        "max_file_size_mb": 500,
        "retry_attempts": 3,
        "retry_delay_seconds": 30,
        "scan_interval_seconds": 60,
        "auto_cleanup_days": 30,
        "log_level": "INFO"
    }
    
    with open("folder_bridge_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: folder_bridge_config.json")
    print("⚠️  Please edit the configuration file with your API credentials")
    
    return True

def create_start_script():
    """Create start script for different platforms"""
    print("\n🚀 Creating start script...")
    
    if platform.system() == "Windows":
        # Create Windows batch file
        batch_content = """@echo off
echo Starting DeepSight Imaging AI - DICOM Folder Bridge
python dicom_folder_bridge.py
pause
"""
        with open("start_bridge.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created start script: start_bridge.bat")
        
        # Create Windows service script
        service_content = """@echo off
echo Installing DeepSight DICOM Bridge as Windows Service
python -m pip install pywin32
python install_windows_service.py
echo Service installed. Use "net start DeepSightBridge" to start
pause
"""
        with open("install_service.bat", "w") as f:
            f.write(service_content)
        print("✅ Created service installer: install_service.bat")
        
    else:
        # Create Linux/macOS shell script
        shell_content = """#!/bin/bash
echo "Starting DeepSight Imaging AI - DICOM Folder Bridge"
python3 dicom_folder_bridge.py
"""
        with open("start_bridge.sh", "w") as f:
            f.write(shell_content)
        
        # Make executable
        os.chmod("start_bridge.sh", 0o755)
        print("✅ Created start script: start_bridge.sh")
        
        # Create systemd service file
        service_content = """[Unit]
Description=DeepSight Imaging AI DICOM Bridge
After=network.target

[Service]
Type=simple
User=%i
WorkingDirectory=%h/deepsight-bridge
ExecStart=/usr/bin/python3 %h/deepsight-bridge/dicom_folder_bridge.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        with open("deepsight-bridge.service", "w") as f:
            f.write(service_content)
        print("✅ Created systemd service file: deepsight-bridge.service")
    
    return True

def create_windows_service():
    """Create Windows service installer"""
    if platform.system() != "Windows":
        return True
    
    service_script = '''import win32serviceutil
import win32service
import win32event
import servicemanager
import sys
import os
from dicom_folder_bridge import DICOMFolderBridge

class DeepSightBridgeService(win32serviceutil.ServiceFramework):
    _svc_name_ = "DeepSightBridge"
    _svc_display_name_ = "DeepSight Imaging AI DICOM Bridge"
    _svc_description_ = "Monitors folders for DICOM files and uploads to DeepSight Imaging AI cloud API"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.bridge = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        if self.bridge:
            self.bridge.stop()

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                            servicemanager.PYS_SERVICE_STARTED,
                            (self._svc_name_, ''))
        try:
            self.bridge = DICOMFolderBridge()
            self.bridge.start_watching()
        except Exception as e:
            servicemanager.LogErrorMsg(f"Service error: {e}")

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(DeepSightBridgeService)
'''
    
    with open("install_windows_service.py", "w") as f:
        f.write(service_script)
    
    return True

def create_readme():
    """Create README file"""
    readme_content = """# DeepSight Imaging AI - DICOM Folder Bridge

## Overview
This bridge monitors a local folder for DICOM files and automatically uploads them to the DeepSight Imaging AI cloud API.

## Quick Start

### 1. Configure
Edit `folder_bridge_config.json` with your API credentials:
```json
{
  "api_token": "your_api_token_here",
  "customer_id": "your_customer_id_here"
}
```

### 2. Start Bridge
- **Windows**: Double-click `start_bridge.bat`
- **Linux/macOS**: Run `./start_bridge.sh`

### 3. Place DICOM Files
Copy DICOM files to the `dicom_incoming` folder. They will be automatically processed and uploaded.

## Configuration

### Watch Folder
- Place DICOM files in `./dicom_incoming/`
- Bridge monitors this folder for new files
- Supports subdirectories

### Processed Files
- Successfully uploaded files are moved to `./dicom_processed/`
- Failed files are moved to `./dicom_errors/`

### Network Folders
You can configure the bridge to watch network folders:
```json
{
  "watch_folder": "\\\\server\\share\\dicom_export",
  "processed_folder": "\\\\server\\share\\dicom_processed"
}
```

## MRI Scanner Configuration

### Siemens Skyra 3T
1. Configure DICOM export to save files to the watch folder
2. Set export destination to: `[Bridge Server]\\dicom_incoming`

### GE Discovery MR750
1. Configure DICOM export to save files to the watch folder
2. Set export destination to: `[Bridge Server]\\dicom_incoming`

### Philips Ingenia
1. Configure DICOM export to save files to the watch folder
2. Set export destination to: `[Bridge Server]\\dicom_incoming`

## Running as Service

### Windows
```cmd
install_service.bat
net start DeepSightBridge
```

### Linux
```bash
sudo cp deepsight-bridge.service /etc/systemd/system/
sudo systemctl enable deepsight-bridge
sudo systemctl start deepsight-bridge
```

## Troubleshooting

### Check Logs
- Log file: `dicom_folder_bridge.log`
- Monitor for errors and upload status

### Test API Connection
```bash
python dicom_folder_bridge.py
```

### Verify Files
- Check `dicom_processed/` for successfully uploaded files
- Check `dicom_errors/` for failed uploads

## Support
- Email: support@deepsightimaging.ai
- Documentation: https://docs.deepsightimaging.ai/bridge
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md")
    return True

def main():
    """Main installation function"""
    print("DeepSight Imaging AI - Bridge Installer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Create configuration
    if not create_config():
        return False
    
    # Create start scripts
    if not create_start_script():
        return False
    
    # Create Windows service (if applicable)
    if not create_windows_service():
        return False
    
    # Create README
    if not create_readme():
        return False
    
    print("\n" + "=" * 50)
    print("✅ Installation completed successfully!")
    print("\nNext steps:")
    print("1. Edit folder_bridge_config.json with your API credentials")
    print("2. Start the bridge using the start script")
    print("3. Place DICOM files in the dicom_incoming folder")
    print("\nFor more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

