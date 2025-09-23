#!/usr/bin/env python3
"""
DeepSight Imaging AI - DICOM Folder Bridge
Monitors network folders for DICOM files and uploads them to cloud API
"""

import os
import json
import requests
import time
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional
import shutil

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available. Install with: pip install watchdog")

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. Install with: pip install pydicom")

class DICOMFileHandler(FileSystemEventHandler):
    """Handles file system events for DICOM files"""
    
    def __init__(self, bridge):
        self.bridge = bridge
        self.logger = bridge.logger
    
    def on_created(self, event):
        """Called when a file is created"""
        if not event.is_directory and self.is_dicom_file(event.src_path):
            self.logger.info(f"New DICOM file detected: {event.src_path}")
            # Wait a moment for file to be fully written
            time.sleep(2)
            self.bridge.process_dicom_file(event.src_path)
    
    def on_moved(self, event):
        """Called when a file is moved"""
        if not event.is_directory and self.is_dicom_file(event.dest_path):
            self.logger.info(f"DICOM file moved to: {event.dest_path}")
            time.sleep(2)
            self.bridge.process_dicom_file(event.dest_path)
    
    def is_dicom_file(self, file_path: str) -> bool:
        """Check if file is a DICOM file"""
        if not os.path.exists(file_path):
            return False
        
        # Check file extension
        ext = Path(file_path).suffix.lower()
        if ext in ['.dcm', '.dicom']:
            return True
        
        # Check file content (DICOM files start with specific bytes)
        try:
            with open(file_path, 'rb') as f:
                header = f.read(132)
                # DICOM files start with DICM or have specific byte patterns
                if header.startswith(b'DICM') or (len(header) >= 132 and header[128:132] == b'DICM'):
                    return True
        except:
            pass
        
        return False

class DICOMFolderBridge:
    """Bridge that monitors folders for DICOM files and uploads to cloud API"""
    
    def __init__(self, config_file: str = "folder_bridge_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        
        # Folder Configuration
        self.watch_folder = Path(self.config.get('watch_folder', './dicom_incoming'))
        self.processed_folder = Path(self.config.get('processed_folder', './dicom_processed'))
        self.error_folder = Path(self.config.get('error_folder', './dicom_errors'))
        
        # API Configuration
        self.api_base_url = self.config.get('api_base_url', 'https://api.deepsightimaging.ai')
        self.api_token = self.config.get('api_token')
        self.customer_id = self.config.get('customer_id')
        
        # Processing Configuration
        self.max_file_size = self.config.get('max_file_size_mb', 500) * 1024 * 1024  # Convert to bytes
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay_seconds', 30)
        self.scan_interval = self.config.get('scan_interval_seconds', 60)
        
        # Track processed files to avoid duplicates
        self.processed_files: Set[str] = set()
        self.load_processed_files_log()
        
        # Validate configuration
        self.validate_config()
        
        # Create necessary directories
        self.create_directories()
        
        self.logger.info(f"DICOM Folder Bridge initialized for customer: {self.customer_id}")
        self.logger.info(f"Watching folder: {self.watch_folder}")
    
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = {
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
            
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"Created default configuration file: {self.config_file}")
            print("Please edit the configuration file with your settings")
            
            return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dicom_folder_bridge.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self):
        """Validate configuration settings"""
        if not self.api_token or self.api_token == "YOUR_API_TOKEN_HERE":
            raise ValueError("API token must be configured")
        
        if not self.customer_id or self.customer_id == "YOUR_CUSTOMER_ID_HERE":
            raise ValueError("Customer ID must be configured")
        
        if not os.path.exists(self.watch_folder):
            self.logger.warning(f"Watch folder does not exist: {self.watch_folder}")
    
    def create_directories(self):
        """Create necessary directories"""
        self.watch_folder.mkdir(exist_ok=True)
        self.processed_folder.mkdir(exist_ok=True)
        self.error_folder.mkdir(exist_ok=True)
        
        self.logger.info(f"Created directories: {self.watch_folder}, {self.processed_folder}, {self.error_folder}")
    
    def load_processed_files_log(self):
        """Load log of previously processed files"""
        log_file = Path("processed_files.json")
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                self.logger.info(f"Loaded {len(self.processed_files)} previously processed files")
            except Exception as e:
                self.logger.warning(f"Could not load processed files log: {e}")
    
    def save_processed_files_log(self):
        """Save log of processed files"""
        try:
            log_file = Path("processed_files.json")
            with open(log_file, 'w') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save processed files log: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def extract_dicom_metadata(self, file_path: str) -> Dict:
        """Extract metadata from DICOM file"""
        metadata = {
            'file_path': str(file_path),
            'file_size': os.path.getsize(file_path),
            'upload_timestamp': datetime.now().isoformat()
        }
        
        if PYDICOM_AVAILABLE:
            try:
                ds = pydicom.dcmread(file_path)
                metadata.update({
                    'patient_id': str(ds.get('PatientID', 'Unknown')),
                    'patient_name': str(ds.get('PatientName', 'Unknown')),
                    'study_description': str(ds.get('StudyDescription', 'Unknown')),
                    'series_description': str(ds.get('SeriesDescription', 'Unknown')),
                    'modality': str(ds.get('Modality', 'Unknown')),
                    'study_date': str(ds.get('StudyDate', 'Unknown')),
                    'series_number': str(ds.get('SeriesNumber', 'Unknown')),
                    'instance_number': str(ds.get('InstanceNumber', 'Unknown')),
                    'acquisition_date': str(ds.get('AcquisitionDate', 'Unknown')),
                    'manufacturer': str(ds.get('Manufacturer', 'Unknown')),
                    'model': str(ds.get('ManufacturerModelName', 'Unknown'))
                })
            except Exception as e:
                self.logger.warning(f"Could not read DICOM metadata from {file_path}: {e}")
                metadata['error'] = str(e)
        
        return metadata
    
    def upload_to_cloud_api(self, file_path: str, metadata: Dict) -> bool:
        """Upload DICOM file to cloud API"""
        for attempt in range(self.retry_attempts):
            try:
                # Check file size
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size:
                    self.logger.error(f"File too large: {file_size} bytes > {self.max_file_size} bytes")
                    return False
                
                # Prepare headers
                headers = {
                    'Authorization': f'Bearer {self.api_token}',
                    'X-Customer-ID': self.customer_id,
                    'X-File-Hash': self.get_file_hash(file_path)
                }
                
                # Prepare files and data
                with open(file_path, 'rb') as f:
                    files = {'file': f}
                    data = {'metadata': json.dumps(metadata)}
                    
                    # Upload to API
                    response = requests.post(
                        f"{self.api_base_url}/api/v1/upload",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=300
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"Upload successful: {result.get('image_id', 'Unknown')}")
                    return True
                elif response.status_code == 409:
                    # File already exists (duplicate)
                    self.logger.info(f"File already exists in cloud: {file_path}")
                    return True
                else:
                    self.logger.error(f"Upload failed (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error during upload (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error during upload (attempt {attempt + 1}): {e}")
                break
        
        return False
    
    def process_dicom_file(self, file_path: str):
        """Process a DICOM file"""
        file_path = Path(file_path)
        
        # Check if file is still being written
        if not self.is_file_complete(file_path):
            self.logger.info(f"File still being written, skipping: {file_path}")
            return
        
        # Check if already processed
        file_hash = self.get_file_hash(str(file_path))
        if file_hash in self.processed_files:
            self.logger.info(f"File already processed, skipping: {file_path}")
            return
        
        self.logger.info(f"Processing DICOM file: {file_path}")
        
        try:
            # Extract metadata
            metadata = self.extract_dicom_metadata(str(file_path))
            
            # Upload to cloud API
            success = self.upload_to_cloud_api(str(file_path), metadata)
            
            if success:
                # Move to processed folder
                processed_path = self.processed_folder / file_path.name
                shutil.move(str(file_path), str(processed_path))
                
                # Add to processed files log
                self.processed_files.add(file_hash)
                self.save_processed_files_log()
                
                self.logger.info(f"Successfully processed and moved: {file_path} → {processed_path}")
            else:
                # Move to error folder
                error_path = self.error_folder / file_path.name
                shutil.move(str(file_path), str(error_path))
                self.logger.error(f"Failed to process, moved to error folder: {file_path} → {error_path}")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            # Move to error folder
            try:
                error_path = self.error_folder / file_path.name
                shutil.move(str(file_path), str(error_path))
            except:
                pass
    
    def is_file_complete(self, file_path: Path) -> bool:
        """Check if file is complete (not being written)"""
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return False
            
            # Try to open file in exclusive mode
            with open(file_path, 'r+b') as f:
                pass
            
            # Check if file size is stable
            size1 = file_path.stat().st_size
            time.sleep(1)
            size2 = file_path.stat().st_size
            
            return size1 == size2 and size1 > 0
            
        except:
            return False
    
    def scan_existing_files(self):
        """Scan for existing files in watch folder"""
        if not self.watch_folder.exists():
            return
        
        self.logger.info("Scanning for existing DICOM files...")
        
        for file_path in self.watch_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.dcm', '.dicom']:
                self.process_dicom_file(file_path)
    
    def start_watching(self):
        """Start watching the folder for new files"""
        if not WATCHDOG_AVAILABLE:
            self.logger.error("watchdog not available. Install with: pip install watchdog")
            return False
        
        # Scan existing files first
        self.scan_existing_files()
        
        # Start file system watcher
        event_handler = DICOMFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.watch_folder), recursive=True)
        
        self.logger.info(f"Started watching folder: {self.watch_folder}")
        observer.start()
        
        try:
            while True:
                time.sleep(self.scan_interval)
                # Periodic scan as backup
                self.scan_existing_files()
        except KeyboardInterrupt:
            self.logger.info("Stopping file watcher...")
            observer.stop()
        
        observer.join()
        return True
    
    def test_api_connection(self) -> bool:
        """Test connection to cloud API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'X-Customer-ID': self.customer_id
            }
            
            response = requests.get(
                f"{self.api_base_url}/api/v1/status",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("API connection test successful")
                return True
            else:
                self.logger.error(f"API connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"API connection test error: {e}")
            return False

def main():
    """Main function to run the DICOM folder bridge"""
    print("DeepSight Imaging AI - DICOM Folder Bridge")
    print("=" * 50)
    
    try:
        # Initialize bridge
        bridge = DICOMFolderBridge()
        
        # Test API connection
        print("Testing API connection...")
        if not bridge.test_api_connection():
            print("❌ API connection test failed. Please check your configuration.")
            return
        
        print("✅ API connection test successful")
        
        # Start watching
        print(f"Watching folder: {bridge.watch_folder}")
        print(f"Customer ID: {bridge.customer_id}")
        print(f"Processed files will be moved to: {bridge.processed_folder}")
        print(f"Error files will be moved to: {bridge.error_folder}")
        print("\nPress Ctrl+C to stop the bridge")
        
        bridge.start_watching()
        
    except KeyboardInterrupt:
        print("\nShutting down DICOM folder bridge...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

