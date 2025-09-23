#!/usr/bin/env python3
"""
DeepSight Imaging AI - DICOM to API Bridge
Converts traditional DICOM C-STORE requests to REST API calls
"""

import os
import json
import requests
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from pynetdicom import AE, evt
    from pynetdicom.sop_class import CTImageStorage, MRImageStorage, SecondaryCaptureImageStorage
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pynetdicom not available. Install with: pip install pynetdicom")

class DICOMToAPIBridge:
    """Bridge that receives DICOM images and forwards them to cloud API"""
    
    def __init__(self, config_file: str = "bridge_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        
        # DICOM Server Configuration
        self.ae_title = self.config.get('dicom_ae_title', 'DEEPSIGHT_BRIDGE')
        self.port = self.config.get('dicom_port', 104)
        self.host = self.config.get('dicom_host', '0.0.0.0')
        
        # API Configuration
        self.api_base_url = self.config.get('api_base_url', 'https://api.deepsightimaging.ai')
        self.api_token = self.config.get('api_token')
        self.customer_id = self.config.get('customer_id')
        
        if not self.api_token or not self.customer_id:
            raise ValueError("API token and customer_id must be configured")
        
        self.logger.info(f"DICOM Bridge initialized for customer: {self.customer_id}")
    
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = {
                "dicom_ae_title": "DEEPSIGHT_BRIDGE",
                "dicom_port": 104,
                "dicom_host": "0.0.0.0",
                "api_base_url": "https://api.deepsightimaging.ai",
                "api_token": "YOUR_API_TOKEN_HERE",
                "customer_id": "YOUR_CUSTOMER_ID_HERE",
                "max_retries": 3,
                "timeout": 300
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            self.logger.info(f"Created default configuration file: {self.config_file}")
            self.logger.info("Please edit the configuration file with your API credentials")
            
            return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dicom_bridge.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_dicom_server(self):
        """Start DICOM C-STORE SCP server"""
        if not DICOM_AVAILABLE:
            self.logger.error("pynetdicom not available. Cannot start DICOM server.")
            return False
        
        try:
            # Create Application Entity
            ae = AE()
            
            # Add supported SOP classes
            ae.add_supported_context(CTImageStorage)
            ae.add_supported_context(MRImageStorage)
            ae.add_supported_context(SecondaryCaptureImageStorage)
            
            # Start server
            self.logger.info(f"Starting DICOM server on {self.host}:{self.port}")
            self.logger.info(f"AE Title: {self.ae_title}")
            
            ae.start_server(
                (self.host, self.port),
                evt.EVT_C_STORE,
                handle_store=self.handle_c_store,
                ae_title=self.ae_title
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start DICOM server: {e}")
            return False
    
    def handle_c_store(self, event):
        """Handle incoming DICOM C-STORE requests"""
        try:
            # Get the dataset
            ds = event.dataset
            
            # Extract metadata
            metadata = {
                'patient_id': str(ds.get('PatientID', 'Unknown')),
                'patient_name': str(ds.get('PatientName', 'Unknown')),
                'study_description': str(ds.get('StudyDescription', 'Unknown')),
                'series_description': str(ds.get('SeriesDescription', 'Unknown')),
                'modality': str(ds.get('Modality', 'Unknown')),
                'study_date': str(ds.get('StudyDate', 'Unknown')),
                'series_number': str(ds.get('SeriesNumber', 'Unknown')),
                'instance_number': str(ds.get('InstanceNumber', 'Unknown'))
            }
            
            # Save DICOM to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                ds.save_as(tmp_file.name)
                dicom_file_path = tmp_file.name
            
            self.logger.info(f"Received DICOM: {metadata['patient_id']} - {metadata['series_description']}")
            
            # Upload to cloud API
            success = self.upload_to_cloud_api(dicom_file_path, metadata)
            
            # Clean up temporary file
            os.unlink(dicom_file_path)
            
            if success:
                self.logger.info(f"Successfully uploaded to cloud API: {metadata['patient_id']}")
                return 0x0000  # Success
            else:
                self.logger.error(f"Failed to upload to cloud API: {metadata['patient_id']}")
                return 0xA700  # Processing failure
                
        except Exception as e:
            self.logger.error(f"Error handling C-STORE: {e}")
            return 0xA700  # Processing failure
    
    def upload_to_cloud_api(self, dicom_file_path: str, metadata: Dict) -> bool:
        """Upload DICOM file to cloud API"""
        try:
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'X-Customer-ID': self.customer_id
            }
            
            # Prepare files
            with open(dicom_file_path, 'rb') as f:
                files = {'file': f}
                data = {'metadata': json.dumps(metadata)}
                
                # Upload to API
                response = requests.post(
                    f"{self.api_base_url}/api/v1/upload",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.config.get('timeout', 300)
                )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"API upload successful: {result.get('image_id', 'Unknown')}")
                return True
            else:
                self.logger.error(f"API upload failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during API upload: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during API upload: {e}")
            return False
    
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
    """Main function to run the DICOM bridge"""
    print("DeepSight Imaging AI - DICOM to API Bridge")
    print("=" * 50)
    
    try:
        # Initialize bridge
        bridge = DICOMToAPIBridge()
        
        # Test API connection
        print("Testing API connection...")
        if not bridge.test_api_connection():
            print("❌ API connection test failed. Please check your configuration.")
            return
        
        print("✅ API connection test successful")
        
        # Start DICOM server
        print(f"Starting DICOM server on port {bridge.port}...")
        print(f"AE Title: {bridge.ae_title}")
        print(f"Customer ID: {bridge.customer_id}")
        print("\nMRI Configuration:")
        print(f"  AE Title: {bridge.ae_title}")
        print(f"  IP Address: [This server's IP address]")
        print(f"  Port: {bridge.port}")
        print("\nPress Ctrl+C to stop the server")
        
        bridge.start_dicom_server()
        
    except KeyboardInterrupt:
        print("\nShutting down DICOM bridge...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

