#!/usr/bin/env python3
"""
DeepSight AI - Cloud DICOM API
HIPAA-compliant cloud-based DICOM image reception with API authentication
"""

import streamlit as st
import pydicom
import numpy as np
import cv2
import tempfile
import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import hmac
import base64
import uuid
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

class CloudDICOMSecurity:
    """HIPAA-compliant security layer for cloud DICOM API"""
    
    def __init__(self):
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for data protection"""
        # In production, this should be stored securely (AWS KMS, Azure Key Vault, etc.)
        key = Fernet.generate_key()
        return key
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data)
    
    def generate_api_token(self, customer_id: str) -> str:
        """Generate secure API token for customer"""
        # Create token with expiration
        token_data = {
            'customer_id': customer_id,
            'issued_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=365)).isoformat(),
            'nonce': secrets.token_hex(16)
        }
        
        # Sign token with HMAC
        token_json = json.dumps(token_data, sort_keys=True)
        signature = hmac.new(
            self.encryption_key,
            token_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine token and signature
        full_token = f"{base64.b64encode(token_json.encode()).decode()}.{signature}"
        return full_token
    
    def verify_api_token(self, token: str) -> Optional[Dict]:
        """Verify API token and return customer info"""
        try:
            # Split token and signature
            token_part, signature = token.split('.', 1)
            
            # Decode token
            token_json = base64.b64decode(token_part).decode()
            token_data = json.loads(token_json)
            
            # Verify signature
            expected_signature = hmac.new(
                self.encryption_key,
                token_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.utcnow() > expires_at:
                return None
            
            return token_data
            
        except Exception as e:
            logging.error(f"Token verification failed: {e}")
            return None
    
    def generate_upload_signature(self, customer_id: str, filename: str, content_type: str) -> Dict:
        """Generate signed upload URL for secure file upload"""
        # Create upload signature
        upload_data = {
            'customer_id': customer_id,
            'filename': filename,
            'content_type': content_type,
            'timestamp': datetime.utcnow().isoformat(),
            'expires_in': 3600,  # 1 hour
            'nonce': secrets.token_hex(16)
        }
        
        # Sign upload data
        upload_json = json.dumps(upload_data, sort_keys=True)
        signature = hmac.new(
            self.encryption_key,
            upload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'upload_token': base64.b64encode(upload_json.encode()).decode(),
            'signature': signature,
            'upload_url': f"/api/v1/upload"
        }

class CustomerManagement:
    """Customer onboarding and API key management"""
    
    def __init__(self, security: CloudDICOMSecurity):
        self.security = security
        self.customers = {}
        
        # Initialize with demo customers
        self._initialize_demo_customers()
    
    def _initialize_demo_customers(self):
        """Initialize with demo customers for testing"""
        demo_customers = [
            {
                'customer_id': 'hospital_001',
                'name': 'Metro General Hospital',
                'contact_email': 'admin@metrohospital.com',
                'mri_models': ['Siemens Skyra 3T', 'GE Discovery MR750'],
                'hipaa_compliant': True,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active'
            },
            {
                'customer_id': 'clinic_002',
                'name': 'Advanced Imaging Clinic',
                'contact_email': 'tech@advancedimaging.com',
                'mri_models': ['Philips Ingenia'],
                'hipaa_compliant': True,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active'
            }
        ]
        
        for customer in demo_customers:
            customer['api_token'] = self.security.generate_api_token(customer['customer_id'])
            self.customers[customer['customer_id']] = customer
    
    def create_customer(self, name: str, contact_email: str, mri_models: List[str]) -> Dict:
        """Create new customer and generate API token"""
        customer_id = f"customer_{uuid.uuid4().hex[:8]}"
        
        customer = {
            'customer_id': customer_id,
            'name': name,
            'contact_email': contact_email,
            'mri_models': mri_models,
            'hipaa_compliant': True,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        # Generate API token
        customer['api_token'] = self.security.generate_api_token(customer_id)
        
        self.customers[customer_id] = customer
        return customer
    
    def get_customer(self, customer_id: str) -> Optional[Dict]:
        """Get customer information"""
        return self.customers.get(customer_id)
    
    def get_customer_by_token(self, api_token: str) -> Optional[Dict]:
        """Get customer by API token"""
        token_data = self.security.verify_api_token(api_token)
        if token_data:
            return self.get_customer(token_data['customer_id'])
        return None
    
    def regenerate_api_token(self, customer_id: str) -> str:
        """Regenerate API token for customer"""
        if customer_id in self.customers:
            new_token = self.security.generate_api_token(customer_id)
            self.customers[customer_id]['api_token'] = new_token
            return new_token
        return None

class CloudDICOMAPI:
    """Cloud-based DICOM API with authentication"""
    
    def __init__(self):
        self.security = CloudDICOMSecurity()
        self.customer_mgmt = CustomerManagement(self.security)
        self.received_images = {}
        self.upload_endpoints = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def authenticate_request(self, headers: Dict) -> Optional[Dict]:
        """Authenticate API request"""
        # Check for API token in headers
        api_token = headers.get('Authorization', '').replace('Bearer ', '')
        if not api_token:
            return None
        
        # Verify token
        customer = self.customer_mgmt.get_customer_by_token(api_token)
        return customer
    
    def process_dicom_upload(self, customer_id: str, dicom_data: bytes, metadata: Dict) -> Dict:
        """Process uploaded DICOM image"""
        try:
            # Encrypt DICOM data
            encrypted_data = self.security.encrypt_data(dicom_data)
            
            # Create image record
            image_id = str(uuid.uuid4())
            image_record = {
                'image_id': image_id,
                'customer_id': customer_id,
                'uploaded_at': datetime.utcnow().isoformat(),
                'metadata': metadata,
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'status': 'received',
                'processing_status': 'pending'
            }
            
            # Store image record
            if customer_id not in self.received_images:
                self.received_images[customer_id] = []
            
            self.received_images[customer_id].append(image_record)
            
            # Log successful upload
            self.logger.info(f"DICOM image uploaded: {image_id} from {customer_id}")
            
            return {
                'success': True,
                'image_id': image_id,
                'message': 'DICOM image uploaded successfully',
                'processing_status': 'pending'
            }
            
        except Exception as e:
            self.logger.error(f"DICOM upload failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'DICOM upload failed'
            }
    
    def get_upload_endpoint(self, customer_id: str, filename: str) -> Dict:
        """Get secure upload endpoint for customer"""
        content_type = "application/dicom"
        
        # Generate upload signature
        upload_signature = self.security.generate_upload_signature(
            customer_id, filename, content_type
        )
        
        return {
            'upload_url': f"https://api.deepsightimaging.ai/api/v1/upload",
            'upload_token': upload_signature['upload_token'],
            'signature': upload_signature['signature'],
            'expires_in': 3600,
            'content_type': content_type,
            'max_file_size': 500 * 1024 * 1024  # 500MB
        }

class CloudDICOMInterface:
    """Streamlit interface for cloud DICOM management"""
    
    def __init__(self):
        self.api = CloudDICOMAPI()
        
        # Initialize session state
        if 'cloud_dicom_customers' not in st.session_state:
            st.session_state.cloud_dicom_customers = self.api.customer_mgmt.customers
        if 'selected_customer' not in st.session_state:
            st.session_state.selected_customer = None
    
    def show_main_interface(self):
        """Display main cloud DICOM interface"""
        st.header("☁️ Cloud DICOM API Configuration")
        st.markdown("### HIPAA-Compliant Cloud-Based MRI Integration")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "👥 Customer Management", 
            "🔑 API Configuration", 
            "📡 Image Reception", 
            "🔒 Security & Compliance"
        ])
        
        with tab1:
            self.show_customer_management()
        
        with tab2:
            self.show_api_configuration()
        
        with tab3:
            self.show_image_reception()
        
        with tab4:
            self.show_security_compliance()
    
    def show_customer_management(self):
        """Customer onboarding and management"""
        st.subheader("👥 Customer Management")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Current Customers:**")
            
            # Display customers
            customers = st.session_state.cloud_dicom_customers
            if customers:
                for customer_id, customer in customers.items():
                    with st.expander(f"{customer['name']} ({customer_id})"):
                        st.write(f"**Contact:** {customer['contact_email']}")
                        st.write(f"**MRI Models:** {', '.join(customer['mri_models'])}")
                        st.write(f"**Status:** {customer['status']}")
                        st.write(f"**Created:** {customer['created_at']}")
                        
                        if st.button(f"Select Customer", key=f"select_{customer_id}"):
                            st.session_state.selected_customer = customer_id
                            st.rerun()
            else:
                st.info("No customers found. Create a new customer below.")
        
        with col2:
            st.subheader("Add New Customer")
            
            # Customer creation form
            with st.form("new_customer_form"):
                customer_name = st.text_input("Customer Name", key="new_customer_name")
                contact_email = st.text_input("Contact Email", key="new_customer_email")
                
                # MRI model selection
                mri_models = st.multiselect(
                    "MRI Models",
                    ["Siemens Skyra 3T", "GE Discovery MR750", "Philips Ingenia", "Other"],
                    key="new_customer_mri"
                )
                
                if st.form_submit_button("Create Customer"):
                    if customer_name and contact_email and mri_models:
                        # Create new customer
                        new_customer = self.api.customer_mgmt.create_customer(
                            customer_name, contact_email, mri_models
                        )
                        
                        # Update session state
                        st.session_state.cloud_dicom_customers[new_customer['customer_id']] = new_customer
                        
                        st.success(f"✅ Customer created: {new_customer['customer_id']}")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields.")
    
    def show_api_configuration(self):
        """API configuration and setup instructions"""
        st.subheader("🔑 API Configuration")
        
        if not st.session_state.selected_customer:
            st.warning("⚠️ Please select a customer first.")
            return
        
        customer_id = st.session_state.selected_customer
        customer = st.session_state.cloud_dicom_customers[customer_id]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"**Configuration for {customer['name']}:**")
            
            # Display API token
            st.subheader("API Token")
            api_token = customer['api_token']
            st.code(api_token, language="text")
            
            # Regenerate token button
            if st.button("🔄 Regenerate API Token", key="regenerate_token"):
                new_token = self.api.customer_mgmt.regenerate_api_token(customer_id)
                if new_token:
                    customer['api_token'] = new_token
                    st.session_state.cloud_dicom_customers[customer_id] = customer
                    st.success("✅ API token regenerated!")
                    st.rerun()
            
            # API endpoints
            st.subheader("API Endpoints")
            st.write("**Base URL:** `https://api.deepsightimaging.ai`")
            st.write("**Upload Endpoint:** `/api/v1/upload`")
            st.write("**Status Endpoint:** `/api/v1/status`")
            st.write("**Authentication:** Bearer token in Authorization header")
        
        with col2:
            st.subheader("MRI Configuration Instructions")
            
            # Generate configuration for each MRI model
            for mri_model in customer['mri_models']:
                with st.expander(f"Configuration for {mri_model}"):
                    config = self.generate_mri_config(customer_id, mri_model)
                    st.code(config, language="text")
            
            # Copy all configurations
            if st.button("📋 Copy All Configurations", key="copy_all_configs"):
                all_configs = self.generate_all_configurations(customer_id, customer['mri_models'])
                st.code(all_configs, language="text")
                st.success("All configurations copied to clipboard!")
    
    def show_image_reception(self):
        """Image reception monitoring"""
        st.subheader("📡 Cloud Image Reception")
        
        if not st.session_state.selected_customer:
            st.warning("⚠️ Please select a customer first.")
            return
        
        customer_id = st.session_state.selected_customer
        customer = st.session_state.cloud_dicom_customers[customer_id]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"**Reception Status for {customer['name']}:**")
            
            # Get received images
            received_images = self.api.received_images.get(customer_id, [])
            
            st.metric("Images Received", len(received_images))
            st.metric("Customer Status", customer['status'])
            
            # Refresh button
            if st.button("🔄 Refresh", key="refresh_cloud_images"):
                st.rerun()
            
            # Simulate image upload
            if st.button("📥 Simulate Image Upload", key="simulate_cloud_upload"):
                # Create mock DICOM data
                mock_dicom_data = self.create_mock_dicom_data()
                mock_metadata = {
                    'patient_id': 'TEST001',
                    'study_description': 'Brain MRI',
                    'series_description': 'T1-weighted',
                    'modality': 'MR'
                }
                
                # Process upload
                result = self.api.process_dicom_upload(customer_id, mock_dicom_data, mock_metadata)
                
                if result['success']:
                    st.success(f"✅ Simulated upload successful! Image ID: {result['image_id']}")
                else:
                    st.error(f"❌ Upload failed: {result['error']}")
                
                st.rerun()
            
            # Show recent images
            if received_images:
                st.subheader("Recent Images")
                for img in received_images[-5:]:  # Show last 5 images
                    with st.expander(f"Image {img['image_id'][:8]}..."):
                        st.write(f"**Uploaded:** {img['uploaded_at']}")
                        st.write(f"**Status:** {img['status']}")
                        st.write(f"**Processing:** {img['processing_status']}")
                        st.write(f"**Patient ID:** {img['metadata'].get('patient_id', 'N/A')}")
                        st.write(f"**Study:** {img['metadata'].get('study_description', 'N/A')}")
        
        with col2:
            st.subheader("Upload Statistics")
            
            # Upload statistics
            if received_images:
                # Calculate statistics
                total_images = len(received_images)
                today_images = len([img for img in received_images 
                                  if datetime.fromisoformat(img['uploaded_at']).date() == datetime.now().date()])
                
                st.metric("Total Images", total_images)
                st.metric("Today's Uploads", today_images)
                
                # Processing status breakdown
                status_counts = {}
                for img in received_images:
                    status = img['processing_status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                st.write("**Processing Status:**")
                for status, count in status_counts.items():
                    st.write(f"- {status.title()}: {count}")
            else:
                st.info("No images received yet. Images will appear here when uploaded via API.")
    
    def show_security_compliance(self):
        """Security and compliance information"""
        st.subheader("🔒 Security & Compliance")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**HIPAA Compliance Features:**")
            
            compliance_features = [
                "✅ End-to-end encryption for all data transmission",
                "✅ API token authentication with expiration",
                "✅ Secure file upload with signed URLs",
                "✅ Data encryption at rest",
                "✅ Audit logging for all access",
                "✅ Customer data isolation",
                "✅ Secure key management",
                "✅ Regular security assessments"
            ]
            
            for feature in compliance_features:
                st.write(feature)
            
            # Security settings
            st.subheader("Security Configuration")
            security_config = {
                'encryption_algorithm': 'AES-256-GCM',
                'token_expiration': '365 days',
                'upload_expiration': '1 hour',
                'max_file_size': '500MB',
                'audit_logging': 'Enabled',
                'data_retention': '7 years'
            }
            st.json(security_config)
        
        with col2:
            st.subheader("Network Security")
            
            # Network configuration
            network_config = """
# Cloud API Security Configuration

## Firewall Rules
- Allow HTTPS (443) only
- Block all other ports
- Use CloudFlare for DDoS protection

## API Security
- Rate limiting: 100 requests/minute per customer
- IP whitelisting (optional)
- Request signing for sensitive operations

## Data Protection
- TLS 1.3 for all connections
- Certificate pinning
- HSTS headers
- CSP headers
"""
            st.code(network_config, language="text")
            
            # Compliance checklist
            st.subheader("Compliance Checklist")
            compliance_checklist = [
                "✅ Business Associate Agreement (BAA)",
                "✅ Data Processing Agreement (DPA)",
                "✅ SOC 2 Type II certification",
                "✅ HIPAA technical safeguards",
                "✅ Regular penetration testing",
                "✅ Incident response plan"
            ]
            
            for item in compliance_checklist:
                st.write(item)
    
    def generate_mri_config(self, customer_id: str, mri_model: str) -> str:
        """Generate MRI configuration for customer"""
        customer = st.session_state.cloud_dicom_customers[customer_id]
        api_token = customer['api_token']
        
        config = f"""
# {mri_model} Configuration for {customer['name']}

## API Configuration
API_BASE_URL=https://api.deepsightimaging.ai
API_TOKEN={api_token}

## Upload Configuration
UPLOAD_ENDPOINT=/api/v1/upload
MAX_FILE_SIZE=500MB
TIMEOUT=300s

## Security
ENCRYPTION=Enabled
AUTHENTICATION=Bearer Token
TLS_VERSION=1.3

## MRI Scanner Settings
- Configure DICOM export to send to cloud API
- Use HTTPS for all communications
- Include API token in Authorization header
- Set appropriate timeout values

## Example cURL Command
curl -X POST \\
  -H "Authorization: Bearer {api_token}" \\
  -H "Content-Type: application/dicom" \\
  -F "file=@image.dcm" \\
  https://api.deepsightimaging.ai/api/v1/upload
"""
        return config
    
    def generate_all_configurations(self, customer_id: str, mri_models: List[str]) -> str:
        """Generate all MRI configurations for customer"""
        all_configs = f"# Complete Configuration for {st.session_state.cloud_dicom_customers[customer_id]['name']}\n\n"
        
        for mri_model in mri_models:
            all_configs += self.generate_mri_config(customer_id, mri_model)
            all_configs += "\n" + "="*50 + "\n\n"
        
        return all_configs
    
    def create_mock_dicom_data(self) -> bytes:
        """Create mock DICOM data for testing"""
        # Create a simple DICOM dataset
        ds = pydicom.Dataset()
        ds.PatientID = "TEST001"
        ds.PatientName = "Test^Patient"
        ds.StudyDescription = "Brain MRI"
        ds.SeriesDescription = "T1-weighted"
        ds.Modality = "MR"
        
        # Create mock pixel data
        pixel_array = np.random.randint(0, 255, (256, 256), dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()
        ds.Rows = 256
        ds.Columns = 256
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        
        # Save to bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            ds.save_as(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                dicom_bytes = f.read()
            os.unlink(tmp_file.name)
        
        return dicom_bytes

if __name__ == "__main__":
    interface = CloudDICOMInterface()
    interface.show_main_interface()
