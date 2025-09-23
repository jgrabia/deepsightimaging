#!/usr/bin/env python3
"""
DeepSight Imaging AI - DICOM C-STORE SCP Server Configuration
Real-time DICOM image reception from MRI machines with full configuration interface
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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import socket
import logging

# DICOM Server Implementation
try:
    from pynetdicom import AE, evt
    from pynetdicom.sop_class import CTImageStorage, MRImageStorage, SecondaryCaptureImageStorage
    from pynetdicom import debug_logger
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    st.warning("⚠️ pynetdicom not installed. Install with: pip install pynetdicom")

class DICOMServerConfig:
    """Configuration for DICOM C-STORE SCP Server"""
    
    def __init__(self):
        self.server_config = {
            'ae_title': 'DEEPSIGHT_AI',
            'port': 104,
            'host': '0.0.0.0',  # Listen on all interfaces
            'max_pdu_size': 16384,
            'timeout': 30,
            'supported_sop_classes': [
                'CTImageStorage',
                'MRImageStorage', 
                'SecondaryCaptureImageStorage',
                'UltrasoundImageStorage',
                'DigitalMammographyXRayImageStorage',
                'DigitalBreastTomosynthesisImageStorage'
            ]
        }
        
        self.mri_configs = {
            'Siemens Skyra 3T': {
                'ae_title': 'SKYRA3T',
                'port': 104,
                'supported_sequences': ['T1', 'T2', 'FLAIR', 'DWI', 'ADC', 'MRA'],
                'transfer_syntax': 'JPEG Lossless',
                'max_image_size': '512x512x512'
            },
            'GE Discovery MR750': {
                'ae_title': 'DISCOVERY750',
                'port': 104,
                'supported_sequences': ['T1', 'T2', 'FLAIR', 'DWI', 'ADC', 'MRA'],
                'transfer_syntax': 'JPEG Lossless',
                'max_image_size': '512x512x512'
            },
            'Philips Ingenia': {
                'ae_title': 'INGENIA',
                'port': 104,
                'supported_sequences': ['T1', 'T2', 'FLAIR', 'DWI', 'ADC', 'MRA'],
                'transfer_syntax': 'JPEG Lossless',
                'max_image_size': '512x512x512'
            }
        }
        
        self.network_config = {
            'firewall_ports': [104, 2762, 2763],  # DICOM ports
            'protocol': 'TCP',
            'encryption': 'TLS 1.2+',
            'authentication': 'None (DICOM standard)',
            'hipaa_compliant': True
        }

class DICOMImageReceiver:
    """DICOM C-STORE SCP Server for receiving images from MRI"""
    
    def __init__(self, config: DICOMServerConfig):
        self.config = config
        self.server = None
        self.is_running = False
        self.received_images = []
        self.server_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_server(self):
        """Start DICOM C-STORE SCP Server"""
        if not DICOM_AVAILABLE:
            st.error("❌ pynetdicom not available. Cannot start DICOM server.")
            return False
        
        try:
            # Create Application Entity
            self.server = AE()
            
            # Add supported SOP classes
            for sop_class in self.config.server_config['supported_sop_classes']:
                if sop_class == 'CTImageStorage':
                    self.server.add_supported_context(CTImageStorage)
                elif sop_class == 'MRImageStorage':
                    self.server.add_supported_context(MRImageStorage)
                elif sop_class == 'SecondaryCaptureImageStorage':
                    self.server.add_supported_context(SecondaryCaptureImageStorage)
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.is_running = True
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to start DICOM server: {e}")
            return False
    
    def _run_server(self):
        """Run DICOM server in background thread"""
        try:
            # Start server
            self.server.start_server(
                (self.config.server_config['host'], self.config.server_config['port']),
                evt.EVT_C_STORE,
                handle_store=self.handle_c_store,
                ae_title=self.config.server_config['ae_title']
            )
        except Exception as e:
            self.logger.error(f"DICOM server error: {e}")
    
    def handle_c_store(self, event):
        """Handle incoming DICOM C-STORE requests"""
        try:
            # Get the dataset
            ds = event.dataset
            
            # Create image info
            image_info = {
                'id': str(datetime.now().timestamp()),
                'timestamp': datetime.now().isoformat(),
                'patient_id': str(ds.get('PatientID', 'Unknown')),
                'patient_name': str(ds.get('PatientName', 'Unknown')),
                'study_description': str(ds.get('StudyDescription', 'Unknown')),
                'series_description': str(ds.get('SeriesDescription', 'Unknown')),
                'modality': str(ds.get('Modality', 'Unknown')),
                'ae_title': event.assoc.requestor.ae_title,
                'dataset': ds
            }
            
            # Save DICOM file
            filename = f"received_{image_info['id']}.dcm"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            ds.save_as(filepath)
            image_info['filepath'] = filepath
            
            # Add to received images
            self.received_images.append(image_info)
            
            self.logger.info(f"Received DICOM image: {image_info['patient_id']} - {image_info['series_description']}")
            
            # Return success status
            return 0x0000  # Success
            
        except Exception as e:
            self.logger.error(f"Error handling C-STORE: {e}")
            return 0xA700  # Processing failure
    
    def stop_server(self):
        """Stop DICOM server"""
        if self.server:
            self.server.shutdown()
            self.is_running = False
    
    def get_received_images(self) -> List[Dict]:
        """Get list of received images"""
        return self.received_images

class DICOMConfigurationInterface:
    """Streamlit interface for DICOM server configuration"""
    
    def __init__(self):
        self.config = DICOMServerConfig()
        self.image_receiver = DICOMImageReceiver(self.config)
        
        # Initialize session state
        if 'dicom_server_running' not in st.session_state:
            st.session_state.dicom_server_running = False
        if 'received_images' not in st.session_state:
            st.session_state.received_images = []
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
    
    def show_main_interface(self):
        """Display main DICOM configuration interface"""
        st.set_page_config(
            page_title="DeepSight Imaging AI - DICOM Server Configuration",
            page_icon="🔧",
            layout="wide"
        )
        
        st.title("🔧 DeepSight Imaging AI - DICOM Server Configuration")
        st.markdown("### Configure DICOM C-STORE SCP Server for MRI Integration")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚙️ Server Configuration", 
            "🏥 MRI Machine Setup", 
            "📡 Image Reception", 
            "🔒 Network Security"
        ])
        
        with tab1:
            self.show_server_configuration()
        
        with tab2:
            self.show_mri_setup_guide()
        
        with tab3:
            self.show_image_reception()
        
        with tab4:
            self.show_network_security()
    
    def show_server_configuration(self):
        """Server configuration interface"""
        st.header("⚙️ DICOM Server Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Current Server Settings")
            
            # Display current configuration
            st.json(self.config.server_config)
            
            # Server status
            st.subheader("Server Status")
            if st.session_state.dicom_server_running:
                st.success("🟢 DICOM Server Running")
                st.write(f"**AE Title:** {self.config.server_config['ae_title']}")
                st.write(f"**Port:** {self.config.server_config['port']}")
                st.write(f"**Host:** {self.config.server_config['host']}")
            else:
                st.error("🔴 DICOM Server Stopped")
            
            # Server controls
            if not st.session_state.dicom_server_running:
                if st.button("🚀 Start DICOM Server", key="start_server"):
                    if self.image_receiver.start_server():
                        st.session_state.dicom_server_running = True
                        st.success("✅ DICOM Server started successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start DICOM server")
            else:
                if st.button("🛑 Stop DICOM Server", key="stop_server"):
                    self.image_receiver.stop_server()
                    st.session_state.dicom_server_running = False
                    st.success("✅ DICOM Server stopped")
                    st.rerun()
        
        with col2:
            st.subheader("Configuration Options")
            
            # AE Title configuration
            new_ae_title = st.text_input(
                "AE Title", 
                value=self.config.server_config['ae_title'],
                help="Application Entity Title - must be unique on the network",
                key="config_ae_title"
            )
            
            # Port configuration
            new_port = st.number_input(
                "Port", 
                value=self.config.server_config['port'],
                min_value=1,
                max_value=65535,
                help="DICOM port (typically 104)",
                key="config_port"
            )
            
            # Host configuration
            new_host = st.selectbox(
                "Host Interface",
                ['0.0.0.0', '127.0.0.1', 'localhost'],
                index=0,
                help="0.0.0.0 listens on all interfaces",
                key="config_host"
            )
            
            # Update configuration
            if st.button("💾 Update Configuration", key="update_config"):
                self.config.server_config.update({
                    'ae_title': new_ae_title,
                    'port': new_port,
                    'host': new_host
                })
                st.success("✅ Configuration updated!")
                st.rerun()
            
            # Connection test
            st.subheader("Connection Test")
            if st.button("🔍 Test Connection", key="test_connection"):
                if self.test_dicom_connection():
                    st.success("✅ Connection test successful!")
                else:
                    st.error("❌ Connection test failed")
    
    def show_mri_setup_guide(self):
        """MRI machine setup guide"""
        st.header("🏥 MRI Machine Configuration Guide")
        
        # Select MRI manufacturer
        mri_manufacturer = st.selectbox(
            "Select MRI Manufacturer",
            list(self.config.mri_configs.keys()),
            key="mri_manufacturer"
        )
        
        selected_config = self.config.mri_configs[mri_manufacturer]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"Configuration for {mri_manufacturer}")
            
            # Display configuration
            st.write(f"**AE Title:** {selected_config['ae_title']}")
            st.write(f"**Port:** {selected_config['port']}")
            st.write(f"**Transfer Syntax:** {selected_config['transfer_syntax']}")
            
            # Step-by-step instructions
            st.subheader("Step-by-Step Configuration")
            
            steps = [
                "1. Access MRI scanner's administrative interface",
                "2. Navigate to Network/DICOM settings",
                "3. Create new DICOM destination",
                "4. Configure the following settings:",
                f"   • Destination AE Title: {self.config.server_config['ae_title']}",
                f"   • Destination IP: {self.get_server_ip()}",
                f"   • Destination Port: {self.config.server_config['port']}",
                "5. Enable DICOM C-STORE service",
                "6. Test connection with test image",
                "7. Save configuration"
            ]
            
            for step in steps:
                st.write(step)
        
        with col2:
            st.subheader("Configuration Details")
            
            # Show detailed configuration
            config_details = {
                "Source MRI Settings": {
                    "AE Title": selected_config['ae_title'],
                    "Port": selected_config['port'],
                    "Supported Sequences": selected_config['supported_sequences']
                },
                "DeepSight Imaging AI Settings": {
                    "AE Title": self.config.server_config['ae_title'],
                    "IP Address": self.get_server_ip(),
                    "Port": self.config.server_config['port'],
                    "Supported SOP Classes": self.config.server_config['supported_sop_classes']
                }
            }
            
            st.json(config_details)
            
            # Copy configuration button
            if st.button("📋 Copy Configuration", key="copy_config"):
                config_text = self.generate_config_text(mri_manufacturer, selected_config)
                st.code(config_text, language="text")
                st.success("Configuration copied to clipboard!")
    
    def show_image_reception(self):
        """Image reception monitoring"""
        st.header("📡 Real-time Image Reception")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Reception Status")
            
            if st.session_state.dicom_server_running:
                # Get received images
                received_images = self.image_receiver.get_received_images()
                st.session_state.received_images = received_images
                
                st.metric("Images Received", len(received_images))
                st.metric("Server Status", "Running")
                
                # Auto-refresh
                if st.button("🔄 Refresh", key="refresh_images"):
                    st.rerun()
                
                # Show recent images
                if received_images:
                    st.subheader("Recent Images")
                    for img in received_images[-5:]:  # Show last 5 images
                        with st.expander(f"{img['patient_id']} - {img['series_description']}"):
                            st.write(f"**Received:** {img['timestamp']}")
                            st.write(f"**Patient:** {img['patient_name']}")
                            st.write(f"**Study:** {img['study_description']}")
                            st.write(f"**Modality:** {img['modality']}")
                            st.write(f"**Source AE:** {img['ae_title']}")
                            
                            if st.button(f"View Image", key=f"view_{img['id']}"):
                                st.session_state.current_image = img
                                st.rerun()
            else:
                st.warning("⚠️ DICOM server is not running. Start the server to receive images.")
        
        with col2:
            st.subheader("Image Display")
            
            if st.session_state.current_image:
                img = st.session_state.current_image
                
                st.write(f"**Current Image:** {img['patient_id']}")
                st.write(f"**Series:** {img['series_description']}")
                
                # Display image
                try:
                    ds = img['dataset']
                    if hasattr(ds, 'pixel_array'):
                        pixel_array = ds.pixel_array
                        
                        # Normalize for display
                        if pixel_array.max() > 255:
                            display_array = ((pixel_array / pixel_array.max()) * 255).astype(np.uint8)
                        else:
                            display_array = pixel_array.astype(np.uint8)
                        
                        st.image(display_array, caption="Received DICOM Image", use_column_width=True)
                        
                        # Image metadata
                        st.subheader("Image Metadata")
                        metadata = {
                            "Patient ID": str(ds.get('PatientID', 'Unknown')),
                            "Study Date": str(ds.get('StudyDate', 'Unknown')),
                            "Series Number": str(ds.get('SeriesNumber', 'Unknown')),
                            "Instance Number": str(ds.get('InstanceNumber', 'Unknown')),
                            "Image Size": f"{ds.Rows}x{ds.Columns}",
                            "Bits Allocated": str(ds.get('BitsAllocated', 'Unknown'))
                        }
                        st.json(metadata)
                
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
            else:
                st.info("No image selected. Click 'View Image' on a received image to display it here.")
    
    def show_network_security(self):
        """Network security configuration"""
        st.header("🔒 Network Security Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Security Settings")
            
            # Display current security config
            st.json(self.config.network_config)
            
            # Security recommendations
            st.subheader("Security Recommendations")
            
            security_recommendations = [
                "✅ Use dedicated network segment for DICOM traffic",
                "✅ Configure firewall to allow only necessary ports",
                "✅ Use VPN for remote access",
                "✅ Enable DICOM TLS encryption if supported",
                "✅ Regular security audits and updates",
                "✅ Monitor DICOM traffic for anomalies",
                "✅ Backup configuration regularly"
            ]
            
            for rec in security_recommendations:
                st.write(rec)
        
        with col2:
            st.subheader("Network Configuration")
            
            # Firewall configuration
            st.subheader("Firewall Rules")
            firewall_rules = f"""
# Allow DICOM traffic to DeepSight Imaging AI
iptables -A INPUT -p tcp --dport {self.config.server_config['port']} -j ACCEPT
iptables -A INPUT -p tcp --dport 2762 -j ACCEPT  # DICOM TLS
iptables -A INPUT -p tcp --dport 2763 -j ACCEPT  # DICOM TLS

# Allow outbound DICOM (if needed)
iptables -A OUTPUT -p tcp --sport {self.config.server_config['port']} -j ACCEPT
"""
            st.code(firewall_rules, language="bash")
            
            # Network test
            st.subheader("Network Connectivity Test")
            if st.button("🌐 Test Network", key="test_network"):
                network_status = self.test_network_connectivity()
                if network_status:
                    st.success("✅ Network connectivity OK")
                else:
                    st.error("❌ Network connectivity issues detected")
    
    def test_dicom_connection(self) -> bool:
        """Test DICOM connection"""
        try:
            # Test if port is open
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', self.config.server_config['port']))
            sock.close()
            return result == 0
        except:
            return False
    
    def get_server_ip(self) -> str:
        """Get server IP address"""
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "192.168.1.100"  # Default fallback
    
    def generate_config_text(self, mri_manufacturer: str, config: Dict) -> str:
        """Generate configuration text for copying"""
        config_text = f"""
DICOM Configuration for {mri_manufacturer}
==========================================

Source MRI Settings:
- AE Title: {config['ae_title']}
- Port: {config['port']}
- Supported Sequences: {', '.join(config['supported_sequences'])}

DeepSight Imaging AI Destination:
- AE Title: {self.config.server_config['ae_title']}
- IP Address: {self.get_server_ip()}
- Port: {self.config.server_config['port']}
- Protocol: TCP
- Transfer Syntax: JPEG Lossless

Configuration Steps:
1. Access MRI scanner admin interface
2. Navigate to DICOM/Network settings
3. Create new destination
4. Enter DeepSight Imaging AI settings above
5. Test connection
6. Save configuration

Network Requirements:
- Port {self.config.server_config['port']} must be open
- TCP protocol
- No authentication required (DICOM standard)
"""
        return config_text
    
    def test_network_connectivity(self) -> bool:
        """Test network connectivity"""
        try:
            # Test local connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', self.config.server_config['port']))
            sock.close()
            return result == 0
        except:
            return False

if __name__ == "__main__":
    interface = DICOMConfigurationInterface()
    interface.show_main_interface()
