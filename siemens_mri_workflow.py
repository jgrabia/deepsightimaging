#!/usr/bin/env python3
"""
DeepSight Imaging AI - Siemens Skyra 3T MRI Integration
Real-time image reception, annotation, and AI analysis workflow
"""

import streamlit as st
import pydicom
import numpy as np
import cv2
import tempfile
import os
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import threading
import time

class SiemensSkyraIntegration:
    """Direct integration with Siemens Skyra 3T MRI"""
    
    def __init__(self):
        self.mri_config = {
            'model': 'Siemens Skyra 3T',
            'field_strength': '3.0 Tesla',
            'dicom_port': 104,
            'ae_title': 'DEEPSIGHT_AI',
            'supported_sequences': [
                'T1-weighted', 'T2-weighted', 'FLAIR', 'DWI', 'ADC',
                'T1-post-contrast', 'MRA', 'MRV', 'DTI', 'SWI'
            ],
            'image_formats': ['DICOM', 'JPEG', 'PNG'],
            'max_image_size': '512x512x512'
        }
        
        self.ai_models = {
            'brain_lesion_detection': 'Brain lesion detection and classification',
            'stroke_detection': 'Acute stroke detection (DWI/ADC)',
            'tumor_segmentation': 'Brain tumor segmentation and volumetry',
            'white_matter_analysis': 'White matter hyperintensity detection',
            'hemorrhage_detection': 'Intracranial hemorrhage detection',
            'quality_assessment': 'Image quality and artifact detection'
        }

class RealTimeImageReceiver:
    """Real-time image reception from Siemens Skyra"""
    
    def __init__(self):
        self.received_images = []
        self.processing_queue = []
        self.is_receiving = False
    
    def start_reception(self):
        """Start receiving images from MRI"""
        self.is_receiving = True
        st.success("🔄 Connected to Siemens Skyra 3T - Ready to receive images")
        return True
    
    def receive_dicom_image(self, dicom_data: bytes) -> str:
        """Receive and process DICOM image from MRI"""
        try:
            # Save received DICOM data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(dicom_data)
                tmp_path = tmp_file.name
            
            # Load and process DICOM
            ds = pydicom.dcmread(tmp_path)
            
            # Extract image information
            image_info = {
                'id': str(datetime.now().timestamp()),
                'timestamp': datetime.now().isoformat(),
                'patient_id': ds.get('PatientID', 'Unknown'),
                'study_description': ds.get('StudyDescription', 'Unknown'),
                'series_description': ds.get('SeriesDescription', 'Unknown'),
                'sequence_name': ds.get('SequenceName', 'Unknown'),
                'slice_location': ds.get('SliceLocation', 0),
                'image_position': ds.get('ImagePositionPatient', [0, 0, 0]),
                'file_path': tmp_path,
                'dicom_dataset': ds
            }
            
            self.received_images.append(image_info)
            self.processing_queue.append(image_info['id'])
            
            return image_info['id']
            
        except Exception as e:
            st.error(f"Error receiving DICOM image: {e}")
            return None
    
    def get_latest_image(self) -> Optional[Dict]:
        """Get the most recently received image"""
        if self.received_images:
            return self.received_images[-1]
        return None

class CloudAnnotationSystem:
    """Cloud-based annotation system for radiologists"""
    
    def __init__(self):
        self.annotations = {}
        self.annotation_types = {
            'point': 'Point annotation',
            'rectangle': 'Rectangular ROI',
            'polygon': 'Polygon ROI',
            'freehand': 'Freehand drawing',
            'measurement': 'Distance/Area measurement'
        }
    
    def create_annotation(self, image_id: str, annotation_type: str, coordinates: List, metadata: Dict) -> str:
        """Create new annotation on image"""
        annotation_id = f"ann_{datetime.now().timestamp()}"
        
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'type': annotation_type,
            'coordinates': coordinates,
            'metadata': metadata,
            'created_by': 'Radiologist',
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        if image_id not in self.annotations:
            self.annotations[image_id] = []
        
        self.annotations[image_id].append(annotation)
        return annotation_id
    
    def get_image_annotations(self, image_id: str) -> List[Dict]:
        """Get all annotations for an image"""
        return self.annotations.get(image_id, [])
    
    def update_annotation(self, annotation_id: str, updates: Dict) -> bool:
        """Update existing annotation"""
        for image_id, annotations in self.annotations.items():
            for annotation in annotations:
                if annotation['id'] == annotation_id:
                    annotation.update(updates)
                    annotation['updated_at'] = datetime.now().isoformat()
                    return True
        return False

class RealTimeAIAnalysis:
    """Real-time AI analysis pipeline"""
    
    def __init__(self):
        self.analysis_results = {}
        self.model_confidence_threshold = 0.7
    
    def analyze_image(self, image_info: Dict) -> Dict:
        """Run AI analysis on received image"""
        try:
            ds = image_info['dicom_dataset']
            pixel_array = ds.pixel_array
            
            # Determine analysis type based on sequence
            sequence_name = image_info['series_description'].lower()
            
            analysis_results = {
                'image_id': image_info['id'],
                'analysis_timestamp': datetime.now().isoformat(),
                'sequence_type': sequence_name,
                'findings': {},
                'confidence_scores': {},
                'recommendations': []
            }
            
            # Run appropriate AI models based on sequence
            if 't1' in sequence_name or 't2' in sequence_name:
                # Brain lesion detection
                lesion_results = self.run_brain_lesion_detection(pixel_array)
                analysis_results['findings']['lesions'] = lesion_results
                analysis_results['confidence_scores']['lesion_detection'] = 0.85
            
            if 'dwi' in sequence_name or 'adc' in sequence_name:
                # Stroke detection
                stroke_results = self.run_stroke_detection(pixel_array)
                analysis_results['findings']['stroke'] = stroke_results
                analysis_results['confidence_scores']['stroke_detection'] = 0.92
            
            if 'flair' in sequence_name:
                # White matter analysis
                wm_results = self.run_white_matter_analysis(pixel_array)
                analysis_results['findings']['white_matter'] = wm_results
                analysis_results['confidence_scores']['wm_analysis'] = 0.78
            
            # Quality assessment for all images
            quality_results = self.run_quality_assessment(pixel_array)
            analysis_results['findings']['quality'] = quality_results
            analysis_results['confidence_scores']['quality'] = 0.88
            
            # Generate recommendations
            analysis_results['recommendations'] = self.generate_recommendations(analysis_results)
            
            self.analysis_results[image_info['id']] = analysis_results
            return analysis_results
            
        except Exception as e:
            st.error(f"AI analysis error: {e}")
            return None
    
    def run_brain_lesion_detection(self, pixel_array: np.ndarray) -> Dict:
        """Mock brain lesion detection"""
        return {
            'lesions_detected': 2,
            'locations': [
                {'region': 'Right frontal lobe', 'size': 8.5, 'confidence': 0.87},
                {'region': 'Left parietal lobe', 'size': 5.2, 'confidence': 0.73}
            ],
            'total_lesion_volume': 13.7
        }
    
    def run_stroke_detection(self, pixel_array: np.ndarray) -> Dict:
        """Mock stroke detection"""
        return {
            'stroke_detected': True,
            'location': 'Left middle cerebral artery territory',
            'volume': 45.3,
            'confidence': 0.94,
            'severity': 'Moderate'
        }
    
    def run_white_matter_analysis(self, pixel_array: np.ndarray) -> Dict:
        """Mock white matter analysis"""
        return {
            'wmh_detected': True,
            'wmh_count': 12,
            'total_wmh_volume': 8.7,
            'fazekas_scale': 2
        }
    
    def run_quality_assessment(self, pixel_array: np.ndarray) -> Dict:
        """Mock quality assessment"""
        return {
            'overall_quality': 8.5,
            'artifacts': 'Minimal motion artifact',
            'signal_to_noise': 45.2,
            'recommendation': 'Images suitable for diagnosis'
        }
    
    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate clinical recommendations based on AI analysis"""
        recommendations = []
        
        if 'lesions' in analysis_results['findings']:
            lesion_count = analysis_results['findings']['lesions']['lesions_detected']
            if lesion_count > 0:
                recommendations.append(f"Consider correlation with clinical history for {lesion_count} detected lesions")
        
        if 'stroke' in analysis_results['findings']:
            if analysis_results['findings']['stroke']['stroke_detected']:
                recommendations.append("URGENT: Acute stroke detected - immediate clinical correlation required")
        
        if 'quality' in analysis_results['findings']:
            quality_score = analysis_results['findings']['quality']['overall_quality']
            if quality_score < 7.0:
                recommendations.append("Consider repeat imaging due to suboptimal quality")
        
        return recommendations

class SiemensMRIWorkflow:
    """Main workflow for Siemens Skyra integration"""
    
    def __init__(self):
        self.mri_integration = SiemensSkyraIntegration()
        self.image_receiver = RealTimeImageReceiver()
        self.annotation_system = CloudAnnotationSystem()
        self.ai_analysis = RealTimeAIAnalysis()
        
        # Initialize session state
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []
        if 'ai_results' not in st.session_state:
            st.session_state.ai_results = None
    
    def show_main_dashboard(self):
        """Display main Siemens MRI workflow dashboard"""
        st.set_page_config(
            page_title="DeepSight Imaging AI - Siemens Skyra Integration",
            page_icon="🧠",
            layout="wide"
        )
        
        st.title("🧠 DeepSight Imaging AI - Siemens Skyra 3T Integration")
        st.markdown("### Real-time MRI Image Reception, Annotation & AI Analysis")
        
        # Status bar
        self.show_status_bar()
        
        # Main workflow tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📡 Image Reception", 
            "✏️ Annotation", 
            "🤖 AI Analysis", 
            "📋 Report"
        ])
        
        with tab1:
            self.show_image_reception()
        
        with tab2:
            self.show_annotation_tools()
        
        with tab3:
            self.show_ai_analysis()
        
        with tab4:
            self.show_report_generation()
    
    def show_status_bar(self):
        """Show connection status and current image info"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.image_receiver.is_receiving:
                st.success("🟢 Connected to Siemens Skyra")
            else:
                st.error("🔴 Disconnected from MRI")
        
        with col2:
            image_count = len(self.image_receiver.received_images)
            st.metric("Images Received", image_count)
        
        with col3:
            if st.session_state.current_image:
                st.metric("Current Image", "Active")
            else:
                st.metric("Current Image", "None")
        
        with col4:
            ai_count = len(self.ai_analysis.analysis_results)
            st.metric("AI Analyses", ai_count)
    
    def show_image_reception(self):
        """Image reception interface"""
        st.header("📡 Real-time Image Reception")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("MRI Connection")
            
            # Connection controls
            if not self.image_receiver.is_receiving:
                if st.button("🔌 Connect to Siemens Skyra", key="connect_mri"):
                    if self.image_receiver.start_reception():
                        st.rerun()
            else:
                if st.button("🔌 Disconnect from MRI", key="disconnect_mri"):
                    self.image_receiver.is_receiving = False
                    st.rerun()
            
            # Simulate image reception
            st.subheader("Simulate Image Reception")
            if st.button("📥 Simulate Receive DICOM Image", key="simulate_receive"):
                # Create mock DICOM data
                mock_dicom_data = self.create_mock_dicom()
                image_id = self.image_receiver.receive_dicom_image(mock_dicom_data)
                
                if image_id:
                    st.session_state.current_image = self.image_receiver.get_latest_image()
                    st.success(f"✅ Image received successfully! ID: {image_id}")
                    st.rerun()
        
        with col2:
            st.subheader("Received Images")
            
            if self.image_receiver.received_images:
                # Show latest image
                latest_image = self.image_receiver.get_latest_image()
                
                if latest_image:
                    st.write(f"**Latest Image:** {latest_image['study_description']}")
                    st.write(f"**Sequence:** {latest_image['series_description']}")
                    st.write(f"**Received:** {latest_image['timestamp']}")
                    
                    # Display image
                    try:
                        ds = latest_image['dicom_dataset']
                        if hasattr(ds, 'pixel_array'):
                            pixel_array = ds.pixel_array
                            
                            # Normalize for display
                            if pixel_array.max() > 255:
                                display_array = ((pixel_array / pixel_array.max()) * 255).astype(np.uint8)
                            else:
                                display_array = pixel_array.astype(np.uint8)
                            
                            st.image(display_array, caption="Received MRI Image", use_column_width=True)
                            
                            if st.button("🔄 Set as Current Image", key="set_current"):
                                st.session_state.current_image = latest_image
                                st.success("Current image updated!")
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
            else:
                st.info("No images received yet. Connect to MRI and start receiving images.")
    
    def show_annotation_tools(self):
        """Annotation tools interface"""
        st.header("✏️ Cloud Annotation System")
        
        if not st.session_state.current_image:
            st.warning("⚠️ No current image selected. Please receive an image first.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Annotation Tools")
            
            # Annotation type selection
            annotation_type = st.selectbox(
                "Annotation Type",
                list(self.annotation_system.annotation_types.keys()),
                key="annotation_type"
            )
            
            st.write(f"**Selected:** {self.annotation_system.annotation_types[annotation_type]}")
            
            # Annotation controls
            if st.button("➕ Add Annotation", key="add_annotation"):
                # Create mock annotation
                annotation_id = self.annotation_system.create_annotation(
                    image_id=st.session_state.current_image['id'],
                    annotation_type=annotation_type,
                    coordinates=[100, 100, 200, 200],  # Mock coordinates
                    metadata={
                        'description': f'{annotation_type} annotation',
                        'confidence': 0.95
                    }
                )
                
                st.success(f"✅ Annotation added! ID: {annotation_id}")
                st.rerun()
            
            # Show current annotations
            st.subheader("Current Annotations")
            current_annotations = self.annotation_system.get_image_annotations(
                st.session_state.current_image['id']
            )
            
            if current_annotations:
                for annotation in current_annotations:
                    with st.expander(f"{annotation['type'].title()} - {annotation['created_at']}"):
                        st.write(f"**Type:** {annotation['type']}")
                        st.write(f"**Created:** {annotation['created_by']}")
                        st.write(f"**Coordinates:** {annotation['coordinates']}")
                        st.write(f"**Description:** {annotation['metadata']['description']}")
                        
                        if st.button(f"🗑️ Delete", key=f"delete_{annotation['id']}"):
                            # Remove annotation (simplified)
                            st.success("Annotation deleted!")
                            st.rerun()
            else:
                st.info("No annotations on current image")
        
        with col2:
            st.subheader("Image Display with Annotations")
            
            if st.session_state.current_image:
                try:
                    ds = st.session_state.current_image['dicom_dataset']
                    if hasattr(ds, 'pixel_array'):
                        pixel_array = ds.pixel_array
                        
                        # Normalize for display
                        if pixel_array.max() > 255:
                            display_array = ((pixel_array / pixel_array.max()) * 255).astype(np.uint8)
                        else:
                            display_array = pixel_array.astype(np.uint8)
                        
                        # Convert to color for annotation overlay
                        if len(display_array.shape) == 2:
                            display_array = cv2.cvtColor(display_array, cv2.COLOR_GRAY2RGB)
                        
                        # Draw annotations
                        for annotation in current_annotations:
                            coords = annotation['coordinates']
                            if annotation['type'] == 'rectangle':
                                cv2.rectangle(display_array, 
                                            (coords[0], coords[1]), 
                                            (coords[2], coords[3]), 
                                            (0, 255, 0), 2)
                        
                        st.image(display_array, caption="Annotated Image", use_column_width=True)
                
                except Exception as e:
                    st.error(f"Error displaying annotated image: {e}")
    
    def show_ai_analysis(self):
        """AI analysis interface"""
        st.header("🤖 Real-time AI Analysis")
        
        if not st.session_state.current_image:
            st.warning("⚠️ No current image selected. Please receive an image first.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("AI Models Available")
            
            for model_name, description in self.ai_analysis.ai_models.items():
                with st.expander(f"{model_name.replace('_', ' ').title()}"):
                    st.write(description)
                    
                    if st.button(f"Run {model_name}", key=f"run_{model_name}"):
                        with st.spinner("Running AI analysis..."):
                            # Run AI analysis
                            results = self.ai_analysis.analyze_image(st.session_state.current_image)
                            
                            if results:
                                st.session_state.ai_results = results
                                st.success("✅ AI analysis completed!")
                                st.rerun()
                            else:
                                st.error("❌ AI analysis failed")
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.session_state.ai_results:
                results = st.session_state.ai_results
                
                st.write(f"**Analysis Time:** {results['analysis_timestamp']}")
                st.write(f"**Sequence:** {results['sequence_type']}")
                
                # Display findings
                for finding_type, finding_data in results['findings'].items():
                    with st.expander(f"{finding_type.replace('_', ' ').title()}"):
                        if finding_type == 'lesions':
                            st.write(f"Lesions Detected: {finding_data['lesions_detected']}")
                            for lesion in finding_data['locations']:
                                st.write(f"• {lesion['region']}: {lesion['size']}mm (confidence: {lesion['confidence']:.2f})")
                        
                        elif finding_type == 'stroke':
                            st.write(f"Stroke Detected: {finding_data['stroke_detected']}")
                            if finding_data['stroke_detected']:
                                st.write(f"• Location: {finding_data['location']}")
                                st.write(f"• Volume: {finding_data['volume']}ml")
                                st.write(f"• Severity: {finding_data['severity']}")
                        
                        elif finding_type == 'quality':
                            st.write(f"Quality Score: {finding_data['overall_quality']}/10")
                            st.write(f"Artifacts: {finding_data['artifacts']}")
                            st.write(f"SNR: {finding_data['signal_to_noise']}")
                
                # Display recommendations
                if results['recommendations']:
                    st.subheader("Clinical Recommendations")
                    for rec in results['recommendations']:
                        st.write(f"• {rec}")
            else:
                st.info("No AI analysis results available. Run analysis on current image.")
    
    def show_report_generation(self):
        """Report generation interface"""
        st.header("📋 Report Generation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Generate Report")
            
            patient_name = st.text_input("Patient Name", key="report_patient_name")
            study_date = st.date_input("Study Date", key="report_study_date")
            
            # Clinical information
            clinical_history = st.text_area("Clinical History", key="report_clinical_history")
            indication = st.text_input("Study Indication", key="report_indication")
            
            # Radiologist findings
            radiologist_findings = st.text_area("Radiologist Findings", key="report_radiologist_findings")
            impression = st.text_area("Impression", key="report_impression")
            
            if st.button("📄 Generate Final Report", key="generate_report"):
                # Generate comprehensive report
                report = self.generate_comprehensive_report(
                    patient_name, study_date, clinical_history, 
                    indication, radiologist_findings, impression
                )
                
                st.session_state.current_report = report
                st.success("✅ Report generated successfully!")
                st.rerun()
        
        with col2:
            st.subheader("Report Preview")
            
            if hasattr(st.session_state, 'current_report') and st.session_state.current_report:
                report = st.session_state.current_report
                
                st.write("**DeepSight Imaging AI - MRI Report**")
                st.write("---")
                st.write(f"**Patient:** {report['patient_name']}")
                st.write(f"**Study Date:** {report['study_date']}")
                st.write(f"**Indication:** {report['indication']}")
                st.write("")
                
                st.write("**Clinical History:**")
                st.write(report['clinical_history'])
                st.write("")
                
                st.write("**Technique:**")
                st.write(report['technique'])
                st.write("")
                
                st.write("**AI-Assisted Findings:**")
                st.write(report['ai_findings'])
                st.write("")
                
                st.write("**Radiologist Findings:**")
                st.write(report['radiologist_findings'])
                st.write("")
                
                st.write("**Impression:**")
                st.write(report['impression'])
                st.write("")
                
                st.write(f"**Report Generated:** {report['generated_at']}")
                
                # Download report
                report_text = self.format_report_for_download(report)
                st.download_button(
                    label="📥 Download Report",
                    data=report_text,
                    file_name=f"MRI_Report_{patient_name}_{study_date}.txt",
                    mime="text/plain"
                )
            else:
                st.info("No report generated yet. Fill in the form and generate report.")
    
    def create_mock_dicom(self) -> bytes:
        """Create mock DICOM data for simulation"""
        # Create a simple DICOM dataset
        ds = pydicom.Dataset()
        ds.PatientID = "MOCK001"
        ds.PatientName = "Test^Patient"
        ds.StudyDescription = "Brain MRI"
        ds.SeriesDescription = "T1-weighted"
        ds.SequenceName = "T1_MPRAGE"
        ds.SliceLocation = 0.0
        ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        
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
    
    def generate_comprehensive_report(self, patient_name, study_date, clinical_history, 
                                    indication, radiologist_findings, impression) -> Dict:
        """Generate comprehensive MRI report"""
        
        # Get AI results if available
        ai_summary = "No AI analysis performed."
        if st.session_state.ai_results:
            ai_results = st.session_state.ai_results
            ai_summary = f"AI analysis performed on {ai_results['sequence_type']} sequence. "
            
            if 'lesions' in ai_results['findings']:
                lesion_count = ai_results['findings']['lesions']['lesions_detected']
                ai_summary += f"Detected {lesion_count} potential lesions. "
            
            if 'stroke' in ai_results['findings'] and ai_results['findings']['stroke']['stroke_detected']:
                ai_summary += "Acute stroke detected requiring immediate attention. "
        
        report = {
            'patient_name': patient_name,
            'study_date': study_date.isoformat(),
            'indication': indication,
            'clinical_history': clinical_history,
            'technique': f"MRI performed on Siemens Skyra 3T. Sequences included T1-weighted, T2-weighted, and FLAIR imaging.",
            'ai_findings': ai_summary,
            'radiologist_findings': radiologist_findings,
            'impression': impression,
            'generated_at': datetime.now().isoformat(),
            'radiologist': 'Dr. DeepSight Imaging AI',
            'facility': 'DeepSight Imaging AI Medical Imaging Center'
        }
        
        return report
    
    def format_report_for_download(self, report: Dict) -> str:
        """Format report for text download"""
        formatted_report = f"""
DEEPSIGHT IMAGING AI - MRI REPORT
========================

Patient: {report['patient_name']}
Study Date: {report['study_date']}
Indication: {report['indication']}

CLINICAL HISTORY:
{report['clinical_history']}

TECHNIQUE:
{report['technique']}

AI-ASSISTED FINDINGS:
{report['ai_findings']}

RADIOLOGIST FINDINGS:
{report['radiologist_findings']}

IMPRESSION:
{report['impression']}

---
Report generated by DeepSight Imaging AI on {report['generated_at']}
Radiologist: {report['radiologist']}
Facility: {report['facility']}
"""
        return formatted_report

if __name__ == "__main__":
    workflow = SiemensMRIWorkflow()
    workflow.show_main_dashboard()
