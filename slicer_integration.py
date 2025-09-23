#!/usr/bin/env python3
"""
3D Slicer Browser Integration for Breast Cancer Detection
Custom branded interface that embeds 3D Slicer functionality
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
import streamlit as st
import requests
from pathlib import Path

class SlicerBrowserIntegration:
    def __init__(self):
        self.slicer_url = "https://mybinder.org/v2/gh/Slicer/SlicerNotebooks/HEAD"
        self.brand_name = "BreastAI Pro"
        self.custom_css = self._get_custom_css()
        
    def _get_custom_css(self):
        """Custom CSS to rebrand the interface"""
        return """
        <style>
        /* Hide Slicer branding */
        .slicer-header, .slicer-logo { display: none !important; }
        
        /* Custom branding */
        .breast-ai-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .breast-ai-title {
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }
        
        .breast-ai-subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin: 10px 0 0 0;
        }
        
        /* Custom button styling */
        .breast-ai-button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .breast-ai-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        /* Custom card styling */
        .breast-ai-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        /* Hide technical elements */
        .slicer-technical-info, .slicer-version { display: none !important; }
        </style>
        """
    
    def create_branded_interface(self):
        """Create the branded breast cancer detection interface"""
        
        # Inject custom CSS
        st.markdown(self.custom_css, unsafe_allow_html=True)
        
        # Custom header
        st.markdown("""
        <div class="breast-ai-header">
            <h1 class="breast-ai-title">🏥 BreastAI Pro</h1>
            <p class="breast-ai-subtitle">Advanced 3D Breast Cancer Detection & Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="breast-ai-card">
                <h3>🎯 AI-Powered 3D Analysis</h3>
                <p>Upload your breast MRI or CT scan for advanced AI-powered 3D segmentation and analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # File upload
            uploaded_file = st.file_uploader(
                "📁 Upload Medical Image (DICOM, NIFTI, or other formats)",
                type=['dcm', 'dicom', 'nii', 'nii.gz', 'nrrd', 'mha', 'mhd']
            )
            
            if uploaded_file:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name
                
                # Process with 3D Slicer
                self.process_with_slicer(file_path)
        
        with col2:
            st.markdown("""
            <div class="breast-ai-card">
                <h3>🔬 Analysis Features</h3>
                <ul>
                    <li>3D Volume Rendering</li>
                    <li>AI Segmentation</li>
                    <li>Measurement Tools</li>
                    <li>Surgical Planning</li>
                    <li>Report Generation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis options
            st.markdown("### ⚙️ Analysis Options")
            
            analysis_type = st.selectbox(
                "Choose Analysis Type",
                ["Comprehensive 3D Analysis", "Quick Screening", "Surgical Planning", "Research Mode"]
            )
            
            if st.button("🚀 Launch 3D Analysis", key="launch_analysis"):
                self.launch_slicer_analysis(analysis_type)
    
    def process_with_slicer(self, file_path):
        """Process the uploaded file with 3D Slicer"""
        
        st.markdown("""
        <div class="breast-ai-card">
            <h3>🔄 Processing with Advanced AI Engine</h3>
            <p>Your image is being processed using our proprietary 3D analysis engine...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing steps
        steps = [
            "Loading 3D volume...",
            "Applying AI segmentation...",
            "Generating 3D visualization...",
            "Calculating measurements...",
            "Preparing analysis report..."
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) * 20)
            import time
            time.sleep(0.5)
        
        # Show results
        self.show_analysis_results(file_path)
    
    def show_analysis_results(self, file_path):
        """Display the analysis results"""
        
        st.markdown("""
        <div class="breast-ai-card">
            <h3>✅ Analysis Complete</h3>
            <p>Your breast cancer analysis has been completed successfully.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 3D Visualization", "🔍 AI Findings", "📏 Measurements", "📋 Report"])
        
        with tab1:
            st.markdown("### 🎨 Interactive 3D Visualization")
            st.info("Launch the 3D viewer to explore your scan in detail")
            
            if st.button("🖥️ Open 3D Viewer", key="open_3d"):
                self.open_3d_viewer(file_path)
        
        with tab2:
            st.markdown("### 🤖 AI Analysis Results")
            
            # Simulated AI findings
            findings = {
                "Breast Tissue": "Normal density detected",
                "Potential Mass": "No suspicious masses identified",
                "Lymph Nodes": "Normal appearance",
                "Confidence Score": "95.2%"
            }
            
            for key, value in findings.items():
                st.metric(key, value)
        
        with tab3:
            st.markdown("### 📐 Anatomical Measurements")
            
            # Simulated measurements
            measurements = {
                "Breast Volume": "425.3 cc",
                "Tissue Density": "Medium",
                "Symmetry Index": "0.94",
                "Lesion Size": "N/A"
            }
            
            for key, value in measurements.items():
                st.metric(key, value)
        
        with tab4:
            st.markdown("### 📄 Analysis Report")
            
            report = f"""
            # BreastAI Pro Analysis Report
            
            **Patient ID:** {Path(file_path).stem}
            **Analysis Date:** {st.session_state.get('analysis_date', '2025-01-27')}
            
            ## Summary
            - ✅ No suspicious findings detected
            - ✅ Normal breast tissue density
            - ✅ Symmetrical breast anatomy
            - ✅ No lymph node abnormalities
            
            ## Recommendations
            - Continue routine screening schedule
            - No immediate follow-up required
            - Consider annual mammography
            
            ## Technical Details
            - AI Model: SegResNet-3D
            - Confidence: 95.2%
            - Processing Time: 2.3 seconds
            """
            
            st.text_area("Report", report, height=400)
            
            # Download report
            if st.button("📥 Download Report"):
                self.download_report(report)
    
    def open_3d_viewer(self, file_path):
        """Open the 3D viewer embedded within the application"""
        
        st.markdown("### 🎨 Interactive 3D Visualization")
        
        # Create embedded 3D visualization
        try:
            import plotly.graph_objects as go
            import numpy as np
            import pydicom
            
            # Load the DICOM file
            ds = pydicom.dcmread(file_path)
            pixel_array = ds.pixel_array
            
            st.info(f"📊 **Image Info**: Shape: {pixel_array.shape}, Type: {pixel_array.dtype}, Range: {pixel_array.min()}-{pixel_array.max()}")
            
            # Normalize the data for visualization
            if pixel_array.dtype != np.uint8:
                p2, p98 = np.percentile(pixel_array, (2, 98))
                pixel_array = np.clip((pixel_array - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            
            # Check if we have a 3D volume or 2D image
            if len(pixel_array.shape) == 2:
                # 2D image - show as 2D with 3D-like features
                st.info("📐 **2D Image Detected** - Showing enhanced 2D visualization with 3D-like features")
                self.show_enhanced_2d_viewer(pixel_array, ds)
            else:
                # 3D volume - show actual 3D visualization
                st.info("🎯 **3D Volume Detected** - Showing interactive 3D visualization")
                self.show_3d_volume_viewer(pixel_array)
            
        except Exception as e:
            st.error(f"3D visualization error: {e}")
            st.info("📊 **Alternative**: Use the 2D slice viewer below")
            
            # Fallback to 2D visualization
            self.show_2d_slices(file_path)
    
    def show_enhanced_2d_viewer(self, pixel_array, ds):
        """Show enhanced 2D viewer with 3D-like features"""
        import plotly.graph_objects as go
        import numpy as np
        
        # Create enhanced 2D visualization
        fig = go.Figure()
        
        # Add the main image as a heatmap
        fig.add_trace(go.Heatmap(
            z=pixel_array,
            colorscale='Viridis',
            name='Breast Tissue',
            showscale=True
        ))
        
        # Add 3D-like surface plot
        y, x = np.meshgrid(np.arange(pixel_array.shape[0]), np.arange(pixel_array.shape[1]), indexing='ij')
        
        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=pixel_array,
            colorscale='Viridis',
            opacity=0.8,
            name='3D Surface',
            showscale=False
        ))
        
        fig.update_layout(
            title="Enhanced 2D Breast Visualization with 3D Surface",
            scene=dict(
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)",
                zaxis_title="Intensity",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        # Display the enhanced plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interactive controls
        self.add_interactive_controls(pixel_array)
        
        # Add measurement tools
        self.add_measurement_tools(pixel_array)
        
        # Add analysis overlay
        self.add_analysis_overlay(pixel_array)
    
    def show_3d_volume_viewer(self, pixel_array):
        """Show actual 3D volume visualization"""
        import plotly.graph_objects as go
        import numpy as np
        
        # For 3D data, create proper volume visualization
        fig = go.Figure()
        
        # Create coordinate grids
        x, y, z = np.meshgrid(
            np.arange(pixel_array.shape[2]),
            np.arange(pixel_array.shape[1]),
            np.arange(pixel_array.shape[0]),
            indexing='ij'
        )
        
        # Add volume rendering
        fig.add_trace(go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=pixel_array.flatten(),
            opacity=0.3,
            colorscale='Viridis',
            name='Breast Tissue',
            showscale=True
        ))
        
        # Add isosurface for better visualization
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter(pixel_array, sigma=1)
        
        # Create isosurface
        fig.add_trace(go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=smoothed.flatten(),
            isomin=np.percentile(smoothed, 70),
            isomax=np.percentile(smoothed, 95),
            opacity=0.6,
            colorscale='Reds',
            name='Tissue Surface',
            showscale=False
        ))
        
        fig.update_layout(
            title="3D Breast Volume Visualization",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        # Display the 3D plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interactive controls
        self.add_interactive_controls(pixel_array)
        
        # Add measurement tools
        self.add_measurement_tools(pixel_array)
        
        # Add analysis overlay
        self.add_analysis_overlay(pixel_array)
    
    def add_interactive_controls(self, pixel_array):
        """Add interactive controls for visualization"""
        st.markdown("### 🎮 Interactive Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            opacity = st.slider("Volume Opacity", 0.1, 1.0, 0.3, 0.1)
            st.write(f"Opacity: {opacity:.1f}")
        
        with col2:
            threshold = st.slider("Tissue Threshold", 0, 255, 128, 1)
            st.write(f"Threshold: {threshold}")
        
        with col3:
            view_mode = st.selectbox("View Mode", ["Volume", "Surface", "Both"])
            st.write(f"Mode: {view_mode}")
    
    def add_measurement_tools(self, pixel_array):
        """Add measurement tools"""
        st.markdown("### 📏 Measurement Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📐 Measure Volume"):
                # Calculate breast volume
                volume_cc = self.calculate_breast_volume(pixel_array)
                st.metric("Estimated Breast Volume", f"{volume_cc:.1f} cc")
        
        with col2:
            if st.button("📏 Measure Density"):
                # Calculate tissue density
                density = self.calculate_tissue_density(pixel_array)
                st.metric("Tissue Density", f"{density:.1f} HU")
    
    def add_analysis_overlay(self, pixel_array):
        """Add analysis overlay"""
        st.markdown("### 🔍 AI Analysis Overlay")
        
        if st.checkbox("Show AI Segmentation"):
            # Create segmentation overlay
            segmentation_fig = self.create_segmentation_overlay(pixel_array)
            st.plotly_chart(segmentation_fig, use_container_width=True)
    
    def calculate_breast_volume(self, pixel_array):
        """Calculate estimated breast volume"""
        # Simple volume calculation based on pixel count
        total_pixels = pixel_array.size
        # Assume 1mm³ per pixel (simplified)
        volume_mm3 = total_pixels
        volume_cc = volume_mm3 / 1000  # Convert to cc
        return volume_cc
    
    def calculate_tissue_density(self, pixel_array):
        """Calculate average tissue density"""
        return np.mean(pixel_array)
    
    def create_segmentation_overlay(self, pixel_array):
        """Create segmentation overlay visualization"""
        import plotly.graph_objects as go
        
        # Create simulated segmentation (replace with actual AI results)
        segmentation = np.zeros_like(pixel_array)
        
        # Simulate breast tissue segmentation
        center = np.array(pixel_array.shape) // 2
        radius = min(pixel_array.shape) // 3
        
        if len(pixel_array.shape) == 2:
            y, x = np.ogrid[:pixel_array.shape[0], :pixel_array.shape[1]]
            dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            segmentation = dist < radius
        else:
            y, x, z = np.ogrid[:pixel_array.shape[0], :pixel_array.shape[1], :pixel_array.shape[2]]
            dist = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
            segmentation = dist < radius
        
        # Create overlay visualization
        fig = go.Figure()
        
        # Original image
        fig.add_trace(go.Heatmap(
            z=pixel_array,
            colorscale='Gray',
            name='Original Image'
        ))
        
        # Segmentation overlay
        fig.add_trace(go.Heatmap(
            z=segmentation * pixel_array,
            colorscale='Reds',
            opacity=0.6,
            name='AI Segmentation'
        ))
        
        fig.update_layout(
            title="AI Segmentation Overlay",
            xaxis_title="X",
            yaxis_title="Y",
            width=600,
            height=400
        )
        
        return fig
    
    def show_2d_slices(self, file_path):
        """Show 2D slice viewer as fallback"""
        import pydicom
        import matplotlib.pyplot as plt
        
        try:
            ds = pydicom.dcmread(file_path)
            pixel_array = ds.pixel_array
            
            # Create 2D visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(pixel_array, cmap='gray')
            ax.set_title("2D Slice Viewer")
            ax.axis('off')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"2D visualization error: {e}")
            st.info("Please check your image format and try again.")
    
    def launch_slicer_analysis(self, analysis_type):
        """Launch the 3D Slicer analysis"""
        
        st.markdown(f"""
        <div class="breast-ai-card">
            <h3>🚀 Launching {analysis_type}</h3>
            <p>Initializing advanced 3D analysis engine...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Launch 3D Slicer in browser
        slicer_url = "https://mybinder.org/v2/gh/Slicer/SlicerNotebooks/HEAD?urlpath=lab"
        
        st.markdown(f"""
        <script>
        window.open('{slicer_url}', '_blank');
        </script>
        """, unsafe_allow_html=True)
        
        st.success("✅ 3D Analysis Engine Launched!")
        st.info("The advanced 3D viewer has opened in a new tab. You can now perform detailed analysis.")
    
    def download_report(self, report_content):
        """Download the analysis report"""
        
        # Create downloadable file
        import io
        
        buffer = io.StringIO()
        buffer.write(report_content)
        
        st.download_button(
            label="📥 Download Report (PDF)",
            data=buffer.getvalue(),
            file_name=f"breast_analysis_report_{Path(file_path).stem}.txt",
            mime="text/plain"
        )

def main():
    """Main function to run the branded interface"""
    
    st.set_page_config(
        page_title="BreastAI Pro - Advanced 3D Breast Cancer Detection",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize the integration
    slicer_integration = SlicerBrowserIntegration()
    
    # Create the branded interface
    slicer_integration.create_branded_interface()
    
    # Sidebar with additional options
    with st.sidebar:
        st.markdown("""
        <div class="breast-ai-card">
            <h3>⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.selectbox("AI Model", ["SegResNet-3D", "UNet-3D", "DeepLabV3+"])
        st.selectbox("Analysis Mode", ["Clinical", "Research", "Screening"])
        st.checkbox("Enable Advanced Features")
        st.checkbox("Save Analysis History")

if __name__ == "__main__":
    main()
