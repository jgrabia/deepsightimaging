#!/usr/bin/env python3
"""
Simple 3D Slicer Viewer for Breast Cancer Detection
Works with local MONAI model instead of external services
"""

import streamlit as st
import requests
import tempfile
import os
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def main():
    """Main function for simple 3D Slicer viewer"""
    
    st.set_page_config(
        page_title="BreastAI Pro - 3D Viewer",
        page_icon="🏥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 BreastAI Pro - 3D Viewer</h1>
        <p>Advanced AI-Powered Breast Cancer Detection & 3D Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📁 Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Choose a DICOM file for analysis",
        type=['dcm', 'dicom', 'nii', 'nii.gz']
    )
    
    if uploaded_file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Process with MONAI
        if st.button("🔬 Run AI Analysis"):
            with st.spinner("Running AI analysis..."):
                result = run_monai_analysis(file_path)
                if result:
                    display_3d_results(result, file_path)
        
        # Clean up
        os.unlink(file_path)

def run_monai_analysis(file_path):
    """Run analysis with local MONAI model"""
    
    try:
        # MONAI server URL (adjust if needed)
        monai_url = "http://localhost:8000/infer/advanced_breast_cancer_detection"
        
        # Prepare file for upload
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            
            # Send to MONAI server
            response = requests.post(monai_url, files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"MONAI server error: {response.status_code}")
                return None
                
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        return None

def display_3d_results(result, file_path):
    """Display 3D visualization of results"""
    
    st.markdown("""
    <div class="result-card">
        <h3>✅ AI Analysis Complete</h3>
        <p>Your breast cancer analysis has been completed successfully.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎨 3D Visualization", "📊 Analysis Results", "📋 Export"])
    
    with tab1:
        st.markdown("### 🎨 Interactive 3D Visualization")
        
        if 'pred' in result:
            # Create 3D visualization using Plotly
            create_3d_visualization(result['pred'])
        else:
            st.warning("No prediction data available for 3D visualization")
    
    with tab2:
        st.markdown("### 📊 Analysis Results")
        
        # Display prediction statistics
        if 'pred' in result:
            pred_data = np.array(result['pred'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction Shape", f"{pred_data.shape[0]} x {pred_data.shape[1]}")
            
            with col2:
                st.metric("Min Value", f"{pred_data.min():.3f}")
            
            with col3:
                st.metric("Max Value", f"{pred_data.max():.3f}")
            
            # Show prediction distribution
            st.markdown("#### 📈 Prediction Distribution")
            hist_data = pred_data.flatten()
            st.histogram_chart(hist_data)
            
            # Show label information
            if 'label_names' in result:
                st.markdown("#### 🏷️ Detected Labels")
                for i, label in enumerate(result['label_names']):
                    st.write(f"**{i}**: {label}")
    
    with tab3:
        st.markdown("### 📋 Export Results")
        
        # Export options
        if st.button("📥 Download 3D Results"):
            export_3d_results(result)
        
        if st.button("📄 Generate Report"):
            generate_report(result)

def create_3d_visualization(pred_data):
    """Create 3D visualization using Plotly"""
    
    try:
        # Convert prediction data to numpy array
        pred_array = np.array(pred_data)
        
        # Create 3D surface plot
        x, y = np.meshgrid(np.arange(pred_array.shape[1]), np.arange(pred_array.shape[0]))
        
        fig = go.Figure(data=[go.Surface(z=pred_array, x=x, y=y, colorscale='Viridis')])
        
        fig.update_layout(
            title="3D Breast Cancer Detection Results",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Prediction Confidence"
            ),
            width=800,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Add 2D slice viewer
        st.markdown("#### 📐 2D Slice Viewer")
        
        # Create slider for slice selection
        slice_idx = st.slider("Select Slice", 0, pred_array.shape[0]-1, pred_array.shape[0]//2)
        
        # Show selected slice
        fig_2d = go.Figure(data=go.Heatmap(z=pred_array[slice_idx, :], colorscale='Viridis'))
        fig_2d.update_layout(
            title=f"Slice {slice_idx}",
            xaxis_title="X",
            yaxis_title="Y"
        )
        
        st.plotly_chart(fig_2d)
        
    except Exception as e:
        st.error(f"Error creating 3D visualization: {e}")

def export_3d_results(result):
    """Export 3D results"""
    
    try:
        # Create downloadable file
        import json
        
        # Save results as JSON
        result_str = json.dumps(result, indent=2)
        
        st.download_button(
            label="📥 Download 3D Results (JSON)",
            data=result_str,
            file_name="breast_3d_results.json",
            mime="application/json"
        )
        
        st.success("✅ Results exported successfully!")
        
    except Exception as e:
        st.error(f"Error exporting results: {e}")

def generate_report(result):
    """Generate analysis report"""
    
    try:
        # Create report content
        report = f"""
        BreastAI Pro Analysis Report
        ===========================
        
        Analysis Date: {Path(__file__).stat().st_mtime}
        
        Results Summary:
        - Prediction Shape: {result.get('pred_shape', 'Unknown')}
        - Min Value: {result.get('pred_min', 'Unknown')}
        - Max Value: {result.get('pred_max', 'Unknown')}
        - Mean Value: {result.get('pred_mean', 'Unknown')}
        
        Detected Labels:
        {result.get('label_names', ['Unknown'])}
        
        Analysis completed successfully.
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name="breast_analysis_report.txt",
            mime="text/plain"
        )
        
        st.success("✅ Report generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()




