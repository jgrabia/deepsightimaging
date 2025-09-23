#!/usr/bin/env python3
"""
Lightweight DBT Viewer Test Script
Tests the enhanced DBT viewer functionality locally without heavy dependencies
"""

import streamlit as st
import pydicom
import numpy as np
import tempfile
import os
from pathlib import Path
import io
from PIL import Image
import matplotlib.pyplot as plt

# Configure Streamlit for better performance
st.set_page_config(
    page_title="DBT Viewer Test",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_test_dbt_data():
    """Create synthetic DBT data for testing"""
    # Create a synthetic 3D volume (simulating 66 slices)
    height, width = 512, 512
    num_slices = 66
    
    # Create synthetic breast tissue pattern
    volume = np.zeros((num_slices, height, width), dtype=np.uint16)
    
    for z in range(num_slices):
        # Create a simple synthetic breast pattern
        y, x = np.ogrid[:height, :width]
        
        # Breast outline (ellipse)
        center_y, center_x = height // 2, width // 2
        a, b = height // 3, width // 3  # Semi-major and semi-minor axes
        
        # Distance from ellipse center
        ellipse_mask = ((y - center_y) / a) ** 2 + ((x - center_x) / b) ** 2
        
        # Create tissue pattern
        tissue = np.zeros_like(ellipse_mask)
        tissue[ellipse_mask <= 1] = 1000  # Base tissue
        
        # Add some variation
        noise = np.random.normal(0, 50, (height, width))
        tissue = np.clip(tissue + noise, 0, 4095)
        
        # Add some "lesions" (bright spots)
        if z > 20 and z < 50:  # Add lesions in middle slices
            lesion_y, lesion_x = height // 2 + 50, width // 2 - 30
            lesion_mask = ((y - lesion_y) ** 2 + (x - lesion_x) ** 2) < 400
            tissue[lesion_mask] = 2500
        
        volume[z] = tissue.astype(np.uint16)
    
    return volume

def show_dbt_viewer():
    """Enhanced DBT viewer for testing"""
    st.header("🏥 DBT Viewer Test")
    
    st.info("Testing enhanced DBT viewer functionality with synthetic data.")
    
    # Initialize session state
    if 'dbt_slice_idx' not in st.session_state:
        st.session_state.dbt_slice_idx = 33  # Middle slice
    if 'dbt_volume_data' not in st.session_state:
        st.session_state.dbt_volume_data = None
    
    # File upload or use synthetic data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a DBT DICOM file", type=['dcm'], key="dbt_test_uploader")
    
    with col2:
        st.write("**Or use test data:**")
        if st.button("🧪 Load Synthetic DBT Data"):
            st.session_state.dbt_volume_data = load_test_dbt_data()
            st.success("✅ Synthetic DBT data loaded!")
    
    # Process DBT data
    img_array = None
    metadata = {}
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load DICOM
            ds = pydicom.dcmread(tmp_path)
            metadata = {
                'PatientID': ds.get('PatientID', 'Test Patient'),
                'StudyDate': ds.get('StudyDate', '2024-01-01'),
                'Modality': ds.get('Modality', 'MG'),
                'Manufacturer': ds.get('Manufacturer', 'Test Manufacturer'),
                'SeriesDescription': ds.get('SeriesDescription', 'DBT Test Series'),
                'ViewPosition': ds.get('ViewPosition', 'MLO'),
                'Laterality': ds.get('ImageLaterality', 'L'),
                'CompressionForce': ds.get('CompressionForce', 'N/A')
            }
            
            if hasattr(ds, 'pixel_array'):
                img_array = ds.pixel_array
                st.session_state.dbt_volume_data = img_array
        
        except Exception as e:
            st.error(f"Error reading DBT file: {e}")
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    elif st.session_state.dbt_volume_data is not None:
        img_array = st.session_state.dbt_volume_data
        metadata = {
            'PatientID': 'SYNTHETIC-001',
            'StudyDate': '2024-01-01',
            'Modality': 'MG',
            'Manufacturer': 'Synthetic Generator',
            'SeriesDescription': 'Synthetic DBT Volume',
            'ViewPosition': 'MLO',
            'Laterality': 'L',
            'CompressionForce': 'N/A'
        }
    
    if img_array is not None:
        # Display metadata
        st.subheader("📋 DBT Volume Information")
        
        with st.expander("🏥 Patient & Study Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Patient ID:** {metadata.get('PatientID', 'N/A')}")
                st.write(f"**Study Date:** {metadata.get('StudyDate', 'N/A')}")
                st.write(f"**Modality:** {metadata.get('Modality', 'N/A')}")
                st.write(f"**Manufacturer:** {metadata.get('Manufacturer', 'N/A')}")
            
            with col2:
                st.write(f"**Series Description:** {metadata.get('SeriesDescription', 'N/A')}")
                st.write(f"**View Position:** {metadata.get('ViewPosition', 'N/A')}")
                st.write(f"**Laterality:** {metadata.get('Laterality', 'N/A')}")
                st.write(f"**Compression Force:** {metadata.get('CompressionForce', 'N/A')}")
            
            with col3:
                st.write(f"**Volume Shape:** {img_array.shape}")
                st.write(f"**Data Type:** {img_array.dtype}")
                st.write(f"**Min Value:** {img_array.min()}")
                st.write(f"**Max Value:** {img_array.max()}")
        
        if len(img_array.shape) == 3:
            st.subheader(f"🔍 DBT Volume Analysis: {img_array.shape[0]} slices")
            
            # Volume statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Slices", img_array.shape[0])
            with col2:
                st.metric("Width", f"{img_array.shape[1]} px")
            with col3:
                st.metric("Height", f"{img_array.shape[2]} px")
            with col4:
                volume_size_mb = img_array.nbytes / (1024 * 1024)
                st.metric("Volume Size", f"{volume_size_mb:.1f} MB")
            
            # Enhanced slice navigation
            st.subheader("🎛️ Advanced Navigation")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                current_slice_idx = st.session_state.get('dbt_slice_idx', img_array.shape[0]//2)
                slice_idx = st.slider(
                    "Slice Navigation", 
                    0, 
                    img_array.shape[0]-1, 
                    current_slice_idx,
                    key="test_slice_selector"
                )
                st.session_state.dbt_slice_idx = slice_idx
            
            with col2:
                st.write("**Quick Navigation:**")
                if st.button("🏠 Start", key="test_nav_start"):
                    st.session_state.dbt_slice_idx = 0
                    st.rerun()
                if st.button("🎯 Middle", key="test_nav_middle"):
                    st.session_state.dbt_slice_idx = img_array.shape[0] // 2
                    st.rerun()
                if st.button("🏁 End", key="test_nav_end"):
                    st.session_state.dbt_slice_idx = img_array.shape[0] - 1
                    st.rerun()
            
            with col3:
                st.write("**Step Navigation:**")
                if st.button("⏪ -10", key="test_nav_minus_10"):
                    new_idx = max(0, slice_idx - 10)
                    st.session_state.dbt_slice_idx = new_idx
                    st.rerun()
                if st.button("⏩ +10", key="test_nav_plus_10"):
                    new_idx = min(img_array.shape[0]-1, slice_idx + 10)
                    st.session_state.dbt_slice_idx = new_idx
                    st.rerun()
            
            # Display current slice
            current_slice = img_array[slice_idx, :, :]
            
            st.subheader(f"📊 Slice {slice_idx+1}/{img_array.shape[0]} Analysis")
            
            # Enhanced window/level controls
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("🖼️ Display Controls")
                
                # Window/level presets
                preset = st.selectbox("Window/Level Preset", 
                                    ["Custom", "Soft Tissue", "Bone", "Lung", "Auto"])
                
                if preset == "Auto":
                    window_center = int(current_slice.mean())
                    window_width = int(current_slice.std() * 6)
                elif preset == "Soft Tissue":
                    window_center = 40
                    window_width = 400
                elif preset == "Bone":
                    window_center = 300
                    window_width = 1500
                elif preset == "Lung":
                    window_center = -600
                    window_width = 1500
                else:  # Custom
                    window_center = st.slider("Window Center", 
                                            int(current_slice.min()), 
                                            int(current_slice.max()), 
                                            int(current_slice.mean()))
                    window_width = st.slider("Window Width", 
                                           1, 
                                           int(current_slice.max() - current_slice.min()), 
                                           int(current_slice.std() * 4))
                
                # Apply window/level
                lower_bound = window_center - window_width // 2
                upper_bound = window_center + window_width // 2
                windowed_slice = np.clip(current_slice, lower_bound, upper_bound)
                windowed_slice = ((windowed_slice - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
                
                # Image enhancement options
                st.subheader("🎨 Image Enhancement")
                enhance_contrast = st.checkbox("Enhance Contrast")
                reduce_noise = st.checkbox("Reduce Noise")
                
                if enhance_contrast:
                    # Simple contrast enhancement
                    windowed_slice = np.clip(windowed_slice * 1.2, 0, 255).astype(np.uint8)
                
                if reduce_noise:
                    # Simple noise reduction (median filter approximation)
                    from scipy import ndimage
                    windowed_slice = ndimage.median_filter(windowed_slice, size=3)
            
            with col1:
                # Display image
                st.image(windowed_slice, caption=f"DBT Slice {slice_idx+1}/{img_array.shape[0]}", use_column_width=True)
                
                # Show slice statistics
                st.caption(f"**Slice Statistics:** Min: {current_slice.min()}, Max: {current_slice.max()}, Mean: {current_slice.mean():.1f}, Std: {current_slice.std():.1f}")
                
                # Navigation buttons
                nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
                
                with nav_col1:
                    if st.button("⏮️", help="First slice", key="test_nav_first"):
                        st.session_state.dbt_slice_idx = 0
                        st.rerun()
                
                with nav_col2:
                    if st.button("⏪", help="Previous slice", key="test_nav_prev"):
                        if slice_idx > 0:
                            st.session_state.dbt_slice_idx = slice_idx - 1
                            st.rerun()
                
                with nav_col3:
                    if st.button("⏯️", help="Play/Pause", key="test_nav_play"):
                        st.info("Auto-play feature coming soon!")
                
                with nav_col4:
                    if st.button("⏩", help="Next slice", key="test_nav_next"):
                        if slice_idx < img_array.shape[0]-1:
                            st.session_state.dbt_slice_idx = slice_idx + 1
                            st.rerun()
                
                with nav_col5:
                    if st.button("⏭️", help="Last slice", key="test_nav_last"):
                        st.session_state.dbt_slice_idx = img_array.shape[0] - 1
                        st.rerun()
            
            # Advanced features
            st.subheader("🔬 Advanced Analysis")
            
            tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Histogram", "💾 Export"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Current Slice:**")
                    st.write(f"- Min: {current_slice.min()}")
                    st.write(f"- Max: {current_slice.max()}")
                    st.write(f"- Mean: {current_slice.mean():.2f}")
                    st.write(f"- Std Dev: {current_slice.std():.2f}")
                    st.write(f"- Median: {np.median(current_slice):.2f}")
                
                with col2:
                    st.write("**Entire Volume:**")
                    st.write(f"- Min: {img_array.min()}")
                    st.write(f"- Max: {img_array.max()}")
                    st.write(f"- Mean: {img_array.mean():.2f}")
                    st.write(f"- Std Dev: {img_array.std():.2f}")
                    st.write(f"- Total Pixels: {img_array.size:,}")
            
            with tab2:
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(current_slice.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram - Slice {slice_idx+1}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with tab3:
                st.write("**Export Options:**")
                
                if st.button("💾 Export Current Slice (PNG)"):
                    img = Image.fromarray(windowed_slice)
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download PNG",
                        data=byte_im,
                        file_name=f"dbt_slice_{slice_idx+1}.png",
                        mime="image/png"
                    )
        
        else:
            st.warning("This doesn't appear to be a DBT volume (not 3D)")
            if len(img_array.shape) == 2:
                st.subheader("2D DICOM Image")
                st.image(img_array, caption="DICOM Image", use_column_width=True)

def main():
    """Main function"""
    st.title("🏥 DBT Viewer Test Application")
    
    st.markdown("""
    This is a lightweight test version of the enhanced DBT viewer.
    It focuses on the core functionality without heavy dependencies.
    """)
    
    show_dbt_viewer()

if __name__ == "__main__":
    main()

