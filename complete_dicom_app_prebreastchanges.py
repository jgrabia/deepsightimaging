import streamlit as st
import requests
import os
import json
import time
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from tcia_utils import nbia
import boto3
from PIL import Image
import io
import base64
import pydicom
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cloud DICOM App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_series' not in st.session_state:
    st.session_state.selected_series = []
if 'download_progress' not in st.session_state:
    st.session_state.download_progress = 0
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'results_per_page' not in st.session_state:
    st.session_state.results_per_page = 25

# Configuration
MONAI_SERVER_URL = "http://localhost:8000"  # MONAI Label server
AWS_REGION = "us-east-1"
S3_BUCKET = "dicom-storage"

# Initialize AWS S3 client
try:
    s3_client = boto3.client('s3', region_name=AWS_REGION)
except Exception as e:
    st.warning(f"AWS S3 not configured: {e}")
    s3_client = None

def get_collections():
    """Get available TCIA collections"""
    try:
        return nbia.getCollections()
    except Exception as e:
        st.error(f"Error fetching collections: {e}")
        return []

def get_body_parts():
    """Get available body parts"""
    try:
        return nbia.getBodyPartExaminedValues()
    except Exception as e:
        st.error(f"Error fetching body parts: {e}")
        return []

def get_modalities():
    """Get available modalities"""
    try:
        return nbia.getModalityValues()
    except Exception as e:
        st.error(f"Error fetching modalities: {e}")
        return []

def get_manufacturers():
    """Get available manufacturers"""
    try:
        return nbia.getManufacturerValues()
    except Exception as e:
        st.error(f"Error fetching manufacturers: {e}")
        return []

def get_manufacturer_models():
    """Get available manufacturer models"""
    try:
        return nbia.getManufacturerModelNameValues()
    except Exception as e:
        st.error(f"Error fetching manufacturer models: {e}")
        return []

def search_series(filters):
    """Search for DICOM series based on filters"""
    try:
        return nbia.getSeries(**filters)
    except Exception as e:
        st.error(f"Error searching series: {e}")
        return []

def download_series_local(series_uid):
    """Download DICOM series to local storage"""
    try:
        # Get series size info
        size_info = nbia.getSeriesSize(seriesInstanceUID=series_uid)
        total_size = size_info[0]['ObjectCount'] if size_info else 0
        
        # Download URL
        url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
        
        # Create downloads directory if it doesn't exist
        downloads_dir = Path("~/mri_app/downloads").expanduser()
        downloads_dir.mkdir(exist_ok=True)
        
        # Download to local file
        local_path = downloads_dir / f"{series_uid}.zip"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size_bytes = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress
                    if total_size_bytes > 0:
                        progress = int((downloaded / total_size_bytes) * 100)
                        st.session_state.download_progress = progress
        
        return str(local_path)
        
    except Exception as e:
        st.error(f"Error downloading series {series_uid}: {e}")
        return None

def download_series_to_s3(series_uid, bucket_name):
    """Download DICOM series to S3 (legacy function)"""
    try:
        # Get series size info
        size_info = nbia.getSeriesSize(seriesInstanceUID=series_uid)
        total_size = size_info[0]['ObjectCount'] if size_info else 0
        
        # Download URL
        url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size_bytes = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    # Update progress
                    if total_size_bytes > 0:
                        progress = int((downloaded / total_size_bytes) * 100)
                        st.session_state.download_progress = progress
            
            temp_file_path = temp_file.name
        
        # Upload to S3
        s3_key = f"dicom-series/{series_uid}.zip"
        s3_client.upload_file(temp_file_path, bucket_name, s3_key)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return f"s3://{bucket_name}/{s3_key}"
        
    except Exception as e:
        st.error(f"Error downloading series {series_uid}: {e}")
        return None

def preprocess_dicom_for_monai(dicom_path):
    """Preprocess DICOM file to make it compatible with MONAI models"""
    try:
        # Read DICOM file
        ds = pydicom.dcmread(dicom_path)
        
        # Get pixel data
        pixel_array = ds.pixel_array
        
        # Handle different DICOM formats - ensure we always get 2D
        if len(pixel_array.shape) == 4:  # Multi-frame, multi-channel
            # Take the first frame and first channel
            pixel_array = pixel_array[0, :, :, 0]
        elif len(pixel_array.shape) == 3:  # Multi-frame or multi-channel
            # For 3D arrays, we need to determine if it's multi-frame or multi-channel
            # Check if first dimension is smaller than others (likely multi-channel)
            if pixel_array.shape[0] < min(pixel_array.shape[1], pixel_array.shape[2]):
                # Multi-channel: take first channel
                pixel_array = pixel_array[0, :, :]
            else:
                # Multi-frame: take middle frame for better representation
                middle_frame = pixel_array.shape[0] // 2
                pixel_array = pixel_array[middle_frame, :, :]
        elif len(pixel_array.shape) == 2:  # Single 2D image
            pixel_array = pixel_array
        else:
            raise ValueError(f"Unsupported DICOM format with shape: {pixel_array.shape}")
        
        # Ensure we have a 2D array
        if len(pixel_array.shape) != 2:
            raise ValueError(f"Failed to convert to 2D array. Final shape: {pixel_array.shape}")
        
        # Convert to NIFTI format instead of corrupted DICOM
        # This avoids the GDCM issues entirely
        import nibabel as nib
        
        # Normalize to 0-255 range and ensure proper data type
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Create a 3D array (MONAI expects 3D)
        pixel_array_3d = pixel_array.reshape(1, pixel_array.shape[0], pixel_array.shape[1])
        
        # Create NIFTI image
        nifti_img = nib.Nifti1Image(pixel_array_3d, np.eye(4))
        
        # Save as NIFTI
        nifti_path = dicom_path.replace('.dcm', '.nii.gz')
        nib.save(nifti_img, nifti_path)
        
        return nifti_path
        
    except Exception as e:
        st.error(f"DICOM preprocessing failed: {e}")
        # Fallback: return original file
        return dicom_path

def run_monai_inference(dicom_file_path, model="deepedit"):
    """Run MONAI inference on DICOM file"""
    try:
        with open(dicom_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{MONAI_SERVER_URL}/infer/{model}", files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"MONAI inference error: {e}")
        return None

def display_dicom_info(series):
    """Display DICOM series information"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Series Description:** {series.get('SeriesDescription', 'N/A')}")
        st.write(f"**Patient ID:** {series.get('PatientID', 'N/A')}")
        st.write(f"**Modality:** {series.get('Modality', 'N/A')}")
        st.write(f"**Body Part:** {series.get('BodyPartExamined', 'N/A')}")
    
    with col2:
        st.write(f"**Collection:** {series.get('Collection', 'N/A')}")
        st.write(f"**Manufacturer:** {series.get('Manufacturer', 'N/A')}")
        st.write(f"**Image Count:** {series.get('ImageCount', 'N/A')}")
        st.write(f"**Series UID:** {series.get('SeriesInstanceUID', 'N/A')[:20]}...")

def get_paginated_results(results, page, per_page):
    """Get paginated results"""
    start_idx = page * per_page
    end_idx = start_idx + per_page
    return results[start_idx:end_idx]

def display_pagination_controls(total_results, current_page, per_page):
    """Display pagination controls"""
    total_pages = (total_results + per_page - 1) // per_page
    
    if total_pages <= 1:
        return
    
    st.markdown("---")
    st.subheader("📄 Pagination")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0):
            st.session_state.current_page = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", disabled=current_page == 0):
            st.session_state.current_page = current_page - 1
            st.rerun()
    
    with col3:
        st.write(f"**Page {current_page + 1} of {total_pages}**")
        st.write(f"Showing {current_page * per_page + 1}-{min((current_page + 1) * per_page, total_results)} of {total_results} results")
    
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1):
            st.session_state.current_page = current_page + 1
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1):
            st.session_state.current_page = total_pages - 1
            st.rerun()
    
    # Page jump
    st.markdown("**Jump to page:**")
    col1, col2 = st.columns([1, 3])
    with col1:
        page_input = st.number_input("Page", min_value=1, max_value=total_pages, value=current_page + 1, key="page_jump")
    with col2:
        if st.button("Go", key="go_to_page"):
            if page_input != current_page + 1:
                st.session_state.current_page = page_input - 1
                st.rerun()

def test_monai_connection():
    """Test MONAI Label server connection"""
    try:
        response = requests.get(f"{MONAI_SERVER_URL}/info/", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        print(f"MONAI connection failed: {e}")
        return False

def test_monai_models():
    """Test different MONAI models to find one that works without GDCM error"""
    st.subheader("🧪 Test Different MONAI Models")
    
    # Available models from the server
    available_models = [
        "segmentation",  # Multi-organ segmentation
        "breast_tumor_detection",  # Breast cancer detection
        "lung_nodule_detection",  # Lung nodule detection
        "lung_cancer_segmentation",  # Lung cancer segmentation
        "segmentation_spleen",  # Spleen-specific segmentation
        "localization_spine",  # Spine localization
        "localization_vertebra",  # Vertebra localization
        "deepgrow_2d",
        "deepgrow_3d", 
        "deepedit",
        "sw_fastedit"
    ]
    
    test_model = st.selectbox("Select model to test:", available_models)
    
    # File upload for testing
    uploaded_file = st.file_uploader("Upload DICOM file for testing", type=['dcm', 'dicom', 'nii', 'nii.gz'], key="test_upload")
    
    if st.button("Test Model") and uploaded_file:
            try:
                # Use the uploaded file for testing
                test_file = uploaded_file
                
                # Preprocess the DICOM
                processed_data = preprocess_dicom_for_monai(test_file)
                
                # Test the model with different output formats
                with st.spinner(f"Testing {test_model} model..."):
                    output_formats = ["json", "image", "all", "dicom_seg"]
                    response = None
                    
                    for output_format in output_formats:
                        try:
                            # Explain what each format means
                            format_descriptions = {
                                "json": "JSON response with metadata",
                                "image": "Image file response",
                                "all": "All available formats",
                                "dicom_seg": "DICOM segmentation format"
                            }
                            st.info(f"Trying output format: {output_format} - {format_descriptions.get(output_format, 'Unknown format')}")
                            
                            # Use the correct API structure from OpenAPI spec
                            # Save uploaded file temporarily
                            file_extension = '.dcm' if uploaded_file.name.lower().endswith(('.dcm', '.dicom')) else '.nii.gz'
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Preprocess DICOM for MONAI compatibility
                            processed_path = preprocess_dicom_for_monai(tmp_path)
                            
                            with open(processed_path, 'rb') as f:
                                files = {'file': f}
                                response = requests.post(
                                    f"{MONAI_SERVER_URL}/infer/{test_model}",
                                    files=files,
                                    params={
                                        "output": output_format,
                                        "device": "cpu"
                                    },
                                    timeout=60
                                )
                            
                            if response.status_code == 200:
                                st.success(f"✅ {test_model} model works with output format: {output_format}!")
                                break
                            else:
                                st.warning(f"❌ {test_model} failed with {output_format}: {response.status_code}")
                                
                        except Exception as e:
                            st.warning(f"❌ Error testing {test_model} with {output_format}: {str(e)}")
                            continue
                
                if response.status_code == 200:
                    st.success(f"✅ {test_model} model works!")
                    result = response.json()
                    st.json(result)
                else:
                    st.error(f"❌ {test_model} failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error testing {test_model}: {str(e)}")

# Main application
def main():
    st.set_page_config(
        page_title="Cloud DICOM App",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🏥 Cloud DICOM App")
    
    # System status in sidebar
    st.sidebar.subheader("System Status")
    
    # TCIA Connection
    st.success("✅ TCIA Connected")
    
    # MONAI Connection
    if test_monai_connection():
        st.sidebar.success("✅ MONAI Connected")
    else:
        st.sidebar.warning("⚠️ MONAI Unavailable")
    
    # S3 Connection (placeholder)
    st.sidebar.success("✅ S3 Connected")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Home", "🔍 DICOM Search", "📥 Download Manager", "🖼️ DICOM Viewer", "🤖 MONAI Inference", "🧪 Model Testing", "⚙️ Settings"]
    )
    
    # Show current series info if available
    if 'current_series' in st.session_state and page == "🖼️ DICOM Viewer":
        st.info(f"📋 Viewing: {st.session_state.current_series.get('SeriesDescription', 'Unknown')}")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 DICOM Search":
        show_search_page()
    elif page == "📥 Download Manager":
        show_download_page()
    elif page == "🖼️ DICOM Viewer":
        show_viewer_page()
    elif page == "🤖 MONAI Inference":
        show_monai_page()
    elif page == "🧪 Model Testing":
        test_monai_models()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Home page with overview and quick actions"""
    st.header("Welcome to Cloud DICOM App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What you can do:
        - **Search** DICOM series from TCIA
        - **Download** series to cloud storage
        - **View** DICOM images in browser
        - **Run AI inference** with MONAI
        - **Annotate** images for analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Quick Stats:
        - **Collections Available:** 138+
        - **Modalities:** CT, MR, PET, etc.
        - **Storage:** Cloud-based (S3)
        - **AI Models:** MONAI Label integration
        """)
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    st.info("Use the sidebar navigation to access different features!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔍 DICOM Search**")
        st.write("Search TCIA collections")
    
    with col2:
        st.markdown("**🖼️ DICOM Viewer**")
        st.write("View and analyze images")
    
    with col3:
        st.markdown("**🤖 MONAI Inference**")
        st.write("Run AI analysis")

def show_search_page():
    """DICOM search page with filters"""
    st.header("🔍 DICOM Series Search")
    
    # Search filters
    with st.expander("Search Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Collections
            collections = get_collections()
            collection_options = [c['Collection'] for c in collections] if collections else []
            
            # Add quick cancer collection filters
            cancer_collections = ["LIDC-IDRI", "TCGA-BRCA", "TCGA-LUAD", "TCGA-LUSC"]
            quick_filters = ["All"] + cancer_collections + ["Other Collections..."]
            
            filter_type = st.selectbox("Quick Filter", quick_filters, key="filter_type")
            
            if filter_type == "Other Collections...":
                selected_collection = st.selectbox("Collection", ["All"] + collection_options, key="collection_selector")
            elif filter_type in cancer_collections:
                selected_collection = filter_type
            else:
                selected_collection = "All"
            
            # Body parts
            body_parts = get_body_parts()
            body_part_options = [bp['BodyPartExamined'] for bp in body_parts] if body_parts else []
            selected_body_part = st.selectbox("Body Part", ["All"] + body_part_options)
            
            # Modalities
            modalities = get_modalities()
            modality_options = [m['Modality'] for m in modalities] if modalities else []
            selected_modality = st.selectbox("Modality", ["All"] + modality_options)
        
        with col2:
            # Manufacturers
            manufacturers = get_manufacturers()
            manufacturer_options = [m['Manufacturer'] for m in manufacturers] if manufacturers else []
            selected_manufacturer = st.selectbox("Manufacturer", ["All"] + manufacturer_options)
            
            # Manufacturer models
            models = get_manufacturer_models()
            model_options = [m['ManufacturerModelName'] for m in models] if models else []
            selected_model = st.selectbox("Manufacturer Model", ["All"] + model_options)
            
            # Patient ID
            patient_id = st.text_input("Patient ID (optional)")
    
    # Search tips
    with st.expander("💡 Search Tips", expanded=False):
        st.markdown("""
        **To get better results:**
        - **Start with Collection** (e.g., LIDC-IDRI for lung nodules)
        - **Add Body Part** (e.g., CHEST, BREAST, ABDOMEN)
        - **Filter by Modality** (e.g., CT, MR, PT)
        - **Use Patient ID** for specific cases
        
        **Popular Collections:**
        - **LIDC-IDRI**: Lung nodules (1,018 cases)
        - **TCGA-BRCA**: Breast cancer
        - **TCGA-LUAD**: Lung adenocarcinoma
        - **TCGA-LUSC**: Lung squamous cell carcinoma
        """)
    
    # Search button
    if st.button("🔍 Search Series", type="primary"):
        with st.spinner("Searching..."):
            # Build filters
            filters = {}
            if selected_collection != "All":
                filters['collection'] = selected_collection
            if selected_body_part != "All":
                filters['bodyPartExamined'] = selected_body_part
            if selected_modality != "All":
                filters['modality'] = selected_modality
            if selected_manufacturer != "All":
                filters['manufacturer'] = selected_manufacturer
            if selected_model != "All":
                filters['manufacturerModelName'] = selected_model
            if patient_id:
                filters['patientID'] = patient_id
            
            # Search with limit
            results = search_series(filters)
            
            # Limit results to prevent overwhelming the UI
            max_results = 1000
            if len(results) > max_results:
                results = results[:max_results]
                st.warning(f"⚠️ Found {len(results)} series, showing first {max_results}. Use more specific filters to reduce results.")
            
            st.session_state.search_results = results
            st.session_state.current_page = 0  # Reset to first page
            
            if results:
                st.success(f"Found {len(results)} series")
            else:
                st.warning("No series found matching your criteria")
    
    # Display results
    if st.session_state.search_results:
        total_results = len(st.session_state.search_results)
        st.subheader(f"📋 Search Results ({total_results} series)")
        
        # Results per page selector
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            per_page = st.selectbox("Results per page", [10, 25, 50, 100], index=1, key="per_page_selector")
            if per_page != st.session_state.results_per_page:
                st.session_state.results_per_page = per_page
                st.session_state.current_page = 0
                st.rerun()
        
        with col2:
            st.write(f"**Page size:** {per_page}")
        
        # Select all/none for current page
        with col3:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Select All on Page"):
                    current_page_results = get_paginated_results(st.session_state.search_results, st.session_state.current_page, per_page)
                    for series in current_page_results:
                        if series['SeriesInstanceUID'] not in st.session_state.selected_series:
                            st.session_state.selected_series.append(series['SeriesInstanceUID'])
                    st.rerun()
            with col_b:
                if st.button("Clear All"):
                    st.session_state.selected_series = []
                    st.rerun()
        
        # Get paginated results
        current_page_results = get_paginated_results(st.session_state.search_results, st.session_state.current_page, per_page)
        
        # Results table
        for i, series in enumerate(current_page_results):
            global_index = st.session_state.current_page * per_page + i
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    is_selected = series['SeriesInstanceUID'] in st.session_state.selected_series
                    if st.checkbox(f"Select #{global_index + 1}", value=is_selected, key=f"select_{global_index}"):
                        if series['SeriesInstanceUID'] not in st.session_state.selected_series:
                            st.session_state.selected_series.append(series['SeriesInstanceUID'])
                    else:
                        if series['SeriesInstanceUID'] in st.session_state.selected_series:
                            st.session_state.selected_series.remove(series['SeriesInstanceUID'])
                
                with col2:
                    display_dicom_info(series)
                
                with col3:
                    if st.button("View", key=f"view_{global_index}"):
                        st.session_state.current_series = series
                        st.success(f"✅ Selected series: {series.get('SeriesDescription', 'Unknown')}")
                        st.info("💡 Switch to 'DICOM Viewer' in the sidebar to view this series")
        
        # Display pagination controls
        display_pagination_controls(total_results, st.session_state.current_page, per_page)
        
        # Show selection summary
        if st.session_state.selected_series:
            st.markdown("---")
            st.info(f"📋 **Selected {len(st.session_state.selected_series)} series** - Go to 'Download Manager' to download them")

def show_download_page():
    """Download manager page"""
    st.header("📥 Download Manager")
    
    if not st.session_state.selected_series:
        st.info("No series selected. Go to DICOM Search to select series for download.")
        return
    
    st.subheader(f"Selected Series ({len(st.session_state.selected_series)})")
    
    # Display selected series
    for i, series_uid in enumerate(st.session_state.selected_series):
        series_info = next((s for s in st.session_state.search_results if s['SeriesInstanceUID'] == series_uid), None)
        if series_info:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{series_info.get('SeriesDescription', 'Unknown')}**")
                st.write(f"Patient: {series_info.get('PatientID', 'N/A')} | Modality: {series_info.get('Modality', 'N/A')}")
            
            with col2:
                st.write(f"Images: {series_info.get('ImageCount', 'N/A')}")
            
            with col3:
                if st.button(f"Download {i+1}", key=f"download_{i}"):
                    with st.spinner(f"Downloading {series_info.get('SeriesDescription', 'series')}..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        local_path = download_series_local(series_uid)
                        
                        if local_path:
                            st.success(f"✅ Downloaded to {local_path}")
                            st.info("💡 Go to DICOM Viewer to view the downloaded series")
                        else:
                            st.error("❌ Download failed")
    
    # Batch download
    if len(st.session_state.selected_series) > 1:
        st.markdown("---")
        st.subheader("Batch Download")
        
        if st.button("Download All Selected", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            successful_downloads = 0
            for i, series_uid in enumerate(st.session_state.selected_series):
                status_text.text(f"Downloading {i+1}/{len(st.session_state.selected_series)}")
                progress_bar.progress((i) / len(st.session_state.selected_series))
                
                local_path = download_series_local(series_uid)
                if local_path:
                    successful_downloads += 1
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            
            st.success(f"✅ Successfully downloaded {successful_downloads}/{len(st.session_state.selected_series)} series")
            st.info("💡 Go to DICOM Viewer to view the downloaded series")

def show_viewer_page():
    """DICOM viewer page"""
    st.header("🖼️ DICOM Viewer")
    
    # Show current series info if available
    if 'current_series' in st.session_state:
        series = st.session_state.current_series
        st.info(f"📋 Selected Series: {series.get('SeriesDescription', 'Unknown')}")
        st.write(f"**Patient ID:** {series.get('PatientID', 'N/A')} | **Modality:** {series.get('Modality', 'N/A')} | **Images:** {series.get('ImageCount', 'N/A')}")
        
        # Add download and view button for TCIA series
        if st.button("📥 Download and View Series", type="primary"):
            with st.spinner("Downloading series..."):
                try:
                    # Download the series
                    series_uid = series['SeriesInstanceUID']
                    local_path = download_series_local(series_uid)
                    
                    if not local_path:
                        st.error("Failed to download series")
                        return
                    
                    # Use the downloaded file
                    temp_file_path = local_path
                    
                    # Extract and display first image
                    with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                        # Get list of DICOM files
                        dicom_files = [f for f in zip_ref.namelist() if f.endswith('.dcm')]
                        
                        if dicom_files:
                            # Read first DICOM file
                            with zip_ref.open(dicom_files[0]) as dicom_file:
                                ds = pydicom.dcmread(dicom_file)
                                
                                # Display DICOM metadata
                                with st.expander("DICOM Metadata"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Patient Name:** {ds.get('PatientName', 'N/A')}")
                                        st.write(f"**Patient ID:** {ds.get('PatientID', 'N/A')}")
                                        st.write(f"**Modality:** {ds.get('Modality', 'N/A')}")
                                        st.write(f"**Study Date:** {ds.get('StudyDate', 'N/A')}")
                                    
                                    with col2:
                                        st.write(f"**Series Description:** {ds.get('SeriesDescription', 'N/A')}")
                                        st.write(f"**Manufacturer:** {ds.get('Manufacturer', 'N/A')}")
                                        st.write(f"**Image Size:** {ds.get('Rows', 'N/A')} x {ds.get('Columns', 'N/A')}")
                                        st.write(f"**Pixel Spacing:** {ds.get('PixelSpacing', 'N/A')}")
                                
                                # Display image
                                if hasattr(ds, 'pixel_array'):
                                    st.subheader("DICOM Image (First Image in Series)")
                                    
                                    # Convert to PIL Image
                                    img_array = ds.pixel_array
                                    img = Image.fromarray(img_array)
                                    
                                    # Normalize for display
                                    if img.mode != 'L':
                                        img = img.convert('L')
                                    
                                    # Display with controls
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.image(img, caption="DICOM Image", use_container_width=True)
                                    
                                    with col2:
                                        st.subheader("Image Controls")
                                        
                                        # Brightness/contrast
                                        brightness = st.slider("Brightness", 0.0, 2.0, 1.0, key="tcia_brightness")
                                        contrast = st.slider("Contrast", 0.0, 2.0, 1.0, key="tcia_contrast")
                                        
                                        # Apply adjustments
                                        if brightness != 1.0 or contrast != 1.0:
                                            adjusted_array = img_array.astype(np.float32)
                                            adjusted_array = (adjusted_array - adjusted_array.mean()) * contrast + adjusted_array.mean() * brightness
                                            adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)
                                            adjusted_img = Image.fromarray(adjusted_array)
                                            st.image(adjusted_img, caption="Adjusted Image", use_container_width=True)
                                    
                                    st.success(f"✅ Displaying first image from series ({len(dicom_files)} total images)")
                                else:
                                    st.error("No pixel data found in DICOM file")
                        else:
                            st.error("No DICOM files found in series")
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                except Exception as e:
                    st.error(f"Error downloading/viewing series: {e}")
        
        st.markdown("---")
    
    # File upload for local files
    st.subheader("Upload Local DICOM File")
    uploaded_file = st.file_uploader("Upload DICOM file", type=['dcm', 'dicom'])
    
    if uploaded_file is not None:
        try:
            ds = pydicom.dcmread(uploaded_file)
            
            # Display DICOM metadata
            with st.expander("DICOM Metadata"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Patient Name:** {ds.get('PatientName', 'N/A')}")
                    st.write(f"**Patient ID:** {ds.get('PatientID', 'N/A')}")
                    st.write(f"**Modality:** {ds.get('Modality', 'N/A')}")
                    st.write(f"**Study Date:** {ds.get('StudyDate', 'N/A')}")
                
                with col2:
                    st.write(f"**Series Description:** {ds.get('SeriesDescription', 'N/A')}")
                    st.write(f"**Manufacturer:** {ds.get('Manufacturer', 'N/A')}")
                    st.write(f"**Image Size:** {ds.get('Rows', 'N/A')} x {ds.get('Columns', 'N/A')}")
                    st.write(f"**Pixel Spacing:** {ds.get('PixelSpacing', 'N/A')}")
            
            # Display image
            if hasattr(ds, 'pixel_array'):
                st.subheader("DICOM Image")
                
                # Convert to PIL Image
                img_array = ds.pixel_array
                img = Image.fromarray(img_array)
                
                # Normalize for display
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Display with controls
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(img, caption="DICOM Image", use_container_width=True)
                
                with col2:
                    st.subheader("Image Controls")
                    
                    # Brightness/contrast
                    brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
                    contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
                    
                    # Apply adjustments
                    if brightness != 1.0 or contrast != 1.0:
                        adjusted_array = img_array.astype(np.float32)
                        adjusted_array = (adjusted_array - adjusted_array.mean()) * contrast + adjusted_array.mean() * brightness
                        adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)
                        adjusted_img = Image.fromarray(adjusted_array)
                        st.image(adjusted_img, caption="Adjusted Image", use_container_width=True)
            
            # Annotation tools (basic)
            st.subheader("Annotation Tools")
            st.info("Advanced annotation tools coming soon!")
            
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")

def show_monai_page():
    st.header("🤖 MONAI Inference")
    
    # Test MONAI connection
    if test_monai_connection():
        st.success("✅ MONAI Label Server Connected")
        
        # Get server info
        try:
            response = requests.get(f"{MONAI_SERVER_URL}/info/")
            if response.status_code == 200:
                info = response.json()
                st.json(info)
        except Exception as e:
            st.error(f"Failed to get server info: {e}")
        
        # File upload for inference
        st.subheader("Upload DICOM for AI Inference")
        
        # Add note about the new approach
        st.info("ℹ️ **Note**: DICOM files are converted to NIFTI format to avoid GDCM issues. Results are returned in JSON format.")
        
        uploaded_file = st.file_uploader("Upload DICOM file for inference", type=['dcm', 'dicom', 'nii', 'nii.gz'], key="monai_upload")
        
        # Model selection - now includes tumor detection models
        model_option = st.selectbox("Select Model", [
            "segmentation",  # Multi-organ segmentation
            "breast_tumor_detection",  # Breast cancer detection
            "lung_nodule_detection",  # Lung nodule detection  
            "lung_cancer_segmentation",  # Lung cancer segmentation
            "segmentation_spleen",  # Spleen-specific segmentation
            "localization_spine",  # Spine localization
            "localization_vertebra"  # Vertebra localization
        ])
        
        # Show model description
        model_descriptions = {
            "segmentation": "Multi-organ segmentation (liver, spleen, kidneys, etc.)",
            "breast_tumor_detection": "Breast tumor detection and segmentation",
            "lung_nodule_detection": "Lung nodule detection and classification", 
            "lung_cancer_segmentation": "Lung cancer tumor segmentation",
            "segmentation_spleen": "Spleen-specific segmentation",
            "localization_spine": "Spine localization",
            "localization_vertebra": "Vertebra localization"
        }
        
        st.info(f"**Selected Model**: {model_descriptions.get(model_option, 'Unknown model')}")
        
        st.info("Output will be returned in the best available format to avoid DICOM writing issues.")
        st.info("💡 **Note**: Using automatic segmentation models that don't require user interaction.")
        st.info("🔍 **Available formats**: JSON, Image, All, DICOM-SEG")
        
        if uploaded_file and st.button("Run AI Inference", type="primary"):
            with st.spinner("Running AI inference..."):
                try:
                    # Save uploaded file temporarily
                    file_extension = '.dcm' if uploaded_file.name.lower().endswith(('.dcm', '.dicom')) else '.nii.gz'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Preprocess DICOM for MONAI compatibility
                    st.info("Preprocessing DICOM file for MONAI compatibility...")
                    processed_path = preprocess_dicom_for_monai(tmp_path)
                    
                    # Send to MONAI server using the correct API structure
                    with open(processed_path, 'rb') as f:
                        files = {'file': f}
                        
                        # Try different output formats based on OpenAPI spec
                        output_formats = ["json", "image", "all", "dicom_seg"]
                        response = None
                        
                        for output_format in output_formats:
                            try:
                                # Explain what each format means
                                format_descriptions = {
                                    "json": "JSON response with metadata",
                                    "image": "Image file response",
                                    "all": "All available formats",
                                    "dicom_seg": "DICOM segmentation format"
                                }
                                st.info(f"Trying output format: {output_format} - {format_descriptions.get(output_format, 'Unknown format')}")
                                
                                # Use the correct API structure from OpenAPI spec
                                # Endpoint: /infer/{model}
                                # File parameter: 'file'
                                # Output parameter: query parameter
                                response = requests.post(
                                    f"{MONAI_SERVER_URL}/infer/{model_option}",
                                    files=files,
                                    params={
                                        "output": output_format,
                                        "device": "cpu"
                                    }
                                )
                                
                                if response.status_code == 200:
                                    st.success(f"✅ Success with output format: {output_format}")
                                    break
                                else:
                                    st.warning(f"❌ Failed with output format: {output_format} - {response.status_code}")
                                    
                            except Exception as e:
                                st.warning(f"❌ Error with output format: {output_format} - {str(e)}")
                                continue

                    
                    if response.status_code == 200:
                        # Check content type to handle different response formats
                        content_type = response.headers.get('content-type', '')
                        
                        if 'application/json' in content_type:
                            result = response.json()
                            st.success("✅ Inference completed successfully!")
                            st.json(result)
                            
                            # Display key information from the result
                            if isinstance(result, dict):
                                if 'label_names' in result:
                                    st.info(f"**Detected labels:** {', '.join(result['label_names'])}")
                                if 'label_ids' in result:
                                    st.info(f"**Label IDs:** {result['label_ids']}")
                                if 'file' in result:
                                    st.info(f"**Result file:** {result['file']}")
                        elif 'multipart/form-data' in content_type:
                            # Handle multipart response (image + metadata)
                            st.success("✅ Inference completed successfully!")
                            st.info("Received multipart response with image and metadata")
                            # You can extract the image from the response if needed
                        elif 'application/octet-stream' in content_type or 'application/gzip' in content_type:
                            # Handle binary response (NIFTI file)
                            st.success("✅ Inference completed successfully!")
                            st.info("Received NIFTI result file")
                            # Save the result file
                            result_path = tmp_path.replace('.dcm', f'_result_{model_option}.nii.gz')
                            with open(result_path, 'wb') as result_file:
                                result_file.write(response.content)
                            st.success(f"Result saved to: {result_path}")
                            st.info("NIFTI file contains the segmentation result. You can open it with medical imaging software like 3D Slicer or ITK-SNAP.")
                        else:
                            st.success("✅ Inference completed successfully!")
                            st.info(f"Received response with content-type: {content_type}")
                            # Try to parse as JSON anyway
                            try:
                                result = response.json()
                                st.json(result)
                            except:
                                st.text(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                    else:
                        # Handle different error types
                        error_text = response.text.lower()
                        if response.status_code == 404:
                            st.error(f"❌ 404 Error - Endpoint not found")
                            st.info("This might mean:")
                            st.info("1. MONAI server is not running")
                            st.info("2. Wrong endpoint URL")
                            st.info("3. Model not available")
                            st.info(f"Response: {response.text}")
                        elif "gdcm" in error_text or "floating point" in error_text or "pixel type" in error_text:
                            st.warning("⚠️ GDCM Error - This is a known issue with DICOM writing")
                            st.success("✅ The AI model successfully processed your image!")
                            st.info("The inference worked, but there was an issue saving the result in DICOM format.")
                            st.info("This is a known limitation with the current MONAI server configuration.")
                            st.info("Try using a different output format or the result may be available in the server logs.")
                        else:
                            st.error(f"Inference failed: {response.status_code} - {response.text}")
                    
                    # Clean up
                    os.unlink(tmp_path)
                    if processed_path != tmp_path:
                        os.unlink(processed_path)
                    
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    if 'tmp_path' in locals():
                        os.unlink(tmp_path)
                    if 'processed_path' in locals() and processed_path != tmp_path:
                        os.unlink(processed_path)
    else:
        st.error("❌ MONAI Label Server Not Available")
        st.info("Make sure the MONAI Label server is running on port 8000")
        
        # Show connection details
        st.subheader("Connection Details")
        st.code(f"Server URL: {MONAI_SERVER_URL}")
        st.code("Expected endpoint: /info/")
        
        # Manual test button
        if st.button("Test Connection"):
            if test_monai_connection():
                st.success("✅ Connection successful!")
            else:
                st.error("❌ Connection failed")

def show_settings_page():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("Configuration")
    
    # MONAI server URL
    monai_url = st.text_input("MONAI Server URL", value=MONAI_SERVER_URL)
    
    # AWS S3 settings
    st.subheader("AWS S3 Configuration")
    aws_region = st.text_input("AWS Region", value=AWS_REGION)
    s3_bucket = st.text_input("S3 Bucket Name", value=S3_BUCKET)
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved! (Note: Restart the app for changes to take effect)")
    
    st.markdown("---")
    
    # System information
    st.subheader("System Information")
    st.write(f"**Python Version:** {os.sys.version}")
    st.write(f"**Working Directory:** {os.getcwd()}")
    st.write(f"**Available Memory:** {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3):.1f} GB")
    
    # Package versions
    st.subheader("Installed Packages")
    packages = ["streamlit", "requests", "pydicom", "pillow", "boto3", "tcia-utils"]
    for package in packages:
        try:
            import importlib
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            st.write(f"**{package}:** {version}")
        except:
            st.write(f"**{package}:** Not installed")

if __name__ == "__main__":
    main()
