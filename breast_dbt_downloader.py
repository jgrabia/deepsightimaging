#!/usr/bin/env python3
"""
Breast Cancer Screening DBT Collection Downloader
=================================================

A dedicated script to download the complete Breast-Cancer-Screening-DBT collection
from TCIA, including all DICOM images and annotation files.

Collection Details:
- 5,060 subjects with breast cancer screening data
- 1.63TB of digital breast tomosynthesis (DBT) images
- Complete annotation files (boxes, labels, paths)
- Perfect for training lesion detection models

Usage:
    python breast_dbt_downloader.py
"""

import streamlit as st
import requests
import os
import zipfile
import tempfile
from pathlib import Path
from tcia_utils import nbia
import pandas as pd
from datetime import datetime
import time

# Configuration
BASE_DOWNLOAD_DIR = Path("~/mri_app/breast_dbt_collection").expanduser()
ANNOTATIONS_DIR = BASE_DOWNLOAD_DIR / "annotations"
DICOM_DIR = BASE_DOWNLOAD_DIR / "dicom_images"

# TCIA URLs for the Breast DBT collection
TCIA_URLS = {
    # Training set
    "training_images": "https://www.cancerimagingarchive.net/wp-content/uploads/BSC-DBT-Train-manifest.tcia",
    "training_paths": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-file-paths-train.csv",
    "training_labels": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-labels-train.csv",
    "training_boxes": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-boxes-train-v2.csv",
    
    # Validation set
    "validation_images": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-Challenge-Validation.tcia",
    "validation_paths": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-file-paths-validation.csv",
    "validation_labels": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-labels-validation.csv",
    "validation_boxes": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-boxes-validation.csv",
    
    # Test set
    "test_images": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-Challenge-Test.tcia",
    "test_paths": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-file-paths-test.csv",
    "test_labels": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-labels-test.csv",
    "test_boxes": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-boxes-test.csv",
    
    # Team predictions
    "team_predictions": "https://www.cancerimagingarchive.net/wp-content/uploads/team_predictions_bothphases.zip"
}

def setup_directories():
    """Create necessary directories"""
    BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each dataset
    for dataset in ["training", "validation", "test"]:
        (ANNOTATIONS_DIR / dataset).mkdir(exist_ok=True)
        (DICOM_DIR / dataset).mkdir(exist_ok=True)

def download_file(url, filepath, description=""):
    """Download a file with progress tracking"""
    try:
        st.info(f"📥 Downloading {description}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        
        progress_bar.progress(1.0)
        status_text.text("Download complete!")
        st.success(f"✅ Downloaded {filepath.name}")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to download {description}: {e}")
        return False

def download_annotation_files():
    """Download all CSV annotation files"""
    st.subheader("📋 Downloading Annotation Files")
    
    annotation_files = [
        # Training set
        ("training_paths", "training", "File paths for training set"),
        ("training_labels", "training", "Classification labels for training set"),
        ("training_boxes", "training", "Lesion bounding boxes for training set"),
        
        # Validation set
        ("validation_paths", "validation", "File paths for validation set"),
        ("validation_labels", "validation", "Classification labels for validation set"),
        ("validation_boxes", "validation", "Lesion bounding boxes for validation set"),
        
        # Test set
        ("test_paths", "test", "File paths for test set"),
        ("test_labels", "test", "Classification labels for test set"),
        ("test_boxes", "test", "Lesion bounding boxes for test set"),
    ]
    
    successful_downloads = 0
    
    for url_key, dataset, description in annotation_files:
        url = TCIA_URLS[url_key]
        filename = url.split('/')[-1]
        filepath = ANNOTATIONS_DIR / dataset / filename
        
        if download_file(url, filepath, description):
            successful_downloads += 1
    
    # Download team predictions
    url = TCIA_URLS["team_predictions"]
    filepath = ANNOTATIONS_DIR / "team_predictions_bothphases.zip"
    if download_file(url, filepath, "Team predictions from DBTex challenge"):
        successful_downloads += 1
    
    st.success(f"✅ Downloaded {successful_downloads} annotation files")
    return successful_downloads

def download_dicom_manifests():
    """Download TCIA manifest files for DICOM images"""
    st.subheader("📁 Downloading DICOM Manifest Files")
    
    manifest_files = [
        ("training_images", "training", "Training set DICOM manifest (1.42TB)"),
        ("validation_images", "validation", "Validation set DICOM manifest (84.71GB)"),
        ("test_images", "test", "Test set DICOM manifest (135.14GB)"),
    ]
    
    successful_downloads = 0
    
    for url_key, dataset, description in manifest_files:
        url = TCIA_URLS[url_key]
        filepath = DICOM_DIR / dataset / f"{dataset}_manifest.tcia"
        
        if download_file(url, filepath, description):
            successful_downloads += 1
    
    st.success(f"✅ Downloaded {successful_downloads} DICOM manifest files")
    return successful_downloads

def analyze_downloaded_data():
    """Analyze the downloaded annotation files"""
    st.subheader("📊 Data Analysis")
    
    try:
        # Analyze training set
        training_paths = ANNOTATIONS_DIR / "training" / "BCS-DBT-file-paths-train.csv"
        training_boxes = ANNOTATIONS_DIR / "training" / "BCS-DBT-boxes-train-v2.csv"
        
        if training_paths.exists() and training_boxes.exists():
            # Load data
            paths_df = pd.read_csv(training_paths)
            boxes_df = pd.read_csv(training_boxes)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Images", len(paths_df))
                st.metric("Unique Patients", paths_df['PatientID'].nunique())
                st.metric("Views", len(paths_df['View'].unique()))
            
            with col2:
                st.metric("Total Lesions", len(boxes_df))
                st.metric("Patients with Lesions", boxes_df['PatientID'].nunique())
                st.metric("Lesion Classes", len(boxes_df['Class'].unique()))
            
            with col3:
                avg_width = boxes_df['Width'].mean()
                avg_height = boxes_df['Height'].mean()
                st.metric("Avg Lesion Size", f"{avg_width:.0f} x {avg_height:.0f}")
                st.metric("Total Slices", boxes_df['VolumeSlices'].sum())
            
            # Show class distribution
            st.subheader("📈 Lesion Class Distribution")
            class_counts = boxes_df['Class'].value_counts()
            st.bar_chart(class_counts)
            
            # Show sample data
            st.subheader("📋 Sample Data")
            st.write("**File Paths Sample:**")
            st.dataframe(paths_df.head())
            
            st.write("**Lesion Boxes Sample:**")
            st.dataframe(boxes_df.head())
            
        else:
            st.warning("Training data not found. Download annotation files first.")
            
    except Exception as e:
        st.error(f"Error analyzing data: {e}")

def show_download_instructions():
    """Show instructions for downloading DICOM images"""
    st.subheader("📖 Next Steps: Download DICOM Images")
    
    st.info("""
    **Important:** The manifest files (.tcia) contain the download instructions for the DICOM images.
    You need to use the NBIA Data Retriever to download the actual image files.
    
    **Steps:**
    1. Install NBIA Data Retriever from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    2. Use the downloaded .tcia manifest files to download the DICOM images
    3. The images will be organized according to the file paths in the CSV files
    """)
    
    # Show file locations
    st.subheader("📁 File Locations")
    st.write(f"**Base Directory:** {BASE_DOWNLOAD_DIR}")
    st.write(f"**Annotation Files:** {ANNOTATIONS_DIR}")
    st.write(f"**DICOM Manifests:** {DICOM_DIR}")
    
    # List downloaded files
    if ANNOTATIONS_DIR.exists():
        st.subheader("📋 Downloaded Annotation Files")
        for dataset_dir in ANNOTATIONS_DIR.iterdir():
            if dataset_dir.is_dir():
                st.write(f"**{dataset_dir.name.title()} Set:**")
                for file in dataset_dir.glob("*.csv"):
                    st.write(f"  - {file.name}")
                for file in dataset_dir.glob("*.zip"):
                    st.write(f"  - {file.name}")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Breast DBT Collection Downloader",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Breast Cancer Screening DBT Collection Downloader")
    
    st.markdown("""
    This tool downloads the complete **Breast-Cancer-Screening-DBT** collection from TCIA.
    
    **Collection Details:**
    - **5,060 subjects** with breast cancer screening data
    - **1.63TB** of digital breast tomosynthesis (DBT) images
    - **Complete annotations** including lesion bounding boxes and classification labels
    - **Perfect for training** accurate lesion detection models
    """)
    
    # Setup directories
    setup_directories()
    
    # Main download section
    st.markdown("---")
    st.subheader("🚀 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Download All Annotation Files", type="primary"):
            with st.spinner("Downloading annotation files..."):
                download_annotation_files()
    
    with col2:
        if st.button("📁 Download DICOM Manifests", type="secondary"):
            with st.spinner("Downloading DICOM manifest files..."):
                download_dicom_manifests()
    
    with col3:
        if st.button("📊 Analyze Downloaded Data", type="secondary"):
            analyze_downloaded_data()
    
    # Show instructions
    show_download_instructions()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Source:** [TCIA Breast-Cancer-Screening-DBT Collection](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/)
    
    **Citation Required:** Please cite the original paper when using this data.
    """)

if __name__ == "__main__":
    main()
