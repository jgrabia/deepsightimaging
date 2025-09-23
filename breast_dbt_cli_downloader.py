#!/usr/bin/env python3
"""
Breast DBT Collection Command Line Downloader
Downloads the entire Breast-Cancer-Screening-DBT dataset (1.63TB) via command line
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from tcia_utils import nbia
import time
from datetime import datetime

# Configuration
BASE_DOWNLOAD_DIR = Path("~/mri_app/breast_dbt_collection").expanduser()
ANNOTATIONS_DIR = BASE_DOWNLOAD_DIR / "annotations"
DICOM_DIR = BASE_DOWNLOAD_DIR / "dicom_images"

# TCIA URLs for the Breast DBT collection
TCIA_URLS = {
    # Training set
    "training_images": "https://www.cancerimagingarchive.net/wp-content/uploads/BSC-DBT-Train-manifest.tcia",
    "training_boxes": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-boxes-train-v2.csv",
    "training_labels": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-labels-train.csv",
    "training_file_paths": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-file-paths-train.csv",
    
    # Validation set
    "validation_images": "https://www.cancerimagingarchive.net/wp-content/uploads/BSC-DBT-Val-manifest.tcia",
    "validation_file_paths": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-file-paths-validation.csv",
    
    # Test set
    "test_images": "https://www.cancerimagingarchive.net/wp-content/uploads/BSC-DBT-Test-manifest.tcia",
    "test_file_paths": "https://www.cancerimagingarchive.net/wp-content/uploads/BCS-DBT-file-paths-test.csv",
}

def create_directories():
    """Create the directory structure"""
    print("📁 Creating directory structure...")
    BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (ANNOTATIONS_DIR / "training").mkdir(exist_ok=True)
    (ANNOTATIONS_DIR / "validation").mkdir(exist_ok=True)
    (ANNOTATIONS_DIR / "test").mkdir(exist_ok=True)
    (DICOM_DIR / "training").mkdir(exist_ok=True)
    (DICOM_DIR / "validation").mkdir(exist_ok=True)
    (DICOM_DIR / "test").mkdir(exist_ok=True)
    
    print(f"✅ Directories created at: {BASE_DOWNLOAD_DIR}")

def download_file(url, output_path, description=""):
    """Download a file with progress tracking"""
    print(f"\n📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Output: {output_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded_size:,} / {total_size:,} bytes)", end='', flush=True)
        
        print(f"\n✅ Downloaded: {output_path}")
        print(f"   Size: {downloaded_size:,} bytes ({downloaded_size / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {url}: {e}")
        return False

def download_annotation_files():
    """Download all annotation CSV files"""
    print("\n" + "="*60)
    print("📊 DOWNLOADING ANNOTATION FILES")
    print("="*60)
    
    annotation_files = {
        "training_boxes": ("BCS-DBT-boxes-train-v2.csv", "training"),
        "training_labels": ("BCS-DBT-labels-train.csv", "training"),
        "training_file_paths": ("BCS-DBT-file-paths-train.csv", "training"),
        "validation_file_paths": ("BCS-DBT-file-paths-validation.csv", "validation"),
        "test_file_paths": ("BCS-DBT-file-paths-test.csv", "test"),
    }
    
    success_count = 0
    for key, (filename, subdir) in annotation_files.items():
        url = TCIA_URLS[key]
        output_path = ANNOTATIONS_DIR / subdir / filename
        
        if download_file(url, output_path, f"Annotation: {filename}"):
            success_count += 1
    
    print(f"\n✅ Downloaded {success_count}/{len(annotation_files)} annotation files")
    return success_count == len(annotation_files)

def download_dicom_manifests():
    """Download DICOM manifest files"""
    print("\n" + "="*60)
    print("🏥 DOWNLOADING DICOM MANIFESTS")
    print("="*60)
    
    manifest_files = {
        "training_images": ("BSC-DBT-Train-manifest.tcia", "training"),
        "validation_images": ("BSC-DBT-Val-manifest.tcia", "validation"),
        "test_images": ("BSC-DBT-Test-manifest.tcia", "test"),
    }
    
    success_count = 0
    for key, (filename, subdir) in manifest_files.items():
        url = TCIA_URLS[key]
        output_path = DICOM_DIR / subdir / filename
        
        if download_file(url, output_path, f"Manifest: {filename}"):
            success_count += 1
    
    print(f"\n✅ Downloaded {success_count}/{len(manifest_files)} manifest files")
    return success_count == len(manifest_files)

def analyze_downloaded_data():
    """Analyze the downloaded annotation files"""
    print("\n" + "="*60)
    print("📈 ANALYZING DOWNLOADED DATA")
    print("="*60)
    
    try:
        # Analyze training boxes
        boxes_file = ANNOTATIONS_DIR / "training" / "BCS-DBT-boxes-train-v2.csv"
        if boxes_file.exists():
            print(f"\n📊 Analyzing: {boxes_file.name}")
            df_boxes = pd.read_csv(boxes_file)
            print(f"   Rows: {len(df_boxes):,}")
            print(f"   Columns: {list(df_boxes.columns)}")
            print(f"   Sample data:")
            print(df_boxes.head())
            
            # Check for lesion counts
            if 'label' in df_boxes.columns:
                lesion_counts = df_boxes['label'].value_counts()
                print(f"\n   Lesion type distribution:")
                for label, count in lesion_counts.items():
                    print(f"     {label}: {count:,}")
        
        # Analyze training file paths
        paths_file = ANNOTATIONS_DIR / "training" / "BCS-DBT-file-paths-train.csv"
        if paths_file.exists():
            print(f"\n📁 Analyzing: {paths_file.name}")
            df_paths = pd.read_csv(paths_file)
            print(f"   Rows: {len(df_paths):,}")
            print(f"   Columns: {list(df_paths.columns)}")
            print(f"   Sample data:")
            print(df_paths.head())
            
            # Check for unique patients/studies
            if 'PatientID' in df_paths.columns:
                unique_patients = df_paths['PatientID'].nunique()
                print(f"\n   Unique patients: {unique_patients:,}")
            
            if 'StudyInstanceUID' in df_paths.columns:
                unique_studies = df_paths['StudyInstanceUID'].nunique()
                print(f"   Unique studies: {unique_studies:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return False

def show_disk_usage():
    """Show current disk usage"""
    print("\n" + "="*60)
    print("💾 DISK USAGE")
    print("="*60)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(BASE_DOWNLOAD_DIR)
        
        print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
        print(f"   Total space: {total / (1024**3):.1f} GB")
        print(f"   Used space: {used / (1024**3):.1f} GB")
        print(f"   Free space: {free / (1024**3):.1f} GB")
        print(f"   Available: {(free / total) * 100:.1f}%")
        
        # Check if we have enough space for 1.63TB
        required_gb = 1.63 * 1024  # Convert TB to GB
        if free / (1024**3) < required_gb:
            print(f"⚠️  WARNING: Need {required_gb:.0f} GB, but only {free / (1024**3):.1f} GB available!")
        else:
            print(f"✅ Sufficient space available for download")
            
    except Exception as e:
        print(f"❌ Error checking disk usage: {e}")

def main():
    """Main download function"""
    print("🏥 Breast DBT Collection Downloader")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
    
    # Show disk usage
    show_disk_usage()
    
    # Create directories
    create_directories()
    
    # Download annotation files first (small, fast)
    print("\n🚀 Starting download process...")
    annotation_success = download_annotation_files()
    
    # Download DICOM manifests
    manifest_success = download_dicom_manifests()
    
    # Analyze downloaded data
    if annotation_success:
        analyze_downloaded_data()
    
    # Summary
    print("\n" + "="*60)
    print("📋 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Annotations: {'Success' if annotation_success else 'Failed'}")
    print(f"✅ Manifests: {'Success' if manifest_success else 'Failed'}")
    print(f"📁 Files saved to: {BASE_DOWNLOAD_DIR}")
    
    if annotation_success and manifest_success:
        print("\n🎉 All files downloaded successfully!")
        print("\n📝 Next steps:")
        print("   1. Use the manifest files to download actual DICOM images")
        print("   2. Use NBIA Data Retriever or similar tool")
        print("   3. Expected total size: ~1.63TB")
    else:
        print("\n❌ Some downloads failed. Check the errors above.")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()


