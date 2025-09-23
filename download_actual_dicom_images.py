#!/usr/bin/env python3
"""
Download Actual DICOM Images using NBIA Data Retriever
This script uses the manifest files to download the actual 1.63TB of DICOM images
"""

import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

# Configuration
BASE_DOWNLOAD_DIR = Path("~/mri_app/breast_dbt_collection").expanduser()
DICOM_DIR = BASE_DOWNLOAD_DIR / "dicom_images"
MANIFEST_DIR = DICOM_DIR

def check_nbia_retriever():
    """Check if NBIA Data Retriever is installed"""
    try:
        result = subprocess.run(['nbia-data-retriever', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ NBIA Data Retriever found: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ NBIA Data Retriever not found!")
    print("\n📥 To install NBIA Data Retriever:")
    print("   1. Download from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
    print("   2. Or use: wget https://github.com/kirbyju/TCIA_Data_Retriever/releases/download/v1.0.0/nbia-data-retriever")
    print("   3. Make executable: chmod +x nbia-data-retriever")
    print("   4. Move to PATH: sudo mv nbia-data-retriever /usr/local/bin/")
    return False

def download_with_manifest(manifest_file, output_dir, description):
    """Download DICOM images using a manifest file"""
    print(f"\n🏥 Downloading {description}...")
    print(f"   Manifest: {manifest_file}")
    print(f"   Output: {output_dir}")
    
    if not manifest_file.exists():
        print(f"❌ Manifest file not found: {manifest_file}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use NBIA Data Retriever with manifest
        cmd = [
            'nbia-data-retriever',
            '--manifest', str(manifest_file),
            '--output', str(output_dir),
            '--continue'  # Resume interrupted downloads
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        print("⏳ This will take a LONG time (1.63TB download)...")
        print("💡 You can press Ctrl+C to pause and resume later")
        
        # Start the download process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Stream output in real-time
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ Successfully downloaded {description}")
            return True
        else:
            print(f"❌ Download failed for {description}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Download paused for {description}")
        print("💡 You can resume later by running this script again")
        return False
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
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
            return False
        else:
            print(f"✅ Sufficient space available for download")
            return True
            
    except Exception as e:
        print(f"❌ Error checking disk usage: {e}")
        return False

def main():
    """Main download function"""
    print("🏥 Breast DBT DICOM Images Downloader")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
    
    # Check disk space
    if not show_disk_usage():
        print("❌ Insufficient disk space. Please free up space or extend your volume.")
        return
    
    # Check if NBIA Data Retriever is available
    if not check_nbia_retriever():
        print("\n❌ Cannot proceed without NBIA Data Retriever")
        return
    
    # Define manifest files and their output directories
    downloads = [
        {
            "manifest": MANIFEST_DIR / "training" / "BSC-DBT-Train-manifest.tcia",
            "output": DICOM_DIR / "training" / "images",
            "description": "Training DICOM Images"
        },
        {
            "manifest": MANIFEST_DIR / "test" / "BSC-DBT-Test-manifest.tcia", 
            "output": DICOM_DIR / "test" / "images",
            "description": "Test DICOM Images"
        }
    ]
    
    print(f"\n🚀 Starting DICOM image downloads...")
    print(f"📊 Expected total size: ~1.63TB")
    print(f"⏱️  Estimated time: 6-12 hours (depending on connection)")
    
    success_count = 0
    for download in downloads:
        if download_with_manifest(
            download["manifest"], 
            download["output"], 
            download["description"]
        ):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("📋 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successful downloads: {success_count}/{len(downloads)}")
    print(f"📁 Files saved to: {BASE_DOWNLOAD_DIR}")
    
    if success_count == len(downloads):
        print("\n🎉 All DICOM images downloaded successfully!")
        print("📊 You now have the complete Breast DBT dataset with annotations")
    else:
        print(f"\n⚠️  {len(downloads) - success_count} downloads failed or were interrupted")
        print("💡 You can run this script again to resume interrupted downloads")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()


