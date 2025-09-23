#!/usr/bin/env python3
"""
TCIA DICOM Downloader using Python
Downloads DICOM images directly from TCIA using manifest files
"""

import os
import sys
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import time
from datetime import datetime
import zipfile
import tempfile

# Configuration
BASE_DOWNLOAD_DIR = Path("~/mri_app/breast_dbt_collection").expanduser()
DICOM_DIR = BASE_DOWNLOAD_DIR / "dicom_images"
MANIFEST_DIR = DICOM_DIR

def parse_manifest_file(manifest_path):
    """Parse a TCIA manifest file to extract series UIDs"""
    print(f"📄 Parsing manifest: {manifest_path}")
    
    try:
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        
        # Extract server URL and series UIDs
        server_url = None
        series_uids = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('downloadServerUrl='):
                server_url = line.split('=', 1)[1]
            elif line and not line.startswith('#') and not line.startswith('includeAnnotation') and not line.startswith('noOfrRetry') and not line.startswith('databasketId') and not line.startswith('manifestVersion') and not line.startswith('ListOfSeriesToDownload') and not line.startswith('downloadServerUrl'):
                # This is a series UID (long numeric string)
                if len(line) > 50 and line.replace('.', '').isdigit():
                    series_uids.append(line)
        
        print(f"✅ Found {len(series_uids)} series in manifest")
        print(f"🌐 Server URL: {server_url}")
        
        # Create download entries for each series
        downloads = []
        for uid in series_uids:
            downloads.append({
                'series_uid': uid,
                'server_url': server_url,
                'size': 0  # Unknown size
            })
        
        return downloads
        
    except Exception as e:
        print(f"❌ Error parsing manifest: {e}")
        return []

def download_series_direct(series_uid, server_url, output_dir, description=""):
    """Download a DICOM series directly from TCIA server"""
    try:
        print(f"📥 Downloading series: {description}")
        print(f"   Series UID: {series_uid}")
        print(f"   Output: {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct the download URL
        download_url = f"{server_url}?seriesUID={series_uid}&format=zip"
        
        print(f"   URL: {download_url}")
        
        # Download the zip file
        response = requests.get(download_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save as zip file
        zip_filename = f"series_{series_uid[:20]}.zip"
        zip_path = output_dir / zip_filename
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded_size:,} / {total_size:,} bytes)", end='', flush=True)
        
        print(f"\n✅ Downloaded: {zip_path}")
        print(f"   Size: {downloaded_size:,} bytes ({downloaded_size / (1024*1024):.1f} MB)")
        
        # Extract the zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"✅ Extracted DICOM files to: {output_dir}")
            
            # Remove the zip file to save space
            zip_path.unlink()
            print(f"🗑️  Removed zip file to save space")
            
        except Exception as e:
            print(f"⚠️  Could not extract zip file: {e}")
            print(f"📁 Zip file kept at: {zip_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading series {series_uid}: {e}")
        return False

def download_manifest_images(manifest_path, output_dir, description):
    """Download all images from a manifest file"""
    print(f"\n🏥 Downloading {description}...")
    print(f"   Manifest: {manifest_path}")
    print(f"   Output: {output_dir}")
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return False
    
    # Parse manifest
    downloads = parse_manifest_file(manifest_path)
    if not downloads:
        print("❌ No downloads found in manifest")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_series = len(downloads)
    
    print(f"🚀 Starting download of {total_series} series...")
    print(f"⚠️  This will take a VERY long time (1.63TB total)")
    print(f"💡 You can press Ctrl+C to pause and resume later")
    
    for i, download in enumerate(downloads, 1):
        series_uid = download['series_uid']
        
        # Create a subdirectory for this series
        series_dir = output_dir / f"series_{i:04d}_{series_uid[:20]}"
        
        # Skip if series already exists
        if series_dir.exists() and any(series_dir.iterdir()):
            print(f"⏭️  Skipping existing series: {series_uid[:20]}...")
            success_count += 1
            continue
        
        print(f"\n📁 [{i}/{total_series}] Series: {series_uid[:20]}...")
        
        if download_series_direct(series_uid, download['server_url'], series_dir, f"Series {i}"):
            success_count += 1
        
        # Small delay to be respectful to the server
        time.sleep(1)
    
    print(f"\n✅ Downloaded {success_count}/{total_series} series successfully")
    return success_count == total_series

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
    print("🏥 TCIA DICOM Images Downloader")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
    
    # Check disk space
    if not show_disk_usage():
        print("❌ Insufficient disk space. Please free up space or extend your volume.")
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
    print(f"💡 You can press Ctrl+C to pause and resume later")
    
    success_count = 0
    for download in downloads:
        try:
            if download_manifest_images(
                download["manifest"], 
                download["output"], 
                download["description"]
            ):
                success_count += 1
        except KeyboardInterrupt:
            print(f"\n⏸️  Download paused for {download['description']}")
            print("💡 You can resume later by running this script again")
            break
    
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
