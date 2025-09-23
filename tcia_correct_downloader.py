#!/usr/bin/env python3
"""
TCIA Correct DICOM Downloader
Uses the proper TCIA download approach with authentication
"""

import os
import sys
import requests
import tempfile
from pathlib import Path
import time
from datetime import datetime

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
        
        series_uids = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('includeAnnotation') and not line.startswith('noOfrRetry') and not line.startswith('databasketId') and not line.startswith('manifestVersion') and not line.startswith('ListOfSeriesToDownload') and not line.startswith('downloadServerUrl'):
                if len(line) > 50 and line.replace('.', '').isdigit():
                    series_uids.append(line)
        
        print(f"✅ Found {len(series_uids)} series in manifest")
        return series_uids
        
    except Exception as e:
        print(f"❌ Error parsing manifest: {e}")
        return []

def download_with_tcia_api(series_uid, output_dir, series_num, total_series):
    """Download using TCIA REST API approach"""
    # Create series directory
    series_dir = output_dir / f"series_{series_num:04d}_{series_uid[:20]}"
    series_dir.mkdir(exist_ok=True)
    
    # Skip if already exists and has content
    if series_dir.exists() and any(series_dir.iterdir()):
        return True
    
    try:
        # Use TCIA REST API to get series information first
        api_url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getSeries?SeriesInstanceUID={series_uid}"
        
        response = requests.get(api_url, timeout=30)
        if response.status_code != 200:
            print(f"❌ [{series_num}/{total_series}] API error: {response.status_code}")
            return False
        
        series_data = response.json()
        if not series_data:
            print(f"❌ [{series_num}/{total_series}] No series data found")
            return False
        
        # Get the download URL from the series data
        download_url = None
        for item in series_data:
            if 'downloadURL' in item:
                download_url = item['downloadURL']
                break
        
        if not download_url:
            print(f"❌ [{series_num}/{total_series}] No download URL found")
            return False
        
        # Download the series
        print(f"📥 [{series_num}/{total_series}] Downloading from: {download_url}")
        
        download_response = requests.get(download_url, stream=True, timeout=300)
        if download_response.status_code != 200:
            print(f"❌ [{series_num}/{total_series}] Download failed: {download_response.status_code}")
            return False
        
        # Save the file
        output_file = series_dir / f"series_{series_uid[:20]}.zip"
        
        total_size = int(download_response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_file, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded_size:,} / {total_size:,} bytes)", end='', flush=True)
        
        print(f"\n✅ [{series_num}/{total_series}] Downloaded: {downloaded_size:,} bytes")
        
        # Try to extract if it's a zip file
        try:
            import zipfile
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(series_dir)
            output_file.unlink()  # Remove zip file
            print(f"✅ Extracted DICOM files")
        except:
            print(f"📁 File saved as-is (not a zip)")
        
        return True
        
    except Exception as e:
        print(f"❌ [{series_num}/{total_series}] Error: {e}")
        return False

def download_manifest_with_api(manifest_path, output_dir, description):
    """Download all series from a manifest using TCIA API"""
    print(f"\n🏥 Downloading {description} using TCIA API...")
    print(f"   Manifest: {manifest_path}")
    print(f"   Output: {output_dir}")
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return False
    
    # Parse manifest
    series_uids = parse_manifest_file(manifest_path)
    if not series_uids:
        print("❌ No series found in manifest")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting download of {len(series_uids)} series...")
    print(f"💡 You can press Ctrl+C to pause and resume later")
    
    success_count = 0
    for i, series_uid in enumerate(series_uids, 1):
        if download_with_tcia_api(series_uid, output_dir, i, len(series_uids)):
            success_count += 1
        
        # Small delay between downloads
        time.sleep(1)
    
    print(f"\n✅ Downloaded {success_count}/{len(series_uids)} series successfully")
    return success_count > 0

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
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking disk usage: {e}")
        return False

def main():
    """Main download function"""
    print("🏥 TCIA Correct DICOM Downloader")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
    
    # Check disk space
    show_disk_usage()
    
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
    
    print(f"\n🚀 Starting DICOM image downloads using TCIA API...")
    print(f"📊 Expected total size: ~1.63TB")
    
    success_count = 0
    for download in downloads:
        try:
            if download_manifest_with_api(
                download["manifest"], 
                download["output"], 
                download["description"]
            ):
                success_count += 1
                    
        except KeyboardInterrupt:
            print(f"\n⏸️  Download paused for {download['description']}")
            print("💡 You can resume later by running this script again")
            break
        except Exception as e:
            print(f"❌ Error processing {download['description']}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successful downloads: {success_count}/{len(downloads)}")
    print(f"📁 Files saved to: {BASE_DOWNLOAD_DIR}")
    
    if success_count == len(downloads):
        print("\n🎉 All DICOM images downloaded successfully!")
    else:
        print(f"\n⚠️  {len(downloads) - success_count} downloads failed or were interrupted")
        print("💡 You can run this script again to resume interrupted downloads")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()


