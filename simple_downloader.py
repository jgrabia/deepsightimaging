#!/usr/bin/env python3
"""
Simple TCIA DICOM Downloader
Just downloads the files without worrying about format detection
"""

import os
import sys
import subprocess
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

def download_series_simple(series_uid, output_dir, series_num, total_series):
    """Download a single series - simple approach"""
    # Create series directory
    series_dir = output_dir / f"series_{series_num:04d}_{series_uid[:20]}"
    series_dir.mkdir(exist_ok=True)
    
    # Skip if already exists and has content
    if series_dir.exists() and any(series_dir.iterdir()):
        return True
    
    # Download URL - try without format parameter first
    download_url = f"https://public.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet?seriesUID={series_uid}"
    output_file = series_dir / f"series_{series_uid[:20]}.dcm"
    
    try:
        # Download with curl
        curl_cmd = [
            'curl', '-L', '-o', str(output_file),
            '--connect-timeout', '30',
            '--max-time', '300',
            '--silent', '--show-error',
            download_url
        ]
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
            file_size = output_file.stat().st_size
            
            # Check if it's actually a valid file (not an error page)
            if file_size < 1000:  # Likely an error page
                print(f"❌ [{series_num}/{total_series}] Small file ({file_size} bytes) - likely error")
                output_file.unlink()  # Remove the error file
                return False
            
            # Check if it's HTML (error page)
            try:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(100)
                    if '<html' in content.lower() or '<!doctype' in content.lower():
                        print(f"❌ [{series_num}/{total_series}] HTML error page detected")
                        output_file.unlink()  # Remove the error file
                        return False
            except:
                pass  # Binary file, which is good
            
            print(f"✅ [{series_num}/{total_series}] Downloaded: {file_size:,} bytes")
            return True
        else:
            print(f"❌ [{series_num}/{total_series}] Download failed")
            return False
            
    except Exception as e:
        print(f"❌ [{series_num}/{total_series}] Error: {e}")
        return False

def download_manifest_simple(manifest_path, output_dir, description):
    """Download all series from a manifest - simple approach"""
    print(f"\n🏥 Downloading {description}...")
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
        if download_series_simple(series_uid, output_dir, i, len(series_uids)):
            success_count += 1
        
        # Small delay between downloads
        time.sleep(0.5)
    
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
    print("🏥 Simple TCIA DICOM Downloader")
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
    
    print(f"\n🚀 Starting DICOM image downloads...")
    print(f"📊 Expected total size: ~1.63TB")
    
    success_count = 0
    for download in downloads:
        try:
            if download_manifest_simple(
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
