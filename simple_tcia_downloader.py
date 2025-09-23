#!/usr/bin/env python3
"""
Simple TCIA DICOM Downloader
Uses the existing tcia_utils library to download DICOM images
"""

import os
import sys
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
        
        return series_uids
        
    except Exception as e:
        print(f"❌ Error parsing manifest: {e}")
        return []

def download_with_tcia_utils(series_uids, output_dir, description):
    """Download DICOM series using tcia_utils"""
    print(f"\n🏥 Downloading {description}...")
    print(f"   Series count: {len(series_uids)}")
    print(f"   Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import tcia_utils
        from tcia_utils import nbia
        
        print(f"🔍 Available methods in nbia: {[method for method in dir(nbia) if not method.startswith('_')]}")
        
        # Try to find the correct download method
        if hasattr(nbia, 'download_series'):
            print("✅ Using nbia.download_series")
            nbia.download_series(
                seriesUIDs=series_uids,
                path=str(output_dir),
                format="zip"
            )
        elif hasattr(nbia, 'downloadSeries'):
            print("✅ Using nbia.downloadSeries")
            nbia.downloadSeries(
                seriesUIDs=series_uids,
                path=str(output_dir),
                format="zip"
            )
        elif hasattr(nbia, 'download'):
            print("✅ Using nbia.download")
            nbia.download(
                seriesUIDs=series_uids,
                path=str(output_dir)
            )
        else:
            print("❌ No suitable download method found in tcia_utils")
            return False
        
        print(f"✅ Successfully downloaded {description}")
        return True
        
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
    print("🏥 Simple TCIA DICOM Images Downloader")
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
    
    success_count = 0
    for download in downloads:
        try:
            # Parse manifest to get series UIDs
            series_uids = parse_manifest_file(download["manifest"])
            if not series_uids:
                print(f"❌ No series found in {download['manifest']}")
                continue
            
            # Download in batches to avoid overwhelming the server
            batch_size = 100  # Download 100 series at a time
            total_batches = (len(series_uids) + batch_size - 1) // batch_size
            
            print(f"📦 Downloading in {total_batches} batches of {batch_size} series each")
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(series_uids))
                batch_series = series_uids[start_idx:end_idx]
                
                print(f"\n📦 Batch {batch_num + 1}/{total_batches}: Series {start_idx + 1}-{end_idx}")
                
                if download_with_tcia_utils(
                    batch_series, 
                    download["output"] / f"batch_{batch_num + 1:03d}",
                    f"{download['description']} - Batch {batch_num + 1}"
                ):
                    print(f"✅ Batch {batch_num + 1} completed successfully")
                else:
                    print(f"❌ Batch {batch_num + 1} failed")
                
                # Delay between batches to be respectful to the server
                if batch_num < total_batches - 1:
                    print("⏳ Waiting 30 seconds before next batch...")
                    time.sleep(30)
            
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
        print("📊 You now have the complete Breast DBT dataset with annotations")
    else:
        print(f"\n⚠️  {len(downloads) - success_count} downloads failed or were interrupted")
        print("💡 You can run this script again to resume interrupted downloads")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()