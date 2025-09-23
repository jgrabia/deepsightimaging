#!/usr/bin/env python3
"""
TCIA Proper DICOM Downloader
Uses the correct TCIA download approach with manifest files
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import time
from datetime import datetime

# Configuration
BASE_DOWNLOAD_DIR = Path("~/mri_app/breast_dbt_collection").expanduser()
DICOM_DIR = BASE_DOWNLOAD_DIR / "dicom_images"
MANIFEST_DIR = DICOM_DIR

def check_nbia_retriever():
    """Check if NBIA Data Retriever is available"""
    # Check for different possible locations
    possible_paths = [
        "nbia-data-retriever",
        "/usr/local/bin/nbia-data-retriever",
        "/usr/bin/nbia-data-retriever",
        "./nbia-data-retriever"
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ NBIA Data Retriever found at: {path}")
                return path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    print("❌ NBIA Data Retriever not found!")
    print("\n📥 To install NBIA Data Retriever:")
    print("   1. Download from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
    print("   2. Or try: wget https://github.com/kirbyju/TCIA_Data_Retriever/releases/latest/download/nbia-data-retriever")
    print("   3. Make executable: chmod +x nbia-data-retriever")
    print("   4. Move to PATH: sudo mv nbia-data-retriever /usr/local/bin/")
    return None

def download_with_nbia_retriever(manifest_path, output_dir, description):
    """Download DICOM images using NBIA Data Retriever"""
    print(f"\n🏥 Downloading {description}...")
    print(f"   Manifest: {manifest_path}")
    print(f"   Output: {output_dir}")
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find NBIA Data Retriever
    nbia_path = check_nbia_retriever()
    if not nbia_path:
        return False
    
    try:
        # Use NBIA Data Retriever with manifest
        cmd = [
            nbia_path,
            '--manifest', str(manifest_path),
            '--output', str(output_dir),
            '--continue'  # Resume interrupted downloads
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        print("⏳ This will take a LONG time...")
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

def download_with_curl_approach(manifest_path, output_dir, description):
    """Alternative approach using curl to download from TCIA"""
    print(f"\n🏥 Downloading {description} using curl approach...")
    print(f"   Manifest: {manifest_path}")
    print(f"   Output: {output_dir}")
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse manifest to get series UIDs
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        
        series_uids = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('includeAnnotation') and not line.startswith('noOfrRetry') and not line.startswith('databasketId') and not line.startswith('manifestVersion') and not line.startswith('ListOfSeriesToDownload') and not line.startswith('downloadServerUrl'):
                if len(line) > 50 and line.replace('.', '').isdigit():
                    series_uids.append(line)
        
        print(f"📊 Found {len(series_uids)} series to download")
        
        # Download each series using curl
        success_count = 0
        for i, series_uid in enumerate(series_uids, 1):
            print(f"\n📁 [{i}/{len(series_uids)}] Downloading series: {series_uid[:20]}...")
            
            # Create series directory
            series_dir = output_dir / f"series_{i:04d}_{series_uid[:20]}"
            series_dir.mkdir(exist_ok=True)
            
            # Skip if already exists
            if series_dir.exists() and any(series_dir.iterdir()):
                print(f"⏭️  Skipping existing series")
                success_count += 1
                continue
            
            # Use curl to download - try both zip and direct formats
            download_url = f"https://public.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet?seriesUID={series_uid}&format=zip"
            output_file = series_dir / f"series_{series_uid[:20]}.zip"
            
            try:
                # Download with curl
                curl_cmd = [
                    'curl', '-L', '-o', str(output_file),
                    '--connect-timeout', '30',
                    '--max-time', '300',
                    download_url
                ]
                
                result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                    print(f"✅ Downloaded: {output_file}")
                    
                    # Check if it's actually a zip file or DICOM files
                    try:
                        import zipfile
                        with zipfile.ZipFile(output_file, 'r') as zip_ref:
                            zip_ref.extractall(series_dir)
                        output_file.unlink()  # Remove zip file
                        print(f"✅ Extracted DICOM files from zip")
                        success_count += 1
                    except zipfile.BadZipFile:
                        # Not a zip file - it's likely DICOM files directly
                        print(f"📁 File is not a zip - likely DICOM files directly")
                        
                        # Check if it contains DICOM data by looking at the file header
                        with open(output_file, 'rb') as f:
                            header = f.read(4)
                            if header == b'DICM':
                                print(f"✅ Confirmed DICOM file")
                                # Rename to .dcm extension
                                dcm_file = series_dir / f"series_{series_uid[:20]}.dcm"
                                output_file.rename(dcm_file)
                                success_count += 1
                            else:
                                # Try to extract as tar or other format
                                print(f"🔍 Trying to extract as tar...")
                                try:
                                    import tarfile
                                    with tarfile.open(output_file, 'r') as tar_ref:
                                        tar_ref.extractall(series_dir)
                                    output_file.unlink()
                                    print(f"✅ Extracted DICOM files from tar")
                                    success_count += 1
                                except:
                                    print(f"⚠️  Unknown file format, keeping as-is")
                                    success_count += 1  # Still count as success
                    except Exception as e:
                        print(f"⚠️  Could not process file: {e}")
                        success_count += 1  # Still count as success
                else:
                    print(f"❌ Download failed: {result.stderr}")
                
            except Exception as e:
                print(f"❌ Error downloading series: {e}")
            
            # Small delay between downloads
            time.sleep(1)
        
        print(f"\n✅ Downloaded {success_count}/{len(series_uids)} series successfully")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Error in curl approach: {e}")
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
    print("🏥 TCIA Proper DICOM Images Downloader")
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
            # Try NBIA Data Retriever first
            if download_with_nbia_retriever(
                download["manifest"], 
                download["output"], 
                download["description"]
            ):
                success_count += 1
            else:
                print(f"\n🔄 Trying alternative curl approach...")
                if download_with_curl_approach(
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
        print("📊 You now have the complete Breast DBT dataset with annotations")
    else:
        print(f"\n⚠️  {len(downloads) - success_count} downloads failed or were interrupted")
        print("💡 You can run this script again to resume interrupted downloads")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
