#!/usr/bin/env python3
"""
Install NBIA Data Retriever and Download DICOM Images
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import time
from datetime import datetime

# Configuration
BASE_DOWNLOAD_DIR = Path("~/mri_app/breast_dbt_collection").expanduser()
DICOM_DIR = BASE_DOWNLOAD_DIR / "dicom_images"
MANIFEST_DIR = DICOM_DIR

def install_nbia_retriever():
    """Install NBIA Data Retriever"""
    print("🔧 Installing NBIA Data Retriever...")
    
    # Check if already installed
    try:
        result = subprocess.run(['nbia-data-retriever', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ NBIA Data Retriever already installed: {result.stdout.strip()}")
            return True
    except:
        pass
    
    # Try different installation methods
    print("📥 Attempting to download NBIA Data Retriever...")
    
    # Method 1: Try to download from GitHub releases
    try:
        print("🔍 Trying GitHub releases...")
        github_url = "https://github.com/kirbyju/TCIA_Data_Retriever/releases/latest/download/nbia-data-retriever"
        
        response = requests.get(github_url, timeout=30)
        if response.status_code == 200:
            with open("nbia-data-retriever", "wb") as f:
                f.write(response.content)
            
            # Make executable
            os.chmod("nbia-data-retriever", 0o755)
            
            # Test it
            result = subprocess.run(['./nbia-data-retriever', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Downloaded and installed: {result.stdout.strip()}")
                
                # Move to system path
                subprocess.run(['sudo', 'mv', 'nbia-data-retriever', '/usr/local/bin/'], check=True)
                print("✅ Moved to /usr/local/bin/")
                return True
        else:
            print(f"❌ GitHub download failed: {response.status_code}")
    except Exception as e:
        print(f"❌ GitHub method failed: {e}")
    
    # Method 2: Try to install via package manager
    try:
        print("🔍 Trying package manager installation...")
        # Try different package managers
        for cmd in [['sudo', 'apt', 'install', '-y', 'nbia-data-retriever'],
                   ['sudo', 'yum', 'install', '-y', 'nbia-data-retriever'],
                   ['sudo', 'dnf', 'install', '-y', 'nbia-data-retriever']]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ Installed via package manager")
                    return True
            except:
                continue
    except Exception as e:
        print(f"❌ Package manager method failed: {e}")
    
    print("❌ Could not install NBIA Data Retriever automatically")
    print("\n📋 Manual installation instructions:")
    print("1. Visit: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
    print("2. Download the Linux version")
    print("3. Make executable: chmod +x nbia-data-retriever")
    print("4. Move to PATH: sudo mv nbia-data-retriever /usr/local/bin/")
    
    return False

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
    
    try:
        # Use NBIA Data Retriever with manifest
        cmd = [
            'nbia-data-retriever',
            '--manifest', str(manifest_path),
            '--output', str(output_dir),
            '--continue'  # Resume interrupted downloads
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        print("⏳ This will take a VERY long time (1.63TB)...")
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
    """Main function"""
    print("🏥 Install NBIA Data Retriever and Download DICOM Images")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
    
    # Check disk space
    if not show_disk_usage():
        print("❌ Insufficient disk space. Please free up space or extend your volume.")
        return
    
    # Install NBIA Data Retriever
    if not install_nbia_retriever():
        print("❌ Cannot proceed without NBIA Data Retriever")
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
            if download_with_nbia_retriever(
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


