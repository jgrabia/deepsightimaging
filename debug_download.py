#!/usr/bin/env python3
"""
Debug script to test TCIA downloads
"""

import os
import sys
import tempfile
import requests
from datetime import datetime

def test_tcia_download():
    """Test a simple TCIA download to see where files go"""
    
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Python executable: {sys.executable}")
    
    # Test TCIA API
    try:
        from tcia_utils import nbia
        
        print("✅ TCIA utils imported successfully")
        
        # Get a small sample of INbreast series
        filters = {
            "Collection": "INbreast",
            "BodyPartExamined": "BREAST",
            "Modality": "MG"
        }
        
        print("🔍 Querying TCIA for INbreast series...")
        series = nbia.getSeries(**filters)
        print(f"✅ Found {len(series)} series")
        
        if len(series) > 0:
            # Try to download the first series
            first_series = series[0]
            series_uid = first_series.get('SeriesInstanceUID', '')
            
            if series_uid:
                print(f"🔍 Testing download of series: {series_uid}")
                
                # Create test directory
                test_dir = "/home/ubuntu/mri_app/test_download"
                os.makedirs(test_dir, exist_ok=True)
                
                # Download URL
                url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
                
                print(f"🔍 Downloading from: {url}")
                print(f"🔍 Saving to: {test_dir}")
                
                # Download with progress
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Save to test file
                test_file = os.path.join(test_dir, f"{series_uid}.zip")
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(test_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r📥 Downloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                
                print(f"\n✅ Download complete: {test_file}")
                print(f"📁 File size: {os.path.getsize(test_file)} bytes")
                
                # Check if file exists and has content
                if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                    print("✅ File downloaded successfully!")
                    
                    # List contents of test directory
                    print(f"📁 Contents of {test_dir}:")
                    for item in os.listdir(test_dir):
                        item_path = os.path.join(test_dir, item)
                        size = os.path.getsize(item_path)
                        print(f"   {item} ({size} bytes)")
                else:
                    print("❌ File download failed or is empty")
            else:
                print("❌ No SeriesInstanceUID found in first series")
        else:
            print("❌ No series found")
            
    except ImportError as e:
        print(f"❌ Failed to import tcia_utils: {e}")
    except Exception as e:
        print(f"❌ Error during download test: {e}")

if __name__ == "__main__":
    test_tcia_download()




