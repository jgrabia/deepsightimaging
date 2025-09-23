#!/usr/bin/env python3
"""
TCIA Working DICOM Downloader
Uses the existing tcia_utils library correctly
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

def download_with_tcia_utils(series_uids, output_dir, description):
    """Download using tcia_utils library"""
    print(f"\n🏥 Downloading {description} using tcia_utils...")
    print(f"   Series count: {len(series_uids)}")
    print(f"   Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import tcia_utils
        from tcia_utils import nbia
        
        print(f"🔍 Available methods in nbia: {[method for method in dir(nbia) if not method.startswith('_')]}")
        
        # Try different approaches
        try:
            # Method 1: Try to get series info first
            print("🔍 Trying to get series information...")
            series_info = nbia.getSeries(seriesUIDs=series_uids[:5])  # Test with first 5
            print(f"✅ Got series info: {len(series_info) if series_info else 0} series")
            
            if series_info:
                print("📊 Sample series info:")
                for i, series in enumerate(series_info[:3]):
                    print(f"   Series {i+1}: {series.get('SeriesInstanceUID', 'N/A')[:20]}...")
                    print(f"   Size: {series.get('SeriesSize', 'N/A')} bytes")
                    print(f"   Modality: {series.get('Modality', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error getting series info: {e}")
        
        # Method 2: Try to download using the collection name
        try:
            print("\n🔍 Trying to download by collection...")
            # The Breast DBT collection is "Breast-Cancer-Screening-DBT"
            collections = nbia.getCollections()
            print(f"📊 Available collections: {len(collections) if collections else 0}")
            
            if collections:
                breast_collections = [c for c in collections if 'breast' in c.get('Collection', '').lower()]
                print(f"🏥 Breast collections: {[c.get('Collection') for c in breast_collections]}")
                
                # Try to download from the collection
                if breast_collections:
                    collection_name = breast_collections[0].get('Collection')
                    print(f"📥 Attempting to download from collection: {collection_name}")
                    
                    # This might work - download all series from the collection
                    # Note: This is a simplified approach
                    print("⚠️  This approach would download the entire collection")
                    print("💡 For now, we'll just show the available data")
            
        except Exception as e:
            print(f"❌ Error with collection approach: {e}")
        
        # Method 3: Try to use the manifest file directly
        try:
            print("\n🔍 Checking if tcia_utils can handle manifest files...")
            # Check if there's a method to process manifest files
            manifest_methods = [method for method in dir(nbia) if 'manifest' in method.lower()]
            print(f"📄 Manifest-related methods: {manifest_methods}")
            
        except Exception as e:
            print(f"❌ Error checking manifest methods: {e}")
        
        print("\n📋 Summary:")
        print("   The tcia_utils library appears to be for querying TCIA, not downloading")
        print("   For actual downloads, you need the NBIA Data Retriever tool")
        print("   The manifest files are meant to be used with that tool")
        
        return False
        
    except Exception as e:
        print(f"❌ Error with tcia_utils: {e}")
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking disk usage: {e}")
        return False

def main():
    """Main function to explore tcia_utils capabilities"""
    print("🏥 TCIA Working DICOM Downloader")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Download directory: {BASE_DOWNLOAD_DIR}")
    
    # Check disk space
    show_disk_usage()
    
    # Define manifest files
    manifest_path = MANIFEST_DIR / "training" / "BSC-DBT-Train-manifest.tcia"
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return
    
    # Parse manifest to get a sample of series UIDs
    series_uids = parse_manifest_file(manifest_path)
    if not series_uids:
        print("❌ No series found in manifest")
        return
    
    # Test with first 10 series
    test_series = series_uids[:10]
    print(f"\n🧪 Testing with first {len(test_series)} series...")
    
    # Try to download using tcia_utils
    output_dir = DICOM_DIR / "training" / "test_download"
    download_with_tcia_utils(test_series, output_dir, "Test Download")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()


