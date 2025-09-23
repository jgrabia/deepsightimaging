#!/usr/bin/env python3
"""
Test extraction script
"""

import os

def test_extraction():
    """Test if we can find the ZIP files"""
    
    base_dir = "/home/ubuntu/training_data"
    print(f"🔍 Checking directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory {base_dir} does not exist!")
        return
    
    print(f"✅ Directory exists!")
    
    # Check what's in the directory
    items = os.listdir(base_dir)
    print(f"📁 Contents: {items}")
    
    # Check INbreast specifically
    inbreast_dir = os.path.join(base_dir, "INbreast")
    if os.path.exists(inbreast_dir):
        print(f"✅ INbreast directory exists!")
        
        # Count ZIP files
        try:
            zip_files = [f for f in os.listdir(inbreast_dir) if f.endswith('.zip')]
            print(f"📦 Found {len(zip_files)} ZIP files")
            
            if zip_files:
                print(f"📦 First few: {zip_files[:5]}")
                
                # Test opening one ZIP file
                test_zip = os.path.join(inbreast_dir, zip_files[0])
                print(f"🔍 Testing ZIP file: {test_zip}")
                
                import zipfile
                with zipfile.ZipFile(test_zip, 'r') as zf:
                    files_in_zip = zf.namelist()
                    print(f"📄 Files in ZIP: {len(files_in_zip)}")
                    print(f"📄 First few: {files_in_zip[:3]}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ INbreast directory does not exist!")

if __name__ == "__main__":
    test_extraction()




