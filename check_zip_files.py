#!/usr/bin/env python3
"""
Check ZIP files in training data directory
"""

import os

def check_zip_files():
    """Check what ZIP files are in the training data directory"""
    
    base_dir = "training_data"
    print(f"🔍 Checking directory: {os.path.abspath(base_dir)}")
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory {base_dir} does not exist!")
        return
    
    print(f"📁 Contents of {base_dir}:")
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            print(f"   📂 {item}/")
            
            # Check for ZIP files in subdirectories
            try:
                zip_files = [f for f in os.listdir(item_path) if f.endswith('.zip')]
                print(f"      📦 ZIP files: {len(zip_files)}")
                if zip_files:
                    print(f"      📦 First few: {zip_files[:3]}")
            except Exception as e:
                print(f"      ❌ Error reading {item}: {e}")
        else:
            print(f"   📄 {item}")
    
    # Check INbreast specifically
    inbreast_dir = os.path.join(base_dir, "INbreast")
    if os.path.exists(inbreast_dir):
        print(f"\n🔍 Checking INbreast directory specifically:")
        try:
            zip_files = [f for f in os.listdir(inbreast_dir) if f.endswith('.zip')]
            print(f"   📦 ZIP files in INbreast: {len(zip_files)}")
            if zip_files:
                print(f"   📦 First few: {zip_files[:5]}")
        except Exception as e:
            print(f"   ❌ Error reading INbreast: {e}")

if __name__ == "__main__":
    check_zip_files()




