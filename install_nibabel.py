#!/usr/bin/env python3
"""
Script to install nibabel on the MONAI Label server
"""

import subprocess
import sys

def main():
    print("🔧 Installing nibabel for NIFTI support")
    print("=" * 50)
    
    try:
        # Install nibabel
        print("Installing nibabel...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel"])
        print("✅ nibabel installed successfully!")
        
        # Test import
        print("Testing nibabel import...")
        import nibabel as nib
        print(f"✅ nibabel version: {nib.__version__}")
        
        print("\n🎉 nibabel is ready for NIFTI file processing!")
        
    except Exception as e:
        print(f"❌ Error installing nibabel: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()





