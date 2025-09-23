#!/usr/bin/env python3
"""
Debug MONAI server errors and test with proper DICOM files
"""

import requests
import json
import tempfile
import os
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path

def check_monai_server_status():
    """Check MONAI server status and get detailed info"""
    print("🔍 Checking MONAI server status...")
    
    try:
        # Check basic connectivity
        response = requests.get("http://localhost:8000/info/", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print("✅ Server is running")
            print(f"   App: {info.get('name', 'Unknown')}")
            print(f"   Version: {info.get('version', 'Unknown')}")
            print(f"   Models: {list(info.get('models', {}).keys())}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_dicom_file(file_path):
    """Test if DICOM file is valid and readable"""
    print(f"\n🔍 Testing DICOM file: {os.path.basename(file_path)}")
    
    try:
        # Try to read with pydicom
        ds = pydicom.dcmread(file_path)
        print(f"✅ DICOM file is valid")
        print(f"   Modality: {ds.get('Modality', 'Unknown')}")
        print(f"   Body Part: {ds.get('BodyPartExamined', 'Unknown')}")
        print(f"   Image Size: {ds.pixel_array.shape if hasattr(ds, 'pixel_array') else 'Unknown'}")
        print(f"   Bits Allocated: {ds.get('BitsAllocated', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ DICOM file error: {e}")
        return False

def convert_dicom_to_nifti(dicom_path):
    """Convert DICOM to NIFTI format"""
    print(f"\n🔄 Converting DICOM to NIFTI...")
    
    try:
        # Read DICOM
        ds = pydicom.dcmread(dicom_path)
        
        # Get pixel data
        pixel_array = ds.pixel_array
        
        # Create NIFTI image
        nifti_img = nib.Nifti1Image(pixel_array, np.eye(4))
        
        # Save to temporary file
        temp_nifti = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
        nib.save(nifti_img, temp_nifti.name)
        
        print(f"✅ Converted to NIFTI: {temp_nifti.name}")
        return temp_nifti.name
        
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None

def test_model_with_nifti(model_name, nifti_path):
    """Test model with NIFTI file"""
    print(f"\n🔬 Testing {model_name} with NIFTI...")
    
    try:
        url = f"http://localhost:8000/infer/{model_name}"
        
        with open(nifti_path, 'rb') as f:
            files = {'file': f}
            params = {'output': 'json'}
            
            response = requests.post(url, files=files, params=params, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model_name} - Success!")
                print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return True
            else:
                print(f"❌ {model_name} - Failed: {response.status_code}")
                print(f"   Error: {response.text[:200]}...")  # Show first 200 chars
                return False
                
    except Exception as e:
        print(f"❌ {model_name} - Error: {e}")
        return False

def find_better_test_files():
    """Find better test files"""
    print("\n🔍 Looking for better test files...")
    
    search_paths = [
        "~/mri_app/dicom_download",
        "~/mri_app/downloads",
        "./dicom_download"
    ]
    
    good_files = []
    
    for path in search_paths:
        path = Path(path).expanduser()
        if path.exists():
            print(f"   Searching in: {path}")
            
            # Look for DICOM files
            dcm_files = list(path.rglob("*.dcm"))
            for dcm_file in dcm_files[:5]:  # Test first 5 files
                if test_dicom_file(str(dcm_file)):
                    good_files.append(str(dcm_file))
                    if len(good_files) >= 3:  # Get 3 good files
                        break
    
    return good_files

def main():
    print("🏥 Debugging MONAI Server Errors")
    print("=" * 50)
    
    # Check server status
    if not check_monai_server_status():
        print("\n❌ Server is not accessible. Please start the MONAI server first.")
        return
    
    # Find good test files
    test_files = find_better_test_files()
    
    if not test_files:
        print("\n❌ No valid DICOM files found for testing")
        return
    
    print(f"\n📁 Found {len(test_files)} valid test files")
    
    # Test with the first good file
    test_file = test_files[0]
    print(f"\n🧪 Using test file: {os.path.basename(test_file)}")
    
    # Convert to NIFTI
    nifti_path = convert_dicom_to_nifti(test_file)
    if not nifti_path:
        print("❌ Failed to convert DICOM to NIFTI")
        return
    
    # Test a few key models
    key_models = ['segmentation', 'deepgrow_2d', 'sw_fastedit']
    
    print(f"\n🧪 Testing key models...")
    results = {}
    
    for model in key_models:
        success = test_model_with_nifti(model, nifti_path)
        results[model] = success
    
    # Summary
    print(f"\n📋 Test Summary:")
    print("=" * 30)
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model}")
    
    # Cleanup
    try:
        os.unlink(nifti_path)
    except:
        pass
    
    print(f"\n💡 Next Steps:")
    if any(results.values()):
        print("- Some models work! Use the working ones for breast imaging")
    else:
        print("- All models failed. Check server logs for detailed errors")
        print("- Try restarting the MONAI server")
        print("- Consider using different DICOM files")

if __name__ == "__main__":
    main()





