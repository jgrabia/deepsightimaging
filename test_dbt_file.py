#!/usr/bin/env python3
"""
Test script to verify DBT DICOM file reading
"""

import pydicom
import numpy as np
import sys

def test_dbt_file(file_path):
    """Test reading a DBT DICOM file"""
    try:
        print(f"Testing file: {file_path}")
        
        # Read DICOM
        ds = pydicom.dcmread(file_path)
        print("✅ DICOM file read successfully")
        
        # Get basic info
        print(f"📊 Modality: {ds.get('Modality', 'N/A')}")
        print(f"📊 Transfer Syntax: {ds.file_meta.TransferSyntaxUID if hasattr(ds, 'file_meta') else 'N/A'}")
        print(f"📊 SOP Class: {ds.get('SOPClassUID', 'N/A')}")
        
        # Try to get pixel array
        print("\n🔍 Testing pixel array access...")
        pixel_array = ds.pixel_array
        print("✅ Pixel array accessed successfully")
        
        print(f"📊 Shape: {pixel_array.shape}")
        print(f"📊 Data type: {pixel_array.dtype}")
        print(f"📊 Min value: {pixel_array.min()}")
        print(f"📊 Max value: {pixel_array.max()}")
        print(f"📊 Mean value: {pixel_array.mean():.2f}")
        
        # Test slice access
        if len(pixel_array.shape) == 3:
            print(f"\n🔍 Testing slice access...")
            middle_slice = pixel_array.shape[0] // 2
            slice_data = pixel_array[middle_slice, :, :]
            print(f"✅ Middle slice ({middle_slice}) accessed successfully")
            print(f"📊 Slice shape: {slice_data.shape}")
            print(f"📊 Slice data type: {slice_data.dtype}")
            print(f"📊 Slice min/max: {slice_data.min()} / {slice_data.max()}")
            
            # Test normalization
            print(f"\n🔍 Testing normalization...")
            if slice_data.max() > 255:
                normalized = ((slice_data / slice_data.max()) * 255).astype(np.uint8)
                print(f"✅ Normalization successful")
                print(f"📊 Normalized shape: {normalized.shape}")
                print(f"📊 Normalized data type: {normalized.dtype}")
                print(f"📊 Normalized min/max: {normalized.min()} / {normalized.max()}")
            else:
                print("ℹ️  No normalization needed (already 8-bit range)")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_dbt_file.py <path_to_dicom_file>")
        print("\nExample:")
        print("python test_dbt_file.py 'C:/MRIAPP/dicom_download/images/manifest-1617905855234/Breast-Cancer-Screening-DBT/DBT-P00003/01-01-2000-DBT-S01306-MAMMO screening digital bilateral-33603/18377.000000-NA-92351/1-1.dcm'")
    else:
        test_dbt_file(sys.argv[1])


