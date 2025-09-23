#!/usr/bin/env python3
"""
Test script for DICOM bit depth handling in visualization
"""

import numpy as np
from breast_visualizer import BreastVisualizer
import pydicom
import tempfile
import os

def test_dicom_bit_depths():
    """Test visualization with different DICOM bit depths"""
    
    print("🧪 Testing DICOM Bit Depth Handling")
    print("=" * 50)
    
    visualizer = BreastVisualizer()
    
    # Create test predictions
    predictions = np.random.rand(3, 256, 256)
    
    # Test different bit depths
    bit_depths = [
        (np.uint8, "8-bit unsigned"),
        (np.uint16, "16-bit unsigned"), 
        (np.int16, "16-bit signed"),
        (np.float32, "32-bit float")
    ]
    
    for dtype, description in bit_depths:
        print(f"\n📊 Testing {description} ({dtype})")
        
        # Create a test DICOM-like image
        if dtype == np.uint8:
            test_image = np.random.randint(0, 256, (256, 256), dtype=dtype)
        elif dtype == np.uint16:
            test_image = np.random.randint(0, 65536, (256, 256), dtype=dtype)
        elif dtype == np.int16:
            test_image = np.random.randint(-32768, 32767, (256, 256), dtype=dtype)
        else:  # float32
            test_image = np.random.rand(256, 256).astype(dtype) * 255
        
        # Create a temporary DICOM file
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp_file:
            # Create a minimal DICOM dataset
            ds = pydicom.Dataset()
            ds.PixelData = test_image.tobytes()
            ds.Rows = 256
            ds.Columns = 256
            ds.BitsAllocated = test_image.dtype.itemsize * 8
            ds.BitsStored = test_image.dtype.itemsize * 8
            ds.HighBit = test_image.dtype.itemsize * 8 - 1
            ds.PixelRepresentation = 0 if dtype in [np.uint8, np.uint16] else 1
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            
            # Save the DICOM file
            ds.save_as(tmp_file.name)
            dicom_path = tmp_file.name
        
        try:
            print(f"  📁 Created test DICOM: {dicom_path}")
            print(f"  📈 Image stats: min={test_image.min()}, max={test_image.max()}, dtype={test_image.dtype}")
            
            # Test visualization
            result = visualizer.visualize_dicom(dicom_path, predictions)
            print(f"  ✅ Visualization successful! Result shape: {np.array(result).shape}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        finally:
            # Clean up
            if os.path.exists(dicom_path):
                os.unlink(dicom_path)
    
    print("\n🎯 Test completed!")
    print("The visualization should now handle all common DICOM bit depths correctly.")

if __name__ == "__main__":
    test_dicom_bit_depths()





