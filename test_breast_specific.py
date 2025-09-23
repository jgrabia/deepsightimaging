#!/usr/bin/env python3
"""
Test different MONAI models for breast imaging
"""

import requests
import json
import tempfile
import os
from pathlib import Path

def test_model_on_breast_image(model_name, image_path):
    """Test a specific model on breast image"""
    print(f"\n🔬 Testing {model_name} on breast image...")
    
    try:
        # Test the model
        url = f"http://localhost:8000/infer/{model_name}"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'output': 'json'}
            
            response = requests.post(url, files=files, params=params, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model_name} - Success!")
                print(f"   Result: {result}")
                return True
            else:
                print(f"❌ {model_name} - Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ {model_name} - Error: {e}")
        return False

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get("http://localhost:8000/info/")
        if response.status_code == 200:
            info = response.json()
            return info.get('models', [])
    except:
        pass
    return []

def main():
    print("🏥 Testing MONAI Models for Breast Imaging")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/info/")
        if response.status_code != 200:
            print("❌ MONAI server not accessible")
            return
        print("✅ MONAI server is accessible")
    except:
        print("❌ Cannot connect to MONAI server")
        return
    
    # Get available models
    models = get_available_models()
    print(f"\n📊 Available models: {models}")
    
    # Find a breast image to test with
    breast_image = None
    search_paths = [
        "~/mri_app/downloads",
        "~/mri_app/dicom_download",
        "./dicom_download"
    ]
    
    for path in search_paths:
        path = Path(path).expanduser()
        if path.exists():
            # Look for any .dcm files
            dcm_files = list(path.rglob("*.dcm"))
            if dcm_files:
                breast_image = str(dcm_files[0])
                print(f"📁 Found test image: {breast_image}")
                break
    
    if not breast_image:
        print("❌ No DICOM files found for testing")
        return
    
    # Test each model
    print(f"\n🧪 Testing models on: {os.path.basename(breast_image)}")
    
    results = {}
    for model in models:
        success = test_model_on_breast_image(model, breast_image)
        results[model] = success
    
    # Summary
    print(f"\n📋 Test Summary:")
    print("=" * 30)
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model}")
    
    print(f"\n💡 Recommendations:")
    print("- Try 'deepgrow_2d' for manual breast annotation")
    print("- Try 'deepgrow_3d' for 3D breast analysis")
    print("- Consider training a breast-specific model")

if __name__ == "__main__":
    main()





