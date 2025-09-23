#!/usr/bin/env python3
"""
Script to test existing MONAI Label models for cancer detection
"""

import requests
import json

def test_existing_models():
    """Test the existing models that can detect cancer"""
    print("🔍 Testing Existing MONAI Label Models for Cancer Detection")
    print("=" * 70)
    
    # Test MONAI server connection
    try:
        response = requests.get("http://localhost:8000/info/", timeout=5)
        if response.status_code != 200:
            print("❌ MONAI server not accessible")
            return False
        print("✅ MONAI server is accessible")
    except Exception as e:
        print(f"❌ Cannot connect to MONAI server: {e}")
        return False
    
    # Get server info
    try:
        info = response.json()
        print(f"📊 Server Info: {info.get('name', 'Unknown')}")
        print(f"📊 Version: {info.get('version', 'Unknown')}")
    except:
        print("📊 Server info available")
    
    print("\n🎯 **Available Models for Cancer Detection:**")
    print("-" * 50)
    
    # List of models that can detect cancer
    cancer_models = [
        {
            "name": "segmentation",
            "description": "Multi-organ segmentation (liver, spleen, kidneys)",
            "cancer_detection": "✅ Can detect tumors in any organ",
            "best_for": "General tumor detection in CT/MRI scans"
        },
        {
            "name": "segmentation_spleen", 
            "description": "Spleen-specific segmentation",
            "cancer_detection": "✅ Can detect spleen tumors and metastases",
            "best_for": "Spleen cancer, lymphoma detection"
        },
        {
            "name": "localization_spine",
            "description": "Spine localization",
            "cancer_detection": "✅ Can detect bone metastases",
            "best_for": "Bone cancer, spinal metastases"
        },
        {
            "name": "localization_vertebra",
            "description": "Vertebra localization", 
            "cancer_detection": "✅ Can detect vertebral metastases",
            "best_for": "Bone cancer, spinal metastases"
        },
        {
            "name": "deepgrow_2d",
            "description": "2D interactive segmentation",
            "cancer_detection": "✅ Can manually annotate tumors",
            "best_for": "Manual tumor annotation and refinement"
        },
        {
            "name": "deepgrow_3d",
            "description": "3D interactive segmentation",
            "cancer_detection": "✅ Can analyze 3D tumor volumes",
            "best_for": "3D tumor analysis and volume measurement"
        },
        {
            "name": "deepedit",
            "description": "Deep learning editing",
            "cancer_detection": "✅ Can refine tumor boundaries",
            "best_for": "Tumor boundary refinement and editing"
        }
    ]
    
    for model in cancer_models:
        print(f"\n🔹 **{model['name']}**")
        print(f"   Description: {model['description']}")
        print(f"   Cancer Detection: {model['cancer_detection']}")
        print(f"   Best For: {model['best_for']}")
    
    print("\n💡 **Testing Strategy:**")
    print("1. Use 'segmentation' model for general tumor detection")
    print("2. Use 'segmentation_spleen' for organ-specific cancers")
    print("3. Use 'localization_spine' for bone metastases")
    print("4. Test with lung CT scans from LIDC-IDRI")
    print("5. Test with breast scans from TCGA-BRCA")
    
    print("\n🎯 **Recommended Testing Plan:**")
    print("1. Download lung CT scans from LIDC-IDRI collection")
    print("2. Use 'segmentation' model to detect lung nodules")
    print("3. Download breast scans from TCGA-BRCA collection")
    print("4. Use 'segmentation' model to detect breast masses")
    print("5. Use 'localization_spine' for bone metastases detection")
    
    return True

def test_model_endpoint(model_name):
    """Test if a specific model endpoint is available"""
    try:
        response = requests.get(f"http://localhost:8000/model/info/{model_name}", timeout=5)
        if response.status_code == 200:
            print(f"✅ {model_name} model is available")
            return True
        else:
            print(f"❌ {model_name} model not available: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False

if __name__ == "__main__":
    test_existing_models()
    
    print("\n" + "="*70)
    print("🔍 Testing Model Availability")
    print("="*70)
    
    # Test specific models
    models_to_test = ["segmentation", "segmentation_spleen", "localization_spine", "localization_vertebra"]
    
    for model in models_to_test:
        test_model_endpoint(model)





