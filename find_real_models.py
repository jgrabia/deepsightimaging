#!/usr/bin/env python3
"""
Script to find real pre-trained cancer detection models
"""

import requests
import json
import re

def search_monai_models():
    """Search for real MONAI pre-trained models"""
    print("🔍 Searching for Real Pre-trained Cancer Detection Models")
    print("=" * 70)
    
    # Known MONAI model repositories and sources
    sources = [
        {
            "name": "MONAI Model Zoo",
            "url": "https://github.com/Project-MONAI/MONAI",
            "description": "Official MONAI repository"
        },
        {
            "name": "MONAI Label Sample Apps",
            "url": "https://github.com/Project-MONAI/MONAILabel",
            "description": "MONAI Label sample applications"
        },
        {
            "name": "Medical Segmentation Decathlon",
            "url": "https://decathlon-10.grand-challenge.org/",
            "description": "10 medical segmentation challenges"
        },
        {
            "name": "LIDC-IDRI Models",
            "url": "https://www.cancerimagingarchive.net/collections/",
            "description": "Lung nodule detection models"
        }
    ]
    
    print("📋 **Available Model Sources:**")
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['name']}")
        print(f"   URL: {source['url']}")
        print(f"   Description: {source['description']}")
        print()
    
    print("🔧 **Real Model Options:**")
    print()
    
    # Option 1: Use existing MONAI Label models
    print("1. **Use Existing MONAI Label Models**")
    print("   - The radiology sample app already has segmentation models")
    print("   - These can be adapted for cancer detection")
    print("   - Models: segmentation, segmentation_spleen, etc.")
    print()
    
    # Option 2: Download from Medical Segmentation Decathlon
    print("2. **Medical Segmentation Decathlon Models**")
    print("   - Task 04: Hippocampus (brain tumor related)")
    print("   - Task 06: Lung (lung cancer segmentation)")
    print("   - Task 07: Pancreas (pancreatic cancer)")
    print("   - Task 10: Colon (colon cancer)")
    print()
    
    # Option 3: Use LIDC-IDRI specific models
    print("3. **LIDC-IDRI Lung Nodule Models**")
    print("   - Specialized for lung nodule detection")
    print("   - Available from various research groups")
    print("   - Can be found on GitHub and model zoos")
    print()
    
    # Option 4: Create custom models
    print("4. **Custom Model Training**")
    print("   - Train models on TCIA datasets")
    print("   - Use MONAI training framework")
    print("   - Requires labeled data")
    print()

def get_monai_label_models():
    """Get available models from MONAI Label sample apps"""
    print("📊 **Available MONAI Label Models:**")
    print("-" * 50)
    
    models = [
        {
            "name": "segmentation",
            "description": "Multi-organ segmentation (liver, spleen, kidneys)",
            "applicable": "Can be adapted for tumor detection"
        },
        {
            "name": "segmentation_spleen", 
            "description": "Spleen-specific segmentation",
            "applicable": "Good for organ-specific cancer detection"
        },
        {
            "name": "localization_spine",
            "description": "Spine localization",
            "applicable": "Can detect bone metastases"
        },
        {
            "name": "localization_vertebra",
            "description": "Vertebra localization", 
            "applicable": "Can detect bone metastases"
        },
        {
            "name": "deepgrow_2d",
            "description": "2D interactive segmentation",
            "applicable": "Good for manual tumor annotation"
        },
        {
            "name": "deepgrow_3d",
            "description": "3D interactive segmentation",
            "applicable": "Good for 3D tumor analysis"
        },
        {
            "name": "deepedit",
            "description": "Deep learning editing",
            "applicable": "Can refine tumor boundaries"
        }
    ]
    
    for model in models:
        print(f"🔹 {model['name']}")
        print(f"   Description: {model['description']}")
        print(f"   Cancer Detection: {model['applicable']}")
        print()

def recommend_approach():
    """Recommend the best approach for cancer detection"""
    print("🎯 **Recommended Approach for Cancer Detection:**")
    print("=" * 60)
    
    print("1. **Immediate Testing (Use Existing Models)**")
    print("   ✅ Use the existing 'segmentation' model")
    print("   ✅ Test with lung and breast CT scans from TCIA")
    print("   ✅ The model can detect abnormal regions")
    print("   ✅ No additional downloads needed")
    print()
    
    print("2. **Enhanced Detection (Add Specialized Models)**")
    print("   🔍 Download Medical Segmentation Decathlon models")
    print("   🔍 Focus on Task 06 (Lung) for lung cancer")
    print("   🔍 Use LIDC-IDRI specific models for nodules")
    print("   🔍 Requires model conversion to MONAI format")
    print()
    
    print("3. **Custom Training (Long-term Solution)**")
    print("   🎓 Train models on TCIA cancer datasets")
    print("   🎓 Use MONAI training framework")
    print("   🎓 Requires labeled data and GPU resources")
    print("   🎓 Best for production use")
    print()
    
    print("💡 **Immediate Action Plan:**")
    print("1. Test existing 'segmentation' model with cancer datasets")
    print("2. Download lung CT scans from LIDC-IDRI collection")
    print("3. Download breast CT scans from TCGA-BRCA collection")
    print("4. Evaluate detection performance")
    print("5. Consider adding specialized models if needed")

if __name__ == "__main__":
    search_monai_models()
    print("\n" + "="*70)
    get_monai_label_models()
    print("\n" + "="*70)
    recommend_approach()





