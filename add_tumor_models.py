#!/usr/bin/env python3
"""
Script to add breast and lung cancer detection models to MONAI Label server
"""

import os
import requests
import json
import tempfile
from pathlib import Path

def download_model(model_name, model_url, model_path):
    """Download a pre-trained model"""
    print(f"📥 Downloading {model_name}...")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def add_model_to_monai(model_name, model_path, server_url="http://localhost:8000"):
    """Add model to MONAI Label server"""
    print(f"🔧 Adding {model_name} to MONAI server...")
    
    try:
        with open(model_path, 'rb') as f:
            files = {'file': f}
            response = requests.put(f"{server_url}/model/{model_name}", files=files)
        
        if response.status_code == 200:
            print(f"✅ {model_name} added successfully!")
            return True
        else:
            print(f"❌ Failed to add {model_name}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error adding {model_name}: {e}")
        return False

def create_dummy_model(model_name, model_path):
    """Create a dummy model file for testing (since real models aren't available)"""
    print(f"🔧 Creating dummy model for {model_name}...")
    
    try:
        # Create a simple PyTorch model structure
        import torch
        import torch.nn as nn
        
        class DummyTumorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 1, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.sigmoid(self.conv3(x))
                return x
        
        # Create model instance
        model = DummyTumorModel()
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        
        print(f"✅ Dummy model created for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to create dummy model for {model_name}: {e}")
        return False

def main():
    print("🏥 Adding Breast and Lung Cancer Detection Models")
    print("=" * 60)
    
    # Model configurations - using dummy models for now
    models = {
        "breast_tumor_detection": {
            "description": "Breast tumor detection and segmentation model",
            "body_part": "BREAST",
            "modality": "CT",
            "type": "dummy"
        },
        "lung_nodule_detection": {
            "description": "Lung nodule detection and classification model",
            "body_part": "CHEST",
            "modality": "CT",
            "type": "dummy"
        },
        "lung_cancer_segmentation": {
            "description": "Lung cancer tumor segmentation model", 
            "body_part": "CHEST",
            "modality": "CT",
            "type": "dummy"
        }
    }
    
    # Create models directory
    models_dir = Path("~/mri_app/models").expanduser()
    models_dir.mkdir(exist_ok=True)
    
    # Test MONAI server connection
    try:
        response = requests.get("http://localhost:8000/info/", timeout=5)
        if response.status_code != 200:
            print("❌ MONAI server not accessible. Make sure it's running on port 8000")
            return False
        print("✅ MONAI server is accessible")
    except Exception as e:
        print(f"❌ Cannot connect to MONAI server: {e}")
        return False
    
    success_count = 0
    
    for model_name, config in models.items():
        print(f"\n🔍 Processing {model_name}...")
        
        model_path = models_dir / f"{model_name}.pt"
        
        # Create dummy model since real ones aren't available
        if config["type"] == "dummy":
            if create_dummy_model(model_name, model_path):
                # Add to MONAI server
                if add_model_to_monai(model_name, model_path):
                    success_count += 1
                    print(f"📋 Model Info:")
                    print(f"   - Name: {model_name}")
                    print(f"   - Description: {config['description']}")
                    print(f"   - Body Part: {config['body_part']}")
                    print(f"   - Modality: {config['modality']}")
                    print(f"   - Type: Dummy model (for testing)")
        
        print("-" * 40)
    
    print(f"\n🎉 Summary: {success_count}/{len(models)} models added successfully!")
    
    if success_count > 0:
        print("\n📝 Next Steps:")
        print("1. Restart your MONAI Label server")
        print("2. Test with the dummy models")
        print("3. Download real pre-trained models when available")
        print("4. Test with breast and lung cancer datasets from TCIA")
        
        print("\n🔍 **Available Cancer Datasets from TCIA:**")
        print("Based on your search results, you have access to:")
        print("- LIDC-IDRI: Lung nodule dataset")
        print("- TCGA-LUAD: Lung adenocarcinoma dataset") 
        print("- TCGA-LUSC: Lung squamous cell carcinoma dataset")
        print("- TCGA-BRCA: Breast cancer dataset")
        
        print("\n💡 **Testing Strategy:**")
        print("1. Use the dummy models to test the pipeline")
        print("2. Download real cancer datasets from TCIA")
        print("3. Test with actual CT scans of lungs and breasts")
        print("4. Replace dummy models with real pre-trained models")
    
    return success_count > 0

if __name__ == "__main__":
    main()
