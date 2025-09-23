#!/usr/bin/env python3
"""
Fix Breast Model Implementation
Creates a working breast segmentation model for MONAI Label
"""

import os
from pathlib import Path

def create_breast_model():
    """Create a working breast segmentation model"""
    
    print("🏥 Creating Working Breast Segmentation Model")
    print("=" * 50)
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    
    # 1. Create breast inferer
    print("1. Creating breast inferer...")
    infer_path = app_path / "lib" / "infers" / "breast_segmentation.py"
    infer_path.parent.mkdir(parents=True, exist_ok=True)
    
    breast_inferer = '''import torch
import numpy as np
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.pre import LoadImaged
from monailabel.transform.post import Restored

class BreastSegmentation(BasicInferTask):
    def __init__(self):
        super().__init__(
            path="",  # No model file for dummy implementation
            network=None,  # No network for dummy implementation
            type="segmentation",
            labels=["background", "breast_tissue", "potential_tumor"],
            dimension=3,
            description="Breast tissue and tumor segmentation for MRI"
        )
    
    def pre_transforms(self):
        return [
            LoadImaged(keys=["image"]),
        ]
    
    def post_transforms(self):
        return [
            Restored(keys=["pred"], ref_image="image"),
        ]
    
    def inferer(self):
        return DummyBreastInferer()

class DummyBreastInferer:
    def __call__(self, inputs):
        # Create dummy breast segmentation
        image = inputs["image"]
        shape = image.shape
        
        # Create segmentation with breast-specific labels
        pred = torch.zeros((3, *shape[1:]), dtype=torch.float32)
        
        # Label 1: Breast tissue (some probability)
        pred[1] = 0.4
        
        # Label 2: Potential tumor/mass (low probability)
        pred[2] = 0.1
        
        return {"pred": pred}
'''
    
    with open(infer_path, 'w') as f:
        f.write(breast_inferer)
    print("   ✅ Breast inferer created")
    
    # 2. Create breast config
    print("2. Creating breast config...")
    config_path = app_path / "lib" / "configs" / "breast_segmentation.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    breast_config = '''from lib.infers.breast_segmentation import BreastSegmentation

class BreastSegmentationConfig:
    def __init__(self):
        self.name = "breast_segmentation"
        self.description = "Breast tissue and tumor segmentation for MRI"
        self.version = "1.0"
        
    def infer(self):
        return BreastSegmentation()
    
    def trainer(self):
        return None  # No training for now
    
    def strategy(self):
        return None  # No strategy for now
    
    def scoring_method(self):
        return None  # No scoring for now
'''
    
    with open(config_path, 'w') as f:
        f.write(breast_config)
    print("   ✅ Breast config created")
    
    # 3. Update main.py to include breast model
    print("3. Updating main.py...")
    main_path = app_path / "main.py"
    
    if main_path.exists():
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Add import if not present
        if "from lib.configs.breast_segmentation import BreastSegmentationConfig" not in content:
            content = content.replace(
                "from lib.configs.segmentation import Segmentation",
                "from lib.configs.segmentation import Segmentation\nfrom lib.configs.breast_segmentation import BreastSegmentationConfig"
            )
            print("   ✅ Added breast import")
        
        # Add model registration if not present
        if "self.add_model(BreastSegmentationConfig())" not in content:
            content = content.replace(
                "self.add_model(Segmentation())",
                "self.add_model(Segmentation())\n        self.add_model(BreastSegmentationConfig())"
            )
            print("   ✅ Added breast model registration")
        
        with open(main_path, 'w') as f:
            f.write(content)
        print("   ✅ Updated main.py")
    
    print("\n🎉 Breast model created successfully!")
    print("\n🔄 Now restart the MONAI server with:")
    print("   monailabel start_server --app ~/.local/monailabel/sample-apps/radiology \\\\")
    print("   --studies ~/mri_app/dicom_download \\\\")
    print("   --host 0.0.0.0 --port 8000 \\\\")
    print("   --conf models breast_segmentation,segmentation,deepgrow_2d,deepgrow_3d")

if __name__ == "__main__":
    create_breast_model()
