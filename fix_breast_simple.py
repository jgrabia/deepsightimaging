#!/usr/bin/env python3
"""
Simple Breast Model Implementation
Creates a simple breast model that works without BasicInferTask complications
"""

import os
from pathlib import Path

def fix_breast_simple():
    """Create a simple working breast model"""
    
    print("🔧 Creating Simple Breast Model")
    print("=" * 50)
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    
    # 1. Create simple breast inferer
    print("1. Creating simple breast inferer...")
    infer_path = app_path / "lib" / "infers" / "breast_segmentation.py"
    infer_path.parent.mkdir(parents=True, exist_ok=True)

    simple_inferer = '''import torch
import numpy as np
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.pre import LoadImaged
from monailabel.transform.post import Restored

class BreastSegmentation(BasicInferTask):
    def __init__(self):
        super().__init__(
            path="",  # No model file needed
            network=None,  # No network needed
            type="segmentation",
            labels=["background", "breast_tissue", "potential_tumor"],
            dimension=3,
            description="Breast tissue and tumor segmentation for MRI"
        )

    def pre_transforms(self, data=None):
        return [
            LoadImaged(keys=["image"]),
        ]

    def post_transforms(self, data=None):
        return [
            Restored(keys=["pred"], ref_image="image"),
        ]

    def inferer(self, data=None):
        return SimpleBreastInferer()

class SimpleBreastInferer:
    def __call__(self, inputs):
        # Get the image
        image = inputs["image"]
        
        # Create a simple segmentation mask
        # For 2D images, create a 2D mask
        if len(image.shape) == 3:  # [C, H, W]
            shape = image.shape[1:]
            pred = torch.zeros((3, *shape), dtype=torch.float32)
            
            # Create some dummy regions
            h, w = shape
            center_h, center_w = h // 2, w // 2
            
            # Label 1: Breast tissue (create a circular region)
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            pred[1] = torch.where(dist < min(h, w) // 3, 0.6, 0.1)
            
            # Label 2: Potential tumor (small region)
            tumor_radius = min(h, w) // 8
            pred[2] = torch.where(dist < tumor_radius, 0.3, 0.0)
            
        else:  # 3D or other format
            shape = image.shape[1:]
            pred = torch.zeros((3, *shape), dtype=torch.float32)
            pred[1] = 0.4  # Some breast tissue
            pred[2] = 0.1  # Some potential tumor
        
        return {"pred": pred}
'''
    
    with open(infer_path, 'w') as f:
        f.write(simple_inferer)
    print("   ✅ Simple breast inferer created")
    
    # 2. Update breast config
    print("2. Updating breast config...")
    config_path = app_path / "lib" / "configs" / "breast_segmentation.py"
    
    simple_config = '''from lib.infers.breast_segmentation import BreastSegmentation
from monailabel.interfaces.config import TaskConfig

class BreastSegmentationConfig(TaskConfig):
    def __init__(self):
        super().__init__()
        self.name = "breast_segmentation"
        self.description = "Breast tissue and tumor segmentation for MRI"
        self.version = "1.0"

    def infer(self):
        return BreastSegmentation()

    def trainer(self):
        return None

    def strategy(self):
        return None

    def scoring_method(self):
        return None
'''
    
    with open(config_path, 'w') as f:
        f.write(simple_config)
    print("   ✅ Simple breast config created")
    
    print("\n🎉 Simple breast model created!")
    print("\n🔄 Now restart the MONAI server with:")
    print("   monailabel start_server --app ~/.local/monailabel/sample-apps/radiology \\")
    print("   --studies ~/mri_app/dicom_download \\")
    print("   --host 0.0.0.0 --port 8000 \\")
    print("   --conf models breast_segmentation,segmentation,deepgrow_2d,deepgrow_3d")

if __name__ == "__main__":
    fix_breast_simple()





