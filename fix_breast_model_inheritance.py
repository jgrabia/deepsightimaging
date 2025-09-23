#!/usr/bin/env python3
"""
Fix Breast Model Inheritance
Makes BreastSegmentationConfig properly inherit from TaskConfig
"""

import os
from pathlib import Path

def fix_breast_model_inheritance():
    """Fix the breast model to properly inherit from TaskConfig"""
    
    print("🔧 Fixing Breast Model Inheritance")
    print("=" * 50)
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    breast_config_path = app_path / "lib" / "configs" / "breast_segmentation.py"
    
    if not breast_config_path.exists():
        print("❌ Breast config file not found!")
        return
    
    # Read current breast config
    with open(breast_config_path, 'r') as f:
        content = f.read()
    
    print("📄 Current breast config content:")
    print(content)
    print("\n" + "="*50)
    
    # Create the fixed version with proper inheritance
    fixed_content = '''from lib.infers.breast_segmentation import BreastSegmentation
from monailabel.interfaces.tasks.config import TaskConfig

class BreastSegmentationConfig(TaskConfig):
    def __init__(self):
        super().__init__()
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
    
    # Write the fixed content
    with open(breast_config_path, 'w') as f:
        f.write(fixed_content)
    
    print("📄 Fixed breast config content:")
    print(fixed_content)
    
    print("\n🎉 Breast model inheritance fixed!")
    print("\n🔄 Now restart the MONAI server with:")
    print("   monailabel start_server --app ~/.local/monailabel/sample-apps/radiology \\")
    print("   --studies ~/mri_app/dicom_download \\")
    print("   --host 0.0.0.0 --port 8000 \\")
    print("   --conf models breast_segmentation,segmentation,deepgrow_2d,deepgrow_3d")

if __name__ == "__main__":
    fix_breast_model_inheritance()





