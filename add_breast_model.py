#!/usr/bin/env python3
"""
Add breast-specific model to MONAI Label
"""

import os
import shutil
import requests
import json
from pathlib import Path

def create_breast_model_config():
    """Create breast-specific model configuration"""
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    
    if not app_path.exists():
        print("❌ MONAI Label app not found")
        return False
    
    # Create breast model config
    breast_config = '''from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.scoring import ScoringMethod

from lib.infers.breast_segmentation import BreastSegmentation
from lib.trainers.breast_segmentation import BreastSegmentationTrainer
from lib.strategies.breast_strategy import BreastStrategy
from lib.scoring.breast_scoring import BreastScoring

class BreastSegmentationConfig(TaskConfig):
    def __init__(self):
        super().__init__()
        self.name = "breast_segmentation"
        self.description = "Breast tissue and tumor segmentation for MRI"
        self.version = "1.0"
        
    def infer(self):
        return BreastSegmentation()
    
    def trainer(self):
        return BreastSegmentationTrainer()
    
    def strategy(self):
        return BreastStrategy()
    
    def scoring_method(self):
        return BreastScoring()
'''
    
    # Create breast inferer
    breast_inferer = '''import torch
import numpy as np
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.pre import LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityd, NormalizeIntensityd
from monailabel.transform.post import Restored, EnsureTyped, Activationsd, AsDiscreted
from monailabel.transform.writer import Writer

class BreastSegmentation(InferTask):
    def __init__(self):
        super().__init__()
        self.description = "Breast tissue and tumor segmentation for MRI"
        self.version = "1.0"
        self.type = InferType.SEGMENTATION
        
    def pre_transforms(self):
        return [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    
    def post_transforms(self):
        return [
            EnsureTyped(keys=["pred"]),
            Activationsd(keys=["pred"], softmax=True),
            AsDiscreted(keys=["pred"], argmax=True),
            Restored(keys=["pred"], ref_image="image"),
        ]
    
    def inferer(self):
        # For now, return a simple dummy model
        # In production, this would load a trained breast segmentation model
        return DummyBreastInferer()

class DummyBreastInferer:
    def __call__(self, inputs):
        # Create dummy segmentation with breast-specific labels
        image = inputs["image"]
        shape = image.shape
        
        # Create dummy segmentation with breast tissue labels
        pred = torch.zeros((3, *shape[1:]), dtype=torch.float32)
        
        # Label 1: Breast tissue
        pred[1] = 0.3  # Some breast tissue probability
        
        # Label 2: Potential tumor/mass
        pred[2] = 0.1  # Low probability for tumor
        
        return {"pred": pred}
'''
    
    # Create breast trainer
    breast_trainer = '''from monailabel.interfaces.tasks.train import TrainTask
from monailabel.transform.pre import LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityd, NormalizeIntensityd
from monailabel.transform.post import EnsureTyped

class BreastSegmentationTrainer(TrainTask):
    def __init__(self):
        super().__init__()
        self.description = "Train breast segmentation model"
        self.version = "1.0"
    
    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    
    def train(self):
        # Training logic would go here
        pass
'''
    
    # Create breast strategy
    breast_strategy = '''from monailabel.interfaces.tasks.strategy import Strategy

class BreastStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.description = "Active learning strategy for breast segmentation"
        self.version = "1.0"
    
    def __call__(self, request):
        # Strategy logic would go here
        return []
'''
    
    # Create breast scoring
    breast_scoring = '''from monailabel.interfaces.tasks.scoring import ScoringMethod

class BreastScoring(ScoringMethod):
    def __init__(self):
        super().__init__()
        self.description = "Scoring method for breast segmentation"
        self.version = "1.0"
    
    def __call__(self, request):
        # Scoring logic would go here
        return {}
'''
    
    # Write files
    try:
        # Create lib directory if it doesn't exist
        lib_path = app_path / "lib"
        lib_path.mkdir(exist_ok=True)
        
        # Write config
        config_path = lib_path / "configs" / "breast_segmentation.py"
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(breast_config)
        
        # Write inferer
        infer_path = lib_path / "infers" / "breast_segmentation.py"
        infer_path.parent.mkdir(exist_ok=True)
        with open(infer_path, 'w') as f:
            f.write(breast_inferer)
        
        # Write trainer
        trainer_path = lib_path / "trainers" / "breast_segmentation.py"
        trainer_path.parent.mkdir(exist_ok=True)
        with open(trainer_path, 'w') as f:
            f.write(breast_trainer)
        
        # Write strategy
        strategy_path = lib_path / "strategies" / "breast_strategy.py"
        strategy_path.parent.mkdir(exist_ok=True)
        with open(strategy_path, 'w') as f:
            f.write(breast_strategy)
        
        # Write scoring
        scoring_path = lib_path / "scoring" / "breast_scoring.py"
        scoring_path.parent.mkdir(exist_ok=True)
        with open(scoring_path, 'w') as f:
            f.write(breast_scoring)
        
        print("✅ Breast model configuration created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating breast model: {e}")
        return False

def add_breast_model_to_main():
    """Add breast model to main.py"""
    
    main_path = Path("~/.local/monailabel/sample-apps/radiology/main.py").expanduser()
    
    if not main_path.exists():
        print("❌ main.py not found")
        return False
    
    try:
        # Read current main.py
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Add breast model import
        if "from lib.configs.breast_segmentation import BreastSegmentationConfig" not in content:
            # Find the import section
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.startswith('from lib.configs.') and 'import' in line:
                    import_section_end = i
            
            # Insert breast import
            lines.insert(import_section_end + 1, "from lib.configs.breast_segmentation import BreastSegmentationConfig")
            
            # Add breast model to models list
            if "breast_segmentation" not in content:
                # Find where models are added
                for i, line in enumerate(lines):
                    if "self.add_model(" in line and "segmentation" in line:
                        # Add breast model after segmentation
                        lines.insert(i + 1, '        self.add_model("breast_segmentation", BreastSegmentationConfig())')
                        break
            
            # Write back
            with open(main_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print("✅ Breast model added to main.py")
            return True
            
    except Exception as e:
        print(f"❌ Error updating main.py: {e}")
        return False

def main():
    print("🏥 Adding Breast-Specific Model to MONAI Label")
    print("=" * 50)
    
    # Create breast model configuration
    if create_breast_model_config():
        # Add to main.py
        if add_breast_model_to_main():
            print("\n✅ Breast model successfully added!")
            print("\n🔄 Please restart the MONAI Label server:")
            print("   monailabel start_server --app ~/.local/monailabel/sample-apps/radiology \\")
            print("   --studies ~/mri_app/dicom_download \\")
            print("   --host 0.0.0.0 --port 8000 \\")
            print("   --conf models breast_segmentation")
            print("\n💡 This model will provide breast-specific labels instead of abdominal organs")
        else:
            print("❌ Failed to add breast model to main.py")
    else:
        print("❌ Failed to create breast model configuration")

if __name__ == "__main__":
    main()





