#!/usr/bin/env python3

import os
import sys
import json
import shutil
from pathlib import Path
import subprocess

def create_monai_inferer():
    """Create MONAI Label inferer for balanced breast classification"""
    
    # MONAI Label inferer template
    inferer_code = '''import logging
import torch
import numpy as np
from monai.inferers import Inferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    ToTensord,
)
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
import pydicom
from PIL import Image
from pathlib import Path

# Import our balanced model
from balanced_breast_inferer import BalancedBreastInferer

logger = logging.getLogger(__name__)

class BalancedBreastClassificationInferer(Inferer):
    """
    MONAI Label Inferer for Balanced Breast MRI Sequence Classification
    Uses our 99% accurate 10-class classification model
    """
    
    def __init__(self, model_path="best_balanced_breast_model.pth"):
        super().__init__()
        self.model_path = model_path
        self.inferer = BalancedBreastInferer(model_path)
        
        # Class descriptions for MONAI Label
        self.class_descriptions = {
            'axial_t1': 'Axial T1-weighted sequence - anatomical imaging',
            'axial_vibrant': 'Axial VIBRANT sequence - dynamic contrast imaging', 
            'calibration': 'ASSET calibration sequence - coil sensitivity mapping',
            'dynamic_contrast': 'Dynamic contrast sequence - perfusion imaging',
            'other': 'Other/unknown sequence type',
            'post_contrast_sagittal': 'Post-contrast sagittal sequence - enhanced imaging',
            'sagittal_t1': 'Sagittal T1-weighted sequence - anatomical imaging',
            'sagittal_t2': 'Sagittal T2-weighted sequence - fluid-sensitive imaging',
            'scout': 'Scout/localizer sequence - positioning reference',
            'vibrant_sequence': 'VIBRANT sequence - dynamic contrast imaging'
        }
        
        logger.info("Balanced Breast Classification Inferer initialized")
        logger.info(f"Model path: {model_path}")
        logger.info("Available classes: " + ", ".join(self.class_descriptions.keys()))
    
    def __call__(self, inputs, *args, **kwargs):
        """
        Perform inference on input data
        
        Args:
            inputs: Input data (can be file path, DICOM data, or image array)
            
        Returns:
            Dictionary containing classification results
        """
        try:
            # Handle different input types
            if isinstance(inputs, str):
                # File path
                result = self.inferer.classify_sequence(inputs)
            elif isinstance(inputs, dict) and 'image' in inputs:
                # MONAI data dict
                image_path = inputs.get('image_path', '')
                if image_path:
                    result = self.inferer.classify_sequence(image_path)
                else:
                    # Handle image array
                    result = self._classify_array(inputs['image'])
            else:
                # Assume it's an image array
                result = self._classify_array(inputs)
            
            if result is None:
                return {
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'description': 'Classification failed',
                    'error': 'Unable to process input'
                }
            
            # Format result for MONAI Label
            return {
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'description': result['description'],
                'all_probabilities': result['all_probabilities'],
                'model_info': {
                    'model_type': 'Balanced Breast Classification',
                    'accuracy': '99%',
                    'classes': 10,
                    'model_path': self.model_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return {
                'predicted_class': 'error',
                'confidence': 0.0,
                'description': f'Inference error: {str(e)}',
                'error': str(e)
            }
    
    def _classify_array(self, image_array):
        """Classify image array directly"""
        try:
            # Convert to PIL Image and save temporarily
            if len(image_array.shape) == 3:
                # Remove channel dimension if present
                image_array = image_array.squeeze()
            
            # Normalize to 0-255
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
            
            # Create PIL Image
            pil_image = Image.fromarray(image_array)
            pil_image = pil_image.resize((224, 224), Image.LANCZOS)
            
            # Convert back to numpy and create tensor
            image = np.array(pil_image).astype(np.float32) / 255.0
            image = np.stack([image] * 3, axis=0)  # 3-channel
            image_tensor = torch.FloatTensor(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.inferer.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.inferer.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, 1)
                
                predicted_class = self.inferer.idx_to_class[predicted_class_idx.item()]
                confidence_score = confidence.item()
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'description': self.class_descriptions[predicted_class],
                    'all_probabilities': {
                        self.inferer.idx_to_class[i]: prob.item() 
                        for i, prob in enumerate(probabilities[0])
                    }
                }
                
        except Exception as e:
            logger.error(f"Error classifying array: {e}")
            return None
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'model_name': 'Balanced Breast MRI Classification',
            'model_type': 'CNN (ResNet-like)',
            'accuracy': '99%',
            'classes': 10,
            'description': '10-class MRI sequence classification with balanced dataset',
            'classes_list': list(self.class_descriptions.keys()),
            'class_descriptions': self.class_descriptions
        }
'''
    
    return inferer_code

def create_monai_config():
    """Create MONAI Label configuration for balanced breast classification"""
    
    config = {
        "name": "Balanced Breast MRI Classification",
        "description": "Advanced 10-class MRI sequence classification using balanced dataset",
        "version": "1.0.0",
        "labels": {
            "axial_t1": {
                "name": "Axial T1-weighted",
                "description": "Axial T1-weighted sequence - anatomical imaging",
                "color": [255, 0, 0]
            },
            "axial_vibrant": {
                "name": "Axial VIBRANT", 
                "description": "Axial VIBRANT sequence - dynamic contrast imaging",
                "color": [0, 255, 0]
            },
            "calibration": {
                "name": "Calibration",
                "description": "ASSET calibration sequence - coil sensitivity mapping", 
                "color": [0, 0, 255]
            },
            "dynamic_contrast": {
                "name": "Dynamic Contrast",
                "description": "Dynamic contrast sequence - perfusion imaging",
                "color": [255, 255, 0]
            },
            "other": {
                "name": "Other",
                "description": "Other/unknown sequence type",
                "color": [128, 128, 128]
            },
            "post_contrast_sagittal": {
                "name": "Post-contrast Sagittal",
                "description": "Post-contrast sagittal sequence - enhanced imaging",
                "color": [255, 0, 255]
            },
            "sagittal_t1": {
                "name": "Sagittal T1-weighted",
                "description": "Sagittal T1-weighted sequence - anatomical imaging",
                "color": [0, 255, 255]
            },
            "sagittal_t2": {
                "name": "Sagittal T2-weighted", 
                "description": "Sagittal T2-weighted sequence - fluid-sensitive imaging",
                "color": [255, 128, 0]
            },
            "scout": {
                "name": "Scout",
                "description": "Scout/localizer sequence - positioning reference",
                "color": [128, 255, 0]
            },
            "vibrant_sequence": {
                "name": "VIBRANT Sequence",
                "description": "VIBRANT sequence - dynamic contrast imaging",
                "color": [255, 0, 128]
            }
        },
        "infer": {
            "path": "monai_balanced_inferer.py",
            "class": "BalancedBreastClassificationInferer"
        },
        "train": {
            "path": "balanced_breast_training.py", 
            "class": "BalancedBreastTrainer"
        }
    }
    
    return config

def setup_monai_integration():
    """Set up MONAI Label integration"""
    
    print("Setting up MONAI Label integration for Balanced Breast Classification...")
    
    # Create inferer file
    inferer_code = create_monai_inferer()
    with open('monai_balanced_inferer.py', 'w') as f:
        f.write(inferer_code)
    print("✅ Created monai_balanced_inferer.py")
    
    # Create config file
    config = create_monai_config()
    with open('monai_balanced_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Created monai_balanced_config.json")
    
    # Create startup script
    startup_script = '''#!/bin/bash
# MONAI Label startup script for Balanced Breast Classification

echo "Starting MONAI Label with Balanced Breast Classification..."

# Check if model exists
if [ ! -f "best_balanced_breast_model.pth" ]; then
    echo "❌ Model file not found: best_balanced_breast_model.pth"
    echo "Please run balanced_breast_training.py first to train the model"
    exit 1
fi

# Check if required files exist
if [ ! -f "balanced_breast_inferer.py" ]; then
    echo "❌ Inferer file not found: balanced_breast_inferer.py"
    exit 1
fi

if [ ! -f "monai_balanced_inferer.py" ]; then
    echo "❌ MONAI inferer file not found: monai_balanced_inferer.py"
    exit 1
fi

echo "✅ All required files found"
echo "🚀 Starting MONAI Label server..."

# Start MONAI Label server
monailabel start_server --app ./monai_balanced_config.json --studies ./consolidated_training_data --conf models breast_mri_classification
'''
    
    with open('start_monai_balanced.sh', 'w') as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod('start_monai_balanced.sh', 0o755)
    print("✅ Created start_monai_balanced.sh")
    
    print("\n🎯 MONAI Label integration setup complete!")
    print("\nTo start MONAI Label with balanced breast classification:")
    print("  ./start_monai_balanced.sh")
    print("\nOr manually:")
    print("  monailabel start_server --app ./monai_balanced_config.json --studies ./consolidated_training_data")

def test_integration():
    """Test the integration"""
    print("\n🧪 Testing integration...")
    
    try:
        # Test inferer
        from balanced_breast_inferer import BalancedBreastInferer
        inferer = BalancedBreastInferer()
        print("✅ BalancedBreastInferer imported successfully")
        
        # Test model loading
        if Path('best_balanced_breast_model.pth').exists():
            print("✅ Model file found")
        else:
            print("❌ Model file not found")
            return False
            
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    setup_monai_integration()
    test_integration()


