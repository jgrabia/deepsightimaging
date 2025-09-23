#!/usr/bin/env python3
"""
Breast Cancer Classification Inferer
Enterprise-ready classification inference for breast imaging
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pydicom
from pathlib import Path
import json
from datetime import datetime

class BreastClassificationModel(nn.Module):
    """ResNet-based classification model for breast cancer"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Use ResNet50 as backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify first layer for single channel input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class BreastClassificationInferer:
    """Enterprise-ready breast cancer classification inferer"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Normal', 'Benign', 'Malignant']
        self.risk_levels = ['Low', 'Medium', 'High']
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None
            
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """Load trained classification model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            self.model = BreastClassificationModel(num_classes=3, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load config if available
            if 'config' in checkpoint:
                self.config = checkpoint['config']
                if 'class_names' in self.config:
                    self.class_names = self.config['class_names']
            
            print(f"✅ Classification model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_dicom(self, dicom_path):
        """Preprocess DICOM image for classification"""
        try:
            # Load DICOM
            ds = pydicom.dcmread(dicom_path)
            image = ds.pixel_array.astype(np.float32)
            
            # Normalize
            image = (image - image.min()) / (image.max() - image.min())
            
            # Convert to 3-channel
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # Apply transforms
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing DICOM: {e}")
            return None
    
    def classify_image(self, dicom_path):
        """Classify a breast image"""
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'classification': 'Unknown',
                'confidence': 0.0,
                'risk_level': 'Unknown'
            }
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_dicom(dicom_path)
            if image_tensor is None:
                return {
                    'error': 'Failed to preprocess image',
                    'classification': 'Unknown',
                    'confidence': 0.0,
                    'risk_level': 'Unknown'
                }
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                # Get all class probabilities
                class_probabilities = probabilities[0].cpu().numpy()
                
                # Determine risk level
                if predicted_class == 0:  # Normal
                    risk_level = 'Low'
                elif predicted_class == 1:  # Benign
                    risk_level = 'Medium'
                else:  # Malignant
                    risk_level = 'High'
                
                # Generate clinical recommendation
                recommendation = self.generate_recommendation(predicted_class, confidence_score)
                
                return {
                    'classification': self.class_names[predicted_class],
                    'confidence': confidence_score,
                    'risk_level': risk_level,
                    'class_probabilities': {
                        self.class_names[i]: float(class_probabilities[i]) 
                        for i in range(len(self.class_names))
                    },
                    'recommendation': recommendation,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0',
                    'enterprise_ready': True
                }
                
        except Exception as e:
            return {
                'error': f'Classification failed: {str(e)}',
                'classification': 'Unknown',
                'confidence': 0.0,
                'risk_level': 'Unknown'
            }
    
    def generate_recommendation(self, predicted_class, confidence):
        """Generate clinical recommendation based on classification"""
        if predicted_class == 0:  # Normal
            if confidence > 0.8:
                return "No suspicious findings detected. Routine follow-up recommended."
            else:
                return "No obvious abnormalities detected. Consider additional imaging if clinically indicated."
        
        elif predicted_class == 1:  # Benign
            if confidence > 0.7:
                return "Benign findings detected. Follow-up in 6-12 months recommended."
            else:
                return "Possible benign findings. Consider additional imaging or follow-up in 3-6 months."
        
        else:  # Malignant
            if confidence > 0.8:
                return "Suspicious findings detected. Immediate further evaluation recommended."
            else:
                return "Concerning findings detected. Additional imaging and clinical correlation recommended."
    
    def batch_classify(self, dicom_paths):
        """Classify multiple images in batch"""
        results = []
        for dicom_path in dicom_paths:
            result = self.classify_image(dicom_path)
            result['file_path'] = str(dicom_path)
            results.append(result)
        return results
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        return {
            'model_type': 'ResNet50-based Classification',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'device': str(self.device),
            'enterprise_ready': True,
            'version': '1.0'
        }

def main():
    """Test the classification inferer"""
    # Test with a sample DICOM file
    inferer = BreastClassificationInferer()
    
    # Check if model exists
    model_path = Path('trained_classification_models/best_classification_model.pth')
    if model_path.exists():
        inferer.load_model(model_path)
        
        # Test classification
        test_file = Path('consolidated_training_data') / 'test.dcm'
        if test_file.exists():
            result = inferer.classify_image(test_file)
            print("Classification Result:")
            print(json.dumps(result, indent=2))
        else:
            print("No test file found")
    else:
        print("No trained model found. Please train the model first.")

if __name__ == "__main__":
    main()


