import pydicom
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

class BalancedBreastClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(BalancedBreastClassifier, self).__init__()
        
        # Use a pre-trained ResNet backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BalancedBreastInferer:
    def __init__(self, model_path=None):
        if model_path is None:
            # Try different possible locations
            possible_paths = [
                'best_balanced_breast_model.pth',
                './best_balanced_breast_model.pth',
                os.path.join(os.path.expanduser('~'), 'mri_app', 'best_balanced_breast_model.pth'),
                os.path.join(os.getcwd(), 'best_balanced_breast_model.pth')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"Model file not found. Tried: {possible_paths}")
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class mapping (same as training)
        self.idx_to_class = {
            0: 'axial_t1',
            1: 'axial_vibrant', 
            2: 'calibration',
            3: 'dynamic_contrast',
            4: 'other',
            5: 'post_contrast_sagittal',
            6: 'sagittal_t1',
            7: 'sagittal_t2',
            8: 'scout',
            9: 'vibrant_sequence'
        }
        
        # Class descriptions for clinical use
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
        
        # Load the trained model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        model = BalancedBreastClassifier(num_classes=10)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        print(f"Loaded balanced breast classification model from {self.model_path}")
        print(f"Model running on device: {self.device}")
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess DICOM image for inference"""
        try:
            # Read DICOM file
            ds = pydicom.dcmread(image_path)
            image = ds.pixel_array.astype(np.float32)
            
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Resize image to standard size (224x224)
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize((224, 224), Image.LANCZOS)
            image = np.array(pil_image).astype(np.float32) / 255.0
            
            # Convert to 3-channel (repeat grayscale)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=0)
            else:
                image = np.transpose(image, (2, 0, 1))
            
            # Convert to tensor and add batch dimension
            image = torch.FloatTensor(image).unsqueeze(0)
            image = image.to(self.device)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def classify_sequence(self, image_path):
        """Classify a single DICOM image"""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            
            predicted_class = self.idx_to_class[predicted_class_idx.item()]
            confidence_score = confidence.item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'description': self.class_descriptions[predicted_class],
                'all_probabilities': {
                    self.idx_to_class[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
    
    def analyze_dicom_series(self, dicom_directory):
        """Analyze all DICOM files in a directory"""
        dicom_dir = Path(dicom_directory)
        dicom_files = list(dicom_dir.rglob('*.dcm'))
        
        if not dicom_files:
            print(f"No DICOM files found in {dicom_directory}")
            return None
        
        print(f"Analyzing {len(dicom_files)} DICOM files...")
        
        results = []
        sequence_counts = {}
        
        for i, file_path in enumerate(dicom_files):
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(dicom_files)}")
            
            result = self.classify_sequence(file_path)
            if result:
                results.append({
                    'file_path': str(file_path),
                    'sequence_type': result['predicted_class'],
                    'confidence': result['confidence'],
                    'description': result['description']
                })
                
                # Count sequences
                seq_type = result['predicted_class']
                sequence_counts[seq_type] = sequence_counts.get(seq_type, 0) + 1
        
        # Generate summary
        summary = {
            'total_files': len(dicom_files),
            'successfully_analyzed': len(results),
            'sequence_distribution': sequence_counts,
            'detailed_results': results
        }
        
        return summary
    
    def print_analysis_summary(self, summary):
        """Print a formatted analysis summary"""
        if not summary:
            print("No analysis results to display")
            return
        
        print("\n" + "="*60)
        print("BALANCED BREAST MRI SEQUENCE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total DICOM files: {summary['total_files']}")
        print(f"Successfully analyzed: {summary['successfully_analyzed']}")
        print(f"Analysis success rate: {summary['successfully_analyzed']/summary['total_files']*100:.1f}%")
        
        print("\nSEQUENCE DISTRIBUTION:")
        print("-" * 40)
        for seq_type, count in sorted(summary['sequence_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / summary['successfully_analyzed'] * 100
            description = self.class_descriptions[seq_type]
            print(f"{seq_type:25} {count:4d} files ({percentage:5.1f}%) - {description}")
        
        print("\nCLINICAL RECOMMENDATIONS:")
        print("-" * 40)
        
        # Check for key sequences
        vibrant_count = summary['sequence_distribution'].get('vibrant_sequence', 0) + summary['sequence_distribution'].get('axial_vibrant', 0)
        dynamic_count = summary['sequence_distribution'].get('dynamic_contrast', 0)
        post_contrast_count = summary['sequence_distribution'].get('post_contrast_sagittal', 0)
        
        if vibrant_count > 0:
            print(f"✅ VIBRANT sequences detected ({vibrant_count} files) - Good for dynamic contrast analysis")
        if dynamic_count > 0:
            print(f"✅ Dynamic contrast sequences detected ({dynamic_count} files) - Suitable for perfusion analysis")
        if post_contrast_count > 0:
            print(f"✅ Post-contrast sequences detected ({post_contrast_count} files) - Enhanced imaging available")
        
        scout_count = summary['sequence_distribution'].get('scout', 0)
        if scout_count > 0:
            print(f"ℹ️  Scout sequences detected ({scout_count} files) - Positioning references")
        
        calibration_count = summary['sequence_distribution'].get('calibration', 0)
        if calibration_count > 0:
            print(f"ℹ️  Calibration sequences detected ({calibration_count} files) - Coil sensitivity mapping")
        
        print("\n" + "="*60)

def main():
    """Main function for testing the inferer"""
    inferer = BalancedBreastInferer()
    
    # Test with a single file if available
    test_files = [
        'consolidated_training_data',
        'dicom_download',
        'dicom_download/Breast',
        'dicom_download/Breast_2'
    ]
    
    for test_dir in test_files:
        if Path(test_dir).exists():
            print(f"\nTesting with directory: {test_dir}")
            summary = inferer.analyze_dicom_series(test_dir)
            if summary:
                inferer.print_analysis_summary(summary)
            break
    else:
        print("No test directories found. Please provide a path to DICOM files.")

if __name__ == "__main__":
    main()
