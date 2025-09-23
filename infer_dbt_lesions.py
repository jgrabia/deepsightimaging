#!/usr/bin/env python3
"""
DBT Lesion Detection Inference Script
Uses trained model to detect lesions in new DBT images
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
import pydicom
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import the model architecture from training script
import sys
sys.path.append(str(Path(__file__).parent))
from train_dbt_lesion_detection import DBTLesionNet, Config

class DBTLesionInference:
    """Class for running inference on DBT images"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = DBTLesionNet(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"   Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Training accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        print(f"   Training AUC: {checkpoint.get('auc', 'N/A'):.4f}")
        
        return model
    
    def _get_transform(self):
        """Get preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_dicom_volume(self, dicom_path):
        """Load DICOM volume"""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array
            
            # Get metadata
            metadata = {
                'patient_id': ds.get('PatientID', 'Unknown'),
                'study_date': ds.get('StudyDate', 'Unknown'),
                'modality': ds.get('Modality', 'Unknown'),
                'series_description': ds.get('SeriesDescription', 'Unknown'),
                'manufacturer': ds.get('Manufacturer', 'Unknown'),
                'image_shape': pixel_array.shape
            }
            
            return pixel_array, metadata
            
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            return None, None
    
    def _preprocess_slice(self, slice_data):
        """Preprocess a single slice for inference"""
        # Normalize to 0-255
        if slice_data.max() > 255:
            slice_data = ((slice_data / slice_data.max()) * 255).astype(np.uint8)
        else:
            slice_data = slice_data.astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(slice_data).convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def predict_slice(self, slice_tensor):
        """Predict lesion probability for a single slice"""
        with torch.no_grad():
            slice_tensor = slice_tensor.to(self.device)
            outputs = self.model(slice_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get probability of lesion class (class 1)
            lesion_prob = probabilities[0, 1].item()
            prediction = 1 if lesion_prob > 0.5 else 0
            
            return prediction, lesion_prob
    
    def analyze_volume(self, dicom_path, slice_sampling='all', threshold=0.5):
        """Analyze entire DBT volume"""
        # Load DICOM volume
        volume, metadata = self._load_dicom_volume(dicom_path)
        if volume is None:
            return None
        
        print(f"📊 Analyzing DBT volume: {volume.shape}")
        print(f"   Patient ID: {metadata['patient_id']}")
        print(f"   Modality: {metadata['modality']}")
        
        results = {
            'metadata': metadata,
            'slice_results': [],
            'summary': {}
        }
        
        # Determine which slices to analyze
        if len(volume.shape) == 3:
            num_slices = volume.shape[0]
            
            if slice_sampling == 'all':
                slice_indices = list(range(num_slices))
            elif slice_sampling == 'middle':
                slice_indices = [num_slices // 2]
            elif slice_sampling == 'sample':
                # Sample every 5th slice
                slice_indices = list(range(0, num_slices, 5))
            else:
                slice_indices = [num_slices // 2]  # Default to middle
                
        else:
            # 2D image
            slice_indices = [0]
            volume = volume[np.newaxis, ...]  # Add slice dimension
        
        print(f"   Analyzing {len(slice_indices)} slices...")
        
        # Analyze each slice
        positive_slices = 0
        max_probability = 0.0
        suspicious_slices = []
        
        for i, slice_idx in enumerate(slice_indices):
            slice_data = volume[slice_idx, :, :]
            
            # Preprocess slice
            slice_tensor = self._preprocess_slice(slice_data)
            
            # Make prediction
            prediction, probability = self.predict_slice(slice_tensor)
            
            # Store results
            slice_result = {
                'slice_index': slice_idx,
                'prediction': prediction,
                'probability': probability,
                'suspicious': probability > threshold
            }
            results['slice_results'].append(slice_result)
            
            # Update summary statistics
            if prediction == 1:
                positive_slices += 1
            
            if probability > max_probability:
                max_probability = probability
            
            if probability > threshold:
                suspicious_slices.append(slice_idx)
            
            # Progress
            if i % 10 == 0:
                print(f"   Processed {i+1}/{len(slice_indices)} slices...")
        
        # Calculate summary
        results['summary'] = {
            'total_slices_analyzed': len(slice_indices),
            'positive_slices': positive_slices,
            'suspicious_slices': len(suspicious_slices),
            'max_probability': max_probability,
            'mean_probability': np.mean([r['probability'] for r in results['slice_results']]),
            'overall_prediction': 'LESION DETECTED' if max_probability > threshold else 'NO LESION DETECTED',
            'confidence': 'HIGH' if max_probability > 0.8 else 'MEDIUM' if max_probability > 0.6 else 'LOW',
            'suspicious_slice_indices': suspicious_slices
        }
        
        return results
    
    def save_results(self, results, output_path):
        """Save results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON
        clean_results = json.loads(json.dumps(results, default=convert_numpy_types))
        
        # Add timestamp
        clean_results['analysis_timestamp'] = datetime.now().isoformat()
        clean_results['model_info'] = {
            'architecture': 'DBTLesionNet',
            'input_size': Config.IMAGE_SIZE
        }
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def create_visualization(self, dicom_path, results, output_dir):
        """Create visualization of results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load volume for visualization
        volume, _ = self._load_dicom_volume(dicom_path)
        if volume is None:
            return
        
        # Create probability plot
        slice_indices = [r['slice_index'] for r in results['slice_results']]
        probabilities = [r['probability'] for r in results['slice_results']]
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Probability by slice
        plt.subplot(1, 2, 1)
        plt.plot(slice_indices, probabilities, 'b-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
        plt.xlabel('Slice Index')
        plt.ylabel('Lesion Probability')
        plt.title('Lesion Probability by Slice')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Summary statistics
        plt.subplot(1, 2, 2)
        summary = results['summary']
        categories = ['Total Slices', 'Suspicious Slices', 'Positive Slices']
        values = [summary['total_slices_analyzed'], summary['suspicious_slices'], summary['positive_slices']]
        colors = ['lightblue', 'orange', 'red']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Analysis Summary')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'lesion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save most suspicious slices
        if results['summary']['suspicious_slice_indices']:
            suspicious_indices = results['summary']['suspicious_slice_indices'][:4]  # Top 4
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, slice_idx in enumerate(suspicious_indices):
                if i >= 4:
                    break
                
                slice_data = volume[slice_idx, :, :]
                slice_result = next(r for r in results['slice_results'] if r['slice_index'] == slice_idx)
                
                axes[i].imshow(slice_data, cmap='gray')
                axes[i].set_title(f'Slice {slice_idx}\nProb: {slice_result["probability"]:.3f}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(suspicious_indices), 4):
                axes[i].axis('off')
            
            plt.suptitle('Most Suspicious Slices')
            plt.tight_layout()
            plt.savefig(output_dir / 'suspicious_slices.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='DBT Lesion Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input DICOM file')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for lesion detection')
    parser.add_argument('--slices', type=str, default='all',
                       choices=['all', 'middle', 'sample'],
                       help='Which slices to analyze')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    print("🏥 DBT Lesion Detection Inference")
    print("=" * 50)
    
    # Check inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Create inference object
    try:
        inferencer = DBTLesionInference(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run analysis
    print(f"\n🔍 Analyzing: {args.input}")
    results = inferencer.analyze_volume(
        args.input, 
        slice_sampling=args.slices,
        threshold=args.threshold
    )
    
    if results is None:
        print("❌ Analysis failed")
        return
    
    # Print results
    summary = results['summary']
    print(f"\n📋 ANALYSIS RESULTS")
    print("=" * 30)
    print(f"Overall Prediction: {summary['overall_prediction']}")
    print(f"Confidence: {summary['confidence']}")
    print(f"Max Probability: {summary['max_probability']:.3f}")
    print(f"Mean Probability: {summary['mean_probability']:.3f}")
    print(f"Suspicious Slices: {summary['suspicious_slices']}/{summary['total_slices_analyzed']}")
    
    if summary['suspicious_slice_indices']:
        print(f"Suspicious Slice Indices: {summary['suspicious_slice_indices']}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = output_dir / f"results_{Path(args.input).stem}.json"
    inferencer.save_results(results, json_path)
    
    # Create visualizations
    if args.visualize:
        inferencer.create_visualization(args.input, results, output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()


