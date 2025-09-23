#!/usr/bin/env python3
"""
Extract Training Data Script
Extracts all ZIP files from downloaded datasets and organizes them for training
"""

import os
import zipfile
import shutil
from pathlib import Path
import pydicom
import numpy as np
from datetime import datetime

def extract_training_data(base_dir="/home/ubuntu/training_data", output_dir="extracted_training_data"):
    """Extract all ZIP files and organize for training"""
    
    print("🔧 Extracting training data...")
    print(f"📁 Source directory: {base_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_extracted = 0
    total_errors = 0
    
    # Process each dataset directory
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        
        if not os.path.isdir(dataset_path):
            continue
            
        print(f"\n📊 Processing dataset: {dataset_name}")
        
        # Create dataset output directory
        dataset_output = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)
        
        # Find all ZIP files
        zip_files = [f for f in os.listdir(dataset_path) if f.endswith('.zip')]
        print(f"   Found {len(zip_files)} ZIP files")
        
        for i, zip_file in enumerate(zip_files):
            zip_path = os.path.join(dataset_path, zip_file)
            series_uid = zip_file.replace('.zip', '')
            
            print(f"   Extracting {i+1}/{len(zip_files)}: {series_uid}")
            
            try:
                # Create series directory
                series_dir = os.path.join(dataset_output, series_uid)
                os.makedirs(series_dir, exist_ok=True)
                
                # Extract ZIP file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(series_dir)
                
                # Count DICOM files
                dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
                print(f"     ✅ Extracted {len(dicom_files)} DICOM files")
                
                total_extracted += len(dicom_files)
                
            except Exception as e:
                print(f"     ❌ Error extracting {zip_file}: {e}")
                total_errors += 1
    
    print(f"\n🎉 Extraction complete!")
    print(f"   Total DICOM files extracted: {total_extracted}")
    print(f"   Total errors: {total_errors}")
    print(f"   Output directory: {output_dir}")
    
    return output_dir

def create_training_structure(extracted_dir):
    """Create the training directory structure expected by the training script"""
    
    print("\n📁 Creating training directory structure...")
    
    # Create training directories
    training_dir = "dicom_download"
    images_dir = os.path.join(training_dir, "images")
    labels_dir = os.path.join(training_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Copy DICOM files to training structure
    total_copied = 0
    
    for dataset_name in os.listdir(extracted_dir):
        dataset_path = os.path.join(extracted_dir, dataset_name)
        
        if not os.path.isdir(dataset_path):
            continue
            
        print(f"📊 Processing {dataset_name} for training...")
        
        for series_uid in os.listdir(dataset_path):
            series_path = os.path.join(dataset_path, series_uid)
            
            if not os.path.isdir(series_path):
                continue
                
            # Find DICOM files
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            
            for dicom_file in dicom_files:
                src_path = os.path.join(series_path, dicom_file)
                dst_path = os.path.join(images_dir, f"{series_uid}_{dicom_file}")
                
                try:
                    shutil.copy2(src_path, dst_path)
                    total_copied += 1
                except Exception as e:
                    print(f"   ❌ Error copying {dicom_file}: {e}")
    
    print(f"✅ Training structure created!")
    print(f"   Total DICOM files copied: {total_copied}")
    print(f"   Training directory: {training_dir}")
    
    return training_dir

def generate_synthetic_labels(training_dir):
    """Generate synthetic labels for training (temporary until we get real annotations)"""
    
    print("\n🎨 Generating synthetic labels for training...")
    
    images_dir = os.path.join(training_dir, "images")
    labels_dir = os.path.join(training_dir, "labels")
    
    # Find all DICOM files
    dicom_files = [f for f in os.listdir(images_dir) if f.endswith('.dcm')]
    
    print(f"📊 Found {len(dicom_files)} DICOM files")
    
    for i, dicom_file in enumerate(dicom_files):
        if i % 10 == 0:
            print(f"   Processing {i+1}/{len(dicom_files)}")
        
        try:
            # Load DICOM to get dimensions
            dicom_path = os.path.join(images_dir, dicom_file)
            ds = pydicom.dcmread(dicom_path)
            
            # Get image dimensions
            if hasattr(ds, 'pixel_array'):
                height, width = ds.pixel_array.shape
            else:
                # Default dimensions if pixel array not available
                height, width = 512, 512
            
            # Create synthetic label (random regions)
            label = np.zeros((height, width), dtype=np.uint8)
            
            # Add some random "lesion" regions
            num_regions = np.random.randint(0, 3)  # 0-2 regions
            for _ in range(num_regions):
                center_x = np.random.randint(50, width - 50)
                center_y = np.random.randint(50, height - 50)
                radius = np.random.randint(10, 30)
                
                y, x = np.ogrid[:height, :width]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                region_mask = distance < radius
                
                # Assign different labels (1=background, 2=breast_tissue, 3=lesions)
                label[region_mask] = np.random.choice([2, 3], region_mask.sum())
            
            # Save label as numpy array
            label_file = dicom_file.replace('.dcm', '_label.npy')
            label_path = os.path.join(labels_dir, label_file)
            np.save(label_path, label)
            
        except Exception as e:
            print(f"   ❌ Error processing {dicom_file}: {e}")
    
    print(f"✅ Synthetic labels generated!")
    print(f"   Labels saved to: {labels_dir}")

def main():
    """Main function"""
    
    print("🏥 Training Data Preparation")
    print("=" * 50)
    
    try:
        # Step 1: Extract ZIP files
        print("Starting extraction...")
        extracted_dir = extract_training_data()
        print(f"Extraction returned: {extracted_dir}")
        
        # Step 2: Create training structure
        print("Creating training structure...")
        training_dir = create_training_structure(extracted_dir)
        print(f"Training structure returned: {training_dir}")
        
        # Step 3: Generate synthetic labels
        print("Generating synthetic labels...")
        generate_synthetic_labels(training_dir)
        
        print("\n🎉 Training data preparation complete!")
        print(f"📁 Training directory: {training_dir}")
        print(f"📁 Images: {training_dir}/images")
        print(f"📁 Labels: {training_dir}/labels")
        print("\n🚀 Ready to run training!")
        print("   python advanced_breast_training.py")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
