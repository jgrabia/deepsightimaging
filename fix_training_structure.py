#!/usr/bin/env python3
"""
Fix Training Structure Script
Checks what files were extracted and creates the proper training directory structure
"""

import os
import shutil
from pathlib import Path

def check_extracted_files():
    """Check what files were actually extracted"""
    
    print("🔍 Checking extracted files...")
    
    # Check training_data directory
    training_data_dir = "/home/ubuntu/mri_app/training_data"
    if os.path.exists(training_data_dir):
        print(f"📁 Found training_data directory: {training_data_dir}")
        
        for item in os.listdir(training_data_dir):
            item_path = os.path.join(training_data_dir, item)
            if os.path.isdir(item_path):
                print(f"   📂 Dataset: {item}")
                
                # Check for ZIP files
                zip_files = [f for f in os.listdir(item_path) if f.endswith('.zip')]
                print(f"      📦 ZIP files: {len(zip_files)}")
                
                # Check for extracted directories
                extracted_dirs = [f for f in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, f)) and not f.endswith('.zip')]
                print(f"      📂 Extracted directories: {len(extracted_dirs)}")
                
                if extracted_dirs:
                    for ext_dir in extracted_dirs[:3]:  # Show first 3
                        ext_path = os.path.join(item_path, ext_dir)
                        dicom_files = [f for f in os.listdir(ext_path) if f.endswith('.dcm')]
                        print(f"         📄 {ext_dir}: {len(dicom_files)} DICOM files")
    
    # Check for extracted_training_data directory
    extracted_dir = "/home/ubuntu/mri_app/extracted_training_data"
    if os.path.exists(extracted_dir):
        print(f"📁 Found extracted_training_data directory: {extracted_dir}")
        
        for item in os.listdir(extracted_dir):
            item_path = os.path.join(extracted_dir, item)
            if os.path.isdir(item_path):
                print(f"   📂 Dataset: {item}")
                
                # Count DICOM files
                total_dicoms = 0
                for root, dirs, files in os.walk(item_path):
                    dicom_files = [f for f in files if f.endswith('.dcm')]
                    total_dicoms += len(dicom_files)
                
                print(f"      📄 Total DICOM files: {total_dicoms}")

def create_training_structure():
    """Create the proper training directory structure"""
    
    print("\n🔧 Creating training directory structure...")
    
    # Create training directories
    training_dir = "/home/ubuntu/mri_app/dicom_download"
    images_dir = os.path.join(training_dir, "images")
    labels_dir = os.path.join(training_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"📁 Created: {images_dir}")
    print(f"📁 Created: {labels_dir}")
    
    # Find all DICOM files from extracted data
    training_data_dir = "/home/ubuntu/mri_app/training_data"
    total_copied = 0
    
    if os.path.exists(training_data_dir):
        for dataset_name in os.listdir(training_data_dir):
            dataset_path = os.path.join(training_data_dir, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
                
            print(f"📊 Processing {dataset_name}...")
            
            # Look for extracted directories (not ZIP files)
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                
                if os.path.isdir(item_path) and not item.endswith('.zip'):
                    # This is an extracted directory
                    dicom_files = [f for f in os.listdir(item_path) if f.endswith('.dcm')]
                    
                    for dicom_file in dicom_files:
                        src_path = os.path.join(item_path, dicom_file)
                        dst_path = os.path.join(images_dir, f"{item}_{dicom_file}")
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            total_copied += 1
                        except Exception as e:
                            print(f"   ❌ Error copying {dicom_file}: {e}")
    
    print(f"✅ Copied {total_copied} DICOM files to training structure")
    print(f"📁 Training images: {images_dir}")
    print(f"📁 Training labels: {labels_dir}")
    
    return total_copied

def main():
    """Main function"""
    
    print("🏥 Training Structure Fix")
    print("=" * 50)
    
    # Step 1: Check what was extracted
    check_extracted_files()
    
    # Step 2: Create proper training structure
    total_files = create_training_structure()
    
    print(f"\n🎉 Training structure ready!")
    print(f"📊 Total DICOM files: {total_files}")
    print(f"🚀 Ready to run training!")
    print(f"   python advanced_breast_training.py")

if __name__ == "__main__":
    main()




