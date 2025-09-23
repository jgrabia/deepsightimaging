#!/usr/bin/env python3
"""
Organize DBT Training Data
Creates a proper structure for training lesion detection models
"""

import os
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path
import shutil
import json
from PIL import Image

# Configuration
BASE_DIR = Path("C:/MRIAPP")
DICOM_SOURCE = BASE_DIR / "dicom_download/images/manifest-1617905855234"
ANNOTATIONS_DIR = BASE_DIR / "annotations"  # Where you put the CSV files
OUTPUT_DIR = BASE_DIR / "dbt_training_data"

def load_annotations():
    """Load all annotation CSV files"""
    try:
        # Load annotation files
        boxes_file = ANNOTATIONS_DIR / "BCS-DBT-boxes-train-v2.csv"
        labels_file = ANNOTATIONS_DIR / "BCS-DBT-labels-train.csv"
        paths_file = ANNOTATIONS_DIR / "BCS-DBT-file-paths-train.csv"
        
        annotations = {}
        
        if boxes_file.exists():
            annotations['boxes'] = pd.read_csv(boxes_file)
            print(f"✅ Loaded boxes: {len(annotations['boxes'])} entries")
        else:
            print(f"❌ Boxes file not found: {boxes_file}")
            
        if labels_file.exists():
            annotations['labels'] = pd.read_csv(labels_file)
            print(f"✅ Loaded labels: {len(annotations['labels'])} entries")
        else:
            print(f"❌ Labels file not found: {labels_file}")
            
        if paths_file.exists():
            annotations['paths'] = pd.read_csv(paths_file)
            print(f"✅ Loaded paths: {len(annotations['paths'])} entries")
        else:
            print(f"❌ Paths file not found: {paths_file}")
            
        return annotations
        
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return {}

def find_dicom_files():
    """Find all DICOM files in the source directory"""
    dicom_files = []
    
    print(f"🔍 Searching for DICOM files in: {DICOM_SOURCE}")
    
    for root, dirs, files in os.walk(DICOM_SOURCE):
        for file in files:
            if file.endswith('.dcm'):
                full_path = Path(root) / file
                dicom_files.append(full_path)
    
    print(f"✅ Found {len(dicom_files)} DICOM files")
    return dicom_files

def extract_patient_id_from_path(dicom_path):
    """Extract patient ID from DICOM file path"""
    path_parts = str(dicom_path).split('\\')
    for part in path_parts:
        if part.startswith('DBT-P'):
            return part
    return None

def organize_training_data():
    """Organize data into training structure"""
    print("🚀 Starting data organization...")
    
    # Load annotations
    annotations = load_annotations()
    if not annotations:
        print("❌ No annotations loaded. Cannot proceed.")
        return
    
    # Find DICOM files
    dicom_files = find_dicom_files()
    if not dicom_files:
        print("❌ No DICOM files found. Cannot proceed.")
        return
    
    # Create output directories
    (OUTPUT_DIR / "positive_cases").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "negative_cases").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Get list of patients with lesions
    patients_with_lesions = set()
    if 'boxes' in annotations:
        patients_with_lesions = set(annotations['boxes']['PatientID'].unique())
        print(f"📊 Patients with lesions: {len(patients_with_lesions)}")
    
    # Process each DICOM file
    positive_count = 0
    negative_count = 0
    
    for dicom_file in dicom_files:
        try:
            # Extract patient ID
            patient_id = extract_patient_id_from_path(dicom_file)
            if not patient_id:
                print(f"⚠️  Could not extract patient ID from: {dicom_file}")
                continue
            
            # Read DICOM to get basic info
            ds = pydicom.dcmread(dicom_file)
            
            # Determine if positive or negative case
            if patient_id in patients_with_lesions:
                # Positive case (has lesions)
                case_type = "positive_cases"
                positive_count += 1
                
                # Get lesion annotations for this patient
                patient_lesions = annotations['boxes'][annotations['boxes']['PatientID'] == patient_id]
                
                # Create patient directory
                patient_dir = OUTPUT_DIR / case_type / patient_id
                patient_dir.mkdir(exist_ok=True)
                
                # Move DICOM file (cut and paste)
                dest_file = patient_dir / f"{patient_id}_{dicom_file.stem}.dcm"
                shutil.move(str(dicom_file), str(dest_file))
                
                # Create annotation file for this patient
                annotation_file = patient_dir / f"{patient_id}_annotations.json"
                lesion_data = {
                    'patient_id': patient_id,
                    'dicom_file': str(dest_file),
                    'lesions': []
                }
                
                for _, lesion in patient_lesions.iterrows():
                    lesion_data['lesions'].append({
                        'slice': int(lesion.get('Slice', 0)),
                        'x': int(lesion.get('X', 0)),
                        'y': int(lesion.get('Y', 0)),
                        'width': int(lesion.get('Width', 0)),
                        'height': int(lesion.get('Height', 0)),
                        'class': lesion.get('Class', 'unknown'),
                        'view': lesion.get('View', 'unknown')
                    })
                
                with open(annotation_file, 'w') as f:
                    json.dump(lesion_data, f, indent=2)
                
            else:
                # Negative case (no lesions)
                case_type = "negative_cases"
                negative_count += 1
                
                # Create patient directory
                patient_dir = OUTPUT_DIR / case_type / patient_id
                patient_dir.mkdir(exist_ok=True)
                
                # Move DICOM file (cut and paste)
                dest_file = patient_dir / f"{patient_id}_{dicom_file.stem}.dcm"
                shutil.move(str(dicom_file), str(dest_file))
                
                # Create empty annotation file
                annotation_file = patient_dir / f"{patient_id}_annotations.json"
                annotation_data = {
                    'patient_id': patient_id,
                    'dicom_file': str(dest_file),
                    'lesions': []  # Empty for negative cases
                }
                
                with open(annotation_file, 'w') as f:
                    json.dump(annotation_data, f, indent=2)
            
            if (positive_count + negative_count) % 10 == 0:
                print(f"📊 Processed: {positive_count} positive, {negative_count} negative")
                
        except Exception as e:
            print(f"❌ Error processing {dicom_file}: {e}")
            continue
    
    # Create summary
    summary = {
        'total_cases': positive_count + negative_count,
        'positive_cases': positive_count,
        'negative_cases': negative_count,
        'patients_with_lesions': list(patients_with_lesions),
        'created_at': str(pd.Timestamp.now())
    }
    
    with open(OUTPUT_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Data organization complete!")
    print(f"📊 Summary:")
    print(f"   Total cases: {positive_count + negative_count}")
    print(f"   Positive cases (with lesions): {positive_count}")
    print(f"   Negative cases (no lesions): {negative_count}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    return summary

def create_training_splits():
    """Create train/validation/test splits"""
    print("\n🔄 Creating training splits...")
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / "positive").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "negative").mkdir(parents=True, exist_ok=True)
    
    # Get all patients
    positive_patients = list((OUTPUT_DIR / "positive_cases").iterdir())
    negative_patients = list((OUTPUT_DIR / "negative_cases").iterdir())
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(positive_patients)
    random.shuffle(negative_patients)
    
    # 70% train, 15% val, 15% test
    def split_data(patients, train_ratio=0.7, val_ratio=0.15):
        n = len(patients)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': patients[:train_end],
            'val': patients[train_end:val_end],
            'test': patients[val_end:]
        }
    
    pos_splits = split_data(positive_patients)
    neg_splits = split_data(negative_patients)
    
    # Move files to split directories
    for split in ['train', 'val', 'test']:
        # Positive cases
        for patient_dir in pos_splits[split]:
            dest_dir = OUTPUT_DIR / split / "positive" / patient_dir.name
            shutil.move(str(patient_dir), str(dest_dir))
        
        # Negative cases
        for patient_dir in neg_splits[split]:
            dest_dir = OUTPUT_DIR / split / "negative" / patient_dir.name
            shutil.move(str(patient_dir), str(dest_dir))
    
    print(f"✅ Training splits created:")
    print(f"   Train: {len(pos_splits['train'])} pos, {len(neg_splits['train'])} neg")
    print(f"   Val: {len(pos_splits['val'])} pos, {len(neg_splits['val'])} neg")
    print(f"   Test: {len(pos_splits['test'])} pos, {len(neg_splits['test'])} neg")

if __name__ == "__main__":
    print("🏥 DBT Training Data Organizer")
    print("=" * 50)
    
    # Check if annotation files exist
    print(f"📁 Looking for annotations in: {ANNOTATIONS_DIR}")
    
    if not ANNOTATIONS_DIR.exists():
        print(f"❌ Annotations directory not found: {ANNOTATIONS_DIR}")
        print("Please create this directory and put your CSV files there:")
        print("   - BCS-DBT-boxes-train-v2.csv")
        print("   - BCS-DBT-labels-train.csv") 
        print("   - BCS-DBT-file-paths-train.csv")
        exit(1)
    
    # Organize data
    summary = organize_training_data()
    
    if summary:
        # Create training splits
        create_training_splits()
        
        print(f"\n🎯 Next steps:")
        print(f"1. Upload {OUTPUT_DIR} to your server")
        print(f"2. Create a new training script for lesion detection")
        print(f"3. Use the organized positive/negative cases for training")
