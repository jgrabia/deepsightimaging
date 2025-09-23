#!/usr/bin/env python3
"""
Organize Complete DBT Training Data
Creates a proper structure for training lesion detection models with the complete dataset
"""

import os
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path
import shutil
import json
from PIL import Image

# Configuration - Updated for complete dataset
BASE_DIR = Path("/home/ubuntu/mri_app")
DICOM_SOURCE = BASE_DIR / "breast_dbt_complete/training/images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
OUTPUT_DIR = BASE_DIR / "dbt_complete_training_data"

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
            print(f"   Lesion types: {annotations['boxes']['Class'].value_counts().to_dict()}")
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
            print(f"   Unique patients in paths: {annotations['paths']['PatientID'].nunique()}")
        else:
            print(f"❌ Paths file not found: {paths_file}")
            
        return annotations
        
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return {}

def find_dicom_files():
    """Find all DICOM files in the complete dataset"""
    dicom_files = []
    
    print(f"🔍 Searching for DICOM files in: {DICOM_SOURCE}")
    
    if not DICOM_SOURCE.exists():
        print(f"❌ Source directory not found: {DICOM_SOURCE}")
        return []
    
    # Navigate through the nested structure: manifest-*/Breast-Cancer-Screening-DBT/DBT-P*/
    manifest_dirs = [d for d in DICOM_SOURCE.iterdir() if d.is_dir() and 'manifest' in d.name]
    
    for manifest_dir in manifest_dirs:
        print(f"   📁 Processing manifest: {manifest_dir.name}")
        
        # Look for Breast-Cancer-Screening-DBT directory
        breast_dir = manifest_dir / "Breast-Cancer-Screening-DBT"
        if breast_dir.exists():
            # Look for patient directories (DBT-P*)
            patient_dirs = [d for d in breast_dir.iterdir() if d.is_dir() and d.name.startswith('DBT-P')]
            print(f"   👥 Found {len(patient_dirs)} patient directories")
            
            for patient_dir in patient_dirs:
                # Look for DICOM files in all subdirectories of this patient
                for root, dirs, files in os.walk(patient_dir):
                    for file in files:
                        if file.lower().endswith('.dcm'):
                            full_path = Path(root) / file
                            dicom_files.append(full_path)
    
    print(f"✅ Found {len(dicom_files)} DICOM files")
    return dicom_files

def extract_patient_id_from_path(file_path):
    """Extract patient ID from DICOM file path"""
    path_str = str(file_path)
    path_parts = path_str.split('/')
    
    # Look for DBT-P pattern in path parts
    for part in path_parts:
        if part.startswith('DBT-P') and len(part) >= 9:
            # Extract the patient ID (DBT-P followed by 5 digits)
            import re
            match = re.match(r'(DBT-P\d{5})', part)
            if match:
                return match.group(1)
    
    return None

def match_files_with_annotations(dicom_files, annotations):
    """Match DICOM files with their annotations using the paths CSV"""
    print(f"\n🔗 Matching DICOM files with annotations...")
    
    if 'paths' not in annotations:
        print("❌ No paths data available for matching")
        return {}
    
    paths_df = annotations['paths']
    
    # Create a mapping from patient ID to file paths in the CSV
    patient_to_csv_paths = {}
    for _, row in paths_df.iterrows():
        patient_id = row['PatientID']
        csv_path = row.get('descriptive_path', '') or row.get('classic_path', '')
        
        if patient_id not in patient_to_csv_paths:
            patient_to_csv_paths[patient_id] = []
        patient_to_csv_paths[patient_id].append(csv_path)
    
    print(f"📊 Found {len(patient_to_csv_paths)} patients in CSV paths")
    
    # Match downloaded files with annotations
    matched_files = {}
    unmatched_files = []
    
    for dicom_file in dicom_files:
        patient_id = extract_patient_id_from_path(dicom_file)
        
        if patient_id and patient_id in patient_to_csv_paths:
            if patient_id not in matched_files:
                matched_files[patient_id] = []
            matched_files[patient_id].append(dicom_file)
        else:
            unmatched_files.append(dicom_file)
    
    print(f"✅ Matched {len(matched_files)} patients")
    print(f"⚠️  Unmatched files: {len(unmatched_files)}")
    
    return matched_files

def organize_training_data():
    """Organize data into training structure"""
    print("🚀 Starting complete dataset organization...")
    
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
    
    # Match files with annotations
    matched_files = match_files_with_annotations(dicom_files, annotations)
    if not matched_files:
        print("❌ No files could be matched with annotations.")
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
    
    # Process each patient's files
    positive_count = 0
    negative_count = 0
    processed_files = 0
    
    for patient_id, patient_files in matched_files.items():
        try:
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
                
                # Process each file for this patient
                for i, dicom_file in enumerate(patient_files):
                    # Move DICOM file
                    dest_file = patient_dir / f"{patient_id}_{i+1:03d}.dcm"
                    if dicom_file.suffix.lower() == '.dcm':
                        shutil.move(str(dicom_file), str(dest_file))
                    elif dicom_file.suffix.lower() == '.zip':
                        # Handle zip files - extract first
                        import zipfile
                        try:
                            with zipfile.ZipFile(dicom_file, 'r') as zip_ref:
                                zip_ref.extractall(patient_dir)
                            dicom_file.unlink()  # Remove zip after extraction
                        except:
                            # If can't extract, just move the zip
                            dest_file = patient_dir / f"{patient_id}_{i+1:03d}.zip"
                            shutil.move(str(dicom_file), str(dest_file))
                    
                    processed_files += 1
                
                # Create annotation file for this patient
                annotation_file = patient_dir / f"{patient_id}_annotations.json"
                lesion_data = {
                    'patient_id': patient_id,
                    'case_type': 'positive',
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
                
                # Process each file for this patient
                for i, dicom_file in enumerate(patient_files):
                    # Move DICOM file
                    dest_file = patient_dir / f"{patient_id}_{i+1:03d}.dcm"
                    if dicom_file.suffix.lower() == '.dcm':
                        shutil.move(str(dicom_file), str(dest_file))
                    elif dicom_file.suffix.lower() == '.zip':
                        # Handle zip files - extract first
                        import zipfile
                        try:
                            with zipfile.ZipFile(dicom_file, 'r') as zip_ref:
                                zip_ref.extractall(patient_dir)
                            dicom_file.unlink()  # Remove zip after extraction
                        except:
                            # If can't extract, just move the zip
                            dest_file = patient_dir / f"{patient_id}_{i+1:03d}.zip"
                            shutil.move(str(dicom_file), str(dest_file))
                    
                    processed_files += 1
                
                # Create empty annotation file
                annotation_file = patient_dir / f"{patient_id}_annotations.json"
                annotation_data = {
                    'patient_id': patient_id,
                    'case_type': 'negative',
                    'lesions': []  # Empty for negative cases
                }
                
                with open(annotation_file, 'w') as f:
                    json.dump(annotation_data, f, indent=2)
            
            if (positive_count + negative_count) % 50 == 0:
                print(f"📊 Processed: {positive_count} positive, {negative_count} negative patients")
                
        except Exception as e:
            print(f"❌ Error processing {patient_id}: {e}")
            continue
    
    # Create summary
    summary = {
        'total_patients': positive_count + negative_count,
        'positive_patients': positive_count,
        'negative_patients': negative_count,
        'total_files_processed': processed_files,
        'patients_with_lesions': list(patients_with_lesions),
        'created_at': str(pd.Timestamp.now()),
        'dataset_version': 'complete'
    }
    
    with open(OUTPUT_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Complete dataset organization finished!")
    print(f"📊 Summary:")
    print(f"   Total patients: {positive_count + negative_count}")
    print(f"   Positive patients (with lesions): {positive_count}")
    print(f"   Negative patients (no lesions): {negative_count}")
    print(f"   Total files processed: {processed_files}")
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
    print("🏥 Complete DBT Training Data Organizer")
    print("=" * 60)
    
    # Check if source directory exists
    print(f"📁 Looking for DICOM files in: {DICOM_SOURCE}")
    if not DICOM_SOURCE.exists():
        print(f"❌ Source directory not found: {DICOM_SOURCE}")
        print("Please make sure you have downloaded the complete dataset")
        exit(1)
    
    # Check if annotation files exist
    print(f"📁 Looking for annotations in: {ANNOTATIONS_DIR}")
    if not ANNOTATIONS_DIR.exists():
        print(f"❌ Annotations directory not found: {ANNOTATIONS_DIR}")
        print("Please create this directory and put your CSV files there")
        exit(1)
    
    # Organize data
    summary = organize_training_data()
    
    if summary and summary['total_patients'] > 0:
        # Create training splits
        create_training_splits()
        
        print(f"\n🎯 Next steps:")
        print(f"1. Create a lesion detection training script")
        print(f"2. Use the organized positive/negative cases for training")
        print(f"3. Train with proper class balancing techniques")
        print(f"\n📁 Your organized dataset is ready at: {OUTPUT_DIR}")
    else:
        print("❌ No data was organized. Please check your files and try again.")
