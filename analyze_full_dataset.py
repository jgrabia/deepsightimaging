#!/usr/bin/env python3
"""
Analyze Full DBT Dataset
Check the complete lesion vs no-lesion ratio across all splits
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
ANNOTATIONS_DIR = Path("C:/MRIAPP/annotations")

def analyze_dataset():
    """Analyze the complete dataset from all CSV files"""
    print("🏥 DBT Dataset Analysis")
    print("=" * 50)
    
    # Files to check
    files_to_check = {
        'training': {
            'paths': 'BCS-DBT-file-paths-train.csv',
            'boxes': 'BCS-DBT-boxes-train-v2.csv',
            'labels': 'BCS-DBT-labels-train.csv'
        },
        'validation': {
            'paths': 'BCS-DBT-file-paths-validation.csv',
            'boxes': None  # Might not exist
        },
        'test': {
            'paths': 'BCS-DBT-file-paths-test.csv',
            'boxes': None  # Might not exist
        }
    }
    
    total_stats = {
        'total_files': 0,
        'total_patients': set(),
        'patients_with_lesions': set(),
        'total_lesions': 0,
        'splits': {}
    }
    
    # Analyze each split
    for split_name, files in files_to_check.items():
        print(f"\n📊 Analyzing {split_name.upper()} set...")
        split_stats = {
            'total_files': 0,
            'total_patients': set(),
            'patients_with_lesions': set(),
            'lesions': 0
        }
        
        # Load file paths
        paths_file = ANNOTATIONS_DIR / files['paths']
        if paths_file.exists():
            paths_df = pd.read_csv(paths_file)
            split_stats['total_files'] = len(paths_df)
            split_stats['total_patients'] = set(paths_df['PatientID'].unique())
            
            print(f"   📁 Files: {len(paths_df):,}")
            print(f"   👥 Patients: {len(split_stats['total_patients']):,}")
            
            # Add to total
            total_stats['total_files'] += len(paths_df)
            total_stats['total_patients'].update(split_stats['total_patients'])
        else:
            print(f"   ❌ File not found: {paths_file}")
            continue
        
        # Load boxes (lesions)
        boxes_file = ANNOTATIONS_DIR / files['boxes'] if files['boxes'] else None
        if boxes_file and boxes_file.exists():
            boxes_df = pd.read_csv(boxes_file)
            split_stats['patients_with_lesions'] = set(boxes_df['PatientID'].unique())
            split_stats['lesions'] = len(boxes_df)
            
            print(f"   🎯 Patients with lesions: {len(split_stats['patients_with_lesions']):,}")
            print(f"   📍 Total lesions: {len(boxes_df):,}")
            
            # Add to total
            total_stats['patients_with_lesions'].update(split_stats['patients_with_lesions'])
            total_stats['total_lesions'] += len(boxes_df)
            
            # Lesion breakdown
            if 'Class' in boxes_df.columns:
                lesion_classes = boxes_df['Class'].value_counts()
                print(f"   📋 Lesion types:")
                for lesion_type, count in lesion_classes.items():
                    print(f"      {lesion_type}: {count}")
            
        elif split_name == 'training':
            print(f"   ❌ No boxes file found for {split_name}")
        else:
            print(f"   ℹ️  No separate boxes file (may be in training boxes)")
        
        # Calculate ratios for this split
        if split_stats['total_patients']:
            positive_patients = len(split_stats['patients_with_lesions'])
            total_patients = len(split_stats['total_patients'])
            negative_patients = total_patients - positive_patients
            
            print(f"   📊 Positive patients: {positive_patients} ({positive_patients/total_patients*100:.1f}%)")
            print(f"   📊 Negative patients: {negative_patients} ({negative_patients/total_patients*100:.1f}%)")
        
        total_stats['splits'][split_name] = split_stats
    
    # Overall statistics
    print(f"\n🌟 OVERALL DATASET STATISTICS")
    print("=" * 50)
    
    total_patients = len(total_stats['total_patients'])
    positive_patients = len(total_stats['patients_with_lesions'])
    negative_patients = total_patients - positive_patients
    
    print(f"📁 Total files: {total_stats['total_files']:,}")
    print(f"👥 Total patients: {total_patients:,}")
    print(f"📍 Total lesions: {total_stats['total_lesions']:,}")
    print(f"")
    print(f"🎯 Patients WITH lesions: {positive_patients:,} ({positive_patients/total_patients*100:.1f}%)")
    print(f"🚫 Patients WITHOUT lesions: {negative_patients:,} ({negative_patients/total_patients*100:.1f}%)")
    print(f"")
    print(f"📊 Lesion to No-Lesion Ratio: 1:{negative_patients/positive_patients:.1f}")
    
    # Recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS")
    print("=" * 50)
    
    if positive_patients < 50:
        print("⚠️  Very few positive cases - consider:")
        print("   - Heavy data augmentation")
        print("   - Focal loss or weighted sampling")
        print("   - Transfer learning from pre-trained models")
    elif positive_patients < 200:
        print("⚠️  Limited positive cases - consider:")
        print("   - Moderate data augmentation") 
        print("   - Balanced sampling during training")
    else:
        print("✅ Good number of positive cases for training")
    
    if negative_patients / positive_patients > 20:
        print("⚠️  Severe class imbalance - definitely use:")
        print("   - Weighted loss functions")
        print("   - Balanced mini-batches")
        print("   - Proper evaluation metrics (F1, AUC)")
    elif negative_patients / positive_patients > 5:
        print("⚠️  Moderate class imbalance - consider balanced sampling")
    else:
        print("✅ Reasonable class balance")
    
    return total_stats

def estimate_file_sizes():
    """Estimate download sizes based on current data"""
    print(f"\n📦 DOWNLOAD SIZE ESTIMATES")
    print("=" * 50)
    
    # Based on your current data: 1947 files = ~1.6TB
    avg_file_size_mb = (1.6 * 1024) / 1947  # ~0.84 MB per file
    
    paths_files = [
        ('training', 'BCS-DBT-file-paths-train.csv'),
        ('validation', 'BCS-DBT-file-paths-validation.csv'), 
        ('test', 'BCS-DBT-file-paths-test.csv')
    ]
    
    total_estimated_size = 0
    
    for split_name, filename in paths_files:
        file_path = ANNOTATIONS_DIR / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            estimated_size_gb = (len(df) * avg_file_size_mb) / 1024
            total_estimated_size += estimated_size_gb
            print(f"{split_name.capitalize():>12}: {len(df):>6,} files → ~{estimated_size_gb:>6.1f} GB")
        else:
            print(f"{split_name.capitalize():>12}: File not found")
    
    print(f"{'Total':>12}: {total_estimated_size:>13.1f} GB")
    print(f"\nℹ️  This is an estimate based on your current average file size")

if __name__ == "__main__":
    # Check if annotation files exist
    if not ANNOTATIONS_DIR.exists():
        print(f"❌ Annotations directory not found: {ANNOTATIONS_DIR}")
        print("Please make sure you have the CSV files in the annotations directory")
        exit(1)
    
    # Analyze the dataset
    stats = analyze_dataset()
    
    # Estimate download sizes
    estimate_file_sizes()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. If ratios look good, download the missing validation/test data")
    print("2. Re-run the organization script with the complete dataset")
    print("3. Create a balanced training script that handles the class imbalance")


