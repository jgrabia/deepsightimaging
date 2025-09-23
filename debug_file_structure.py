#!/usr/bin/env python3
"""
Debug File Structure
Check the actual structure of downloaded files to understand the naming pattern
"""

import os
from pathlib import Path
import pandas as pd

# Configuration
DICOM_SOURCE = Path("/home/ubuntu/mri_app/breast_dbt_complete/training/images")
ANNOTATIONS_DIR = Path("/home/ubuntu/mri_app/annotations")

def analyze_file_structure():
    """Analyze the actual file structure"""
    print("🔍 ANALYZING FILE STRUCTURE")
    print("=" * 50)
    
    print(f"📁 Source directory: {DICOM_SOURCE}")
    print(f"   Directory exists: {DICOM_SOURCE.exists()}")
    
    if not DICOM_SOURCE.exists():
        return
    
    # Get first 10 files to see the pattern
    files = list(DICOM_SOURCE.iterdir())[:10]
    
    print(f"\n📄 Sample files (first 10):")
    for i, file in enumerate(files, 1):
        print(f"   {i:2d}. {file.name}")
        print(f"       Full path: {file}")
        print(f"       Is file: {file.is_file()}")
        print(f"       Is dir: {file.is_dir()}")
        if file.is_dir():
            # Show contents of directory
            contents = list(file.iterdir())[:3]
            print(f"       Contents (first 3): {[c.name for c in contents]}")
        print()

def analyze_annotations():
    """Analyze annotation files to see patient ID patterns"""
    print("\n🔍 ANALYZING ANNOTATIONS")
    print("=" * 50)
    
    # Load paths file
    paths_file = ANNOTATIONS_DIR / "BCS-DBT-file-paths-train.csv"
    if paths_file.exists():
        df = pd.read_csv(paths_file)
        print(f"📊 Paths CSV columns: {list(df.columns)}")
        print(f"📊 Total rows: {len(df)}")
        
        # Show sample patient IDs and paths
        print(f"\n📄 Sample patient IDs and paths:")
        for i, row in df.head(5).iterrows():
            patient_id = row.get('PatientID', 'N/A')
            desc_path = row.get('descriptive_path', 'N/A')
            classic_path = row.get('classic_path', 'N/A')
            
            print(f"   Patient: {patient_id}")
            print(f"   Desc path: {desc_path}")
            print(f"   Classic path: {classic_path}")
            print()
        
        # Check unique patient count
        unique_patients = df['PatientID'].nunique()
        print(f"📊 Unique patients in CSV: {unique_patients}")
        
        # Show patient ID patterns
        sample_patients = df['PatientID'].unique()[:10]
        print(f"📄 Sample patient IDs: {list(sample_patients)}")
    else:
        print(f"❌ Paths file not found: {paths_file}")

def analyze_boxes():
    """Analyze boxes file to see lesion patient patterns"""
    print("\n🔍 ANALYZING LESION BOXES")
    print("=" * 50)
    
    boxes_file = ANNOTATIONS_DIR / "BCS-DBT-boxes-train-v2.csv"
    if boxes_file.exists():
        df = pd.read_csv(boxes_file)
        print(f"📊 Boxes CSV columns: {list(df.columns)}")
        print(f"📊 Total lesions: {len(df)}")
        
        # Show patients with lesions
        patients_with_lesions = df['PatientID'].unique()
        print(f"📊 Patients with lesions: {len(patients_with_lesions)}")
        print(f"📄 Sample patients with lesions: {list(patients_with_lesions[:10])}")
        
        # Show lesion types
        if 'Class' in df.columns:
            lesion_types = df['Class'].value_counts()
            print(f"📊 Lesion types: {lesion_types.to_dict()}")
    else:
        print(f"❌ Boxes file not found: {boxes_file}")

def check_file_naming_patterns():
    """Check if we can find any patient ID patterns in the downloaded files"""
    print("\n🔍 CHECKING FILE NAMING PATTERNS")
    print("=" * 50)
    
    files = list(DICOM_SOURCE.rglob("*"))[:20]  # Get files recursively
    
    print("📄 Looking for patient ID patterns in file paths:")
    for file in files:
        file_str = str(file)
        
        # Look for DBT-P patterns
        if 'DBT-P' in file_str:
            print(f"   ✅ Found DBT-P pattern: {file}")
        elif any(char.isdigit() for char in file.name):
            print(f"   ? Numeric pattern: {file}")
    
    # Check if files are organized in patient directories
    print(f"\n📁 Directory structure analysis:")
    subdirs = [d for d in DICOM_SOURCE.iterdir() if d.is_dir()][:5]
    print(f"   Subdirectories (first 5): {[d.name for d in subdirs]}")
    
    for subdir in subdirs:
        contents = list(subdir.iterdir())[:3]
        print(f"   {subdir.name} contents: {[c.name for c in contents]}")

if __name__ == "__main__":
    print("🏥 DBT File Structure Debugger")
    print("=" * 60)
    
    analyze_file_structure()
    analyze_annotations()
    analyze_boxes()
    check_file_naming_patterns()
    
    print("\n💡 RECOMMENDATIONS:")
    print("Based on the analysis above, we can determine:")
    print("1. How the downloaded files are actually organized")
    print("2. What patient ID patterns exist in file names/paths")
    print("3. How to modify the organization script to match files correctly")


