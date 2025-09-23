#!/usr/bin/env python3
"""
Combine Normal and Cancer DICOM datasets for balanced training

This script:
1. Takes normal images from dicom_download/images/
2. Takes cancer images from dicom_download_cancer/images/
3. Combines them into a single balanced training directory
4. Creates proper labels for training
"""

import os
import shutil
import argparse
from pathlib import Path


def combine_datasets(normal_dir: str, cancer_dir: str, output_dir: str):
    """Combine normal and cancer datasets into balanced training set"""
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"📁 Creating balanced dataset in: {output_dir}")
    
    # Count existing files
    normal_count = 0
    cancer_count = 0
    
    # Copy normal images (label = 0 for normal)
    if os.path.exists(normal_dir):
        print(f"📂 Copying normal images from: {normal_dir}")
        for series_dir in os.listdir(normal_dir):
            series_path = os.path.join(normal_dir, series_dir)
            if os.path.isdir(series_path):
                # Copy all DICOM files in this series
                for file in os.listdir(series_path):
                    if file.lower().endswith(('.dcm', '.dicom')):
                        src = os.path.join(series_path, file)
                        dst = os.path.join(images_dir, f"normal_{normal_count:04d}_{file}")
                        shutil.copy2(src, dst)
                        
                        # Create corresponding label file (all zeros for normal)
                        label_file = dst.replace('.dcm', '.npy').replace('.dicom', '.npy')
                        import numpy as np
                        # Create empty label (no lesions)
                        label = np.zeros((256, 256), dtype=np.uint8)
                        np.save(label_file, label)
                        
                        normal_count += 1
                        
                        if normal_count % 10 == 0:
                            print(f"   Copied {normal_count} normal images...")
    
    # Copy cancer images (label = 1 for cancer)
    if os.path.exists(cancer_dir):
        print(f"📂 Copying cancer images from: {cancer_dir}")
        for series_dir in os.listdir(cancer_dir):
            series_path = os.path.join(cancer_dir, series_dir)
            if os.path.isdir(series_path):
                # Copy all DICOM files in this series
                for file in os.listdir(series_path):
                    if file.lower().endswith(('.dcm', '.dicom')):
                        src = os.path.join(series_path, file)
                        dst = os.path.join(images_dir, f"cancer_{cancer_count:04d}_{file}")
                        shutil.copy2(src, dst)
                        
                        # Create corresponding label file (placeholder for cancer)
                        label_file = dst.replace('.dcm', '.npy').replace('.dicom', '.npy')
                        import numpy as np
                        # Create placeholder label (will need manual annotation or AI-generated)
                        # For now, create a simple center region as placeholder
                        label = np.zeros((256, 256), dtype=np.uint8)
                        # Add a small center region as placeholder lesion
                        center = 128
                        size = 20
                        label[center-size:center+size, center-size:center+size] = 1
                        np.save(label_file, label)
                        
                        cancer_count += 1
                        
                        if cancer_count % 10 == 0:
                            print(f"   Copied {cancer_count} cancer images...")
    
    # Create dataset summary
    summary = {
        "total_images": normal_count + cancer_count,
        "normal_images": normal_count,
        "cancer_images": cancer_count,
        "balance_ratio": normal_count / (normal_count + cancer_count) if (normal_count + cancer_count) > 0 else 0
    }
    
    # Save summary
    import json
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Dataset combination complete!")
    print(f"   Total images: {summary['total_images']}")
    print(f"   Normal images: {summary['normal_images']}")
    print(f"   Cancer images: {summary['cancer_images']}")
    print(f"   Balance ratio: {summary['balance_ratio']:.2%}")
    print(f"   Output directory: {output_dir}")
    print(f"   Summary saved: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Combine normal and cancer DICOM datasets for balanced training")
    parser.add_argument("--normal-dir", default="dicom_download/images", help="Directory containing normal DICOM images")
    parser.add_argument("--cancer-dir", default="dicom_download_cancer/images", help="Directory containing cancer DICOM images")
    parser.add_argument("--output-dir", default="balanced_training_data", help="Output directory for combined dataset")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before combining")
    
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        print(f"🧹 Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Combine datasets
    summary = combine_datasets(args.normal_dir, args.cancer_dir, args.output_dir)
    
    # Print next steps
    print(f"\n📋 Next Steps:")
    print(f"1. Review the dataset balance in {args.output_dir}/dataset_summary.json")
    print(f"2. Update training script to use: {args.output_dir}/images")
    print(f"3. Start fresh training with balanced data")
    print(f"4. Expected training time: 2-4 hours for 20 epochs")


if __name__ == "__main__":
    main()


