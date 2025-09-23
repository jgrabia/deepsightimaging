#!/usr/bin/env python3
"""
Automated Training Data Collector for Breast Cancer AI
Downloads curated datasets from TCIA with better labels and metadata
"""

import os
import json
import time
import requests
from tcia_utils import nbia
import zipfile
import tempfile
from datetime import datetime
import pandas as pd

class AutomatedDataCollector:
    def __init__(self, output_dir="training_data"):
        self.output_dir = output_dir
        self.metadata_file = os.path.join(output_dir, "dataset_metadata.json")
        self.download_log = os.path.join(output_dir, "download_log.txt")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Curated breast cancer datasets with better labels
        self.curated_datasets = {
            "TCGA-BRCA": {
                "description": "The Cancer Genome Atlas Breast Cancer Collection",
                "filters": {
                    "Collection": "TCGA-BRCA",
                    "BodyPartExamined": "BREAST",
                    "Modality": "MR"
                },
                "expected_count": 1000,
                "label_quality": "High - Clinical annotations available"
            },
            "CBIS-DDSM": {
                "description": "Curated Breast Imaging Subset of DDSM",
                "filters": {
                    "Collection": "CBIS-DDSM",
                    "BodyPartExamined": "BREAST",
                    "Modality": "MG"
                },
                "expected_count": 2000,
                "label_quality": "Very High - Expert radiologist annotations"
            },
            "INbreast": {
                "description": "INbreast Database for Mammographic Image Analysis",
                "filters": {
                    "Collection": "INbreast",
                    "BodyPartExamined": "BREAST",
                    "Modality": "MG"
                },
                "expected_count": 500,
                "label_quality": "Very High - Expert annotations with BI-RADS"
            },
            "BCDR": {
                "description": "Breast Cancer Digital Repository",
                "filters": {
                    "Collection": "BCDR",
                    "BodyPartExamined": "BREAST"
                },
                "expected_count": 800,
                "label_quality": "High - Clinical and pathological data"
            }
        }
    
    def log_message(self, message):
        """Log message to file and print"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.download_log, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    
    def get_dataset_info(self, dataset_name):
        """Get detailed information about a dataset"""
        if dataset_name not in self.curated_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset = self.curated_datasets[dataset_name]
        self.log_message(f"🔍 Analyzing dataset: {dataset_name}")
        self.log_message(f"   Description: {dataset['description']}")
        self.log_message(f"   Expected count: {dataset['expected_count']} images")
        self.log_message(f"   Label quality: {dataset['label_quality']}")
        
        # Get actual series count
        try:
            series = nbia.getSeries(**dataset['filters'])
            actual_count = len(series)
            self.log_message(f"   Actual series found: {actual_count}")
            
            # Get unique patients
            patient_ids = set()
            for s in series:
                if 'PatientID' in s:
                    patient_ids.add(s['PatientID'])
            
            self.log_message(f"   Unique patients: {len(patient_ids)}")
            
            return {
                "dataset_name": dataset_name,
                "series_count": actual_count,
                "patient_count": len(patient_ids),
                "series": series,
                "metadata": dataset
            }
        except Exception as e:
            self.log_message(f"❌ Error analyzing dataset {dataset_name}: {e}")
            return None
    
    def download_dataset(self, dataset_name, max_series=None, include_metadata=True):
        """Download a complete dataset with metadata"""
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            return False
        
        series = dataset_info['series']
        if max_series:
            series = series[:max_series]
            self.log_message(f"📊 Limiting to {max_series} series")
        
        # Create dataset directory
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save metadata
        if include_metadata:
            metadata_path = os.path.join(dataset_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(dataset_info, f, indent=2)
            self.log_message(f"💾 Saved metadata to {metadata_path}")
        
        # Download series
        successful_downloads = 0
        failed_downloads = 0
        
        for i, series_info in enumerate(series):
            series_uid = series_info['SeriesInstanceUID']
            patient_id = series_info.get('PatientID', 'Unknown')
            
            self.log_message(f"📥 Downloading series {i+1}/{len(series)}: {patient_id}")
            
            try:
                # Download series
                zip_path = self.download_series(series_uid, dataset_dir)
                if zip_path:
                    successful_downloads += 1
                    self.log_message(f"✅ Downloaded: {zip_path}")
                else:
                    failed_downloads += 1
                    self.log_message(f"❌ Failed to download series {series_uid}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                failed_downloads += 1
                self.log_message(f"❌ Error downloading series {series_uid}: {e}")
        
        # Summary
        self.log_message(f"🎉 Dataset {dataset_name} download complete!")
        self.log_message(f"   Successful: {successful_downloads}")
        self.log_message(f"   Failed: {failed_downloads}")
        self.log_message(f"   Success rate: {successful_downloads/(successful_downloads+failed_downloads)*100:.1f}%")
        
        return successful_downloads > 0
    
    def download_series(self, series_uid, output_dir):
        """Download a single series from TCIA"""
        try:
            # Get download URL
            url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                
                tmp_path = tmp_file.name
            
            # Move to final location
            final_path = os.path.join(output_dir, f"{series_uid}.zip")
            os.rename(tmp_path, final_path)
            
            return final_path
            
        except Exception as e:
            self.log_message(f"Error downloading series {series_uid}: {e}")
            return None
    
    def create_training_manifest(self):
        """Create a manifest file for training"""
        manifest = {
            "created": datetime.now().isoformat(),
            "datasets": {},
            "total_images": 0,
            "total_patients": 0
        }
        
        for dataset_name in os.listdir(self.output_dir):
            dataset_path = os.path.join(self.output_dir, dataset_name)
            if os.path.isdir(dataset_path):
                # Count ZIP files
                zip_files = [f for f in os.listdir(dataset_path) if f.endswith('.zip')]
                
                # Load metadata if available
                metadata_path = os.path.join(dataset_path, "metadata.json")
                metadata = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                manifest["datasets"][dataset_name] = {
                    "series_count": len(zip_files),
                    "metadata": metadata
                }
                manifest["total_images"] += len(zip_files)
        
        # Save manifest
        manifest_path = os.path.join(self.output_dir, "training_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        self.log_message(f"📋 Created training manifest: {manifest_path}")
        return manifest
    
    def run_full_collection(self, max_series_per_dataset=50):
        """Run full automated data collection"""
        self.log_message("🚀 Starting automated training data collection...")
        
        successful_datasets = []
        
        for dataset_name in self.curated_datasets.keys():
            self.log_message(f"\n{'='*60}")
            self.log_message(f"Processing dataset: {dataset_name}")
            self.log_message(f"{'='*60}")
            
            try:
                success = self.download_dataset(dataset_name, max_series_per_dataset)
                if success:
                    successful_datasets.append(dataset_name)
                
                # Wait between datasets
                time.sleep(5)
                
            except Exception as e:
                self.log_message(f"❌ Failed to process dataset {dataset_name}: {e}")
        
        # Create final manifest
        self.log_message(f"\n{'='*60}")
        self.log_message("Creating training manifest...")
        manifest = self.create_training_manifest()
        
        # Summary
        self.log_message(f"\n🎉 Collection complete!")
        self.log_message(f"Successful datasets: {len(successful_datasets)}")
        self.log_message(f"Total images collected: {manifest['total_images']}")
        self.log_message(f"Output directory: {self.output_dir}")
        
        return manifest

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Training Data Collector")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    parser.add_argument("--dataset", help="Specific dataset to download")
    parser.add_argument("--max-series", type=int, default=50, help="Max series per dataset")
    parser.add_argument("--full-collection", action="store_true", help="Download all datasets")
    
    args = parser.parse_args()
    
    collector = AutomatedDataCollector(args.output_dir)
    
    if args.dataset:
        # Download specific dataset
        collector.download_dataset(args.dataset, args.max_series)
    elif args.full_collection:
        # Download all datasets
        collector.run_full_collection(args.max_series)
    else:
        # Show available datasets
        print("Available datasets:")
        for name, info in collector.curated_datasets.items():
            print(f"  {name}: {info['description']}")
        print("\nUse --dataset <name> to download a specific dataset")
        print("Use --full-collection to download all datasets")

if __name__ == "__main__":
    main()





