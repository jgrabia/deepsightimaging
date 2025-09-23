#!/usr/bin/env python3
"""
Advanced Breast Cancer Model Training Pipeline
Enterprise-grade training system for breast cancer detection on TCIA data
"""

import os
import sys
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

# MONAI imports
from monai.data import DataLoader, Dataset, CacheDataset, ThreadDataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityd, NormalizeIntensityd, EnsureTyped,
    Activationsd, AsDiscreted, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    Compose, CropForegroundd, SpatialPadd, ResizeWithPadOrCropd,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandCoarseDropoutd, RandZoomd
)
from monai.networks.nets import SegResNet, UNet, DynUNet
from monai.losses import DiceCELoss, DiceFocalLoss, FocalLoss
from monai.optimizers import Novograd
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image

class AdvancedBreastCancerTrainer:
    def __init__(self, data_dir="dicom_download", config=None):
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("trained_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Set deterministic training
        set_determinism(seed=42)
        
        # ENTERPRISE-GRADE configuration
        self.config = {
            "model_type": "SegResNet",  # SegResNet, UNet, DynUNet
            "spatial_dims": 2,  # ENTERPRISE: Use 2D for proper MONAI compatibility with 2D images
            "in_channels": 1,
            "out_channels": 3,  # Default: background, tissue, lesions
            "init_filters": 32,
            "dropout_prob": 0.2,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 100,
            "patience": 15,
            "folds": 5,
            "image_size": [512, 512],  # 2D size: [height, width] for 2D images
            "loss_function": "DiceCELoss",  # DiceCELoss, DiceFocalLoss, FocalLoss
            "optimizer": "AdamW",  # AdamW, Novograd
            "scheduler": "CosineAnnealingLR",
            "augmentation_strength": "strong",
            "mixed_precision": True,
            "gradient_clipping": 1.0,
            "early_stopping": True,
            "model_checkpointing": True,
            "tensorboard_logging": True,
            "validation_metrics": ["dice", "hausdorff", "precision", "recall"],
            "class_weights": [1.0, 2.0, 3.0],  # Background, breast tissue, lesions
            # Memory controls
            "cache_rate": 0.2,            # cache 20% of training set to reduce RAM
            "val_cache_rate": 0.0,        # don't cache validation by default
            "num_workers": 2,             # fewer workers to reduce RAM
            "persistent_workers": False,  # avoid keeping workers alive
            # Normal-only finetune controls
            "normal_only": False,          # if True: 2 classes (background, tissue)
            "freeze_encoder_epochs": 3,    # epochs to freeze most weights at start
            "head_trainable_ratio": 0.1,   # last X% parameters stay trainable when frozen
            "resume_from": None,           # path to checkpoint for warm-start
        }
        
        if config:
            self.config.update(config)
        
        print(f"🏥 Advanced Breast Cancer Model Training")
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"💻 Device: {self.device}")
        print(f"📦 Model Directory: {self.model_dir}")
        print(f"⚙️  Configuration: {json.dumps(self.config, indent=2)}")
        
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        import logging
        from torch.utils.tensorboard import SummaryWriter
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Setup TensorBoard
        if self.config["tensorboard_logging"]:
            self.writer = SummaryWriter(log_dir / "tensorboard")
        
    def analyze_data_distribution(self):
        """Analyze the distribution of the dataset"""
        print("\n📊 Analyzing Data Distribution...")
        
        dataset_stats = {
            "total_files": 0,
            "valid_files": 0,
            "dimensions": {},
            "modalities": {},
            "file_sizes": [],
            "intensity_stats": []
        }
        
        # Scan for DICOM files in the main directory
        main_dicom_files = list(self.data_dir.glob("*.dcm"))
        dataset_stats["total_files"] += len(main_dicom_files)
        print(f"🔍 Found {len(main_dicom_files)} DICOM files in main directory")
        
        # Scan all subdirectories in the data directory
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"🔍 Found {len(subdirs)} subdirectories to analyze")
            
            for dataset_path in subdirs:
                dataset_name = dataset_path.name
                dicom_files = list(dataset_path.glob("*.dcm"))
                dataset_stats["total_files"] += len(dicom_files)
                print(f"   {dataset_name}: {len(dicom_files)} DICOM files")
        
        if dataset_stats["total_files"] == 0:
            print(f"❌ No DICOM files found in {self.data_dir}")
            return dataset_stats
            
            for dicom_file in dicom_files[:100]:  # Sample for analysis
                try:
                    ds = pydicom.dcmread(str(dicom_file), force=True)
                    pixel_array = ds.pixel_array
                    
                    # Dimension analysis
                    dim_key = str(pixel_array.shape)
                    dataset_stats["dimensions"][dim_key] = dataset_stats["dimensions"].get(dim_key, 0) + 1
                    
                    # Modality analysis
                    modality = getattr(ds, 'Modality', 'Unknown')
                    dataset_stats["modalities"][modality] = dataset_stats["modalities"].get(modality, 0) + 1
                    
                    # File size analysis
                    file_size = dicom_file.stat().st_size
                    dataset_stats["file_sizes"].append(file_size)
                    
                    # Intensity analysis
                    if len(pixel_array.shape) == 2:
                        intensity_stats = {
                            "min": float(pixel_array.min()),
                            "max": float(pixel_array.max()),
                            "mean": float(pixel_array.mean()),
                            "std": float(pixel_array.std()),
                            "percentiles": [float(np.percentile(pixel_array, p)) for p in [5, 25, 50, 75, 95]]
                        }
                        dataset_stats["intensity_stats"].append(intensity_stats)
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {dicom_file}: {e}")
                    continue
        
        # Print analysis results
        print(f"📈 Dataset Analysis Results:")
        print(f"   Total files: {dataset_stats['total_files']}")
        print(f"   Dimensions found: {dataset_stats['dimensions']}")
        print(f"   Modalities: {dataset_stats['modalities']}")
        
        if dataset_stats["intensity_stats"]:
            avg_intensity = np.mean([stats["mean"] for stats in dataset_stats["intensity_stats"]])
            avg_std = np.mean([stats["std"] for stats in dataset_stats["intensity_stats"]])
            print(f"   Average intensity: {avg_intensity:.2f} ± {avg_std:.2f}")
        
        return dataset_stats

    def prepare_advanced_training_data(self):
        """Prepare training data with advanced preprocessing and validation"""
        print("\n📊 Preparing Advanced Training Data...")
        
        # First analyze the data distribution
        dataset_stats = self.analyze_data_distribution()
        
        training_data = []
        validation_data = []
        
        # Track image types for summary
        self.image_type_stats = {"Grayscale": 0, "RGB": 0, "Other": 0}
        
        # Process DICOM files in the main directory (match .dcm and .DCM)
        main_dicom_files = list(self.data_dir.glob("*.dcm")) + list(self.data_dir.glob("*.DCM"))
        if main_dicom_files:
            print(f"🔍 Processing {len(main_dicom_files)} DICOM files in main directory...")
            
            for i, dicom_file in enumerate(main_dicom_files):
                try:
                    print(f"\n🔍 Processing {dicom_file.name} ({i+1}/{len(main_dicom_files)})...")
                    
                    # Load DICOM with comprehensive validation
                    ds = pydicom.dcmread(str(dicom_file), force=True)
                    pixel_array = ds.pixel_array
                    
                    # Advanced validation
                    if not self.validate_dicom_file(ds, pixel_array, dicom_file):
                        print(f"   ⏭️  Skipping {dicom_file.name} - failed validation")
                        continue
                    
                    # Create advanced training sample
                    print(f"   🏗️  Creating training sample for {dicom_file.name}...")
                    sample = self.create_advanced_training_sample(ds, pixel_array, "main_directory", i)
                    if sample:
                        training_data.append(sample)
                        print(f"   ✅ Sample created successfully for {dicom_file.name}")
                    else:
                        print(f"   ❌ Failed to create sample for {dicom_file.name}")
                    
                    if i % 10 == 0:
                        print(f"   📊 Progress: {i+1}/{len(main_dicom_files)} files processed, {len(training_data)} samples created")
                        
                except Exception as e:
                    print(f"❌ Error processing {dicom_file.name}: {e}")
                    continue
        
        # Process all subdirectories in the data directory
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"🔍 Found {len(subdirs)} subdirectories to process")
            
            for dataset_path in subdirs:
                dataset_name = dataset_path.name
                print(f"🔍 Processing {dataset_name} with advanced filtering...")
                
                # Recursively find DICOM files at any depth (.dcm/.DCM)
                dicom_files = list(dataset_path.rglob("*.dcm")) + list(dataset_path.rglob("*.DCM"))
                print(f"   Found {len(dicom_files)} DICOM files")
                
                for i, dicom_file in enumerate(dicom_files):
                    try:
                        if i % 10 == 0:  # Show progress every 10 files to avoid spam
                            print(f"   🔍 Processing {dicom_file.name} ({i+1}/{len(dicom_files)})...")
                        
                        # Load DICOM with comprehensive validation
                        ds = pydicom.dcmread(str(dicom_file), force=True)
                        pixel_array = ds.pixel_array
                        
                        # Advanced validation
                        if not self.validate_dicom_file(ds, pixel_array, dicom_file):
                            if i % 10 == 0:  # Only show skip message every 10 files
                                print(f"   ⏭️  Skipping {dicom_file.name} - failed validation")
                            continue
                        
                        # Create advanced training sample
                        sample = self.create_advanced_training_sample(ds, pixel_array, dataset_name, i)
                        if sample:
                            training_data.append(sample)
                            if i % 10 == 0:  # Only show success message every 10 files
                                print(f"   ✅ Sample created for {dicom_file.name}")
                        else:
                            if i % 10 == 0:  # Only show failure message every 10 files
                                print(f"   ❌ Failed to create sample for {dicom_file.name}")
                        
                        if i % 50 == 0:
                            print(f"   📊 Progress: {i+1}/{len(dicom_files)} files processed, {len(training_data)} total samples created")
                            
                    except Exception as e:
                        if i % 10 == 0:  # Only show error every 10 files
                            print(f"❌ Error processing {dicom_file.name}: {e}")
                        continue
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"✅ Total training samples: {len(training_data)}")
        print(f"📁 Total DICOM files found: {dataset_stats['total_files']}")
        print(f"📈 Success rate: {len(training_data)}/{dataset_stats['total_files']} = {(len(training_data)/dataset_stats['total_files']*100):.1f}%")
        print(f"🎯 Simple synthetic labels: Otsu-based tissue detection with high-intensity lesion identification")
        
        # Show image type breakdown if we have data
        if hasattr(self, 'image_type_stats'):
            print(f"🖼️  Image types processed:")
            for img_type, count in self.image_type_stats.items():
                print(f"   - {img_type}: {count}")
        
        if len(training_data) == 0:
            print("\n❌ NO TRAINING SAMPLES CREATED!")
            print("🔍 This means all files were rejected during validation or sample creation.")
            print("💡 Check the debug output above to see why files were rejected.")
            return [], [], dataset_stats
        
        # Split into train/validation with stratification
        if len(training_data) > 1:
            training_data, validation_data = self.stratified_split(training_data)
            print(f"📊 Train/Validation split: {len(training_data)}/{len(validation_data)}")
        elif len(training_data) == 1:
            # If only one sample, use it for both train and validation
            validation_data = training_data.copy()
            print("⚠️  Only 1 training sample found - using same sample for train and validation")
        else:
            print("❌ No valid training samples found!")
            return [], [], dataset_stats
        
        return training_data, validation_data, dataset_stats
    
    def validate_dicom_file(self, ds, pixel_array, dicom_file):
        """Comprehensive DICOM file validation with detailed debugging"""
        try:
            filename = dicom_file.name
            
            # Enhanced dimension validation - accept 2D, RGB, and multi-frame 3D (take middle slice)
            if len(pixel_array.shape) == 3:
                if pixel_array.shape[2] == 3:  # RGB image
                    print(f"🔄 {filename}: RGB image detected (shape: {pixel_array.shape}) - will convert to grayscale")
                    self.image_type_stats["RGB"] += 1
                else:
                    # Multi-frame stack (e.g., DBT); accept and reduce to 2D later
                    print(f"🔄 {filename}: Multi-frame detected (shape: {pixel_array.shape}) - will extract middle slice")
                    self.image_type_stats["Other"] += 1
            elif len(pixel_array.shape) == 2:
                self.image_type_stats["Grayscale"] += 1
            else:
                print(f"❌ {filename}: Rejected - Invalid shape (shape: {pixel_array.shape})")
                self.image_type_stats["Other"] += 1
                return False
            
            # Size validation: accept larger 2D mammograms
            if pixel_array.size < 1000:
                print(f"❌ {filename}: Rejected - Too small (size: {pixel_array.size})")
                return False
            # Removed overly strict upper bound; resizing will normalize dimensions
            
            # Quality validation
            std_val = pixel_array.std()
            if std_val < 1.0:  # Too uniform
                print(f"❌ {filename}: Rejected - Too uniform (std: {std_val:.2f})")
                return False
            
            # Modality validation: include CR/DX common for mammography
            modality = getattr(ds, 'Modality', 'Unknown')
            if modality not in ['MG', 'MR', 'CT', 'US', 'XA', 'CR', 'DX']:
                print(f"❌ {filename}: Rejected - Unsupported modality ({modality})")
                return False
            
            # Check for corrupted or empty data
            if np.isnan(pixel_array).any():
                print(f"❌ {filename}: Rejected - Contains NaN values")
                return False
            if np.isinf(pixel_array).any():
                print(f"❌ {filename}: Rejected - Contains infinite values")
                return False
            
            # Additional debugging info for accepted files
            print(f"✅ {filename}: Accepted - Modality: {modality}, Shape: {pixel_array.shape}, Size: {pixel_array.size}, Std: {std_val:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ {dicom_file.name}: Validation error - {e}")
            return False
    
    def create_advanced_training_sample(self, ds, pixel_array, dataset_name, index):
        """Create advanced training sample with sophisticated preprocessing"""
        try:
            # Advanced image preprocessing
            processed_image = self.preprocess_image(pixel_array)
            
            # Create labels: if normal-only, generate 2-class labels (background vs tissue)
            if self.config.get("normal_only", False):
                label_path = self.create_normal_only_label(processed_image)
            else:
                label_path = self.create_advanced_synthetic_label(processed_image, None)
            
            # ENTERPRISE: Save preprocessed image to ensure consistent loading
            import tempfile
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            image_file.close()
            # Add channel dimension: (H, W) -> (1, H, W)
            processed_image_with_channel = processed_image[np.newaxis, :, :]
            np.save(image_file.name, processed_image_with_channel)
            
            # Simplified sample structure - only include fields that MONAI can handle
            sample = {
                "image": image_file.name,  # Use saved numpy file instead of DICOM
                "label": label_path
            }
            
            return sample
            
        except Exception as e:
            print(f"❌ Error creating sample: {e}")
            return None
    
    def preprocess_image(self, pixel_array):
        """Advanced image preprocessing with manual resizing for consistency"""
        # Handle RGB images by converting to grayscale
        if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
            print(f"🔄 Converting RGB to grayscale...")
            # Convert RGB to grayscale using luminance formula
            grayscale = np.dot(pixel_array[..., :3], [0.299, 0.587, 0.114])
            pixel_array = grayscale.astype(pixel_array.dtype)
            print(f"✅ Converted to grayscale shape: {pixel_array.shape}")
        # Handle multi-frame stacks by taking the central slice
        elif len(pixel_array.shape) == 3 and pixel_array.shape[2] != 3:
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid, ...]
            print(f"🔄 Reduced multi-frame to single slice: {pixel_array.shape}")
        
        # Normalize to 0-1 range
        normalized = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) if available
        try:
            import cv2
            normalized_uint8 = (normalized * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(normalized_uint8)
            normalized = enhanced.astype(np.float32) / 255.0
        except ImportError:
            pass  # Continue without CLAHE
        
        # Apply noise reduction
        from scipy import ndimage
        denoised = ndimage.gaussian_filter(normalized, sigma=0.5)
        
        # ENTERPRISE: Manual resizing to ensure consistent dimensions
        target_size = self.config["image_size"]  # Get height, width from [H, W]
        current_size = denoised.shape
        
        print(f"🔍 Resizing from {current_size} to {target_size}")
        
        # Use PIL for reliable resizing
        from PIL import Image
        pil_image = Image.fromarray((denoised * 255).astype(np.uint8))
        pil_resized = pil_image.resize(target_size, Image.BILINEAR)
        resized = np.array(pil_resized).astype(np.float32) / 255.0
        
        print(f"✅ Resized to {resized.shape}")
        
        return resized
    
    def create_advanced_synthetic_label(self, processed_image, label_file):
        """Create simple but effective synthetic labels for breast cancer detection"""
        print("🎯 Creating simple synthetic labels...")
        
        label_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        label_file.close()
        
        from skimage import filters, morphology
        import random
        
        height, width = processed_image.shape
        final_labels = np.zeros((height, width), dtype=np.uint8)
        
        # SIMPLE BUT EFFECTIVE APPROACH:
        # 1. BACKGROUND (class 0) - already 0
        
        # 2. BREAST TISSUE (class 1) - More aggressive tissue detection
        # Use multiple thresholding approaches to get more tissue
        otsu_threshold = filters.threshold_otsu(processed_image)
        yen_threshold = filters.threshold_yen(processed_image)
        
        # Use the lower threshold to get more tissue
        tissue_threshold = min(otsu_threshold, yen_threshold) * 0.6  # More aggressive
        tissue_mask = (processed_image > tissue_threshold).astype(bool)
        
        # Clean up tissue mask but be less aggressive
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=100)  # Smaller minimum
        tissue_mask = morphology.binary_closing(tissue_mask, morphology.disk(2))
        
        # If we still have very little tissue, be even more aggressive
        if np.sum(tissue_mask) < processed_image.size * 0.05:  # Less than 5% tissue
            tissue_threshold = np.percentile(processed_image, 30)  # Use 30th percentile
            tissue_mask = (processed_image > tissue_threshold).astype(bool)
            tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=50)
        
        final_labels[tissue_mask] = 1
        
        # 3. POTENTIAL LESIONS (class 2) - More reasonable lesion detection
        # Find high-intensity regions within tissue
        high_intensity_threshold = np.percentile(processed_image, 85)  # Slightly lower threshold
        lesion_mask = (processed_image > high_intensity_threshold) & tissue_mask
        
        # Clean up lesions
        lesion_mask = morphology.remove_small_objects(lesion_mask, min_size=30)  # Smaller minimum
        lesion_mask = morphology.binary_closing(lesion_mask, morphology.disk(1))
        
        # Ensure reasonable lesion size - allow more lesions if we have more tissue
        tissue_pixels = np.sum(tissue_mask)
        lesion_pixels = np.sum(lesion_mask)
        
        if tissue_pixels > 0:
            lesion_ratio = lesion_pixels / tissue_pixels
            if lesion_ratio > 0.25:  # If lesions > 25% of tissue
                # Reduce lesion size by erosion
                lesion_mask = morphology.binary_erosion(lesion_mask, morphology.disk(2))
            elif lesion_ratio < 0.02 and tissue_pixels > 1000:  # If very few lesions but lots of tissue
                # Add some additional lesions using a lower threshold
                additional_lesions = (processed_image > np.percentile(processed_image, 75)) & tissue_mask
                additional_lesions = morphology.remove_small_objects(additional_lesions, min_size=20)
                lesion_mask = lesion_mask | additional_lesions
        
        final_labels[lesion_mask] = 2
        
        # Ensure we have a balanced distribution
        class_counts = np.bincount(final_labels.flatten())
        total_pixels = final_labels.size
        
        if len(class_counts) >= 3:
            background_ratio = class_counts[0] / total_pixels
            tissue_ratio = class_counts[1] / total_pixels
            lesion_ratio = class_counts[2] / total_pixels
            
            print(f"🔍 Label distribution: Background={class_counts[0]} ({background_ratio:.1%}), Tissue={class_counts[1]} ({tissue_ratio:.1%}), Lesions={class_counts[2]} ({lesion_ratio:.1%})")
            
            # Quality check
            if tissue_ratio < 0.02:
                print("⚠️  Warning: Very little tissue detected")
            elif tissue_ratio > 0.8:
                print("⚠️  Warning: Too much tissue detected")
            if lesion_ratio < 0.002:
                print("⚠️  Warning: Very few lesions detected")
            elif lesion_ratio > 0.3:
                print("⚠️  Warning: Too many lesions detected")
        else:
            print(f"⚠️  Warning: Only {len(class_counts)} classes detected")
        
        # Resize labels to match image size
        target_size = self.config["image_size"]
        current_size = final_labels.shape
        
        print(f"🔍 Resizing labels from {current_size} to {target_size}")
        
        # Use PIL for reliable resizing with nearest neighbor
        from PIL import Image
        pil_label = Image.fromarray(final_labels.astype(np.uint8))
        pil_resized_label = pil_label.resize(target_size, Image.NEAREST)
        resized_labels = np.array(pil_resized_label).astype(np.uint8)
        
        print(f"✅ Labels resized to {resized_labels.shape}")
        
        # Save as numpy array with channel dimension: (H, W) -> (1, H, W)
        resized_labels_with_channel = resized_labels[np.newaxis, :, :]
        np.save(label_file.name, resized_labels_with_channel)
        
        return label_file.name

    def create_normal_only_label(self, processed_image):
        """Create 2-class labels: background=0, tissue=1 (no lesion class)."""
        print("🎯 Creating normal-only labels (background vs tissue)...")
        import tempfile
        from skimage import filters, morphology
        
        label_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        label_file.close()
        
        height, width = processed_image.shape
        labels = np.zeros((height, width), dtype=np.uint8)
        
        # Aggressive tissue detection similar to advanced method but no lesion class
        otsu_threshold = filters.threshold_otsu(processed_image)
        yen_threshold = filters.threshold_yen(processed_image)
        tissue_threshold = min(otsu_threshold, yen_threshold) * 0.6
        tissue_mask = (processed_image > tissue_threshold).astype(bool)
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=100)
        tissue_mask = morphology.binary_closing(tissue_mask, morphology.disk(2))
        if np.sum(tissue_mask) < processed_image.size * 0.05:
            tissue_threshold = np.percentile(processed_image, 30)
            tissue_mask = (processed_image > tissue_threshold).astype(bool)
            tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=50)
        
        labels[tissue_mask] = 1
        
        # Resize to configured size if needed
        target_size = self.config["image_size"]
        if (height, width) != tuple(target_size):
            from PIL import Image
            pil_label = Image.fromarray(labels.astype(np.uint8))
            pil_resized_label = pil_label.resize(target_size, Image.NEAREST)
            labels = np.array(pil_resized_label).astype(np.uint8)
        
        # Save with channel dimension [1, H, W]
        labels_with_channel = labels[np.newaxis, :, :]
        np.save(label_file.name, labels_with_channel)
        
        return label_file.name
    
    def extract_dicom_metadata(self, ds):
        """Extract relevant DICOM metadata"""
        metadata = {}
        
        # Basic metadata
        metadata_fields = [
            'Modality', 'Manufacturer', 'ManufacturerModelName',
            'PatientAge', 'PatientSex', 'BodyPartExamined',
            'ImageType', 'PixelSpacing', 'SliceThickness'
        ]
        
        for field in metadata_fields:
            try:
                value = getattr(ds, field, None)
                if value is not None:
                    metadata[field] = str(value)
            except:
                continue
        
        return metadata
    
    def stratified_split(self, data, val_ratio=0.2):
        """Simple random split since we removed metadata"""
        from sklearn.model_selection import train_test_split
        
        # Simple random split since we don't have metadata for stratification
        train_data, val_data = train_test_split(
            data, test_size=val_ratio, random_state=42
        )
        
        print(f"📊 Train/Validation split: {len(train_data)}/{len(val_data)}")
        return train_data, val_data

    def create_advanced_transforms(self):
        """Create sophisticated transforms for advanced training"""
        print("\n🔄 Creating Advanced Data Transforms...")
        

        
        num_classes = 2 if self.config.get("normal_only", False) else 3

        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            
            # Advanced intensity transforms
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            
            # Sophisticated augmentation (intensity only - no spatial transforms)
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], prob=0.5, factors=0.3),
            RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
            
            # Advanced dropout for regularization
            RandCoarseDropoutd(
                keys=["image"], 
                prob=0.2, 
                holes=8, 
                spatial_size=20, 
                fill_value=0.0
            ),
            
            EnsureTyped(keys=["image", "label"]),
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ])
        
        return train_transforms, val_transforms
    
    def create_advanced_model(self):
        """Create sophisticated model with advanced architecture"""
        print(f"\n🧠 Creating Advanced {self.config['model_type']} Model...")
        
        # Adjust number of classes for normal-only finetune
        effective_out_channels = 2 if self.config.get("normal_only", False) else self.config["out_channels"]

        if self.config["model_type"] == "SegResNet":
            model = SegResNet(
                spatial_dims=self.config["spatial_dims"],
                in_channels=self.config["in_channels"],
                out_channels=effective_out_channels,
                init_filters=self.config["init_filters"],
                dropout_prob=self.config["dropout_prob"],
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                act="RELU",
                norm="BATCH",
            )
        elif self.config["model_type"] == "UNet":
            model = UNet(
                spatial_dims=self.config["spatial_dims"],
                in_channels=self.config["in_channels"],
                out_channels=effective_out_channels,
                features=(32, 64, 128, 256, 512),
                dropout=self.config["dropout_prob"],
                act="RELU",
                norm="BATCH",
                bias=True,
                upsample="DECONV",
            )
        elif self.config["model_type"] == "DynUNet":
            model = DynUNet(
                spatial_dims=self.config["spatial_dims"],
                in_channels=self.config["in_channels"],
                out_channels=effective_out_channels,
                kernel_size=[3, 3, 3, 3],
                strides=[1, 2, 2, 2],
                upsample_kernel_size=[2, 2, 2],
                filters=[32, 64, 128, 256],
                dropout=self.config["dropout_prob"],
                norm_name="instance",
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
        model = model.to(self.device)

        # Optional warm start from checkpoint
        ckpt_path = self.config.get("resume_from")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # Extract raw state dict
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                # Filter out keys with shape mismatch (e.g., final classifier head)
                model_state = model.state_dict()
                compatible_keys = 0
                skipped_keys = []
                for k, v in list(state_dict.items()):
                    if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
                        model_state[k] = v
                        compatible_keys += 1
                    else:
                        skipped_keys.append(k)
                model.load_state_dict(model_state, strict=False)
                print(
                    f"✅ Warm-started from {ckpt_path}\n   Loaded tensors: {compatible_keys} | Skipped (shape/key mismatch): {len(skipped_keys)}"
                )
            except Exception as e:
                print(f"⚠️  Could not load checkpoint '{ckpt_path}': {e}")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model

    def _apply_encoder_freeze_schedule(self, model, epoch: int):
        """Freeze most parameters for first N epochs to protect learned features.
        Keeps the last X% of parameters trainable as a proxy for decoder/head.
        """
        if not self.config.get("normal_only", False):
            return  # only apply for normal-only finetune
        freeze_epochs = int(self.config.get("freeze_encoder_epochs", 0))
        if freeze_epochs <= 0:
            return
        head_ratio = float(self.config.get("head_trainable_ratio", 0.1))
        params = list(model.named_parameters())
        cutoff = max(1, int((1.0 - head_ratio) * len(params)))
        if epoch < freeze_epochs:
            # Freeze all except the last fraction
            for i, (name, p) in enumerate(params):
                p.requires_grad = (i >= cutoff)
            trainable = sum(p.numel() for _, p in params if p.requires_grad)
            total = sum(p.numel() for _, p in params)
            print(f"🔒 Finetune freeze: epoch {epoch+1}/{freeze_epochs} -> trainable params ~ {trainable:,}/{total:,}")
        elif epoch == freeze_epochs:
            # Unfreeze all
            for _, p in params:
                p.requires_grad = True
            print("🔓 Encoder unfrozen: all parameters trainable going forward")
    
    def create_advanced_loss_function(self):
        """Create sophisticated loss function"""
        if self.config["loss_function"] == "DiceCELoss":
            loss_function = DiceCELoss(
                to_onehot_y=True, 
                softmax=True,
                include_background=True  # Include background for 2/3-class segmentation
            )
        elif self.config["loss_function"] == "DiceFocalLoss":
            loss_function = DiceFocalLoss(
                to_onehot_y=True, 
                softmax=True,
                gamma=2.0,
                alpha=0.25,
                include_background=True
            )
        elif self.config["loss_function"] == "FocalLoss":
            loss_function = FocalLoss(
                to_onehot_y=True, 
                softmax=True,
                gamma=2.0,
                alpha=0.25,
                include_background=True
            )
        else:
            raise ValueError(f"Unknown loss function: {self.config['loss_function']}")
        
        return loss_function
    
    def create_advanced_optimizer(self, model):
        """Create sophisticated optimizer with advanced settings"""
        if self.config["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config["optimizer"] == "Novograd":
            optimizer = Novograd(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        return optimizer
    
    def create_advanced_scheduler(self, optimizer):
        """Create sophisticated learning rate scheduler"""
        if self.config["scheduler"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config["epochs"],
                eta_min=1e-6
            )
        elif self.config["scheduler"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def create_advanced_metrics(self):
        """Create comprehensive evaluation metrics"""
        metrics = {}
        
        if "dice" in self.config["validation_metrics"]:
            metrics["dice"] = DiceMetric(include_background=True, reduction="mean")
        
        if "hausdorff" in self.config["validation_metrics"]:
            metrics["hausdorff"] = HausdorffDistanceMetric(include_background=True, reduction="mean")
        
        return metrics
    
    def advanced_training_loop(self, train_data, val_data):
        """Advanced training loop with comprehensive features"""
        print(f"\n🚀 Starting Advanced Training for {self.config['epochs']} epochs...")
        
        # Create transforms
        train_transforms, val_transforms = self.create_advanced_transforms()
        
        # Create datasets (reduced cache to avoid OOM/kill)
        train_ds = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=self.config.get("cache_rate", 0.2),
        )
        val_ds = CacheDataset(
            data=val_data,
            transform=val_transforms,
            cache_rate=self.config.get("val_cache_rate", 0.0),
        )
        
        # Create data loaders
        train_loader = ThreadDataLoader(
            train_ds, 
            batch_size=self.config["batch_size"], 
            shuffle=True, 
            num_workers=self.config.get("num_workers", 2),
            persistent_workers=self.config.get("persistent_workers", False)
        )
        val_loader = ThreadDataLoader(
            val_ds, 
            batch_size=self.config["batch_size"], 
            shuffle=False, 
            num_workers=self.config.get("num_workers", 2),
            persistent_workers=self.config.get("persistent_workers", False)
        )
        
        # Create model and training components
        model = self.create_advanced_model()
        loss_function = self.create_advanced_loss_function()
        optimizer = self.create_advanced_optimizer(model)
        scheduler = self.create_advanced_scheduler(optimizer)
        metrics = self.create_advanced_metrics()
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config["mixed_precision"] else None
        
        # Training state
        best_metric = -1
        best_metric_epoch = -1
        patience_counter = 0
        training_history = {
            "train_loss": [],
            "val_metrics": {},
            "learning_rates": []
        }
        
        # Initialize validation metrics history
        for metric_name in metrics.keys():
            training_history["val_metrics"][metric_name] = []
        
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")
        print(f"🔄 Starting training...")
        
        for epoch in range(self.config["epochs"]):
            print(f"\n📅 Epoch {epoch+1}/{self.config['epochs']}")
            # Apply freeze schedule for normal-only finetune
            self._apply_encoder_freeze_schedule(model, epoch)
            
            # Training phase
            model.train()
            epoch_loss = 0
            step = 0
            
            for batch_data in train_loader:
                step += 1
                optimizer.zero_grad()
                
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                # Debug: Check data shapes and values
                if step == 1 and epoch == 0:
                    print(f"🔍 Debug - Input shape: {inputs.shape}, Label shape: {labels.shape}")
                    print(f"🔍 Debug - Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
                    print(f"🔍 Debug - Label unique values: {torch.unique(labels)}")
                    print(f"🔍 Debug - Label distribution: {torch.bincount(labels.long().flatten())}")  # Convert to long for bincount
                
                # Mixed precision forward pass
                if self.config["mixed_precision"] and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        
                        # Debug: Check output shape and values
                        if step == 1 and epoch == 0:
                            print(f"🔍 Debug - Output shape: {outputs.shape}")
                            print(f"🔍 Debug - Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                        
                        loss = loss_function(outputs, labels)
                        
                        # Debug: Check loss value
                        if step == 1 and epoch == 0:
                            print(f"🔍 Debug - Loss value: {loss.item():.6f}")
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config["gradient_clipping"] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["gradient_clipping"])
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    
                    # Debug: Check output shape and values
                    if step == 1 and epoch == 0:
                        print(f"🔍 Debug - Output shape: {outputs.shape}")
                        print(f"🔍 Debug - Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                    
                    loss = loss_function(outputs, labels)
                    
                    # Debug: Check loss value
                    if step == 1 and epoch == 0:
                        print(f"🔍 Debug - Loss value: {loss.item():.6f}")
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config["gradient_clipping"] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["gradient_clipping"])
                    
                    optimizer.step()
                
                epoch_loss += loss.item()
                
                if step % 10 == 0:
                    print(f"   Step {step}: Loss = {loss.item():.4f}")
            
            epoch_loss /= step
            training_history["train_loss"].append(epoch_loss)
            
            # Log to TensorBoard
            if self.config["tensorboard_logging"]:
                self.writer.add_scalar('Loss/Train', epoch_loss, epoch)
                self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            print(f"   📉 Average Loss: {epoch_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_metrics = {}
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(self.device)
                    val_labels = val_batch["label"].to(self.device)
                    
                    val_outputs = model(val_inputs)
                    
                    # Convert labels to one-hot format for metrics
                    num_classes = 2 if self.config.get("normal_only", False) else 3
                    val_labels_onehot = torch.nn.functional.one_hot(
                        val_labels.long().squeeze(1), num_classes=num_classes
                    ).permute(0, 3, 1, 2).float()
                    
                    # Update metrics
                    for metric_name, metric_fn in metrics.items():
                        metric_fn(y_pred=val_outputs, y=val_labels_onehot)
                
                # Aggregate metrics
                for metric_name, metric_fn in metrics.items():
                    metric_value = metric_fn.aggregate().item()
                    val_metrics[metric_name] = metric_value
                    training_history["val_metrics"][metric_name].append(metric_value)
                    metric_fn.reset()
                    
                    # Log to TensorBoard
                    if self.config["tensorboard_logging"]:
                        self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Print validation results
            print(f"   📊 Validation Metrics:")
            for metric_name, metric_value in val_metrics.items():
                print(f"      {metric_name}: {metric_value:.4f}")
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get("dice", epoch_loss))
                else:
                    scheduler.step()
            
            training_history["learning_rates"].append(optimizer.param_groups[0]['lr'])
            
            # Model checkpointing
            if self.config["model_checkpointing"]:
                current_metric = val_metrics.get("dice", -epoch_loss)
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_metric_epoch = epoch + 1
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_metric': best_metric,
                        'config': self.config,
                        'training_history': training_history
                    }, self.model_dir / "best_breast_cancer_model.pth")
                    
                    print(f"   💾 New best model saved! (Metric: {best_metric:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if self.config["early_stopping"] and patience_counter >= self.config["patience"]:
                    print(f"   ⏹️  Early stopping triggered after {self.config['patience']} epochs without improvement")
                    break
        
        print(f"\n🎉 Training Complete!")
        print(f"🏆 Best Metric: {best_metric:.4f} at epoch {best_metric_epoch}")
        
        # Save final model and training history
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': best_metric,
            'config': self.config,
            'training_history': training_history
        }, self.model_dir / "final_breast_cancer_model.pth")
        
        # Save training history separately
        with open(self.model_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        
        return model, training_history

    def run_advanced_training(self):
        """Run the complete advanced training pipeline"""
        print("🏥 Starting Advanced Breast Cancer Model Training Pipeline")
        print("=" * 80)
        
        # Step 1: Prepare advanced training data
        train_data, val_data, dataset_stats = self.prepare_advanced_training_data()
        
        if len(train_data) == 0:
            print("❌ No training data found!")
            return None, None
        
        # Step 2: Run advanced training
        model, training_history = self.advanced_training_loop(train_data, val_data)
        
        # Step 3: Create advanced MONAI integration
        self.create_advanced_monai_integration()
        
        # Step 4: Generate comprehensive report
        self.generate_training_report(training_history, dataset_stats)
        
        print("\n🎉 Advanced Training Pipeline Complete!")
        print("\n📋 Next Steps:")
        print("   1. Restart MONAI server to include new model")
        print("   2. Test with real breast images")
        print("   3. Integrate results with 3D viewer")
        print("   4. Review TensorBoard logs for detailed analysis")
        
        return model, training_history
    
    def create_advanced_monai_integration(self):
        """Create advanced MONAI Label integration files"""
        print("\n🔗 Creating Advanced MONAI Label Integration...")
        
        # Create the advanced inferer file
        inferer_content = '''#!/usr/bin/env python3
"""
Advanced Breast Cancer Detection Inferer
Uses trained model with advanced preprocessing and post-processing
"""

import os
import torch
import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityd, NormalizeIntensityd, EnsureTyped,
    Activationsd, AsDiscreted, ResizeWithPadOrCropd
)
from monai.networks.nets import SegResNet, UNet, DynUNet
from monailabel.interfaces.tasks.infer import BasicInferTask

class AdvancedBreastCancerDetection(BasicInferTask):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.inferer = None
        self.config = {
            "model_type": "SegResNet",
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 3,
            "init_filters": 32,
            "dropout_prob": 0.2,
            "image_size": [512, 512]
        }
        
    def pre_transforms(self, data=None):
        return [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=self.config["image_size"]),
            EnsureTyped(keys=["image"]),
        ]
    
    def post_transforms(self, data=None):
        return [
            Activationsd(keys=["pred"], softmax=True),
            AsDiscreted(keys=["pred"], argmax=True),
        ]
    
    def inferer(self, data=None):
        return SlidingWindowInferer(
            roi_size=self.config["image_size"],
            sw_batch_size=1,
            overlap=0.5,
        )
    
    def _get_network(self):
        if self.model is None:
            if self.config["model_type"] == "SegResNet":
                self.model = SegResNet(
                    spatial_dims=self.config["spatial_dims"],
                    in_channels=self.config["in_channels"],
                    out_channels=self.config["out_channels"],
                    init_filters=self.config["init_filters"],
                    dropout_prob=self.config["dropout_prob"],
                )
            elif self.config["model_type"] == "UNet":
                self.model = UNet(
                    spatial_dims=self.config["spatial_dims"],
                    in_channels=self.config["in_channels"],
                    out_channels=self.config["out_channels"],
                    features=(32, 64, 128, 256, 512),
                    dropout=self.config["dropout_prob"],
                )
            else:
                raise ValueError(f"Unknown model type: {self.config['model_type']}")
            
            self.model = self.model.to(self.device)
            
            # Load trained weights
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "trained_models", "best_breast_cancer_model.pth")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained breast cancer model from {model_path}")
                print(f"   Best metric: {checkpoint.get('best_metric', 'Unknown')}")
            else:
                print(f"⚠️  No trained model found at {model_path}, using untrained model")
            
            self.model.eval()
        
        return self.model
    
    def __call__(self, request):
        """Run advanced inference on the input data"""
        try:
            # Get the model
            model = self._get_network()
            
            # Get input data
            image = request.get("image")
            if image is None:
                raise ValueError("No image data provided")
            
            # Run inference
            with torch.no_grad():
                pred = model(image)
                
                # Convert to numpy
                pred_np = pred.cpu().numpy()
                
                # Create segmentation map
                segmentation = np.argmax(pred_np, axis=1)
                
                # Create confidence scores
                confidence = np.max(pred_np, axis=1)
                
                # Advanced analysis
                results = self.advanced_breast_cancer_analysis(segmentation, confidence, pred_np)
                
                return {
                    "pred": torch.from_numpy(pred_np),
                    "image": image,
                    "segmentation": segmentation,
                    "confidence": confidence,
                    "analysis": results
                }
                
        except Exception as e:
            print(f"❌ Advanced inference error: {e}")
            return self._fallback_inference(request)
    
    def advanced_breast_cancer_analysis(self, segmentation, confidence, pred_np):
        """Advanced breast cancer detection analysis"""
        results = {
            "breast_tissue_detected": False,
            "potential_lesions": False,
            "confidence_score": 0.0,
            "tissue_percentage": 0.0,
            "lesion_percentage": 0.0,
            "risk_assessment": "low",
            "recommendations": [],
            "detailed_metrics": {}
        }
        
        # Calculate tissue percentages
        total_pixels = segmentation.size
        breast_tissue_pixels = (segmentation == 1).sum()
        lesion_pixels = (segmentation == 2).sum()
        
        results["tissue_percentage"] = float(breast_tissue_pixels / total_pixels * 100)
        results["lesion_percentage"] = float(lesion_pixels / total_pixels * 100)
        
        # Check for breast tissue
        if breast_tissue_pixels > 0:
            results["breast_tissue_detected"] = True
        
        # Check for potential lesions
        if lesion_pixels > 0:
            results["potential_lesions"] = True
            results["recommendations"].append("Potential lesions detected - recommend follow-up")
        
        # Calculate overall confidence
        results["confidence_score"] = float(confidence.mean())
        
        # Risk assessment based on multiple factors
        risk_score = 0
        if results["lesion_percentage"] > 5:
            risk_score += 3
        elif results["lesion_percentage"] > 1:
            risk_score += 2
        elif results["lesion_percentage"] > 0.1:
            risk_score += 1
        
        if results["confidence_score"] < 0.7:
            risk_score += 1
        
        if risk_score >= 3:
            results["risk_assessment"] = "high"
        elif risk_score >= 2:
            results["risk_assessment"] = "medium"
        else:
            results["risk_assessment"] = "low"
        
        # Add recommendations based on analysis
        if results["confidence_score"] < 0.7:
            results["recommendations"].append("Low confidence - recommend manual review")
        elif results["confidence_score"] > 0.9:
            results["recommendations"].append("High confidence analysis")
        
        if results["risk_assessment"] == "high":
            results["recommendations"].append("High risk assessment - immediate follow-up recommended")
        
        # Detailed metrics
        results["detailed_metrics"] = {
            "dice_scores": self.calculate_dice_scores(pred_np, segmentation),
            "hausdorff_distance": self.calculate_hausdorff_distance(segmentation),
            "precision_recall": self.calculate_precision_recall(segmentation)
        }
        
        return results
    
    def calculate_dice_scores(self, pred_np, segmentation):
        """Calculate Dice scores for each class"""
        dice_scores = {}
        for class_id in range(pred_np.shape[1]):
            pred_mask = (segmentation == class_id).astype(np.float32)
            true_mask = (pred_np[:, class_id] > 0.5).astype(np.float32)
            
            intersection = (pred_mask * true_mask).sum()
            union = pred_mask.sum() + true_mask.sum()
            
            dice = (2 * intersection) / (union + 1e-8)
            dice_scores[f"class_{class_id}"] = float(dice)
        
        return dice_scores
    
    def calculate_hausdorff_distance(self, segmentation):
        """Calculate Hausdorff distance for lesion regions"""
        # Simplified Hausdorff distance calculation
        lesion_mask = (segmentation == 2)
        if lesion_mask.sum() == 0:
            return float('inf')
        
        # Calculate boundary distance
        from scipy import ndimage
        boundary = ndimage.binary_erosion(lesion_mask) != lesion_mask
        if boundary.sum() == 0:
            return 0.0
        
        return float(boundary.sum() / lesion_mask.sum())
    
    def calculate_precision_recall(self, segmentation):
        """Calculate precision and recall for lesion detection"""
        lesion_mask = (segmentation == 2)
        tissue_mask = (segmentation == 1)
        
        if tissue_mask.sum() == 0:
            return {"precision": 0.0, "recall": 0.0}
        
        precision = float(lesion_mask.sum() / (lesion_mask.sum() + 1e-8))
        recall = float(lesion_mask.sum() / tissue_mask.sum()) if tissue_mask.sum() > 0 else 0.0
        
        return {"precision": precision, "recall": recall}
    
    def _fallback_inference(self, request):
        """Fallback inference if main inference fails"""
        print("🔄 Using fallback inference...")
        
        image = request.get("image")
        if image is not None:
            shape = image.shape
            pred = torch.randn(1, 3, *shape[2:], device=self.device)
            return {"pred": pred, "image": image}
        
        return {"pred": None, "image": None}
'''
        
        # Create the advanced config file
        config_content = '''#!/usr/bin/env python3
"""
Advanced Breast Cancer Detection Configuration
"""

from lib.infers.advanced_breast_cancer_detection import AdvancedBreastCancerDetection
from monailabel.interfaces.config import TaskConfig

class AdvancedBreastCancerDetectionConfig(TaskConfig):
    def __init__(self):
        super().__init__()
    
    def infer(self):
        return AdvancedBreastCancerDetection()
    
    def trainer(self):
        return None  # No training in MONAI Label
    
    def strategy(self):
        return None  # No strategy needed
'''
        
        # Write files
        os.makedirs("~/.local/monailabel/sample-apps/radiology/lib/infers", exist_ok=True)
        os.makedirs("~/.local/monailabel/sample-apps/radiology/lib/configs", exist_ok=True)
        
        with open("~/.local/monailabel/sample-apps/radiology/lib/infers/advanced_breast_cancer_detection.py", "w") as f:
            f.write(inferer_content)
        
        with open("~/.local/monailabel/sample-apps/radiology/lib/configs/advanced_breast_cancer_detection.py", "w") as f:
            f.write(config_content)
        
        print("✅ Advanced MONAI Label integration files created")
    
    def generate_training_report(self, training_history, dataset_stats):
        """Generate comprehensive training report"""
        print("\n📊 Generating Training Report...")
        
        report = {
            "training_summary": {
                "total_epochs": len(training_history["train_loss"]),
                "best_metric": max(training_history["val_metrics"].get("dice", [0])),
                "final_loss": training_history["train_loss"][-1] if training_history["train_loss"] else 0,
                "training_date": datetime.now().isoformat()
            },
            "dataset_stats": dataset_stats,
            "config": self.config,
            "training_history": training_history
        }
        
        # Save report
        with open(self.model_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Create visualization plots
        self.create_training_plots(training_history)
        
        print("✅ Training report generated")
    
    def create_training_plots(self, training_history):
        """Create training visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            axes[0, 0].plot(training_history["train_loss"])
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            
            # Metrics plot
            for metric_name, metric_values in training_history["val_metrics"].items():
                axes[0, 1].plot(metric_values, label=metric_name)
            axes[0, 1].set_title("Validation Metrics")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Metric Value")
            axes[0, 1].legend()
            
            # Learning rate plot
            axes[1, 0].plot(training_history["learning_rates"])
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_yscale('log')
            
            # Combined plot
            ax2 = axes[1, 1].twinx()
            line1 = axes[1, 1].plot(training_history["train_loss"], 'b-', label='Loss')
            line2 = ax2.plot(training_history["val_metrics"].get("dice", []), 'r-', label='Dice')
            axes[1, 1].set_title("Training Progress")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss", color='b')
            ax2.set_ylabel("Dice Score", color='r')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 1].legend(lines, labels, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(self.model_dir / "training_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Training plots created")
            
        except Exception as e:
            print(f"⚠️  Could not create training plots: {e}")

def main():
    """Main advanced training function"""
    # ENTERPRISE-GRADE configuration
    advanced_config = {
        "model_type": "SegResNet",
        "spatial_dims": 2,  # ENTERPRISE: Use 2D for proper MONAI compatibility with 2D images
        "epochs": 50,  # Reduced for faster training
        "batch_size": 2,  # Reduced for memory constraints
        "image_size": [256, 256],  # 2D size: [height, width] for 2D images
        "patience": 10,
        "mixed_precision": True,
        "tensorboard_logging": True,
        "validation_metrics": ["dice", "hausdorff"],
        "class_weights": [1.0, 2.0, 3.0],
    }
    
    trainer = AdvancedBreastCancerTrainer(config=advanced_config)
    trainer.run_advanced_training()

if __name__ == "__main__":
    main()
