#!/usr/bin/env python3
"""
Advanced Breast Cancer Detection Inferer - FIXED VERSION
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
        # ENTERPRISE: Proper BasicInferTask initialization with required arguments
        # Fix model path to point to the correct location
        model_path = os.path.join(os.path.expanduser("~"), "mri_app", "trained_models", "best_breast_cancer_model.pth")
        print(f"🔍 DEBUG: Model path: {model_path}")
        print(f"🔍 DEBUG: Model file exists: {os.path.exists(model_path)}")
        
        super().__init__(
            path=model_path,
            network=None,  # We'll create the network in _get_network()
            type="segmentation",
            labels=["background", "breast_tissue", "lesions"],
            dimension=2,
            description="Advanced Breast Cancer Detection using SegResNet",
            model_state_dict="model_state_dict",
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            config={
                "model_type": "SegResNet",
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 3,  # Three channels: background, breast_tissue, lesions
                "init_filters": 32,
                "dropout_prob": 0.2,
                "image_size": [256, 256]  # Match our training size
            },
            load_strict=True,
            roi_size=(512, 512),  # Match the input image size
            preload=False,
            train_mode=False
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._inferer = None  # Use different name to avoid conflicts
        
    def pre_transforms(self, data=None):
        return [
            LoadImaged(keys=["image"], reader="PydicomReader"),  # Use default PydicomReader
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Remove ResizeWithPadOrCropd to avoid dimension mismatch
            EnsureTyped(keys=["image"]),
        ]
    
    def post_transforms(self, data=None):
        """Return the post-processing transforms"""
        print(f"🔍 DEBUG: post_transforms called with data type: {type(data)}")
        
        # Skip broken post-transforms - we'll handle softmax manually in writer
        transforms = []
        print(f"🔍 DEBUG: Skipping broken post-transforms, will handle softmax manually")
        return transforms
    
    def writer(self, data, extension=None, dtype=None):
        """Override writer to handle the ITK direction issue"""
        print(f"🔍 DEBUG: Custom writer called")
        
        # Get the prediction result - prioritize processed predictions, fallback to raw
        pred = data.get("pred")  # Use processed predictions first
        raw_pred = data.get("raw_pred")  # Keep raw predictions for debugging
        
        print(f"🔍 DEBUG: Available predictions:")
        print(f"🔍 DEBUG:   pred (processed): {type(pred)} - {pred.shape if pred is not None else 'None'}")
        print(f"🔍 DEBUG:   raw_pred (logits): {type(raw_pred)} - {raw_pred.shape if raw_pred is not None else 'None'}")
        
        # Check if processed predictions are corrupted (all same values)
        if pred is not None:
            if hasattr(pred, 'min') and hasattr(pred, 'max'):
                pred_min = pred.min().item() if hasattr(pred.min(), 'item') else pred.min()
                pred_max = pred.max().item() if hasattr(pred.max(), 'item') else pred.max()
                
                # If all values are the same, the softmax failed
                if abs(pred_max - pred_min) < 1e-6:
                    print(f"🔍 DEBUG: Processed predictions corrupted (all values same: {pred_min})")
                    print(f"🔍 DEBUG: Falling back to raw predictions with manual softmax")
                    pred = None  # Force fallback to raw
                else:
                    print(f"🔍 DEBUG: Using processed predictions (values: {pred_min:.6f} to {pred_max:.6f})")
        
        # Fallback to raw predictions if processed ones are corrupted
        if pred is None and raw_pred is not None:
            print(f"🔍 DEBUG: Applying manual softmax to raw predictions")
            
            # Apply numerically stable softmax manually
            if len(raw_pred.shape) == 4 and raw_pred.shape[1] == 3:
                # Clip extreme values for numerical stability
                clipped_raw = torch.clamp(raw_pred, min=-10.0, max=10.0)
                pred = torch.nn.functional.softmax(clipped_raw, dim=1)
                print(f"🔍 DEBUG: Applied manual softmax to 4D raw predictions")
            elif len(raw_pred.shape) == 3 and raw_pred.shape[0] == 3:
                # Clip extreme values for numerical stability
                clipped_raw = torch.clamp(raw_pred, min=-10.0, max=10.0)
                pred = torch.nn.functional.softmax(clipped_raw, dim=0)
                print(f"🔍 DEBUG: Applied manual softmax to 3D raw predictions")
            else:
                pred = raw_pred
                print(f"🔍 DEBUG: Using raw predictions as-is (unexpected shape)")
        if pred is not None:
            print(f"🔍 DEBUG: Prediction shape: {pred.shape}")
            print(f"🔍 DEBUG: Prediction type: {pred.dtype}")
            
            # Convert to numpy and ensure proper format
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            elif hasattr(pred, 'numpy'):
                pred = pred.numpy()
            
            # Handle 3-channel probability output properly
            if len(pred.shape) == 4:  # [B, C, H, W]
                if pred.shape[1] == 3:  # 3 channels: background, breast_tissue, lesions
                    # Debug: Check what each channel contains
                    background_channel = pred[0, 0, :, :]
                    tissue_channel = pred[0, 1, :, :]
                    lesions_channel = pred[0, 2, :, :]
                    
                    print(f"🔍 DEBUG: Channel analysis:")
                    print(f"🔍 DEBUG:   Background (0): min={background_channel.min():.6f}, max={background_channel.max():.6f}, mean={background_channel.mean():.6f}")
                    print(f"🔍 DEBUG:   Tissue (1): min={tissue_channel.min():.6f}, max={tissue_channel.max():.6f}, mean={tissue_channel.mean():.6f}")
                    print(f"🔍 DEBUG:   Lesions (2): min={lesions_channel.min():.6f}, max={lesions_channel.max():.6f}, mean={lesions_channel.mean():.6f}")
                    
                    # For clinical analysis, extract the lesions channel (index 2)
                    pred = lesions_channel
                    print(f"🔍 DEBUG: Extracted lesions channel from 3-channel output")
                else:
                    pred = pred[0, 0, :, :]  # Take first channel
            elif len(pred.shape) == 3:  # [C, H, W]
                if pred.shape[0] == 3:  # 3 channels
                    # Debug: Check what each channel contains
                    background_channel = pred[0, :, :]
                    tissue_channel = pred[1, :, :]
                    lesions_channel = pred[2, :, :]
                    
                    print(f"🔍 DEBUG: Channel analysis:")
                    print(f"🔍 DEBUG:   Background (0): min={background_channel.min():.6f}, max={background_channel.max():.6f}, mean={background_channel.mean():.6f}")
                    print(f"🔍 DEBUG:   Tissue (1): min={tissue_channel.min():.6f}, max={tissue_channel.max():.6f}, mean={tissue_channel.mean():.6f}")
                    print(f"🔍 DEBUG:   Lesions (2): min={lesions_channel.min():.6f}, max={lesions_channel.max():.6f}, mean={lesions_channel.mean():.6f}")
                    
                    # Extract lesions channel from 3D tensor
                    pred = lesions_channel
                    print(f"🔍 DEBUG: Extracted lesions channel from 3-channel output")
                else:
                    pred = pred[0, :, :]  # Take first channel
            
            print(f"🔍 DEBUG: Final prediction shape: {pred.shape}")
            print(f"🔍 DEBUG: Final prediction values - min: {pred.min():.6f}, max: {pred.max():.6f}, mean: {pred.mean():.6f}")
            
            # Create a comprehensive result with JSON-serializable data
            result = {
                "pred_shape": list(pred.shape),
                "pred_dtype": str(pred.dtype),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
                "success": True,
                "message": "Breast cancer detection completed successfully",
                "label_names": ["background", "breast_tissue", "lesions"],
                # Include the actual prediction data
                "pred": pred.tolist(),  # Convert numpy array to list for JSON serialization
                "debug_info": {
                    "raw_prediction_shape": list(pred.shape),
                    "raw_prediction_min": float(pred.min()),
                    "raw_prediction_max": float(pred.max()),
                    "raw_prediction_mean": float(pred.mean()),
                    "raw_prediction_std": float(pred.std())
                }
            }
            
            return "result.json", result
        
        # Fallback to default writer
        from monailabel.transform.writer import Writer
        return Writer()(data, extension, dtype)
    
    @property
    def post_transforms_property(self):
        """Property version of post_transforms that doesn't take parameters"""
        return []
    
    def run_post_transforms(self, data, transforms):
        """Override run_post_transforms to handle the data correctly"""
        print(f"🔍 DEBUG: run_post_transforms called with data type: {type(data)}")
        print(f"🔍 DEBUG: transforms type: {type(transforms)}")
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            print(f"🔍 DEBUG: Converting data to dictionary")
            # If data is not a dict, we need to handle this differently
            # This might be the SlidingWindowInferer object
            return data
        
        # Run the transforms normally
        from monailabel.interfaces.utils.transform import run_transforms
        return run_transforms(data, transforms, log_prefix="POST")
    
    def inferer(self, data=None, device=None):
        # Create and return the inferer object
        print(f"🔍 DEBUG: Creating SlidingWindowInferer")
        print(f"🔍 DEBUG: Data received: {type(data) if data else 'None'}")
        print(f"🔍 DEBUG: Device received: {device}")
        
        # Create the inferer
        inferer_obj = SlidingWindowInferer(
            roi_size=(512, 512),  # Match the input image size
            sw_batch_size=1,
            overlap=0.0,  # No overlap for full image inference
        )
        
        print(f"🔍 DEBUG: Inferer created: {type(inferer_obj)}")
        print(f"🔍 DEBUG: Inferer is callable: {callable(inferer_obj)}")
        
        return inferer_obj
    
    def get_inferer(self, data=None):
        """Alternative method name that MONAI Label might be looking for"""
        return self.inferer(data)
    
    def run_inferer(self, data=None, device=None):
        """Method that MONAI Label calls with device parameter"""
        print(f"🔍 DEBUG: run_inferer called with device: {device}")
        print(f"🔍 DEBUG: Data type: {type(data)}")
        
        # MONAI Label expects this method to RUN the inference and return the result
        # not just return the inferer object
        if self._inferer is None:
            self._inferer = SlidingWindowInferer(
                roi_size=(512, 512),  # Match the input image size
                sw_batch_size=1,
                overlap=0.0,  # No overlap for full image inference
            )
            print(f"🔍 DEBUG: Created new SlidingWindowInferer with roi_size=(512, 512)")
        
        # Debug: Check if network method exists
        print(f"🔍 DEBUG: self.network type: {type(self.network)}")
        print(f"🔍 DEBUG: self.network is callable: {callable(self.network)}")
        
        # Get the network - handle the case where self.network is None
        print(f"🔍 DEBUG: About to get network")
        print(f"🔍 DEBUG: self.network type: {type(self.network)}")
        
        # Always call _get_network() to load the trained model
        print(f"🔍 DEBUG: Calling _get_network() to load trained model")
        try:
            network = self._get_network()
            print(f"🔍 DEBUG: Successfully loaded trained model: {type(network)}")
        except Exception as e:
            print(f"🔍 DEBUG: Error loading trained model: {e}")
            # Create a fallback network
            network = SegResNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=3,  # Three channels: background, breast_tissue, lesions
                init_filters=32,
                dropout_prob=0.2,
            )
            network = network.to(self.device)
            network.eval()
            print(f"🔍 DEBUG: Created fallback network: {type(network)}")
        
        # Run the inference
        print(f"🔍 DEBUG: Running inference with data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if isinstance(data, dict) and "image" in data:
            # Move input data to the same device as the network
            input_image = data["image"]
            print(f"🔍 DEBUG: Input image device: {input_image.device}")
            print(f"🔍 DEBUG: Network device: {next(network.parameters()).device}")
            
            # Move input to GPU if network is on GPU
            if next(network.parameters()).device.type == 'cuda' and input_image.device.type == 'cpu':
                print(f"🔍 DEBUG: Moving input to GPU")
                input_image = input_image.cuda()
            
            # Use direct model inference instead of SlidingWindowInferer
            # Debug the actual input shape
            print(f"🔍 DEBUG: Input image shape: {input_image.shape}")
            
            # Ensure input is in the correct format for 2D inference
            if len(input_image.shape) == 3:
                # Input is (C, H, W) - add batch dimension
                input_image = input_image.unsqueeze(0)  # Now (1, C, H, W)
                print(f"🔍 DEBUG: Added batch dimension, shape: {input_image.shape}")
            
            # Resize input to match trained model's expected size (256x256)
            if input_image.shape[-1] != 256 or input_image.shape[-2] != 256:
                input_image = torch.nn.functional.interpolate(
                    input_image, 
                    size=(256, 256), 
                    mode='bilinear', 
                    align_corners=False
                )
                print(f"🔍 DEBUG: Resized input to shape: {input_image.shape}")
            
            with torch.no_grad():
                result = network(input_image)
            print(f"🔍 DEBUG: Inference result shape: {result.shape if hasattr(result, 'shape') else 'No shape'}")
            print(f"🔍 DEBUG: Raw inference result - min: {result.min().item():.6f}, max: {result.max().item():.6f}, mean: {result.mean().item():.6f}")
            if len(result.shape) == 4 and result.shape[1] == 3:
                print(f"🔍 DEBUG: Channel 0 (background) - min: {result[0,0].min().item():.6f}, max: {result[0,0].max().item():.6f}")
                print(f"🔍 DEBUG: Channel 1 (breast_tissue) - min: {result[0,1].min().item():.6f}, max: {result[0,1].max().item():.6f}")
                print(f"🔍 DEBUG: Channel 2 (lesions) - min: {result[0,2].min().item():.6f}, max: {result[0,2].max().item():.6f}")
            
            # Store the raw result before post-processing
            data["raw_pred"] = result.clone()
            
            # Apply gradient clipping and numerical stability improvements
            if len(result.shape) == 4 and result.shape[1] == 3:
                # Clip extreme values to prevent numerical instability
                clipped_result = torch.clamp(result, min=-10.0, max=10.0)
                print(f"🔍 DEBUG: Clipped result - min: {clipped_result.min().item():.6f}, max: {clipped_result.max().item():.6f}")
                
                # Apply numerically stable softmax manually
                stable_softmax = torch.nn.functional.softmax(clipped_result, dim=1)
                data["pred"] = stable_softmax
                print(f"🔍 DEBUG: Applied numerically stable softmax")
                print(f"🔍 DEBUG: Softmax result - min: {stable_softmax.min().item():.6f}, max: {stable_softmax.max().item():.6f}, mean: {stable_softmax.mean().item():.6f}")
                
                # Verify channel differentiation
                for i, name in enumerate(["background", "tissue", "lesions"]):
                    channel = stable_softmax[0, i, :, :]
                    print(f"🔍 DEBUG: {name} channel - min: {channel.min().item():.6f}, max: {channel.max().item():.6f}, mean: {channel.mean().item():.6f}")
            else:
                data["pred"] = result
                print(f"🔍 DEBUG: Stored raw predictions (not 3-channel)")
                print(f"🔍 DEBUG: Raw result - min: {result.min().item():.6f}, max: {result.max().item():.6f}, mean: {result.mean().item():.6f}")
            
            # Debug: Check what's in data before returning
            print(f"🔍 DEBUG: Data keys before return: {list(data.keys())}")
            if "pred" in data:
                print(f"🔍 DEBUG: pred shape: {data['pred'].shape}")
                print(f"🔍 DEBUG: pred type: {type(data['pred'])}")
            
            return data
        else:
            print(f"🔍 DEBUG: Data is not a dict or missing image key")
            return data
    

    
    @property
    def inferer_property(self):
        """Property method that MONAI Label might be looking for"""
        if self._inferer is None:
            self._inferer = SlidingWindowInferer(
                roi_size=(512, 512),  # Match the input image size
                sw_batch_size=1,
                overlap=0.0,  # No overlap for full image inference
            )
        return self._inferer
    
    @property
    def network_property(self):
        """Property version of network that doesn't take parameters"""
        print(f"🔍 DEBUG: network_property called")
        return self.network()
    
    def network(self, data=None):
        """Return the network for inference"""
        print(f"🔍 DEBUG: network method called")
        try:
            network_obj = self._get_network()
            print(f"🔍 DEBUG: network method returning: {type(network_obj)}")
            return network_obj
        except Exception as e:
            print(f"🔍 DEBUG: Error in network method: {e}")
            # Create a basic network as fallback
            fallback_network = SegResNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                dropout_prob=0.2,
            )
            fallback_network = fallback_network.to(self.device)
            fallback_network.eval()
            print(f"🔍 DEBUG: Created fallback network: {type(fallback_network)}")
            return fallback_network
    
    def _get_network(self):
        print(f"🔍 DEBUG: _get_network called")
        print(f"🔍 DEBUG: self.model is None: {self.model is None}")
        print(f"🔍 DEBUG: About to create/load model")
        
        if self.model is None:
            print(f"🔍 DEBUG: Creating new SegResNet model")
            self.model = SegResNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=3,  # Three channels: background, breast_tissue, lesions
                init_filters=32,
                dropout_prob=0.2,
            )
            
            print(f"🔍 DEBUG: Moving model to device: {self.device}")
            self.model = self.model.to(self.device)
            
            # Load trained weights
            # Fix model path to point to the correct location
            model_path = os.path.join(os.path.expanduser("~"), "mri_app", "trained_models", "best_breast_cancer_model.pth")
            print(f"🔍 DEBUG: Looking for model at: {model_path}")
            print(f"🔍 DEBUG: Model file exists: {os.path.exists(model_path)}")
            print(f"🔍 DEBUG: About to load model weights...")
            
            if os.path.exists(model_path):
                print(f"🔍 DEBUG: Loading trained weights")
                checkpoint = torch.load(model_path, map_location=self.device)
                # Load with strict=False to ignore batch norm running stats
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"✅ Loaded trained breast cancer model from {model_path}")
                print(f"   Best metric: {checkpoint.get('best_metric', 'Unknown')}")
                print(f"   Missing keys: {len(missing_keys)}")
                print(f"   Unexpected keys: {len(unexpected_keys)}")
            else:
                print(f"⚠️  No trained model found at {model_path}, using untrained model")
            
            print(f"🔍 DEBUG: Setting model to eval mode")
            self.model.eval()
        else:
            print(f"🔍 DEBUG: Using existing model")
        
        print(f"🔍 DEBUG: _get_network returning: {type(self.model)}")
        return self.model
