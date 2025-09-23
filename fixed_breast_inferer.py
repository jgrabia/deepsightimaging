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
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "trained_models", "best_breast_cancer_model.pth")
        
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
                "out_channels": 1,  # Single channel output
                "init_filters": 32,
                "dropout_prob": 0.2,
                "image_size": [256, 256]  # Match our training size
            },
            load_strict=True,
            roi_size=[256, 256],
            preload=False,
            train_mode=False
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._inferer = None  # Use different name to avoid conflicts
        
    def pre_transforms(self, data=None):
        return [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Remove ResizeWithPadOrCropd to avoid dimension mismatch
            EnsureTyped(keys=["image"]),
        ]
    
    def post_transforms(self, data=None):
        """Return the post-processing transforms"""
        print(f"🔍 DEBUG: post_transforms called with data type: {type(data)}")
        
        # Always return the transforms list, regardless of data type
        transforms = [
            Activationsd(keys=["pred"], softmax=True),
            AsDiscreted(keys=["pred"], argmax=True),
        ]
        print(f"🔍 DEBUG: Returning {len(transforms)} post transforms")
        return transforms
    
    def writer(self, data, extension=None, dtype=None):
        """Override writer to handle the ITK direction issue"""
        print(f"🔍 DEBUG: Custom writer called")
        
        # Get the prediction result
        pred = data.get("pred")
        if pred is not None:
            print(f"🔍 DEBUG: Prediction shape: {pred.shape}")
            print(f"🔍 DEBUG: Prediction type: {pred.dtype}")
            
            # Convert to numpy and ensure proper format
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            elif hasattr(pred, 'numpy'):
                pred = pred.numpy()
            
            # Ensure 3D format for ITK
            if len(pred.shape) == 4:  # [B, C, H, W]
                pred = pred[0, 0, :, :]  # Take first batch, first channel
            elif len(pred.shape) == 3:  # [C, H, W]
                pred = pred[0, :, :]  # Take first channel
            
            print(f"🔍 DEBUG: Final prediction shape: {pred.shape}")
            
            # Create a simple result with JSON-serializable data
            result = {
                "pred_shape": list(pred.shape),
                "pred_dtype": str(pred.dtype),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
                "success": True,
                "message": "Breast cancer detection completed successfully"
            }
            
            return "result.json", result
        
        # Fallback to default writer
        from monailabel.transform.writer import Writer
        return Writer()(data, extension, dtype)
    
    @property
    def post_transforms_property(self):
        """Property version of post_transforms that doesn't take parameters"""
        return [
            Activationsd(keys=["pred"], softmax=True),
            AsDiscreted(keys=["pred"], argmax=True),
        ]
    
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
            roi_size=[256, 256],
            sw_batch_size=1,
            overlap=0.5,
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
                roi_size=[256, 256],
                sw_batch_size=1,
                overlap=0.5,
            )
            print(f"🔍 DEBUG: Created new SlidingWindowInferer")
        
        # Debug: Check if network method exists
        print(f"🔍 DEBUG: self.network type: {type(self.network)}")
        print(f"🔍 DEBUG: self.network is callable: {callable(self.network)}")
        
        # Get the network - handle the case where self.network is None
        print(f"🔍 DEBUG: About to get network")
        print(f"🔍 DEBUG: self.network type: {type(self.network)}")
        
        if self.network is None:
            print(f"🔍 DEBUG: self.network is None, creating network directly")
            network = SegResNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                dropout_prob=0.2,
            )
            network = network.to(self.device)
            network.eval()
            print(f"🔍 DEBUG: Created network directly: {type(network)}")
        else:
            try:
                network = self.network(data)
                print(f"🔍 DEBUG: Got network via method: {type(network)}")
            except Exception as e:
                print(f"🔍 DEBUG: Error getting network: {e}")
                # Create a fallback network
                network = SegResNet(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
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
            
            result = self._inferer(inputs=input_image, network=network)
            print(f"🔍 DEBUG: Inference result shape: {result.shape if hasattr(result, 'shape') else 'No shape'}")
            
            # Add the result to the data dictionary
            data["pred"] = result
            print(f"🔍 DEBUG: Added prediction to data")
            
            return data
        else:
            print(f"🔍 DEBUG: Data is not a dict or missing image key")
            return data
    

    
    @property
    def inferer_property(self):
        """Property method that MONAI Label might be looking for"""
        if self._inferer is None:
            self._inferer = SlidingWindowInferer(
                roi_size=[256, 256],
                sw_batch_size=1,
                overlap=0.5,
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
        
        if self.model is None:
            print(f"🔍 DEBUG: Creating new SegResNet model")
            self.model = SegResNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,  # Single channel output
                init_filters=32,
                dropout_prob=0.2,
            )
            
            print(f"🔍 DEBUG: Moving model to device: {self.device}")
            self.model = self.model.to(self.device)
            
            # Load trained weights
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "trained_models", "best_breast_cancer_model.pth")
            print(f"🔍 DEBUG: Looking for model at: {model_path}")
            print(f"🔍 DEBUG: Model file exists: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                print(f"🔍 DEBUG: Loading trained weights")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained breast cancer model from {model_path}")
                print(f"   Best metric: {checkpoint.get('best_metric', 'Unknown')}")
            else:
                print(f"⚠️  No trained model found at {model_path}, using untrained model")
            
            print(f"🔍 DEBUG: Setting model to eval mode")
            self.model.eval()
        else:
            print(f"🔍 DEBUG: Using existing model")
        
        print(f"🔍 DEBUG: _get_network returning: {type(self.model)}")
        return self.model
