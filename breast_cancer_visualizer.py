import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont
import pydicom
import json
import os
from pathlib import Path

class BreastCancerVisualizer:
    """Visualization module for breast cancer AI predictions with Pearl-style overlays"""
    
    def __init__(self):
        # Define Pearl-inspired color scheme
        self.colors = {
            'background': (0, 0, 0, 0),  # Transparent
            'breast_tissue': (173, 216, 230, 80),  # Light blue with transparency
            'lesions': (255, 182, 193, 120),  # Light pink with transparency
            'high_confidence_lesions': (255, 105, 180, 160)  # Hot pink for high confidence
        }
        
        # Create custom colormaps
        self.lesion_cmap = LinearSegmentedColormap.from_list(
            'lesion_cmap', 
            ['transparent', 'lightpink', 'hotpink', 'red']
        )
        
        self.tissue_cmap = LinearSegmentedColormap.from_list(
            'tissue_cmap', 
            ['transparent', 'lightblue', 'blue']
        )
    
    def convert_predictions_to_overlay(self, predictions, confidence_threshold=0.5):
        """
        Convert raw model predictions to colored overlay masks
        
        Args:
            predictions: Raw model output [256, 256] or [3, 256, 256]
            confidence_threshold: Threshold for high confidence lesions
        
        Returns:
            overlay_image: RGBA overlay with transparent colors
            lesion_mask: Binary mask of detected lesions
            tissue_mask: Binary mask of breast tissue
        """
        if len(predictions.shape) == 2:
            # Single channel output - assume it's lesion probability
            lesion_probs = predictions
            tissue_probs = np.zeros_like(predictions)
        elif len(predictions.shape) == 3 and predictions.shape[0] == 3:
            # 3-channel output: [background, breast_tissue, lesions]
            background_probs = predictions[0]
            tissue_probs = predictions[1]
            lesion_probs = predictions[2]
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        
        # Create overlay image
        overlay = np.zeros((*predictions.shape[-2:], 4), dtype=np.uint8)
        
        # Apply tissue overlay (light blue)
        tissue_mask = tissue_probs > 0.3
        if tissue_mask.any():
            tissue_color = np.array(self.colors['breast_tissue'])
            overlay[tissue_mask] = tissue_color
        
        # Apply lesion overlay (light pink to hot pink based on confidence)
        lesion_mask = lesion_probs > 0.2
        if lesion_mask.any():
            # Normalize lesion probabilities for color intensity
            lesion_norm = np.clip((lesion_probs - 0.2) / 0.8, 0, 1)
            
            for i in range(lesion_mask.shape[0]):
                for j in range(lesion_mask.shape[1]):
                    if lesion_mask[i, j]:
                        confidence = lesion_norm[i, j]
                        if confidence > confidence_threshold:
                            # High confidence - hot pink
                            color = np.array(self.colors['high_confidence_lesions'])
                        else:
                            # Low confidence - light pink
                            color = np.array(self.colors['lesions'])
                            color[3] = int(color[3] * confidence)  # Adjust transparency
                        overlay[i, j] = color
        
        return overlay, lesion_mask, tissue_mask
    
    def create_pearl_style_overlay(self, dicom_path, predictions, output_path=None):
        """
        Create Pearl-style overlay visualization
        
        Args:
            dicom_path: Path to original DICOM file
            predictions: Raw model predictions
            output_path: Path to save visualization
        
        Returns:
            overlay_image: PIL Image with overlay
        """
        # Load DICOM
        ds = pydicom.dcmread(dicom_path)
        original_image = ds.pixel_array
        
        # Resize original to match predictions if needed
        if original_image.shape != predictions.shape[-2:]:
            original_image = cv2.resize(original_image, (256, 256))
        
        # Convert to RGB - handle different bit depths
        if len(original_image.shape) == 2:
            try:
                # Normalize to 8-bit range first
                if original_image.dtype == np.uint16:
                    # 16-bit to 8-bit conversion
                    original_8bit = (original_image / 256).astype(np.uint8)
                elif original_image.dtype == np.int16:
                    # Signed 16-bit to 8-bit conversion
                    original_8bit = ((original_image + 32768) / 256).astype(np.uint8)
                elif original_image.dtype == np.float32 or original_image.dtype == np.float64:
                    # Float to 8-bit conversion
                    original_8bit = np.clip(original_image, 0, 255).astype(np.uint8)
                else:
                    # Already 8-bit or other format
                    original_8bit = original_image.astype(np.uint8)
                
                print(f"🔍 DEBUG: Converted to 8-bit, shape: {original_8bit.shape}, dtype: {original_8bit.dtype}")
                
                # Convert to RGB
                original_rgb = cv2.cvtColor(original_8bit, cv2.COLOR_GRAY2RGB)
                print(f"🔍 DEBUG: Converted to RGB, shape: {original_rgb.shape}")
                
            except Exception as e:
                print(f"🔍 DEBUG: Error in color conversion: {e}")
                # Fallback: create a simple grayscale visualization
                if original_image.dtype == np.uint16:
                    original_8bit = (original_image / 256).astype(np.uint8)
                elif original_image.dtype == np.int16:
                    original_8bit = ((original_image + 32768) / 256).astype(np.uint8)
                else:
                    original_8bit = np.clip(original_image, 0, 255).astype(np.uint8)
                
                # Create RGB by repeating the grayscale channel
                original_rgb = np.stack([original_8bit] * 3, axis=-1)
                print(f"🔍 DEBUG: Using fallback RGB conversion, shape: {original_rgb.shape}")
        else:
            original_rgb = original_image
        
        # Create overlay
        overlay, lesion_mask, tissue_mask = self.convert_predictions_to_overlay(predictions)
        
        # Combine original image with overlay
        result = original_rgb.copy().astype(np.float32)
        alpha = overlay[:, :, 3:4] / 255.0
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        
        # Blend images
        result = result * (1 - alpha) + overlay_rgb * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        result_pil = Image.fromarray(result)
        
        # Add legend
        result_pil = self.add_legend(result_pil, lesion_mask, tissue_mask)
        
        if output_path:
            result_pil.save(output_path)
        
        return result_pil
    
    def add_legend(self, image, lesion_mask, tissue_mask):
        """Add Pearl-style legend to the image"""
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Legend background
        legend_height = 80
        legend_width = 200
        legend_x = 10
        legend_y = image.height - legend_height - 10
        
        # Draw semi-transparent background
        legend_bg = Image.new('RGBA', (legend_width, legend_height), (0, 0, 0, 128))
        image.paste(legend_bg, (legend_x, legend_y), legend_bg)
        
        # Legend text
        y_offset = legend_y + 10
        
        # Breast tissue
        tissue_color = self.colors['breast_tissue']
        draw.rectangle([legend_x + 10, y_offset, legend_x + 30, y_offset + 15], 
                      fill=tissue_color[:3])
        draw.text((legend_x + 35, y_offset), "Breast Tissue", fill="white", font=font)
        
        # Lesions
        lesion_color = self.colors['lesions']
        draw.rectangle([legend_x + 10, y_offset + 20, legend_x + 30, y_offset + 35], 
                      fill=lesion_color[:3])
        draw.text((legend_x + 35, y_offset + 20), "Potential Lesions", fill="white", font=font)
        
        # High confidence lesions
        high_conf_color = self.colors['high_confidence_lesions']
        draw.rectangle([legend_x + 10, y_offset + 40, legend_x + 30, y_offset + 55], 
                      fill=high_conf_color[:3])
        draw.text((legend_x + 35, y_offset + 40), "High Confidence", fill="white", font=font)
        
        return image
    
    def export_for_slicer(self, predictions, output_path):
        """
        Export predictions in format suitable for 3D Slicer
        
        Args:
            predictions: Raw model predictions
            output_path: Path to save Slicer-compatible file
        """
        # Convert predictions to Slicer-compatible format
        if len(predictions.shape) == 3 and predictions.shape[0] == 3:
            # Take the lesion channel (index 2)
            lesion_probs = predictions[2]
        else:
            lesion_probs = predictions
        
        # Create binary mask
        lesion_mask = (lesion_probs > 0.3).astype(np.uint8) * 255
        
        # Save as PNG for Slicer
        lesion_image = Image.fromarray(lesion_mask, mode='L')
        lesion_image.save(output_path)
        
        return output_path
    
    def create_circled_areas(self, image, predictions, confidence_threshold=0.5):
        """
        Create circled areas of concern (alternative visualization method)
        
        Args:
            image: Original DICOM image
            predictions: Raw model predictions
            confidence_threshold: Threshold for circling areas
        
        Returns:
            circled_image: Image with circles around areas of concern
        """
        if len(predictions.shape) == 3 and predictions.shape[0] == 3:
            lesion_probs = predictions[2]
        else:
            lesion_probs = predictions
        
        # Find connected components
        lesion_mask = lesion_probs > confidence_threshold
        if not lesion_mask.any():
            return image
        
        # Find contours
        contours, _ = cv2.findContours(
            lesion_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CONTOUR_APPROX_SIMPLE
        )
        
        # Draw circles around areas of concern
        circled_image = image.copy()
        for contour in contours:
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw circle
            cv2.circle(circled_image, center, radius, (255, 0, 0), 2)
            
            # Add confidence text
            confidence = np.max(lesion_probs[int(y-radius):int(y+radius), 
                                           int(x-radius):int(x+radius)])
            cv2.putText(circled_image, f"{confidence:.2f}", 
                       (center[0] - 20, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return circled_image

# Example usage
if __name__ == "__main__":
    visualizer = BreastCancerVisualizer()
    
    # Example with dummy predictions
    dummy_predictions = np.random.rand(3, 256, 256)
    
    # Create overlay
    overlay, lesion_mask, tissue_mask = visualizer.convert_predictions_to_overlay(dummy_predictions)
    
    print("Visualization module ready!")
    print("Use create_pearl_style_overlay() for Pearl-style transparent overlays")
    print("Use create_circled_areas() for circled areas of concern")
    print("Use export_for_slicer() for 3D Slicer integration")
