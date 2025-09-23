import numpy as np
import cv2
from PIL import Image, ImageDraw
import pydicom

class BreastVisualizer:
    def __init__(self):
        # Pearl-inspired colors (RGBA)
        self.colors = {
            'tissue': (173, 216, 230, 80),  # Light blue
            'lesions': (255, 182, 193, 120),  # Light pink
            'high_conf': (255, 105, 180, 160)  # Hot pink
        }
    
    def create_overlay(self, predictions, confidence_threshold=0.5):
        """Convert predictions to Pearl-style overlay"""
        if len(predictions.shape) == 3 and predictions.shape[0] == 3:
            lesion_probs = predictions[2]
            tissue_probs = predictions[1]
        else:
            lesion_probs = predictions
            tissue_probs = np.zeros_like(predictions)
        
        # Create overlay
        overlay = np.zeros((*predictions.shape[-2:], 4), dtype=np.uint8)
        
        # Tissue overlay (light blue)
        tissue_mask = tissue_probs > 0.3
        if tissue_mask.any():
            overlay[tissue_mask] = self.colors['tissue']
        
        # Lesion overlay (light pink to hot pink)
        lesion_mask = lesion_probs > 0.2
        if lesion_mask.any():
            for i in range(lesion_mask.shape[0]):
                for j in range(lesion_mask.shape[1]):
                    if lesion_mask[i, j]:
                        confidence = lesion_probs[i, j]
                        if confidence > confidence_threshold:
                            overlay[i, j] = self.colors['high_conf']
                        else:
                            color = list(self.colors['lesions'])
                            color[3] = int(color[3] * confidence)
                            overlay[i, j] = color
        
        return overlay
    
    def visualize_dicom(self, dicom_path, predictions, output_path=None):
        """Create Pearl-style visualization"""
        # Load DICOM
        ds = pydicom.dcmread(dicom_path)
        original = ds.pixel_array
        
        # Debug information
        print(f"🔍 DEBUG: DICOM image shape: {original.shape}")
        print(f"🔍 DEBUG: DICOM image dtype: {original.dtype}")
        print(f"🔍 DEBUG: DICOM image min/max: {original.min()}/{original.max()}")
        print(f"🔍 DEBUG: Predictions shape: {predictions.shape}")
        print(f"🔍 DEBUG: Predictions dtype: {predictions.dtype}")
        
        # Resize if needed
        if original.shape != predictions.shape[-2:]:
            original = cv2.resize(original, (256, 256))
        
        # Convert to RGB - handle different bit depths
        if len(original.shape) == 2:
            try:
                # Normalize to 8-bit range first
                if original.dtype == np.uint16:
                    # 16-bit to 8-bit conversion
                    original_8bit = (original / 256).astype(np.uint8)
                elif original.dtype == np.int16:
                    # Signed 16-bit to 8-bit conversion
                    original_8bit = ((original + 32768) / 256).astype(np.uint8)
                elif original.dtype == np.float32 or original.dtype == np.float64:
                    # Float to 8-bit conversion
                    original_8bit = np.clip(original, 0, 255).astype(np.uint8)
                else:
                    # Already 8-bit or other format
                    original_8bit = original.astype(np.uint8)
                
                print(f"🔍 DEBUG: Converted to 8-bit, shape: {original_8bit.shape}, dtype: {original_8bit.dtype}")
                
                # Convert to RGB
                original_rgb = cv2.cvtColor(original_8bit, cv2.COLOR_GRAY2RGB)
                print(f"🔍 DEBUG: Converted to RGB, shape: {original_rgb.shape}")
                
            except Exception as e:
                print(f"🔍 DEBUG: Error in color conversion: {e}")
                # Fallback: create a simple grayscale visualization
                if original.dtype == np.uint16:
                    original_8bit = (original / 256).astype(np.uint8)
                elif original.dtype == np.int16:
                    original_8bit = ((original + 32768) / 256).astype(np.uint8)
                else:
                    original_8bit = np.clip(original, 0, 255).astype(np.uint8)
                
                # Create RGB by repeating the grayscale channel
                original_rgb = np.stack([original_8bit] * 3, axis=-1)
                print(f"🔍 DEBUG: Using fallback RGB conversion, shape: {original_rgb.shape}")
        else:
            original_rgb = original
        
        # Create overlay
        overlay = self.create_overlay(predictions)
        
        # Blend images
        alpha = overlay[:, :, 3:4] / 255.0
        result = original_rgb * (1 - alpha) + overlay[:, :, :3] * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Add legend
        result_pil = Image.fromarray(result)
        result_pil = self.add_legend(result_pil)
        
        if output_path:
            result_pil.save(output_path)
        
        return result_pil
    
    def add_legend(self, image):
        """Add Pearl-style legend"""
        draw = ImageDraw.Draw(image)
        
        # Legend background
        legend_bg = Image.new('RGBA', (200, 80), (0, 0, 0, 128))
        image.paste(legend_bg, (10, image.height - 90), legend_bg)
        
        # Legend items
        y = image.height - 80
        draw.rectangle([20, y, 40, y + 15], fill=self.colors['tissue'][:3])
        draw.text((45, y), "Breast Tissue", fill="white")
        
        draw.rectangle([20, y + 20, 40, y + 35], fill=self.colors['lesions'][:3])
        draw.text((45, y + 20), "Potential Lesions", fill="white")
        
        draw.rectangle([20, y + 40, 40, y + 55], fill=self.colors['high_conf'][:3])
        draw.text((45, y + 40), "High Confidence", fill="white")
        
        return image
