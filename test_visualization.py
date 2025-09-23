#!/usr/bin/env python3
"""
Test script for Pearl-style breast cancer visualization
"""

import numpy as np
from breast_visualizer import BreastVisualizer
import matplotlib.pyplot as plt

def test_visualization():
    """Test the Pearl-style visualization"""
    
    print("🎨 Testing Pearl-style Breast Cancer Visualization")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = BreastVisualizer()
    
    # Create sample predictions (simulating your AI model output)
    print("📊 Creating sample predictions...")
    
    # Simulate 3-channel output: [background, breast_tissue, lesions]
    predictions = np.zeros((3, 256, 256))
    
    # Background (mostly negative values)
    predictions[0] = np.random.normal(-0.5, 0.3, (256, 256))
    
    # Breast tissue (moderate positive values in certain areas)
    tissue_mask = np.random.rand(256, 256) > 0.7
    predictions[1] = np.random.normal(0.2, 0.4, (256, 256))
    predictions[1][tissue_mask] = np.random.normal(0.8, 0.2, tissue_mask.sum())
    
    # Lesions (high positive values in small areas)
    lesion_centers = [(100, 100), (150, 80), (80, 150)]
    for center_x, center_y in lesion_centers:
        y, x = np.ogrid[:256, :256]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        lesion_mask = distance < 20
        predictions[2][lesion_mask] = np.random.normal(3.0, 0.5, lesion_mask.sum())
    
    # Create overlay
    print("🎨 Creating Pearl-style overlay...")
    overlay = visualizer.create_overlay(predictions, confidence_threshold=2.0)
    
    # Display results
    print("📈 Prediction Statistics:")
    print(f"  Background: min={predictions[0].min():.3f}, max={predictions[0].max():.3f}, mean={predictions[0].mean():.3f}")
    print(f"  Breast Tissue: min={predictions[1].min():.3f}, max={predictions[1].max():.3f}, mean={predictions[1].mean():.3f}")
    print(f"  Lesions: min={predictions[2].min():.3f}, max={predictions[2].max():.3f}, mean={predictions[2].mean():.3f}")
    
    # Create visualization
    print("🖼️ Creating visualization...")
    
    # Create a sample DICOM-like image
    sample_image = np.random.normal(128, 30, (256, 256)).astype(np.uint8)
    
    # Convert to RGB
    sample_rgb = np.stack([sample_image] * 3, axis=-1)
    
    # Blend with overlay
    alpha = overlay[:, :, 3:4] / 255.0
    result = sample_rgb * (1 - alpha) + overlay[:, :, :3] * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(sample_image, cmap='gray')
    axes[0, 0].set_title('Original DICOM Image')
    axes[0, 0].axis('off')
    
    # Predictions
    axes[0, 1].imshow(predictions[2], cmap='hot')
    axes[0, 1].set_title('Lesion Predictions (Raw)')
    axes[0, 1].axis('off')
    
    # Overlay
    axes[1, 0].imshow(overlay[:, :, :3])
    axes[1, 0].set_title('Pearl-Style Overlay')
    axes[1, 0].axis('off')
    
    # Final result
    axes[1, 1].imshow(result)
    axes[1, 1].set_title('Final Visualization')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('pearl_style_visualization_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization test completed!")
    print("📁 Saved as: pearl_style_visualization_test.png")
    print("\n🎯 Next steps:")
    print("1. Run your Streamlit app")
    print("2. Upload a DICOM file")
    print("3. Run AI inference")
    print("4. View the Pearl-style overlay visualization")
    print("5. Download for 3D Slicer integration")

if __name__ == "__main__":
    test_visualization()





