#!/usr/bin/env python3
"""
DICOM Viewer FastAPI Backend
Converts Streamlit DICOM viewer functionality to REST API endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import json
import numpy as np
import pydicom
from PIL import Image
import cv2
import base64
import io
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepSight DICOM Viewer API",
    description="API for DICOM image viewing and analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store current session data
current_dicom_data = {}
current_annotations = {}
current_slice = 0

def load_annotations_for_dicom(ds, annotation_folder: str = None):
    """Load annotations for a given DICOM file"""
    try:
        patient_id = getattr(ds, 'PatientID', 'unknown')
        annotations = []
        
        # Try JSON file first (in same directory as DICOM)
        if hasattr(ds, 'filename'):
            dicom_dir = Path(ds.filename).parent
            json_file = dicom_dir / f"{patient_id}_annotations.json"
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'lesions' in data:
                        for lesion in data['lesions']:
                            annotations.append({
                                'slice': lesion.get('slice', 0),
                                'x': lesion.get('x', 0),
                                'y': lesion.get('y', 0),
                                'width': lesion.get('width', 0),
                                'height': lesion.get('height', 0),
                                'class': lesion.get('class', 'unknown'),
                                'view': lesion.get('view', 'unknown')
                            })
                return annotations
        
        # Fallback to CSV files
        if annotation_folder:
            csv_files = [
                Path(annotation_folder) / "BCS-DBT-boxes-train-v2.csv",
                Path(annotation_folder) / "BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv"
            ]
            
            for csv_file in csv_files:
                if csv_file.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    patient_annotations = df[df['patient_id'] == patient_id]
                    
                    for _, row in patient_annotations.iterrows():
                        annotations.append({
                            'slice': int(row.get('slice', 0)),
                            'x': int(row.get('x', 0)),
                            'y': int(row.get('y', 0)),
                            'width': int(row.get('width', 0)),
                            'height': int(row.get('height', 0)),
                            'class': row.get('class', 'unknown'),
                            'view': row.get('view', 'unknown')
                        })
        
        return annotations
        
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        return []

def draw_annotations_on_image(image, annotations, current_slice):
    """Draw medical-grade annotations on image"""
    if not annotations:
        return image
    
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Convert to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Draw annotations for current slice
    for i, ann in enumerate(annotations):
        if ann['slice'] == current_slice:
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            
            # Medical-grade annotation style
            color = (255, 255, 0)  # Cyan
            thickness = 2
            
            # Draw bounding box
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)
            
            # Draw corner markers
            marker_size = 8
            cv2.circle(img_bgr, (x, y), marker_size, color, -1)
            cv2.circle(img_bgr, (x + w, y), marker_size, color, -1)
            cv2.circle(img_bgr, (x, y + h), marker_size, color, -1)
            cv2.circle(img_bgr, (x + w, y + h), marker_size, color, -1)
            
            # Draw center crosshair
            center_x, center_y = x + w // 2, y + h // 2
            crosshair_size = 10
            cv2.line(img_bgr, (center_x - crosshair_size, center_y), 
                    (center_x + crosshair_size, center_y), color, 1)
            cv2.line(img_bgr, (center_x, center_y - crosshair_size), 
                    (center_x, center_y + crosshair_size), color, 1)
            
            # Add annotation number
            cv2.putText(img_bgr, str(i + 1), (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Convert back to RGB
    if len(img_bgr.shape) == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def apply_window_level(image, window_center, window_width):
    """Apply window/level adjustment to image"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Calculate window/level
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    
    # Apply window/level
    img_array = np.clip(img_array, window_min, window_max)
    img_array = ((img_array - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def enhance_image(image, contrast_factor=1.2, noise_reduction=True):
    """Apply image enhancement"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Contrast adjustment
    if contrast_factor != 1.0:
        img_array = np.clip(img_array * contrast_factor, 0, 255).astype(np.uint8)
    
    # Noise reduction (median filter)
    if noise_reduction:
        img_array = cv2.medianBlur(img_array, 3)
    
    return Image.fromarray(img_array)

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DeepSight DICOM Viewer API", "status": "running"}

@app.post("/api/dicom/upload")
async def upload_dicom(file: UploadFile = File(...)):
    """Upload and process DICOM file"""
    try:
        # Read DICOM file
        content = await file.read()
        
        # Save temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load DICOM
        ds = pydicom.dcmread(temp_path)
        ds.filename = temp_path  # Store path for annotation loading
        
        # Extract pixel data
        pixel_array = ds.pixel_array
        
        # Handle 3D volumes
        if len(pixel_array.shape) == 3:
            num_slices = pixel_array.shape[0]
            current_slice = num_slices // 2
        else:
            num_slices = 1
            current_slice = 0
        
        # Store in global state
        global current_dicom_data, current_annotations
        current_dicom_data = {
            'ds': ds,
            'pixel_array': pixel_array,
            'num_slices': num_slices,
            'filename': file.filename
        }
        current_slice = current_slice
        
        # Load annotations
        current_annotations = load_annotations_for_dicom(ds)
        
        # Get first slice image
        if len(pixel_array.shape) == 3:
            slice_data = pixel_array[current_slice, :, :]
        else:
            slice_data = pixel_array
        
        # Normalize and convert to PIL
        if slice_data.max() > 255:
            slice_data = ((slice_data / slice_data.max()) * 255).astype(np.uint8)
        else:
            slice_data = slice_data.astype(np.uint8)
        
        image = Image.fromarray(slice_data)
        
        return {
            "status": "success",
            "filename": file.filename,
            "num_slices": num_slices,
            "current_slice": current_slice,
            "image": image_to_base64(image),
            "annotations_count": len(current_annotations),
            "patient_id": getattr(ds, 'PatientID', 'unknown'),
            "study_date": getattr(ds, 'StudyDate', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Error processing DICOM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dicom/slice/{slice_number}")
async def get_slice(slice_number: int, 
                   window_center: Optional[int] = Query(None),
                   window_width: Optional[int] = Query(None),
                   show_annotations: bool = Query(True),
                   contrast_factor: float = Query(1.0),
                   noise_reduction: bool = Query(False)):
    """Get specific slice with optional enhancements"""
    try:
        if not current_dicom_data:
            raise HTTPException(status_code=400, detail="No DICOM file loaded")
        
        pixel_array = current_dicom_data['pixel_array']
        num_slices = current_dicom_data['num_slices']
        
        # Validate slice number
        if slice_number < 0 or slice_number >= num_slices:
            raise HTTPException(status_code=400, detail="Invalid slice number")
        
        # Get slice data
        if len(pixel_array.shape) == 3:
            slice_data = pixel_array[slice_number, :, :]
        else:
            slice_data = pixel_array
        
        # Normalize
        if slice_data.max() > 255:
            slice_data = ((slice_data / slice_data.max()) * 255).astype(np.uint8)
        else:
            slice_data = slice_data.astype(np.uint8)
        
        image = Image.fromarray(slice_data)
        
        # Apply window/level if specified
        if window_center is not None and window_width is not None:
            image = apply_window_level(image, window_center, window_width)
        
        # Apply enhancements
        if contrast_factor != 1.0 or noise_reduction:
            image = enhance_image(image, contrast_factor, noise_reduction)
        
        # Apply annotations if requested
        if show_annotations and current_annotations:
            image = draw_annotations_on_image(image, current_annotations, slice_number)
        
        # Update current slice
        global current_slice
        current_slice = slice_number
        
        return {
            "slice_number": slice_number,
            "total_slices": num_slices,
            "image": image_to_base64(image),
            "annotations_count": len([a for a in current_annotations if a['slice'] == slice_number])
        }
        
    except Exception as e:
        logger.error(f"Error getting slice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dicom/annotations")
async def get_annotations():
    """Get all annotations for current DICOM"""
    if not current_annotations:
        return {"annotations": []}
    
    return {
        "annotations": current_annotations,
        "total_count": len(current_annotations),
        "cancer_count": len([a for a in current_annotations if a.get('class') == 'cancer']),
        "benign_count": len([a for a in current_annotations if a.get('class') == 'benign'])
    }

@app.get("/api/dicom/info")
async def get_dicom_info():
    """Get DICOM metadata information"""
    if not current_dicom_data:
        raise HTTPException(status_code=400, detail="No DICOM file loaded")
    
    ds = current_dicom_data['ds']
    
    return {
        "patient_id": getattr(ds, 'PatientID', 'unknown'),
        "patient_name": getattr(ds, 'PatientName', 'unknown'),
        "study_date": getattr(ds, 'StudyDate', 'unknown'),
        "study_description": getattr(ds, 'StudyDescription', 'unknown'),
        "modality": getattr(ds, 'Modality', 'unknown'),
        "manufacturer": getattr(ds, 'Manufacturer', 'unknown'),
        "model": getattr(ds, 'ManufacturerModelName', 'unknown'),
        "num_slices": current_dicom_data['num_slices'],
        "image_size": list(ds.pixel_array.shape) if hasattr(ds, 'pixel_array') else None
    }

@app.get("/api/dicom/window-level/presets")
async def get_window_level_presets():
    """Get window/level presets"""
    return {
        "presets": {
            "soft_tissue": {"center": 40, "width": 400},
            "bone": {"center": 400, "width": 1800},
            "lung": {"center": -600, "width": 1500},
            "auto": {"center": None, "width": None}
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
