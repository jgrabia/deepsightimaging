#!/usr/bin/env python3
"""
Enhanced Training API Server for DeepSight Imaging
- Training status and monitoring
- File upload and DICOM processing
- TCIA integration
- AI analysis endpoints
- DICOM viewer API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from pathlib import Path
import psutil
import subprocess
from datetime import datetime
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import shutil
import tempfile
import logging
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Training API Server", version="2.0.0")

# CORS middleware for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
training_process = None
upload_dir = Path("/tmp/uploads")
upload_dir.mkdir(exist_ok=True)

# Pydantic models
class TrainingConfig(BaseModel):
    model_type: str = "UNet"
    learning_rate: float = 1e-3
    batch_size: int = 8
    epochs: int = 30

class AIAnalysisRequest(BaseModel):
    image_path: str
    model_name: str = "deepedit"
    confidence_threshold: float = 0.5

class TCIAFilter(BaseModel):
    collection: Optional[str] = None
    bodyPart: Optional[str] = None
    modality: Optional[str] = None
    manufacturer: Optional[str] = None
    patientId: Optional[str] = None
    studyUid: Optional[str] = None

# TCIA API Configuration
TCIA_BASE_URL = "https://services.cancerimagingarchive.net/services/v4/NBIA"

@app.get("/")
async def root():
    return {"message": "Enhanced Training API Server", "version": "2.0.0", "status": "running"}

# =============================================================================
# TRAINING ENDPOINTS
# =============================================================================

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status and progress"""
    try:
        # Look for the latest training log
        log_dir = Path("/home/ubuntu/mri_app/logs")
        if not log_dir.exists():
            return {
                "status": "No Training",
                "currentPhase": "No Training Started",
                "filesProcessed": 0,
                "totalFiles": 13421,
                "epoch": 0,
                "totalEpochs": 30,
                "accuracy": 0,
                "loss": 0,
                "lastUpdate": datetime.now().isoformat()
            }
        
        log_files = list(log_dir.glob("training_*.log"))
        if not log_files:
            return {
                "status": "No Training",
                "currentPhase": "No Training Started",
                "filesProcessed": 0,
                "totalFiles": 13421,
                "epoch": 0,
                "totalEpochs": 30,
                "accuracy": 0,
                "loss": 0,
                "lastUpdate": datetime.now().isoformat()
            }
        
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getctime)
        
        # Parse the log file for current status
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        # Initialize default values
        status = "Processing"
        current_phase = "Data Processing"
        files_processed = 0
        epoch = 0
        accuracy = 0
        loss = 0
        
        # Parse the last few lines for current status
        for line in reversed(lines[-50:]):
            if "Epoch" in line and "/" in line:
                try:
                    epoch_part = line.split("Epoch")[1].split()[0]
                    epoch = int(epoch_part.split("/")[0])
                except:
                    pass
            if "Train Acc:" in line:
                try:
                    accuracy = float(line.split("Train Acc:")[1].split("%")[0])
                except:
                    pass
            if "Train Loss:" in line:
                try:
                    loss = float(line.split("Train Loss:")[1].split(",")[0])
                except:
                    pass
            if "files processed" in line.lower():
                try:
                    files_processed = int(line.split()[0])
                except:
                    pass
        
        # Determine status based on epoch
        if epoch > 0:
            status = "Training"
            current_phase = f"Training Epoch {epoch}"
        else:
            status = "Processing"
            current_phase = "Data Processing"
        
        return {
            "status": status,
            "currentPhase": current_phase,
            "filesProcessed": files_processed,
            "totalFiles": 13421,
            "epoch": epoch,
            "totalEpochs": 30,
            "accuracy": accuracy / 100 if accuracy > 1 else accuracy,
            "loss": loss,
            "lastUpdate": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return {
            "status": "Error",
            "currentPhase": "Error",
            "filesProcessed": 0,
            "totalFiles": 13421,
            "epoch": 0,
            "totalEpochs": 30,
            "accuracy": 0,
            "loss": 0,
            "lastUpdate": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start training with specified configuration"""
    global training_process
    
    try:
        if training_process and training_process.poll() is None:
            return {"message": "Training already running", "status": "running"}
        
        # Create training script with config
        training_script = f"""
import sys
sys.path.append('/home/ubuntu/mri_app')
from advanced_breast_training_improved import AdvancedBreastCancerTrainer

config = {{
    "model_type": "{config.model_type}",
    "learning_rate": {config.learning_rate},
    "batch_size": {config.batch_size},
    "epochs": {config.epochs},
    "patience": 8,
    "use_real_annotations": True
}}

trainer = AdvancedBreastCancerTrainer(
    data_dir="/home/ubuntu/mri_app/dbt_complete_training_data/train",
    config=config
)

trainer.run_advanced_training()
"""
        
        # Write and execute training script
        script_path = "/tmp/start_training.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        training_process = subprocess.Popen([
            "python", script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {"message": "Training started", "status": "started", "pid": training_process.pid}
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/stop")
async def stop_training():
    """Stop current training"""
    global training_process
    
    try:
        if training_process and training_process.poll() is None:
            training_process.terminate()
            return {"message": "Training stopped", "status": "stopped"}
        else:
            return {"message": "No training running", "status": "stopped"}
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FILE UPLOAD ENDPOINTS
# =============================================================================

@app.post("/api/files/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload DICOM files for processing"""
    try:
        uploaded_files = []
        
        for file in files:
            if not file.filename.lower().endswith(('.dcm', '.dicom')):
                continue
                
            # Save file to upload directory
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append({
                "filename": file.filename,
                "size": file_path.stat().st_size,
                "path": str(file_path)
            })
        
        return {
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
        
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/list")
async def list_uploaded_files():
    """List all uploaded files"""
    try:
        files = []
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "uploaded": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DICOM VIEWER ENDPOINTS
# =============================================================================

@app.get("/api/dicom/info/{filename}")
async def get_dicom_info(filename: str):
    """Get DICOM file information"""
    try:
        file_path = upload_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Use pydicom to read DICOM info
        try:
            import pydicom
            ds = pydicom.dcmread(str(file_path))
            
            return {
                "filename": filename,
                "patient_id": getattr(ds, 'PatientID', 'Unknown'),
                "study_description": getattr(ds, 'StudyDescription', 'Unknown'),
                "series_description": getattr(ds, 'SeriesDescription', 'Unknown'),
                "modality": getattr(ds, 'Modality', 'Unknown'),
                "manufacturer": getattr(ds, 'Manufacturer', 'Unknown'),
                "rows": getattr(ds, 'Rows', 0),
                "columns": getattr(ds, 'Columns', 0),
                "slices": getattr(ds, 'NumberOfFrames', 1)
            }
        except ImportError:
            return {
                "filename": filename,
                "error": "pydicom not available",
                "size": file_path.stat().st_size
            }
        
    except Exception as e:
        logger.error(f"Error getting DICOM info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dicom/image/{filename}")
async def get_dicom_image(filename: str, slice_index: int = 0):
    """Get DICOM image as PNG"""
    try:
        file_path = upload_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Convert DICOM to PNG (simplified - you'd use proper DICOM processing)
        # For now, return a placeholder
        return {"message": f"DICOM image for {filename}, slice {slice_index}", "status": "placeholder"}
        
    except Exception as e:
        logger.error(f"Error getting DICOM image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# AI ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/api/ai/analyze")
async def analyze_image(request: AIAnalysisRequest):
    """Run AI analysis on uploaded image"""
    try:
        file_path = Path(request.image_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Simulate AI analysis (replace with actual MONAI inference)
        await asyncio.sleep(2)  # Simulate processing time
        
        # Mock analysis results
        analysis_results = {
            "filename": file_path.name,
            "model_used": request.model_name,
            "confidence": 0.87,
            "findings": [
                {
                    "type": "Mass",
                    "location": "Upper outer quadrant",
                    "confidence": 0.92,
                    "coordinates": {"x": 150, "y": 200, "width": 50, "height": 40}
                },
                {
                    "type": "Calcification",
                    "location": "Central",
                    "confidence": 0.78,
                    "coordinates": {"x": 300, "y": 250, "width": 20, "height": 20}
                }
            ],
            "recommendations": "Follow-up recommended in 6 months",
            "processing_time": 2.3,
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/models")
async def get_available_models():
    """Get list of available AI models"""
    return {
        "models": [
            {
                "name": "deepedit",
                "description": "DeepEdit for lesion segmentation",
                "type": "segmentation",
                "status": "available"
            },
            {
                "name": "segresnet",
                "description": "SegResNet for organ segmentation",
                "type": "segmentation",
                "status": "available"
            },
            {
                "name": "breast_cancer_detector",
                "description": "Custom breast cancer detection model",
                "type": "classification",
                "status": "training"
            }
        ]
    }

# =============================================================================
# TCIA ENDPOINTS
# =============================================================================

@app.get("/api/tcia/collections")
async def get_tcia_collections():
    """Get available TCIA collections"""
    try:
        response = requests.get(f"{TCIA_BASE_URL}/query/getCollectionValues")
        if response.status_code == 200:
            return response.json()
        else:
            # Return mock data if TCIA is unavailable
            return [
                {"Collection": "Breast-Cancer-Screening-DBT"},
                {"Collection": "CBIS-DDSM"},
                {"Collection": "INbreast"},
                {"Collection": "MIAS"}
            ]
    except Exception as e:
        logger.error(f"Error fetching TCIA collections: {e}")
        return [
            {"Collection": "Breast-Cancer-Screening-DBT"},
            {"Collection": "CBIS-DDSM"},
            {"Collection": "INbreast"},
            {"Collection": "MIAS"}
        ]

@app.get("/api/tcia/body-parts")
async def get_tcia_body_parts():
    """Get available body parts from TCIA"""
    try:
        response = requests.get(f"{TCIA_BASE_URL}/query/getBodyPartValues")
        if response.status_code == 200:
            return response.json()
        else:
            return [
                {"BodyPartExamined": "BREAST"},
                {"BodyPartExamined": "CHEST"},
                {"BodyPartExamined": "ABDOMEN"},
                {"BodyPartExamined": "HEAD"}
            ]
    except Exception as e:
        logger.error(f"Error fetching TCIA body parts: {e}")
        return [
            {"BodyPartExamined": "BREAST"},
            {"BodyPartExamined": "CHEST"},
            {"BodyPartExamined": "ABDOMEN"},
            {"BodyPartExamined": "HEAD"}
        ]

@app.post("/api/tcia/search")
async def search_tcia_series(filters: TCIAFilter):
    """Search TCIA for series matching filters"""
    try:
        # Build query parameters
        params = {}
        if filters.collection:
            params["Collection"] = filters.collection
        if filters.bodyPart:
            params["BodyPartExamined"] = filters.bodyPart
        if filters.modality:
            params["Modality"] = filters.modality
        if filters.manufacturer:
            params["Manufacturer"] = filters.manufacturer
        if filters.patientId:
            params["PatientID"] = filters.patientId
        if filters.studyUid:
            params["StudyInstanceUID"] = filters.studyUid
        
        # Make request to TCIA
        response = requests.get(f"{TCIA_BASE_URL}/query/getSeries", params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Format the response for the frontend
            formatted_results = []
            for item in data:
                formatted_results.append({
                    "SeriesInstanceUID": item.get("SeriesInstanceUID", ""),
                    "PatientID": item.get("PatientID", ""),
                    "StudyInstanceUID": item.get("StudyInstanceUID", ""),
                    "SeriesDescription": item.get("SeriesDescription", ""),
                    "Modality": item.get("Modality", ""),
                    "BodyPartExamined": item.get("BodyPartExamined", ""),
                    "Manufacturer": item.get("Manufacturer", ""),
                    "NumberOfImages": item.get("NumberOfImages", 0),
                    "Collection": item.get("Collection", "")
                })
            return formatted_results
        else:
            # Return mock data if TCIA is unavailable
            return [
                {
                    "SeriesInstanceUID": "1.2.3.4.5.6.7.8.9.10",
                    "PatientID": "DBT-P00013",
                    "StudyInstanceUID": "1.2.3.4.5.6.7.8.9.11",
                    "SeriesDescription": "DBT Series",
                    "Modality": "MG",
                    "BodyPartExamined": "BREAST",
                    "Manufacturer": "HOLOGIC",
                    "NumberOfImages": 75,
                    "Collection": "Breast-Cancer-Screening-DBT"
                }
            ]
            
    except Exception as e:
        logger.error(f"Error searching TCIA: {e}")
        # Return mock data on error
        return [
            {
                "SeriesInstanceUID": "1.2.3.4.5.6.7.8.9.10",
                "PatientID": "DBT-P00013",
                "StudyInstanceUID": "1.2.3.4.5.6.7.8.9.11",
                "SeriesDescription": "DBT Series",
                "Modality": "MG",
                "BodyPartExamined": "BREAST",
                "Manufacturer": "HOLOGIC",
                "NumberOfImages": 75,
                "Collection": "Breast-Cancer-Screening-DBT"
            }
        ]

@app.post("/api/tcia/download")
async def download_tcia_series(request: Dict[str, Any]):
    """Download TCIA series (placeholder)"""
    try:
        series_uids = request.get("seriesUids", [])
        
        # This would integrate with actual TCIA download functionality
        # For now, return a placeholder response
        
        return {
            "message": f"Download initiated for {len(series_uids)} series",
            "series_count": len(series_uids),
            "status": "initiated",
            "download_id": f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        logger.error(f"Error downloading TCIA series: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SYSTEM STATUS ENDPOINTS
# =============================================================================

@app.get("/api/system/status")
async def get_system_status():
    """Get system status (CPU, memory, disk)"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Check if training is running
        training_running = False
        try:
            result = subprocess.run(['pgrep', '-f', 'advanced_breast_training'], 
                                  capture_output=True, text=True)
            training_running = result.returncode == 0
        except:
            pass
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
            "disk_available_gb": disk.free / (1024**3),
            "training_running": training_running,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Training API Server...")
    print("📊 API will be available at: http://0.0.0.0:8000")
    print("🔗 React app should connect to: http://3.88.157.239:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
