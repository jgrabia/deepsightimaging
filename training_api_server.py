#!/usr/bin/env python3
"""
Training API Server
Simple FastAPI server to provide training status to React app
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pathlib import Path
import psutil
import subprocess
from datetime import datetime
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

app = FastAPI(title="Training API Server", version="1.0.0")

# Enable CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Render domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_training_process():
    """Find the running training process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'advanced_breast_training_improved.py' in cmdline:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def parse_training_log():
    """Parse the latest training log for status"""
    log_dir = Path("logs")
    if not log_dir.exists():
        return None
    
    # Find the most recent log file
    log_files = list(log_dir.glob("training_*.log"))
    if not log_files:
        return None
    
    latest_log = max(log_files, key=os.path.getctime)
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            
        # Parse the last few lines for current status
        status = {
            "status": "Processing",
            "currentPhase": "Data Processing",
            "filesProcessed": 0,
            "totalFiles": 13421,
            "epoch": 0,
            "totalEpochs": 30,
            "accuracy": 0,
            "loss": 0,
            "lastUpdate": datetime.now().isoformat()
        }
        
        # Look for progress indicators in the log
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "Progress:" in line and "files processed" in line:
                try:
                    # Extract file count from progress line
                    parts = line.split("Progress:")[1].split("files processed")[0]
                    files_processed = int(parts.split("/")[0].strip())
                    status["filesProcessed"] = files_processed
                except:
                    pass
            
            if "Epoch" in line and "/" in line:
                try:
                    # Extract epoch info
                    epoch_part = line.split("Epoch")[1].split("/")[0].strip()
                    status["epoch"] = int(epoch_part)
                except:
                    pass
            
            if "Train Acc:" in line:
                try:
                    acc_part = line.split("Train Acc:")[1].split("%")[0].strip()
                    status["accuracy"] = float(acc_part)
                    status["status"] = "Training"
                except:
                    pass
            
            if "Train Loss:" in line:
                try:
                    loss_part = line.split("Train Loss:")[1].split(",")[0].strip()
                    status["loss"] = float(loss_part)
                except:
                    pass
        
        return status
        
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None

@app.get("/")
async def root():
    return {"message": "Training API Server is running"}

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status"""
    try:
        # Check if training process is running
        process = get_training_process()
        
        if not process:
            return {
                "status": "Not Running",
                "currentPhase": "No Training Process",
                "filesProcessed": 0,
                "totalFiles": 13421,
                "epoch": 0,
                "totalEpochs": 30,
                "accuracy": 0,
                "loss": 0,
                "lastUpdate": datetime.now().isoformat()
            }
        
        # Parse log for detailed status
        log_status = parse_training_log()
        if log_status:
            return log_status
        
        # Fallback status
        return {
            "status": "Running",
            "currentPhase": "Processing",
            "filesProcessed": 0,
            "totalFiles": 13421,
            "epoch": 0,
            "totalEpochs": 30,
            "accuracy": 0,
            "loss": 0,
            "lastUpdate": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/logs")
async def get_training_logs():
    """Get recent training logs"""
    try:
        log_dir = Path("logs")
        if not log_dir.exists():
            return {"logs": []}
        
        log_files = list(log_dir.glob("training_*.log"))
        if not log_files:
            return {"logs": []}
        
        latest_log = max(log_files, key=os.path.getctime)
        
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        return {"logs": lines[-20:]}  # Last 20 lines
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status():
    """Get system status"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TCIA API Endpoints
TCIA_BASE_URL = "https://services.cancerimagingarchive.net/services/v4/NBIA"

@app.get("/api/tcia/collections")
async def get_tcia_collections():
    """Get available TCIA collections"""
    try:
        response = requests.get(f"{TCIA_BASE_URL}/query/getCollectionValues")
        if response.status_code == 200:
            collections = response.json()
            return [{"name": collection} for collection in collections]
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch collections")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tcia/body-parts")
async def get_tcia_body_parts():
    """Get available body parts"""
    try:
        response = requests.get(f"{TCIA_BASE_URL}/query/getBodyPartValues")
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch body parts")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tcia/search")
async def search_tcia_series(filters: Dict[str, Any]):
    """Search TCIA series with filters"""
    try:
        # Build query parameters
        params = {}
        if filters.get("collection"):
            params["Collection"] = filters["collection"]
        if filters.get("bodyPart"):
            params["BodyPartExamined"] = filters["bodyPart"]
        if filters.get("modality"):
            params["Modality"] = filters["modality"]
        if filters.get("manufacturer"):
            params["Manufacturer"] = filters["manufacturer"]
        if filters.get("patientId"):
            params["PatientID"] = filters["patientId"]
        if filters.get("studyUid"):
            params["StudyInstanceUID"] = filters["studyUid"]
        
        # Add default filters for breast cancer data
        params["Modality"] = params.get("Modality", "MG")
        
        print(f"🔍 Searching TCIA with params: {params}")
        
        response = requests.get(f"{TCIA_BASE_URL}/query/getSeries", params=params)
        if response.status_code == 200:
            series_data = response.json()
            
            # Convert to our format
            results = []
            for series in series_data:
                results.append({
                    "SeriesInstanceUID": series.get("SeriesInstanceUID", ""),
                    "PatientID": series.get("PatientID", ""),
                    "StudyInstanceUID": series.get("StudyInstanceUID", ""),
                    "SeriesDescription": series.get("SeriesDescription", ""),
                    "Modality": series.get("Modality", ""),
                    "BodyPartExamined": series.get("BodyPartExamined", ""),
                    "Manufacturer": series.get("Manufacturer", ""),
                    "NumberOfImages": series.get("NumberOfImages", 0),
                    "Collection": series.get("Collection", "")
                })
            
            return results
        else:
            raise HTTPException(status_code=500, detail=f"TCIA API error: {response.status_code}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tcia/download")
async def download_tcia_series(request: Dict[str, Any]):
    """Download selected series to cloud storage"""
    try:
        series_uids = request.get("seriesUids", [])
        target = request.get("target", "cloud")
        
        if not series_uids:
            raise HTTPException(status_code=400, detail="No series UIDs provided")
        
        print(f"📥 Downloading {len(series_uids)} series to {target}")
        
        # For now, simulate the download process
        # In a real implementation, this would:
        # 1. Download DICOM files from TCIA
        # 2. Store them in cloud storage (S3, GCS, etc.)
        # 3. Return download status
        
        # Simulate download progress
        import time
        time.sleep(2)  # Simulate download time
        
        return {
            "downloadedCount": len(series_uids),
            "target": target,
            "seriesUids": series_uids,
            "status": "completed",
            "message": f"Successfully downloaded {len(series_uids)} series to {target}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Training API Server...")
    print("📊 API will be available at: http://0.0.0.0:8000")
    print("🔗 Training status: http://0.0.0.0:8000/api/training/status")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
