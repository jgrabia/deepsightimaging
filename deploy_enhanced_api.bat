@echo off
echo 🚀 DEPLOYING ENHANCED TRAINING API SERVER TO AWS
echo ================================================

REM Configuration
set SERVER_IP=3.88.157.239
set SERVER_USER=ubuntu
set KEY_PATH=C:\Keys\DeepSight.pem
set API_FILE=enhanced_training_api_server.py

REM Check if key file exists
if not exist "%KEY_PATH%" (
    echo ❌ Error: Key file not found at %KEY_PATH%
    echo Please ensure your SSH key is at the correct location
    pause
    exit /b 1
)

REM Check if API file exists
if not exist "%API_FILE%" (
    echo ❌ Error: API file not found: %API_FILE%
    pause
    exit /b 1
)

echo 📁 Uploading enhanced API server...
scp -i "%KEY_PATH%" "%API_FILE%" %SERVER_USER%@%SERVER_IP%:~/mri_app/

if %errorlevel% neq 0 (
    echo ❌ Failed to upload API file
    pause
    exit /b 1
)

echo 🔧 Installing additional dependencies and starting server...
ssh -i "%KEY_PATH%" %SERVER_USER%@%SERVER_IP% << 'EOF'
cd ~/mri_app

# Install additional Python packages if needed
pip install fastapi uvicorn python-multipart aiofiles

# Create uploads directory
mkdir -p /tmp/uploads

# Stop any existing API server
pkill -f "training_api_server.py" || true
pkill -f "enhanced_training_api_server.py" || true

# Start the enhanced API server in background
nohup python enhanced_training_api_server.py > api_server.log 2>&1 &
API_PID=$!

# Wait a moment for server to start
sleep 5

# Check if server is running
if ps -p $API_PID > /dev/null; then
    echo "✅ Enhanced API server started successfully (PID: $API_PID)"
    echo "📊 API available at: http://3.88.157.239:8000"
    echo "🔗 React app can now connect to all endpoints"
else
    echo "❌ Failed to start API server"
    echo "📋 Checking logs..."
    tail -20 api_server.log
    exit 1
fi

# Save PID for later reference
echo $API_PID > api_server.pid
echo "💾 Server PID saved to api_server.pid"
EOF

if %errorlevel% equ 0 (
    echo.
    echo 🎉 ENHANCED API SERVER DEPLOYED SUCCESSFULLY!
    echo ==============================================
    echo 📊 API Base URL: http://3.88.157.239:8000
    echo.
    echo 🔗 Available Endpoints:
    echo   • GET  /api/training/status     - Training progress
    echo   • POST /api/training/start      - Start training
    echo   • POST /api/training/stop       - Stop training
    echo   • POST /api/files/upload        - Upload DICOM files
    echo   • GET  /api/files/list          - List uploaded files
    echo   • GET  /api/dicom/info/{file}   - DICOM file info
    echo   • GET  /api/dicom/image/{file}  - DICOM image data
    echo   • POST /api/ai/analyze          - AI analysis
    echo   • GET  /api/ai/models           - Available models
    echo   • GET  /api/tcia/collections    - TCIA collections
    echo   • GET  /api/tcia/body-parts     - TCIA body parts
    echo   • POST /api/tcia/search         - Search TCIA
    echo   • POST /api/tcia/download       - Download TCIA series
    echo   • GET  /api/system/status       - System status
    echo.
    echo 📱 Next Steps:
    echo 1. Deploy updated React app to Render
    echo 2. Test all features with real server data
    echo 3. Start training with your 13,421 DICOM files
) else (
    echo ❌ Failed to deploy enhanced API server
)

pause
