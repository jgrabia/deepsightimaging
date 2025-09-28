#!/bin/bash

# Deploy Enhanced Training API Server to AWS
# This script uploads and starts the enhanced API server with all React app endpoints

echo "🚀 DEPLOYING ENHANCED TRAINING API SERVER"
echo "=========================================="

# Configuration
SERVER_IP="3.88.157.239"
SERVER_USER="ubuntu"
KEY_PATH="C:/Keys/DeepSight.pem"
API_FILE="enhanced_training_api_server.py"

# Check if key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "❌ Error: Key file not found at $KEY_PATH"
    echo "Please ensure your SSH key is at the correct location"
    exit 1
fi

# Check if API file exists
if [ ! -f "$API_FILE" ]; then
    echo "❌ Error: API file not found: $API_FILE"
    exit 1
fi

echo "📁 Uploading enhanced API server..."
scp -i "$KEY_PATH" "$API_FILE" "$SERVER_USER@$SERVER_IP:~/mri_app/"

if [ $? -ne 0 ]; then
    echo "❌ Failed to upload API file"
    exit 1
fi

echo "🔧 Installing additional dependencies..."
ssh -i "$KEY_PATH" "$SERVER_USER@$SERVER_IP" << 'EOF'
cd ~/mri_app

# Install additional Python packages if needed
pip install fastapi uvicorn python-multipart aiofiles

# Create uploads directory
mkdir -p /tmp/uploads

# Stop any existing API server
pkill -f "training_api_server.py" || true
pkill -f "enhanced_training_api_server.py" || true

echo "✅ Dependencies installed and directories created"
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "🚀 Starting enhanced API server..."
ssh -i "$KEY_PATH" "$SERVER_USER@$SERVER_IP" << 'EOF'
cd ~/mri_app

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

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 ENHANCED API SERVER DEPLOYED SUCCESSFULLY!"
    echo "=============================================="
    echo "📊 API Base URL: http://3.88.157.239:8000"
    echo ""
    echo "🔗 Available Endpoints:"
    echo "  • GET  /api/training/status     - Training progress"
    echo "  • POST /api/training/start      - Start training"
    echo "  • POST /api/training/stop       - Stop training"
    echo "  • POST /api/files/upload        - Upload DICOM files"
    echo "  • GET  /api/files/list          - List uploaded files"
    echo "  • GET  /api/dicom/info/{file}   - DICOM file info"
    echo "  • GET  /api/dicom/image/{file}  - DICOM image data"
    echo "  • POST /api/ai/analyze          - AI analysis"
    echo "  • GET  /api/ai/models           - Available models"
    echo "  • GET  /api/tcia/collections    - TCIA collections"
    echo "  • GET  /api/tcia/body-parts     - TCIA body parts"
    echo "  • POST /api/tcia/search         - Search TCIA"
    echo "  • POST /api/tcia/download       - Download TCIA series"
    echo "  • GET  /api/system/status       - System status"
    echo ""
    echo "📱 Next Steps:"
    echo "1. Update React app to use AWS endpoints"
    echo "2. Test all features with real server data"
    echo "3. Deploy updated React app to Render"
else
    echo "❌ Failed to deploy enhanced API server"
    exit 1
fi
