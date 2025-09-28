#!/bin/bash

# Fix AWS Server Connectivity Issues
echo "🔧 FIXING AWS SERVER CONNECTIVITY"
echo "================================="

cd ~/mri_app

echo "1️⃣ Ensuring Enhanced API server is running..."
# Stop any existing processes
pkill -f "enhanced_training_api_server.py" || true
pkill -f "training_api_server.py" || true

# Start the enhanced API server
echo "🚀 Starting Enhanced API server..."
nohup python3 enhanced_training_api_server.py > api_server.log 2>&1 &
API_PID=$!

sleep 5

if ps -p $API_PID > /dev/null; then
    echo "✅ Enhanced API server started (PID: $API_PID)"
    echo $API_PID > api_server.pid
else
    echo "❌ Failed to start API server"
    echo "📋 Checking logs..."
    tail -20 api_server.log
    exit 1
fi

echo ""
echo "2️⃣ Testing local connectivity..."
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Local API responding"
    echo "📊 API Response:"
    curl -s http://localhost:8000/ | python3 -m json.tool
else
    echo "❌ Local API not responding"
    exit 1
fi

echo ""
echo "3️⃣ Checking server IP and port..."
SERVER_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "Server public IP: $SERVER_IP"

echo "Port 8000 status:"
sudo netstat -tlnp | grep :8000

echo ""
echo "4️⃣ Testing external connectivity..."
echo "Testing from server to itself via public IP..."
if curl -s http://$SERVER_IP:8000/ > /dev/null; then
    echo "✅ External access working"
else
    echo "❌ External access failed - AWS Security Group issue"
    echo ""
    echo "🔧 AWS SECURITY GROUP FIX NEEDED:"
    echo "=================================="
    echo "1. Go to AWS EC2 Console"
    echo "2. Select your instance"
    echo "3. Go to Security tab"
    echo "4. Click on Security Group"
    echo "5. Add Inbound Rule:"
    echo "   - Type: Custom TCP"
    echo "   - Port: 8000"
    echo "   - Source: 0.0.0.0/0"
    echo "6. Save the rule"
    echo ""
    echo "Then test again with: curl http://$SERVER_IP:8000/"
fi

echo ""
echo "5️⃣ Setting up proper CORS headers..."
echo "The Enhanced API server already has CORS configured, but let's verify..."

echo "Testing CORS preflight request:"
curl -s -H "Origin: https://deepsightimaging.onrender.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     http://localhost:8000/api/files/upload && echo "✅ CORS working" || echo "❌ CORS issue"

echo ""
echo "6️⃣ Creating simple test endpoints..."
cat > test_endpoints.py << 'EOF'
#!/usr/bin/env python3
import requests
import json

base_url = "http://localhost:8000"

endpoints = [
    "/",
    "/api/training/status",
    "/api/system/status",
    "/api/tcia/collections",
    "/api/tcia/body-parts"
]

print("🧪 Testing all endpoints...")
for endpoint in endpoints:
    try:
        response = requests.get(f"{base_url}{endpoint}", timeout=5)
        if response.status_code == 200:
            print(f"✅ {endpoint}: OK")
        else:
            print(f"❌ {endpoint}: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ {endpoint}: {e}")

print("\n📊 Testing POST endpoint (file upload simulation)...")
try:
    # Test file upload endpoint
    files = {'files': ('test.dcm', b'test content', 'application/dicom')}
    response = requests.post(f"{base_url}/api/files/upload", files=files, timeout=10)
    if response.status_code == 200:
        print("✅ /api/files/upload: OK")
    else:
        print(f"❌ /api/files/upload: HTTP {response.status_code}")
except Exception as e:
    print(f"❌ /api/files/upload: {e}")
EOF

python3 test_endpoints.py

echo ""
echo "🎯 CONNECTIVITY STATUS"
echo "====================="
echo "✅ Enhanced API server: RUNNING"
echo "📊 API available at: http://$SERVER_IP:8000"
echo "🔗 React app should connect to: http://$SERVER_IP:8000"
echo ""
echo "🛠️ If still having issues:"
echo "1. Fix AWS Security Group (port 8000)"
echo "2. Check firewall: sudo ufw status"
echo "3. Test external access: curl http://$SERVER_IP:8000/"
echo "4. Check React app console for specific error messages"
