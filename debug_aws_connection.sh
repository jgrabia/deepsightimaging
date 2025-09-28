#!/bin/bash

# Debug AWS Server Connection Issues
echo "🔍 DEBUGGING AWS SERVER CONNECTION"
echo "=================================="

cd ~/mri_app

echo "1️⃣ Checking if Enhanced API server is running..."
if systemctl is-active --quiet enhanced-api; then
    echo "✅ Enhanced API service: RUNNING"
else
    echo "❌ Enhanced API service: NOT RUNNING"
    echo "🔄 Starting service..."
    sudo systemctl start enhanced-api
    sleep 3
fi

echo ""
echo "2️⃣ Testing local API endpoints..."
echo "Testing basic endpoint:"
if curl -s http://localhost:8000/; then
    echo "✅ Local API responding"
else
    echo "❌ Local API not responding"
fi

echo ""
echo "Testing training status:"
curl -s http://localhost:8000/api/training/status | python3 -m json.tool || echo "❌ Training status failed"

echo ""
echo "Testing system status:"
curl -s http://localhost:8000/api/system/status | python3 -m json.tool || echo "❌ System status failed"

echo ""
echo "3️⃣ Checking server ports and processes..."
echo "Processes listening on port 8000:"
sudo netstat -tlnp | grep :8000 || echo "❌ Nothing listening on port 8000"

echo ""
echo "All Python processes:"
ps aux | grep python | grep -v grep

echo ""
echo "4️⃣ Checking AWS Security Group..."
echo "Current server IP:"
curl -s http://169.254.169.254/latest/meta-data/public-ipv4

echo ""
echo "5️⃣ Testing external connectivity..."
echo "Testing from external source:"
curl -s http://3.88.157.239:8000/ || echo "❌ External access failed"

echo ""
echo "6️⃣ Checking CORS configuration..."
echo "Testing with CORS headers:"
curl -s -H "Origin: https://deepsightimaging.onrender.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     http://localhost:8000/api/files/upload || echo "❌ CORS test failed"

echo ""
echo "7️⃣ Checking server logs..."
echo "Recent API server logs:"
sudo journalctl -u enhanced-api --no-pager -n 20

echo ""
echo "8️⃣ Checking if nginx is running (for reverse proxy)..."
if systemctl is-active --quiet nginx; then
    echo "✅ Nginx is running"
    echo "Nginx status:"
    sudo systemctl status nginx --no-pager -l
else
    echo "❌ Nginx not running - this might be needed for CORS"
fi

echo ""
echo "🎯 SUMMARY"
echo "=========="
echo "If you see ❌ errors above, those are the issues to fix."
echo "Most likely issues:"
echo "1. AWS Security Group not allowing port 8000"
echo "2. CORS not properly configured"
echo "3. API server not running or crashed"
echo ""
echo "Next steps:"
echo "1. Fix AWS Security Group to allow port 8000"
echo "2. Restart the API server"
echo "3. Test external connectivity"
