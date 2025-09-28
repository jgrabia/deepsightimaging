#!/bin/bash

echo "Starting DeepSight Imaging AI - Local Development"
echo "================================================"

echo ""
echo "Step 1: Installing React dependencies..."
npm install

echo ""
echo "Step 2: Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 3: Starting FastAPI backend..."
cd backend
python main.py &
BACKEND_PID=$!

echo ""
echo "Step 4: Waiting for backend to start..."
sleep 5

echo ""
echo "Step 5: Starting React frontend..."
cd ..
npm start &
FRONTEND_PID=$!

echo ""
echo "================================================"
echo "Both servers are starting up..."
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
