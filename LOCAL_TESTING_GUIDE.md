# 🧪 Local Testing Guide - DeepSight Imaging AI

## Prerequisites

### 1. Install Node.js
- Download from [nodejs.org](https://nodejs.org/)
- Install the **LTS version** (recommended)
- Restart your terminal after installation

### 2. Verify Installation
```bash
node --version    # Should show v18.x.x or higher
npm --version     # Should show 9.x.x or higher
```

## 🚀 Quick Start (Windows)

### Option 1: One-Click Start
```cmd
# Double-click the start_local.bat file
start_local.bat
```

### Option 2: Manual Start
```cmd
# Install dependencies
npm install

# Start backend (Terminal 1)
cd backend
pip install -r ../requirements.txt
python main.py

# Start frontend (Terminal 2)
npm start
```

## 🚀 Quick Start (Mac/Linux)

### Option 1: One-Click Start
```bash
# Make executable and run
chmod +x start_local.sh
./start_local.sh
```

### Option 2: Manual Start
```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Start backend (Terminal 1)
cd backend
python main.py

# Start frontend (Terminal 2)
npm start
```

## 🌐 Access URLs

Once both servers are running:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

## 🔐 Login Credentials

Use these demo credentials to log in:
- **Username**: `admin`
- **Password**: `admin`

## 🧪 Testing Features

### 1. Dashboard
- View live statistics
- Check recent activity
- Monitor training progress

### 2. DICOM Viewer
- Upload DICOM files (drag & drop)
- Navigate through slices
- Test window/level controls
- Try cine mode playback
- Load annotations (if available)

### 3. AI Analysis
- Select DICOM files
- Choose AI models
- Run analysis
- View results

### 4. API Testing
- Visit http://localhost:8000/api/docs
- Test endpoints with Swagger UI
- Try file uploads
- Check authentication

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Kill processes on ports 3000 and 8000
# Windows:
netstat -ano | findstr :3000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Mac/Linux:
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

### Node.js Not Found
- Make sure Node.js is installed
- Restart your terminal
- Check PATH environment variable

### Python Dependencies Error
```bash
# Try with pip3 instead of pip
pip3 install -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### CORS Errors
- Make sure backend is running on port 8000
- Check that frontend proxy is configured in package.json

## 📱 Features to Test

### ✅ Core Functionality
- [ ] Login system
- [ ] Dashboard navigation
- [ ] DICOM file upload
- [ ] Image viewer controls
- [ ] AI analysis workflow
- [ ] API endpoints

### ✅ DICOM Viewer Features
- [ ] Slice navigation
- [ ] Zoom/pan controls
- [ ] Window/level presets
- [ ] Cine mode playback
- [ ] Annotation overlay
- [ ] Metadata display

### ✅ AI Analysis Features
- [ ] Model selection
- [ ] Batch processing
- [ ] Results visualization
- [ ] Export functionality

## 🚀 Next Steps

Once local testing is successful:

1. **Deploy to Server**: Use `./deploy.sh` for production deployment
2. **Configure Domain**: Point `deepsightimaging.ai` to your server
3. **SSL Setup**: Configure HTTPS with Let's Encrypt
4. **Monitor**: Set up logging and monitoring

## 📞 Support

If you encounter issues:
1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure ports 3000 and 8000 are available
4. Check firewall settings

Happy testing! 🎉
