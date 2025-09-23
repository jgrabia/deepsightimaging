@echo off
echo Starting DeepSight Imaging AI - Local Development
echo ================================================

echo.
echo Step 1: Installing React dependencies...
call npm install

echo.
echo Step 2: Starting FastAPI backend...
start "FastAPI Backend" cmd /k "cd backend && python -m pip install -r ../requirements.txt && python main.py"

echo.
echo Step 3: Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Step 4: Starting React frontend...
start "React Frontend" cmd /k "npm start"

echo.
echo ================================================
echo Both servers are starting up...
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/api/docs
echo.
echo Press any key to close this window...
pause > nul
