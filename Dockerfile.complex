# Multi-stage build for DeepSight Imaging AI

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY package*.json ./
    RUN npm install

COPY src/ ./src/
COPY public/ ./public/
COPY tsconfig.json ./

RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create directories for DICOM processing
RUN mkdir -p /app/data/dicom_incoming \
    /app/data/dicom_processed \
    /app/data/dicom_errors \
    /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DICOM_DATA_PATH=/app/data
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
