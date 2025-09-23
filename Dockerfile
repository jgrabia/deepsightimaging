# Simple Dockerfile for DeepSight Imaging AI
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and install Node dependencies
COPY package.json .
RUN npm install

# Copy all source files
COPY . .

# Build React app and move to frontend directory
RUN npm run build && mkdir -p frontend && mv build frontend/

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
