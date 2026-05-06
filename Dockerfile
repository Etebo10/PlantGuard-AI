# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies (for scientific stack, TensorFlow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    git-lfs \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Download model files directly from GitHub
RUN curl -L https://github.com/Etebo10/PlantGuard-AI/raw/main/mobilenet_best.h5 -o mobilenet_best.h5
RUN curl -L https://github.com/Etebo10/PlantGuard-AI/raw/main/efficientnet_best.h5 -o efficientnet_best.h5

# Copy dependency files
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code
COPY . .

# Expose port (change if your app uses a different port)
EXPOSE 8080

# Command to run your app (edit this if your app uses something else)
CMD ["python", "app.py"]
