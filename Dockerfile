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
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Initialize a git repo and pull actual LFS model files from remote
RUN git init
RUN git remote add origin https://github.com/Etebo10/PlantGuard-AI.git
RUN git lfs pull origin main

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
