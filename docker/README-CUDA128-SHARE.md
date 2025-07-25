# MLOps Application - CUDA 12.8 Shareable Docker Setup

This directory contains the optimized Docker configuration for sharing the MLOps application with CUDA 12.8 support.

## 📁 Files Overview

- **`Dockerfile.cuda128-share`**: Production-ready Dockerfile with CUDA 12.8 and PyTorch 2.7.0
- **`docker-compose.cuda128-share.yml`**: Docker Compose configuration for the shareable setup
- **`README-CUDA128-SHARE.md`**: This instruction file

## 🚀 Quick Start

### 1. Build the Shareable Image

```bash
# Navigate to project root
cd /path/to/your/project

# Build using the shareable Dockerfile
docker build -f docker/Dockerfile.cuda128-share -t mlops-cuda-app:cuda128-v1.0 .
```

### 2. Run with Docker Compose

```bash
# Use the shareable compose file
docker-compose -f docker/docker-compose.cuda128-share.yml up -d
```

### 3. Verify the Setup

```bash
# Check containers are running
docker ps

# Test the services
python tests/test_docker_image.py

# Verify CUDA compatibility
docker exec mlops_cuda_app python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📦 Sharing the Docker Image

### Option 1: Save to TAR file
```bash
# Save image to file (will be large ~25GB)
docker save mlops-cuda-app:cuda128-v1.0 | gzip > mlops-cuda-app-cuda128.tar.gz

# Load on another machine
gunzip -c mlops-cuda-app-cuda128.tar.gz | docker load
```

### Option 2: Push to Docker Registry
```bash
# Tag for registry
docker tag mlops-cuda-app:cuda128-v1.0 your-registry/mlops-cuda-app:cuda128-v1.0

# Push to registry
docker push your-registry/mlops-cuda-app:cuda128-v1.0

# Pull on another machine
docker pull your-registry/mlops-cuda-app:cuda128-v1.0
```

## ⚙️ System Requirements

- **NVIDIA GPU**: Compatible with CUDA 12.8+
- **NVIDIA Driver**: 520.61.05 or newer
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Docker Compose**: 2.0+

## 🔧 Configuration Details

### CUDA & PyTorch Versions
- **CUDA**: 12.8.0
- **PyTorch**: 2.7.0+cu128
- **TorchVision**: 0.22.0
- **TorchAudio**: 2.7.0

### Exposed Ports
- **8000**: FastAPI application
- **5000**: MLflow tracking server

### Volume Mounts
- `./models` → `/app/models`
- `./data` → `/app/data`
- `./logs` → `/app/logs`
- `./mlruns` → `/app/mlruns`
- `./mlflow_db` → `/app/mlflow_db`

## 🏥 Health Checks

Both services include health checks:
- **FastAPI**: `curl -f http://localhost:8000/health`
- **MLflow**: `curl -f http://localhost:5000/health`

## 🐛 Troubleshooting

### GPU Not Available
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Container Fails to Start
```bash
# Check logs
docker logs mlops_cuda_app
docker logs mlops_mlflow
```

### Build Issues
```bash
# Clean build (no cache)
docker build --no-cache -f docker/Dockerfile.cuda128-share -t mlops-cuda-app:cuda128-v1.0 .
```

## 📋 Pre-requisites Setup

### Install NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🎯 Usage Examples

### Development Mode
```bash
# Run with live code mounting
docker run --gpus all -v $(pwd):/app -p 8000:8000 mlops-cuda-app:cuda128-v1.0
```

### Production Mode
```bash
# Run the full stack
docker-compose -f docker/docker-compose.cuda128-share.yml up -d
```

### Custom Configuration
```bash
# Override environment variables
docker run --gpus all -e MLFLOW_TRACKING_URI=http://your-mlflow:5000 mlops-cuda-app:cuda128-v1.0
```

---
**Note**: This setup is optimized for CUDA 12.8 compatibility and easy sharing across different environments.