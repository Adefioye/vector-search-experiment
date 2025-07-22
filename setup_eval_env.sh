#!/bin/bash

# Exit on any error
set -e

# Accept ToS for Anaconda channels
# Remove these lines (not supported on macOS)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with Python 3.11
echo "Creating conda environment with Python 3.10..."
conda create -n pyserini python=3.11 -y

# Install core dependencies inside the conda environment
echo "Installing conda packages (wget, Java, Maven, faiss-gpu)..."
conda run -n pyserini conda install -c anaconda wget -y
conda run -n pyserini conda install -c conda-forge openjdk=21 maven -y
# Use faiss-cpu for CPU-only installations, or faiss-gpu for GPU support
# Uncomment the next line if you want to use CPU version instead
# conda run -n pyserini conda install -c pytorch faiss-cpu -y
# For GPU support, uncomment the next line  and comment the above line
conda run -n pyserini conda install -c pytorch faiss-gpu -y

# Upgrade pip and install Python packages
echo "Installing pip requirements..."
conda run -n pyserini pip install --upgrade pip

# Install torch explicitly first as flash_attn requires it
conda run -n pyserini pip install torch

# Make sure faiss-gpu is REMOVED from requirements.txt before this step
conda run -n pyserini pip install -r requirements.txt

echo "✅ Environment setup complete."
echo "👉 Activate it with: conda activate pyserini"

